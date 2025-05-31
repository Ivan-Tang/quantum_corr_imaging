import torch
from torch.utils.data import DataLoader
from dataset import GhostImagingDataset
from model import UNet
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from utils import perceptual_loss
import json
import os
import time
import optuna
from metric import run_metric

config = {
    'root_dir': 'data/train',
    'img_size': (384, 512),
    'epochs': 50,
    'learning_rate': 1e-3,
    'stack_num': 5,
    'batch_size': 1,
    'max_signal': 100,
    'max_idler': 100,
    'out_channels': 1,
    'loss_weight_ssim': 1,
    'loss_weight_mse': 0.1,
    'loss_weight_perceptual': 0.1, 
}


#根据max_signal, max_idler, stack_num计算输入通道数
config['in_channels'] = (config['max_signal'] // config['stack_num']) + (config['max_idler'] // config['stack_num'])

def train(config, return_best_val_loss=False, trial=None):
    exp_name = f"exp_{time.strftime('%Y%m%d_%H%M%S')}_epochs{config['epochs']}_stack{config['stack_num']}_lr{config['learning_rate']:.3g}_sig{config['max_signal']}"
    exp_dir = os.path.join('results', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    train_dataset = GhostImagingDataset(
        config['root_dir'],
        config['img_size'],
        stack_num=config['stack_num'],
        split='train',
        max_signal=config['max_signal'],
        max_idler=config['max_idler']
    )
    val_dataset = GhostImagingDataset(
        config['root_dir'],
        config['img_size'],
        stack_num=config['stack_num'],
        split='val',
        max_signal=config['max_signal'],
        max_idler=config['max_idler']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = UNet(in_channels=config['in_channels'], out_channels=config['out_channels'])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')
    losses = []
    psnrs = []

    def loss_fn(output, target):
        return config['loss_weight_ssim'] * (1 - ssim(output, target)) + \
               config['loss_weight_mse'] * nn.MSELoss()(output, target) + \
               config['loss_weight_perceptual'] * perceptual_loss(output, target)

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        for X, target in train_loader:
            X, target = X.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_psnr += float(psnr(output, target).cpu().item())
        train_loss /= len(train_loader)
        train_psnr /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for X, target in val_loader:
                X, target = X.to(device), target.to(device)
                output = model(X)
                loss = loss_fn(output, target)
                val_loss += loss.item()
                val_psnr += float(psnr(output, target).cpu().item())
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        # optuna pruner
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()

        print(f"Epoch [{epoch + 1}/{config['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        losses.append((train_loss, val_loss))
        psnrs.append((float(train_psnr), float(val_psnr)))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            print("Saved Best Model")

    #plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([x[0] for x in losses], label='train_loss')
    plt.plot([x[1] for x in losses], label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'losses.png'))
    plt.close()

    plt.figure()
    plt.plot([x[0] for x in psnrs], label = 'train_psnr')
    plt.plot([x[1] for x in psnrs], label = 'val_psnr')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'psnrs.png'))
    plt.close()

    if return_best_val_loss:
        return best_val_loss

if __name__ == "__main__":
    train(config)
    run_metric()

