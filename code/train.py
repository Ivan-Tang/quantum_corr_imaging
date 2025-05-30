import torch
from torch.utils.data import DataLoader
from dataset import GhostImagingDataset
from model import UNet
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
<<<<<<< HEAD



=======
>>>>>>> 852e5ba (为GhostImagingDataset类添加stack_num参数，更新数据处理逻辑以支持图像叠加；在train.py中更新数据集初始化，添加损失函数计算；在metric.py中添加空行以提高可读性；创建utils.py文件。)

config = {
    'root_dir': 'data/train',
    'img_size': (512, 384),
    'epochs': 50,
    'learning_rate': 1e-3,
<<<<<<< HEAD
    'keep_ratio': 0.2,
    'stack_num': 5,
    'batch_size': 1,
    'loss_weight_psnr': 1,
    'loss_weight_ssim': 1
}

def loss_fn(output, target):
    B = output.shape[0]
    H, W = config['img_size']
    output = output.view(B, 1, H, W)
    loss = config['loss_weight_ssim'] * (1 - ssim(output, target)) + config['loss_weight_psnr'] * psnr(output, target)
    return loss

def train():
    # 加载训练集和验证集
    train_dataset = GhostImagingDataset(config['root_dir'], config['img_size'], keep_ratio=config['keep_ratio'], split='train', stack_num=config['stack_num'])
    val_dataset = GhostImagingDataset(config['root_dir'], config['img_size'], keep_ratio=config['keep_ratio'], split='val', stack_num=config['stack_num'])
=======
    'stack_num': 5,
    'batch_size': 1,
    'max_signal': 100,
    'max_idler': 100,
    'in_channels': 200,  # max_signal+max_idler
    'out_channels': 1
}

def loss_fn(output, target):
    # output, target: [B, 1, H, W]
    return 1 - ssim(output, target)
>>>>>>> 852e5ba (为GhostImagingDataset类添加stack_num参数，更新数据处理逻辑以支持图像叠加；在train.py中更新数据集初始化，添加损失函数计算；在metric.py中添加空行以提高可读性；创建utils.py文件。)

def train():
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

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for X, target in train_loader:
            X, target = X.to(device), target.to(device)
            # X: [B, C, H, W], target: [B, 1, H, W]
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
<<<<<<< HEAD
            for X, mask, target in val_loader:
                X, mask, target = X.to(device), mask.to(device), target.to(device)

                output = model(X, mask)

=======
            for X, target in val_loader:
                X, target = X.to(device), target.to(device)
                output = model(X)
>>>>>>> 852e5ba (为GhostImagingDataset类添加stack_num参数，更新数据处理逻辑以支持图像叠加；在train.py中更新数据集初始化，添加损失函数计算；在metric.py中添加空行以提高可读性；创建utils.py文件。)
                loss = loss_fn(output, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch + 1}/{config['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        losses.append((train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print("Saved Best Model")

    #plot
    import matplotlib.pyplot as plt
    plt.plot([x[0] for x in losses], label='train_loss')
    plt.plot([x[1] for x in losses], label='val_loss')
    plt.legend()
    plt.savefig('losses.png')

if __name__ == "__main__":
    train()

