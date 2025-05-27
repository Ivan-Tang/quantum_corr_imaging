import torch
from torch.utils.data import DataLoader
from dataset import GhostImagingDataset
from model import CompressiveImagingModel
import torch.nn as nn

# =====================
# 配置区（超参数集中管理）
# =====================
config = {
    'root_dir': 'data/train',
    'img_size': (512, 384),
    'input_dim': 100,
    'hidden_dim': 256,
    'n_heads': 4,
    'n_layers': 2,
    'output_size': 512 * 384,
    'epochs': 10,
    'learning_rate': 1e-3,
    'keep_ratio': 0.2,
    'batch_size': 1
}

def train():
    # 加载训练集和验证集
    train_dataset = GhostImagingDataset(config['root_dir'], config['img_size'], keep_ratio=config['keep_ratio'], split='train')
    val_dataset = GhostImagingDataset(config['root_dir'], config['img_size'], keep_ratio=config['keep_ratio'], split='val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = CompressiveImagingModel(
        config['input_dim'], config['hidden_dim'], config['n_heads'], config['n_layers'], config['output_size']
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    losses = []

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        for X, mask, target in train_loader:
            X, mask, target = X.to(device), mask.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(X, mask)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, mask, target in val_loader:
                X, mask, target = X.to(device), mask.to(device), target.to(device)

                output = model(X, mask)
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

