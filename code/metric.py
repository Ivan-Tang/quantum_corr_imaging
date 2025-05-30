from model import UNet
from dataset import GhostImagingDataset
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image



<<<<<<< HEAD
def predict_and_save(test_root_dir, img_size, model_path, save_dir):
=======
def predict_and_save(test_root_dir, img_size, model_path, save_dir, max_signal=100, max_idler=100):
>>>>>>> 852e5ba (为GhostImagingDataset类添加stack_num参数，更新数据处理逻辑以支持图像叠加；在train.py中更新数据集初始化，添加损失函数计算；在metric.py中添加空行以提高可读性；创建utils.py文件。)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = GhostImagingDataset(test_root_dir, img_size, split='all', max_signal=max_signal, max_idler=max_idler)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    in_channels = max_signal + max_idler
    model = UNet(in_channels=in_channels, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    for idx, (X, target) in enumerate(loader):
        X = X.to(device)
        with torch.no_grad():
            pred = model(X)
        pred_img = pred.squeeze().cpu()
        save_path = os.path.join(save_dir, f'pred_{idx}.png')
        save_image(pred_img, save_path)

        target_img = target.squeeze().cpu()
        save_image(target_img, os.path.join(save_dir, f'target_{idx}.png'))
        print(f'Saved: {save_path}')

if __name__ == "__main__":
    test_root_dir = 'data/test'  # 按实际路径修改
    img_size = (512, 384)
    model_path = 'checkpoints/best_model.pth'
    save_dir = 'results/'
    predict_and_save(test_root_dir, img_size, model_path, save_dir)

