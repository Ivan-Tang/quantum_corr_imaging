from model import CompressiveImagingModel
from dataset import GhostImagingDataset
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image



def predict_and_save(test_root_dir, img_size, model_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = GhostImagingDataset(test_root_dir, img_size, split='all')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = CompressiveImagingModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    for idx, (X, mask, target) in enumerate(loader):
        X, mask = X.to(device), mask.to(device)
        with torch.no_grad():
            pred = model(X, mask)
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

