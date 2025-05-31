from model import UNet
from dataset import GhostImagingDataset
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import json

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
def get_latest_config(results_dir='results'):
    # 查找results下最新的exp_xxx/config.json
    exp_dirs = [d for d in os.listdir(results_dir) if d.startswith('exp_') and os.path.isdir(os.path.join(results_dir, d))]
    if not exp_dirs:
        raise FileNotFoundError('No exp_xxx directory found in results/')
    exp_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    latest_exp = exp_dirs[0]
    config_path = os.path.join(results_dir, latest_exp, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")
    return config, latest_exp

def predict_and_save_with_config(test_root_dir, img_size, model_path, save_dir, config):
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    max_signal = config['max_signal']
    max_idler = config['max_idler']
    stack_num = config.get('stack_num', 1)
    in_channels = (max_signal // stack_num) + (max_idler // stack_num)
    dataset = GhostImagingDataset(test_root_dir, img_size, split='all', max_signal=max_signal, max_idler=max_idler, stack_num=stack_num)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = UNet(in_channels=in_channels, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    metrics = []
    start_time = time.time()
    for idx, (X, target) in enumerate(loader):
        X = X.to(device)
        target = target.to(device)
        with torch.no_grad():
            pred = model(X)
        metric = psnr(pred, target).cpu().item()
        metrics.append(metric)
        pred_img = pred.squeeze().cpu()
        save_path_pred = os.path.join(save_dir, f'pred_{idx}.png')
        save_image(pred_img, save_path_pred)
        print(f'Saved: {save_path_pred}')
        target_img = target.squeeze().cpu()
        save_path_target = os.path.join(save_dir, f'target_{idx}.png')
        save_image(target_img, save_path_target)
        print(f"Saved: {save_path_target}")
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(loader)
    print(f'Total inference time {total_time:.3f}, avg time per image {avg_time:.3f}')
    return list(metrics)

def run_metric():
    test_root_dir = 'data/test'
    img_size = (512, 384)
    model_path = 'checkpoints/best_model.pth'
    config, latest_exp = get_latest_config('results')
    save_dir = os.path.join('results/', latest_exp)
    psnrs = predict_and_save_with_config(test_root_dir, img_size, model_path, save_dir, config)
    print(f'PSNRs: {psnrs}')
    with open(os.path.join(save_dir, 'psnrs.txt'), 'w') as f:
        f.write(str(psnrs))


if __name__ == "__main__":
    run_metric()



