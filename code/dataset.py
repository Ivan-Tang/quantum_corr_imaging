import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GhostImagingDataset(Dataset):
    def __init__(self, root_dir, img_size=(512, 384), compress_dim=100, keep_ratio=0.05, split='train', split_ratio=0.8):
        """
        root_dir: 形如 data/dataset/train，每个子文件夹为一个物体，内含 signal/、idler/、target.JPG
        split: 'train' 或 'val'，决定返回每个物体的前 split_ratio 还是后 1-split_ratio 部分数据
        split_ratio: 训练集比例
        """
        self.object_dirs = []
        for name in sorted(os.listdir(root_dir)):
            obj_path = os.path.join(root_dir, name)
            if os.path.isdir(obj_path):
                self.object_dirs.append(obj_path)
        self.img_size = img_size
        self.compress_dim = compress_dim
        self.keep_ratio = keep_ratio
        self.split = split
        self.split_ratio = split_ratio
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        np.random.seed(42)
        img_flatten_size = img_size[0] * img_size[1]
        self.Phi = np.random.randn(compress_dim, img_flatten_size).astype(np.float32)

    def __len__(self):
        return len(self.object_dirs)

    def __getitem__(self, idx):
        obj_dir = self.object_dirs[idx]
        signal_dir = os.path.join(obj_dir, 'signal')
        idler_dir = os.path.join(obj_dir, 'idler')
        target_path = os.path.join(obj_dir, 'target.JPG')
        # 读取 signal
        signal_paths = sorted([
            os.path.join(signal_dir, f) for f in os.listdir(signal_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        idler_paths = sorted([
            os.path.join(idler_dir, f) for f in os.listdir(idler_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        assert len(signal_paths) == len(idler_paths), "Mismatch between signal and idler images"
        # 按 split 划分
        n = len(signal_paths)
        split_idx = int(n * self.split_ratio)
        if self.split == 'train':
            signal_paths = signal_paths[:split_idx]
            idler_paths = idler_paths[:split_idx]
        else:
            signal_paths = signal_paths[split_idx:]
            idler_paths = idler_paths[split_idx:]
        # 读取 signal 图像
        X_raw = []
        for path in signal_paths:
            img = Image.open(path).convert("L")
            img = self.transform(img).view(-1)
            X_raw.append(img)
        X_raw = torch.stack(X_raw, dim=0)  # [N, H*W]
        X_comp = torch.matmul(torch.from_numpy(self.Phi), X_raw.T).T  # [N, compress_dim]
        # 读取 idler 图像
        mask = torch.zeros(len(signal_paths))
        num_keep = int(self.keep_ratio * len(signal_paths))
        keep_indices = np.random.choice(len(signal_paths), num_keep, replace=False)
        mask[keep_indices] = 1.0
        # 读取 target
        target = self.transform(Image.open(target_path).convert("L")).float()  # [1, H, W]
        return X_comp.float(), mask, target

