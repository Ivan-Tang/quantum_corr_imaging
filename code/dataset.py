import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GhostImagingDataset(Dataset):
    def __init__(self, root_dir, img_size=(512, 384), stack_num=5, split='train', split_ratio=0.8, max_signal=100, max_idler=100):
        self.object_dirs = []
        for name in sorted(os.listdir(root_dir)):
            obj_path = os.path.join(root_dir, name)
            if os.path.isdir(obj_path):
                self.object_dirs.append(obj_path)
        self.img_size = img_size
        self.split = split
        self.split_ratio = split_ratio
        self.stack_num = stack_num
        self.max_signal = max_signal
        self.max_idler = max_idler
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])

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
        # 读取 idler
        idler_paths = sorted([
            os.path.join(idler_dir, f) for f in os.listdir(idler_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        assert len(signal_paths) == len(idler_paths), "Mismatch between signal and idler images"
        n = len(signal_paths)
        split_idx = int(n * self.split_ratio)
        if self.split == 'train':
            signal_paths = signal_paths[:split_idx]
            idler_paths = idler_paths[:split_idx]
        else:
            signal_paths = signal_paths[split_idx:]
            idler_paths = idler_paths[split_idx:]
        # 只取前max_signal和max_idler张
        signal_imgs = []
        for path in signal_paths[:self.max_signal]:
            img = Image.open(path).convert('L')
            img = self.transform(img)  # [1, H, W]
            signal_imgs.append(img)
        idler_imgs = []
        for path in idler_paths[:self.max_idler]:
            img = Image.open(path).convert('L')
            img = self.transform(img)
            idler_imgs.append(img)
        # 堆叠为多通道输入
        X = torch.cat(signal_imgs + idler_imgs, dim=0)  # [C, H, W], C=max_signal+max_idler
        target = self.transform(Image.open(target_path).convert("L")).float()  # [1, H, W]
        return X, target

