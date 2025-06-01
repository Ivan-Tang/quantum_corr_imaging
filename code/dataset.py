import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import RandomNoiseTransform # Changed from random_mask

class GhostImagingDataset(Dataset):
    def __init__(self, root_dir, img_size=None, stack_num=5, split='train', split_ratio=0.8, max_signal=100, max_idler=100, add_noise=False): # Added add_noise parameter
        self.object_dirs = []
        for name in sorted(os.listdir(root_dir)):
            obj_path = os.path.join(root_dir, name)
            if os.path.isdir(obj_path):
                self.object_dirs.append(obj_path)
        # 自动推断图片尺寸
        if img_size is None:
            # 找到第一个signal图片
            for obj_dir in self.object_dirs:
                signal_dir = os.path.join(obj_dir, 'signal')
                for f in os.listdir(signal_dir):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img = Image.open(os.path.join(signal_dir, f))
                        w, h = img.size
                        img_size = (h, w)
                        break
                if img_size is not None:
                    break
        self.img_size = img_size
        self.split = split
        self.split_ratio = split_ratio
        self.stack_num = stack_num
        self.max_signal = max_signal
        self.max_idler = max_idler
        self.add_noise = add_noise # Store add_noise

        base_transforms = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]

        if split == 'train':
            train_specific_transforms = [
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
            ]
            current_transforms = train_specific_transforms + base_transforms
        else:
            current_transforms = base_transforms
        
        if self.add_noise:
            current_transforms.append(RandomNoiseTransform(mask_ratio=0.1, block_ratio=0.1))

        self.transform = transforms.Compose(current_transforms)

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
        # n = len(signal_paths) # This was not used
        # split_idx = int(n * self.split_ratio) # This was not used

        # The split logic seems to be applied differently than typical train/test splits.
        # Usually, you'd partition the object_dirs itself in __init__ based on split_ratio.
        # The current logic applies split_ratio to each object's image sequence, which might be intended.
        # For now, I will keep the existing logic for signal_paths and idler_paths slicing.
        
        # Determine which part of the data to use based on the split
        num_images_per_object = len(signal_paths)
        split_point = int(num_images_per_object * self.split_ratio)

        if self.split == 'train':
            current_signal_paths = signal_paths[:split_point]
            current_idler_paths = idler_paths[:split_point]
        elif self.split == 'val' or self.split == 'test': # Assuming 'val' or 'test' for the other part
            current_signal_paths = signal_paths[split_point:]
            current_idler_paths = idler_paths[split_point:]
        else:
            # Default to using all if split is not recognized, or handle error
            current_signal_paths = signal_paths 
            current_idler_paths = idler_paths

        # 只取前max_signal和max_idler张
        signal_imgs = []
        for path in current_signal_paths[:self.max_signal]: # Use current_signal_paths
            img = Image.open(path).convert('L')
            img = self.transform(img)  # [1, H, W], transform now includes noise if add_noise is True
            signal_imgs.append(img)
        idler_imgs = []
        for path in current_idler_paths[:self.max_idler]: # Use current_idler_paths
            img = Image.open(path).convert('L')
            img = self.transform(img)
            idler_imgs.append(img)

        # 预处理：随机噪声 - This is now part of self.transform if add_noise is True
        # #for img in signal_imgs + idler_imgs:
        #     #img = random_mask(img, 0.1, 0.1) # Kept commented out

        # 预叠加：每stack_num张合成一张（均值）
        def stack_and_merge(imgs, stack_num):
            merged = []
            for i in range(0, len(imgs), stack_num):
                group = imgs[i:i+stack_num]
                if len(group) < stack_num:
                    # 不足stack_num的丢弃
                    continue
                group_tensor = torch.stack(group, dim=0)  # [stack_num, 1, H, W]
                merged_img = group_tensor.sum(dim=0)  # [1, H, W]
                merged.append(merged_img)
            return merged
        signal_merged = stack_and_merge(signal_imgs, self.stack_num)
        idler_merged = stack_and_merge(idler_imgs, self.stack_num)
        # 堆叠为多通道输入
        X = torch.cat(signal_merged + idler_merged, dim=0)  # [C, H, W], C=(max_signal//stack_num)+(max_idler//stack_num)
        target = self.transform(Image.open(target_path).convert("L")).float()  # [1, H, W]
        return X, target

