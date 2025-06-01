import torchvision.models as models
import torch.nn.functional as F
import torch
import random
from PIL import Image

# 兼容新版torchvision的VGG16加载方式
try:
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
except AttributeError:
    vgg = models.vgg16(pretrained=True).features.eval()

def perceptual_loss(output, target):
    # output, target: [B, 1, H, W]
    device = output.device
    vgg.to(device)
    if output.shape[1] == 1:
        output = output.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
    output = output.to(device)
    target = target.to(device)
    feat_out = vgg(output)
    feat_tar = vgg(target)
    loss = F.mse_loss(feat_out, feat_tar)
    return loss

#图片随机掩码
def random_mask(img, mask_ratio=0.1, block_ratio=0.1):
    _, H, W = img.shape
    mask_amount = int(H * W * mask_ratio)
    block_amount = int(H * W * block_ratio)
    for i in range(mask_amount+block_amount):
        y = random.randint(0, H - 1)
        x = random.randint(0, W - 1)
        if i < mask_amount:
            img[..., y, x] = random.uniform(0, 1)  # 随机像素值
        else:
            img[..., y, x] = 0  # 掩码像素值

    return img

class RandomNoiseTransform:
    def __init__(self, mask_ratio=0.1, block_ratio=0.1):
        self.mask_ratio = mask_ratio
        self.block_ratio = block_ratio

    def __call__(self, img_tensor):
        _, H, W = img_tensor.shape
        
        # Apply random pixel noise
        num_pixels_to_mask = int(H * W * self.mask_ratio)
        if num_pixels_to_mask > 0:
            mask_indices = torch.randperm(H * W, device=img_tensor.device)[:num_pixels_to_mask]
            mask_y = mask_indices // W
            mask_x = mask_indices % W
            noise_values = torch.rand(num_pixels_to_mask, device=img_tensor.device)
            img_tensor[..., mask_y, mask_x] = noise_values

        # Apply blocking noise (setting pixels to 0)
        num_pixels_to_block = int(H * W * self.block_ratio)
        if num_pixels_to_block > 0:
            # Ensure block indices are different from noise indices if needed,
            # but for simplicity here, we draw fresh random indices.
            # If overlap is a concern, a more complex sampling without replacement would be needed.
            block_indices = torch.randperm(H * W, device=img_tensor.device)[:num_pixels_to_block]
            block_y = block_indices // W
            block_x = block_indices % W
            img_tensor[..., block_y, block_x] = 0
            
        return img_tensor



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    img_path = 'data/sample_data/train/1/target.jpg'
    img = Image.open(img_path).convert('L')
    img_tensor = torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0)  # 修正
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(img_tensor.squeeze().numpy(), cmap='gray')

    masked = random_mask(img_tensor.clone(), mask_ratio=0.2, block_ratio=0.1)
    plt.subplot(1, 2, 2)
    plt.title('Masked')
    plt.imshow(masked.squeeze().numpy(), cmap='gray')
    plt.savefig('reports/masking_comparison.png')
    plt.close()




