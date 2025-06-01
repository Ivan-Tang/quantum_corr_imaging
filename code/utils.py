import torchvision.models as models
import torch.nn.functional as F
import torch
import random


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
    # img: torch.Tensor, shape [1, H, W]
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



