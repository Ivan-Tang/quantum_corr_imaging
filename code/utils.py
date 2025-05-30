import torchvision.models as models
import torch.nn.functional as F
import torch

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

