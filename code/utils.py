import torchvision.models as models
import torch.nn.functional as F

vgg = models.vgg16(pretrained=True).features.eval()
def perceptual_loss(output, target):
    fear_out = vgg(output)
    feat_tar = vgg(target)
    loss = F.mse_loss(fear_out, feat_tar)
    return loss

