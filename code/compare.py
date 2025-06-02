from torchmetrics.functional import peak_signal_noise_ratio as psnr
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def psnr_metric(output_path, fused_img_path, qci_img_path, unet_img_path, target_path):
    def image_to_tensor(img_path):
        """将图片转换为tensor格式"""
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        # 转换为torch tensor并归一化到[0,1]
        tensor = torch.from_numpy(img_array).float() / 255.0
        # 调整维度顺序 (H,W,C) -> (C,H,W)
        tensor = tensor.permute(2, 0, 1)
        return tensor

    def tensor_to_image(tensor):
        """将tensor转换回图片格式用于显示"""
        # 调整维度顺序 (C,H,W) -> (H,W,C)
        img_array = tensor.permute(1, 2, 0).numpy()
        # 确保值在[0,1]范围内
        img_array = np.clip(img_array, 0, 1)
        return img_array

    # 加载所有图片
    target_img = image_to_tensor(target_path)
    fused_tensor = image_to_tensor(fused_img_path)
    qci_tensor = image_to_tensor(qci_img_path)
    unet_tensor = image_to_tensor(unet_img_path)

    # 计算PSNR值
    psnr_fused = psnr(fused_tensor, target_img, data_range=1.0)
    psnr_qci = psnr(qci_tensor, target_img, data_range=1.0)
    psnr_unet = psnr(unet_tensor, target_img, data_range=1.0)

    # 创建可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Image Comparison with PSNR Values', fontsize=16, fontweight='bold')

    # 图片和标题
    images = [fused_tensor, qci_tensor, unet_tensor, target_img]
    titles = ['Fused', 'QCI', 'UNet', 'Target']
    psnr_values = [psnr_fused.item(), psnr_qci.item(), psnr_unet.item(), None]  # Target没有PSNR值

    for i, (ax, img_tensor, title, psnr_val) in enumerate(zip(axes, images, titles, psnr_values)):
        # 显示图片
        img_array = tensor_to_image(img_tensor)
        ax.imshow(img_array)
        ax.axis('off')
        
        # 设置标题
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        # 设置PSNR标签
        if psnr_val is not None:
            ax.text(0.5, -0.1, f'PSNR: {psnr_val:.2f} dB', 
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue'))
        else:
            ax.text(0.5, -0.1, 'Reference Image', 
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.2)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Comparison image saved to {output_path}')
    plt.close()

    # 打印PSNR结果
    print("\nPSNR Results:")
    print(f"Fused vs Target: {psnr_fused:.2f} dB")
    print(f"QCI vs Target: {psnr_qci:.2f} dB")
    print(f"UNet vs Target: {psnr_unet:.2f} dB")

if __name__ == '__main__':
    fused_img_path = 'reports/imgs/fused_image_1.png'
    qci_img_path = 'reports/imgs/quantum_corr_x.jpg'
    unet_img_path = 'reports/imgs/pred_1.png'
    target_path = 'reports/imgs/target_1.jpg'
    psnr_metric('reports/imgs/psnr_comparison_x.png', fused_img_path, qci_img_path, unet_img_path, target_path)

    fused_img_path = 'reports/imgs/fused_image_3.png'
    qci_img_path = 'reports/imgs/quantum_corr_one.jpg'
    unet_img_path = 'reports/imgs/pred_3.png'
    target_path = 'reports/imgs/target_3.JPG'
    psnr_metric('reports/imgs/psnr_comparison_one.png', fused_img_path, qci_img_path, unet_img_path, target_path)
