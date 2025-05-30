# 量子关联成像 UNet 项目

## 项目简介
本项目实现了基于 UNet 的量子关联成像（Quantum Correlated Imaging）端到端重建流程，支持多帧 signal/idler 图像堆叠为多通道输入，灵活的损失函数组合，以及自动化实验记录。适用于量子成像、计算成像等相关方向的深度学习建模与实验。

## 主要特性
- **多帧堆叠输入**：支持 signal/idler 多帧图片，每 stack_num 张图片预叠加为 1 通道，自动适配 UNet 输入。
- **灵活损失函数**：支持 SSIM、MSE、感知损失（VGG16）等多种损失加权组合。
- **自动实验归档**：每次训练自动生成实验名，保存 config、loss 曲线、PSNR 曲线等，便于对比和复现。
- **多平台兼容**：支持 CPU、CUDA、Apple MPS。
- **详细注释与易扩展性**：代码结构清晰，便于自定义模型、损失、数据处理等。

## 目录结构
```
quantumn-corr-imaging/
  code/
    dataset.py      # 数据集定义，支持多帧堆叠与预叠加
    model.py        # UNet 模型，支持自定义 in_channels
    train.py        # 训练主流程，自动记录实验
    metric.py       # 推理与指标评估，支持PSNR/SSIM
    utils.py        # 感知损失等工具
  data/
    train/val/test/ # 数据集，结构见下
  results/
    exp_xxx/        # 每次实验自动归档
  checkpoints/
    best_model.pth  # 最优模型权重
  README.md         # 项目说明
```

## 数据格式
每个样本目录下：
```
object_dir/
  signal/   # 多帧signal图片
  idler/    # 多帧idler图片
  target.JPG
```

## 快速开始
1. **准备数据**：按上述格式放置 train/val/test 数据。
2. **训练模型**：
   ```bash
   python code/train.py
   ```
   - 支持自动保存每次实验的参数、loss曲线、psnr曲线等到 results/exp_xxx/。
3. **推理与评估**：
   ```bash
   python code/metric.py
   ```
   - 自动评估PSNR等指标，保存预测结果。

## 主要参数说明
- `stack_num`：每 stack_num 张 signal/idler 图片预叠加为 1 通道。
- `max_signal`/`max_idler`：每个样本最多取多少帧 signal/idler。
- `learning_rate`：学习率。
- `loss_weight_ssim`/`loss_weight_mse`/`loss_weight_perceptual`：损失加权。
- 其余参数详见 code/train.py。

## 实验记录与对比
- 每次训练自动生成唯一实验名（含主要参数），所有结果归档于 results/exp_xxx/。
- 包含 config.json、losses.png、psnrs.png 等，便于横向对比。

## 常见问题
- **PSNR/SSIM 只用于评估，不建议作为 loss。**
- **stack_num、max_signal、损失权重等建议先手动粗调，再用 Optuna 等自动化工具细调。**
- **如需自定义感知损失、模型结构、数据处理等，代码已高度模块化，便于扩展。**



## 拍摄相关问题

### 安装依赖
建议使用MATLAB2015B，必须安装**Image Acquisition Toolbox**

### 信号过强
先确定相机编号，再降低对应相机的Brightness。
### 拍摄缓慢
以Mindvision工业相机为例：停止^takejpg.m‘，打开'Camera1_Photos, Camera2_Photos'
1．如果图片中间出现黑线，打开 MVDCP，反复重启相机，直到预览界面中图片恢复正常。
2．如果图片全黑，重连相机后再运行。
### 摄像头被其他程序占用
1.关闭MATLAB2015B，关闭MVDCP。再重新运行'takejpg.m’。
2． 打开设备管理器，卸载摄像机驱动，重启电脑。
3．Windows更新可能占用摄像机。关闭Windows更新。

## 联系与贡献
如有问题或建议，欢迎 issue 或 PR。
