# Quantum Correlation Imaging (量子关联成像) 项目

本项目实现了基于深度学习的量子关联成像重建方法，支持多帧信号（signal）和闲光子（idler）图像堆叠输入，采用 UNet 网络结构进行高质量图像重建。

## 目录结构

```
quantumn-corr-imaging/
├── code/
│   ├── dataset.py         # 数据集加载与预处理
│   ├── model.py           # UNet模型定义
│   ├── train.py           # 训练主程序
│   ├── metric.py          # 推理与结果保存
│   └── utils.py           # 工具函数
├── data/
│   ├── train/             # 训练集
│   ├── val/               # 验证集
│   └── test/              # 测试集
├── checkpoints/           # 训练中保存的模型权重
├── results/               # 推理结果图片
├── losses.png             # 训练损失曲线
└── README.md              # 项目说明
```

## 数据格式
- 每个样本为一个文件夹，包含：
  - `signal/`：多张信号光子图片
  - `idler/`：多张闲光子图片
  - `target.JPG`：目标重建图像

## 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision torchmetrics matplotlib
```

### 2. 训练模型
```bash
python code/train.py
```
- 训练完成后，最优模型保存在 `checkpoints/best_model.pth`

### 3. 推理与结果保存
```bash
python code/metric.py
```
- 预测结果保存在 `results/` 目录下。
- `pred_*.png` 为模型重建图像，`target_*.png` 为真实target。

## 主要文件说明
- `code/train.py`：训练主程序，支持超参数集中管理。
- `code/metric.py`：批量预测与结果保存脚本。
- `code/model.py`：UNet模型定义。
- `code/dataset.py`：自定义数据集，支持物体级/采样mask等灵活配置。
- `code/peek.py`：可选的可视化/调试工具。

## 备注
- 可根据实际需求调整 `train.py` 和 `metric.py` 中的超参数。
- 支持自定义mask采样比例、全观测/稀疏观测等多种实验场景。
- 如需更复杂的模型结构（如UNet、卷积解码器等），可在 `model.py` 基础上扩展。
## 数据集制作指南

### 安装依赖

建议使用MATLAB2015B，必须安装**Image Acquisition Toolbox**

### 常见问题说明

#### 信号过强
先确定相机编号，再降低对应相机的Brightness。
#### 拍摄缓慢
以MindVision工业相机为例：停止 `takejpg.m `，打开  `Camera1_Photos、Camera2_Photos`
1. 如果图片中间出现黑线，打开 MVDCP ，反复重启相机，直到预览界面中图片恢复正常。
2. 如果图片全黑，重连相机后再运行。
#### 摄像头被其他程序占用
1. 关闭MATLAB2015B，关闭MVDCP。再重新运行`takejpg.m`。
2. 打开设备管理器，卸载摄像机驱动，重启电脑。
3. Windows更新可能占用摄像机。关闭Windows更新。

---
如有问题欢迎联系作者。
