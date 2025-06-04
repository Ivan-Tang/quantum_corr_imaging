# 端到端关联成像 UNet 项目

## 项目简介
本项目基于UNet结构，针对量子关联成像（Ghost Imaging）任务，支持多帧signal/idler图像堆叠为多通道输入，实现端到端的高质量重建。支持多种损失函数加权组合（SSIM、MSE、感知损失），自动实验归档、超参数搜索、推理评估与后处理。兼容多种设备（CPU/CUDA/MPS），并支持形态学后处理接口扩展。

---

## 目录结构
```
quantumn-corr-imaging/
  code/
    dataset.py      # 数据集定义，支持多帧堆叠与预叠加
    model.py        # UNet模型，支持自定义in_channels
    train.py        # 训练主流程，自动归档与评估
    metric.py       # 推理与PSNR/SSIM评测，支持后处理
    optuna_search.py# 超参数自动搜索
    utils.py        # 感知损失等工具
    ...
  data/
    train/val/test/ # 数据集，结构见下
  results/
    exp_xxx/        # 每次实验自动归档
  checkpoints/
    best_model.pth  # 最优模型权重
  reports/
    quantum_corr_imaging_presentation.tex/pdf # 自动生成的汇报PPT
  README.md         # 项目说明
```

---

## 原理简介
- **量子关联成像**：利用信号光与闲光的统计相关性，通过多帧采集与重建算法恢复目标图像。
- **多通道堆叠**：将多帧signal/idler图片每stack_num张预叠加为1通道，输入UNet，提升重建质量。
- **损失函数**：支持SSIM、MSE、感知损失加权组合，兼顾结构、像素与感知一致性。
- **评测指标**：主要采用PSNR（峰值信噪比）、SSIM等，自动评估与归档。

---

## 数据格式
每个样本目录下：
```
object_dir/
  signal/   # 多帧signal图片
  idler/    # 多帧idler图片
  target.JPG
```

---

## 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision torchmetrics matplotlib optuna pillow
```

### 2. 数据准备
- 将原始数据放入`data/train/`、`data/val/`、`data/test/`等目录，结构参考`GhostImagingDataset`实现。

### 3. 训练模型
```bash
python code/train.py
```
- 自动归档实验到`results/exp_xxx/`，保存config、loss/psnr曲线、最优模型等。
- 训练结束后自动调用推理评估，输出预测图片和PSNR结果。

### 4. 超参数搜索
```bash
python code/optuna_search.py
```
- 支持Optuna自动调参，结果自动保存为csv/json/txt。

### 5. 推理与评测
- 训练结束后自动推理评估，也可单独运行`metric.py`，自动读取最新实验参数，输出预测图片、PSNR/SSIM等。

---

## 主要功能
- **多帧堆叠输入**：支持signal/idler多帧堆叠为多通道输入，shape自适应。
- **端到端UNet重建**：支持自定义输入通道数，适配多帧输入。
- **多损失加权**：SSIM、MSE、感知损失可加权组合，提升重建质量。
- **自动实验归档**：每次实验自动保存config、曲线、模型，便于复现与对比。
- **自动推理评估**：训练后自动推理，输出预测图片、PSNR/SSIM等指标。
- **超参数搜索**：集成Optuna，支持自动调参与早停。
- **后处理接口**：预留形态学后处理（如闭运算、膨胀等）接口，修复输出断裂。
- **设备兼容**：支持CPU、CUDA、MPS自动切换。
- **可视化与汇报**：自动生成loss/psnr曲线、实验结果图片，支持beamer PPT自动化汇报。

---

## 自动化测试与CI

本项目已集成自动化smoke测试脚本 `code/test_pipeline.py`，可一键验证主流程（训练、推理、超参数搜索）是否正常运行，并自动清理测试产物。

- **本地测试**：
  ```bash
  python code/test_pipeline.py
  ```
  运行后会依次测试训练、推理、optuna调参主流程，所有测试产物自动删除。

- **CI集成**：
  推荐将 `test_pipeline.py` 集成到GitHub Actions等CI工具，保障每次提交主流程可用。
  可参考如下workflow配置：
  ```yaml
  name: Python CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - uses: actions/setup-python@v4
          with:
            python-version: '3.9'
        - run: pip install torch torchvision torchmetrics matplotlib optuna pillow
        - run: python code/test_pipeline.py
  ```

---

## 主要参数说明
- `stack_num`：每stack_num张signal/idler图片预叠加为1通道。
- `max_signal`/`max_idler`：每个样本最多取多少帧signal/idler。
- `learning_rate`：学习率。
- `loss_weight_ssim`/`loss_weight_mse`/`loss_weight_perceptual`：损失加权。
- 其余参数详见`code/train.py`。

---

## 实验记录与对比
- 每次训练自动生成唯一实验名（含主要参数），所有结果归档于`results/exp_xxx/`。
- 包含`config.json`、`losses.png`、`psnrs.png`等，便于横向对比。

---

## 评测指标说明
- **PSNR**：峰值信噪比，反映重建图像与目标图像的相对误差，数值越高表示重建质量越好。常用范围20-40dB。
- **SSIM**：结构相似性，衡量结构信息保真度，1为最优。

---

## 调参建议
- **stack_num**：增大可提升信噪比，但过大可能丢失细节。
- **loss权重**：建议以SSIM为主，MSE/感知损失为辅，具体可用Optuna搜索。
- **学习率**：建议1e-3~1e-4，支持自动调参。
- **后处理**：可用形态学操作修复断裂，或再训一个UNet做后处理。

---

## 常见问题与注意事项
- **图像shape顺序**：所有图片resize为(img_size[0], img_size[1])，即(512, 384)，保存/可视化时需确保shape为(H, W)，避免H/W混淆。
- **设备兼容**：所有数据与模型均to(device)，避免设备不一致报错。
- **实验归档**：每次实验自动归档，便于复现与对比。
- **后处理**：如需集成形态学操作，可在`metric.py`中调用`scipy.ndimage`或`cv2`相关API。

---

## 拍摄相关问题

### 安装依赖
建议使用MATLAB2015B，必须安装**Image Acquisition Toolbox**

### 信号过强
先确定相机编号，再降低对应相机的Brightness。

### 拍摄缓慢
以Mindvision工业相机为例：停止`takejpg.m`，打开`Camera1_Photos, Camera2_Photos`
1. 如果图片中间出现黑线，打开MVDCP，反复重启相机，直到预览界面中图片恢复正常。
2. 如果图片全黑，重连相机后再运行。

### 摄像头被其他程序占用
1. 关闭MATLAB2015B，关闭MVDCP。再重新运行`takejpg.m`。
2. 打开设备管理器，卸载摄像机驱动，重启电脑。
3. Windows更新可能占用摄像机。关闭Windows更新。

---

## 扩展与TODO
- 集成形态学后处理到`metric.py`，如闭运算、膨胀等。
- 支持“再训一个UNet做后处理”。
- 批量评测、实验对比、更多可视化等功能扩展。

---

## 联系与贡献
如有问题或建议，欢迎issue或PR。
