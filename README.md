# Quantum Correlation Imaging (量子关联成像) 项目说明

本项目实现了基于量子关联成像的图像重建流程，包含数据准备、模型训练、预测与结果可视化等完整流程。适用于压缩感知、量子成像等相关研究与实验。

## 目录结构
```
quantumn-corr-imaging/
├── code/                # 主要代码目录
│   ├── train.py         # 训练主程序
│   ├── metric.py        # 测试与结果保存脚本
│   ├── model.py         # 模型结构定义
│   ├── dataset.py       # 数据集加载与处理
│   ├── peek.py          # 可视化/调试工具
│   └── __pycache__/     # Python缓存
├── data/                # 数据目录
│   ├── train/           # 训练集（每个物体一个子文件夹，含 signal、idler、target.JPG）
│   ├── test/            # 测试集（结构同上）
│   └── sample_data/     # 示例数据
├── checkpoints/         # 训练模型权重
│   └── best_model.pth   # 最优模型
├── results/             # 预测结果图片
│   ├── pred_*.png       # 预测重建图像
│   └── target_*.png     # 对应真实target
├── losses.png           # 训练损失曲线
├── README.md            # 项目说明
```

## 快速开始

### 1. 安装依赖
建议使用 Python 3.8+，安装 PyTorch、torchvision、numpy、matplotlib 等依赖。

```bash
pip install torch torchvision numpy matplotlib
```

### 2. 数据准备
- 将训练/测试数据按如下结构放置：
  - `data/train/物体编号/signal/`、`idler/`、`target.JPG`
  - `data/test/物体编号/signal/`、`idler/`、`target.JPG`

### 3. 训练模型

```bash
python code/train.py
```
- 训练完成后，最优模型保存在 `checkpoints/best_model.pth`。
- 损失曲线保存在 `losses.png`。

### 4. 测试与预测

```bash
python code/metric.py
```
- 预测结果保存在 `results/` 目录下。
- `pred_*.png` 为模型重建图像，`target_*.png` 为真实target。

## 主要文件说明
- `code/train.py`：训练主程序，支持超参数集中管理。
- `code/metric.py`：批量预测与结果保存脚本。
- `code/model.py`：Transformer+MLP结构的成像重建模型。
- `code/dataset.py`：自定义数据集，支持物体级/采样mask等灵活配置。
- `code/peek.py`：可选的可视化/调试工具。

## 备注
- 可根据实际需求调整 `train.py` 和 `metric.py` 中的超参数。
- 支持自定义mask采样比例、全观测/稀疏观测等多种实验场景。
- 如需更复杂的模型结构（如UNet、卷积解码器等），可在 `model.py` 基础上扩展。

---
如有问题欢迎联系或提交 issue。
