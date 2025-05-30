# 皮肤病变分类数据集的扩散模型实验

本项目实现了基于扩散模型的皮肤病变图像生成实验，包括以下实验：

1. Image-level Conditional Diffusion Model (Classifier-free Guidance)
2. Latent-level Conditional Diffusion Model
3. Stable Diffusion Fine-tuning

## 项目结构

```
.
├── configs/                 # 配置文件目录
│   ├── image_diffusion.yaml
│   ├── latent_diffusion.yaml
│   └── stable_diffusion.yaml
├── data/                    # 数据处理相关代码
│   ├── dataset.py          # 数据集加载和预处理
│   └── transforms.py       # 数据增强和转换
├── models/                  # 模型定义
│   ├── image_diffusion.py  # Image-level扩散模型
│   ├── latent_diffusion.py # Latent-level扩散模型
│   └── stable_diffusion.py # Stable Diffusion相关代码
├── utils/                   # 工具函数
│   ├── metrics.py          # 评估指标计算
│   └── visualization.py    # 可视化工具
├── train.py                 # 训练脚本
├── evaluate.py             # 评估脚本
└── requirements.txt        # 项目依赖
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 数据集准备

将数据集放置在 `data/raw/` 目录下，每个类别对应一个子文件夹。

## 使用方法

1. 训练 Image-level Diffusion Model:
```bash
python train.py --config configs/image_diffusion.yaml
```

2. 训练 Latent-level Diffusion Model:
```bash
python train.py --config configs/latent_diffusion.yaml
```

3. Stable Diffusion Fine-tuning:
```bash
python train.py --config configs/stable_diffusion.yaml
```

4. 评估生成结果:
```bash
python evaluate.py --config configs/evaluation.yaml
```

## 评估指标

- FID (Fréchet Inception Distance)
- IS (Inception Score)
- 分类准确率
- 多样性指标 