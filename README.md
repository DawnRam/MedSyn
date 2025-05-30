# Medical Image Synthesis with Diffusion Models

This project implements diffusion model-based experiments for medical image synthesis, focusing on skin lesion classification dataset. The experiments include:

1. Image-level Conditional Diffusion Model (Classifier-free Guidance)
2. Latent-level Conditional Diffusion Model
3. Stable Diffusion Fine-tuning

## Project Structure

```
.
├── configs/                 # Configuration files
│   ├── image_diffusion.yaml
│   ├── latent_diffusion.yaml
│   └── stable_diffusion.yaml
├── data/                    # Data processing
│   ├── dataset.py          # Dataset loading and preprocessing
│   └── transforms.py       # Data augmentation and transformations
├── models/                  # Model definitions
│   ├── image_diffusion.py  # Image-level diffusion model
│   ├── latent_diffusion.py # Latent-level diffusion model
│   └── stable_diffusion.py # Stable Diffusion related code
├── utils/                   # Utility functions
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Visualization tools
├── train.py                 # Training script
├── evaluate.py             # Evaluation script
└── requirements.txt        # Project dependencies
```

## Environment Setup

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Place your dataset in the specified data directory (default: `/home/eechengyang/Data/ISIC`), with each class in a separate subfolder. You can specify a custom data directory using the `-d` or `--data-dir` option when running the training script.

## Usage

1. Train Image-level Diffusion Model:
```bash
python train.py --config configs/image_diffusion.yaml
```

2. Train Latent-level Diffusion Model:
```bash
python train.py --config configs/latent_diffusion.yaml
```

3. Fine-tune Stable Diffusion:
```bash
python train.py --config configs/stable_diffusion.yaml
```

4. Evaluate Generated Images:
```bash
python evaluate.py --config configs/evaluation.yaml
```

## Evaluation Metrics

- FID (Fréchet Inception Distance)
- IS (Inception Score)
- Classification Accuracy
- Diversity Metrics

## Features

- Multi-GPU training support with DistributedDataParallel
- Mixed precision training for faster training
- Gradient accumulation for larger effective batch sizes
- Learning rate scheduling with warmup
- Comprehensive evaluation metrics
- Weights & Biases integration for experiment tracking
- Checkpoint saving and resuming
- Data augmentation and preprocessing pipeline

## TODO List

- [ ] Implement Representation Alignment
- [ ] Implement Density-aware Sampling
- [ ] Add Medical Image-specific Evaluation Metrics
- [ ] Implement adaptive learning rate scheduling
- [ ] Add gradient clipping and regularization
- [ ] Optimize multi-GPU training efficiency
- [ ] Implement Medical Image-specific Augmentations
- [ ] Model quantization and optimization
- [ ] Batch inference optimization
- [ ] Add Deployment documentation and examples

