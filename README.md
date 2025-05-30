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

Place the dataset in the `data/raw/` directory, with each class in a separate subfolder.

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

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+
- diffusers
- transformers
- wandb
- scikit-learn
- scipy
- numpy
- pillow
- tqdm
- yaml

## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@misc{medsyn2024,
  author = {Your Name},
  title = {Medical Image Synthesis with Diffusion Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/MedSyn}
}
``` 