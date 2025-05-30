import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid, save_image

from data.dataset import get_dataloader
from utils.metrics import InceptionScore, calculate_fid, calculate_classification_accuracy, calculate_diversity

def parse_args():
    parser = argparse.ArgumentParser(description="评估扩散模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_type", type=str, required=True, 
                      choices=["image", "latent", "stable"], 
                      help="模型类型：image/latent/stable")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="评估结果输出目录")
    parser.add_argument("--num_samples", type=int, default=1000, help="每个类别生成的样本数")
    return parser.parse_args()

def setup_logging(output_dir):
    """设置日志"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "evaluate.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(model_type, config, checkpoint_path, device):
    """加载模型"""
    if model_type == "image":
        from models.image_diffusion import ImageDiffusionModel
        model = ImageDiffusionModel(**config["model"]).to(device)
    
    elif model_type == "latent":
        from models.latent_diffusion import LatentDiffusionModel, Autoencoder
        model = LatentDiffusionModel(**config["model"]).to(device)
        autoencoder = Autoencoder(**config["autoencoder"]).to(device)
        # 加载自编码器检查点
        ae_checkpoint = torch.load(checkpoint_path.replace("model", "autoencoder"))
        autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
        model.autoencoder = autoencoder
    
    elif model_type == "stable":
        from diffusers import StableDiffusionPipeline
        from peft import PeftModel
        model = StableDiffusionPipeline.from_pretrained(
            config["model"]["pretrained_model_name_or_path"],
            torch_dtype=torch.float16 if config["model"]["torch_dtype"] == "float16" else torch.float32,
            revision=config["model"]["revision"]
        ).to(device)
        # 加载LoRA权重
        model = PeftModel.from_pretrained(model, checkpoint_path)
    
    # 加载模型检查点
    checkpoint = torch.load(checkpoint_path)
    if model_type != "stable":  # Stable Diffusion使用PeftModel加载
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    return model

def generate_samples(model, num_classes, num_samples_per_class, config, device):
    """生成样本"""
    all_samples = []
    all_labels = []
    
    for class_idx in range(num_classes):
        labels = torch.full((num_samples_per_class,), class_idx, device=device)
        
        with torch.no_grad():
            samples = model.sample(
                labels,
                num_inference_steps=config["sampling"]["num_inference_steps"],
                guidance_scale=config["sampling"]["guidance_scale"]
            )
        
        all_samples.append(samples)
        all_labels.append(labels)
    
    return torch.cat(all_samples, dim=0), torch.cat(all_labels, dim=0)

def calculate_metrics(model, val_loader, gen_samples, gen_labels, config, device):
    """计算评估指标"""
    metrics = {}
    
    # 获取真实图像
    real_images = []
    real_labels = []
    for images, labels in val_loader:
        real_images.append(images.to(device))
        real_labels.append(labels.to(device))
    real_images = torch.cat(real_images, dim=0)
    real_labels = torch.cat(real_labels, dim=0)
    
    # 计算FID
    if "fid" in config["evaluation"]["metrics"]:
        fid = calculate_fid(real_images, gen_samples, device)
        metrics["fid"] = fid
    
    # 计算Inception Score
    if "inception_score" in config["evaluation"]["metrics"]:
        inception_score = InceptionScore(device=device)
        is_mean, is_std = inception_score(gen_samples)
        metrics["inception_score_mean"] = is_mean
        metrics["inception_score_std"] = is_std
    
    # 计算分类准确率
    if "classification_accuracy" in config["evaluation"]["metrics"]:
        # 这里需要预训练的分类器
        # accuracy = calculate_classification_accuracy(classifier, gen_samples, gen_labels, device)
        # metrics["classification_accuracy"] = accuracy
        pass
    
    # 计算多样性
    if "diversity" in config["evaluation"]["metrics"]:
        diversity = calculate_diversity(gen_samples)
        metrics["diversity"] = diversity
    
    return metrics

def visualize_samples(samples, labels, num_classes, output_dir):
    """可视化生成的样本"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个类别创建网格图
    for class_idx in range(num_classes):
        class_samples = samples[labels == class_idx]
        if len(class_samples) > 0:
            grid = make_grid(class_samples[:64], nrow=8, normalize=True)
            save_image(grid, output_dir / f"class_{class_idx}_samples.png")
    
    # 创建所有类别的对比图
    samples_per_class = 8
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(20, 2.5*num_classes))
    
    for class_idx in range(num_classes):
        class_samples = samples[labels == class_idx][:samples_per_class]
        for i, sample in enumerate(class_samples):
            if num_classes > 1:
                ax = axes[class_idx, i]
            else:
                ax = axes[i]
            ax.imshow(sample.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
            ax.axis('off')
            if i == 0:
                ax.set_title(f"Class {class_idx}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "all_classes_comparison.png")
    plt.close()

def main():
    args = parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    logger.info(f"Using device: {device}")
    
    # 加载模型
    model = load_model(args.model_type, config, args.checkpoint, device)
    
    # 准备验证数据加载器
    val_loader = get_dataloader(
        config["data"]["root_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        split="val"
    )
    
    # 生成样本
    logger.info("Generating samples...")
    samples, labels = generate_samples(
        model,
        config["model"]["num_classes"],
        args.num_samples // config["model"]["num_classes"],
        config,
        device
    )
    
    # 计算评估指标
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(model, val_loader, samples, labels, config, device)
    
    # 记录指标
    logger.info(f"Evaluation metrics: {metrics}")
    with open(Path(args.output_dir) / "metrics.txt", "w") as f:
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value}\n")
    
    # 可视化样本
    logger.info("Visualizing samples...")
    visualize_samples(samples, labels, config["model"]["num_classes"], args.output_dir)

if __name__ == "__main__":
    main() 