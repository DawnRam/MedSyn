import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import logging
from pathlib import Path
import datetime
import torchvision
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms

from data.dataset import get_dataloader, SkinLesionDataset, preprocess_imgs_vae
from utils.metrics import InceptionScore, calculate_fid, calculate_classification_accuracy, calculate_diversity
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.transforms import Compose, ToTensor, Normalize

def parse_args():
    parser = argparse.ArgumentParser(description="训练扩散模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_type", type=str, required=True, 
                      choices=["image", "latent", "stable"], 
                      help="模型类型：image/latent/stable")
    parser.add_argument("--resume", type=str, help="从检查点恢复训练")
    parser.add_argument("--dataset", type=str, default="skin", choices=["skin", "cifar10"], help="数据集类型")
    return parser.parse_args()

def setup_logging(config):
    """设置日志"""
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_wandb(config, model_type):
    """设置Weights & Biases日志"""
    if config["logging"]["use_wandb"]:
        wandb.init(
            project=config["logging"]["wandb_project"],
            name=f"{config['logging']['wandb_name']}_{model_type}",
            config=config
        )

def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    # 设置设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    # 初始化进程组
    if world_size > 1:
        try:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
        except Exception as e:
            print(f"Failed to initialize process group: {str(e)}")
            raise e
    
    return rank, world_size, device

def get_dataset(name, root, split, image_size):
    if name == 'skin':
        return SkinLesionDataset(root_dir=root, split=split, image_size=image_size)
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        train = (split == 'train')
        return CIFAR10(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset')

def get_dataloader_distributed_dataset(dataset, batch_size, num_workers, split, rank, world_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train")
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )

def euler_sampler(model, latents, labels, noise_scheduler, num_steps=50, cfg_scale=1.0):
    """Euler采样器，借鉴REPA-E的实现，适配UNet2DModel"""
    device = latents.device
    dtype = latents.dtype
    
    # 时间步从1到0
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64, device=device)
    x_next = latents.to(torch.float64)
    
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            
            # 处理CFG
            if cfg_scale > 1.0:
                # 创建无条件标签（使用类别数作为null标签）
                null_labels = torch.full_like(labels, 7)  # 假设7是null类别
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([labels, null_labels], dim=0)
            else:
                model_input = x_cur
                y_cur = labels
            
            # 转换时间步格式为UNet2DModel期望的格式
            time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur
            # 将时间步转换为UNet2DModel期望的格式（0到num_train_timesteps-1）
            timesteps = (time_input * (noise_scheduler.config.num_train_timesteps - 1)).long()
            
            # 模型前向传播
            noise_pred = model(
                model_input.to(dtype=dtype), 
                timesteps.to(dtype=torch.long), 
                class_labels=y_cur
            ).sample.to(torch.float64)
            
            # CFG处理
            if cfg_scale > 1.0:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Euler步进
            x_next = x_cur + (t_next - t_cur) * noise_pred
    
    return x_next.to(dtype=dtype)

def train_image_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size, num_classes, dataset_name):
    """训练Image-level扩散模型，完全仿照classifier-free diffusion guidance逻辑"""
    if world_size > 1:
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["training"]["num_timesteps"],
        beta_schedule=config["training"].get("beta_schedule", "linear")
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(config["training"]["beta1"], config["training"]["beta2"])
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"].get("warmup_steps", 0),
        num_training_steps=len(train_loader) * config["training"]["num_epochs"]
    )
    scaler = GradScaler() if config["training"]["mixed_precision"] else None
    prob_cf = 0.1  # classifier-free概率
    for epoch in range(config["training"]["num_epochs"]):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}", disable=rank != 0)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = preprocess_imgs_vae(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=device).long()
            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            # classifier-free标签
            mask = torch.rand(labels.shape, device=labels.device) < prob_cf
            labels_cf = labels.clone()
            if dataset_name == 'cifar10':
                labels_cf[mask] = 10  # 10为无条件标签
            else:
                labels_cf[mask] = num_classes  # 皮肤镜用num_classes为无条件
            with autocast(enabled=config["training"]["mixed_precision"]):
                noise_pred = model(noisy_images, timesteps, class_labels=labels_cf).sample
                loss = nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_val"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_val"])
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({"loss": loss.item()})
            if (epoch > 0) and ((epoch + 1) % 10 == 0) and (batch_idx == len(train_loader) - 1):
                evaluate_model(model, val_loader, config, logger, device, noise_scheduler, epoch, num_classes, dataset_name)
                model.train()
        if (epoch + 1) % config["logging"]["save_every_n_epochs"] == 0 and rank == 0:
            save_dir = Path(config["logging"]["save_dir"])
            save_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(model, DDP):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "noise_scheduler": noise_scheduler,
                "loss": total_loss / len(train_loader)
            }, save_dir / f"checkpoint_epoch_{epoch+1}.pt")

def train_latent_diffusion(model, autoencoder, train_loader, val_loader, config, logger, device, rank, world_size):
    """训练Latent-level扩散模型"""
    # 首先训练自编码器
    logger.info("开始训练自编码器...")
    train_autoencoder(autoencoder, train_loader, config, logger, device)
    
    # 然后训练扩散模型
    logger.info("开始训练扩散模型...")
    train_image_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size, 7, "skin")

def train_stable_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size):
    """训练Stable Diffusion模型"""
    # 使用LoRA进行微调
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=config["training"]["lora"]["r"],
        lora_alpha=config["training"]["lora"]["alpha"],
        target_modules=config["training"]["lora"]["target_modules"],
        lora_dropout=config["training"]["lora"]["dropout"],
        bias=config["training"]["lora"]["bias"]
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 训练循环
    train_image_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size, 7, "skin")

def train_autoencoder(autoencoder, train_loader, config, logger, device):
    """训练自编码器"""
    optimizer = AdamW(
        autoencoder.parameters(),
        lr=config["training"]["ae_learning_rate"],
        weight_decay=config["training"]["ae_weight_decay"],
        betas=(config["training"]["ae_beta1"], config["training"]["ae_beta2"])
    )
    
    for epoch in range(config["training"]["ae_num_epochs"]):
        autoencoder.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Autoencoder Epoch {epoch+1}/{config['training']['ae_num_epochs']}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # 前向传播
            reconstructed, latent = autoencoder(images)
            loss = nn.MSELoss()(reconstructed, images)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
            if batch_idx % config["logging"]["log_every_n_steps"] == 0:
                logger.info(f"Autoencoder Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                if config["logging"]["use_wandb"]:
                    wandb.log({
                        "autoencoder/train/loss": loss.item(),
                        "autoencoder/train/epoch": epoch + 1,
                        "autoencoder/train/step": epoch * len(train_loader) + batch_idx
                    })

def evaluate_model(model, val_loader, config, logger, device, noise_scheduler, epoch, num_classes, dataset_name):
    """评估函数，完全仿照classifier-free diffusion采样逻辑"""
    model.eval()
    num_samples_per_class = 8 if dataset_name == 'cifar10' else 4
    image_size = config["data"]["image_size"]
    guidance_scale = 3.0 if dataset_name == 'cifar10' else 4.0
    noise_scheduler.set_timesteps(50)
    save_dir = Path(config["logging"]["save_dir"]) / "samples"
    if not dist.is_initialized() or dist.get_rank() == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    all_samples = []
    with torch.no_grad():
        for class_idx in range(num_classes):
            labels = torch.full((num_samples_per_class,), class_idx, device=device)
            latents = torch.randn((num_samples_per_class, 3, image_size, image_size), device=device)
            for t in noise_scheduler.timesteps:
                if dataset_name == 'cifar10':
                    labels_uncond = torch.full_like(labels, 10)
                else:
                    labels_uncond = torch.full_like(labels, num_classes)
                model_input = torch.cat([latents, latents], dim=0)
                labels_input = torch.cat([labels, labels_uncond], dim=0)
                noise_pred = model(model_input, t, class_labels=labels_input).sample
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            all_samples.append(latents)
        all_samples = torch.cat(all_samples, dim=0)
        images = (all_samples.clamp(-1, 1) + 1) / 2
        if not dist.is_initialized() or dist.get_rank() == 0:
            for idx, img in enumerate(images):
                class_idx = idx // num_samples_per_class
                sample_idx = idx % num_samples_per_class
                img_path = save_dir / f"sample_class_{class_idx:02d}_idx_{sample_idx:03d}.png"
                torchvision.utils.save_image(img, img_path)
            # 直接拼成grid，每行一个类别（如10行8列，每行同一类别）
            grid = torchvision.utils.make_grid(
                images,
                nrow=num_samples_per_class,  # 每行num_samples_per_class张
                padding=2,
                normalize=False
            )
            grid_path = save_dir / f"samples_grid_epoch_{epoch+1:03d}.png"
            torchvision.utils.save_image(grid, grid_path)
            logger.info(f"Saved generated samples grid to {grid_path}")
        if config["logging"]["use_wandb"] and (not dist.is_initialized() or dist.get_rank() == 0):
            wandb.log({
                "val/generated_images_grid": wandb.Image(grid.cpu()),
                "val/generated_images": [wandb.Image(img) for img in images.cpu()]
            })
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info("Evaluation completed - generated samples for visualization")

def update_config_from_env(config):
    """从环境变量更新配置"""
    env_mapping = {
        "TRAIN_BATCH_SIZE": ("data", "batch_size"),
        "TRAIN_NUM_WORKERS": ("data", "num_workers"),
        "TRAIN_MIXED_PRECISION": ("training", "mixed_precision"),
        "TRAIN_ACCUMULATION_STEPS": ("training", "accumulate_grad_batches"),
        "TRAIN_WARMUP_STEPS": ("training", "warmup_steps"),
        "TRAIN_LEARNING_RATE": ("training", "learning_rate"),
        "TRAIN_NUM_EPOCHS": ("training", "num_epochs"),
        "WANDB_PROJECT": ("logging", "wandb_project"),
        "WANDB_NAME": ("logging", "wandb_name")
    }
    
    for env_var, (section, key) in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            # 转换布尔值
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            # 转换数值
            else:
                try:
                    # 尝试转换为整数
                    value = int(value)
                except ValueError:
                    try:
                        # 尝试转换为浮点数
                        value = float(value)
                    except ValueError:
                        # 如果不是数值，则保留为字符串
                        pass
            config[section][key] = value
    
    return config

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # 动态参数设置
    if args.dataset == 'skin':
        num_classes = 7
        image_size = 256
        num_class_embeds = 8
        block_out_channels = [128, 256, 512, 512]
        data_root = config["data"]["root_dir"]
    elif args.dataset == 'cifar10':
        num_classes = 10
        image_size = 32
        num_class_embeds = 11
        block_out_channels = [64, 128, 256, 256]
        data_root = './cifar10_data'
    else:
        raise ValueError('Unknown dataset')
    # 分布式设置
    rank, world_size, device = setup_distributed()
    logger = setup_logging(config)
    if rank == 0:
        logger.info(f"Using device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Rank: {rank}")
        logger.info("训练配置:")
        logger.info(yaml.dump(config, default_flow_style=False))
    if rank == 0:
        setup_wandb(config, args.model_type)
    # 数据集加载
    train_dataset = get_dataset(args.dataset, data_root, 'train', image_size)
    val_dataset = get_dataset(args.dataset, data_root, 'val', image_size)
    train_loader = get_dataloader_distributed_dataset(
        train_dataset,
        batch_size=2 if args.dataset == 'cifar10' else config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        split="train",
        rank=rank,
        world_size=world_size
    )
    val_loader = get_dataloader_distributed_dataset(
        val_dataset,
        batch_size=2 if args.dataset == 'cifar10' else config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        split="val",
        rank=rank,
        world_size=world_size
    )
    # 构建UNet2DModel
    unet_config = config["model"].copy()
    unet_config['sample_size'] = image_size
    unet_config['block_out_channels'] = block_out_channels
    unet_config['num_class_embeds'] = num_class_embeds
    unet_config['in_channels'] = 3
    unet_config['out_channels'] = 3
    model = UNet2DModel.from_config(unet_config).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"从检查点恢复: {args.resume}")
    try:
        train_image_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size, num_classes, args.dataset)
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main() 