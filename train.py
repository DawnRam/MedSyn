import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import logging
from pathlib import Path

from data.dataset import get_dataloader
from utils.metrics import InceptionScore, calculate_fid, calculate_classification_accuracy, calculate_diversity
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

def parse_args():
    parser = argparse.ArgumentParser(description="训练扩散模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_type", type=str, required=True, 
                      choices=["image", "latent", "stable"], 
                      help="模型类型：image/latent/stable")
    parser.add_argument("--resume", type=str, help="从检查点恢复训练")
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
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    return rank, world_size, gpu

def get_dataloader_distributed(
    root_dir: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    split: str,
    rank: int,
    world_size: int
) -> DataLoader:
    """创建分布式数据加载器"""
    dataset = SkinLesionDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size
    )
    
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

def train_image_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size):
    """训练Image-level扩散模型"""
    # 将模型转换为DDP模型
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(config["training"]["beta1"], config["training"]["beta2"])
    )
    
    # 添加学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"].get("warmup_steps", 0),
        num_training_steps=len(train_loader) * config["training"]["num_epochs"]
    )
    
    scaler = GradScaler() if config["training"]["mixed_precision"] else None
    
    # 训练循环
    for epoch in range(config["training"]["num_epochs"]):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}", 
                   disable=rank != 0)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            with autocast(enabled=config["training"]["mixed_precision"]):
                loss = model(images, labels)
            
            # 反向传播
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config["training"]["gradient_clip_val"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config["training"]["gradient_clip_val"]
                )
                optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({"loss": loss.item()})
            
            # 记录日志
            if batch_idx % config["logging"]["log_every_n_steps"] == 0 and rank == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                if config["logging"]["use_wandb"]:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_loader) + batch_idx,
                        "train/lr": scheduler.get_last_lr()[0]
                    })
            
            # 评估
            if batch_idx % config["evaluation"]["eval_freq"] == 0 and rank == 0:
                evaluate_model(model, val_loader, config, logger, device)
        
        # 保存检查点
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
                "loss": total_loss / len(train_loader)
            }, save_dir / f"checkpoint_epoch_{epoch+1}.pt")

def train_latent_diffusion(model, autoencoder, train_loader, val_loader, config, logger, device, rank, world_size):
    """训练Latent-level扩散模型"""
    # 首先训练自编码器
    logger.info("开始训练自编码器...")
    train_autoencoder(autoencoder, train_loader, config, logger, device)
    
    # 然后训练扩散模型
    logger.info("开始训练扩散模型...")
    train_image_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size)

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
    train_image_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size)

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

def evaluate_model(model, val_loader, config, logger, device):
    """评估模型性能"""
    model.eval()
    
    # 初始化评估指标
    inception_score = InceptionScore(device=device)
    all_real_images = []
    all_gen_images = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            all_real_images.append(images)
            all_labels.append(labels)
            
            # 生成图像
            gen_images = model.sample(
                labels,
                num_inference_steps=config["sampling"]["num_inference_steps"],
                guidance_scale=config["sampling"]["guidance_scale"]
            )
            all_gen_images.append(gen_images)
    
    # 计算评估指标
    real_images = torch.cat(all_real_images, dim=0)
    gen_images = torch.cat(all_gen_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    metrics = {}
    
    # 计算FID
    if "fid" in config["evaluation"]["metrics"]:
        fid = calculate_fid(real_images, gen_images, device)
        metrics["fid"] = fid
    
    # 计算Inception Score
    if "inception_score" in config["evaluation"]["metrics"]:
        is_mean, is_std = inception_score(gen_images)
        metrics["inception_score_mean"] = is_mean
        metrics["inception_score_std"] = is_std
    
    # 计算分类准确率
    if "classification_accuracy" in config["evaluation"]["metrics"]:
        # 这里需要预训练的分类器
        # accuracy = calculate_classification_accuracy(classifier, gen_images, labels, device)
        # metrics["classification_accuracy"] = accuracy
        pass
    
    # 计算多样性
    if "diversity" in config["evaluation"]["metrics"]:
        diversity = calculate_diversity(gen_images)
        metrics["diversity"] = diversity
    
    # 记录指标
    logger.info(f"Evaluation metrics: {metrics}")
    if config["logging"]["use_wandb"]:
        wandb.log(metrics)

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
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            config[section][key] = value
    
    return config

def main():
    args = parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # 从环境变量更新配置
    config = update_config_from_env(config)
    
    # 初始化分布式训练
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f"cuda:{gpu}")
    
    # 设置日志
    logger = setup_logging(config)
    if rank == 0:
        logger.info(f"Using device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info("训练配置:")
        logger.info(yaml.dump(config, default_flow_style=False))
    
    # 设置wandb（仅在主进程上）
    if rank == 0:
        setup_wandb(config, args.model_type)
    
    # 准备数据
    train_loader = get_dataloader_distributed(
        config["data"]["root_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        split="train",
        rank=rank,
        world_size=world_size
    )
    
    val_loader = get_dataloader_distributed(
        config["data"]["root_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        split="val",
        rank=rank,
        world_size=world_size
    )
    
    # 根据模型类型选择训练函数
    if args.model_type == "image":
        from models.image_diffusion import ImageDiffusionModel
        model = ImageDiffusionModel(**config["model"]).to(device)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"从检查点恢复: {args.resume}")
        train_image_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size)
    
    elif args.model_type == "latent":
        from models.latent_diffusion import LatentDiffusionModel, Autoencoder
        model = LatentDiffusionModel(**config["model"]).to(device)
        autoencoder = Autoencoder(**config["autoencoder"]).to(device)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            ae_checkpoint = torch.load(args.resume.replace("model", "autoencoder"), map_location=device)
            autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
            logger.info(f"从检查点恢复: {args.resume}")
        train_latent_diffusion(model, autoencoder, train_loader, val_loader, config, logger, device, rank, world_size)
    
    elif args.model_type == "stable":
        from diffusers import StableDiffusionPipeline
        model = StableDiffusionPipeline.from_pretrained(
            config["model"]["pretrained_model_name_or_path"],
            torch_dtype=torch.float16 if config["model"]["torch_dtype"] == "float16" else torch.float32,
            revision=config["model"]["revision"]
        ).to(device)
        if args.resume:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.resume)
            logger.info(f"从检查点恢复: {args.resume}")
        train_stable_diffusion(model, train_loader, val_loader, config, logger, device, rank, world_size)
    
    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 