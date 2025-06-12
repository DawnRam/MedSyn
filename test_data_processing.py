#!/usr/bin/env python3
"""
测试数据处理和生成功能
"""

import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import SkinLesionDataset, preprocess_imgs_vae
from diffusers import UNet2DModel, DDPMScheduler

def test_data_processing():
    """测试数据处理功能"""
    print("测试数据处理功能...")
    
    # 创建数据集
    dataset = SkinLesionDataset(
        root_dir="/home/eechengyang/Data/ISIC",
        split="train",
        image_size=256
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数: {len(dataset.classes)}")
    print(f"类别映射: {dataset.class_to_idx}")
    
    # 测试数据加载
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"图像形状: {image.shape}")
        print(f"图像范围: [{image.min():.3f}, {image.max():.3f}]")
        print(f"标签: {label}")
        
        # 测试预处理
        image_processed = preprocess_imgs_vae(image.unsqueeze(0))
        print(f"预处理后图像范围: [{image_processed.min():.3f}, {image_processed.max():.3f}]")
        
        # 保存测试图像
        save_dir = Path("test_outputs")
        save_dir.mkdir(exist_ok=True)
        
        # 保存原始图像
        torchvision.utils.save_image(image, save_dir / "original_image.png")
        
        # 保存预处理后的图像
        torchvision.utils.save_image(
            (image_processed.squeeze() + 1) / 2, 
            save_dir / "processed_image.png"
        )
        
        print("测试图像已保存到 test_outputs/ 目录")
    else:
        print("警告: 数据集为空，请检查数据路径")

def test_model_creation():
    """测试模型创建功能"""
    print("\n测试模型创建功能...")
    
    # 创建UNet2DModel配置
    model_config = {
        'sample_size': 256,
        'in_channels': 3,
        'out_channels': 3,
        'layers_per_block': 2,
        'block_out_channels': [128, 256, 512, 512],
        'down_block_types': ["DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"],
        'up_block_types': ["AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"],
        'mid_block_type': "UNetMidBlock2DCrossAttn",
        'cross_attention_dim': 768,
        'attention_head_dim': 8,
        'dropout': 0.1,
        'num_class_embeds': 7,
        'class_embed_type': "timestep",
    }
    
    try:
        # 创建模型
        model = UNet2DModel.from_config(model_config)
        print(f"模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 创建测试输入
        batch_size = 2
        sample = torch.randn(batch_size, 3, 256, 256).to(device)
        timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        class_labels = torch.randint(0, 7, (batch_size,)).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(sample, timesteps, class_labels=class_labels)
            print(f"模型输出形状: {output.sample.shape}")
            print("模型前向传播测试成功")
            
    except Exception as e:
        print(f"模型创建或测试失败: {str(e)}")

def test_noise_scheduler():
    """测试噪声调度器"""
    print("\n测试噪声调度器...")
    
    try:
        # 创建噪声调度器
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear"
        )
        
        # 创建测试图像
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = torch.randn(1, 3, 256, 256).to(device)
        
        # 添加噪声
        timesteps = torch.randint(0, 1000, (1,)).to(device)
        noise = torch.randn_like(image)
        noisy_image = noise_scheduler.add_noise(image, noise, timesteps)
        
        print(f"原始图像范围: [{image.min():.3f}, {image.max():.3f}]")
        print(f"噪声图像范围: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
        print("噪声调度器测试成功")
        
    except Exception as e:
        print(f"噪声调度器测试失败: {str(e)}")

def main():
    """主函数"""
    print("开始测试数据处理和生成功能...")
    
    # 测试数据处理
    test_data_processing()
    
    # 测试模型创建
    test_model_creation()
    
    # 测试噪声调度器
    test_noise_scheduler()
    
    print("\n所有测试完成!")

if __name__ == "__main__":
    main() 