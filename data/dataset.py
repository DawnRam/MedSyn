import os
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class SkinLesionDataset(Dataset):
    """皮肤病变数据集加载器"""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        参数:
            root_dir (str): 数据集根目录，包含各个类别的子文件夹
            split (str): 数据集划分，可选 "train" 或 "val"
            image_size (int): 图像大小
            transform (Optional[transforms.Compose]): 数据增强转换
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 设置默认的数据增强
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        # 收集所有图像路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        # 数据集划分
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * 0.8)  # 80% 训练集，20% 验证集
        
        if split == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:  # val
            self.samples = [self.samples[i] for i in indices[split_idx:]]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloader(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 256,
    split: str = "train"
) -> DataLoader:
    """
    创建数据加载器
    
    参数:
        root_dir (str): 数据集根目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载的工作进程数
        image_size (int): 图像大小
        split (str): 数据集划分
        
    返回:
        DataLoader: PyTorch数据加载器
    """
    dataset = SkinLesionDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    ) 