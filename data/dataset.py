import os
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class SkinLesionDataset(Dataset):
    """皮肤病变数据集加载器，借鉴REPA-E的数据处理方式"""
    
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
        try:
            all_items = os.listdir(root_dir)
            self.classes = sorted([d for d in all_items 
                                 if os.path.isdir(os.path.join(root_dir, d))])
        except FileNotFoundError:
            self.classes = []
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 设置默认的数据增强，借鉴REPA-E的处理方式
        if transform is None:
            if split == "train":
                # 训练时的数据增强
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    # transforms.RandomRotation(degrees=10),
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                    transforms.ToTensor(),
                    # 不进行标准化，保持[0, 1]范围，在训练时再处理
                ])
            else:
                # 验证时只进行resize和转换为tensor
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transform
            
        # 收集所有图像路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            if os.path.isdir(class_dir): 
                 try:
                     img_list = os.listdir(class_dir)
                     for img_name in img_list:
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_dir, img_name)
                            self.samples.append((img_path, class_idx))
                 except Exception as e:
                     # Consider logging this error instead of printing
                     pass # Suppress error printing for now

        # 数据集划分 - 移除划分逻辑，加载所有数据
        # np.random.seed(42)
        # indices = np.random.permutation(len(self.samples))
        # split_idx = int(len(indices) * 0.8)  # 80% 训练集，20% 验证集
        
        # if split == "train":
        #     self.samples = [self.samples[i] for i in indices[:split_idx]]
        # else:  # val
        #     self.samples = [self.samples[i] for i in indices[split_idx:]]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def preprocess_imgs_vae(imgs):
    """预处理图像用于VAE，借鉴REPA-E的实现"""
    # imgs: (B, C, H, W) -> (B, C, H, W), [0, 1] float32 -> [-1, 1] float32
    return imgs * 2.0 - 1.0

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