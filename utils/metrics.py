import torch
import numpy as np
from torch import nn
from torchvision.models import inception_v3
from scipy import linalg
from pytorch_fid import fid_score
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

class InceptionScore(nn.Module):
    """计算Inception Score"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        
    @torch.no_grad()
    def forward(self, images, splits=10):
        """
        计算Inception Score
        
        参数:
            images (torch.Tensor): 生成的图像批次 [B, C, H, W]
            splits (int): 计算平均值的分割数
            
        返回:
            float: Inception Score
            float: Inception Score的标准差
        """
        preds = []
        batch_size = images.size(0)
        
        for i in range(0, batch_size, 100):
            batch = images[i:i+100].to(self.device)
            pred = F.softmax(self.model(batch), dim=1)
            preds.append(pred.cpu())
            
        preds = torch.cat(preds, 0)
        
        # 计算每个分割的KL散度
        scores = []
        for i in range(splits):
            part = preds[i * (batch_size // splits):(i + 1) * (batch_size // splits)]
            kl = part * (torch.log(part) - torch.log(torch.mean(part, 0, keepdim=True)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
            
        scores = torch.stack(scores)
        return scores.mean().item(), scores.std().item()

def calculate_fid(real_images, gen_images, device='cuda'):
    """
    计算FID分数
    
    参数:
        real_images (torch.Tensor): 真实图像
        gen_images (torch.Tensor): 生成的图像
        device (str): 计算设备
        
    返回:
        float: FID分数
    """
    # 将图像保存为临时文件
    import tempfile
    import os
    from torchvision.utils import save_image
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        real_dir = os.path.join(tmp_dir, 'real')
        gen_dir = os.path.join(tmp_dir, 'gen')
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)
        
        # 保存图像
        for i, img in enumerate(real_images):
            save_image(img, os.path.join(real_dir, f'{i}.png'))
        for i, img in enumerate(gen_images):
            save_image(img, os.path.join(gen_dir, f'{i}.png'))
            
        # 计算FID
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir],
            batch_size=50,
            device=device,
            dims=2048
        )
        
    return fid

def calculate_classification_accuracy(classifier, gen_images, labels, device='cuda'):
    """
    计算生成图像的分类准确率
    
    参数:
        classifier (nn.Module): 预训练的分类器
        gen_images (torch.Tensor): 生成的图像
        labels (torch.Tensor): 真实标签
        device (str): 计算设备
        
    返回:
        float: 分类准确率
    """
    classifier.eval()
    with torch.no_grad():
        preds = classifier(gen_images.to(device))
        pred_labels = torch.argmax(preds, dim=1)
        accuracy = (pred_labels == labels.to(device)).float().mean().item()
    return accuracy

def calculate_diversity(gen_images):
    """
    计算生成图像的多样性指标
    
    参数:
        gen_images (torch.Tensor): 生成的图像 [B, C, H, W]
        
    返回:
        float: 多样性分数 (平均L2距离)
    """
    # 将图像展平为向量
    images_flat = gen_images.view(gen_images.size(0), -1)
    
    # 计算所有图像对之间的L2距离
    distances = []
    for i in range(len(images_flat)):
        for j in range(i + 1, len(images_flat)):
            dist = torch.norm(images_flat[i] - images_flat[j], p=2)
            distances.append(dist.item())
            
    return np.mean(distances) if distances else 0.0 