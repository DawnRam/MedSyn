import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_channels, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # 时间嵌入
        time_emb = self.time_mlp(t)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        h = h + time_emb
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """注意力块"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                           qkv=3, heads=self.num_heads)
        
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                       heads=self.num_heads, h=H, w=W)
        
        return self.proj(out) + x

class Downsample(nn.Module):
    """下采样块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """上采样块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class ImageDiffusionModel(nn.Module):
    """Image-level条件扩散模型"""
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        dropout=0.1,
        channel_mult=[1, 2, 2, 2],
        num_heads=8,
        num_classes=7,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 类别嵌入
        self.label_embed = nn.Embedding(num_classes, time_embed_dim)
        
        # 下采样
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, model_channels * mult, time_embed_dim, dropout)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.input_blocks.append(nn.Sequential(*layers))
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch))
                ds *= 2
        
        # 中间块
        self.middle_block = nn.Sequential(
            ResBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, ch, time_embed_dim, dropout)
        )
        
        # 上采样
        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(ch + model_channels * mult, model_channels * mult, 
                            time_embed_dim, dropout)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        
        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, labels, t):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入图像 [B, C, H, W]
            labels (torch.Tensor): 类别标签 [B]
            t (torch.Tensor): 时间步 [B]
        """
        # 时间嵌入
        # 模仿标准的Sinusoidal位置编码并映射到model_channels维度
        # 这里的实现是一个简化的版本，标准Diffusers实现更复杂
        timesteps = t.long()
        # Sinusoidal embeddings
        freqs = torch.exp(
            -torch.arange(0, self.model_channels, 2, dtype=torch.float32, device=t.device) *
            torch.log(torch.tensor(10000.0, device=t.device)) / (self.model_channels - 2)
        )
        # Ensure freqs has the correct dtype for mixed precision if needed
        if self.time_embed[0].weight.dtype == torch.float16:
             freqs = freqs.to(dtype=torch.float16)

        # Apply sinusoidal embeddings
        # t has shape (B,), freqs has shape (model_channels/2,)
        # Unsqueeze t to (B, 1) for broadcasting
        # Output shape will be (B, model_channels/2)
        t_emb = timesteps.unsqueeze(-1) * freqs # Shape: (B, model_channels / 2)
        # Concatenate sin and cos embeddings
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1) # Shape: (B, model_channels)

        # 现在t_emb的形状是 (B, model_channels)，可以传递给self.time_embed
        t = self.time_embed(t_emb)
        
        # 类别嵌入
        label_emb = self.label_embed(labels)
        t = t + label_emb
        
        # 下采样
        h = x
        hs = []
        for module in self.input_blocks:
            h = module(h, t)
            hs.append(h)
        
        # 中间块
        h = self.middle_block(h, t)
        
        # 上采样
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, t)
        
        return self.out(h)
    
    def get_loss(self, x, labels, t, noise=None):
        """计算损失"""
        if noise is None:
            noise = torch.randn_like(x)
        
        x_noisy = self.q_sample(x, t, noise)
        predicted = self(x_noisy, labels, t)
        
        return F.mse_loss(predicted, noise)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 计算噪声调度
        alphas = self.get_noise_schedule(t)
        alphas = alphas.view(-1, 1, 1, 1)
        
        return torch.sqrt(alphas) * x_start + torch.sqrt(1 - alphas) * noise
    
    def get_noise_schedule(self, t):
        """获取噪声调度"""
        # 线性噪声调度
        return 1 - t / 1000
    
    @torch.no_grad()
    def sample(self, labels, num_inference_steps=50, guidance_scale=7.5):
        """采样生成图像"""
        device = labels.device
        b = labels.shape[0]
        
        # 初始化随机噪声
        x = torch.randn(b, self.in_channels, 64, 64, device=device)
        
        # 逐步去噪
        for i in tqdm(reversed(range(num_inference_steps)), desc='Sampling'):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            # 无条件和条件预测
            noise_pred_uncond = self(x, torch.zeros_like(labels), t)
            noise_pred_cond = self(x, labels, t)
            
            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 更新x
            alpha = self.get_noise_schedule(t)
            alpha_prev = self.get_noise_schedule(t - 1) if i > 0 else torch.ones_like(alpha)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha)) * noise_pred)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(1 - alpha_prev) * noise
        
        return x 