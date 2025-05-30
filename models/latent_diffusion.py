import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Autoencoder(nn.Module):
    """自编码器"""
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        hidden_dims=[64, 128, 256, 512],
        latent_dim=4,
        image_size=256,
        use_attention=True,
        dropout=0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.use_attention = use_attention
        
        # 编码器
        modules = []
        in_channels = self.in_channels
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = h_dim
        
        if use_attention:
            modules.append(SelfAttention(hidden_dims[-1]))
        
        self.encoder = nn.Sequential(*modules)
        
        # 计算编码器输出大小
        self.encoded_size = self._get_encoded_size()
        
        # 潜在空间映射
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.encoded_size[0] * self.encoded_size[1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.encoded_size[0] * self.encoded_size[1], latent_dim)
        
        # 解码器
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.encoded_size[0] * self.encoded_size[1])
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ])
        
        modules.extend([
            nn.ConvTranspose2d(hidden_dims[-1], out_channels,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        ])
        
        self.decoder = nn.Sequential(*modules)
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def _get_encoded_size(self):
        """计算编码器输出大小"""
        x = torch.randn(1, self.in_channels, self.image_size, self.image_size)
        for layer in self.encoder:
            x = layer(x)
        return x.shape[2:]
    
    def encode(self, x):
        """编码"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """重参数化"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """解码"""
        z = self.decoder_input(z)
        z = z.view(z.size(0), -1, self.encoded_size[0], self.encoded_size[1])
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), z

class SelfAttention(nn.Module):
    """自注意力层"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, H * W)
        v = self.value(x).view(batch_size, -1, H * W)
        
        attention = F.softmax(torch.bmm(q, k), dim=2)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class LatentDiffusionModel(nn.Module):
    """Latent-level条件扩散模型"""
    def __init__(
        self,
        in_channels=4,  # latent_dim
        out_channels=4,
        model_channels=256,
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
        
        # 使用与ImageDiffusionModel相同的UNet架构
        from .image_diffusion import ImageDiffusionModel
        self.diffusion_model = ImageDiffusionModel(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_heads=num_heads,
            num_classes=num_classes,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order
        )
        
        # 自编码器将在训练时设置
        self.autoencoder = None
    
    def set_autoencoder(self, autoencoder):
        """设置自编码器"""
        self.autoencoder = autoencoder
    
    def forward(self, x, labels, t):
        """前向传播"""
        return self.diffusion_model(x, labels, t)
    
    def get_loss(self, x, labels, t, noise=None):
        """计算损失"""
        # 首先通过自编码器获取潜在表示
        with torch.no_grad():
            _, z = self.autoencoder(x)
        
        # 在潜在空间中进行扩散
        if noise is None:
            noise = torch.randn_like(z)
        
        z_noisy = self.diffusion_model.q_sample(z, t, noise)
        predicted = self.diffusion_model(z_noisy, labels, t)
        
        return F.mse_loss(predicted, noise)
    
    @torch.no_grad()
    def sample(self, labels, num_inference_steps=50, guidance_scale=7.5):
        """采样生成图像"""
        device = labels.device
        b = labels.shape[0]
        
        # 在潜在空间中采样
        z = self.diffusion_model.sample(
            labels,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # 通过解码器生成图像
        return self.autoencoder.decode(z) 