import torch
from torch import nn
from torch.nn import functional as F
from sd.attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channel)
        self.attention = SelfAttention(1, channel)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, channel, Width, Height)
        residue = x
        # (Batch_size, channel, Width, Height) -> (Batch_size, channel, Width, Height)
        x = self.group_norm(x)
        n, c, h, w = x.shape
        # (Batch_size, channel, Width, Height) -> (Batch_size, channel, Width * Height)
        x = x.view((n, c, h * w))
        # (Batch_size, channel, Width * Height) -> (Batch_size, Width * Height, channel)
        x = x.transpose(-1, -2)
        # (Batch_size, Width * Height, channel) -> (Batch_size, Width * Height, channel)
        x = self.attention(x)
        # (Batch_size, Width * Height, channel) -> (Batch_size, channel, Width * Height)
        x = x.transpose(-1, -2)
        # (Batch_size, channel, Width * Height) -> (Batch_size, channel, Width, Height)
        x = x.view((n, c, h, w))
        # (Batch_size, channel, Width, Height) -> (Batch_size, channel, Width, Height)
        x = x + residue
        return x
        
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1, padding = 0)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, in_channels, Width, Height)
        residue = x
        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        x = x + self.residual_layer(residue)
        return x

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_size, 4, width/8, height/8) -> (Batch_size, 4, width/8, height/8)
            nn.Conv2d(4, 4, kernel_size = 1, padding = 0),
            # (Batch_size, 4, width/8, height/8) -> (Batch_size, 512, width/8, height/8)
            nn.Conv2d(4, 512, kernel_size = 3, padding = 1),
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/8, height/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/8, height/8)
            VAE_AttentionBlock(512),
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/8, height/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/8, height/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/8, height/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/8, height/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/4, height/4)
            nn.Upsample(scale_factor = 2),
            # (Batch_size, 512, width/8, height/8) -> (Batch_size, 512, width/8, height/8)
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            # (Batch_size, 512, width/4, height/4) -> (Batch_size, 512, width/4, height/4)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, width/4, height/4) -> (Batch_size, 512, width/4, height/4)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, width/4, height/4) -> (Batch_size, 512, width/4, height/4)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, width/4, height/4) -> (Batch_size, 512, width/2, height/2)
            nn.Upsample(scale_factor = 2),
            #(Batch_size, 512, width/2, height/2) -> (Batch_size, 512, width/2, height/2)
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            # (Batch_size, 512, width/2, height/2) -> (Batch_size, 256, width/2, height/2)
            VAE_ResidualBlock(512, 256),
            # (Batch_size, 256, width/2, height/2) -> (Batch_size, 256, width/2, height/2)
            VAE_ResidualBlock(256, 256),
            # (Batch_size, 256, width/2, height/2) -> (Batch_size, 256, width/2, height/2)
            VAE_ResidualBlock(256, 256),
            # (Batch_size, 256, width/2, height/2) -> (Batch_size, 256, width, height)
            nn.Upsample(scale_factor = 2),
            # (Batch_size, 256, width, height) -> (Batch_size, 256, width, height)
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            # (Batch_size, 256, width, height) -> (Batch_size, 128, width, height)
            VAE_ResidualBlock(256, 128),
            # (Batch_size, 128, width, height) -> (Batch_size, 128, width, height)
            VAE_ResidualBlock(128, 128),
            # (Batch_size, 128, width, height) -> (Batch_size, 128, width, height)
            VAE_ResidualBlock(128, 128),
            # (Batch_size, 128, width, height) -> (Batch_size, 128, width, height)
            nn.GroupNorm(32, 128),
            # (Batch_size, 128, width, height) -> (Batch_size, 128, width, height)
            nn.SiLU(),
            # (Batch_size, 128, width, height) -> (Batch_size, 3, width, height)
            nn.Conv2d(128, 3, kernel_size = 3, padding = 1),
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, 4, width/8, height/8)
        x /= 0.18215
        for module in self:
            x = module(x)
        # (Batch_size, 3, width, height)
        return x