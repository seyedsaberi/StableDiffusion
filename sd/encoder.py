import torch
from torch import nn
from torch.nn import functional as F
from sd.decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, Width, Height) -> (Batch_size, 128, Width, Height)
            nn.Conv2d(3, 128, kernel_size = 3, padding = 1),
            # (Batch_size, 128, Width, Height) -> (Batch_size, 128, Width, Height)
            VAE_ResidualBlock(128, 128),
            # (Batch_size, 128, Width, Height) -> (Batch_size, 128, Width, Height)
            VAE_ResidualBlock(128, 128),
            # (Batch_size, 128, Width, Height) -> (Batch_size, 128, Width/2, Height/2)
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0),
            # (Batch_size, 128, Width/2, Height/2) -> (Batch_size, 256, Width/2, Height/2)
            VAE_ResidualBlock(128, 256),
            # (Batch_size, 256, Width/2, Height/2) -> (Batch_size, 256, Width/2, Height/2)
            VAE_ResidualBlock(256, 256),
            # (Batch_size, 256, Width/2, Height/2) -> (Batch_size, 256, Width/4, Height/4)
            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 0),
            # (Batch_size, 256, Width/4, Height/4) -> (Batch_size, 256, Width/4, Height/4)
            VAE_ResidualBlock(256, 512),
            # (Batch_size, 512, Width/4, Height/4) -> (Batch_size, 512, Width/4, Height/4)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Width/4, Height/4) -> (Batch_size, 512, Width/8, Height/8)
            nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 0),
            # (Batch_size, 512, Width/8, Height/8) -> (Batch_size, 512, Width/8, Height/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Width/8, Height/8) -> (Batch_size, 512, Width/8, Height/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Width/8, Height/8) -> (Batch_size, 512, Width/8, Height/8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Width/8, Height/8) -> (Batch_size, 512, Width/8, Height/8)
            VAE_AttentionBlock(512),
            # (Batch_size, 512, Width/8, Height/8) -> (Batch_size, 512, Width/8, Height/8)
            nn.GroupNorm(32, 512),
            # (Batch_size, 512, Width/8, Height/8) -> (Batch_size, 512, Width/8, Height/8)
            nn.SiLU(),
            # (Batch_size, 512, Width/8, Height/8) -> (Batch_size, 8, Width/8, Height/8)
            nn.Conv2d(512, 8, kernel_size = 3, padding = 1),
            # (Batch_size, 8, Width/8, Height/8) -> (Batch_size, 8, Width/8, Height/8)
            nn.Conv2d(8, 8 , kernel_size = 1, padding = 0),
        )
    def forward(self, x : torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel, Width, Height)
        # noise: (Batch_size, out_channels, Width, Height)
        for name, module in self.named_children():
            # if getattr(module, 'stride', None) == (2, 2):
                # # (padding_left, padding_right, padding_top, padding_bottom)
                # F.pad(x, (0, 1, 0, 1))
            x = module(x)
            print(name, x.shape)
        # (Batch_size, 8, Width/8, Height/8) -> (Batch_size, 4, Width/8, Height/8), (Batch_size, 4, Width/8, Height/8)
        mean, log_variance = torch.chunk(x, 2, dim = 1)
        # (Batch_size, 4, Width/8, Height/8) -> (Batch_size, 4, Width/8, Height/8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_size, 4, Width/8, Height/8) -> (Batch_size, 4, Width/8, Height/8)
        variance = log_variance.exp()
        # (Batch_size, 4, Width/8, Height/8) -> (Batch_size, 4, Width/8, Height/8)
        stdev = variance.sqrt()
        # (Batch_size, 4, Width/8, Height/8) -> (Batch_size, 4, Width/8, Height/8)
        print(mean.shape, stdev.shape, noise.shape)
        x = mean + stdev * noise
        x *= 0.18215
        return x, mean, stdev