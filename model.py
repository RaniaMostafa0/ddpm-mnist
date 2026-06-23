import torch
import torch.nn as nn
import math

class NoiseSchedule(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.T = T
        
        betas = torch.linspace(0.0001, 0.02, T)
        
        alphas = 1 - betas
        
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        # sinusoidal embedding
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        embedding = torch.relu(self.linear1(embedding))
        embedding = self.linear2(embedding)
        return embedding 
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels)

    def forward(self, x, t_emb):
        x = torch.relu(self.norm1(self.conv1(x)))
        t = self.time_mlp(t_emb)[:, :, None, None] 
        x = x + t                                     # add time embedding
        x = torch.relu(self.norm2(self.conv2(x)))
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, time_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.conv_block(x, t_emb)    # process features
        skip = x                          # save for skip connection
        x = self.downsample(x)            # halve spatial size
        return x, skip                    
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, time_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)                        # double spatial size
        x = torch.cat([x, skip], dim=1)             # concatenate skip connection
        x = self.conv_block(x, t_emb)               # process combined features
        return x
    
class UNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        
        # initial conv
        self.init_conv = nn.Conv2d(1, 32, 3, padding=1)
        
        # downsampling
        self.down1 = DownBlock(32, 64, time_dim)
        self.down2 = DownBlock(64, 128, time_dim)
        
        # bottleneck
        self.bottleneck = ConvBlock(128, 128, time_dim)
        
        # upsampling
        self.up1 = UpBlock(128, 64, 128, time_dim)  # skip2 has 128 channels
        self.up2 = UpBlock(64, 32, 64, time_dim)    # skip1 has 64 channels
        
        # final conv
        self.final_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        
        x = self.init_conv(x)
        
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        
        x = self.bottleneck(x, t_emb)
        
        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        
        return self.final_conv(x)
    
@torch.no_grad()
def sample(unet, noise_schedule, n_images=8, device='cuda'):
    unet.eval()
    x = torch.randn(n_images, 1, 28, 28).to(device)  # start from pure noise
    
    for t in reversed(range(noise_schedule.T)):
        t_batch = torch.full((n_images,), t, device=device, dtype=torch.long)
        predicted_noise = unet(x, t_batch)
        
        alpha = noise_schedule.alphas[t]
        alpha_bar = noise_schedule.alpha_bars[t]
        beta = noise_schedule.betas[t]
        
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)
        
        x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise) + torch.sqrt(beta) * z
    
    return x