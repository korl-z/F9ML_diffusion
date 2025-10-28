# import torch.nn as nn
# # Assuming layers.py is in the same directory
# from layers import ResidualBlock, RefineNetBlock, CondInstanceNorm, AdaptiveConvBlock

# class RefineNet(nn.Module):
#     def __init__(self, in_channels, hidden_channels=(128, 256, 512, 1024), n_noise_scale=10):
#         super().__init__()
#         self.res1 = ResidualBlock(in_channels, hidden_channels[0], n_layers=2, downsample='stride')
#         self.res2 = ResidualBlock(hidden_channels[0], hidden_channels[1], n_layers=2, downsample='dilation', dilation=2)
#         self.res3 = ResidualBlock(hidden_channels[1], hidden_channels[2], n_layers=2, downsample='dilation', dilation=4)
#         self.res4 = ResidualBlock(hidden_channels[2], hidden_channels[3], n_layers=2, downsample='dilation', dilation=8)
        
#         self.refine1 = RefineNetBlock(x1_in=hidden_channels[-1], x2_in=hidden_channels[-1], channels=hidden_channels[-1], n_noise_scale=n_noise_scale)
#         self.refine2 = RefineNetBlock(x1_in=hidden_channels[-2], x2_in=hidden_channels[-1], channels=hidden_channels[-2], n_noise_scale=n_noise_scale)
#         self.refine3 = RefineNetBlock(x1_in=hidden_channels[-3], x2_in=hidden_channels[-2], channels=hidden_channels[-3], n_noise_scale=n_noise_scale)
#         self.refine4 = RefineNetBlock(x1_in=hidden_channels[-4], x2_in=hidden_channels[-3], channels=hidden_channels[-4], n_noise_scale=n_noise_scale)

#         self.up_norm = CondInstanceNorm(hidden_channels[-4], n_noise_scale)
#         self.up_conv = nn.ConvTranspose2d(hidden_channels[-4], hidden_channels[-4], kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.out = AdaptiveConvBlock(hidden_channels[-4], in_channels, n_noise_scale=n_noise_scale)

#     def forward(self, x, noise_scale_idx):
#         h1 = self.res1(x, noise_scale_idx)
#         h2 = self.res2(h1, noise_scale_idx)
#         h3 = self.res3(h2, noise_scale_idx)
#         h4 = self.res4(h3, noise_scale_idx)

#         h = self.refine1(h4, x2=None, noise_scale_idx=noise_scale_idx)
#         h = self.refine2(h3, h, noise_scale_idx)
#         h = self.refine3(h2, h, noise_scale_idx)
#         h = self.refine4(h1, h, noise_scale_idx)

#         h = self.up_norm(h, noise_scale_idx)
#         h = self.up_conv(h)
#         h = self.out(h, noise_scale_idx)
#         return h

import torch.nn as nn
import torch 
import math

from ml.diffusion.score.layers import ResidualBlock, RefineNetBlock, CondInstanceNorm, AdaptiveConvBlock


class RefineNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=(128, 256, 512, 1024), n_noise_scale=10):
        super().__init__()
        
        self.res1 = ResidualBlock(in_channels, hidden_channels[0], n_layers=2, downsample='dilation', dilation=1)
        
        self.res2 = ResidualBlock(hidden_channels[0], hidden_channels[1], n_layers=2, downsample='dilation', dilation=2)
        self.res3 = ResidualBlock(hidden_channels[1], hidden_channels[2], n_layers=2, downsample='dilation', dilation=4)
        self.res4 = ResidualBlock(hidden_channels[2], hidden_channels[3], n_layers=2, downsample='dilation', dilation=8)
        
        self.refine1 = RefineNetBlock(x1_in=hidden_channels[-1], x2_in=hidden_channels[-1], channels=hidden_channels[-1], n_noise_scale=n_noise_scale)
        self.refine2 = RefineNetBlock(x1_in=hidden_channels[-2], x2_in=hidden_channels[-1], channels=hidden_channels[-2], n_noise_scale=n_noise_scale)
        self.refine3 = RefineNetBlock(x1_in=hidden_channels[-3], x2_in=hidden_channels[-2], channels=hidden_channels[-3], n_noise_scale=n_noise_scale)
        self.refine4 = RefineNetBlock(x1_in=hidden_channels[-4], x2_in=hidden_channels[-3], channels=hidden_channels[-4], n_noise_scale=n_noise_scale)

        self.up_norm = CondInstanceNorm(hidden_channels[-4], n_noise_scale)
        
        self.up_conv = nn.Conv2d(hidden_channels[-4], hidden_channels[-4], kernel_size=3, padding='same')
        
        self.out = AdaptiveConvBlock(hidden_channels[-4], in_channels, n_noise_scale=n_noise_scale)


    def forward(self, x, noise_scale_idx):
        h1 = self.res1(x, noise_scale_idx)
        h2 = self.res2(h1, noise_scale_idx)
        h3 = self.res3(h2, noise_scale_idx)
        h4 = self.res4(h3, noise_scale_idx)

        h = self.refine1(h4, x2=None, noise_scale_idx=noise_scale_idx)
        h = self.refine2(h3, h, noise_scale_idx)
        h = self.refine3(h2, h, noise_scale_idx)
        h = self.refine4(h1, h, noise_scale_idx)

        h = self.up_norm(h, noise_scale_idx)
        h = self.up_conv(h) 
        h = self.out(h, noise_scale_idx)

        return h
    
class TimeEmbedding(nn.Module):
    """
    Converts a tensor of noise levels to a sinusoidal time embedding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    """
    a simple convolutional block with Group Normalization and SiLU activation.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.norm = nn.GroupNorm(1, out_channels) # GroupNorm is often more stable for small batches
        self.act = nn.SiLU() # SiLU (or Swish) is a common activation in modern U-Nets
        
        # A linear layer to project the time embedding to the number of output channels
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        
        h = self.conv1(x)
        h = self.norm(h)
        
        time_cond = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_cond
        
        return self.act(h)

class SimpleUNet(nn.Module):
    """
    shape-preserving U-Net for low-dimensional data.
    """
    def __init__(self, in_channels=1, n_noise_scale=10):
        super().__init__()
        
        time_emb_dim = 32  # Dimension for the time embedding
        
        # --- Time Embedding ---
        self.time_embedding = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # --- Encoder Path (Downsampling) ---
        # No actual downsampling occurs; we just increase channels
        self.conv_in = ConvBlock(in_channels, 32, time_emb_dim)
        self.down1 = ConvBlock(32, 64, time_emb_dim)
        
        # --- Bottleneck ---
        self.bottleneck = ConvBlock(64, 128, time_emb_dim)
        
        # --- Decoder Path (Upsampling) ---
        # No actual upsampling occurs; we decrease channels
        self.up1 = ConvBlock(128 + 64, 64, time_emb_dim) # We add skip connection channels
        self.up2 = ConvBlock(64 + 32, 32, time_emb_dim)
        
        # --- Output Layer ---
        # Final convolution to map back to the original number of channels
        self.conv_out = nn.Conv2d(32, in_channels, kernel_size=1)

    def forward(self, x, noise_scale_idx):
        # x: (batch, in_channels, h, w)
        # noise_scale_idx: (batch,) tensor of integers representing the noise level
        
        # 1. Get time embedding
        t_emb = self.time_embedding(noise_scale_idx)
        
        # 2. Encoder
        x1 = self.conv_in(x, t_emb)
        x2 = self.down1(x1, t_emb)
        
        # 3. Bottleneck
        bn = self.bottleneck(x2, t_emb)
        
        # 4. Decoder with Skip Connections
        # Concatenate bottleneck output with the skip connection from the encoder
        up1_in = torch.cat([bn, x2], dim=1) 
        u1 = self.up1(up1_in, t_emb)
        
        up2_in = torch.cat([u1, x1], dim=1)
        u2 = self.up2(up2_in, t_emb)
        
        # 5. Final Output
        output = self.conv_out(u2)
        
        return output


#heavier, modular Unet
class ResnetBlock(nn.Module):
    """A standard residual block with two convolutional layers."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU()
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        time_cond = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_cond
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip_connection(x)

class AttentionBlock(nn.Module):
    """A self-attention block."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        attn = torch.bmm(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj_out(out)

# --- The New, Working, Modular U-Net ---

class ModularUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32,
                 channel_mults=(1, 2, 4), num_res_blocks=2, use_attention_at_level=(False, False, True)):
        super().__init__()

        time_emb_dim = base_channels * 4

        # Time Embedding
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder Path
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding='same')
        self.downs = nn.ModuleList()
        
        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResnetBlock(current_channels, out_channels, time_emb_dim))
                current_channels = out_channels
            if use_attention_at_level[i]:
                level_blocks.append(AttentionBlock(current_channels))
            self.downs.append(level_blocks)

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResnetBlock(current_channels, current_channels, time_emb_dim),
            AttentionBlock(current_channels),
            ResnetBlock(current_channels, current_channels, time_emb_dim)
        ])

        # Decoder Path
        self.ups = nn.ModuleList()
        # Iterate through encoder levels in reverse to build decoder
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList()
            
            # The channel dimension of the skip connection from the corresponding encoder level
            skip_channels = base_channels * channel_mults[i]
            
            # The input to this level is the output from the deeper level + the skip connection
            in_ch = current_channels + skip_channels
            
            for _ in range(num_res_blocks):
                level_blocks.append(ResnetBlock(in_ch, out_channels, time_emb_dim))
                in_ch = out_channels # Subsequent blocks in the same level have the same input/output channels
                
            if use_attention_at_level[i]:
                level_blocks.append(AttentionBlock(out_channels))

            self.ups.append(level_blocks)
            current_channels = out_channels

        # Output Layer
        self.conv_out = nn.Sequential(
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=1)
        )

    def forward(self, x, noise_scale_idx):
        t_emb = self.time_embedding(noise_scale_idx)
        
        h = self.conv_in(x)
        skip_connections = [h]

        # Encoder
        for level_blocks in self.downs:
            for block in level_blocks:
                h = block(h, t_emb) if isinstance(block, ResnetBlock) else block(h)
            skip_connections.append(h)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb) if isinstance(block, ResnetBlock) else block(h)

        # Decoder
        for level_blocks in self.ups:
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            for block in level_blocks:
                h = block(h, t_emb) if isinstance(block, ResnetBlock) else block(h)
        
        return self.conv_out(h)