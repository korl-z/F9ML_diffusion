import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ml.common.nn.positional_emb import TimeEmbedding

#Tiny unet implementation (no downsampling on 2x3x3 imgs)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
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

# --- The New Shape-Preserving U-Net for Tiny Data ---

class TinyUNet(nn.Module):
    """
    A shape-preserving U-Net for low-dimensional data (e.g., 2-channel, 3x3 'images').
    It does not use spatial downsampling but follows the U-Net pattern of increasing
    and decreasing channel depth.
    """
    def __init__(self, in_channels=2, base_channels=32, time_emb_dim=32,
                 channel_mults=[1, 2, 4], use_attention_at_level=[False, True, True]):
        super().__init__()
        self.in_channels = in_channels

        # Time Embedding
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder Path
        self.conv_in = ResnetBlock(in_channels, base_channels, time_emb_dim)
        self.downs = nn.ModuleList()
        
        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList([
                ResnetBlock(current_channels, out_channels, time_emb_dim)
            ])
            if use_attention_at_level[i]:
                level_blocks.append(AttentionBlock(out_channels))
            self.downs.append(level_blocks)
            current_channels = out_channels

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResnetBlock(current_channels, current_channels, time_emb_dim),
            AttentionBlock(current_channels),
            ResnetBlock(current_channels, current_channels, time_emb_dim)
        ])

        # Decoder Path
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            skip_channels = base_channels * channel_mults[i]
            in_ch = current_channels + skip_channels
            
            level_blocks = nn.ModuleList()
            if use_attention_at_level[i]:
                level_blocks.append(AttentionBlock(in_ch)) # Attention on concatenated features
            level_blocks.append(ResnetBlock(in_ch, out_channels, time_emb_dim))

            self.ups.append(level_blocks)
            current_channels = out_channels

        # Output Layer
        self.conv_out = nn.Sequential(
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=1)
        )

    def forward(self, x, sigma):
        t_emb = self.time_embedding(sigma)
        
        h = self.conv_in(x, t_emb)
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

#DDPM implementation, for testing, 1d array implementation
class NoisePredictorUNet(nn.Module):
    def __init__(
        self,
        data_dim: int,
        base_dim: int = 128,
        depth: int = 3,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.data_dim = int(data_dim)
        self.base_dim = int(base_dim)
        self.depth = int(depth)
        self.time_emb_dim = int(time_emb_dim)
            
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # time MLP projector
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Build hidden dims: base, base*2, base*4, ...
        hidden_dims = [base_dim * (2**i) for i in range(depth)]
        self.hidden_dims = hidden_dims  # list of ints

        # input projection: D -> hidden_dims[0]
        self.input_fc = nn.Linear(self.data_dim, hidden_dims[0])
        self.input_ln = nn.LayerNorm(hidden_dims[0])

        # encoder: list of (Linear, LayerNorm)
        self.enc_layers = nn.ModuleList()
        self.enc_ln = nn.ModuleList()
        for i in range(depth - 1):
            self.enc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.enc_ln.append(nn.LayerNorm(hidden_dims[i + 1]))

        # bottleneck (two small FCs)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.SiLU(),
        )
        self.bottleneck_ln = nn.LayerNorm(hidden_dims[-1])

        self.dec_layers = nn.ModuleList()
        self.dec_ln = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            in_dim = hidden_dims[i] + hidden_dims[i]
            out_dim = hidden_dims[i - 1]
            self.dec_layers.append(nn.Linear(in_dim, out_dim))
            self.dec_ln.append(nn.LayerNorm(out_dim))

        self.out_proj = nn.Linear(hidden_dims[0], self.data_dim)

        self.time_proj_input = nn.Linear(time_emb_dim, hidden_dims[0])
        self.time_proj_enc = nn.ModuleList(
            [nn.Linear(time_emb_dim, d) for d in hidden_dims[1:]]
        )
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, hidden_dims[-1])

        self.time_proj_dec = nn.ModuleList(
            [nn.Linear(time_emb_dim, d) for d in hidden_dims[:-1][::-1]]
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t):
        """
        x: [B, D]
        t: [B] ints 0..T-1
        returns: eps_pred [B, D]
        """
        assert x.ndim == 2 and x.shape[1] == self.data_dim

        temb = self.time_embedding(t.float() + 1)
        temb = self.time_mlp(temb)

        h = self.input_fc(x) + self.time_proj_input(temb)
        h = self.input_ln(h)
        h = F.silu(h)

        skips = []
        for i, enc in enumerate(self.enc_layers):
            h = enc(h)
            h = h + self.time_proj_enc[i](temb)
            h = self.enc_ln[i](h)
            h = F.silu(h)
            skips.append(h)

        # bottleneck
        h = self.bottleneck(h)
        h = h + self.time_proj_bottleneck(temb)
        h = self.bottleneck_ln(h)
        h = F.silu(h)

        # decoder path (mirror)
        for j, dec in enumerate(self.dec_layers):
            skip = skips[-1 - j]
            h = torch.cat([h, skip], dim=-1)
            h = dec(h)
            h = h + self.time_proj_dec[j](temb)
            h = self.dec_ln[j](h)
            h = F.silu(h)

        # final projection to data_dim
        eps_pred = self.out_proj(h)
        return eps_pred
    

class VarPredictorUNet(nn.Module):
    """
    A U-Net that predicts both the noise (eps) and the variance interpolation (v).
    """

    def __init__(
        self,
        data_dim: int,
        base_dim: int = 128,
        depth: int = 3,
        time_emb_dim: int = 32,
    ):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        self.data_dim = int(data_dim)
        self.base_dim = int(base_dim)
            
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # time MLP projector
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Build hidden dims: base, base*2, base*4, ...
        hidden_dims = [base_dim * (2**i) for i in range(depth)]
        self.hidden_dims = hidden_dims  # list of ints

        # input projection: D -> hidden_dims[0]
        self.input_fc = nn.Linear(self.data_dim, hidden_dims[0])
        self.input_ln = nn.LayerNorm(hidden_dims[0])

        # encoder: list of (Linear, LayerNorm)
        self.enc_layers = nn.ModuleList()
        self.enc_ln = nn.ModuleList()
        for i in range(depth - 1):
            self.enc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.enc_ln.append(nn.LayerNorm(hidden_dims[i + 1]))

        # bottleneck (two small FCs)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.SiLU(),
        )
        self.bottleneck_ln = nn.LayerNorm(hidden_dims[-1])

        self.dec_layers = nn.ModuleList()
        self.dec_ln = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            # skip has size hidden_dims[i], and h currently has size hidden_dims[i],
            # concat size is hidden_dims[i] + hidden_dims[i]
            in_dim = hidden_dims[i] + hidden_dims[i]
            out_dim = hidden_dims[i - 1]
            self.dec_layers.append(nn.Linear(in_dim, out_dim))
            self.dec_ln.append(nn.LayerNorm(out_dim))

        self.out_proj = nn.Linear(hidden_dims[0], self.data_dim * 2)

        # time-to-feature projection layers 
        # - for input projection
        self.time_proj_input = nn.Linear(time_emb_dim, hidden_dims[0])
        # - for each encoder output (i -> hidden_dims[i+1])
        self.time_proj_enc = nn.ModuleList(
            [nn.Linear(time_emb_dim, d) for d in hidden_dims[1:]]
        )
        # - for bottleneck
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, hidden_dims[-1])
        # - for each decoder output (matching dec out dims)
        self.time_proj_dec = nn.ModuleList(
            [nn.Linear(time_emb_dim, d) for d in hidden_dims[:-1][::-1]]
        )

        # initialization (similar style to MLP)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t):
        """
        x: [B, D]
        t: [B] ints 0..T-1
        returns: A single tensor [B, 2*D]. The first D channels are eps_pred,
                 and the second D channels are v_pred.
        """
        assert x.ndim == 2 and x.shape[1] == self.data_dim
        
        temb = self.time_embedding(t.float() + 1)
        temb = self.time_mlp(temb)

        # input proj + add time
        h = self.input_fc(x) + self.time_proj_input(temb)
        h = self.input_ln(h)
        h = F.silu(h)

        # encoder path, collect skips
        skips = []
        for i, enc in enumerate(self.enc_layers):
            h = enc(h)
            h = h + self.time_proj_enc[i](temb)
            h = self.enc_ln[i](h)
            h = F.silu(h)
            skips.append(h)

        # bottleneck
        h = self.bottleneck(h)
        h = h + self.time_proj_bottleneck(temb)
        h = self.bottleneck_ln(h)
        h = F.silu(h)

        # decoder path (mirror)
        for j, dec in enumerate(self.dec_layers):
            skip = skips[-1 - j]
            h = torch.cat([h, skip], dim=-1)
            h = dec(h)
            h = h + self.time_proj_dec[j](temb)
            h = self.dec_ln[j](h)
            h = F.silu(h)

        model_output = self.out_proj(h)
        return model_output    
    


# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""
import numpy as np
from ml.diffusion.EDM.torch_utils import const_like
from ml.common.nn.positional_emb import MPFourier

# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


# ----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.


def resample(x, f=[1, 1], mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(
            x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
        )
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(
        x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
    )


# ----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

class MP_SiLU(nn.Module):
    def forward(self, x):
        return mp_silu(x)

# ----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


# ----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


# ----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding="same")

class MPResnetBlock(nn.Module):
    """
    A ResNet block refactored to use magnitude-preserving operations.
    It preserves the shape of the input tensor.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
                MP_SiLU(),
                nn.Linear(time_emb_dim, out_channels))
        self.conv1 = MPConv(in_channels, out_channels, kernel=(3,3))
        self.conv2 = MPConv(out_channels, out_channels, kernel=(3,3))
        self.act = MP_SiLU()
        self.skip_connection = MPConv(in_channels, out_channels, kernel=(1,1)) if in_channels != out_channels else None

    def forward(self, x, t_emb):
        h = mp_silu(self.conv1(x))
        time_cond = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = mp_sum(h, time_cond, t=0.5)
        h = mp_silu(self.conv2(h))
        if self.skip_connection is not None:
            skip = self.skip_connection(x)
        else:
            skip = x
        return mp_sum(h, skip, t=0.5)
    
class MPAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.qkv = MPConv(channels, channels * 3, kernel=(1,1))
        self.proj_out = MPConv(channels, channels, kernel=(1,1))
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        attn = torch.bmm(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj_out(out)

# --- The New Shape-Preserving U-Net for Tiny Data --
class MPTinyUNet(nn.Module):
    """
    A shape-preserving U-Net for low-dimensional data (e.g., 2-channel, 3x3 'images').
    It does not use spatial downsampling but follows the U-Net pattern of increasing
    and decreasing channel depth.
    """
    def __init__(self, in_channels=2, base_channels=32, time_emb_dim=32,
                 channel_mults=[1, 2, 4], use_attention_at_level=[False, True, True]):
        super().__init__()
        self.in_channels = in_channels

        # Time Embedding
        self.time_embedding = nn.Sequential(
            MPFourier(base_channels),  #MPFourier <--> TimeEmbedding ?
            nn.Linear(base_channels, time_emb_dim),
            MP_SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder Path
        self.conv_in = MPResnetBlock(in_channels, base_channels, time_emb_dim)
        self.downs = nn.ModuleList()
        
        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList([
                MPResnetBlock(current_channels, out_channels, time_emb_dim)
            ])
            if use_attention_at_level[i]:
                level_blocks.append(MPAttentionBlock(out_channels))
            self.downs.append(level_blocks)
            current_channels = out_channels

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            MPResnetBlock(current_channels, current_channels, time_emb_dim),
            MPAttentionBlock(current_channels),
            MPResnetBlock(current_channels, current_channels, time_emb_dim)
        ])

        # Decoder Path
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            skip_channels = base_channels * channel_mults[i]
            in_ch = current_channels + skip_channels
            
            level_blocks = nn.ModuleList()
            if use_attention_at_level[i]:
                level_blocks.append(MPAttentionBlock(in_ch)) # Attention on concatenated features
            level_blocks.append(MPResnetBlock(in_ch, out_channels, time_emb_dim))

            self.ups.append(level_blocks)
            current_channels = out_channels

        # Output Layer
        self.conv_out = nn.Sequential(
            MP_SiLU(),
            MPConv(base_channels, in_channels, kernel=(1,1))
        )

    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        t_emb = self.time_embedding(sigma)
        
        h = self.conv_in(x, t_emb)
        skip_connections = [h]

        # Encoder
        for level_blocks in self.downs:
            for block in level_blocks:
                h = block(h, t_emb) if isinstance(block, MPResnetBlock) else block(h)
            skip_connections.append(h)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb) if isinstance(block, MPResnetBlock) else block(h)

        # Decoder
        for level_blocks in self.ups:
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            for block in level_blocks:
                h = block(h, t_emb) if isinstance(block, MPResnetBlock) else block(h)
        
        return self.conv_out(h)