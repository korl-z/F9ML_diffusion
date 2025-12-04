import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import silu
from ml.common.nn.positional_emb import TimeEmbedding, TimeEmbedding2


# Tiny unet implementation (no downsampling on 2x3x3 imgs)
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same"
        )
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU()
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

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
        self.scale = channels**-0.5

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


# shape preserving U-Net (channel increase - decrease only), not optimal


class TinyUNet(nn.Module):
    """
    A shape preserving U-Net for low-dimensional data (2-channel, 3x3 pix. for 18 features).
    """

    def __init__(
        self,
        in_channels=2,
        base_channels=32,
        time_emb_dim=32,
        channel_mults=[1, 2, 4],
        use_attention_at_level=[False, True, True],
    ):
        super().__init__()
        self.in_channels = in_channels

        # Time emb
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder path
        self.conv_in = ResnetBlock(in_channels, base_channels, time_emb_dim)
        self.downs = nn.ModuleList()

        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList(
                [ResnetBlock(current_channels, out_channels, time_emb_dim)]
            )
            if use_attention_at_level[i]:
                level_blocks.append(AttentionBlock(out_channels))
            self.downs.append(level_blocks)
            current_channels = out_channels

        # Bottleneck
        self.bottleneck = nn.ModuleList(
            [
                ResnetBlock(current_channels, current_channels, time_emb_dim),
                AttentionBlock(current_channels),
                ResnetBlock(current_channels, current_channels, time_emb_dim),
            ]
        )

        # Decoder path
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            skip_channels = base_channels * channel_mults[i]
            in_ch = current_channels + skip_channels

            level_blocks = nn.ModuleList()
            if use_attention_at_level[i]:
                level_blocks.append(
                    AttentionBlock(in_ch)
                )  # Attention on concatenated features
            level_blocks.append(ResnetBlock(in_ch, out_channels, time_emb_dim))

            self.ups.append(level_blocks)
            current_channels = out_channels

        # Output layer
        self.conv_out = nn.Sequential(
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
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


# DDPM modules


# for initial tests, usless otherwise
class Diffusion1DModel(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.var_net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Softplus()
        )

    def forward(self, x, t):
        """
        x : shape [batch, 1] — the noisy sample x_t
        t : shape [batch, 1] — the timestep (scaled to [0,1] ideally)
        returns:
            mu: predicted mean [batch, 1]
            sigma: predicted std dev [batch, 1]
        """
        xin = torch.cat((x, t), dim=1)
        mu = self.mean_net(xin)
        sigma = self.var_net(xin)
        return mu, sigma


# Not in use anymore
def sinusoidal_time_embedding(t: torch.Tensor, dim: int):
    """
    t: 1D tensor of shape [B] (floats in [0,1] or ints).
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / half)
    args = t[:, None].float() * freqs[None, :] * 2 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb  # [B, dim]


# doesnt perform well
class NoisePredictorMLP(nn.Module):
    """
    Predicts noise eps for x_t: eps_pred = model(x_t, t)
    - data_dim: D
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        time_emb_dim: int = 32,
        timesteps: int = 1000,
    ):
        super().__init__()
        assert n_hidden_layers >= 1
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim
        self.timesteps = int(timesteps)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        input_dim = data_dim + time_emb_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        hidden_layers = []
        for _ in range(n_hidden_layers - 1):
            hidden_layers.append(nn.LayerNorm(hidden_dim))
            hidden_layers.append(nn.SiLU())
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_net = nn.Sequential(*hidden_layers)

        self.out = nn.Linear(hidden_dim, data_dim)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prepare_t(self, t):
        t = torch.as_tensor(t)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t.view(-1)
        if t.dtype in (torch.long, torch.int) or (t.max() > 1.0):
            t = t.float() / float(self.timesteps - 1)  # map 0..T-1 to [0,1]
        else:
            t = t.float()
        return t.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor, t):
        """
        x: [B, D]
        t: [B] ints 0..T-1 or floats in [0,1]
        returns: eps_pred [B, D]
        """
        assert x.ndim == 2 and x.shape[1] == self.data_dim
        t_f = self._prepare_t(t)
        temb = sinusoidal_time_embedding(t_f, self.time_emb_dim)
        temb = self.time_mlp(temb)
        inp = torch.cat([x, temb], dim=-1)
        h = F.silu(self.input_fc(inp))
        if len(self.hidden_net) > 0:
            h = self.hidden_net(h)
        eps_pred = self.out(h)
        return eps_pred


class UNet1D(nn.Module):
    """
    1D unet, expands features vector using MLP to base dim, the rest is unet but with mlps not 1d conv.
    """

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
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # hidden dims: base, base*2, base*4, ...
        hidden_dims = [base_dim * (2**i) for i in range(depth)]
        self.hidden_dims = hidden_dims

        #input projection: D -> hidden_dims[0]
        self.input_fc = nn.Linear(self.data_dim, hidden_dims[0])
        self.input_ln = nn.LayerNorm(hidden_dims[0])
        self.enc_layers = nn.ModuleList()
        self.enc_ln = nn.ModuleList()

        for i in range(depth - 1):
            self.enc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.enc_ln.append(nn.LayerNorm(hidden_dims[i + 1]))

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

    def forward(self, x: torch.Tensor, sigma, class_labels=None, augment_labels=None):
        """
        forward pass for EDM framework compatibility.
        """
        original_shape = x.shape
        batch_size = original_shape[0]
        x_1d = x.view(batch_size, -1)

        temb = self.time_embedding(sigma)
        temb = self.time_mlp(temb)
        h = self.input_fc(x_1d) + self.time_proj_input(temb)
        h = self.input_ln(h)
        h = F.silu(h)
        skips = []
        for i, enc in enumerate(self.enc_layers):
            h = enc(h)
            h = h + self.time_proj_enc[i](temb)
            h = self.enc_ln[i](h)
            h = F.silu(h)
            skips.append(h)
        h = self.bottleneck(h)
        h = h + self.time_proj_bottleneck(temb)
        h = self.bottleneck_ln(h)
        h = F.silu(h)
        for j, dec in enumerate(self.dec_layers):
            skip = skips[-1 - j]
            h = torch.cat([h, skip], dim=-1)
            h = dec(h)
            h = h + self.time_proj_dec[j](temb)
            h = self.dec_ln[j](h)
            h = F.silu(h)
        output_1d = self.out_proj(h)

        output_4d = output_1d.view(original_shape)
        return output_4d


def double_conv(in_ch, out_ch):
    """2x (conv3 + ReLU)"""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


def double_conv_norm(in_channels, out_channels, num_groups=32):
    """
    A block of two 1D convolutions with Group Normalization and SiLU activation.
    """
    if out_channels % num_groups != 0:
        for i in range(num_groups, 0, -1):
             if out_channels % i == 0:
                 num_groups = i
                 break
        if out_channels % num_groups != 0: 
            num_groups = 1

    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
        nn.SiLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
        nn.SiLU(inplace=True),
    )

class UNet1DconvDDPM(nn.Module):
    """
    Final, robust 1D Convolutional U-Net.
    - Fixes the decoder channel mismatch error permanently.
    - Uses a stable log(t+1) time embedding.
    """
    def __init__(
        self,
        data_dim: int,
        base_dim: int = 64,
        depth: int = 4,
        time_emb_dim: int = 32
    ):
        super().__init__()
        self.depth = depth
        self.data_dim = int(data_dim)
        self.time_emb_dim = int(time_emb_dim)

        self.time_embedding = TimeEmbedding2(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
        )

        self.chs = [base_dim * (2**i) for i in range(depth)]
        
        def robust_group_norm(num_channels, num_groups=32):
            if num_channels < num_groups: num_groups = 1
            while num_channels % num_groups != 0: num_groups //= 2
            return nn.GroupNorm(num_groups, num_channels)

        # --- Input Block ---
        self.input_conv = nn.Conv1d(1, self.chs[0], kernel_size=3, padding=1)
        self.input_ln = robust_group_norm(self.chs[0])
        self.time_proj_input = nn.Linear(time_emb_dim, self.chs[0])

        # --- Encoder Blocks ---
        self.enc_layers, self.enc_lns, self.pools, self.time_proj_enc = (nn.ModuleList() for _ in range(4))
        for i in range(depth - 1):
            in_ch, out_ch = self.chs[i], self.chs[i+1]
            self.enc_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            self.enc_lns.append(robust_group_norm(out_ch))
            self.pools.append(nn.MaxPool1d(2))
            self.time_proj_enc.append(nn.Linear(time_emb_dim, out_ch))

        # --- Bottleneck Block ---
        b_dim = self.chs[-1]
        self.bottleneck_conv1 = nn.Conv1d(b_dim, b_dim, 3, padding=1)
        self.bottleneck_ln1 = robust_group_norm(b_dim)
        self.bottleneck_conv2 = nn.Conv1d(b_dim, b_dim, 3, padding=1)
        self.bottleneck_ln2 = robust_group_norm(b_dim)
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, b_dim)

        # --- Decoder Blocks  ---
        self.dec_layers, self.dec_lns, self.upconvs, self.time_proj_dec = (nn.ModuleList() for _ in range(4))
        for i in range(depth - 1, 0, -1):
            upconv_in_ch, upconv_out_ch = self.chs[i], self.chs[i-1]
            self.upconvs.append(nn.ConvTranspose1d(upconv_in_ch, upconv_out_ch, kernel_size=2, stride=2))
            
            # Input channels = upsampled channels (chs[i-1]) + skip channels (chs[i])
            dec_in_ch = self.chs[i-1] + self.chs[i]
            dec_out_ch = self.chs[i-1]
            
            self.dec_layers.append(nn.Conv1d(dec_in_ch, dec_out_ch, kernel_size=3, padding=1))
            self.dec_lns.append(robust_group_norm(dec_out_ch))
            self.time_proj_dec.append(nn.Linear(time_emb_dim, dec_out_ch))

        # --- Output Block ---
        self.out_conv = nn.Conv1d(self.chs[0], 1, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None: nn.init.zeros_(m.bias)

    def _right_pad_to_multiple(self, L: int, multiple: int):
        return (multiple - (L % multiple)) % multiple

    def forward(self, x: torch.Tensor, sigma, class_labels=None, augment_labels=None):
        original_shape = x.shape
        B = original_shape[0]

        x_flat = x.view(B, -1)
        L = x_flat.shape[1]
        min_div = 2 ** (self.depth - 1)
        pad_len = self._right_pad_to_multiple(L, min_div)
        if pad_len > 0: x_flat = F.pad(x_flat, (0, pad_len))
        
        h = x_flat.unsqueeze(1)
        temb = self.time_embedding(sigma)
        temb = self.time_mlp(temb)

        h = self.input_conv(h)
        h = h + self.time_proj_input(temb).unsqueeze(-1)
        h = self.input_ln(h)
        h = F.silu(h)
        
        skips = [h]

        for enc, norm, pool, t_proj in zip(self.enc_layers, self.enc_lns, self.pools, self.time_proj_enc):
            h = enc(h); h = h + t_proj(temb).unsqueeze(-1); h = norm(h); h = F.silu(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck_conv1(h); h = self.bottleneck_ln1(h); h = F.silu(h)
        h = self.bottleneck_conv2(h); h = h + self.time_proj_bottleneck(temb).unsqueeze(-1); h = self.bottleneck_ln2(h); h = F.silu(h)

        for upconv, dec, norm, t_proj in zip(self.upconvs, self.dec_layers, self.dec_lns, self.time_proj_dec):
            skip = skips.pop()
            h = upconv(h)
            if h.shape[-1] != skip.shape[-1]: h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = dec(h); h = h + t_proj(temb).unsqueeze(-1); h = norm(h); h = F.silu(h)

        h = self.out_conv(h)
        
        out_flat = h.view(B, -1)[:, :L]
        return out_flat.view(original_shape)


class UNet1Dconv(nn.Module):
    """
    1D U-Net: conv3 ReLU blocks, 2 maxpool down, 2 up-conv up
    - Input: dims flattened to length data_dim
    - Output: input shape
    """

    def __init__(
        self, 
        data_dim: int, 
        base_dim: int = 64, 
        depth: int = 4, 
        time_emb_dim: int = 32
    ):
        super().__init__()
        self.data_dim = int(data_dim)
        self.base_dim = int(base_dim)
        self.depth = int(depth)
        self.time_emb_dim = int(time_emb_dim)

        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        self.chs = [base_dim * (2**i) for i in range(depth)]

        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth):
            in_ch = 1 if i == 0 else self.chs[i - 1]
            out_ch = self.chs[i]
            self.enc_convs.append(double_conv(in_ch, out_ch))
            if i < depth - 1:
                self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.upconvs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose1d(
                    self.chs[i], self.chs[i - 1], kernel_size=2, stride=2
                )
            )
            self.dec_convs.append(double_conv(self.chs[i - 1] * 2, self.chs[i - 1]))

        self.final_conv1x1 = nn.Conv1d(self.chs[0], 1, kernel_size=1)

        self.time_proj_input = nn.Linear(time_emb_dim, self.chs[0])
        self.time_proj_enc = nn.ModuleList(
            [nn.Linear(time_emb_dim, c) for c in self.chs[1:]]
        )
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, self.chs[-1])
        self.time_proj_dec = nn.ModuleList(
            [nn.Linear(time_emb_dim, c) for c in self.chs[:-1][::-1]]
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _right_pad_to_multiple(self, L: int, multiple: int):
        return (multiple - (L % multiple)) % multiple

    def forward(self, x: torch.Tensor, sigma, class_labels=None, augment_labels=None):
        """
        x: ex. (B, 18)
        returns: tensor shaped as x
        """
        original_shape = x.shape
        B = original_shape[0]

        x_flat = x.view(B, -1)
        L = x_flat.shape[1]

        min_div = 2 ** (self.depth - 1)
        pad_len = self._right_pad_to_multiple(L, min_div)
        if pad_len > 0:
            x_flat = F.pad(x_flat, (0, pad_len))

        L_padded = x_flat.shape[1]
        h = x_flat.unsqueeze(1)

        temb = self.time_embedding(sigma)
        temb = self.time_mlp(temb)

        skips = []

        for i, enc_block in enumerate(self.enc_convs):
            h = enc_block(h)
            if i == 0:
                h = h + self.time_proj_input(temb).unsqueeze(-1)
            else:
                h = h + self.time_proj_enc[i - 1](temb).unsqueeze(-1)
            skips.append(h)
            if i < self.depth - 1:
                h = self.pools[i](h)

        h = h + self.time_proj_bottleneck(temb).unsqueeze(-1)

        for j, (upconv, dec_conv) in enumerate(zip(self.upconvs, self.dec_convs)):
            h = upconv(h)
            skip = skips[-2 - j]
            if h.shape[-1] != skip.shape[-1]:
                if h.shape[-1] > skip.shape[-1]:
                    h = h[..., : skip.shape[-1]]
                else:
                    diff = skip.shape[-1] - h.shape[-1]
                    h = F.pad(h, (0, diff))
            h = torch.cat([h, skip], dim=1)
            h = dec_conv(h) + self.time_proj_dec[j](temb).unsqueeze(-1)
            h = F.relu(h)

        out = self.final_conv1x1(h)
        out_flat = out.view(B, -1)[:, :L]
        out_reshaped = out_flat.view(*original_shape)
        return out_reshaped


# DDPM implementation, for testing, 1d array implementation
class NoisePredictorUNet(nn.Module):
    """
    1d unet version for ddpm modules.
    """

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
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # hidden dims: base, base*2, base*4, ...
        hidden_dims = [base_dim * (2**i) for i in range(depth)]
        self.hidden_dims = hidden_dims

        # input projection: D -> hidden_dims[0]
        self.input_fc = nn.Linear(self.data_dim, hidden_dims[0])
        self.input_ln = nn.LayerNorm(hidden_dims[0])
        self.enc_layers = nn.ModuleList()
        self.enc_ln = nn.ModuleList()

        for i in range(depth - 1):
            self.enc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.enc_ln.append(nn.LayerNorm(hidden_dims[i + 1]))

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

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        hidden_dims = [base_dim * (2**i) for i in range(depth)]
        self.hidden_dims = hidden_dims

        self.input_fc = nn.Linear(self.data_dim, hidden_dims[0])
        self.input_ln = nn.LayerNorm(hidden_dims[0])

        self.enc_layers = nn.ModuleList()
        self.enc_ln = nn.ModuleList()
        for i in range(depth - 1):
            self.enc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.enc_ln.append(nn.LayerNorm(hidden_dims[i + 1]))

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

        self.out_proj = nn.Linear(hidden_dims[0], self.data_dim * 2)

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
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel)
        )

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
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
        self.time_mlp = nn.Sequential(MP_SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1 = MPConv(in_channels, out_channels, kernel=(3, 3))
        self.conv2 = MPConv(out_channels, out_channels, kernel=(3, 3))
        self.act = MP_SiLU()
        self.skip_connection = (
            MPConv(in_channels, out_channels, kernel=(1, 1))
            if in_channels != out_channels
            else None
        )

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
        self.qkv = MPConv(channels, channels * 3, kernel=(1, 1))
        self.proj_out = MPConv(channels, channels, kernel=(1, 1))
        self.scale = channels**-0.5

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


# --- Shape preserving U-Net --
class MPTinyUNet(nn.Module):
    """
    A shape-preserving U-Net for low-dimensional data, still hase increasing
    and decreasing channel depths.
    """

    def __init__(
        self,
        in_channels=2,
        base_channels=32,
        time_emb_dim=32,
        channel_mults=[1, 2, 4],
        use_attention_at_level=[False, True, True],
    ):
        super().__init__()
        self.in_channels = in_channels

        # Time emb
        self.time_embedding = nn.Sequential(
            MPFourier(base_channels),  # MPFourier <--> TimeEmbedding ?
            nn.Linear(base_channels, time_emb_dim),
            MP_SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder path
        self.conv_in = MPResnetBlock(in_channels, base_channels, time_emb_dim)
        self.downs = nn.ModuleList()

        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            level_blocks = nn.ModuleList(
                [MPResnetBlock(current_channels, out_channels, time_emb_dim)]
            )
            if use_attention_at_level[i]:
                level_blocks.append(MPAttentionBlock(out_channels))
            self.downs.append(level_blocks)
            current_channels = out_channels

        # Bottleneck
        self.bottleneck = nn.ModuleList(
            [
                MPResnetBlock(current_channels, current_channels, time_emb_dim),
                MPAttentionBlock(current_channels),
                MPResnetBlock(current_channels, current_channels, time_emb_dim),
            ]
        )

        # Decoder path
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            skip_channels = base_channels * channel_mults[i]
            in_ch = current_channels + skip_channels

            level_blocks = nn.ModuleList()
            if use_attention_at_level[i]:
                level_blocks.append(
                    MPAttentionBlock(in_ch)
                )  # Attention on concatenated features
            level_blocks.append(MPResnetBlock(in_ch, out_channels, time_emb_dim))

            self.ups.append(level_blocks)
            current_channels = out_channels

        # Output
        self.conv_out = nn.Sequential(
            MP_SiLU(), MPConv(base_channels, in_channels, kernel=(1, 1))
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


# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))

        skip_x = self.skip(orig) if self.skip is not None else orig
        
        if skip_x.shape[-1] > x.shape[-1]:
            diff_y = skip_x.size()[2] - x.size()[2]
            diff_x = skip_x.size()[3] - x.size()[3]
            skip_x = skip_x[:, :, diff_y // 2 : diff_y // 2 + x.size()[2], diff_x // 2 : diff_x // 2 + x.size()[3]]

        x = x.add_(skip_x)
        
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

