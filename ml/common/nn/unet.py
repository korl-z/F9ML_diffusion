import sys 
sys.path.insert(1, "C:/Users/Uporabnik/Documents/IJS-F9/korlz")

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


###DDPM modules

#for initial tests, usless otherwise
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

#No in use anymore
def sinusoidal_time_embedding(t: torch.Tensor, dim: int):
    """
    t: 1D tensor of shape [B] (floats in [0,1] or ints). Returns [B, dim].
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / half)
    args = t[:, None].float() * freqs[None, :] * 2 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb  # [B, dim]

#doesnt perform well
class DiffusionModelMLP(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        time_emb_dim: int = 32,
        timesteps: int = 100,
    ):
        """
        data_dim: D (number of features)
        hidden_dim: width of MLP trunk
        n_hidden_layers: number of hidden linear layers (>=1)
        time_emb_dim: dimension of time embedding appended to x
        timesteps: integer T used to normalize integer t inputs (if you pass ints)
        """
        super().__init__()
        assert n_hidden_layers >= 1
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim
        self.timesteps = timesteps

        # time MLP to project sinusoidal embedding (optional small projector)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # trunk: input = D + time_emb_dim
        input_dim = data_dim + time_emb_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        hidden_layers = []
        for _ in range(n_hidden_layers - 1):
            hidden_layers.append(nn.LayerNorm(hidden_dim))
            hidden_layers.append(nn.SiLU())
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_net = nn.Sequential(*hidden_layers)

        # outputs: both map to D
        self.mu_out = nn.Linear(hidden_dim, data_dim)
        self.sigma_out = nn.Linear(hidden_dim, data_dim)
        # small positive offset on sigma
        self.sigma_eps = 1e-6

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prepare_t(self, t):
        """
        Accepts:
          - t float tensor [B] or [B,1] with values in [0,1]
          - OR t long/int tensor with values in [0..timesteps-1]
        Returns: t_float [B] in [0,1]
        """
        if t is None:
            raise ValueError("t must be provided")
        t = torch.as_tensor(t)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t.view(-1)
        # normalize int timesteps to [0,1]
        if t.dtype in (torch.long, torch.int) or t.max() > 1.0:
            t = t.float() / float(self.timesteps)
        else:
            t = t.float()
        return t.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor, t):
        """
        x: [B, D]
        t: either [B] or [B,1], ints in [0..T-1] or floats in [0,1]
        returns:
            mu: [B, D]
            sigma: [B, D] (positive)
        """
        assert (
            x.ndim == 2 and x.shape[1] == self.data_dim
        ), f"x must be [B,{self.data_dim}]"

        # prepare t
        t_f = self._prepare_t(t)  # [B]
        temb = sinusoidal_time_embedding(t_f, self.time_emb_dim)  # [B, time_emb_dim]
        temb = self.time_mlp(temb)  # [B, time_emb_dim]

        # concat time embedding once to each sample
        inp = torch.cat([x, temb], dim=-1)  # [B, D + time_emb_dim]
        h = F.silu(self.input_fc(inp))
        if len(self.hidden_net) > 0:
            h = self.hidden_net(h)

        mu = self.mu_out(h)  # [B, D]
        sigma = F.softplus(self.sigma_out(h)) + self.sigma_eps  # [B, D] positive
        return mu, sigma

#doesnt perform well
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

        # small MLP for time embedding projector
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
        # Accept ints or floats. Normalize ints by self.timesteps
        if t is None:
            raise ValueError("t must be provided")
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
    




#Old implementation


# class NoisePredictorUNet(nn.Module):
#     """
#     Small, fully-connected U-Net style architecture for vector-valued DDPM.
#     Interface matches NoisePredictorMLP: forward(x: [B, D], t) -> eps_pred [B, D].

#     Args:
#         data_dim: number of input features D
#         base_dim: number of channels at the first level (width)
#         depth: how many down/up blocks (depth >= 1). Each level doubles channels.
#         time_emb_dim: dimension of sinusoidal time embedding (projected by time_mlp)
#         timesteps: used to normalize integer timestep inputs
#     """

#     def __init__(
#         self,
#         data_dim: int,
#         base_dim: int = 128,
#         depth: int = 3,
#         time_emb_dim: int = 32,
#         timesteps: int = 1000,
#         diffuser: Optional[Any] = None,
#         sampler_cls: Optional[Any] = None,
#     ):
#         super().__init__()
#         assert depth >= 1, "depth must be >= 1"
#         self.data_dim = int(data_dim)
#         self.base_dim = int(base_dim)
#         self.depth = int(depth)
#         self.time_emb_dim = int(time_emb_dim)
#         self.timesteps = int(timesteps)

#         self.diffuser = diffuser

#         from ml.diffusion.ddpm.samplers import SamplerNoise  # adjust import path if needed
#         self.sampler_cls = sampler_cls if sampler_cls is not None else SamplerNoise
            
#         # time MLP projector
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_emb_dim, time_emb_dim),
#             nn.SiLU(),
#             nn.Linear(time_emb_dim, time_emb_dim),
#             nn.SiLU(),
#         )

#         # Build hidden dims: base, base*2, base*4, ...
#         hidden_dims = [base_dim * (2**i) for i in range(depth)]
#         self.hidden_dims = hidden_dims  # list of ints

#         # input projection: D -> hidden_dims[0]
#         self.input_fc = nn.Linear(self.data_dim, hidden_dims[0])
#         self.input_ln = nn.LayerNorm(hidden_dims[0])

#         # encoder: list of (Linear, LayerNorm)
#         self.enc_layers = nn.ModuleList()
#         self.enc_ln = nn.ModuleList()
#         for i in range(depth - 1):
#             self.enc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
#             self.enc_ln.append(nn.LayerNorm(hidden_dims[i + 1]))

#         # bottleneck (two small FCs)
#         self.bottleneck = nn.Sequential(
#             nn.Linear(hidden_dims[-1], hidden_dims[-1]),
#             nn.SiLU(),
#             nn.Linear(hidden_dims[-1], hidden_dims[-1]),
#             nn.SiLU(),
#         )
#         self.bottleneck_ln = nn.LayerNorm(hidden_dims[-1])

#         self.dec_layers = nn.ModuleList()
#         self.dec_ln = nn.ModuleList()
#         for i in range(depth - 1, 0, -1):
#             # skip has size hidden_dims[i], and h currently has size hidden_dims[i],
#             # concat size is hidden_dims[i] + hidden_dims[i]
#             in_dim = hidden_dims[i] + hidden_dims[i]
#             out_dim = hidden_dims[i - 1]
#             self.dec_layers.append(nn.Linear(in_dim, out_dim))
#             self.dec_ln.append(nn.LayerNorm(out_dim))

#         self.out_proj = nn.Linear(hidden_dims[0], self.data_dim)

#         # time-to-feature projection layers 
#         # - for input projection
#         self.time_proj_input = nn.Linear(time_emb_dim, hidden_dims[0])
#         # - for each encoder output (i -> hidden_dims[i+1])
#         self.time_proj_enc = nn.ModuleList(
#             [nn.Linear(time_emb_dim, d) for d in hidden_dims[1:]]
#         )
#         # - for bottleneck
#         self.time_proj_bottleneck = nn.Linear(time_emb_dim, hidden_dims[-1])
#         # - for each decoder output (matching dec out dims)
#         self.time_proj_dec = nn.ModuleList(
#             [nn.Linear(time_emb_dim, d) for d in hidden_dims[:-1][::-1]]
#         )

#         # initialization (similar style to MLP)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def set_diffuser(self, diffuser):
#         """Attach a Diffuser instance post-construction."""
#         self.diffuser = diffuser
#         return self
    
#     def _prepare_t(self, t):
#         # same semantics as NoisePredictorMLP
#         if t is None:
#             raise ValueError("t must be provided")
#         t = torch.as_tensor(t)
#         if t.ndim == 2 and t.shape[1] == 1:
#             t = t.view(-1)
#         if t.dtype in (torch.long, torch.int) or (t.max() > 1.0):
#             t = t.float() / float(self.timesteps - 1)  # map 0..T-1 to [0,1]
#         else:
#             t = t.float()
#         return t.to(next(self.parameters()).device)

#     def forward(self, x: torch.Tensor, t):
#         """
#         x: [B, D]
#         t: [B] ints 0..T-1 or floats in [0,1]
#         returns: eps_pred [B, D]
#         """
#         assert x.ndim == 2 and x.shape[1] == self.data_dim

#         t_f = self._prepare_t(t)  # [B]
#         temb = sinusoidal_time_embedding(t_f, self.time_emb_dim)  # [B, time_emb_dim]
#         temb = self.time_mlp(temb)  # [B, time_emb_dim]

#         # input proj + add time
#         h = self.input_fc(x) + self.time_proj_input(temb)
#         h = self.input_ln(h)
#         h = F.silu(h)

#         # encoder path, collect skips
#         skips = []
#         for i, enc in enumerate(self.enc_layers):
#             # enc input -> enc output dim hidden_dims[i+1]
#             h = enc(h)
#             # add time projection for this encoder level
#             h = h + self.time_proj_enc[i](temb)
#             h = self.enc_ln[i](h)
#             h = F.silu(h)
#             skips.append(h)

#         # bottleneck
#         h = self.bottleneck(h)
#         h = h + self.time_proj_bottleneck(temb)
#         h = self.bottleneck_ln(h)
#         h = F.silu(h)

#         # decoder path (mirror)
#         for j, dec in enumerate(self.dec_layers):
#             # corresponding skip (reverse order)
#             skip = skips[-1 - j]
#             # concat along feature dimension
#             h = torch.cat([h, skip], dim=-1)
#             h = dec(h)
#             # add time projection for this decoder level
#             h = h + self.time_proj_dec[j](temb)
#             h = self.dec_ln[j](h)
#             h = F.silu(h)

#         # final projection to data_dim
#         eps_pred = self.out_proj(h)
#         return eps_pred
    
#     def sample(self, num_samples: int, batch_size: Optional[int] = None, device: Optional[torch.device] = None, mean_only: bool = False):
#         """
#         Draw `num_samples` samples using the attached diffuser and sampler_cls.
#         Returns:
#             torch.Tensor of shape (num_samples, D) on CPU (detached).
#         Notes:
#             - If `batch_size` is provided, sampling is done in chunks of that size.
#             - Requires self.diffuser to be set (either passed at construction or via set_diffuser()).
#         """
#         if self.diffuser is None:
#             raise RuntimeError("No diffuser attached to the model. Call model.set_diffuser(diffuser) or pass diffuser at construction.")

#         # determine device for sampling
#         if device is None:
#             try:
#                 device = next(self.parameters()).device
#             except StopIteration:
#                 device = torch.device("cpu")

#         sampler = self.sampler_cls(self, self.diffuser)

#         # chunking
#         total = int(num_samples)
#         if total <= 0:
#             return torch.empty((0, self.data_dim), dtype=torch.float32)

#         if batch_size is None:
#             # ask sampler for all at once (sampler returns CPU tensor)
#             out = sampler.sample(total, device=device, mean_only=mean_only)
#             if isinstance(out, torch.Tensor):
#                 return out.cpu().detach()
#             else:
#                 return torch.as_tensor(np.asarray(out)).cpu()
#         else:
#             bs = int(batch_size)
#             chunks = []
#             remaining = total
#             while remaining > 0:
#                 cur = min(bs, remaining)
#                 out = sampler.sample(cur, device=device, mean_only=mean_only)
#                 if isinstance(out, torch.Tensor):
#                     chunks.append(out.cpu().detach())
#                 else:
#                     chunks.append(torch.as_tensor(np.asarray(out)))
#                 remaining -= cur
#             return torch.cat(chunks, dim=0)


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
            TimeEmbedding(base_channels),
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