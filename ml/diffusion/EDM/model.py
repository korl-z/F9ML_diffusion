import torch.nn as nn
import torch
import math
from tqdm import tqdm
import numpy as np

from ml.common.nn.positional_emb import TimeEmbedding, MPFourier
from ml.common.nn.unet import normalize, resample, mp_cat, mp_sum, mp_silu, MPConv  #magnitute preserving functions/blocks

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same")
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU()
        # self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm(h)
        time_cond = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_cond
        return self.act(h)
    
class SimpleUNet(nn.Module):
    """
    Simple unet for initial tests.
    """
    def __init__(self, in_channels=3, time_emb_dim=32, base_channels=32, channel_mults=[1, 2, 4]):
        super().__init__()
        self.in_channels = in_channels

        self.time_embedding = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        self.conv_in = ConvBlock(in_channels, base_channels*channel_mults[0], time_emb_dim)
        self.down1 = ConvBlock(base_channels*channel_mults[0], base_channels*channel_mults[1], time_emb_dim)
        self.bottleneck = ConvBlock(base_channels*channel_mults[1], base_channels*channel_mults[2], time_emb_dim)
        self.up1 = ConvBlock(base_channels*channel_mults[2] + base_channels*channel_mults[1], base_channels*channel_mults[1], time_emb_dim)
        self.up2 = ConvBlock(base_channels*channel_mults[1] + base_channels*channel_mults[0], base_channels*channel_mults[0], time_emb_dim)
        self.conv_out = nn.Conv2d(base_channels*channel_mults[0], in_channels, kernel_size=1)

    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        t_emb = self.time_embedding(sigma)

        # Encoder
        x1 = self.conv_in(x, t_emb)
        x2 = self.down1(x1, t_emb)

        # Bottleneck
        bn = self.bottleneck(x2, t_emb)

        # Decoder
        up1_in = torch.cat([bn, x2], dim=1)
        u1 = self.up1(up1_in, t_emb)

        up2_in = torch.cat([u1, x1], dim=1)
        u2 = self.up2(up2_in, t_emb)
        
        output = self.conv_out(u2)
        return output

class EDMPrecond(nn.Module):
    """
    Wrapper for the raw nn, outputs D_theta (denoiser prediction), see Karras et al. (2022).
    """
    def __init__(self, net, sigma_data=1.0, img_shape=None):
        super().__init__()
        self.net = net # raw nn (F_theta)
        self.sigma_data = sigma_data

        if img_shape is None:
            raise ValueError("'img_shape' missing (channels, height, width).")
        self.img_shape = tuple(img_shape)
        self.n_features = np.prod(self.img_shape)
    
    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        x = x.to(torch.float32)
        
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device)

        if sigma.ndim == 0:
            sigma_1d = sigma.expand(x.shape[0])
        else:
            sigma_1d = sigma.squeeze()
        
        sigma_4d = sigma_1d.view(-1, 1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma_4d**2 + self.sigma_data**2)
        c_out = sigma_4d * self.sigma_data / (sigma_4d**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma_4d**2 + self.sigma_data**2).sqrt()
        
        # F_x = self.net(c_in * x, sigma_1d, class_labels, augment_labels)
        F_x = self.net(c_in * x, sigma_1d)
        
        D_x = c_skip * x + c_out * F_x
        return D_x
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    def _run_sampler(self, latents, sampler_cfg, randn_like=torch.randn_like):
        """Internal helper sampler, stolen form Karras 2022"""
        num_steps = sampler_cfg["num_steps"]
        sigma_min = sampler_cfg["sigma_min"]
        sigma_max = sampler_cfg["sigma_max"]
        rho = sampler_cfg["rho"]
        S_churn = sampler_cfg.get("S_churn", 0)
        S_min = sampler_cfg.get("S_min", 0)
        S_max = sampler_cfg.get("S_max", float("inf"))
        S_noise = sampler_cfg.get("S_noise", 1)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps =  (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self(x_hat, t_hat).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self(x_next, t_next).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sample(self, num_samples: int, chunks: int = 20, **sampler_kwargs):
        device = next(self.parameters()).device
        self.eval()

        if hasattr(self, "sampler_cfg"):
            sampler_cfg = {**self.sampler_cfg, **sampler_kwargs}
        else:
            sampler_cfg = sampler_kwargs

        chunk_size = math.ceil(num_samples / chunks)
        all_samples = []

        with torch.no_grad():
            for _ in tqdm(
                range(chunks), desc=f"Sampling in {chunks} chunks, using Heun 2nd order"
            ):
                n_to_sample = min(chunk_size, num_samples - len(all_samples))
                if n_to_sample == 0:
                    break
                latents = torch.randn((n_to_sample, *self.img_shape), device=device)
                samples_reshaped = self._run_sampler(latents, sampler_cfg)

                samples_chunk = (
                    samples_reshaped.view(n_to_sample, self.n_features).cpu().numpy()
                )
                all_samples.append(samples_chunk)

        return np.concatenate(all_samples, axis=0)
    

# EDM2 model implementation:
# ----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        emb_channels,  # Number of embedding channels.
        flavor="enc",  # Flavor: 'enc' or 'dec'.
        resample_mode="keep",  # Resampling: 'keep', 'up', or 'down'.
        resample_filter=[1, 1],  # Resampling filter.
        attention=False,  # Include self-attention?
        channels_per_head=64,  # Number of channels per attention head.
        dropout=0,  # Dropout probability.
        res_balance=0.3,  # Balance between main branch (0) and residual branch (1).
        attn_balance=0.3,  # Balance between main branch (0) and self-attention (1).
        clip_act=256,  # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(
            out_channels if flavor == "enc" else in_channels,
            out_channels,
            kernel=[3, 3],
        )
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = (
            MPConv(in_channels, out_channels, kernel=[1, 1])
            if in_channels != out_channels
            else None
        )
        self.attn_qkv = (
            MPConv(out_channels, out_channels * 3, kernel=[1, 1])
            if self.num_heads != 0
            else None
        )
        self.attn_proj = (
            MPConv(out_channels, out_channels, kernel=[1, 1])
            if self.num_heads != 0
            else None
        )

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel norm & split
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(
                dim=3
            )
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


# ----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).


class UNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        in_channels,  # Image channels.
        label_dim,  # Class label dimensionality. 0 = unconditional.
        model_channels=32,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            4,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_noise=None,  # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb=None,  # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks=1,  # Number of residual blocks per resolution.
        attn_resolutions=[2],  # List of resolutions with self-attention.
        label_balance=0.5,  # Balance between noise embedding (0) and class embedding (1).
        concat_balance=0.5,  # Balance between skip connections (0) and main path (1).
        **block_kwargs,  # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = (
            model_channels * channel_mult_noise
            if channel_mult_noise is not None
            else cblock[0]
        )
        cemb = (
            model_channels * channel_mult_emb
            if channel_mult_emb is not None
            else max(cblock)
        )
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{res}x{res}_down"] = Block(
                    cout, cout, cemb, flavor="enc", resample_mode="down", **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="enc",
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f"{res}x{res}_in0"] = Block(
                    cout, cout, cemb, flavor="dec", attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = Block(
                    cout, cout, cemb, flavor="dec", **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = Block(
                    cout, cout, cemb, flavor="dec", resample_mode="up", **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"{res}x{res}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="dec",
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        self.out_conv = MPConv(cout, in_channels, kernel=[3, 3])

    def forward(self, x, noise_labels, class_labels):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(
                emb,
                self.emb_label(class_labels * np.sqrt(class_labels.shape[1])),
                t=self.label_balance,
            )
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x


# ----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.


class EDMPrecond2(torch.nn.Module):
    def __init__(
        self,
        net,
        img_shape=None, # (num_channels, h, w)
        label_dim=0,   # Class label dimensionality. 0 = unconditional.
        use_fp16=False,  # Run the model at FP16 precision?
        sigma_data=0.5,  # Expected standard deviation of the training data.
        logvar_channels=32,  # Intermediate dimensionality for uncertainty estimation.
        sigma_min=0.002, 
        sigma_max=80.0
    ):      
        super().__init__()
        self.net = net 
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min   #required for the built in sampler
        self.sigma_max = sigma_max     #required for the built in sampler

        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

        self.img_resolution = tuple(img_shape[1], img_shape[2])
        self.in_channels = img_shape[0]  

    def forward(
        self,
        x,
        sigma,
        class_labels=None,
        augment_labels=None,
        force_fp32=False,
        return_logvar=False,
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        # Preconditioning weights.
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model.
        x_in = (c_in * x).to(dtype)
        F_x = self.net(x_in, c_noise, class_labels, augment_labels)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(
                -1, 1, 1, 1
            )
            return D_x, logvar  # u(sigma) in Equation 21
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def _run_sampler(self, latents, sampler_cfg, randn_like=torch.randn_like):
        """Internal helper sampler, stolen form Karras 2022"""
        num_steps = sampler_cfg["num_steps"]
        sigma_min = max(sampler_cfg["sigma_min"], self.sigma_min)
        sigma_max = min(sampler_cfg["sigma_max"], self.sigma_max)
        rho = sampler_cfg["rho"]
        S_churn = sampler_cfg.get("S_churn", 0)
        S_min = sampler_cfg.get("S_min", 0)
        S_max = sampler_cfg.get("S_max", float("inf"))
        S_noise = sampler_cfg.get("S_noise", 1)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps =  (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self(x_hat, t_hat).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self(x_next, t_next).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sample(self, num_samples: int, chunks: int = 20, **sampler_kwargs):
        device = next(self.parameters()).device
        self.eval()

        if hasattr(self, "sampler_cfg"):
            sampler_cfg = {**self.sampler_cfg, **sampler_kwargs}
        else:
            sampler_cfg = sampler_kwargs

        chunk_size = math.ceil(num_samples / chunks)
        all_samples = []

        with torch.no_grad():
            for _ in tqdm(
                range(chunks), desc=f"Sampling in {chunks} chunks, using Heun 2nd order"
            ):
                n_to_sample = min(chunk_size, num_samples - len(all_samples))
                if n_to_sample == 0:
                    break
                latents = torch.randn((n_to_sample, *self.img_shape), device=device)
                samples_reshaped = self._run_sampler(latents, sampler_cfg)

                samples_chunk = (
                    samples_reshaped.view(n_to_sample, self.n_features).cpu().numpy()
                )
                all_samples.append(samples_chunk)

        return np.concatenate(all_samples, axis=0)


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

    def forward(self, x, sigma, class_labels=None, augment_labels=None):
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