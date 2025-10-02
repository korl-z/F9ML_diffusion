import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from typing import Optional, Any
from ml.diffusion.ddpm.samplers import SamplerVar
    
#standard ddpm implemenation, wrapper model with built in sampler methods
class DDPMPrecond(nn.Module):
    """
    A wrapper for the convolutional TinyUNet to make it compatible with an API
    that expects flat vector data (e.g., [B, 18]). It handles the reshaping
    of data from vector to a 2x3x3 image and back. The sampling and diffuser
    logic is preserved from the original MLP-based U-Net.
    """
    def __init__(
        self,
        net,
        data_dim,
        diffuser: Optional[Any] = None,
    ):
        super().__init__()
        self.net = net
        self.diffuser = diffuser
        self.data_dim = data_dim
            
    def set_diffuser(self, diffuser):
        """Attach a Diffuser instance post-construction."""
        self.diffuser = diffuser
        return self

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        The forward pass that bridges the vector and image domains.
        x: [B, D] - Flat input data (e.g., [B, 18])
        t: [B] - Integer timesteps 0..T-1
        returns: eps_pred [B, D] - Predicted noise in the same flat format
        """
        eps_pred = self.net(x, t)
        return eps_pred

    @torch.no_grad()
    def _run_sampler(self, latents, mean_only=False):
        """Internal helper."""
        T = self.diffuser.T
        x = latents
        device = latents.device
        
        for t_idx in range(T - 1, -1, -1):
            t = torch.full((x.shape[0],), t_idx, device=device, dtype=torch.long)
            eps_pred = self(x, t)

            alpha_t = self.diffuser.alphas[t_idx]
            alpha_cum_t = self.diffuser.alpha_cum[t_idx]
            
            coef = (1 - alpha_t) / torch.sqrt(1 - alpha_cum_t)
            mean = (1 / torch.sqrt(alpha_t)) * (x - coef * eps_pred)

            if t_idx > 0:
                if not mean_only:
                    var = self.diffuser.posterior_variance(t_idx)
                    noise = torch.randn_like(x, device=device)
                    x = mean + math.sqrt(var) * noise
                else:
                    x = mean
            else:
                x = mean
        return x

    def sample(self, num_samples: int, chunks: int = 20, mean_only: bool = False):
        """
        Public method for generating samples in chunks.
        """ 
        device = next(self.parameters()).device
        self.eval()

        chunk_size = math.ceil(num_samples / chunks)
        all_samples = []

        with torch.no_grad():
            for _ in tqdm(range(chunks), desc="Sampling DDPM in chunks"):
                n_to_sample = min(chunk_size, num_samples - len(all_samples))
                if n_to_sample <= 0:
                    break

                latents = torch.randn((n_to_sample, self.data_dim), device=device)
                samples_chunk = self._run_sampler(latents, mean_only=mean_only)
                all_samples.append(samples_chunk.cpu().numpy())

        return np.concatenate(all_samples, axis=0)
    

class iDDPMPrecond(nn.Module):
    def __init__(
        self,
        net,
        data_dim,
        diffuser: Optional[Any] = None,
    ):
        super().__init__()
        self.net = net
        self.diffuser = diffuser
        self.data_dim = data_dim

    def set_diffuser(self, diffuser):
        self.diffuser = diffuser
        return self
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: [B, D]
        t: [B] ints 0..T-1
        returns: A single tensor [B, 2*D]. The first D channels are eps_pred,
                 and the second D channels are v_pred.
        """
        model_out = self.net(x, t)
        return model_out #eps_pred + v_pred (v_pred is used as interpolation parameter (more stable than directly predicting Sigma))

    @torch.no_grad()
    def _run_sampler(self, latents):
        """Internal helper."""
        T = self.diffuser.T
        x = latents
        device = latents.device
        
        # The original, full T-step reverse loop
        for t_idx in range(T - 1, -1, -1):
            t = torch.full((x.shape[0],), t_idx, device=device, dtype=torch.long)
            
            model_output = self.model(x, t)
            
            pred_mean, pred_log_var = self.diffuser.p_mean_variance(model_output, x, t)
            
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = pred_mean + (0.5 * pred_log_var).exp() * noise
            else:
                x = pred_mean
                
        return x

    def sample(self, num_samples: int, chunks: int = 20):
        """
        Public method for generating samples in chunks.
        """ 
        device = next(self.parameters()).device
        self.eval()

        chunk_size = math.ceil(num_samples / chunks)
        all_samples = []

        with torch.no_grad():
            for _ in tqdm(range(chunks), desc="Sampling DDPM in chunks"):
                n_to_sample = min(chunk_size, num_samples - len(all_samples))
                if n_to_sample <= 0:
                    break

                latents = torch.randn((n_to_sample, self.data_dim), device=device)
                samples_chunk = self._run_sampler(latents)
                all_samples.append(samples_chunk.cpu().numpy())

        return np.concatenate(all_samples, axis=0)
    
class iDDPM2Precond(nn.Module):
    def __init__(self,
        net,                               
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        img_shape       = None,             # image shape (channels, h, w)
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_shape[1]
        self.img_channels = img_shape[0]
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.net = net
        self.img_shape = tuple(img_shape)
        self.n_features = np.prod(self.img_shape)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        dtype = torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.net((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, augment_labels=augment_labels)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)
    
    def _run_sampler(self, latents, sampler_cfg):
        """Internal helper for running the Euler ODE solver."""
        num_steps = sampler_cfg["num_steps"]
        sigma_min = sampler_cfg["sigma_min"]
        sigma_max = sampler_cfg["sigma_max"]

        # Time step discretization, uses "ti<N" formula in Table 1 for iDDPM.
        j_max = self.round_sigma(torch.tensor(sigma_max), return_index=True)
        j_min = self.round_sigma(torch.tensor(sigma_min), return_index=True)
        j_steps = torch.linspace(j_max, j_min, num_steps + 1, device=latents.device).long()
        t_steps = self.u[j_steps]

        x_next = latents.to(torch.float64) * t_steps[0]

        # Main sampling loop.
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # Euler step.
            denoised = self(x_cur, t_cur).to(torch.float64)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

        return x_next

    def sample(self, num_samples: int, chunks: int = 1, **sampler_kwargs):
        device = next(self.parameters()).device
        self.eval()

        # Merge default and user-provided sampler settings
        cfg = {**self.sampler_cfg, **sampler_kwargs}

        chunk_size = math.ceil(num_samples / chunks)
        all_samples = []

        with torch.no_grad():
            for _ in tqdm(
                range(chunks), desc=f"Sampling in {chunks} chunks, using Euler"
            ):
                n_to_sample = min(chunk_size, num_samples - len(all_samples))
                if n_to_sample == 0:
                    break
                latents = torch.randn((n_to_sample, *self.img_shape), device=device)
                samples_reshaped = self._run_sampler(latents, cfg)

                samples_chunk = (
                    samples_reshaped.view(n_to_sample, self.n_features).cpu().numpy()
                )
                all_samples.append(samples_chunk)

        return np.concatenate(all_samples, axis=0)