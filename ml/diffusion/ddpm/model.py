import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from typing import Dict, Optional, Any, List
from ml.common.nn.positional_emb import TimeEmbedding
from ml.common.nn.modules import Module
from ml.diffusion.ddpm.samplers import SamplerVar
from ml.common.nn.unet import NoisePredictorUNet, TinyUNet
    
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
        """Internal helper that runs the core DDPM sampling loop."""
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
        if self.diffuser is None:
            raise RuntimeError("Diffuser must be attached to the model via set_diffuser()")
        
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
    

#needs to be edited into a wrapper, so its modular
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
        return model_out #eps_pred + v_pred

    def sample(self, num_samples: int, chunks: int = 20):
        """
        Public method for generating samples in memory-efficient chunks.
        This method instantiates and delegates the core logic to the SamplerVar class.
        """
        if self.diffuser is None:
            raise RuntimeError("A diffuser must be attached to the model via set_diffuser().")
        
        device = next(self.parameters()).device
        self.eval()

        sampler = SamplerVar(self, self.diffuser)

        chunk_size = math.ceil(num_samples / chunks)
        all_samples = []

        with torch.no_grad():
            for _ in tqdm(range(chunks), desc="Sampling DDPM in chunks"):
                n_to_sample = min(chunk_size, num_samples - len(all_samples))
                if n_to_sample <= 0:
                    break

                latents = torch.randn((n_to_sample, self.data_dim), device=device)

                samples_chunk = sampler.run(latents)
                all_samples.append(samples_chunk.cpu().numpy())
        
        return np.concatenate(all_samples, axis=0)