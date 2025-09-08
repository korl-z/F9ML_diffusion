import torch.nn as nn
import numpy as np
import torch
import math

class TimeEmbedding(nn.Module):
    """
    Stanndard sinusuidal embedding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.ndim == 0:  #sampling problem with shapes/unsqueeze (sigmas in sampling have shapes [1, 1, 1, 1])
            t = t.unsqueeze(0)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = torch.log(t).unsqueeze(1) * embeddings.unsqueeze(0)  # Use log(t)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75). (karras 2024)
class MPFourier(torch.nn.Module):
    """
    Magnitude preserving fourier time embedding, used in Karras - Analyzing and Improving the Training Dynamics of Diffusion Models.
    """
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        if x.ndim == 0:   #fix for sampling problem
            x = x.unsqueeze(0)
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)
