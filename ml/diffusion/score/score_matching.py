import numpy as np
import torch
import torch.nn.functional as F

def linear_noise_scale(start=1., end=0.01, length=10):
    return torch.linspace(start, end, length)

def geometric_noise_scale(start=1., end=0.01, length=10, base=1.2):
    startlog = np.emath.logn(base, start)
    endlog = np.emath.logn(base, end)
    return torch.logspace(startlog, endlog, length, base)

from scipy.spatial.distance import pdist

def maximum_eucl_dist(x):
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        x_np = x
    else:
        raise TypeError(f"Input must be a NumPy array or a PyTorch tensor, but got {type(x)}")
    
    if x_np.ndim != 2:
        raise ValueError(f"Input must be a 2D array, but got shape {x_np.shape}")

    distances = pdist(x_np, 'euclidean')

    if distances.size == 0:
        return 0.0
    
    max_dist = distances.max()
    
    return float(max_dist)

def q_sample(x, sigma, noise=None):
    if noise is None:
        noise = torch.randn_like(x)
    while sigma.dim() < x.dim():
        sigma = sigma.unsqueeze(-1)
    return x + sigma * noise

def score_matching_loss(model, x, noise_scales):
    noise_scale_idx = torch.randint(0, noise_scales.shape[0], (x.shape[0],), device=x.device)
    noise_scale_batch = noise_scales[noise_scale_idx].view(-1, 1, 1, 1)

    noise = torch.randn_like(x)
    x_noisy = q_sample(x, noise_scale_batch, noise=noise)
    score = model(x_noisy, noise_scale_idx)
    
    # Target is -noise, prediction is score * sigma
    # The loss aims to make score = -noise / sigma
    loss = F.mse_loss(noise_scale_batch * score, -noise)
    return loss