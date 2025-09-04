import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import itertools
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from ml.common.utils.utils import make_beta_schedule

class DiffuserDDPM1D:
    def __init__(self, data, timesteps, beta):
        self.data = data
        self.steps = timesteps
        self.beta = beta

    def do_diffusion(self):
        # perform diffusion following equation 2
        # returns a list of q(x(t)) and x(t)
        # starting from t=0 (i.e., the dataset)

        distributions, samples = [None], [self.data]
        xt = self.data
        for t in range(int(self.steps)):
            q = torch.distributions.Normal(
                np.sqrt(1 - self.beta) * xt, np.sqrt(self.beta)
            )
            xt = q.sample()

            distributions.append(q)
            samples.append(xt)

        return distributions, samples


class DiffuserDDPM:
    """
    Simple forward diffusion for DDPM-style experiments that works for
    multidimensional data (shape [B, D], where D = num features).

    Args:
        data: torch.Tensor of shape [B, D]
        timesteps: int, number of forward diffusion steps
        beta: float or torch.Tensor
            - if float: same beta applied to every feature at every step
            - if tensor with shape (D,): per-feature beta (broadcast to samples)
    """

    def __init__(self, data: torch.Tensor, timesteps: int, beta: float):
        assert isinstance(data, torch.Tensor), "data must be a torch.Tensor"
        self.data = data
        self.steps = int(timesteps)
        if isinstance(beta, torch.Tensor):
            self.beta = beta.to(data.device).float()
        else:
            self.beta = torch.tensor(
                float(beta), device=data.device, dtype=torch.float32
            )

    def do_diffusion(self, timesteps: int = None, beta=None):
        """
        Run forward diffusion.

        Returns:
            distributions: list where distributions[t] is q(x_t | x_{t-1}) (None for t=0)
            samples: list of tensors [x0, x1, ..., x_T] where each x_t has shape [N, D]
        """
        T = self.steps if timesteps is None else int(timesteps)
        beta_val = (
            self.beta
            if beta is None
            else (
                beta.to(self.data.device)
                if isinstance(beta, torch.Tensor)
                else torch.tensor(float(beta), device=self.data.device)
            )
        )

        distributions = [None]
        samples = [self.data]
        xt = self.data

        # precompute sqrt factors; allow beta_val to be scalar or per-dim
        # if beta_val has shape (D,), broadcast it to sample shape when needed
        for t in range(T):
            # sqrt(1 - beta) * xt  -> same shape as xt
            # handle scalar beta or per-dim beta
            if beta_val.numel() == 1:
                sqrt_one_minus = torch.sqrt(1.0 - beta_val)
                scale = torch.sqrt(beta_val)
                loc = sqrt_one_minus * xt
                std = scale * torch.ones_like(xt)
            else:
                # beta_val shape (D,) -> reshape to (1, D) to broadcast over batch
                b = beta_val.view(1, -1)
                sqrt_one_minus = torch.sqrt(1.0 - b)
                scale = torch.sqrt(b)
                loc = sqrt_one_minus * xt
                std = scale.expand_as(xt)

            q = torch.distributions.Normal(loc, std)
            xt = q.sample()

            distributions.append(q)
            samples.append(xt)

        return distributions, samples


class DiffuserDDPMeps:
    """
    Responsible for the forward noise schedule and q-sampling operations.
    - timesteps: T
    - beta_schedule: either 'linear' or pass a tensor of betas
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,  #if using linear scheduler
        beta_end: float = 0.02,    #if using linear scheduler  
        betas: Optional[torch.Tensor] = None, #if given explicit beta schedule
        scheduler: str = "linear",
        device: Optional[torch.device] = None,
    ):
        self.T = int(timesteps)
        self.device = torch.device("cpu") if device is None else device

        if betas is not None:
            betas = betas.to(self.device).float()
            assert betas.shape[0] == self.T
            self.betas = betas
        else:
            # create schedule by name
            self.betas = make_beta_schedule(scheduler, self.T, beta_start, beta_end, device=self.device)

        self.alphas = 1.0 - self.betas 
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)  
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - self.alpha_cum)

    def to(self, device: torch.device):
        """Move all buffers to device (in-place)"""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cum = self.alpha_cum.to(device)
        self.sqrt_alpha_cum = self.sqrt_alpha_cum.to(device)
        self.sqrt_one_minus_alpha_cum = self.sqrt_one_minus_alpha_cum.to(device)
        return self

    def extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape):
        """
        Extract values from 1D arr at indices t and reshape to broadcast into x_shape.
        arr: [T] tensor
        t: [B] long tensor with values in 0..T-1
        x_shape: desired broadcast shape (B, D)
        """
        assert t.dim() == 1
        out = arr.gather(0, t.to(arr.device))  # shape [B]
        # reshape to [B, 1, ...] matching x_shape
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out.to(arr.device)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ):
        """
        Draw x_t from q(x_t | x_0) = sqrt(alpha_cum_t) * x0 + sqrt(1-alpha_cum_t) * noise
        x0: [B, D]
        t: [B] long tensor
        noise: optional noise [B, D] (if None, sampled)
        Returns: x_t [B, D], noise [B, D]
        """
        B, D = x0.shape
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)

        sqrt_ac = self.extract(self.sqrt_alpha_cum, t, x0.shape)  # [B,1] broadcast
        sqrt_om = self.extract(self.sqrt_one_minus_alpha_cum, t, x0.shape)
        x_t = sqrt_ac * x0 + sqrt_om * noise
        return x_t, noise

    def q_mean_var(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Get the analytic mean and variance of q(x_t | x_0)
        returns: mean [B,D] and var [B,D] (var is scalar broadcasted)
        """
        sqrt_ac = self.extract(self.sqrt_alpha_cum, t, x0.shape)
        sqrt_om = self.extract(self.sqrt_one_minus_alpha_cum, t, x0.shape)
        mean = sqrt_ac * x0
        var = sqrt_om**2
        return mean, var

    # helpers to get scalars for posterior sampling (for t single scalar)
    def posterior_variance(self, t_idx: int):
        """returns scalar posterior var for step t_idx (for t_idx>0)."""
        if t_idx == 0:
            return 0.0
        beta_t = float(self.betas[t_idx])
        alpha_cum_t = float(self.alpha_cum[t_idx])
        alpha_cum_prev = float(self.alpha_cum[t_idx - 1])
        var = beta_t * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum_t)
        return var

    def T_steps(self):
        return self.T


class DiffuseriDDPMeps:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,  #if using linear scheduler
        beta_end: float = 0.02,    #if using linear scheduler  
        betas: Optional[torch.Tensor] = None, #if given explicit beta schedule
        scheduler: str = "linear",
        device: Optional[torch.device] = None,
    ):
        self.T = int(timesteps)
        self.device = torch.device("cpu") if device is None else device

        if betas is not None:
            betas = betas.to(self.device).float()
            assert betas.shape[0] == self.T
            self.betas = betas
        else:
            # create schedule by name
            self.betas = make_beta_schedule(scheduler, self.T, beta_start, beta_end, device=self.device)

        self.alphas = 1.0 - self.betas 
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)  
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - self.alpha_cum)

        #buffers
        self.alpha_cum_prev = F.pad(self.alpha_cum[:-1], (1, 0), value=1.0)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_cum_prev) / (1.0 - self.alpha_cum)
        self.posterior_mean_coef2 = (1.0 - self.alpha_cum_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_cum)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cum_prev) / (1.0 - self.alpha_cum)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.sqrt_recip_alpha_cum = torch.sqrt(1.0 / self.alpha_cum)
        self.sqrt_recipm1_alpha_cum = torch.sqrt(1.0 / self.alpha_cum - 1)

    def to(self, device: torch.device):
        """Move all buffers to device (in-place)"""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cum = self.alpha_cum.to(device)
        self.sqrt_alpha_cum = self.sqrt_alpha_cum.to(device)
        self.sqrt_one_minus_alpha_cum = self.sqrt_one_minus_alpha_cum.to(device)
        return self

    def extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape):
        """
        Extract values from 1D arr at indices t and reshape to broadcast into x_shape.
        arr: [T] tensor
        t: [B] long tensor with values in 0..T-1
        x_shape: desired broadcast shape (B, D)
        """
        assert t.dim() == 1
        out = arr.gather(0, t.to(arr.device))  # shape [B]
        # reshape to [B, 1, ...] matching x_shape
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out.to(arr.device)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ):
        """
        Draw x_t from q(x_t | x_0) = sqrt(alpha_cum_t) * x0 + sqrt(1-alpha_cum_t) * noise
        x0: [B, D]
        t: [B] long tensor
        noise: optional noise [B, D] (if None, sampled)
        Returns: x_t [B, D], noise [B, D]
        """
        B, D = x0.shape
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)

        sqrt_ac = self.extract(self.sqrt_alpha_cum, t, x0.shape)  # [B,1] broadcast
        sqrt_om = self.extract(self.sqrt_one_minus_alpha_cum, t, x0.shape)
        x_t = sqrt_ac * x0 + sqrt_om * noise
        return x_t, noise
    
    def q_posterior_mean_variance(self, x0, x_t, t):
        """Compute the mean and variance of the true posterior q(x_{t-1} | x_t, x_0)."""
        alpha_cum_prev = self.extract(self.alpha_cum_prev, t, x_t.shape)
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x0 +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, model_output, x_t, t):
        """Compute the mean and variance of the learned model p_theta(x_{t-1} | x_t)."""
        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        # Split the model output into eps and v predictions
        eps_pred = model_output[:, :C]
        v_pred = model_output[:, C:]

        # Interpolate between beta_t and tilde_beta_t using v
        log_tilde_beta = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        log_beta = self.extract(torch.log(self.betas), t, x_t.shape)
        
        # The model predicts v in [-1, 1], we scale it to [0, 1] for interpolation
        frac = (v_pred.clamp(-1, 1) + 1) / 2
        model_log_variance = frac * log_tilde_beta + (1 - frac) * log_beta

        # Calculate the predicted mean
        pred_x0 = self._predict_x0_from_eps(x_t, t, eps_pred)
        model_mean, _ = self.q_posterior_mean_variance(pred_x0, x_t, t)

        return model_mean, model_log_variance

    def _predict_x0_from_eps(self, x_t, t, eps):
        return (
            self.extract(self.sqrt_recip_alpha_cum, t, x_t.shape) * x_t -
            self.extract(self.sqrt_recipm1_alpha_cum, t, x_t.shape) * eps
        )