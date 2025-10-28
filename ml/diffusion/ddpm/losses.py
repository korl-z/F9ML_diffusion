import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Any
import torch.nn as nn
import torch.nn.functional as F


class DDPMLoss:
    """
    DDPM noise prediction (MSE loss).
    """

    def __init__(self, diffuser: Any):
        self.diffuser = diffuser

    def __call__(self, model: nn.Module, x0: torch.Tensor):
        device = x0.device
        B = x0.shape[0]

        t = torch.randint(0, self.diffuser.T, (B,), device=device).long()
        x_t, noise_target = self.diffuser.q_sample(x0, t)

        noise_pred = model(x_t, t)
        loss = F.mse_loss(noise_pred, noise_target)

        return loss


# Improved DDPM loss (simple + vlb)
"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


class HybridLoss:
    """
    loss from Improved DDPM: L_simple + lambda * L_vlb.
    """

    def __init__(self, diffuser: Any, vlb_weight: float = 0.001):
        self.diffuser = diffuser
        self.vlb_weight = vlb_weight

    def _get_vlb_loss(self, model_output, x0, x_t, t):
        """L_vlb term."""
        true_mean, true_log_var = self.diffuser.q_posterior_mean_variance(
            x0=x0, x_t=x_t, t=t
        )

        pred_mean, pred_log_var = self.diffuser.p_mean_variance(model_output, x_t, t)

        kl_div = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var)
        kl_div = kl_div.mean(dim=list(range(1, len(kl_div.shape)))) / np.log(2.0)

        vlb_loss = torch.where(t == 0, kl_div.new_zeros(kl_div.shape), kl_div).mean()
        return vlb_loss

    def __call__(self, model: nn.Module, x0: torch.Tensor):
        device = x0.device
        B = x0.shape[0]

        t = torch.randint(0, self.diffuser.T, (B,), device=device).long()
        x_t, noise_target = self.diffuser.q_sample(x0, t)

        model_output = model(x_t, t)
        eps_pred = model_output[:, : x0.shape[1]]

        simple_loss = F.mse_loss(eps_pred, noise_target)

        with torch.no_grad():
            vlb_loss = self._get_vlb_loss(model_output.detach(), x0, x_t, t)

        # print(f"r = {simple_loss / vlb_loss}")
        return simple_loss + self.vlb_weight * vlb_loss


# iDDPM loss (simple) from EDM paper
class iDDPMloss:
    """
    Loss function for the iDDPM (EDM paper framework).
    """

    def __init__(self, u_buffer):
        super().__init__()
        self.register_buffer("u", u_buffer)
        self.M = len(self.u) - 1

    def __call__(self, net, images, labels=None, augment_pipe=None):
        j = torch.randint(0, self.M, (images.shape[0],), device=images.device)
        sigma = self.u[j].reshape(-1, 1, 1, 1)
        weight = 1 / (sigma**2)
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, class_labels=labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def to(self, device):
        self.u = self.u.to(device)
        return self
