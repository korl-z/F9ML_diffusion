import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Any
import torch.nn as nn
import torch.nn.functional as F

class LossDDPM1D:
    def __init__(self, model):
        self.model = model

    def __call__(self, forward_dist, forward_samples):
        loss = (
            -torch.distributions.Normal(
                torch.zeros(forward_samples[0].shape),
                torch.ones(forward_samples[0].shape),
            )
            .log_prob(forward_samples[-1])
            .mean()
        )

        for t in range(1, len(forward_samples)):
            xt = forward_samples[t]  # x(t)
            xprev = forward_samples[t - 1]  # x(t-1)
            q = forward_dist[t]  # q( x(t) | x(t-1) )

            # normalize timestep
            t_scaled = torch.full((xt.shape[0], 1), t / len(forward_samples))

            mu, sigma = self.model(xt, t_scaled)

            p = torch.distributions.Normal(mu, sigma)

            loss -= torch.mean(p.log_prob(xprev))
            loss += torch.mean(q.log_prob(xt))

        return loss / len(forward_samples)


class LossDDPM:
    def __init__(self, model, timesteps=None):
        """
        model: the diffusion model that implements `mu, sigma = model(x, t)`
        timesteps: optional int T (used only if you want to pass normalized floats;
                   model._prepare_t handles ints too if implemented)
        """
        self.model = model
        self.timesteps = timesteps

    def __call__(self, forward_dist, forward_samples):
        """
        forward_dist: list of torch.distributions (q(x_t | x_{t-1})) with None at index 0
        forward_samples: list of tensors [x0, x1, ..., x_T], each of shape [B, D]
        """
        # x_T term: p(x_T) is standard normal per-dimension
        xT = forward_samples[-1]  # [B, D]
        standard_logprob = torch.distributions.Normal(
            torch.zeros_like(xT), torch.ones_like(xT)
        ).log_prob(
            xT
        )  # [B, D]
        # sum over features, mean over batch
        loss = -torch.mean(torch.sum(standard_logprob, dim=1))

        T = len(forward_samples) - 1  # number of forward steps

        for t in range(1, len(forward_samples)):
            xt = forward_samples[t]  # x(t)    shape [B, D]
            xprev = forward_samples[t - 1]  # x(t-1)  shape [B, D]
            q = forward_dist[t]  # distribution object for q(x_t | x_{t-1})

            # timestep for the model: pass integers (model can normalize internally)
            t_tensor = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)

            # model predicts mu and sigma for p(x_{t-1} | x_t)
            mu, sigma = self.model(xt, t_tensor)  # each [B, D]

            p = torch.distributions.Normal(mu, sigma)
            p_logprob = p.log_prob(xprev)  # [B, D]
            q_logprob = q.log_prob(xt)  # [B, D]

            loss -= torch.mean(torch.sum(p_logprob, dim=1))
            loss += torch.mean(torch.sum(q_logprob, dim=1))

        loss = loss / (T + 1)  # forward_samples length == T+1
        return loss

class LossDDPMNoise:
    """
    Loss that implements the DDPM noise-prediction objective:
      sample t ~ Uniform({0..T-1}), noise ~ N(0,I)
      x_t = sqrt(alpha_cum_t) * x0 + sqrt(1 - alpha_cum_t) * noise
      loss = MSE(eps_pred, noise)
    """

    def __init__(self, model: nn.Module, diffuser):
        self.model = model
        self.diffuser = diffuser

    def __call__(self, x0: torch.Tensor):
        """
        x0: [B, D] raw batch
        returns: scalar loss (torch.tensor)
        """
        device = next(self.model.parameters()).device
        B = x0.shape[0]

        t = torch.randint(0, self.diffuser.T, (B,), device=device).long()
        noise = torch.randn_like(x0, device=device)

        x_t, eps = self.diffuser.q_sample(x0, t, noise=noise)
        eps_pred = self.model(x_t, t)

        loss = F.mse_loss(eps_pred, eps)
        return loss
    

class DDPMLoss:
    """
    Implements the standard DDPM noise-prediction objective.
    """
    def __init__(self, diffuser: Any):
        if diffuser is None:
            raise ValueError("DDPMLoss requires a 'diffuser' instance.")
        self.diffuser = diffuser

    def __call__(self, model: nn.Module, x0: torch.Tensor):
        """
        Calculates the DDPM loss for a given batch of clean data.
        Args:
            model: The noise prediction network.
            x0: A batch of clean data, shape [B, D].
        Returns:
            A single scalar tensor representing the mean loss over the batch.
        """
        device = x0.device
        B = x0.shape[0]

        t = torch.randint(0, self.diffuser.T, (B,), device=device).long()
        x_t, noise_target = self.diffuser.q_sample(x0, t)
        
        noise_pred = model(x_t, t)
        loss = F.mse_loss(noise_pred, noise_target)
        
        return loss
    
#Improved DDPM loss (simple + vlb)
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
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
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
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class HybridLoss:
    """
    Implements the hybrid loss from 'Improved DDPM': L_simple + lambda * L_vlb.
    """
    def __init__(self, diffuser: Any, vlb_weight: float = 0.001):
        if diffuser is None:
            raise ValueError("HybridLoss requires a 'diffuser' instance.")
        self.diffuser = diffuser
        self.vlb_weight = vlb_weight

    def _get_vlb_loss(self, model_output, x0, x_t, t):
        """Calculates the L_vlb term of the loss."""
        # Get the true posterior mean and variance
        true_mean, true_log_var = self.diffuser.q_posterior_mean_variance(x0=x0, x_t=x_t, t=t)
        
        # Get the models predicted mean and variance
        pred_mean, pred_log_var = self.diffuser.p_mean_variance(model_output, x_t, t)
        
        # Calculate the KL divergence
        kl_div = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var)
        kl_div = kl_div.mean(dim=list(range(1, len(kl_div.shape)))) / np.log(2.0) # Average over dims, convert to bits
        
        vlb_loss = torch.where(t == 0, kl_div.new_zeros(kl_div.shape), kl_div).mean()
        return vlb_loss

    def __call__(self, model: nn.Module, x0: torch.Tensor):
        device = x0.device
        B = x0.shape[0]

        t = torch.randint(0, self.diffuser.T, (B,), device=device).long()
        x_t, noise_target = self.diffuser.q_sample(x0, t)
        
        # Get the joint prediction from the model
        model_output = model(x_t, t)
        eps_pred = model_output[:, :x0.shape[1]] # First half is noise prediction

        # L_simple is the standard MSE loss on the noise
        simple_loss = F.mse_loss(eps_pred, noise_target)
        
        # L_vlb is the KL divergence term, calculated with a stop_gradient on the mean
        # as recommended by the paper to stabilize training.
        with torch.no_grad():
            vlb_loss = self._get_vlb_loss(model_output.detach(), x0, x_t, t)

        # The final hybrid loss
        # print(f"loss_ratio = {simple_loss / vlb_loss}")
        return simple_loss + self.vlb_weight * vlb_loss
    

#iDDPM loss (simple) from EDM paper

class iDDPMloss:
    """
    Loss function for the iDDPM model as described in the EDM paper framework.
    """
    def __init__(self, u_buffer):
        super().__init__()
        self.register_buffer('u', u_buffer)
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