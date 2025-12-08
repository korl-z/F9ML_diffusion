import torch.nn as nn
import torch
import math
from tqdm import tqdm
import numpy as np

from ml.common.nn.positional_emb import TimeEmbedding, MPFourier
from ml.common.nn.unet import (
    normalize,
    resample,
    mp_cat,
    mp_sum,
    mp_silu,
    MPConv,
)  # magnitute preserving functions/blocks

class EDMPrecond(nn.Module):
    """
    Wrapper for the raw nn, outputs D_theta (denoiser prediction), see Karras et al. (2022).
    """

    def __init__(self, net, sigma_data=1.0, img_shape=None):
        super().__init__()
        self.net = net  # raw nn (F_theta)
        self.sigma_data = sigma_data
        self.img_shape = tuple(img_shape)
        self.n_features = np.prod(self.img_shape)

    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        #original formulation, has problems with predicting low variance, usually too low on high level features
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        # c_noise = 0.2 * sigma.flatten().sqrt()
        c_noise = sigma.flatten().log() / 4.0

        F_x = self.net(c_in * x, c_noise, class_labels, augment_labels)

        D_x = c_skip * x + c_out * F_x
        #----------------------------------------------------------------------------------------

        #raw network test
        # D_x = self.net(x, sigma.flatten(), class_labels, augment_labels)
        #----------------------------------------------------------------------------------------

        #simple net 1
        # c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        # c_noise = 0.2 * sigma.flatten().sqrt()
        # F_x = self.net(c_in * x, c_noise, class_labels, augment_labels)

        # D_x = x - sigma * F_x
        #----------------------------------------------------------------------------------------
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def _run_sampler(self, latents, sampler_cfg, randn_like=torch.randn_like):
        """Internal helper sampler, stolen form Karras 2022"""
        num_steps = sampler_cfg["num_steps"]
        sigma_min = sampler_cfg["sigma_min"]
        sigma_max = sampler_cfg["sigma_max"]
        rho = sampler_cfg["rho"]
        S_churn = sampler_cfg["S_churn"]
        S_min = sampler_cfg["S_min"]
        S_max = sampler_cfg["S_max"]
        S_noise = sampler_cfg["S_noise"]

        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # increase noise temporarily
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            # euler step.
            denoised = self(x_hat, t_hat).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # spply 2nd order correction.
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
# Preconditioning and uncertainty estimation.


class EDMPrecond2(torch.nn.Module):
    def __init__(
        self,
        net,
        img_shape=None,  # (num_channels, h, w)
        sigma_data=1.0,  # Expected standard deviation of the training data.
        logvar_channels=32,  # Intermediate dimensionality for uncertainty estimation.
    ):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

        self.sigma_data = sigma_data
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

        self.img_shape = tuple(img_shape)
        self.n_features = np.prod(self.img_shape)

    def forward(
        self,
        x,
        sigma,
        class_labels=None,
        augment_labels=None,
        return_logvar=True,
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        dtype = torch.float

        # Precond weights
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        x_in = (c_in * x).to(dtype)
        F_x = self.net(x_in, sigma.flatten(), class_labels, augment_labels)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty
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
        sigma_min = sampler_cfg["sigma_min"]
        sigma_max = sampler_cfg["sigma_max"]
        rho = sampler_cfg["rho"]
        S_churn = sampler_cfg["S_churn"]
        S_min = sampler_cfg["S_min"]
        S_max = sampler_cfg["S_max"]
        S_noise = sampler_cfg["S_noise"]

        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            denoised = self(x_hat, t_hat, return_logvar=False).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < num_steps - 1:
                denoised = self(x_hat, t_hat, return_logvar=False).to(torch.float64)
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


# Vp predcond:
class VPPrecond(nn.Module):
    def __init__(
        self,
        net,
        img_shape=None,  # (num_channels, h, w)
        M: int = 1000,
    ):
        super().__init__()
        self.net = net
        self.img_shape = tuple(img_shape)
        self.n_features = np.prod(self.img_shape)
        self.M = M

    def forward(
        self,
        x,
        sigma,
        class_labels=None,
        augment_labels=None,
    ):
        x = x.to(torch.float32)
        sigma_t = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        dtype = torch.float32

        vp_beta_d = getattr(self, "sampler_cfg", {}).get("beta_d", 19.9)
        vp_beta_min = getattr(self, "sampler_cfg", {}).get("beta_min", 0.1)

        # local helpers
        vp_sigma_inv = (
            lambda beta_d, beta_min: lambda sigma: (
                (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
            )
            / beta_d
        )
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)

        # precond weights
        c_skip = 1.0
        c_out = -sigma_t
        c_in = 1.0 / (sigma_t**2 + 1.0).sqrt()
        c_noise = (self.M - 1) * sigma_inv(sigma_t)

        F_x = self.net(
            (c_in * x).to(dtype), c_noise.flatten(), class_labels, augment_labels
        )
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    @torch.no_grad()
    def _run_sampler(self, latents, sampler_cfg):
        num_steps = sampler_cfg["num_steps"]
        vp_beta_d = sampler_cfg["beta_d"]
        vp_beta_min = sampler_cfg["beta_min"]
        epsilon_s = sampler_cfg["epsilon_s"]

        S_churn = sampler_cfg["S_churn"]
        S_min = sampler_cfg["S_min"]
        S_max = sampler_cfg["S_max"]
        S_noise = sampler_cfg["S_noise"]

        solver = sampler_cfg["solver"]
        subvp = sampler_cfg["subvp"]

        # for linear schedule #currently not modular, just comment out
        vp_sigma = (
            lambda beta_d, beta_min: lambda t: (
                np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1
            )
            ** 0.5
        )
        vp_sigma_deriv = (
            lambda beta_d, beta_min: lambda t: 0.5
            * (beta_min + beta_d * t)
            * (sigma(t) + 1 / sigma(t))
        )
        vp_sigma_inv = (
            lambda beta_d, beta_min: lambda sigma: (
                (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
            )
            / beta_d
        )

        # for cosine schedule
        # vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** ((beta_min + beta_d * 0.5) * t - beta_d * np.sin(np.pi * t)/ (2 * np.pi)) - 1) ** 0.5
        # vp_sigma_deriv = ... #need to be added
        # vp_sigma_inv = ... #(no anal. solution, need numerics)

        sigma_min = vp_sigma(vp_beta_d, vp_beta_min)(t=epsilon_s)
        sigma_max = vp_sigma(vp_beta_d, vp_beta_min)(t=1)

        beta_d = (
            2
            * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
            / (epsilon_s - 1)
        )
        beta_min = np.log(sigma_max**2 + 1) - 0.5 * beta_d

        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=latents.device
        )
        t_vals = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(beta_d, beta_min)(t_vals)

        # define schedule
        sigma = vp_sigma(beta_d, beta_min)
        sigma_deriv = vp_sigma_deriv(beta_d, beta_min)
        sigma_inv = vp_sigma_inv(beta_d, beta_min)

        # define scaling schedule.
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)

        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(self.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        t_next = t_steps[0]
        x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= sigma(t_cur) <= S_max
                else 0
            )
            t_hat = sigma_inv(self.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (
                sigma(t_hat) ** 2 - sigma(t_cur) ** 2
            ).clip(min=0).sqrt() * s(t_hat) * S_noise * torch.randn_like(x_cur)

            # Euler step.
            h = t_next - t_hat
            denoised = self(x_hat / s(t_hat), sigma(t_hat), class_labels=None).to(
                torch.float64
            )
            d_cur = (
                sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
            ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised

            if subvp == True:
                subvp_scale_cur = 1 - s(t_hat) ** 4
                x_prime = x_hat + h * d_cur * subvp_scale_cur
            else:
                x_prime = x_hat + h * d_cur
            t_prime = t_hat + h

            # apply 2nd order correction.
            if solver == "euler" or i == num_steps - 1:
                x_next = x_hat + h * d_cur
            else:
                assert solver == "heun"
                denoised = self(
                    x_prime / s(t_prime), sigma(t_prime), class_labels=None
                ).to(torch.float64)
                d_prime = (
                    sigma_deriv(t_prime) / sigma(t_prime)
                    + s_deriv(t_prime) / s(t_prime)
                ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(
                    t_prime
                ) * denoised

                if subvp == True:
                    subvp_scale_prime = 1 - s(t_prime) ** 4
                    x_next = x_hat + h * (
                        0.5 * d_cur + 0.5 * d_prime * subvp_scale_prime
                    )
                else:
                    x_next = x_hat + h * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sample(self, num_samples: int, chunks: int = 20, **sampler_kwargs):
        device = next(self.parameters()).device
        self.eval()

        sampler_cfg = {**self.sampler_cfg, **sampler_kwargs}
        chunk_size = math.ceil(num_samples / chunks)
        all_samples = []

        with torch.no_grad():
            for _ in tqdm(
                range(chunks),
                desc=f"Sampling in {chunks} chunks, using {sampler_cfg["solver"]} ODE solver. S_churn = {sampler_cfg['S_churn']}.",
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


# VE precond:
class VEPrecond(nn.Module):
    def __init__(
        self,
        net,
        img_shape=None,  # (num_channels, h, w)
        M: int = 1000,
    ):
        super().__init__()
        self.net = net
        self.img_shape = tuple(img_shape)
        self.n_features = np.prod(self.img_shape)
        self.M = M

    def forward(
        self,
        x,
        sigma,
        class_labels=None,
        augment_labels=None,
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        dtype = torch.float32

        #changed some stuff with c_noise, as the log was causing problems (similar structure to VEPrecond) I have to check again, if its working correctly!!
        vp_beta_d = getattr(self, "sampler_cfg", {}).get("beta_d", 19.9)
        vp_beta_min = getattr(self, "sampler_cfg", {}).get("beta_min", 0.1)
        vp_sigma_inv = (
            lambda beta_d, beta_min: lambda s: (
                (beta_min**2 + 2 * beta_d * (s**2 + 1).log()).sqrt() - beta_min
            ) / beta_d
        )
        
        t_conditioning = (self.M - 1) * vp_sigma_inv(vp_beta_d, vp_beta_min)(sigma)
        c_noise = t_conditioning.flatten()
        #changed some stuff with c_noise, as the log was causing problems SS

        c_skip = 1
        c_out = sigma
        c_in = 1
        # c_noise = (0.5 * sigma).log()
        
        F_x = self.net(
            (c_in * x).to(dtype), c_noise.flatten(), class_labels, augment_labels
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    @torch.no_grad()
    def _run_sampler(self, latents, sampler_cfg):
        num_steps = sampler_cfg["num_steps"]
        sigma_min = sampler_cfg["sigma_min"]
        sigma_max = sampler_cfg["sigma_max"]

        S_churn = sampler_cfg["S_churn"]
        S_min = sampler_cfg["S_min"]
        S_max = sampler_cfg["S_max"]
        S_noise = sampler_cfg["S_noise"]

        solver = sampler_cfg["solver"]

        ve_sigma = lambda t: t.sqrt()
        ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
        ve_sigma_inv = lambda sigma: sigma**2

        # create timestep arrays and convert to sigma
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=latents.device
        )
        t_vals = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = ve_sigma(t_vals)

        # define schedule
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv

        # define scaling schedule.
        s = lambda t: 1
        s_deriv = lambda t: 0

        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(self.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        t_next = t_steps[0]
        x_next = latents.to(torch.float64) * (
            sigma(t_next) * s(t_next)
        )  # rescale starting gaussian to proper width
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= sigma(t_cur) <= S_max
                else 0
            )
            t_hat = sigma_inv(self.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (
                sigma(t_hat) ** 2 - sigma(t_cur) ** 2
            ).clip(min=0).sqrt() * s(t_hat) * S_noise * torch.randn_like(x_cur)

            # Euler step.
            h = t_next - t_hat
            denoised = self(x_hat / s(t_hat), sigma(t_hat), class_labels=None).to(
                torch.float64
            )
            d_cur = (
                sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
            ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
            x_prime = x_hat + h * d_cur
            t_prime = t_hat + h

            # Apply 2nd order correction.
            if solver == "euler" or i == num_steps - 1:
                x_next = x_hat + h * d_cur
            else:
                assert solver == "heun"
                denoised = self(
                    x_prime / s(t_prime), sigma(t_prime), class_labels=None
                ).to(torch.float64)
                d_prime = (
                    sigma_deriv(t_prime) / sigma(t_prime)
                    + s_deriv(t_prime) / s(t_prime)
                ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(
                    t_prime
                ) * denoised
                x_next = x_hat + h * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def sample(self, num_samples: int, chunks: int = 20, **sampler_kwargs):
        device = next(self.parameters()).device
        self.eval()

        sampler_cfg = {**self.sampler_cfg, **sampler_kwargs}
        chunk_size = math.ceil(num_samples / chunks)
        all_samples = []

        with torch.no_grad():
            for _ in tqdm(
                range(chunks),
                desc=f"Sampling in {chunks} chunks, using {sampler_cfg["solver"]} ODE solver",
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
