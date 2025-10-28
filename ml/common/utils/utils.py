import math
import torch
import numpy as np
from typing import Optional
from copy import deepcopy
import lightning as L
from lightning.pytorch import Callback
from torch.optim.lr_scheduler import _LRScheduler


# schedulers
def linear_beta_schedule(
    T: int, beta_start: float, beta_end: float, device: Optional[torch.device] = None
):
    """linear schedule."""
    device = torch.device("cpu") if device is None else device
    return torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float32)


def cosine_beta_schedule(
    T: int,
    beta_start: float,
    beta_end: float,
    device: Optional[torch.device] = None,
    s: float = 0.008,
):
    """
    Cosine schedule from 'Improved DDPM'
    """
    device = torch.device("cpu") if device is None else device
    steps = T
    t = torch.linspace(0, steps, steps + 1, device=device, dtype=torch.float64) / steps
    alphas_cum = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_cum = (alphas_cum / alphas_cum[0]).clamp(min=1e-8)
    betas = []
    for i in range(steps):
        numerator = float(max(1e-20, 1.0 - (alphas_cum[i + 1] / alphas_cum[i])))
        betas.append(numerator)
    betas = torch.tensor(betas, device=device, dtype=torch.float32)
    return betas.clamp(min=1e-8, max=0.999)


def make_beta_schedule(
    name: str,
    T: int,
    beta_start: float,
    beta_end: float,
    device: Optional[torch.device] = None,
):
    name = name.lower()
    if name in ("linear", "lin"):
        return linear_beta_schedule(T, beta_start, beta_end, device=device)
    elif name in ("cosine",):
        return cosine_beta_schedule(T, beta_start, beta_end, device=device)
    else:
        raise ValueError(f"Unknown beta schedule '{name}'")


# LR scheduler
class InverseSqrtLR(_LRScheduler):
    """
    Inverse square root learning rate scheduler (per epoch):
        alpha(t) = alpha_ref / sqrt(max(t/t_ref, 1))
    """

    def __init__(self, optimizer, t_ref=1, last_epoch=-1):
        self.t_ref = t_ref
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = max(self.last_epoch, 1)
        factor = 1.0 / math.sqrt(max(epoch / self.t_ref, 1.0))
        return [base_lr * factor for base_lr in self.base_lrs]


# register so getattr works
torch.optim.lr_scheduler.InverseSqrtLR = InverseSqrtLR


class EMACallback(Callback):
    def __init__(self, decay_halflife_kimg=500, rampup_ratio=0.05, batchsize=512):
        """
        Args:
            decay_halflife_kimg: Half-life of the EMA decay
            rampup_ratio: A coefficient to speed up EMA during the initial phase of training.
            Set to 0.0 to disable ramp-up.
        """
        super().__init__()
        self.decay_halflife_kimg = decay_halflife_kimg
        self.rampup_ratio = rampup_ratio
        self.ema_state_dict = None
        self.batch_size = batchsize

    def on_fit_start(self, trainer, pl_module):
        pl_module.model_ema = deepcopy(pl_module.model)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        cur_nimg = (trainer.global_step + 1) * self.batch_size

        ema_halflife_nimg = self.decay_halflife_kimg * 1000
        if self.rampup_ratio > 0:
            ema_halflife_nimg = min(
                ema_halflife_nimg, cur_nimg / 1e6 * self.rampup_ratio
            )

        ema_beta = 0.5 ** (self.batch_size / max(ema_halflife_nimg * 1e6, 1e-8))

        with torch.no_grad():
            for p_ema, p_net in zip(
                pl_module.model_ema.parameters(), pl_module.model.parameters()
            ):
                p_ema.copy_(p_net.detach().lerp(p_ema, 1 - ema_beta))


# Exponential movin average implementation from Analyzing and Improving the Training Dynamics of Diffusion Models (Karras)
def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp


# ----------------------------------------------------------------------------
# Construct response functions for the given EMA profiles
# according to Equations 121 and 108.


def power_function_response(ofs, std, len, axis=0):
    ofs, std = np.broadcast_arrays(ofs, std)
    ofs = np.stack([np.float64(ofs)], axis=axis)
    exp = np.stack([std_to_exp(std)], axis=axis)
    s = [1] * exp.ndim
    s[axis] = -1
    t = np.arange(len).reshape(s)
    resp = np.where(t <= ofs, (t / ofs) ** exp, 0) / ofs * (exp + 1)
    resp = resp / np.sum(resp, axis=axis, keepdims=True)
    return resp


# ----------------------------------------------------------------------------
# Compute inner products between the given pairs of EMA profiles
# according to Equation 151 and Algorithm 3.


def power_function_correlation(a_ofs, a_std, b_ofs, b_std):
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio**t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den


# ----------------------------------------------------------------------------
# Calculate beta for tracking a given EMA profile during training
# according to Equation 127.


def power_function_beta(std, t_next, t_delta):
    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta


# ----------------------------------------------------------------------------
# Solve the coefficients for post-hoc EMA reconstruction
# according to Algorithm 3.


def solve_posthoc_coefficients(in_ofs, in_std, out_ofs, out_std):  # => [in, out]
    in_ofs, in_std = np.broadcast_arrays(in_ofs, in_std)
    out_ofs, out_std = np.broadcast_arrays(out_ofs, out_std)
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = power_function_correlation(rv(in_ofs), rv(in_std), cv(in_ofs), cv(in_std))
    B = power_function_correlation(rv(in_ofs), rv(in_std), cv(out_ofs), cv(out_std))
    X = np.linalg.solve(A, B)
    X = X / np.sum(X, axis=0)
    return X


# ----------------------------------------------------------------------------
# Class for tracking power function EMA during the training.
class PowerFunctionEMA:
    @torch.no_grad()
    def __init__(self, net, stds=[0.050, 0.100]):
        self.net = net
        self.stds = stds
        self.emas = [deepcopy(net) for _std in stds]

    @torch.no_grad()
    def reset(self):
        for ema in self.emas:
            for p_net, p_ema in zip(self.net.parameters(), ema.parameters()):
                p_ema.copy_(p_net)

    @torch.no_grad()
    def update(self, cur_nimg, batch_size):
        for std, ema in zip(self.stds, self.emas):
            beta = power_function_beta(std=std, t_next=cur_nimg, t_delta=batch_size)
            for p_net, p_ema in zip(self.net.parameters(), ema.parameters()):
                p_ema.lerp_(p_net, 1 - beta)

    @torch.no_grad()
    def get(self):
        for ema in self.emas:
            for p_net, p_ema in zip(self.net.buffers(), ema.buffers()):
                p_ema.copy_(p_net)
        return [(ema, f"-{std:.3f}") for std, ema in zip(self.stds, self.emas)]

    def state_dict(self):
        return dict(stds=self.stds, emas=[ema.state_dict() for ema in self.emas])

    def load_state_dict(self, state):
        self.stds = state["stds"]
        for ema, s_ema in zip(self.emas, state["emas"]):
            ema.load_state_dict(s_ema)


# modified class that subclasses Callback, should work ok with mlflow logger now
# class PFEMACallback(Callback):
#     def __init__(self, std, batchsize):
#         super().__init__()
#         self.std = std
#         self.batch_size = batchsize

#     def on_fit_start(self, trainer, pl_module):
#         pl_module.model_ema = deepcopy(pl_module.model)

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         cur_nimg = (trainer.global_step + 1) * self.batch_size

#         beta = power_function_beta(
#             std=self.std, t_next=cur_nimg, t_delta=self.batch_size
#         )

#         with torch.no_grad():
#             for p_ema, p_net in zip(
#                 pl_module.model_ema.parameters(), pl_module.model.parameters()
#             ):
#                 p_ema.copy_(p_net.detach().lerp(p_ema, 1 - beta))

class PFEMACallback(Callback):
    def __init__(self, std, batchsize):
        super().__init__()
        self.std = std
        self.batch_size = batchsize

    def on_fit_start(self, trainer, pl_module):
        pl_module.model_ema = deepcopy(pl_module.model)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        cur_nimg = (trainer.global_step + 1) * self.batch_size
        beta = power_function_beta(std=self.std, t_next=cur_nimg, t_delta=self.batch_size)

        #update EMA in-place 
        with torch.no_grad():
            for p_ema, p_net in zip(pl_module.model_ema.parameters(), pl_module.model.parameters()):
                # ema = ema * (1 - w) + net * w
                w = (1 - beta)
                p_ema.lerp_(p_net.detach(), w)

            # update buffers
            for b_ema, b_net in zip(pl_module.model_ema.buffers(), pl_module.model.buffers()):
                b_ema.copy_(b_net)