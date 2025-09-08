import math
import torch
import numpy as np
from typing import Optional
from copy import deepcopy
import lightning as L
from lightning.pytorch import Callback

#EMA
class EMA:
    def __init__(self, model, decay=0.9999, device="cpu", start_step=0):
        """
        model: nn.Module to track
        decay: alpha in [0,1). Typical: 0.999 -> 0.9999
        device: where to store shadow params
        start_step: do not update EMA until step >= start_step
        """
        self.decay = float(decay)
        self.device = torch.device(device)
        self.start_step = int(start_step)
        self.step = 0

        # shadow holds copies of model.state_dict() tensors on self.device
        self.shadow = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        self.backup = None

    def update(self, model):
        """Call after optimizer.step()."""
        self.step += 1
        if self.step < self.start_step:
            # initialize shadow to current params on first relevant step
            for k, v in model.state_dict().items():
                self.shadow[k] = v.detach().cpu().clone()
            return

        # standard EMA: shadow = decay * shadow + (1 - decay) * param_cpu
        for k, v in model.state_dict().items():
            v_cpu = v.detach().cpu()
            self.shadow[k].mul_(self.decay).add_(v_cpu, alpha=(1.0 - self.decay))

    def state_dict(self):
        """Return a serializable dict (for checkpointing)."""
        return {"decay": self.decay, "step": self.step, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.step = sd.get("step", 0)
        self.shadow = {k: v.clone().cpu() for k, v in sd["shadow"].items()}

    def apply_to(self, model):
        """
        Swap EMA weights into `model` in-place. Saves original params to `self.backup`.
        Use before sampling/evaluation.
        """
        if self.backup is not None:
            raise RuntimeError(
                "apply_to called but backup already exists. Call restore() first."
            )
        self.backup = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        # copy shadow into model (on model device)
        model_device = next(model.parameters()).device
        sd = model.state_dict()
        for k in sd.keys():
            sd[k].copy_(self.shadow[k].to(model_device))
        model.load_state_dict(sd)

    def restore(self, model):
        """Restore model params that were saved by apply_to()."""
        if self.backup is None:
            raise RuntimeError("No backup found. Call apply_to first.")
        model_device = next(model.parameters()).device
        sd = model.state_dict()
        for k in sd.keys():
            sd[k].copy_(self.backup[k].to(model_device))
        model.load_state_dict(sd)
        self.backup = None


# schedulers
def linear_beta_schedule(
    T: int, beta_start: float, beta_end: float, device: Optional[torch.device] = None
):
    """Simple linear schedule (beta_start -> beta_end)."""
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
    Cosine schedule from 'Improved DDPM' (Nichol & Dhariwal).
    Returns a length-T tensor of betas in (0,1).
    The beta_start / beta_end args are ignored for cosine but kept for API compatibility.
    """
    device = torch.device("cpu") if device is None else device
    steps = T
    t = torch.linspace(0, steps, steps + 1, device=device, dtype=torch.float64) / steps
    # compute alphas_cum using the cosine schedule
    alphas_cum = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    # normalize so that alpha_cum[0] == 1
    alphas_cum = (alphas_cum / alphas_cum[0]).clamp(min=1e-8)
    # betas_t = 1 - alpha_cum[t+1] / alpha_cum[t] for t=0..T-1
    betas = []
    for i in range(steps):
        numerator = float(max(1e-20, 1.0 - (alphas_cum[i + 1] / alphas_cum[i])))
        betas.append(numerator)
    betas = torch.tensor(betas, device=device, dtype=torch.float32)
    # clamp to avoid extreme values
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

#LR scheduler

from torch.optim.lr_scheduler import _LRScheduler

class InverseSqrtLR(_LRScheduler):
    """
    Inverse square root learning rate scheduler (per epoch):
        alpha(t) = alpha_ref / sqrt(max(t/t_ref, 1))
    """
    def __init__(self, optimizer, t_ref=1, last_epoch=-1, verbose=False):
        self.t_ref = t_ref
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        epoch = max(self.last_epoch, 1)
        factor = 1.0 / math.sqrt(max(epoch / self.t_ref, 1.0))
        return [base_lr * factor for base_lr in self.base_lrs]

# register it so getattr works
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
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg / 1e6 * self.rampup_ratio)
        
        ema_beta = 0.5 ** (self.batch_size / max(ema_halflife_nimg * 1e6, 1e-8))

        with torch.no_grad():
            for p_ema, p_net in zip(pl_module.model_ema.parameters(), pl_module.model.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, 1 - ema_beta)) 


#Exponential movin average implementation from Analyzing and Improving the Training Dynamics of Diffusion Models (Karras)
def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp

#----------------------------------------------------------------------------
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

#----------------------------------------------------------------------------
# Compute inner products between the given pairs of EMA profiles
# according to Equation 151 and Algorithm 3.

def power_function_correlation(a_ofs, a_std, b_ofs, b_std):
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio ** t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den

#----------------------------------------------------------------------------
# Calculate beta for tracking a given EMA profile during training
# according to Equation 127.

def power_function_beta(std, t_next, t_delta):
    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta

#----------------------------------------------------------------------------
# Solve the coefficients for post-hoc EMA reconstruction
# according to Algorithm 3.

def solve_posthoc_coefficients(in_ofs, in_std, out_ofs, out_std): # => [in, out]
    in_ofs, in_std = np.broadcast_arrays(in_ofs, in_std)
    out_ofs, out_std = np.broadcast_arrays(out_ofs, out_std)
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = power_function_correlation(rv(in_ofs), rv(in_std), cv(in_ofs), cv(in_std))
    B = power_function_correlation(rv(in_ofs), rv(in_std), cv(out_ofs), cv(out_std))
    X = np.linalg.solve(A, B)
    X = X / np.sum(X, axis=0)
    return X

#----------------------------------------------------------------------------
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
        return [(ema, f'-{std:.3f}') for std, ema in zip(self.stds, self.emas)]

    def state_dict(self):
        return dict(stds=self.stds, emas=[ema.state_dict() for ema in self.emas])

    def load_state_dict(self, state):
        self.stds = state['stds']
        for ema, s_ema in zip(self.emas, state['emas']):
            ema.load_state_dict(s_ema)

#modified class that subclasses Callback, should work with mlflow logger
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

        with torch.no_grad():
            for p_ema, p_net in zip(pl_module.model_ema.parameters(), pl_module.model.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, 1 - beta))

import functools # <--- ADD THIS IMPORT

class MagnitudeMonitor(Callback):
    """
    A Lightning Callback to monitor and log the L2 norms of weights and activations
    for specified layers during training.
    
    This version uses a picklable hook to be compatible with Lightning's checkpointing.
    """
    def __init__(self, layer_prefixes_to_monitor=None):
        super().__init__()
        if layer_prefixes_to_monitor is None:
            self.layer_prefixes = [
                'conv_in',
                'bottleneck',
                'downs.0',
                'ups.0',
            ]
        else:
            self.layer_prefixes = layer_prefixes_to_monitor
            
        self.activations = {}
        # We store handles in a non-persistent attribute so Lightning doesn't try to pickle them.
        self._handles = []

    # --- FIX 1: The hook is now a proper method of the class ---
    def _activation_hook(self, name, module, input, output):
        """
        The hook function itself. It's a method, so it can be pickled.
        'name' is the layer name, pre-filled by functools.partial.
        """
        if isinstance(output, torch.Tensor):
            norm = output.detach().norm(2)
            dim_weighted_norm = norm / np.sqrt(np.prod(output.shape[1:]))
            self.activations[name] = dim_weighted_norm.cpu().item()

    def on_train_start(self, trainer, pl_module):
        """Called when training is about to start."""
        print("\n[MagnitudeMonitor] Attaching forward hooks to monitor activations...")
        for name, module in pl_module.model.net.named_modules():
            for prefix in self.layer_prefixes:
                if name.startswith(prefix):
                    # --- FIX 2: Use functools.partial to create a picklable hook closure ---
                    hook_fn = functools.partial(self._activation_hook, name)
                    handle = module.register_forward_hook(hook_fn)
                    self._handles.append(handle)
        print(f"[MagnitudeMonitor] Attached {len(self._handles)} hooks.")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of the validation epoch."""
        # Log weights (this part was already correct)
        for name, param in pl_module.model.net.named_parameters():
            for prefix in self.layer_prefixes:
                if name.startswith(prefix) and 'weight' in name:
                    norm = param.detach().norm(2)
                    dim_weighted_norm = norm / np.sqrt(np.prod(param.shape[1:]))
                    pl_module.log(f"weight_norm/{name}", dim_weighted_norm, on_step=False, on_epoch=True)

        # Log activations
        for name, act_norm in self.activations.items():
            pl_module.log(f"act_norm/{name}", act_norm, on_step=False, on_epoch=True)
            
        self.activations = {}

    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        print("\n[MagnitudeMonitor] Removing forward hooks.")
        for handle in self._handles:
            handle.remove()
        self._handles = [] # Clear the handles