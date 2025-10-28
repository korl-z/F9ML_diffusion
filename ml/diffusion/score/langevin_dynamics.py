from tqdm import tqdm
import torch

@torch.no_grad()
def langevin_step(model, x, step_size, noise_idx):
    # The score is the gradient of the log-probability density function
    score = model(x, noise_idx.to(x.device))
    noise = torch.randn_like(x)
    # Move towards higher probability density (score) with some random noise
    x = x + (step_size / 2) * score + torch.sqrt(step_size) * noise
    return x

@torch.no_grad()
def langevin_dynamics(model, x, step_size, noise_idx, n_steps=100):
    for _ in range(n_steps):
        x = langevin_step(model, x, step_size, noise_idx)
    return x

@torch.no_grad()
def annealed_langevin_dynamics(model, x, noise_scales, n_steps=100, eps=2e-5):
    bsz = x.size(0)
    for i in tqdm(range(len(noise_scales)), desc="Annealed Langevin Dynamics"):
        # Create a noise index tensor for the current step for the whole batch
        noise_idx = (torch.ones(bsz, device=x.device) * i).long()
        step_size = eps * (noise_scales[i] / noise_scales[-1]) ** 2
        x = langevin_dynamics(model, x, step_size, noise_idx, n_steps)
    return x

@torch.no_grad()
def sample(model, shape, noise_scales, device, n_steps=100, eps=2e-5):
    model.eval()
    # Start from random noise
    x = torch.rand(shape).to(device)
    x = annealed_langevin_dynamics(model, x, noise_scales.to(device), n_steps, eps)
    return x