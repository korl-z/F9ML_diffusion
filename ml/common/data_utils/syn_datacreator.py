import numpy as np
import torch


# synthetic data creator
def create_custom_multidim_dataset(
    n_samples: int = 10000,
    n_features: int = 22,
    label_random: bool = True,
    signal_frac: float = 0.5,
    seed: int = 42,
    class0_mean: float = -4.0,
    class1_mean: float = 4.0,
    class0_std: float = 1.5,
    class1_std: float = 1.0,
) -> np.ndarray:
    """
    Create synthetic dataset shaped (n_samples, n_features + 1),last column a label.
    """
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    n = int(n_samples)
    D = int(n_features)

    if label_random:
        probs = torch.full((n,), float(signal_frac))
        labels = torch.bernoulli(probs, generator=rng).long()
    else:
        latent = torch.randn(n, generator=rng) * 0.8
        labels = (latent > 0.0).long()

    feat_offsets = (torch.rand(D, generator=rng) - 0.5) * 0.6  # shape (D,)

    X = torch.empty((n, D), dtype=torch.float32)
    for cls in (0, 1):
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        m = idx.numel()

        if cls == 0:
            base_mean = float(class0_mean)
            base_std = float(class0_std)
        else:
            base_mean = float(class1_mean)
            base_std = float(class1_std)

        chosen_means = base_mean + feat_offsets.unsqueeze(0)
        chosen_means = chosen_means.expand(m, D).contiguous()
        chosen_stds = torch.full((m, D), base_std)

        sampled = torch.normal(chosen_means, chosen_stds, generator=rng)

        skew_mask = torch.rand((m, D), generator=rng) < 0.01
        if skew_mask.any():
            expo = torch.distributions.Exponential(torch.tensor(1.0)).sample((m, D))
            sampled = torch.where(skew_mask, chosen_means + expo * 1.0, sampled)

        X[idx, :] = sampled

    perm = torch.randperm(n, generator=rng)
    X = X[perm]
    labels = labels[perm]

    out = torch.cat([X, labels.unsqueeze(1).float()], dim=1).numpy().astype(np.float32)
    return out
