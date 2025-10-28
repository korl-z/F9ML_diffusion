import torch
import math


class SimpleSampler:
    """
    Multidimensional reverse sampler for diffusion models.

    Args:
        model: a model with signature `mu, sigma = model(x, t, y=None)` and attribute `data_dim`.
        timesteps: int, number of reverse steps (T)
        count: int, number of samples to generate in total (batched internally)
        device: optional torch.device; if None the device of model params is used
    """

    def __init__(self, model, timesteps, count, device=None):
        self.model = model
        self.steps = int(timesteps)
        self.count = int(count)
        self.device = device if device is not None else next(model.parameters()).device

        if not hasattr(model, "data_dim"):
            raise ValueError(
                "Model must have attribute `data_dim` (number of features)."
            )
        self.data_dim = int(model.data_dim)

    def sample(
        self,
        batch_size=None,
        class_label=None,
        mean_only=False,
        return_history=False,
        t_as_float=False,
    ):
        """
        Draw samples.

        Args:
            batch_size: int or None. If None, generate all samples in one pass. If int, generate in chunks.
            class_label: optional int if model is conditional (model.class_emb exists)
            mean_only: if True, use the predicted mean each step (deterministic).
            return_history: if True, return stacked tensor of shape (T+1, N, D) else return final samples (N, D).
            t_as_float: if True, pass normalized float timesteps t/self.steps to model; otherwise pass integer tensors.

        Returns:
            If return_history: tensor shape [(T+1), count, D]
            Else: tensor shape [count, D] (final samples)
        """
        device = self.device
        total = self.count
        batch_size = total if batch_size is None else int(batch_size)
        results = []

        for start in range(0, total, batch_size):
            cur = min(batch_size, total - start)

            xt = torch.randn((cur, self.data_dim), device=device)

            if getattr(self.model, "class_emb", None) is not None:
                if class_label is None:
                    raise ValueError("Model is conditional; provide class_label.")
                y = torch.full(
                    (cur,), int(class_label), device=device, dtype=torch.long
                )
            else:
                y = None

            # history collector (optional)
            if return_history:
                history = [xt.detach().cpu()]

            # iterate backward
            for t_idx in range(self.steps, 0, -1):
                # prepare timestep argument for the model
                if t_as_float:
                    t_tensor = (
                        torch.full((cur,), float(t_idx), device=device)
                        / float(self.steps)
                    ).float()
                else:
                    t_tensor = torch.full(
                        (cur,), t_idx, device=device, dtype=torch.long
                    )

                # predict parameters
                # model should return mu, sigma both shaped [B, D]
                mu, sigma = (
                    self.model(xt, t_tensor, y)
                    if y is not None
                    else self.model(xt, t_tensor)
                )

                # ensure shapes match
                assert (
                    mu.shape == xt.shape and sigma.shape == xt.shape
                ), f"mu/sigma shape {mu.shape},{sigma.shape} must match xt {xt.shape}"

                if mean_only:
                    xt = mu
                else:
                    p = torch.distributions.Normal(mu, sigma)
                    xt = p.sample()

                if return_history:
                    history.append(xt.detach().cpu())

            # final step done; collect
            if return_history:
                # history is list length steps with tensors on CPU; stack to shape (T+1, cur, D)
                results.append(torch.stack(history, dim=0))
            else:
                results.append(xt.detach().cpu())

        # concat all chunks
        if return_history:
            # each item is (T+1, batch, D) -> concat on batch dim=1
            stacked = torch.cat(results, dim=1)  # (T+1, total, D)
            return stacked
        else:
            final = torch.cat(results, dim=0)  # (total, D)
            return final


class SamplerNoise:
    """
    Reverse DDPM sampler operating with a noise-predicting model (eps_pred).
    Uses Diffuser for schedule math.
    """

    def __init__(self, model, diffuser):
        """
        model: NoisePredictorMLP (predicts eps)
        diffuser: Diffuser instance
        """
        self.model = model
        self.diffuser = diffuser

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 100,
        device: torch.device = None,
        mean_only: bool = False,
    ):
        """
        Draw `batch_size` samples (final x0) from the model using DDPM reverse steps.
        Returns: tensor [batch_size, D] on CPU (detached).
        """
        if device is None:
            device = next(self.model.parameters()).device
        self.model.to(device)
        self.diffuser.to(device)

        D = self.model.data_dim
        T = self.diffuser.T

        x = torch.randn((batch_size, D), device=device)  # start from prior x_T ~ N(0,I)

        for t_idx in range(T - 1, -1, -1):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            # predict eps
            eps_pred = self.model(x, t)  # [B, D]

            # posterior mean: (1 / sqrt(alpha_t)) * (x_t - beta_t / sqrt(1 - alpha_cum_t) * eps_pred)
            alpha_t = float(self.diffuser.alphas[t_idx].item())
            beta_t = float(self.diffuser.betas[t_idx].item())
            alpha_cum_t = float(self.diffuser.alpha_cum[t_idx].item())
            # compute scalar broadcast tensors
            alpha_t_b = torch.full((batch_size, D), alpha_t, device=device)
            beta_t_b = torch.full((batch_size, D), beta_t, device=device)
            alpha_cum_t_b = torch.full((batch_size, D), alpha_cum_t, device=device)

            coef = beta_t_b / torch.sqrt(1.0 - alpha_cum_t_b)
            mean = (1.0 / torch.sqrt(alpha_t_b)) * (x - coef * eps_pred)

            if t_idx > 0:
                # posterior variance
                alpha_cum_prev = float(self.diffuser.alpha_cum[t_idx - 1].item())
                var = beta_t * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum_t)
                if mean_only:
                    x = mean
                else:
                    noise = torch.randn_like(x, device=device)
                    x = mean + math.sqrt(max(var, 0.0)) * noise
            else:
                x = mean

        return x.cpu().detach()


class SamplerVar:
    """
    Reverse DDPM sampler for a model that predicts both noise (eps) and variance (v).
    Performs the full T-step reverse process.
    """

    def __init__(self, model, diffuser):
        """
        Args:
            model: A model that outputs a [B, 2*D] tensor (e.g., VarPredictorUNet).
            diffuser: The DiffuserDDPMeps instance with the noise schedule.
        """
        self.model = model
        self.diffuser = diffuser

    @torch.no_grad()
    def run(self, latents):
        """
        Runs the full sampling loop on a single chunk of latents.

        Args:
            latents: The initial noise tensor of shape [B, D].

        Returns:
            A tensor of denoised samples of shape [B, D] on the same device.
        """
        T = self.diffuser.T
        x = latents
        device = latents.device

        # The original, full T-step reverse loop
        for t_idx in range(T - 1, -1, -1):
            # Create a 1D tensor for the current timestep, matching the batch size
            t = torch.full((x.shape[0],), t_idx, device=device, dtype=torch.long)

            # Get the joint output (eps and v) from the model
            model_output = self.model(x, t)

            # Use the diffuser's helper to get the learned mean and variance
            pred_mean, pred_log_var = self.diffuser.p_mean_variance(model_output, x, t)

            if t_idx > 0:
                noise = torch.randn_like(x)
                # Sample from the learned reverse distribution p_theta(x_{t-1} | x_t)
                x = pred_mean + (0.5 * pred_log_var).exp() * noise
            else:
                # At the final step, there's no noise
                x = pred_mean

        return x
