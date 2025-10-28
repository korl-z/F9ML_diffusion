import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import torch
import lightning as L
from typing import Optional, Dict, Any
import mlflow
from sklearn import preprocessing
from lightning.pytorch.callbacks import Callback

from ml.diffusion.score.model import RefineNet, SimpleUNet, ModularUNet
from ml.diffusion.score.score_matching import score_matching_loss, linear_noise_scale, geometric_noise_scale, maximum_eucl_dist
from ml.diffusion.score.langevin_dynamics import sample

from ml.common.nn.modules import Module
from ml.common.data_utils.downloadutils import load_dataset_variables
from ml.common.utils.plot_utils import add_data_mc_ratio


#temporary list for plotting
features_list = [
    "lepton pT",
    "lepton eta",
    "missing energy",
    "jet1 pT",
    "jet1 eta",
    "jet2 pT",
    "jet2 eta",
    "jet3 pT",
    "jet3 eta",
    "jet4 pT",
    "jet4 eta",
    "m jj",
    "m jjj",
    "m lv",
    "m jlv",
    "m bb",
    "m wbb",
    "m wwbb",
]


class NCNSModule(Module):
    """

    """

    def __init__(
        self,
        datamodule: Any,
        model_conf: Dict[str, Any],
        training_conf: Dict[str, Any],
        data_conf: Optional[Dict[str, Any]] = None,
        model: Optional[torch.nn.Module] = None,
        test_sample_count: int = 1024,
        tracker = None
    ):
        super().__init__(model_conf, training_conf, data_conf, model, tracker=None)
        self.save_hyperparameters(ignore=["model", "tracker"])
        
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.model = model
        self.datamodule = datamodule
        self.test_sample_count = test_sample_count

        self.noise_conf = model_conf["noise_schedule"]
        self.noise_scales = linear_noise_scale(self.noise_conf["sigma_start"], self.noise_conf["sigma_end"], self.noise_conf["num_scales"])

        self._train_losses = []
        self._val_losses = []

        self.train_loss_history = []
        self.val_loss_history = []

        self._test_outputs = []

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x0 = batch[0]
        else:
            x0 = batch
        x0 = x0.to(self.device)
        reshaped_x0 = x0.view(-1, *self.model_conf["img_shape"])

        loss = score_matching_loss(self.model, reshaped_x0, self.noise_scales)

        try:
            self._train_losses.append(loss.detach().cpu().item())
        except Exception:
            pass

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x0.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x0 = batch[0]
        else:
            x0 = batch
        x0 = x0.to(self.device)
        reshaped_x0 = x0.view(-1, *self.model_conf["img_shape"])

        loss = score_matching_loss(self.model, reshaped_x0, self.noise_scales)

        try:
            self._val_losses.append(loss.detach().cpu().item())
        except Exception:
            pass

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x0.size(0))

        return None

    def test_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x0 = batch[0]
        else:
            x0 = batch
        self._test_outputs.append(x0.detach().cpu())
        return None

    def on_train_epoch_end(self) -> None:
        if len(self._train_losses) > 0:
            epoch_mean = float(np.mean(self._train_losses))
            self.train_loss_history.append(epoch_mean)
            self.log("train_epoch_loss", epoch_mean, prog_bar=True)
        self._train_losses = []

    def on_validation_epoch_end(self) -> None:
        if len(self._val_losses) > 0:
            epoch_mean = float(np.mean(self._val_losses))
            self.val_loss_history.append(epoch_mean)
            self.log("val_epoch_loss", epoch_mean, prog_bar=True)
        self._val_losses = []

    def on_test_start(self) -> None:
        self._test_outputs = []

    def on_train_end(self) -> None:
        """
        At end of training generate samples using the (trained) model and log per-feature histograms + loss plot.
        Uses the sampler class (SamplerSBGM) instead of a model.sample(...) call.
        """
        dm = self.datamodule
        dm.setup(stage='test')

        test_loader = dm.test_dataloader()

        reals_list = []
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            if isinstance(x, torch.Tensor):
                reals_list.append(x.detach().cpu().numpy())
            else:
                reals_list.append(np.asarray(x))

        real = np.concatenate(reals_list, axis=0)

        shape = (self.test_sample_count, *self.model_conf["img_shape"])
        samples_2d = sample(self.model, shape, self.noise_scales, next(self.model.parameters()).device, n_steps=100, eps=2e-5)
        gen = samples_2d.view(self.test_sample_count, self.data_conf["input_dim"])

        if isinstance(gen, torch.Tensor):
            gen_np = gen.detach().cpu().numpy()
        else:
            gen_np = np.asarray(gen)

        D = real.shape[1]
        ncols = 6
        nrows = math.ceil(D / ncols)

        fig1, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        fig1.subplots_adjust(hspace=0.55, wspace=0.2, left=0.03, right=0.98, top=0.97, bottom=0.1)

        ax_flat = np.array(axs).reshape(-1)

        for feat_idx in range(D):
            ax = ax_flat[feat_idx]
            bin_edges = np.histogram_bin_edges(real[:, feat_idx], bins=40)

            real_counts, _ = np.histogram(real[:, feat_idx], bins=bin_edges)
            gen_counts, _  = np.histogram(gen_np[:, feat_idx], bins=bin_edges)

            sum_real = real_counts.sum()
            sum_gen = gen_counts.sum()

            # division by 0 errors
            if sum_gen == 0:
                gen_counts_scaled = np.full_like(gen_counts, 1e-8, dtype=float)
            else:
                scale_factor = float(sum_real) / float(sum_gen)
                gen_counts_scaled = gen_counts.astype(float) * scale_factor

            real_yerr = np.sqrt(real_counts.astype(float))
            gen_yerr  = np.sqrt(gen_counts.astype(float)) * (scale_factor if sum_gen != 0 else 1.0)

            real_counts_safe = real_counts.astype(float).copy()
            zero_mask = real_counts_safe == 0
            if np.any(zero_mask):
                real_counts_safe[zero_mask] = 1e-8
                real_yerr[zero_mask] = 1e-8  # if no MC data

            try:
                import seaborn as sns
                sns.histplot(real[:, feat_idx], bins=bin_edges, ax=ax, stat="density", color="C0", alpha=0.6, label="real")
            except Exception:
                ax.hist(real[:, feat_idx], bins=bin_edges, density=True, alpha=0.6, label="real", color="C0")

            ax.hist(gen_np[:, feat_idx], bins=bin_edges, density=True, histtype="step", lw=1.6, color="gray", label="gen")
            ax.set_yscale("log")
            plt.legend()
            ax.set_xlabel(features_list[feat_idx])

            add_data_mc_ratio(
                ax=ax,
                bin_edges=bin_edges,
                data_hist=gen_counts_scaled.astype(float),
                data_yerr=gen_yerr.astype(float),
                mc_hists=real_counts_safe[None, :].astype(float),
                mc_yerrs=real_yerr[None, :].astype(float),
                ylim=(0.5, 1.5),
                lower_ylabel="gen / real",
            )

        for j in range(D, len(ax_flat)):
            ax_flat[j].axis("off")
        plt.tight_layout()

        run_id = getattr(self.logger, "run_id", None)
        if run_id is not None:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_figure(fig1, "samples_hist.png")
        else:
            mlflow.log_figure(fig1, "samples_hist.png")
        self.print("Logged figures via mlflow")

        try:
            plt.close("all")
        except Exception:
            pass