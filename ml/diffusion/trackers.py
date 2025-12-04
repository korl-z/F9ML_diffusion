import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from lightning import Callback

from ml.common.nn.modules import Tracker
from ml.common.stats.two_sample_tests import two_sample_plot
from ml.common.utils.plot_utils import (
    handle_plot_exception,
    iqr_remove_outliers,
    make_subplots_grid,
)


class DDPMTracker(Tracker, Callback):
    def __init__(self, experiment_conf, tracker_path, n_samples=10**3, n_bins=40):
        super().__init__(experiment_conf, tracker_path)
        self.n_samples = n_samples
        self.n_bins = n_bins
        self.generated = None

    def on_validation_epoch_end(self, trainer, pl_module):
        self.module = pl_module
        if self.compute(stage="val"):
            self.plot()
    
    def on_test_end(self, trainer, pl_module):
        self.module = pl_module
        if self.compute(stage="test"):
            self.plot()

    def make_plotting_dirs(self):
        return {
            "generated": f"{self.base_dir}/generated/",
        }

    def get_predictions(self, stage):
        self.stage = stage

        if stage == "val" or stage is None:
            dl = self.module._trainer.datamodule.val_dataloader()
        elif stage == "test":
            dl = self.module._trainer.datamodule.test_dataloader()
        else:
            raise ValueError(f"Stage must be one of ['val', 'test', None], got {stage} instead!")

        reference = dl.dataset.X.cpu().numpy()

        self.generated = self.module.model.sample(reference.shape[0]//10)

        num_generated_features = self.generated.shape[1]
        self.reference = reference[:, :num_generated_features]

        torch.cuda.empty_cache()

    def compute(self, stage=None):
        return super().compute(stage)

    def plot(self):
        if self.current_epoch % self.experiment_conf["check_metrics_n_epoch"] != 0:
            return False

        self.gen_vs_ref_plot()

        self.module.logger.experiment.log_artifact(local_path=self.base_dir, run_id=self.module.logger.run_id)

        return True

    @handle_plot_exception
    def gen_vs_ref_plot(self):
        D = self.generated.shape[1]
        ncols = 4
        nrows = int(np.ceil(D / ncols))

        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        fig.subplots_adjust(hspace=0.55, wspace=0.2, left=0.03, right=0.98, top=0.97, bottom=0.1)
        ax_flat = np.array(axs).reshape(-len(axs)) 

        two_sample_plot(
            A=self.reference,
            B=self.generated,
            axs=ax_flat,
            n_bins=self.n_bins,
        )

        for j in range(D, len(ax_flat)):
            ax_flat[j].axis("off")

        plt.tight_layout()
        fig.savefig(f"{self.plotting_dirs['generated']}generated_epoch{self.current_epoch:02d}_{self.stage}.png")
        plt.close(fig)