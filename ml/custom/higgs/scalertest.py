import sys

sys.path.insert(1, "C:/Users/Uporabnik/Documents/IJS-F9/korlz")

import os
import time
import logging
import torch
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers import MLFlowLogger

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.common.utils.register_model import register_from_checkpoint
from higgs_dataset import HiggsDataModule
from process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)
from ml.common.utils.utils import EMACallback, PFEMACallback

from ml.diffusion.EDM.model import SimpleUNet, TinyUNet
from ml.diffusion.EDM.lightning_EDM import EDMModule

from ml.common.utils.loggers import timeit, log_num_trainable_params, setup_logger
from ml.diffusion.trackers import DDPMTracker


class DropLabelProcessor:
    def __init__(self, drop_labels):
        self.drop_labels = drop_labels

    def __call__(self, data, selection, scalers):
        data = self.drop(data, selection)
        return data, selection, scalers

    def drop(self, data, selection):
        logging.info(f"Dropping labels {self.drop_labels}!")

        labels_idx = selection[selection["type"] == "label"].index

        for label_idx in labels_idx:
            for drop_label in self.drop_labels:
                mask_label = data[:, label_idx] == drop_label
                data = data[~mask_label]
                logging.info(
                    f"Dropped label {drop_label}! New data shape: {data.shape}."
                )

        return data


@hydra.main(version_base=None, config_path="config/edm/", config_name="main_config")
def main(cfg: DictConfig) -> None:
    setup_logger()

    experiment_conf = cfg.experiment_config
    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    experiment_name = "EDM_inital_test"

    data_conf = cfg.data_config
    model_conf = cfg.model_config
    training_conf = cfg.training_config

    accelerator_cfg = str(experiment_conf["accelerator"])
    use_gpu = accelerator_cfg.startswith("gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu is True else "cpu")
    print("Using device:", device)

    # train on background
    on_train = data_conf["feature_selection"]["on_train"]

    # hack for mixed scaling (on both signal and background data and then drop 1 labels after)
    if on_train == "mixed":
        logging.warning("Will scale on mixed data and drop signal labels after!")
        data_conf["feature_selection"]["on_train"] = None
        experiment_conf["model_postfix"] += "_mixed"
        drop_labels = [1]

    torch.set_float32_matmul_precision("high")
    L.seed_everything(experiment_conf["seed"], workers=True)

    # data processing
    npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])

    f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])

    pre = Preprocessor(**data_conf["preprocessing"])

    chainer = ProcessorChainer(npy_proc, f_sel, pre)

    # dm = HiggsDataModule(
    #     chainer,
    #     train_split=data_conf["train_split"],
    #     val_split=data_conf["val_split"],
    #     **data_conf["dataloader_config"],
    # )

    # dm.setup(stage='test')
    # real = dm.test.X.numpy()

    data, selection, scalers = chainer()   

    idx = selection[selection["type"] != "label"].index

    inv_data = scalers["cont"][0][1].inverse_transform(data[:, idx])

    fig, axs = plt.subplots(6, 3, figsize=(8, 14))
    axs = axs.flatten()

    labels = selection[selection["type"] != "label"]["feature"].values

    for i, (ax, label) in enumerate(zip(axs, labels)):
        ax.hist(data[:, i], bins=50, histtype="step", lw=2, label="scaled", density=True)
        ax.hist(inv_data[:, i], bins=50, histtype="step", lw=2, label="original", density=True)
        ax.set_xlabel(label)
        ax.set_yscale("log")

    axs[0].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()