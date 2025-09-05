import os

import logging
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from ml.common.data_utils.syn_datacreator import create_custom_multidim_dataset
from higgs_dataset import HiggsDataset, HiggsDataModule
from process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)

from ml.diffusion.ddpm.model import NoisePredictorUNet
from ml.diffusion.ddpm.diffusers import DiffuserDDPMeps
from ml.diffusion.ddpm.lightning_ddpm import DDPMLightning, EMAUpdateCallback

from ml.common.utils.loggers import timeit, log_num_trainable_params, setup_logger

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
                logging.info(f"Dropped label {drop_label}! New data shape: {data.shape}.")

        return data
    

@timeit(unit="min")
@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logger()
    
    print("Loaded Hydra config:\n", OmegaConf.to_yaml(cfg))

    accelerator_cfg = str(cfg["trainer"]["accelerator"])
    use_gpu = accelerator_cfg.startswith("gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu is True else "cpu")
    print("Using device:", device)

    # Create synthetic dataset
    D = int(cfg["model"]["data_dim"])
    n_samples = int(cfg["train"]["n_samples"])
    print(f"Creating synthetic dataset: n_samples={n_samples}, n_features={D}")
    datasetraw = create_custom_multidim_dataset(
        n_samples=n_samples,
        n_features=D,
        label_random=bool(cfg["train"]["label_random"]),
        signal_frac=float(cfg["train"]["signal_frac"]),
        seed=int(cfg["train"]["seed"]),
    )

    train_val, test = train_test_split(
        datasetraw, train_size=0.95, random_state=int(cfg["train"]["seed"])
    )
    train, val = train_test_split(
        train_val, train_size=0.95, random_state=int(cfg["train"]["seed"])
    )
    print("shapes: train", train.shape, "val", val.shape, "test", test.shape)
    print("train", train)

    train_ds = HiggsDataset(train)
    val_ds = HiggsDataset(val)
    test_ds = HiggsDataset(test)

    batch_size = int(cfg["train"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"])
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"])
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"])
    )

    time_steps = int(cfg["model"]["timesteps"])
    diffuser2 = DiffuserDDPMeps(
        timesteps=time_steps,
        scheduler=cfg["model"]["scheduler"],
        device=device
    )

    unet_kwargs = {
        "data_dim": int(cfg["model"]["data_dim"]),
        "base_dim": int(cfg["model"]["base_dim"]),
        "depth": int(cfg["model"]["depth"]),
        "time_emb_dim": int(cfg["model"]["time_emb_dim"]),
        "timesteps": time_steps
    }

    modelUnet = NoisePredictorUNet(**unet_kwargs)

    ddpm_pl_module = DDPMLightning(
        model=modelUnet,
        diffuser=diffuser2,
        raw_data=datasetraw,
        training_conf=cfg.train,
        model_conf=cfg.model,
        test_sample_count=int(cfg["train"]["test_sample_count"]),
    )

    checkpoint_dir = to_absolute_path(str(cfg["trainer"]["checkpoint_path"]))
    save_name = str(cfg["trainer"]["save_name"])
    ckpt_dir_full = os.path.join(checkpoint_dir, save_name)
    os.makedirs(ckpt_dir_full, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir_full,
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=10,
        mode="min",
        verbose=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    ema_cb = EMAUpdateCallback()
    tqdm_cb = (
        TQDMProgressBar(refresh_rate=int(cfg["trainer"]["progress_refresh_rate"]))
        if bool(cfg["trainer"]["enable_progress_bar"])
        else None
    )

    callbacks = [ckpt_cb, lr_cb, ema_cb]
    if tqdm_cb is not None:
        callbacks.append(tqdm_cb)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg["trainer"]["experiment_name"],
        run_name=save_name,
    )
    logger = mlf_logger

    trainer = L.Trainer(
        default_root_dir=ckpt_dir_full,
        accelerator=str(cfg["trainer"]["accelerator"]),
        devices=int(cfg["trainer"]["devices"]),
        max_epochs=int(cfg["train"]["max_epochs"]),
        callbacks=callbacks,
        logger=logger,
        precision=int(cfg["trainer"]["precision"]),
        enable_progress_bar=bool(cfg["trainer"]["enable_progress_bar"]),
    )

    print("Starting training.")
    trainer.fit(ddpm_pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished. Running test (sampling + hist plots)...")
    # trainer.test(ddpm_pl_module, dataloaders=test_loader)
    # ddpm_pl_module.on_train_end()
    print("Done. Checkpoints and artifacts are in:", ckpt_dir_full)


if __name__ == "__main__":
    main()
