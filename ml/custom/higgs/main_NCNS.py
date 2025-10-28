import os
import time
import logging
import torch

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

from ml.diffusion.score.model import RefineNet, SimpleUNet, ModularUNet
from ml.diffusion.score.lightning_score import NCNSModule

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


@timeit(unit="min")
@hydra.main(version_base=None, config_path="config/ncns", config_name="main_config")
def main(cfg: DictConfig) -> None:
    setup_logger()

    experiment_conf = cfg.experiment_config
    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    experiment_name = "score_inital_test"

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

    if on_train == "mixed":
        drop_proc = DropLabelProcessor(drop_labels)
        chainer = ProcessorChainer(npy_proc, f_sel, pre, drop_proc)
    else:
        chainer = ProcessorChainer(npy_proc, f_sel, pre)

    dm = HiggsDataModule(
        chainer,
        train_split=data_conf["train_split"],
        val_split=data_conf["val_split"],
        **data_conf["dataloader_config"],
    )

    logging.info(f"Setting up {model_conf['model_name']} model.")


    unet_kwargs = {
        "in_channels": int(model_conf["in_channels"]),
        "base_channels": int(model_conf["base_channels"]),
        "num_res_blocks": int(model_conf["num_res_blocks"]),
        "channel_mults": model_conf["channel_mults"],
        "use_attention_at_level": model_conf["use_attention_at_level"],
    }

    tracker = DDPMTracker(experiment_conf, tracker_path="ml/custom/higgs/metrics")
    model = ModularUNet(**unet_kwargs)

    logging.info("Done model setup.")
    log_num_trainable_params(model, unit="k")

    # callbacks
    ckpt_cb = ModelCheckpoint(
        save_weights_only=True,
        # dirpath=ckpt_dir_full,
        # filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        verbose=True,
    )
    
    lr_cb = LearningRateMonitor(logging_interval="step")
    tqdm_cb = (
        TQDMProgressBar(refresh_rate=int(experiment_conf["progress_refresh_rate"]))
        if bool(experiment_conf["enable_progress_bar"])
        else None
    )

    # ne dela for some reason
    # es_cb = EarlyStopping(
    #         monitor="val_loss",
    #         mode="min",
    #         patience=(
    #             experiment_conf["num_epochs"]
    #             if training_conf["early_stop_patience"] is None
    #             else training_conf["early_stop_patience"]
    #         ),
    #     ),

    callbacks = [ckpt_cb, lr_cb, tqdm_cb]

    # initialize mlflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,  # ni vec v yaml
        run_name=f'{model_conf["model_name"]}_{experiment_conf["run_name"]}',
        save_dir=experiment_conf["save_dir"],
        log_model=True,
    )

    trainer = L.Trainer(
        max_epochs=int(training_conf["max_epochs"]),
        accelerator=str(experiment_conf["accelerator"]),
        devices=int(experiment_conf["devices"]),
        check_val_every_n_epoch=experiment_conf["check_eval_n_epoch"],
        log_every_n_steps=experiment_conf["log_every_n_steps"],
        num_sanity_val_steps=experiment_conf["num_sanity_val_steps"],
        precision=int(experiment_conf["precision"]),
        logger=mlf_logger,
        callbacks=callbacks,
        enable_progress_bar=bool(experiment_conf["enable_progress_bar"]),
        # default_root_dir=ckpt_dir_full,
    )

    NCNS_L_module = NCNSModule(
        datamodule=dm,
        model_conf=model_conf,
        data_conf=data_conf,
        training_conf=training_conf,
        model=model,
        test_sample_count=training_conf["test_sample_count"], 
        tracker=tracker,
    )


    print("Trainer callbacks:", trainer.callbacks)
    print("Has ModelCheckpoint?", any(isinstance(cb, ModelCheckpoint) for cb in trainer.callbacks))

    print("logger:", trainer.logger)
    print("logger._checkpoint_callback exists?", getattr(trainer.logger, "_checkpoint_callback", None))

    model_name = f"{model_conf['model_name']}_model"

    print("Starting training.")
    trainer.fit(NCNS_L_module, dm)
    print("Training finished. Running test (sampling + hist plots)...")

    register_from_checkpoint(trainer, NCNS_L_module, model_name=model_name)

if __name__ == "__main__":
    main()
