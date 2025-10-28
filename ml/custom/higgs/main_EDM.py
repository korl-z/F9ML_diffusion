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

# common imports
from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.common.utils.register_model import register_from_checkpoint
from higgs_dataset import HiggsDataModule
from process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)
from ml.common.utils.loggers import timeit, log_num_trainable_params, setup_logger
from ml.common.utils.utils import PFEMACallback
from ml.common.nn.unet import MPTinyUNet, UNet1D, UNet1Dconv

# custom imports
from ml.diffusion.EDM.lightning_EDM import EDMModule, VPModule, VEModule, EDM2Module
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
@hydra.main(version_base=None, config_path="config/edm/", config_name="main_config")
def main(cfg: DictConfig) -> None:
    setup_logger()

    experiment_conf = cfg.experiment_config
    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    experiment_name = "EDM"

    data_conf = cfg.data_config
    model_conf = cfg.model_config
    training_conf = cfg.training_config

    accelerator_cfg = str(experiment_conf["accelerator"])
    use_gpu = accelerator_cfg.startswith("gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu is True else "cpu")
    logging.info(f"Using device: {device}")

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

    dm = HiggsDataModule(
        chainer,
        train_split=data_conf["train_split"],
        val_split=data_conf["val_split"],
        **data_conf["dataloader_config"],
    )

    logging.info(f"Setting up {model_conf['model_name']} model.")

    tracker = DDPMTracker(
        experiment_conf, tracker_path="/data0/korlz/f9-ml/ml/custom/higgs/metrics/"
    )

    model = UNet1D(
        data_dim=data_conf["input_dim"],
        **model_conf["network"],
    )

    # model = MPTinyUNet(
    #     **model_conf["network"],
    # )

    # model = UNet1Dconv(
    #     data_dim=data_conf["input_dim"],
    #     **model_conf["network"],
    # )

    logging.info("Done model setup.")
    log_num_trainable_params(model, unit="M")

    # callbacks
    ckpt_cb = ModelCheckpoint(
        save_weights_only=True,
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        verbose=True,
    )

    lr_cb = LearningRateMonitor(logging_interval="step")

    ema_cb = PFEMACallback(
        std=training_conf["std"],
        batchsize=data_conf["dataloader_config"]["batch_size"],
    )

    tqdm_cb = (
        TQDMProgressBar(refresh_rate=int(experiment_conf["progress_refresh_rate"]))
        if bool(experiment_conf["enable_progress_bar"])
        else None
    )

    es_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=training_conf["early_stop_patience"],
        verbose=True,
    )

    callbacks = [ckpt_cb, lr_cb, 
                #  ema_cb, 
                 tqdm_cb, es_cb, tracker]

    # initialize mlflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,  # ni vec v yaml
        run_name=f'{model_conf["model_name"]}_{experiment_conf["run_name"]}',
        save_dir=experiment_conf["save_dir"],
        log_model=True,
    )

    trainer = L.Trainer(
        max_epochs=int(training_conf["max_epochs"]),
        accelerator=experiment_conf["accelerator"],
        devices=experiment_conf["devices"],
        check_val_every_n_epoch=experiment_conf["check_eval_n_epoch"],
        log_every_n_steps=experiment_conf["log_every_n_steps"],
        num_sanity_val_steps=experiment_conf["num_sanity_val_steps"],
        precision=experiment_conf["precision"],
        logger=mlf_logger,
        callbacks=callbacks,
        enable_progress_bar=bool(experiment_conf["enable_progress_bar"]),
    )

    module_name = "EDM"

    if module_name == "EDM":
        EDM_L_module = EDMModule(
            datamodule=dm,
            model_conf=model_conf,
            training_conf=training_conf,
            data_conf=data_conf,
            model=model,
            tracker=tracker,
        )

        model_name = f"{model_conf['model_name']}_model"

        print("Starting training.")
        trainer.fit(EDM_L_module, dm)

        register_from_checkpoint(trainer, EDM_L_module, model_name=model_name)

    elif module_name == "VP":
        VP_L_module = VPModule(  # turn on subVP by changing subVP = True in VP_conifg.
            datamodule=dm,
            model_conf=model_conf,
            training_conf=training_conf,
            data_conf=data_conf,
            model=model,
            tracker=tracker,
        )

        model_name = f"{model_conf['model_name']}_model"

        print("Starting training.")
        trainer.fit(VP_L_module, dm)

        register_from_checkpoint(trainer, VP_L_module, model_name=model_name)

    elif module_name == "VE":
        VE_L_module = VEModule(
            datamodule=dm,
            model_conf=model_conf,
            training_conf=training_conf,
            data_conf=data_conf,
            model=model,
            tracker=tracker,
        )

        model_name = f"{model_conf['model_name']}_model"

        print("Starting training.")
        trainer.fit(VE_L_module, dm)

        register_from_checkpoint(trainer, VE_L_module, model_name=model_name)

    elif module_name == "EDM2":
        EDM2_L_module = EDM2Module(
            datamodule=dm,
            model_conf=model_conf,
            training_conf=training_conf,
            data_conf=data_conf,
            model=model, 
            tracker=tracker,
        )

        model_name = f"{model_conf['model_name']}_model"

        print("Starting training.")
        trainer.fit(EDM2_L_module, dm)

        register_from_checkpoint(trainer, EDM2_L_module, model_name=model_name)

if __name__ == "__main__":
    main()
