import os
import time
import sys
import logging

sys.path.insert(1, "C:/Users/Uporabnik/Documents/IJS-F9/korlz")

import mlflow
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch 

import hydra
import lightning as L

# lightning imports
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

# import common classes
from ml.classifiers.models.binary_model import BinaryClassifier
from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.common.stats.c2st import GeneratedProcessor, TwoSampleBuilder
from ml.common.utils.loggers import setup_logger
from ml.common.utils.register_model import register_from_checkpoint
from ml.custom.higgs.analysis.utils import get_model, sample_from_models

# custom imports
from ml.custom.higgs.higgs_dataset import HiggsDataModule
from ml.custom.higgs.process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)

def get_generated_data(select_model, N, chunks=100, sample_i=0, ver=-1):
    model_dct = {select_model: get_model(select_model, ver=ver).eval()}

    if ver == -1:
        ver = ""

    model_dct[f"{select_model}{ver}"] = model_dct.pop(select_model)
    select_model = f"{select_model}{ver}"

    _, npy_file_names = sample_from_models(model_dct, N, ver=ver, chunks=chunks, resample=1, return_npy_files=True)

    file_path = npy_file_names[select_model][sample_i]
    head, tail = os.path.split(file_path)

    tail = tail.split(".")[0]

    return head, tail


@hydra.main(config_path="config/dnn/", config_name="main_config", version_base=None)
def main(cfg):
    setup_logger()
    logging.basicConfig(level=logging.INFO)

    experiment_conf = cfg.experiment_config
    data_conf = cfg.data_config
    model_conf = cfg.model_config
    training_conf = cfg.training_config

    if experiment_conf["run_name"] is None:
        experiment_conf["run_name"] = time.asctime(time.localtime())

    experiment_name = "c2st_all"

    select_models = [
        "unet_noise_ddpm_model",
        "unet_noise_imp_ddpm_model",
        "simpleunet_EDM_model",
        "simpleunet_EDM_model",
        "tinyunet_EDM_model",
        "MPtinyunet_EDM_model",
    ]
    versions = [18, 18, 11, 13, 3, 3]

    model_name = select_models[-1]
    ver = versions[-1]

    N = 32000

    data_conf["feature_selection"]["n_data"] = 2 * N

    # change hold modes
    data_conf["input_processing"]["hold_mode"] = True
    data_conf["input_processing"]["use_hold"] = True

    # hijack background data for training in c2st
    data_conf["feature_selection"]["on_train"] = "bkg"

    # match model postfix to rescale type
    experiment_conf["model_postfix"] = f"{data_conf['preprocessing']['cont_rescale_type']}"

    # set early stop patience and epochs
    # training_conf["early_stop_patience"] = 4
    # training_conf["epochs"] = 7

    # small batch size
    data_conf["dataloader_config"]["batch_size"] = 512

    # matmul precision and seed
    L.seed_everything(experiment_conf["seed"], workers=True)

    # data processing
    npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])

    f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])

    pre = Preprocessor(**data_conf["preprocessing"])

    chainer = ProcessorChainer(npy_proc, f_sel, pre)

    # load model
    file_dir, file_name = get_generated_data(
        select_model=model_name, N=N, chunks=100, ver=ver
    )

    logging.info(f"Generated file located at: {os.path.join(file_dir, file_name + '.npy')}")

    # --- Wrap in GeneratedProcessor ---
    gen_proc = GeneratedProcessor(file_dir, file_name, shuffle=True)
    gen_np, _, _ = gen_proc()

    logging.info(f"Loaded dataset from GeneratedProcessor with shape {gen_np.shape}")

    # file_dir, file_name = get_generated_data(select_model=select_model, N=N, chunks=20, ver=ver, timesteps=1000, scheduler="cosine")
    # gen_proc = GeneratedProcessor(file_dir, file_name, shuffle=True)


    two_samples_proc = TwoSampleBuilder(
        processor_X=chainer,
        processor_Y=gen_proc,
        add_label_X=False,
        add_label_Y=True,
        hold_out_ratio=0.99,
        shuffle_random_state=0,
    )

    # create MC+gen two sample data module
    dm = HiggsDataModule(two_samples_proc, **data_conf["dataloader_config"])
   
    
    # dm.setup(stage='fit') 
    # print(next(iter(dm.train_dataloader())))
    # make model
    classifier = BinaryClassifier(model_conf, training_conf, tracker=None)

    # define callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=(
                training_conf["max_epochs"]
                if training_conf["early_stop_patience"] is None
                else training_conf["early_stop_patience"]
            ),
        ),
        ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
    ]

    # initialize mlflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f'{model_conf["model_name"]}_{experiment_conf["run_name"]}',
        save_dir=experiment_conf["save_dir"],
        log_model=True,
    )
    # define trainer
    trainer = L.Trainer(
        max_epochs=training_conf["max_epochs"],
        accelerator=experiment_conf["accelerator"],
        devices=experiment_conf["devices"],
        check_val_every_n_epoch=experiment_conf["check_eval_n_epoch"],
        log_every_n_steps=experiment_conf["log_every_n_steps"],
        num_sanity_val_steps=experiment_conf["num_sanity_val_steps"],
        logger=mlf_logger,
        callbacks=callbacks,
    )

    from ml.common.utils.register_model import list_registered_objects
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    models = list_registered_objects()
    print("Registered model names:", list(models.keys()))
    if "unet_noise_ddpm_model" in models:
        print("versions:", sorted(list(models["unet_noise_ddpm_model"].keys())))


    model_name = f"{model_conf['model_name']}_{model_name}_c2st_gen_model"

    model_name += "_all"

    print("Classifier training.")
    # run training
    trainer.fit(classifier, dm)

    # save model
    register_from_checkpoint(trainer, classifier, model_name=model_name)

    device = 'cpu'

    dm.setup(stage='test')
    loader = dm.test_dataloader()
    ys_list = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            # typical TwoSample / Higgs dataloader returns (X, y) or (X, y, ...)
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                # if dataloader returns tensors directly
                x = batch
                # no labels available
                raise RuntimeError("Expected (X, y) batches from datamodule.test_dataloader()")

            x = x.to(device)
            y = y.to(device)

            # get raw model scores (use underlying .model if this is a Lightning wrapper)
            if hasattr(classifier, "model") and callable(getattr(classifier, "model")):
                scores = classifier.model(x)
            else:
                # fallback to calling classifier directly (may call forward)
                scores = classifier(x)

            # Flatten arrays (binary classifier returns [B,1] or [B])
            scores = scores.detach().cpu().numpy().ravel()
            y_np = y.detach().cpu().numpy().ravel()

            preds_list.append(scores)
            ys_list.append(y_np)

    y_all = np.concatenate(ys_list, axis=0)
    preds_all = np.concatenate(preds_list, axis=0)

    # compute ROC
    fpr, tpr, thresholds = roc_curve(y_all, preds_all)
    roc_auc = auc(fpr, tpr)

    # plot
    fig, ax = plt.subplots(figsize=(3, 2.8))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="k", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # run_id = getattr(mlf_logger, "run_id", None)
    # if run_id is not None:
    #     with mlflow.start_run(run_id=run_id):
    #         mlflow.log_figure(fig, f"roc_{model_name}.png")
    # else:
    #     mlflow.log_figure(fig, f"roc_{model_name}.png")
    #     print("Logged figures via mlflow")

    try:
        plt.close("all")
    except Exception:
        pass


if __name__ == "__main__":
    main()

