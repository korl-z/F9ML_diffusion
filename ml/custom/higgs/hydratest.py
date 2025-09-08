import os
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
    EarlyStopping,
)
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from higgs_dataset import HiggsDataModule
from process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)
from ml.common.data_utils.downloadutils import load_dataset_variables
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
from ml.diffusion.EDM.model import EDMPrecond
from ml.common.utils.plot_utils import add_data_mc_ratio


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

def get_generated_data(select_model, N, chunks=20, sample_i=0, ver=-1):
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

@hydra.main(config_path="config/edm/", config_name="main_config", version_base=None)
def main(cfg):
    setup_logger()

    experiment_conf = cfg.experiment_config
    logging.basicConfig(level=logging.INFO)
    data_conf = cfg.data_config
    model_conf = cfg.model_config
    training_conf = cfg.training_config

    accelerator_cfg = str(experiment_conf["accelerator"])
    use_gpu = accelerator_cfg.startswith("gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu is True else "cpu")
    print("Using device:", device)

    torch.set_float32_matmul_precision("high")
    L.seed_everything(experiment_conf["seed"], workers=True)

    # data processing
    npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])

    f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])

    pre = Preprocessor(**data_conf["preprocessing"])

    chainer = ProcessorChainer(npy_proc, f_sel, pre)

    real, selection, scalers = chainer()   

    idx = selection[selection["type"] != "label"].index
    inv_real = scalers["cont"][0][1].inverse_transform(real[:, idx])

    # pick model name and version
    # model_name = "simpleunet_EDM_model"
    # ver = 17
    model_name = "MPtinyunet_EDM_model"
    ver = 15
    N = 1600

    # --- Generate data file ---
    file_dir, file_name = get_generated_data(
        select_model=model_name, N=N, chunks=10, ver=ver
    )

    logging.info(f"Using S_churn = {model_conf['sampler']['S_churn']}")
    logging.info(f"Generated file located at: {os.path.join(file_dir, file_name + '.npy')}")

    # --- Wrap in GeneratedProcessor ---
    gen_proc = GeneratedProcessor(file_dir, file_name, shuffle=True)
    gen_np, _, _ = gen_proc()
    inv_gen = scalers["cont"][0][1].inverse_transform(gen_np[:, idx])

    logging.info(f"Loaded dataset from GeneratedProcessor with shape {gen_np.shape}")

    D = gen_np.shape[1]
    ncols = 6
    nrows = int(np.ceil(D / ncols))

    fig1, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    fig1.subplots_adjust(hspace=0.55, wspace=0.2, left=0.03, right=0.98, top=0.97, bottom=0.1)

    ax_flat = np.array(axs).reshape(-1)

    for feat_idx in range(D):
        ax = ax_flat[feat_idx]
        bin_edges = np.histogram_bin_edges(inv_real[:, feat_idx], bins=50)

        real_counts, _ = np.histogram(inv_real[:, feat_idx], bins=bin_edges) 
        gen_counts, _  = np.histogram(inv_gen[:, feat_idx], bins=bin_edges) 

        sum_real = real_counts.sum()
        sum_gen = gen_counts.sum()

        #division by 0 errors
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
            real_yerr[zero_mask] = 1e-8 #if no MC data


        import seaborn as sns
        sns.histplot(inv_real[:, feat_idx], bins=bin_edges, ax=ax, stat="density", color="gray", alpha=0.3, label="real")

        ax.hist(inv_gen[:, feat_idx], bins=bin_edges, density=True, histtype="step", lw=1.0, color="blue", label="ML")
        ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel(features_list[feat_idx])

        add_data_mc_ratio(
            ax=ax,
            bin_edges=bin_edges,
            data_hist=gen_counts_scaled.astype(float),    
            data_yerr=gen_yerr.astype(float),          
            mc_hists=real_counts_safe[None, :].astype(float),  
            mc_yerrs=real_yerr[None, :].astype(float),         
            ylim=(0.5, 1.5),
            lower_ylabel="ML / MC",
        )

    for j in range(D, len(ax_flat)):
        ax_flat[j].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()