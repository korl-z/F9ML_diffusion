import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import colormaps as cmaps
import lightning as L
import seaborn as sns
import scipy

import hydra

from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)

# import common classes
from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.common.stats.c2st import GeneratedProcessor
from ml.common.utils.loggers import setup_logger
from ml.custom.higgs.analysis.utils import get_model, sample_from_models

# custom imports
from ml.custom.higgs.process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)
from ml.common.utils.plot_utils import add_data_mc_ratio

# plt.rcParams.update(
#     {"text.usetex": True, "font.family": "Libertinus Serif", "font.size": 10}
# )

features_list = [
    r"lepton $p_T$",
    r"lepton $\eta$",
    r"missing energy",
    r"jet1 $p_T$",
    r"jet1 $\eta$",
    r"jet2 $p_T$",
    r"jet2 $\eta$",
    r"jet3 $p_T$",
    r"jet3 $\eta$",
    r"jet4 $p_T$",
    r"jet4 $\eta$",
    r"$m_{jj}$",
    r"$m_{jjj}$",
    r"$m_{lv}$",
    r"$m_{jlv}$",
    r"$m_{bb}$",
    r"$m_{wbb}$",
    r"$m_{wwbb}$",
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

    local_generated_data_path = r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ml\data\HIGGS\HIGGS_generated_unet1d_ddpm_model6_6_1mil.npy"
    # model_name = "unet1d_ddpm_model"
    # ver = 6
    # N = 1000000

    # # --- Generate data file ---
    # file_dir, file_name = get_generated_data(
        # select_model=model_name, N=N, chunks=10, ver=ver
    # )

    # logging.info(f"Using S_churn = {model_conf['sampler']['S_churn']}")
    # logging.info(f"Generated file located at: {os.path.join(file_dir, file_name + '.npy')}")

    # # --- Wrap in GeneratedProcessor ---
    # gen_proc = GeneratedProcessor(file_dir, file_name, shuffle=True)
    # gen_np, _, _ = gen_proc()

    logging.info(f"Loading generated data directly from: {local_generated_data_path}")
    gen_np = np.load(local_generated_data_path)
    logging.info(f"Loaded dataset with shape {gen_np.shape}")

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

        # x = real[:, feat_idx]
        # y = gen_np[:, feat_idx]

        x = inv_real[:, feat_idx]
        y = inv_gen[:, feat_idx]
        bin_edges = np.histogram_bin_edges(x, bins=50)

        real_counts, _ = np.histogram(x, bins=bin_edges) 
        gen_counts, _  = np.histogram(y, bins=bin_edges) 

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
        sns.histplot(x, bins=bin_edges, ax=ax, stat="density", color="gray", alpha=0.3, label="MC")

        ax.hist(y, bins=bin_edges, density=True, histtype="step", lw=1.0, color="blue", label="ML")
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



@hydra.main(config_path="config/edm/", config_name="main_config", version_base=None)
def main2(cfg):
    setup_logger()

    experiment_conf = cfg.experiment_config
    logging.basicConfig(level=logging.INFO)
    data_conf = cfg.data_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.set_float32_matmul_precision("high")
    L.seed_everything(experiment_conf["seed"], workers=True)

    # This part is still necessary to get the scalers for the inverse transform
    logging.info("Processing real data to get scalers...")
    npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])
    f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])
    pre = Preprocessor(**data_conf["preprocessing"])
    chainer = ProcessorChainer(npy_proc, f_sel, pre)
    real, selection, scalers = chainer()

    idx = selection[selection["type"] != "label"].index
    inv_real = scalers["cont"][0][1].inverse_transform(real[:, idx])
    logging.info("Finished processing real data.")

    # --- [MODIFICATION 1] ---
    # Define a dictionary of models to plot.
    # Key: The name that will appear in the legend.
    # Value: The full path to the .npy file.
    generated_files_to_plot = {
        "UNet DDPM v6": r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ml\data\HIGGS\HIGGS_generated_unet1d_ddpm_model6_6_1mil.npy",
        # "EMA Model v17": r"C:\path\to\your\second_model_file.npy",
        # "Another Model": r"C:\path\to\your\third_model_file.npy",
    }
    
    # --- [MODIFICATION 2] ---
    # Load all generated data files and apply the inverse transform
    inverse_generated_data = {}
    for label, path in generated_files_to_plot.items():
        if os.path.exists(path):
            logging.info(f"Loading '{label}' from: {path}")
            gen_np = np.load(path)
            inverse_generated_data[label] = scalers["cont"][0][1].inverse_transform(gen_np[:, idx])
        else:
            logging.warning(f"File not found, skipping: {path}")

    # --- Plotting logic ---
    D = inv_real.shape[1]
    ncols = 6
    nrows = int(np.ceil(D / ncols))

    # The figure setup is simpler as we don't need to make room for a ratio plot
    fig1, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows)) # Reduced height slightly
    ax_flat = np.array(axs).reshape(-1)

    # Define a color cycle for the different models
    colors = plt.cm.get_cmap('tab10').colors
    color_cycle = itertools.cycle(colors)

    for feat_idx in range(D):
        ax = ax_flat[feat_idx]
        x = inv_real[:, feat_idx]
        
        # Use a common set of bins for all histograms on this axis
        bin_edges = np.histogram_bin_edges(x, bins=50)

        # Plot the real data
        sns.histplot(x, bins=bin_edges, ax=ax, stat="density", color="gray", alpha=0.3, label="Real Data")

        # --- [MODIFICATION 3] ---
        # Loop through and plot each loaded generated dataset
        for label, inv_gen_data in inverse_generated_data.items():
            y = inv_gen_data[:, feat_idx]
            ax.hist(y, bins=bin_edges, density=True, histtype="step", lw=1.5, color=next(color_cycle), label=label)
        
        ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel(features_list[feat_idx])
        ax.set_ylabel("Density")

        # --- [MODIFICATION 4] ---
        # The code for calculating counts, errors, and the add_data_mc_ratio call has been removed.

    # Turn off any unused subplots
    for j in range(D, len(ax_flat)):
        ax_flat[j].axis("off")
        
    plt.tight_layout(pad=1.0)
    plt.show()

if __name__ == "__main__":
    main()