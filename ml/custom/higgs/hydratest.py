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
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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

    logging.info(f"Using device: {experiment_conf["accelerator"]}")
    logging.info(f"Using S_churn = {model_conf['sampler']['S_churn']}")

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
    # model_name = "unet1d_VE_model"
    # ver = 1

    select_models = [
        # "unet1d_ddpm_model",

        # "unet1d_VP_model",
        # "MPtinyunet_VE_model",
        # "unet1d_EDM2_model",
        # "unet1d_EDMnoEMA_model",
        # "unet1d_ddpm_model",

        # "unet1d_EDMraw_model",
        # "unet1d_EDMraw_model",
        
        # "unet1dconv_EDMsimple_model",
        # "unet1dconv_VP_model",

        "unet1D_EDM_s_model",
    ]
    versions = [
                # 2, 
                # 3, 3, 1, 1, 6, #v1 EDM noEMA
                # 1, 2, #no. 2 is EDMsimple
                # 1, 1, #1dconv models
                2
                ]
    
    N = 1000000
    all_generated_data = {} # Dictionary to store results
    model_info = {}

    for model_name, ver in zip(select_models, versions):
        print("\n" + "="*50)
        print(f"Generating data for model: {model_name}, version: {ver}")
        
        file_dir, file_name = get_generated_data(
            select_model=model_name, N=N, chunks=100, ver=ver
        )
        gen_proc = GeneratedProcessor(file_dir, file_name, shuffle=True)
        gen_np, _, _ = gen_proc()
        inv_gen = scalers["cont"][0][1].inverse_transform(gen_np[:, idx])

        # all_generated_data[f"{model_name}_v{ver}"] = inv_gen
        all_generated_data[f"{model_name}_v{ver}"] = gen_np[:, idx]

    logging.info(f"Loaded dataset from GeneratedProcessor with shape {gen_np.shape}")

    D = gen_np.shape[1]
    ncols = 6
    nrows = int(np.ceil(D / ncols))

    fig1, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols + 3, 4 * nrows))
    fig1.subplots_adjust(hspace=0.55, wspace=0.2, left=0.03, right=0.98, top=0.97, bottom=0.1)
    
    ax_flat = np.array(axs).reshape(-1)
    
    model_colors = cmaps.vivid(np.linspace(0, 1, len(all_generated_data)))
    color_cycle = itertools.cycle(model_colors)


    for feat_idx in range(D):
        ax = ax_flat[feat_idx]

        x = real[:, feat_idx]
        # y = inv_gen[:, feat_idx]

        bin_edges = np.histogram_bin_edges(x, bins=50)

        real_counts, _ = np.histogram(x, bins=bin_edges) 
        # gen_counts, _  = np.histogram(y, bins=bin_edges) 

        sum_real = real_counts.sum()
        # sum_gen = gen_counts.sum()

        #division by 0 errors
        # if sum_gen == 0:
        #     # gen_counts_scaled = np.full_like(gen_counts, 1e-8, dtype=float)
        # else:
        #     scale_factor = float(sum_real) / float(sum_gen)
        #     # gen_counts_scaled = gen_counts.astype(float) * scale_factor

        real_yerr = np.sqrt(real_counts.astype(float))
        # gen_yerr  = np.sqrt(gen_counts.astype(float)) * (scale_factor if sum_gen != 0 else 1.0)

        real_counts_safe = real_counts.astype(float).copy()
        zero_mask = real_counts_safe == 0
        if np.any(zero_mask):
            real_counts_safe[zero_mask] = 1e-8
            real_yerr[zero_mask] = 1e-8 #if no MC data

        bin_width = bin_edges[1] - bin_edges[0]

        #MC generated plot
        sns.histplot(x, bins=bin_edges, ax=ax, stat="density", color="gray", alpha=0.3, label="MC")

        for model_label, gen_data in all_generated_data.items():
            y_gen = gen_data[:, feat_idx]
            color = next(color_cycle)

            gen_counts, _ = np.histogram(y_gen, bins=bin_edges)
            
            # Prevent division by zero if a model generates no data
            if gen_counts.sum() == 0:
                continue

            # Calculate density and the sqrt(N) error on the density
            gen_density = gen_counts / (gen_counts.sum() * bin_width)
            gen_err = np.sqrt(gen_counts) / (gen_counts.sum() * bin_width)

            # Plot the main line as a step plot
            ax.step(bin_edges[:-1], gen_density, where='post', color=color, lw=1.2, label=model_label)
            
            # Plot the error band using fill_between
            ax.fill_between(bin_edges[:-1], gen_density - gen_err, gen_density + gen_err, step='post', color=color, alpha=0.4)

        ax.set_yscale("log")
        ax.set_xlabel(features_list[feat_idx])
        color_cycle = itertools.cycle(model_colors)

    #     # X = np.linspace(-6, 6, 100)    
    #     # ax.plot(X, scipy.stats.norm.pdf(X, 0, 1), '-k')
    #     add_data_mc_ratio(
    #         ax=ax,
    #         bin_edges=bin_edges,
    #         data_hist=gen_counts_scaled.astype(float),    
    #         data_yerr=gen_yerr.astype(float),          
    #         mc_hists=real_counts_safe[None, :].astype(float),  
    #         mc_yerrs=real_yerr[None, :].astype(float),         
    #         ylim=(0.5, 1.5),
    #         lower_ylabel="ML / MC",
    #     )

    handles, labels = ax_flat[0].get_legend_handles_labels()
    # Place the legend on the right side of the figure
    fig1.legend(handles, labels, loc='center right')
    
    # Adjust the subplot layout to make space for the legend
    # fig1.subplots_adjust(right=0.85)

    # Save the figure
    output_filename = "analysis/plots/testplots/combined_comparison_plot_raw.pdf"
    plt.savefig(output_filename)
    logging.info(f"Combined plot saved successfully as {output_filename}")

# @hydra.main(config_path="config/edm/", config_name="main_config", version_base=None)
# def main(cfg):
#     setup_logger()

#     experiment_conf = cfg.experiment_config
#     logging.basicConfig(level=logging.INFO)
#     data_conf = cfg.data_config
#     model_conf = cfg.model_config
#     training_conf = cfg.training_config

#     logging.info(f"Using device: {experiment_conf['accelerator']}")
#     torch.set_float32_matmul_precision("high")
#     L.seed_everything(experiment_conf["seed"], workers=True)

#     # --- 1. Load and Prepare Real Data ---
#     npy_proc = HIGGSNpyProcessor(**data_conf["input_processing"])
#     f_sel = HIGGSFeatureSelector(npy_proc.npy_file, **data_conf["feature_selection"])
#     pre = Preprocessor(**data_conf["preprocessing"])
#     chainer = ProcessorChainer(npy_proc, f_sel, pre)
#     real, selection, scalers = chainer()   
#     idx = selection[selection["type"] != "label"].index
#     inv_real = scalers["cont"][0][1].inverse_transform(real[:, idx])

#     # --- 2. Define Models and Generate Data for Each ---
#     select_models = [
#         "unet1d_ddpm_model",

#         "unet1d_VP_model",
#         "MPtinyunet_VE_model",
#         "unet1d_EDM2_model",
#         "unet1d_EDM_model",
#         "unet1d_ddpm_model",

#         "unet1d_EDMraw_model",
#         "unet1d_EDMraw_model",
        
#         "unet1dconv_EDMsimple_model",
#         "unet1dconv_VP_model",
#     ]
#     versions = [
#                 2, 
#                 3, 3, 1, 10, 6,
#                 1, 2, #no. 2 is EDMsimple
#                 1, 1 #1dconv models
#                 ]
    
#     N = 10000  # Number of samples to generate per model
#     all_generated_data = {}

#     for model_name, ver in zip(select_models, versions):
#         print("\n" + "="*50)
#         logging.info(f"Generating data for model: {model_name}, version: {ver}")
        
#         file_dir, file_name = get_generated_data(
#             select_model=model_name, N=N, chunks=100, ver=ver
#         )
#         gen_proc = GeneratedProcessor(file_dir, file_name, shuffle=True)
#         gen_np, _, _ = gen_proc()
#         inv_gen = scalers["cont"][0][1].inverse_transform(gen_np[:, idx])
        
#         label = f"{model_name}_v{ver}"
#         all_generated_data[label] = inv_gen

#     logging.info(f"Successfully generated data for {len(all_generated_data)} models.")

#     # --- 3. Setup for Plotting ---
#     D = inv_real.shape[1]
#     ncols = 6
#     nrows = int(np.ceil(D / ncols))

#     fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols + 3, 4 * nrows))
#     ax_flat = np.array(axs).reshape(-1)

#     # Use the 'vibrant' colormap from the colormaps library

#     model_colors = cmaps.vivid(np.linspace(0, 1, len(all_generated_data)))
#     color_cycle = itertools.cycle(model_colors)

#     # --- 4. Main Plotting Loop ---
#     for feat_idx in range(D):
#         ax = ax_flat[feat_idx]
#         x_real = inv_real[:, feat_idx]
        
#         bin_edges = np.histogram_bin_edges(x_real, bins=50)
#         bin_width = bin_edges[1] - bin_edges[0]

#         # Plot the "real" data as a gray filled histogram
#         real_counts, _ = np.histogram(x_real, bins=bin_edges)
#         real_density = real_counts / (real_counts.sum() * bin_width)
#         ax.fill_between(bin_edges[:-1], real_density, step='post', color='gray', alpha=0.5, label='real')

#         # Loop through each generated model and plot it
#         for model_label, gen_data in all_generated_data.items():
#             y_gen = gen_data[:, feat_idx]
#             color = next(color_cycle)

#             gen_counts, _ = np.histogram(y_gen, bins=bin_edges)
            
#             # Prevent division by zero if a model generates no data
#             if gen_counts.sum() == 0:
#                 continue

#             # Calculate density and the sqrt(N) error on the density
#             gen_density = gen_counts / (gen_counts.sum() * bin_width)
#             gen_err = np.sqrt(gen_counts) / (gen_counts.sum() * bin_width)

#             # Plot the main line as a step plot
#             ax.step(bin_edges[:-1], gen_density, where='post', color=color, lw=1.2, label=model_label)
            
#             # Plot the error band using fill_between
#             ax.fill_between(bin_edges[:-1], gen_density - gen_err, gen_density + gen_err, step='post', color=color, alpha=0.4)

#         ax.set_yscale("log")
#         ax.set_xlabel(features_list[feat_idx])
#         # A legend is not added here to avoid clutter

#         # Reset color cycle for the next subplot
#         color_cycle = itertools.cycle(model_colors)

#     # --- 5. Finalize and Save Plot ---
    
#     # Deactivate any unused subplots
#     for j in range(D, len(ax_flat)):
#         ax_flat[j].axis("off")
        
#     # Create a single, unified legend for the entire figure
#     handles, labels = ax_flat[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='center right', fontsize=12, bbox_to_anchor=(0.98, 0.5))
    
#     # Adjust subplot layout to make space for the legend
#     fig.subplots_adjust(right=0.85, hspace=0.3, wspace=0.3)
    
#     output_filename = "analysis/plots/final_comparison_plot.pdf"
#     os.makedirs(os.path.dirname(output_filename), exist_ok=True)
#     plt.savefig(output_filename)
#     logging.info(f"Combined plot saved successfully as {output_filename}")
#     plt.close(fig)


if __name__ == "__main__":
    main()
