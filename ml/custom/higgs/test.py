# In a new plotting script, e.g., plot_magnitudes.py
import torch
import lightning as L
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hydra
from omegaconf import DictConfig

from ml.custom.higgs.analysis.utils import get_model

@hydra.main(config_path="config/edm/", config_name="main_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # --- Configuration ---
    model_name = "tinyunet_EDM_model"
    model_version = 8
    
    # Get the MLflow run ID associated with the registered model
    try:
        run_id = mlflow.tracking.MlflowClient().get_model_version(
            name=model_name, version=model_version
        ).run_id
        print(f"Found MLflow Run ID for {model_name} v{model_version}: {run_id}")
    except Exception as e:
        print(f"Could not find MLflow run for {model_name} v{model_version}. Please ensure it is registered.")
        print(f"Error: {e}")
        return

    # --- Load the Model (using your functions) ---
    print("\nLoading model from MLflow registry...")

    model = get_model(model_name, ver=model_version)
    print("Model loaded successfully.")
    print(model)

    # --- Load Metric History from MLflow ---
    print("\nFetching metric history from MLflow...")
    client = mlflow.tracking.MlflowClient()
    all_metrics = client.get_run(run_id).data.metrics

    weight_history = {}
    act_history = {}

    for key in all_metrics.keys():
        if key.startswith('weight_norm/') or key.startswith('act_norm/'):
            history = client.get_metric_history(run_id, key)
            series = pd.Series({item.step: item.value for item in history}, name=key)
            
            if key.startswith('weight_norm/'):
                weight_history[key] = series
            else:
                act_history[key] = series

    if not weight_history or not act_history:
        print("\nNo weight or activation metrics found in the specified run.")
        print("Please ensure the MagnitudeMonitor callback was used during training.")
        return

    df_weights = pd.DataFrame(weight_history).sort_index()
    df_activations = pd.DataFrame(act_history).sort_index()
    print("Metric history fetched successfully.")

    # --- Plotting ---
    print("\nGenerating plots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Magnitude Analysis for {model_name} v{model_version}", fontsize=16)

    # Plot Activations
    df_activations.plot(ax=axes[0], logy=True)
    axes[0].set_title("Activation Magnitudes")
    axes[0].set_ylabel("Dim-Weighted L2 Norm (log)")
    axes[0].legend(title='Layer', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Plot Weights
    df_weights.plot(ax=axes[1], logy=True)
    axes[1].set_title("Weight Magnitudes")
    axes[1].set_ylabel("Dim-Weighted L2 Norm (log)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(title='Layer', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()

if __name__ == "__main__":

    main()