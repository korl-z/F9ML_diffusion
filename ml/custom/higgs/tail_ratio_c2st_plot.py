import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow

from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)

local_mlruns_path = "/data0/korlz/f9-ml/ml/custom/higgs/mlruns"
mlflow.set_tracking_uri(f"file:///{local_mlruns_path}")

plt.rcParams.update(
    {"text.usetex": True, "font.family": "Helvetica", "font.size": 10}
)
set1_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#f781bf", "#999999"]

class DensityRatioAnalysis:
    def __init__(self, c2st_mlflow_path, device="cpu"):
        """
        Initialize density ratio analysis with MLflow saved C2ST classifier.
        
        Parameters
        ----------
        c2st_mlflow_path : str
            Path to MLflow artifacts directory containing the C2ST model
            e.g., "/data0/korlz/f9-ml/ml/custom/higgs/mlruns/406586685500999771/1f072e2af32944dfb1e9473fca5f43e6"
        device : str
            Device to use for inference
        """
        self.device = device
        self.c2st_model = self._load_c2st_model(c2st_mlflow_path)
        

    def _load_c2st_model(self, model_uri):
        """
        Load C2ST classifier from MLflow using a model URI.
        
        Parameters
        ----------
        model_uri : str
            MLflow model URI, e.g., "models:/my_c2st_model/1" or "models:/my_c2st_model/Staging"
        """
        print(f"Loading C2ST model from MLflow URI: {model_uri}")
        
        # Load using the correct MLflow URI
        model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
        model.eval()
        
        return model
    
    def get_density_ratio(self, data):
        """
        Calculate density ratio r(x) = P_ML(x) / P_MC(x) from C2ST classifier.
        
        Uses Equation 24 from paper:
        r(x) = exp(σ^{-1}[p(y=1|x)])
        
        Parameters
        ----------
        data : np.ndarray
            Input data [N, D]
            
        Returns
        -------
        r_x : np.ndarray
            Density ratio for each sample [N,]
        """
        # Convert to tensor
        if isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data)
        else:
            data_tensor = data
            
        data_tensor = data_tensor.to(self.device)
        
        with torch.no_grad():
            # Get model output
            output = self.c2st_model(data_tensor)
            
            probs = torch.clamp(output, min=1e-5, max=1-1e-5)  # Changed from 1e-7

            # Convert probability to logit: logit(p) = log(p / (1-p))
            logits = torch.log(probs / (1 - probs))

            # Calculate r(x) = exp(logit)
            r_x = torch.exp(logits).cpu().numpy().flatten()

            # Additional safety: clip r(x) to reasonable range
            # Paper uses tail cuts at 0.94 and 1.08, so anything beyond ~0.5-2.0 is already extreme
            r_x = np.clip(r_x, 0.01, 100.0)  # ADD THIS

            print(f"  Output (probs) range: [{output.min():.6f}, {output.max():.6f}]")
            print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            print(f"  r(x) range: [{r_x.min():.3f}, {r_x.max():.3f}]")
            print(f"  Extreme probs (< 0.01 or > 0.99): {torch.sum((output < 0.01) | (output > 0.99)).item()}")

        return r_x
    
    def plot_density_ratio_distribution(self, mc_data, ml_data, 
                                       tail_cut_low=0.97, tail_cut_high=1.02,
                                       save_path=None,
                                       save_data_path=None):
        """
        Plot density ratio distribution (Paper's Figure 14).
        """
        print("Calculating density ratios...")
        r_x_mc = self.get_density_ratio(mc_data)
        r_x_ml = self.get_density_ratio(ml_data)

        print(f"MC r(x) range: [{r_x_mc.min():.3f}, {r_x_mc.max():.3f}]")
        print(f"ML r(x) range: [{r_x_ml.min():.3f}, {r_x_ml.max():.3f}]")

        # Calculate fraction in tails
        mc_tail_frac = np.sum((r_x_mc < tail_cut_low) | (r_x_mc > tail_cut_high)) / len(r_x_mc)
        ml_tail_frac = np.sum((r_x_ml < tail_cut_low) | (r_x_ml > tail_cut_high)) / len(r_x_ml)

        print(f"Fraction in tails - MC: {mc_tail_frac*100:.2f}%, ML: {ml_tail_frac*100:.2f}%")

        bin_range = (0.94, 1.06)  # Focus on the region of interest
        all_r_x = np.concatenate([r_x_mc, r_x_ml])

        # Filter to reasonable range for histogram
        r_x_mc_plot = r_x_mc[(r_x_mc >= bin_range[0]) & (r_x_mc <= bin_range[1])]
        r_x_ml_plot = r_x_ml[(r_x_ml >= bin_range[0]) & (r_x_ml <= bin_range[1])]

        bin_edges = np.linspace(bin_range[0], bin_range[1], 100)

        hist_mc, _ = np.histogram(r_x_mc_plot, bins=bin_edges, density=True)
        hist_ml, _ = np.histogram(r_x_ml_plot, bins=bin_edges, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if save_data_path:
            df = pd.DataFrame({
                'bin_centers': bin_centers,
                'bin_edges_left': bin_edges[:-1],
                'bin_edges_right': bin_edges[1:],
                'hist_mc': hist_mc,
                'hist_ml': hist_ml,
            })
            df.to_csv(save_data_path, index=False)
            print(f"Saved histogram data to: {save_data_path}")

            metadata = {
                'tail_cut_low': tail_cut_low,
                'tail_cut_high': tail_cut_high,
                'mc_tail_frac': mc_tail_frac,
                'ml_tail_frac': ml_tail_frac,
                'mc_r_x_min': float(r_x_mc.min()),
                'mc_r_x_max': float(r_x_mc.max()),
                'ml_r_x_min': float(r_x_ml.min()),
                'ml_r_x_max': float(r_x_ml.max()),
                'mc_extreme_count': int(np.sum((r_x_mc < 0.7) | (r_x_mc > 1.3))),  # ADD
                'ml_extreme_count': int(np.sum((r_x_ml < 0.7) | (r_x_ml > 1.3))),  # ADD
            }
            metadata_path = save_data_path.replace('.csv', '_metadata.csv')
            pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
            print(f"Saved metadata to: {metadata_path}")

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(3.47412, 3.47412 * 0.8))

        ax.hist(r_x_mc_plot, bins=bin_edges, histtype='step', 
                label='MC c2st', color=set1_list[1], lw=1.5, density=True, zorder=2)
        ax.hist(r_x_ml_plot, bins=bin_edges, histtype='step', 
                label='ML c2st', color=set1_list[0], lw=1.5, density=True, zorder=3)

        ax.axvline(tail_cut_low, color='gray', ls='--', lw=1, alpha=0.7, 
                  label='tail cut', zorder=1)
        ax.axvline(tail_cut_high, color='gray', ls='--', lw=1, alpha=0.7, zorder=1)

        ax.axvspan(bin_range[0], tail_cut_low, color='gray', alpha=0.15, zorder=0)
        ax.axvspan(tail_cut_high, bin_range[1], color='gray', alpha=0.15, zorder=0)
        ax.set_xlabel(r'$r(x)$', fontsize=10)
        ax.set_ylabel('density [a.u.]', fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
        ax.set_xlim(bin_range)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {save_path}")

        plt.show()

        return r_x_mc, r_x_ml
    
    def get_tail_events(self, data, r_x, tail_cut_low=0.94, tail_cut_high=1.08):
        """
        Extract events in the tails of density ratio distribution.
        
        Parameters
        ----------
        data : np.ndarray
            Input data [N, D]
        r_x : np.ndarray
            Density ratios [N,]
        tail_cut_low : float
            Lower threshold
        tail_cut_high : float
            Upper threshold
            
        Returns
        -------
        tail_mask : np.ndarray
            Boolean mask for tail events
        tail_data : np.ndarray
            Data in tails
        """
        tail_mask = (r_x < tail_cut_low) | (r_x > tail_cut_high)
        tail_data = data[tail_mask]
        
        print(f"Total events: {len(data)}")
        print(f"Tail events: {np.sum(tail_mask)} ({np.sum(tail_mask)/len(data)*100:.2f}%)")
        
        return tail_mask, tail_data


if __name__ == "__main__":
    registered_model_name = "BinaryClassifier_unet1d_ddpm_model_c2st_gen_model_all"
    model_version = 14 
    c2st_model_uri = f"models:/{registered_model_name}/{model_version}"

    # Load and process data correctly (matching C2ST training pipeline)
    print("Processing MC test data...")
    
    # Initialize processors with correct arguments
    npy_proc = HIGGSNpyProcessor(
        data_dir="/data0/korlz/f9-ml/ml/data/HIGGS/",
        base_file_name="HIGGS_data",
        hold_mode=True,
        use_hold=True,  # Use partition 2 (holdout/test set)
    )
    
    # Get npy_file from processor
    npy_file, features = npy_proc()
    
    f_sel = HIGGSFeatureSelector(
        file_path=npy_file,
        features=features,
        drop_types=["uni", "disc"],
        on_train="bkg",  # Only background
    )
    
    pre = Preprocessor(
        cont_rescale_type="gauss_rank",
        disc_rescale_type="none",
    )
    
    chainer = ProcessorChainer(npy_proc, f_sel, pre)
    mc_test_data, selection, scalers = chainer()
    
    label_idx = selection[selection["feature"] == "label"].index
    label_mask = np.ones(mc_test_data.shape[1], dtype=bool)
    label_mask[label_idx] = False
    mc_test_data = mc_test_data[:, label_mask]
    
    # Load generated data (already preprocessed, no label)
    ml_test_data = np.load("/data0/korlz/f9-ml/ml/data/HIGGS/HIGGS_generated_unet1d_ddpm_model6_6.npy")
    
    print(f"MC test data shape: {mc_test_data.shape}")
    print(f"ML test data shape: {ml_test_data.shape}")
    
    # Ensure same size for fair comparison
    min_size = min(len(mc_test_data), len(ml_test_data))
    mc_test_data = mc_test_data[:min_size]
    ml_test_data = ml_test_data[:min_size]
    
    density_analysis = DensityRatioAnalysis(c2st_model_uri, device="cpu")
    
    output_dir = Path("/data0/korlz/f9-ml/ml/custom/higgs/analysis/plots/ratioplots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    r_x_mc, r_x_ml = density_analysis.plot_density_ratio_distribution(
        mc_test_data, 
        ml_test_data,
        save_path=str(output_dir / "density_ratio_distribution.png"),
        save_data_path=str(output_dir / "density_ratio_data.csv")
    )
    
    tail_mask_ml, tail_data_ml = density_analysis.get_tail_events(ml_test_data, r_x_ml)
    tail_mask_mc, tail_data_mc = density_analysis.get_tail_events(mc_test_data, r_x_mc)
    
    np.save(output_dir / "r_x_mc.npy", r_x_mc)
    np.save(output_dir / "r_x_ml.npy", r_x_ml)
    np.save(output_dir / "tail_mask_mc.npy", tail_mask_mc)
    np.save(output_dir / "tail_mask_ml.npy", tail_mask_ml)
    np.save(output_dir / "mc_test_data.npy", mc_test_data)
    np.save(output_dir / "ml_test_data.npy", ml_test_data)