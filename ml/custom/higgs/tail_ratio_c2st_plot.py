import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow

local_mlruns_path = r"C:/Users/Uporabnik/Documents/IJS-F9/korlz/ppt/data/models/mlruns"
mlflow.set_tracking_uri(f"file:///{local_mlruns_path}")

# Apply consistent styling
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
            
            # Check if output is logits or probabilities
            # If sigmoid is already applied, we need to convert back to logits
            if output.min() >= 0 and output.max() <= 1:
                # Output is probability, convert to logits
                # logit(p) = log(p / (1-p))
                eps = 1e-10  # Numerical stability
                output = torch.log((output + eps) / (1 - output + eps))
            
            # Calculate r(x) = exp(logit)
            r_x = torch.exp(output).cpu().numpy().flatten()
        
        return r_x
    
    def plot_density_ratio_distribution(self, mc_data, ml_data, 
                                       tail_cut_low=0.94, tail_cut_high=1.08,
                                       save_path=None):
        """
        Plot density ratio distribution (Paper's Figure 14).
        
        Parameters
        ----------
        mc_data : np.ndarray
            MC test data [N, D]
        ml_data : np.ndarray
            ML generated test data [N, D]
        tail_cut_low : float
            Lower threshold for tail events
        tail_cut_high : float
            Upper threshold for tail events
        save_path : str, optional
            Path to save figure
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
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(3.47412, 3.47412 * 0.8))
        
        # Use common bins
        all_r_x = np.concatenate([r_x_mc, r_x_ml])
        bin_edges = np.histogram_bin_edges(all_r_x, bins=100, 
                                           range=(all_r_x.min(), all_r_x.max()))
        
        ax.hist(r_x_mc, bins=bin_edges, histtype='step', 
                label='MC c2st', color=set1_list[1], lw=1.5, density=True, zorder=2)
        ax.hist(r_x_ml, bins=bin_edges, histtype='step', 
                label='ML c2st', color=set1_list[0], lw=1.5, density=True, zorder=3)
        
        # Add vertical lines for tail cuts
        ax.axvline(tail_cut_low, color='gray', ls='--', lw=1, alpha=0.7, 
                  label='tail cut', zorder=1)
        ax.axvline(tail_cut_high, color='gray', ls='--', lw=1, alpha=0.7, zorder=1)
        
        ax.set_xlabel(r'$r(x)$', fontsize=10)
        ax.set_ylabel('density [a.u.]', fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
        
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


# Example usage
if __name__ == "__main__":
    # --- THIS IS HOW YOU DEFINE THE MODEL TO LOAD ---
    # The URI format is "models:/<REGISTERED_MODEL_NAME>/<VERSION>"
    registered_model_name = "BinaryClassifier_unet1d_ddpm_model_c2st_gen_model_all"
    model_version = 6 # Example version
    c2st_model_uri = f"models:/{registered_model_name}/{model_version}"

    mc_test_data = np.load(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ml\data\HIGGS\HIGGS_data_hold_partition_2.npy") 
    ml_test_data = np.load(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ml\data\HIGGS\HIGGS_generated_unet1d_ddpm_model6_6.npy")  
    
    # Initialize analysis
    density_analysis = DensityRatioAnalysis(c2st_model_uri, device="cpu")
    
    # Plot density ratio distribution (Figure 14)
    r_x_mc, r_x_ml = density_analysis.plot_density_ratio_distribution(
        mc_test_data, 
        ml_test_data,
        save_path="density_ratio_distribution.pdf"
    )
    
    # Get tail events for further analysis (Figure 15)
    tail_mask_ml, tail_data_ml = density_analysis.get_tail_events(ml_test_data, r_x_ml)
    tail_mask_mc, tail_data_mc = density_analysis.get_tail_events(mc_test_data, r_x_mc)