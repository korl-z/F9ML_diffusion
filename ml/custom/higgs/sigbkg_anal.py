import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import mlflow

from ml.common.data_utils.processors import (
    Preprocessor,
)

# custom imports
from ml.custom.higgs.process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)

plt.rcParams.update(
    {"text.usetex": True, "font.family": "Helvetica", "font.size": 10}
)
set1_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#f781bf", "#999999"]

mlflow.set_tracking_uri("file:///data0/korlz/f9-ml/ml/custom/higgs/mlruns")

class SigBkgAnalysis:
    def __init__(self, mc_classifier_uri, ml_classifier_uri, device="cpu"):
        """
        Initialize sig bkg analysis with two classifiers.
        """
        self.device = device
        
        print(f"Loading MC classifier: {mc_classifier_uri}")
        self.mc_classifier = mlflow.pytorch.load_model(mc_classifier_uri, map_location=device)
        self.mc_classifier.eval()
        
        print(f"Loading ML classifier: {ml_classifier_uri}")
        self.ml_classifier = mlflow.pytorch.load_model(ml_classifier_uri, map_location=device)
        self.ml_classifier.eval()
    
    def get_predictions(self, classifier, data):
        """Get classifier scores"""
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        with torch.no_grad():
            output = classifier(data_tensor)
            
            if output.min() < 0 or output.max() > 1:
                output = torch.sigmoid(output)
        
        return output.cpu().numpy().flatten()
    
    def plot_score_distributions(self, sig_data, bkg_mc_data, bkg_ml_data, 
                                 save_path=None, save_data_dir=None, 
                                 score_cut=0.5, use_mc_classifier=True):
        """
        Plot classifier score distributions using SAME classifier on all data.
        
        Parameters
        ----------
        sig_data : np.ndarray
            Signal data
        bkg_mc_data : np.ndarray
            MC background data
        bkg_ml_data : np.ndarray
            ML-generated background data
        use_mc_classifier : bool
            If True, use MC classifier for all (recommended for Figure 16)
        """
        print("\nCalculating classifier scores...")
        
        # Use SAME classifier on all datasets
        if use_mc_classifier:
            classifier = self.mc_classifier
            classifier_name = "MC"
            print("Using MC classifier on all datasets")
        else:
            classifier = self.ml_classifier
            classifier_name = "ML"
            print("Using ML classifier on all datasets")
        
        # Get predictions from SAME classifier
        scores_sig = self.get_predictions(classifier, sig_data)
        scores_bkg_mc = self.get_predictions(classifier, bkg_mc_data)
        scores_bkg_ml = self.get_predictions(classifier, bkg_ml_data)
        
        print(f"{classifier_name} classifier - Sig: [{scores_sig.min():.3f}, {scores_sig.max():.3f}]")
        print(f"{classifier_name} classifier - MC bkg: [{scores_bkg_mc.min():.3f}, {scores_bkg_mc.max():.3f}]")
        print(f"{classifier_name} classifier - ML bkg: [{scores_bkg_ml.min():.3f}, {scores_bkg_ml.max():.3f}]")

        # Save raw scores
        if save_data_dir:
            np.save(save_data_dir / f"{classifier_name}_scores_sig.npy", scores_sig)
            np.save(save_data_dir / f"{classifier_name}_scores_bkg_mc.npy", scores_bkg_mc)
            np.save(save_data_dir / f"{classifier_name}_scores_bkg_ml.npy", scores_bkg_ml)
            print(f"Saved raw scores to {save_data_dir}")

        bins = np.linspace(0, 1, 50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        hist_bkg_mc, _ = np.histogram(scores_bkg_mc, bins=bins, density=True)
        hist_bkg_ml, _ = np.histogram(scores_bkg_ml, bins=bins, density=True)
        hist_sig, _ = np.histogram(scores_sig, bins=bins, density=True)
        
        # Ratio: ML bkg / MC bkg
        ratio_bkg = np.divide(hist_bkg_ml, hist_bkg_mc, 
                              out=np.ones_like(hist_bkg_ml), 
                              where=hist_bkg_mc!=0)
        
        # Save histogram data for local plotting
        if save_data_dir:
            df = pd.DataFrame({
                'bin_centers': bin_centers,
                'bin_edges_left': bins[:-1],
                'bin_edges_right': bins[1:],
                'hist_mc_bkg': hist_bkg_mc,
                'hist_mc_sig': hist_sig,
                'hist_ml_bkg': hist_bkg_ml,
                'hist_ml_sig': hist_sig,  # Same signal for both (it's MC signal)
                'ratio_bkg': ratio_bkg,
            })
            df.to_csv(save_data_dir / f"score_distributions_data{classifier_name}.csv", index=False)
            print(f"Saved histogram data to {save_data_dir / f'score_distributions_data{classifier_name}.csv'}")
        
        # Create figure with ratio
        fig = plt.figure(figsize=(3.47412, 3.47412 * 1.2))
        
        # Main plot
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        
        # Plot all three datasets scored by SAME classifier
        ax1.hist(scores_bkg_mc, bins=bins, histtype='step', density=True,
                color=set1_list[1], ls='-', lw=1.5, label='MC bkg', alpha=0.8)
        ax1.hist(scores_bkg_ml, bins=bins, histtype='step', density=True,
                color=set1_list[2], ls='-', lw=1.5, label='ML bkg', alpha=0.8)
        ax1.hist(scores_sig, bins=bins, histtype='step', density=True,
                color=set1_list[0], ls='-', lw=1.5, label='MC sig', alpha=0.8)
        
        # Add cut line
        ax1.axvline(score_cut, color='gray', ls=':', lw=1.5, alpha=0.7, 
                   label=f'cut={score_cut}')
        
        ax1.set_ylabel('Density', fontsize=10)
        ax1.set_yscale('log')
        ax1.legend(fontsize=7, framealpha=0.9, loc='upper center', ncol=2)
        ax1.set_xlim(0, 1)
        ax1.grid(alpha=0.3)
        ax1.set_xticklabels([])
        ax1.tick_params(which="both", direction='in')
        ax1.set_title(f'Using {classifier_name} classifier', fontsize=9)
        
        # Ratio plot
        ax2 = plt.subplot2grid((4, 1), (3, 0))
        ax2.plot(bin_centers, ratio_bkg, color='k', lw=1.5, marker='o', 
                markersize=3, label='ML bkg / MC bkg')
        ax2.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
        ax2.axvline(score_cut, color='gray', ls=':', lw=1.5, alpha=0.7)
        
        ax2.set_xlabel('Classifier score', fontsize=10)
        ax2.set_ylabel('Ratio', fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0.5, 1.5)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=7, framealpha=0.9, loc='upper right')
        ax2.tick_params(which="both", direction='in')
        
        plt.tight_layout(pad=0.3, h_pad=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.close()
        
        return {
            'sig': scores_sig, 
            'bkg_mc': scores_bkg_mc,
            'bkg_ml': scores_bkg_ml,
            'ratio': ratio_bkg
        }
    

def prepare_test_data():
    """Prepare MC signal, MC background, and ML background test data."""
    print("="*70)
    print("PREPARING TEST DATA")
    print("="*70)
    
    # Load MC data
    print("\n1. Loading MC signal and background...")
    npy_proc = HIGGSNpyProcessor(
        data_dir="/data0/korlz/f9-ml/ml/data/HIGGS/",
        base_file_name="HIGGS_data",
        hold_mode=True,
        use_hold=True,  
    )
    
    npy_file, features = npy_proc()
    
    f_sel = HIGGSFeatureSelector(
        file_path=npy_file,
        features=features,
        drop_types=[],  
        on_train=None,  
    )
    
    data_with_label, selection_with_label = f_sel()
    
    print(f"   Loaded test data shape (with label): {data_with_label.shape}")
    
    # Split by label
    label_idx = selection_with_label[selection_with_label["feature"] == "label"].index[0]
    sig_mask = (data_with_label[:, label_idx] == 1).flatten()
    bkg_mask = (data_with_label[:, label_idx] == 0).flatten()
    
    sig_data_with_label = data_with_label[sig_mask]
    bkg_mc_data_with_label = data_with_label[bkg_mask]
    
    print(f"   Signal events: {len(sig_data_with_label)}")
    print(f"   MC background events: {len(bkg_mc_data_with_label)}")
    
    # Remove label
    label_mask = np.ones(sig_data_with_label.shape[1], dtype=bool)
    label_mask[label_idx] = False
    
    sig_data = sig_data_with_label[:, label_mask]
    bkg_mc_data = bkg_mc_data_with_label[:, label_mask]
    
    selection_no_label = selection_with_label[label_mask].reset_index(drop=True)
    
    # Drop uni/disc
    keep_mask = ~selection_no_label["type"].isin(["uni", "disc"])
    sig_data = sig_data[:, keep_mask]
    bkg_mc_data = bkg_mc_data[:, keep_mask]
    selection_final = selection_no_label[keep_mask].reset_index(drop=True)
    
    print(f"   After dropping uni/disc - Signal shape: {sig_data.shape}")
    print(f"   After dropping uni/disc - MC background shape: {bkg_mc_data.shape}")
    
    # Preprocess MC data
    print("\n2. Preprocessing MC data...")
    pre = Preprocessor(
        cont_rescale_type="gauss_rank",
        disc_rescale_type="none",
    )
    
    sig_data_scaled, _, scalers = pre.preprocess(sig_data, selection_final)
    bkg_mc_data_scaled, _, _ = pre.preprocess(bkg_mc_data, selection_final)
    
    # Load ML background
    print("\n3. Loading ML-generated background...")
    # ml_bkg_path = "/data0/korlz/f9-ml/ml/data/HIGGS/HIGGS_generated_unet1D_EDM_s_model_v2.npy"
    ml_bkg_path = "/data0/korlz/f9-ml/ml/data/HIGGS/HIGGS_generated_unet1d_ddpm_model_v6.npy"
    bkg_ml_data_scaled = np.load(ml_bkg_path)
    
    print(f"   ML background: {len(bkg_ml_data_scaled)} events, shape: {bkg_ml_data_scaled.shape}")
    
    print(f"\nFinal shapes:")
    print(f"   Signal: {sig_data_scaled.shape}")
    print(f"   MC background: {bkg_mc_data_scaled.shape}")
    print(f"   ML background: {bkg_ml_data_scaled.shape}")
    
    return sig_data_scaled, bkg_mc_data_scaled, bkg_ml_data_scaled



if __name__ == "__main__":
    # Prepare data - now returns 3 datasets
    sig_test, bkg_mc_test, bkg_ml_test = prepare_test_data()
    
    analysis = SigBkgAnalysis(
        mc_classifier_uri="models:/BinaryClassifier_sigbkg_gauss_rank_MC/1",
        # ml_classifier_uri="models:/BinaryClassifier_sigbkg_gauss_rank_EDMxl/1",
        ml_classifier_uri="models:/BinaryClassifier_full_sigbkg_gauss_rank_best6/1",
        device="cpu"
    )
    
    output_dir = Path("/data0/korlz/f9-ml/ml/custom/higgs/analysis/plots/sigbkg")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot with MC classifier (recommended)
    print("\n" + "="*70)
    print("FIGURE 16: Using MC Classifier")
    print("="*70)
    scores = analysis.plot_score_distributions(
        sig_test,
        bkg_mc_test,
        bkg_ml_test,
        save_path=output_dir / "f16_score_distributions_MC_classifier.png",
        save_data_dir=output_dir,
        score_cut=0.55,
        use_mc_classifier=True
    )