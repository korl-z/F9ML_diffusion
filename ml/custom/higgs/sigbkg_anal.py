import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import mlflow

from ml.common.data_utils.processors import (
    Preprocessor,
    ProcessorChainer,
)

# custom imports
from ml.custom.higgs.higgs_dataset import HiggsDataModule
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
        Initialize signal/background analysis with two classifiers.
        """
        self.device = device
        
        print(f"Loading MC classifier: {mc_classifier_uri}")
        self.mc_classifier = mlflow.pytorch.load_model(mc_classifier_uri, map_location=device)
        self.mc_classifier.eval()
        
        print(f"Loading ML classifier: {ml_classifier_uri}")
        self.ml_classifier = mlflow.pytorch.load_model(ml_classifier_uri, map_location=device)
        self.ml_classifier.eval()
    
    def get_predictions(self, classifier, data):
        """Get classifier scores (probabilities)."""
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        with torch.no_grad():
            output = classifier(data_tensor)
            
            # If output is logits, apply sigmoid
            if output.min() < 0 or output.max() > 1:
                output = torch.sigmoid(output)
        
        return output.cpu().numpy().flatten()
    
    def plot_score_distributions(self, sig_data, bkg_data, save_path=None, save_data_dir=None):
        """Plot Figure 16: Classifier score distributions."""
        print("\nCalculating classifier scores...")
        
        # Get predictions from both classifiers
        mc_scores_sig = self.get_predictions(self.mc_classifier, sig_data)
        mc_scores_bkg = self.get_predictions(self.mc_classifier, bkg_data)
        
        ml_scores_sig = self.get_predictions(self.ml_classifier, sig_data)
        ml_scores_bkg = self.get_predictions(self.ml_classifier, bkg_data)
        
        print(f"MC classifier - Sig: [{mc_scores_sig.min():.3f}, {mc_scores_sig.max():.3f}]")
        print(f"MC classifier - Bkg: [{mc_scores_bkg.min():.3f}, {mc_scores_bkg.max():.3f}]")
        print(f"ML classifier - Sig: [{ml_scores_sig.min():.3f}, {ml_scores_sig.max():.3f}]")
        print(f"ML classifier - Bkg: [{ml_scores_bkg.min():.3f}, {ml_scores_bkg.max():.3f}]")

        if save_data_dir:
            np.save(save_data_dir / "mc_scores_sig.npy", mc_scores_sig)
            np.save(save_data_dir / "mc_scores_bkg.npy", mc_scores_bkg)
            np.save(save_data_dir / "ml_scores_sig.npy", ml_scores_sig)
            np.save(save_data_dir / "ml_scores_bkg.npy", ml_scores_bkg)
            print(f"Saved raw scores to {save_data_dir}")

        bins = np.linspace(0, 1, 50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        hist_mc_bkg, _ = np.histogram(mc_scores_bkg, bins=bins, density=True)
        hist_mc_sig, _ = np.histogram(mc_scores_sig, bins=bins, density=True)
        hist_ml_bkg, _ = np.histogram(ml_scores_bkg, bins=bins, density=True)
        hist_ml_sig, _ = np.histogram(ml_scores_sig, bins=bins, density=True)
        
        # Calculate ratio: ML_bkg / MC_bkg
        ratio_bkg = np.divide(hist_ml_bkg, hist_mc_bkg, 
                              out=np.ones_like(hist_ml_bkg), 
                              where=hist_mc_bkg!=0)
        
        if save_data_dir:
            df = pd.DataFrame({
                'bin_centers': bin_centers,
                'bin_edges_left': bins[:-1],
                'bin_edges_right': bins[1:],
                'hist_mc_bkg': hist_mc_bkg,
                'hist_mc_sig': hist_mc_sig,
                'hist_ml_bkg': hist_ml_bkg,
                'hist_ml_sig': hist_ml_sig,
                'ratio_bkg': ratio_bkg,
            })
            df.to_csv(save_data_dir / "score_distributions_data.csv", index=False)
            print(f"Saved histogram data to {save_data_dir / 'score_distributions_data.csv'}")
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(3.47412, 3.47412 * 0.8))
        
        bins = np.linspace(0, 1, 50)
        
        # MC classifier
        ax.hist(mc_scores_bkg, bins=bins, histtype='step', density=True,
                color=set1_list[1], ls='-', lw=1.5, label='MC bkg', alpha=0.8)
        ax.hist(mc_scores_sig, bins=bins, histtype='step', density=True,
                color=set1_list[0], ls='-', lw=1.5, label='MC sig', alpha=0.8)
        
        # ML classifier
        ax.hist(ml_scores_bkg, bins=bins, histtype='step', density=True,
                color=set1_list[1], ls='--', lw=1.5, label='ML bkg', alpha=0.8)
        ax.hist(ml_scores_sig, bins=bins, histtype='step', density=True,
                color=set1_list[0], ls='--', lw=1.5, label='ML sig', alpha=0.8)
        
        ax.set_xlabel('Classifier score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_yscale('log')
        ax.legend(fontsize=8, framealpha=0.9, loc='upper center')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        
        plt.tight_layout(pad=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Figure 16 to: {save_path}")
        
        
        return {
            'mc_sig': mc_scores_sig, 'mc_bkg': mc_scores_bkg,
            'ml_sig': ml_scores_sig, 'ml_bkg': ml_scores_bkg
        }
    
    def plot_roc_curves(self, sig_data, bkg_data, save_path=None, save_data_dir=None):
        """Plot Figure 17: ROC curves."""
        print("\nCalculating ROC curves...")
        
        # Get predictions
        mc_scores_sig = self.get_predictions(self.mc_classifier, sig_data)
        mc_scores_bkg = self.get_predictions(self.mc_classifier, bkg_data)
        
        ml_scores_sig = self.get_predictions(self.ml_classifier, sig_data)
        ml_scores_bkg = self.get_predictions(self.ml_classifier, bkg_data)
        
        # Combine scores and labels
        mc_scores = np.concatenate([mc_scores_sig, mc_scores_bkg])
        mc_labels = np.concatenate([np.ones(len(mc_scores_sig)), np.zeros(len(mc_scores_bkg))])
        
        ml_scores = np.concatenate([ml_scores_sig, ml_scores_bkg])
        ml_labels = np.concatenate([np.ones(len(ml_scores_sig)), np.zeros(len(ml_scores_bkg))])
        
        # Calculate ROC curves
        fpr_mc, tpr_mc, _ = roc_curve(mc_labels, mc_scores)
        fpr_ml, tpr_ml, _ = roc_curve(ml_labels, ml_scores)
        
        auc_mc = auc(fpr_mc, tpr_mc)
        auc_ml = auc(fpr_ml, tpr_ml)
        
        print(f"MC classifier AUC: {auc_mc:.4f}")
        print(f"ML classifier AUC: {auc_ml:.4f}")

        if save_data_dir:
            df_roc = pd.DataFrame({
                'fpr_mc': fpr_mc,
                'tpr_mc': tpr_mc,
                'fpr_ml': fpr_ml,
                'tpr_ml': tpr_ml,
            })
            df_roc.to_csv(save_data_dir / "roc_curves_data.csv", index=False)
            
            # Save metadata
            metadata = pd.DataFrame([{
                'auc_mc': auc_mc,
                'auc_ml': auc_ml,
            }])
            metadata.to_csv(save_data_dir / "roc_metadata.csv", index=False)
            print(f"Saved ROC data to {save_data_dir}")

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(3.47412, 3.47412 * 0.8))
        
        ax.plot(fpr_mc, tpr_mc, color=set1_list[1], ls='-', lw=1.5,
                label=f'MC (AUC={auc_mc:.3f})')
        ax.plot(fpr_ml, tpr_ml, color=set1_list[0], ls='--', lw=1.5,
                label=f'ML (AUC={auc_ml:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9, loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        
        plt.tight_layout(pad=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Figure 17 to: {save_path}")
        
        
        return {'auc_mc': auc_mc, 'auc_ml': auc_ml}


def prepare_test_data():
    """Prepare signal and background test data using your pipeline."""
    logging.info("Preparing test data...")
    
    # Use holdout partition 2 for testing
    npy_proc = HIGGSNpyProcessor(
        data_dir="/data0/korlz/f9-ml/ml/data/HIGGS/",
        base_file_name="HIGGS_data",
        hold_mode=True,
        use_hold=True,  # Use partition 2 (holdout)
    )
    
    # Get npy_file and features
    npy_file, features = npy_proc()
    
    # Select features (keep both signal and background, drop uni/disc)
    f_sel = HIGGSFeatureSelector(
        file_path=npy_file,
        features=features,
        drop_types=[],  # Keep everything including label
        on_train=None,  # Keep both sig and bkg
    )
    
    pre = Preprocessor(
        cont_rescale_type="gauss_rank",
        disc_rescale_type="none",
    )
    
    data_with_label, selection_with_label = f_sel()
    
    print(f"Loaded test data shape (with label): {data_with_label.shape}")
    
    # Split by label BEFORE preprocessing
    label_idx = selection_with_label[selection_with_label["feature"] == "label"].index[0]
    sig_mask = (data_with_label[:, label_idx] == 1).flatten()
    bkg_mask = (data_with_label[:, label_idx] == 0).flatten()
    
    sig_data_with_label = data_with_label[sig_mask]
    bkg_data_with_label = data_with_label[bkg_mask]
    
    print(f"Signal events: {len(sig_data_with_label)}")
    print(f"Background events: {len(bkg_data_with_label)}")
    
    # Now remove label and drop uni/disc types
    label_mask = np.ones(sig_data_with_label.shape[1], dtype=bool)
    label_mask[label_idx] = False
    
    sig_data = sig_data_with_label[:, label_mask]
    bkg_data = bkg_data_with_label[:, label_mask]
    
    # Update selection to remove label
    selection_no_label = selection_with_label[label_mask].reset_index(drop=True)
    
    # Now drop uni/disc types from the data
    keep_mask = ~selection_no_label["type"].isin(["uni", "disc"])
    sig_data = sig_data[:, keep_mask]
    bkg_data = bkg_data[:, keep_mask]
    selection_final = selection_no_label[keep_mask].reset_index(drop=True)
    
    print(f"After dropping uni/disc - Signal shape: {sig_data.shape}")
    print(f"After dropping uni/disc - Background shape: {bkg_data.shape}")
    
    # CHANGED: Preprocess signal and background separately
    pre = Preprocessor(
        cont_rescale_type="gauss_rank",
        disc_rescale_type="none",
    )
    
    # Preprocess signal
    sig_data_scaled, _, _ = pre.preprocess(sig_data, selection_final)
    
    # Preprocess background  
    bkg_data_scaled, _, _ = pre.preprocess(bkg_data, selection_final)
    
    print(f"Final signal data shape: {sig_data_scaled.shape}")
    print(f"Final background data shape: {bkg_data_scaled.shape}")
    
    return sig_data_scaled, bkg_data_scaled



if __name__ == "__main__":
    # Prepare test data
    sig_test, bkg_test = prepare_test_data()
    
    # Initialize analysis with your trained classifiers
    analysis = SigBkgAnalysis(
        mc_classifier_uri="models:/BinaryClassifier_sigbkg_gauss_rank_best6/2",
        ml_classifier_uri="models:/BinaryClassifier_full_sigbkg_gauss_rank_best6/1",
        device="cpu"
    )
    
    # Create output directory
    output_dir = Path("/data0/korlz/f9-ml/ml/custom/higgs/analysis/plots/sigbkg")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Figure 16: Score distributions
    print("\n" + "="*70)
    print("GENERATING FIGURE 16: Classifier Score Distributions")
    print("="*70)
    scores = analysis.plot_score_distributions(
        sig_test,
        bkg_test,
        save_path=output_dir / "f16_score_distributions.png", save_data_dir=output_dir
    )
    
    # Figure 17: ROC curves
    print("\n" + "="*70)
    print("GENERATING FIGURE 17: ROC Curves")
    print("="*70)
    roc_results = analysis.plot_roc_curves(
        sig_test,
        bkg_test,
        save_path=output_dir / "f17_roc_curves.png", save_data_dir=output_dir
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Plots saved to: {output_dir}")
    print(f"MC classifier AUC: {roc_results['auc_mc']:.4f}")
    print(f"ML classifier AUC: {roc_results['auc_ml']:.4f}")