import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Apply consistent styling
plt.rcParams.update(
    {"text.usetex": True, "font.family": "Helvetica", "font.size": 10}
)
set1_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#f781bf", "#999999"]


def plot_score_distributions_from_data(data_dir, score_cut=0.55, save_path=None):
    """
    Plot Figure 16: Classifier score distributions with ratio subplot.
    
    Parameters
    ----------
    data_dir : Path or str
        Directory containing the saved data files
    score_cut : float
        Score threshold for classification cut
    save_path : Path or str, optional
        Path to save the figure
    """
    data_dir = Path(data_dir)
    
    # Load histogram data
    df = pd.read_csv(data_dir / "score_distributions_dataEDM.csv")
    
    bin_centers = df['bin_centers'].values
    bin_edges_left = df['bin_edges_left'].values
    bin_edges_right = df['bin_edges_right'].values
    bins = np.append(bin_edges_left, bin_edges_right[-1])
    
    hist_mc_bkg = df['hist_mc_bkg'].values
    hist_mc_sig = df['hist_mc_sig'].values
    hist_ml_bkg = df['hist_ml_bkg'].values
    hist_ml_sig = df['hist_ml_sig'].values
    ratio_bkg = df['ratio_bkg'].values
    
    # Create plot with ratio
    fig = plt.figure(figsize=(3.47412, 3.47412 * 1.2))
    
    # Main plot
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    
    # Plot histograms as step functions
    # MC classifier
    ax1.stairs(hist_mc_bkg, bins, color=set1_list[1], 
              linestyle='-', linewidth=1, label='MC bkg', alpha=0.8)
    ax1.stairs(hist_mc_sig, bins, color=set1_list[0], 
              linestyle='-', linewidth=1, label='MC sig', alpha=0.8)
    
    # ML classifier
    ax1.stairs(hist_ml_bkg, bins, color=set1_list[3], 
              linestyle='-', linewidth=1, label='ML bkg', alpha=0.8)
    ax1.stairs(hist_ml_sig, bins, color=set1_list[4], 
              linestyle='-', linewidth=1, label='ML sig', alpha=0.8)
    
    # Add cut line
    ax1.axvline(score_cut, color='gray', ls=':', lw=1.5, alpha=0.7, 
               label=f'cut={score_cut}')
    
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_yscale('log')
    ax1.legend(fontsize=7, framealpha=0.9, loc='upper center', ncol=3)
    ax1.set_xlim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.set_xticklabels([])
    ax1.tick_params(which="both", direction='in')
    
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
        print(f"Saved Figure 16 to: {save_path}")
    
    plt.show()


def plot_roc_curves_from_data(data_dir, save_path=None):
    """
    Plot Figure 17: ROC curves from saved data.
    
    Parameters
    ----------
    data_dir : Path or str
        Directory containing the saved data files
    save_path : Path or str, optional
        Path to save the figure
    """
    data_dir = Path(data_dir)
    
    # Load ROC data
    df_roc = pd.read_csv(data_dir / "roc_curves_dataEDM.csv")
    metadata = pd.read_csv(data_dir / "roc_metadataEDM.csv")
    
    fpr_mc = df_roc['fpr_mc'].values
    tpr_mc = df_roc['tpr_mc'].values
    fpr_ml = df_roc['fpr_ml'].values
    tpr_ml = df_roc['tpr_ml'].values
    
    auc_mc = metadata['auc_mc'].values[0]
    auc_ml = metadata['auc_ml'].values[0]
    
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
    ax.tick_params(which="both", direction='in')
    
    plt.tight_layout(pad=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Paths
    data_dir = Path(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\sigbkg")
    output_dir = Path(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plot_score_distributions_from_data(
        data_dir,
        score_cut=0.57,
        save_path=output_dir / "f16_score_distributionsEDM.png"
    )