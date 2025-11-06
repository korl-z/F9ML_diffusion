import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Apply consistent styling
plt.rcParams.update(
    {"text.usetex": True, "font.family": "Helvetica", "font.size": 10}
)
set1_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#f781bf", "#999999"]

# Features list (matching paper)
features_list = [
    r"lepton $p_T$", r"lepton $\eta$", "missing energy",
    r"jet1 $p_T$", r"jet1 $\eta$", r"jet2 $p_T$", r"jet2 $\eta$",
    r"jet3 $p_T$", r"jet3 $\eta$", r"jet4 $p_T$", r"jet4 $\eta$",
    r"$m_{jj}$", r"$m_{jjj}$", r"$m_{\ell\nu}$", r"$m_{j\ell\nu}$",
    r"$m_{b\bar{b}}$", r"$m_{Wb\bar{b}}$", r"$m_{WWb\bar{b}}$"
]

class DensityRatioPlotter:
    def __init__(self, data_dir):
        """
        Initialize plotter with data directory.
        
        Parameters
        ----------
        data_dir : str or Path
            Directory containing the saved .npy and .csv files
        """
        self.data_dir = Path(data_dir)
        
        # Load all data
        self.r_x_mc = np.load(self.data_dir / "r_x_mc.npy")
        self.r_x_ml = np.load(self.data_dir / "r_x_ml.npy")
        self.tail_mask_mc = np.load(self.data_dir / "tail_mask_mc.npy")
        self.tail_mask_ml = np.load(self.data_dir / "tail_mask_ml.npy")
        self.mc_test_data = np.load(self.data_dir / "mc_test_data.npy")
        self.ml_test_data = np.load(self.data_dir / "ml_test_data.npy")
        
        # Load histogram data
        self.hist_data = pd.read_csv(self.data_dir / "density_ratio_data.csv")
        self.metadata = pd.read_csv(self.data_dir / "density_ratio_data_metadata.csv")
        
        print(f"Loaded data:")
        print(f"  MC events: {len(self.mc_test_data)}")
        print(f"  ML events: {len(self.ml_test_data)}")
        print(f"  MC tail events: {np.sum(self.tail_mask_mc)} ({np.sum(self.tail_mask_mc)/len(self.tail_mask_mc)*100:.2f}%)")
        print(f"  ML tail events: {np.sum(self.tail_mask_ml)} ({np.sum(self.tail_mask_ml)/len(self.tail_mask_ml)*100:.2f}%)")
    
    def plot_density_ratio_distribution(self, save_path=None):
        """
        Plot Figure 14: Density ratio distribution.
        """
        tail_cut_low = self.metadata['tail_cut_low'].values[0]
        tail_cut_high = self.metadata['tail_cut_high'].values[0]
        
        fig, ax = plt.subplots(1, 1, figsize=(3.47412, 3.47412 * 0.8))
        
        # Plot as step histograms using bin edges
        bin_edges_left = self.hist_data['bin_edges_left'].values
        bin_edges_right = self.hist_data['bin_edges_right'].values
        bin_edges = np.append(bin_edges_left, bin_edges_right[-1])
        
        ax.hist(bin_edges[:-1], bins=bin_edges, weights=self.hist_data['hist_mc'].values,
                histtype='step', label='MC c2st', color=set1_list[1], lw=1.5, zorder=2)
        ax.hist(bin_edges[:-1], bins=bin_edges, weights=self.hist_data['hist_ml'].values,
                histtype='step', label='ML c2st', color=set1_list[0], lw=1.5, zorder=3)
        

        ax.axvspan(bin_edges[0], tail_cut_low, color='gray', alpha=0.15, zorder=0)
        ax.axvspan(tail_cut_high, bin_edges[-1], color='gray', alpha=0.15, zorder=0)
        # Add vertical lines for tail cuts
        ax.axvline(tail_cut_low, color='gray', ls='--', lw=1, alpha=0.7, 
                  label='tail cut', zorder=1)
        ax.axvline(tail_cut_high, color='gray', ls='--', lw=1, alpha=0.7, zorder=1)
        ax.axvline(1, color='k', ls='-', lw=1, alpha=0.7, zorder=1)
        
        ax.set_xlabel(r'$r(x)$', fontsize=10)
        ax.set_ylabel('density [a.u.]', fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        
        plt.tight_layout(pad=0.3)
        ax.set_xlim(0.94, 1.06)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Figure 14 to: {save_path}")
        
        plt.show()
    
    def plot_tail_kinematic_distributions(self, save_path=None):
        """
        Plot Figure 15: Kinematic distributions for tail events.
        """
        D = self.mc_test_data.shape[1]
        ncols = 6
        nrows = int(np.ceil(D / ncols))
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3.47412, nrows * 3.47412 * 0.8))
        ax_flat = np.array(axs).reshape(-1)
        
        tail_cut_low = self.metadata['tail_cut_low'].values[0]
        tail_cut_high = self.metadata['tail_cut_high'].values[0]
        
        # Get tail data
        ml_tail_low = self.ml_test_data[self.r_x_ml < tail_cut_low]
        ml_tail_high = self.ml_test_data[self.r_x_ml > tail_cut_high]
        
        for feat_idx in range(D):
            ax = ax_flat[feat_idx]
            
            # MC reference (all events)
            mc_data = self.mc_test_data[:, feat_idx]
            bin_edges = np.histogram_bin_edges(mc_data, bins=50)
            
            # Plot MC as filled histogram
            ax.hist(mc_data, bins=bin_edges, density=True, 
                   histtype='bar', color='gray', alpha=0.4, 
                   label='MC', linewidth=0)
            # ax.hist(mc_data, bins=bin_edges, density=True, 
            #        histtype='step', color='black', alpha=0.8, linewidth=1.5)
            
            # Plot ML tail events
            if len(ml_tail_low) > 0:
                ax.hist(ml_tail_low[:, feat_idx], bins=bin_edges, density=True,
                       histtype='step', color=set1_list[1], lw=1.5,
                       label=f'ML tail cut $< {tail_cut_low}$', alpha=0.8)
            
            if len(ml_tail_high) > 0:
                ax.hist(ml_tail_high[:, feat_idx], bins=bin_edges, density=True,
                       histtype='step', color=set1_list[0], lw=1.5,
                       label=f'ML tail cut $> {tail_cut_high}$', alpha=0.8)
            
            # ax.set_yscale('log')
            ax.set_xlabel(features_list[feat_idx], fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            
            # Add legend only to first plot
            if feat_idx == 0:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
        
        # Turn off unused subplots
        for j in range(D, len(ax_flat)):
            ax_flat[j].axis('off')
        
        plt.tight_layout(pad=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Figure 15 to: {save_path}")
        
        plt.show()
    
    def plot_tail_kinematic_distributions_alternative(self, save_path=None):
        """
        Alternative version: Show MC vs ML for tail events only.
        """
        D = self.mc_test_data.shape[1]
        ncols = 6
        nrows = int(np.ceil(D / ncols))
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3.47412, nrows * 3.47412 * 0.8))
        ax_flat = np.array(axs).reshape(-1)
        
        # Get tail data for both MC and ML
        mc_tail = self.mc_test_data[self.tail_mask_mc]
        ml_tail = self.ml_test_data[self.tail_mask_ml]
        
        for feat_idx in range(D):
            ax = ax_flat[feat_idx]
            
            # Get data range from all MC data
            mc_all = self.mc_test_data[:, feat_idx]
            bin_edges = np.histogram_bin_edges(mc_all, bins=50)
            
            # Plot MC all events as reference
            ax.hist(mc_all, bins=bin_edges, density=True,
                   histtype='stepfilled', color='gray', alpha=0.3,
                   label='MC (all)', linewidth=0)
            
            # Plot tail events
            if len(mc_tail) > 0:
                ax.hist(mc_tail[:, feat_idx], bins=bin_edges, density=True,
                       histtype='step', color=set1_list[1], lw=1.5,
                       label='MC tail', alpha=0.8)
            
            if len(ml_tail) > 0:
                ax.hist(ml_tail[:, feat_idx], bins=bin_edges, density=True,
                       histtype='step', color=set1_list[0], lw=1.5,
                       label='ML tail', alpha=0.8)
            
            # ax.set_yscale('log')
            ax.set_xlabel(features_list[feat_idx], fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            
            if feat_idx == 0:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
        
        for j in range(D, len(ax_flat)):
            ax_flat[j].axis('off')
        
        plt.tight_layout(pad=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved alternative Figure 15 to: {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("DENSITY RATIO ANALYSIS SUMMARY")
        print("="*70)
        
        meta = self.metadata.iloc[0]
        
        print(f"\nTail Cuts:")
        print(f"  Lower: {meta['tail_cut_low']:.3f}")
        print(f"  Upper: {meta['tail_cut_high']:.3f}")
        
        print(f"\nDensity Ratio Ranges:")
        print(f"  MC: [{meta['mc_r_x_min']:.3f}, {meta['mc_r_x_max']:.3f}]")
        print(f"  ML: [{meta['ml_r_x_min']:.3f}, {meta['ml_r_x_max']:.3f}]")
        
        print(f"\nFraction in Tails:")
        print(f"  MC: {meta['mc_tail_frac']*100:.2f}%")
        print(f"  ML: {meta['ml_tail_frac']*100:.2f}%")
        
        print(f"\nTail Event Counts:")
        print(f"  MC: {np.sum(self.tail_mask_mc):,} / {len(self.tail_mask_mc):,}")
        print(f"  ML: {np.sum(self.tail_mask_ml):,} / {len(self.tail_mask_ml):,}")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    data_dir = Path(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\ratiotest")
    
    plotter = DensityRatioPlotter(data_dir)
    
    plotter.print_summary()
    
    # Create output directory for plots
    output_dir = Path(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # # Plot Figure 14: Density ratio distribution
    # print("\nGenerating Figure 14...")
    # plotter.plot_density_ratio_distribution(
        # save_path=output_dir / "figure14_density_ratio.pdf"
    # )
    
    # Plot Figure 15: Kinematic distributions for tail events
    print("\nGenerating Figure 15...")
    plotter.plot_tail_kinematic_distributions(
        save_path=output_dir / "figure15_tail_kinematics.pdf"
    )
    
    # # Alternative version (optional)
    # print("\nGenerating Figure 15 alternative...")
    # plotter.plot_tail_kinematic_distributions_alternative(
    #     save_path=output_dir / "figure15_tail_kinematics_alt.pdf"
    # )
    
    print(f"\nAll plots saved to: {output_dir}")