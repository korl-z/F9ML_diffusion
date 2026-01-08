import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica", "font.size": 10})
set1_list = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#f781bf",
    "#999999",
]

features_list = [
    r"lepton $p_T$",
    r"lepton $\eta$",
    "missing energy",
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
    r"$m_{\ell\nu}$",
    r"$m_{j\ell\nu}$",
    r"$m_{b\bar{b}}$",
    r"$m_{Wb\bar{b}}$",
    r"$m_{WWb\bar{b}}$",
]


class DensityRatio_plot:
    def __init__(self, data_dir):
        """
        init plotter with data dir
        """
        self.data_dir = Path(data_dir)

        # loaddata
        self.r_x_mc = np.load(self.data_dir / "r_x_mc.npy")
        self.r_x_ml = np.load(self.data_dir / "r_x_ml.npy")
        self.tail_mask_mc = np.load(self.data_dir / "tail_mask_mc.npy")
        self.tail_mask_ml = np.load(self.data_dir / "tail_mask_ml.npy")
        self.mc_test_data = np.load(self.data_dir / "mc_test_data.npy")
        self.ml_test_data = np.load(self.data_dir / "ml_test_data.npy")

        # load hist data
        self.hist_data = pd.read_csv(
            self.data_dir / "density_ratio_dataDDPM.csv"
        )  # DDPM/EDM
        self.metadata = pd.read_csv(
            self.data_dir / "density_ratio_dataDDPM_metadata.csv"
        )

        print(f"MC events: {len(self.mc_test_data)}")
        print(f"ML events: {len(self.ml_test_data)}")
        print(
            f"MC tail events: {np.sum(self.tail_mask_mc)} ({np.sum(self.tail_mask_mc)/len(self.tail_mask_mc)*100:.2f}%)"
        )
        print(
            f"ML tail events: {np.sum(self.tail_mask_ml)} ({np.sum(self.tail_mask_ml)/len(self.tail_mask_ml)*100:.2f}%)"
        )

    def hist_dr_plot(self, save_path=None):
        """
        density ratio distribution
        """
        tail_cut_low = self.metadata["tail_cut_low"].values[0]
        tail_cut_high = self.metadata["tail_cut_high"].values[0]

        fig, ax = plt.subplots(1, 1, figsize=(3.47412, 3.47412 * 0.8))

        bin_edges_left = self.hist_data["bin_edges_left"].values
        bin_edges_right = self.hist_data["bin_edges_right"].values
        bin_edges = np.append(bin_edges_left, bin_edges_right[-1])

        ax.hist(
            bin_edges[:-1],
            bins=bin_edges,
            weights=self.hist_data["hist_mc"].values,
            histtype="step",
            label="MC c2st",
            color=set1_list[1],
            lw=1.5,
            zorder=2,
        )
        ax.hist(
            bin_edges[:-1],
            bins=bin_edges,
            weights=self.hist_data["hist_ml"].values,
            histtype="step",
            label="ML c2st",
            color=set1_list[0],
            lw=1.5,
            zorder=3,
        )

        ax.axvspan(bin_edges[0], tail_cut_low, color="gray", alpha=0.15, zorder=0)
        ax.axvspan(tail_cut_high, bin_edges[-1], color="gray", alpha=0.15, zorder=0)

        ax.axvline(
            tail_cut_low,
            color="gray",
            ls="--",
            lw=1,
            alpha=0.7,
            label="tail cut",
            zorder=1,
        )
        ax.axvline(tail_cut_high, color="gray", ls="--", lw=1, alpha=0.7, zorder=1)
        ax.axvline(1, color="k", ls="-", lw=1, alpha=0.7, zorder=1)

        ax.set_xlabel(r"$r(x)$", fontsize=10)
        ax.set_ylabel("density [a.u.]", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
        ax.set_xlim(bin_edges[0], bin_edges[-1])

        plt.tight_layout(pad=0.3)
        ax.set_xlim(0.94, 1.06)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved Figure 14 to: {save_path}")

        plt.show()

    def tail_distributions_plot(self, save_path=None):
        """
        tail event distributions
        """
        D = self.mc_test_data.shape[1]
        ncols = 6
        nrows = int(np.ceil(D / ncols))

        fig, axs = plt.subplots(
            nrows, ncols, figsize=(ncols * 3.47412, nrows * 3.47412 * 0.8)
        )
        ax_flat = np.array(axs).reshape(-1)

        tail_cut_low = self.metadata["tail_cut_low"].values[0]
        tail_cut_high = self.metadata["tail_cut_high"].values[0]

        # tail data
        ml_tail_low = self.ml_test_data[self.r_x_ml < tail_cut_low]
        ml_tail_high = self.ml_test_data[self.r_x_ml > tail_cut_high]

        for feat_idx in range(D):
            ax = ax_flat[feat_idx]

            mc_data = self.mc_test_data[:, feat_idx]
            bin_edges = np.histogram_bin_edges(mc_data, bins=50)

            # MC plot
            ax.hist(
                mc_data,
                bins=bin_edges,
                density=True,
                histtype="bar",
                color="gray",
                alpha=0.4,
                label="MC",
                linewidth=0,
            )
            # ax.hist(mc_data, bins=bin_edges, density=True,
            #        histtype='step', color='black', alpha=0.8, linewidth=1.5)

            # ML tails plot
            if len(ml_tail_low) > 0:
                ax.hist(
                    ml_tail_low[:, feat_idx],
                    bins=bin_edges,
                    density=True,
                    histtype="step",
                    color=set1_list[1],
                    lw=1.5,
                    label=f"ML tail cut $< {tail_cut_low}$",
                    alpha=0.8,
                )

            if len(ml_tail_high) > 0:
                ax.hist(
                    ml_tail_high[:, feat_idx],
                    bins=bin_edges,
                    density=True,
                    histtype="step",
                    color=set1_list[0],
                    lw=1.5,
                    label=f"ML tail cut $> {tail_cut_high}$",
                    alpha=0.8,
                )

            # ax.set_yscale('log')
            ax.set_xlabel(features_list[feat_idx], fontsize=9)
            ax.set_ylabel("Density", fontsize=9)

            if feat_idx == 0:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

        for j in range(D, len(ax_flat)):
            ax_flat[j].axis("off")

        plt.tight_layout(pad=0.5)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved Figure 15 to: {save_path}")

        plt.show()

    def print_summary(self):
        meta = self.metadata.iloc[0]

        print(f"\ntail cuts:")
        print(f"  L: {meta['tail_cut_low']:.3f}")
        print(f"  U: {meta['tail_cut_high']:.3f}")

        print(f"\nfraction:")
        print(f"  MC: {meta['mc_tail_frac']*100:.2f}%")
        print(f"  ML: {meta['ml_tail_frac']*100:.2f}%")

        print(f"\ntail counts:")
        print(f"  MC: {np.sum(self.tail_mask_mc):,} / {len(self.tail_mask_mc):,}")
        print(f"  ML: {np.sum(self.tail_mask_ml):,} / {len(self.tail_mask_ml):,}")


if __name__ == "__main__":
    data_dir = Path(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\ratiotest")

    plotter = DensityRatio_plot(data_dir)

    plotter.print_summary()

    output_dir = Path(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\plots")
    output_dir.mkdir(exist_ok=True, parents=True)

    plotter.hist_dr_plot(save_path=output_dir / "density_ratioDDPM.png")

    plotter.tail_distributions_plot(save_path=output_dir / "tail_kinematicsDDPM.png")

    print(f"\nAll plots saved to: {output_dir}")
