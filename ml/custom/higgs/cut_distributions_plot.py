import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ml.common.data_utils.processors import Preprocessor
from ml.custom.higgs.process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)

plt.rcParams.update({"text.usetex": True, "font.size": 10})
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
    r"$m_{l\nu}$",
    r"$m_{jl\nu}$",
    r"$m_{bb}$",
    r"$m_{wbb}$",
    r"$m_{wwbb}$",
]


def prep_data():
    """prep and preproc data"""
    npy_proc = HIGGSNpyProcessor(
        data_dir="/data0/korlz/f9-ml/ml/data/HIGGS/",
        base_file_name="HIGGS_data",
        hold_mode=True,
        use_hold=True,
    )

    npy_file, feats = npy_proc()

    f_sel = HIGGSFeatureSelector(
        file_path=npy_file,
        features=feats,
        drop_types=[],
        on_train=None,
    )

    data_w_lbl, sel_w_lbl = f_sel()

    # split sig/bkg
    lbl_idx = sel_w_lbl[sel_w_lbl["feature"] == "label"].index[0]
    sig_msk = (data_w_lbl[:, lbl_idx] == 1).flatten()
    bkg_msk = (data_w_lbl[:, lbl_idx] == 0).flatten()

    sig_data_w_lbl = data_w_lbl[sig_msk]
    bkg_mc_data_w_lbl = data_w_lbl[bkg_msk]

    # remove labels
    lbl_msk = np.ones(sig_data_w_lbl.shape[1], dtype=bool)
    lbl_msk[lbl_idx] = False

    sig_data = sig_data_w_lbl[:, lbl_msk]
    bkg_mc_data = bkg_mc_data_w_lbl[:, lbl_msk]

    sel_no_lbl = sel_w_lbl[lbl_msk].reset_index(drop=True)

    # drop uni/disc
    keep_msk = ~sel_no_lbl["type"].isin(["uni", "disc"])
    sig_data = sig_data[:, keep_msk]
    bkg_mc_data = bkg_mc_data[:, keep_msk]
    sel_fin = sel_no_lbl[keep_msk].reset_index(drop=True)

    # preproc
    pre = Preprocessor(
        cont_rescale_type="gauss_rank",
        disc_rescale_type="none",
    )

    sig_scaled, _, _ = pre.preprocess(sig_data, sel_fin)
    bkg_mc_scaled, _, _ = pre.preprocess(bkg_mc_data, sel_fin)

    # load ML bkg
    ml_path = (
        "/data0/korlz/f9-ml/ml/data/HIGGS/HIGGS_generated_unet1d_ddpm_model_v6.npy"
    )
    bkg_ml_scaled = np.load(ml_path)

    return sig_scaled, bkg_mc_scaled, bkg_ml_scaled, sel_fin


def get_clf_scores(clf, sig, bkg_mc, bkg_ml, dev="cpu"):
    """get clf scores for all data"""

    def pred(data):
        tens = torch.FloatTensor(data).to(dev)
        with torch.no_grad():
            out = clf(tens)
            if out.min() < 0 or out.max() > 1:
                out = torch.sigmoid(out)
        return out.cpu().numpy().flatten()

    sig_sc = pred(sig)
    bkg_mc_sc = pred(bkg_mc)
    bkg_ml_sc = pred(bkg_ml)

    return sig_sc, bkg_mc_sc, bkg_ml_sc


def plot_fig18(sig, bkg_mc, bkg_ml, sel, cut=0.55, save_path=None):
    """plot fig 18 style distrs after clf cut"""

    feat_names = features_list
    n_feats = len(feat_names)

    D = n_feats
    ncols = 6
    nrows = int(np.ceil(D / ncols))

    fig1, axs = plt.subplots(
        nrows * 2,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        gridspec_kw={"height_ratios": [3, 1] * nrows},
    )

    fig1.subplots_adjust(
        hspace=0.55, wspace=0.2, left=0.03, right=0.98, top=0.97, bottom=0.1
    )

    bins = np.linspace(-4.0, 4.0, 50)
    ctrs = 0.5 * (bins[1:] + bins[:-1])

    for i in range(n_feats):
        r = (i // ncols) * 2
        c = i % ncols

        ax = axs[r, c]
        axr = axs[r + 1, c]

        sig_f = sig[:, i]
        bkg_mc_f = bkg_mc[:, i]
        bkg_ml_f = bkg_ml[:, i]

        #raw distributions (no inverse scaling)
        ax.hist(
            bkg_mc_f,
            bins=bins,
            histtype="step",
            lw=2,
            color=set1_list[0],
            label="bkg MC",
        )
        ax.hist(
            bkg_ml_f,
            bins=bins,
            histtype="step",
            lw=2,
            color=set1_list[1],
            label="bkg ML",
        )
        ax.hist(
            sig_f,
            bins=bins,
            histtype="step",
            lw=2,
            color=set1_list[2],
            label="sig MC",
        )

        ax.set_xlim(-4.0, 4.0)
        ax.set_ylabel(r"$N$")
        ax.tick_params(which="both", direction="in")

        if i == 0:
            ax.legend(fontsize=10, framealpha=0.9)

        # density ratio !!!!
        h_mc, _ = np.histogram(bkg_mc_f, bins=bins, density=True)
        h_ml, _ = np.histogram(bkg_ml_f, bins=bins, density=True)

        ratio = np.divide(
            h_ml,
            h_mc,
            out=np.zeros_like(h_ml),
            where=h_mc > 0,
        )

        axr.scatter(ctrs, ratio, color="k", s=7, marker="D")
        axr.axhline(1.0, color="k", lw=1.0, alpha=0.6)

        axr.set_xlim(-4.0, 4.0)
        axr.set_ylim(0.5, 1.5)
        axr.set_ylabel("ML/MC")
        axr.set_xlabel(feat_names[i])
        axr.grid(alpha=0.3)
        axr.tick_params(which="both", direction="in")

    plt.tight_layout(pad=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"saved: {save_path}")

    plt.close()


if __name__ == "__main__":
    sig_test, bkg_mc_test, bkg_ml_test, sel = prep_data()

    # load clf 
    import mlflow

    mlflow.set_tracking_uri("file:///data0/korlz/f9-ml/ml/custom/higgs/mlruns")

    clf = mlflow.pytorch.load_model(
        "models:/BinaryClassifier_full_sigbkg_gauss_rank_best6/1",  # ddpm
        # "models:/BinaryClassifier_sigbkg_gauss_rank_EDMxl/1", #EDM
        map_location="cpu",
    )
    clf.eval()

    sig_sc, bkg_mc_sc, bkg_ml_sc = get_clf_scores(
        clf, sig_test, bkg_mc_test, bkg_ml_test
    )

    #cut data
    cut_val = 0.55
    sig_cut = sig_test[sig_sc >= cut_val]
    bkg_mc_cut = bkg_mc_test[bkg_mc_sc >= cut_val]
    bkg_ml_cut = bkg_ml_test[bkg_ml_sc >= cut_val]

    # plot
    out_dir = Path("/data0/korlz/f9-ml/ml/custom/higgs/analysis/plots/sigbkg")
    out_dir.mkdir(exist_ok=True, parents=True)

    plot_fig18(
        sig_cut,
        bkg_mc_cut,
        bkg_ml_cut,
        sel,
        cut=cut_val,
        save_path=out_dir / "fig18_distrs_after_clf_cut_DDPM.png",
    )
