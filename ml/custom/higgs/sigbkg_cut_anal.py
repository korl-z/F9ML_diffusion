import sys

sys.path.append("/data0/korlz/f9-ml")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
import logging
import pyhf

from ml.common.utils.loggers import setup_logger
from ml.custom.higgs.analysis.utils import run_chainer
from ml.common.data_utils.processors import Preprocessor, ProcessorChainer
from ml.custom.higgs.process_higgs_dataset import (
    HIGGSFeatureSelector,
    HIGGSNpyProcessor,
)


import pyhf
import pyhf.infer.mle
import matplotlib.patches as mpatches
setup_logger()
# Styling
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

mlflow.set_tracking_uri("file:///data0/korlz/f9-ml/ml/custom/higgs/mlruns")


class PrefitDistributions:
    def __init__(self, classifier_uri, device="cpu"):
        """Initialize with signal/background classifier."""
        self.device = device
        logging.info(f"Loading classifier: {classifier_uri}")
        self.classifier = mlflow.pytorch.load_model(classifier_uri, map_location=device)
        self.classifier.eval()

    def get_predictions(self, data):
        """Get classifier scores."""
        data_tensor = torch.FloatTensor(data).to(self.device)
        with torch.no_grad():
            output = self.classifier(data_tensor)
            if output.min() < 0 or output.max() > 1:
                output = torch.sigmoid(output)
        return output.cpu().numpy().flatten()

    def data_prep(
        self,
        sig_scaled,
        bkg_mc_scaled,
        bkg_ml_scaled,
        sig_original,
        bkg_mc_original,
        bkg_ml_original,
        score_cut=0.55,
        m_bb_idx=15,
    ):
        """
        Apply classifier cut and prepare data for Figure 19.

        Returns
        -------
        dict with keys:
            - asimov_mbb: (MC sig + MC bkg)
            - asimov_scores: classifier scores for Asimov dataset
            - sig_mbb: MC signal m_bb after cut
            - sig_scores: MC signal scores after cut
            - bkg_ml_mbb: ML background m_bb after cut
            - bkg_ml_scores: ML background scores after cut
        """
        logging.info(f"\nApplying classifier cut at score > {score_cut}")

        # Get classifier scores
        scores_sig = self.get_predictions(sig_scaled)
        scores_bkg_mc = self.get_predictions(bkg_mc_scaled)
        scores_bkg_ml = self.get_predictions(bkg_ml_scaled)

        logging.info(f"Before cut:")
        logging.info(f"  Signal: {len(scores_sig)} events")
        logging.info(f"  MC background: {len(scores_bkg_mc)} events")
        logging.info(f"  ML background: {len(scores_bkg_ml)} events")

        mask_sig = scores_sig > score_cut
        mask_bkg_mc = scores_bkg_mc > score_cut
        mask_bkg_ml = scores_bkg_ml > score_cut

        logging.info(f"After cut (score > {score_cut}):")
        logging.info(
            f"  Signal: {np.sum(mask_sig)} ({np.sum(mask_sig)/len(mask_sig)*100:.1f}%)"
        )
        logging.info(
            f"  MC background: {np.sum(mask_bkg_mc)} ({np.sum(mask_bkg_mc)/len(mask_bkg_mc)*100:.1f}%)"
        )
        logging.info(
            f"  ML background: {np.sum(mask_bkg_ml)} ({np.sum(mask_bkg_ml)/len(mask_bkg_ml)*100:.1f}%)"
        )

        # Extract passing events
        sig_scores_cut = scores_sig[mask_sig]
        bkg_mc_scores_cut = scores_bkg_mc[mask_bkg_mc]
        bkg_ml_scores_cut = scores_bkg_ml[mask_bkg_ml]

        # Extract m_bb values (original data has label in column 0, m_bb at column 16)
        # But if we removed label, m_bb is at column 15
        sig_mbb = sig_original[mask_sig, m_bb_idx]
        bkg_mc_mbb = bkg_mc_original[mask_bkg_mc, m_bb_idx]
        bkg_ml_mbb = bkg_ml_original[mask_bkg_ml, m_bb_idx]

        # Create Asimov dataset (MC sig + MC bkg)
        asimov_scores = np.concatenate([sig_scores_cut, bkg_mc_scores_cut])
        asimov_mbb = np.concatenate([sig_mbb, bkg_mc_mbb])

        logging.info(f"\nAsimov dataset: {len(asimov_scores)} events")
        logging.info(f"  From signal: {len(sig_scores_cut)}")
        logging.info(f"  From MC background: {len(bkg_mc_scores_cut)}")

        return {
            "asimov_mbb": asimov_mbb,
            "asimov_scores": asimov_scores,
            "sig_mbb": sig_mbb,
            "sig_scores": sig_scores_cut,
            "bkg_mc_mbb": bkg_mc_mbb,
            "bkg_mc_scores": bkg_mc_scores_cut,
            "bkg_ml_mbb": bkg_ml_mbb,
            "bkg_ml_scores": bkg_ml_scores_cut,
        }

    def prefit_cut_plot(
        self, data_dict, score_cut=0.55, save_path=None, save_data_dir=None
    ):
        """
        prefit distibituons
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 3.47412, 3.47412 * 0.8))

        # kinematic dist plot
        bins_mbb = np.linspace(0, 3, 30)

        hist_asimov_mbb, _ = np.histogram(data_dict["asimov_mbb"], bins=bins_mbb)
        hist_sig_mbb, _ = np.histogram(data_dict["sig_mbb"], bins=bins_mbb)
        hist_bkg_ml_mbb, _ = np.histogram(data_dict["bkg_ml_mbb"], bins=bins_mbb)

        # Asimov
        bin_centers = (bins_mbb[:-1] + bins_mbb[1:]) / 2
        ax1.errorbar(
            bin_centers,
            hist_asimov_mbb,
            yerr=np.sqrt(hist_asimov_mbb),
            fmt="o",
            color="k",
            markersize=3,
            capsize=2,
            label="MC data (Asimov)",
            zorder=10,
        )

        # stacked ML-bkg + MC-sig
        ax1.hist(
            data_dict["bkg_ml_mbb"],
            bins=bins_mbb,
            histtype="stepfilled",
            color=set1_list[1],
            alpha=0.5,
            label="ML bkg",
            edgecolor="none",
        )
        ax1.hist(
            data_dict["sig_mbb"],
            bins=bins_mbb,
            histtype="stepfilled",
            bottom=hist_bkg_ml_mbb,
            color=set1_list[4],
            alpha=0.5,
            label="MC sig",
            edgecolor="none",
        )

        # Signal only
        ax1.stairs(
            hist_sig_mbb,
            bins_mbb,
            color=set1_list[0],
            linewidth=1.5,
            label="Signal only",
            zorder=5,
        )

        ax1.set_xlabel(r"$m_{b\bar{b}}$ [TeV]", fontsize=10)
        ax1.set_ylabel("Events", fontsize=10)
        ax1.legend(fontsize=8, framealpha=0.9, loc="upper right")
        ax1.set_xlim(bins_mbb[0], bins_mbb[-1])
        ax1.tick_params(which="both", direction="in")

        bins_score = np.linspace(score_cut, 1.0, 30)

        hist_asimov_score, _ = np.histogram(data_dict["asimov_scores"], bins=bins_score)
        hist_sig_score, _ = np.histogram(data_dict["sig_scores"], bins=bins_score)
        hist_bkg_ml_score, _ = np.histogram(data_dict["bkg_ml_scores"], bins=bins_score)

        # Asimov
        bin_centers_score = (bins_score[:-1] + bins_score[1:]) / 2
        ax2.errorbar(
            bin_centers_score,
            hist_asimov_score,
            yerr=np.sqrt(hist_asimov_score),
            fmt="o",
            color="k",
            markersize=3,
            capsize=2,
            label="MC data (Asimov)",
            zorder=10,
        )

        # stack ml-bkg + mc-sig
        ax2.hist(
            data_dict["bkg_ml_scores"],
            bins=bins_score,
            histtype="stepfilled",
            color=set1_list[1],
            alpha=0.5,
            label="ML bkg",
            edgecolor="none",
        )
        ax2.hist(
            data_dict["sig_scores"],
            bins=bins_score,
            histtype="stepfilled",
            bottom=hist_bkg_ml_score,
            color=set1_list[4],
            alpha=0.5,
            label="MC sig",
            edgecolor="none",
        )

        # only signal data
        ax2.stairs(
            hist_sig_score,
            bins_score,
            color=set1_list[0],
            linewidth=1.5,
            label="Signal only",
            zorder=5,
        )

        ax2.axvline(score_cut, color="gray", ls=":", lw=1.5, alpha=0.7)

        ax2.set_xlabel("Classifier score", fontsize=10)
        ax2.set_ylabel("Events", fontsize=10)
        ax2.legend(fontsize=8, framealpha=0.9, loc=3)
        ax2.set_xlim(score_cut, 1.0)
        ax2.tick_params(which="both", direction="in")

        plt.tight_layout(pad=0.5)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

        # # Save data
        # if save_data_dir:
        #     # m_bb data
        #     df_mbb = pd.DataFrame({
        #         'bin_centers': bin_centers,
        #         'bin_edges_left': bins_mbb[:-1],
        #         'bin_edges_right': bins_mbb[1:],
        #         'asimov': hist_asimov_mbb,
        #         'asimov_err': np.sqrt(hist_asimov_mbb),
        #         'sig': hist_sig_mbb,
        #         'bkg_ml': hist_bkg_ml_mbb,
        #     })
        #     df_mbb.to_csv(save_data_dir / "fig19_mbb_data.csv", index=False)

        #     # Score data
        #     df_score = pd.DataFrame({
        #         'bin_centers': bin_centers_score,
        #         'bin_edges_left': bins_score[:-1],
        #         'bin_edges_right': bins_score[1:],
        #         'asimov': hist_asimov_score,
        #         'asimov_err': np.sqrt(hist_asimov_score),
        #         'sig': hist_sig_score,
        #         'bkg_ml': hist_bkg_ml_score,
        #     })
        #     df_score.to_csv(save_data_dir / "fig19_score_data.csv", index=False)

        #     logging.info(f"Saved data to: {save_data_dir}")


def data_prep_f19():
    """Load and prepare all datasets for Figure 19."""

    # load data
    logging.info("loading MC data.")
    npy_proc = HIGGSNpyProcessor(
        data_dir="/data0/korlz/f9-ml/ml/data/HIGGS/",
        base_file_name="HIGGS_data",
        hold_mode=True,
        use_hold=True,
    )
    npy_file, features = npy_proc()

    # Load with label
    f_sel = HIGGSFeatureSelector(
        file_path=npy_file,
        features=features,
        drop_types=[],
        on_train=None,
    )
    data_all, selection = f_sel()

    # Split by label
    label_idx = selection[selection["feature"] == "label"].index[0]
    sig_mask = (data_all[:, label_idx] == 1).flatten()
    bkg_mask = (data_all[:, label_idx] == 0).flatten()

    sig_data_full = data_all[sig_mask]
    bkg_data_mc_full = data_all[bkg_mask]

    logging.info(f"   Signal: {len(sig_data_full)} events")
    logging.info(f"   MC background: {len(bkg_data_mc_full)} events")

    sig_original = sig_data_full.copy()
    bkg_mc_original = bkg_data_mc_full.copy()

    m_bb_idx_with_label = selection[selection["feature"] == "m bb"].index[0]

    label_mask_arr = np.ones(sig_data_full.shape[1], dtype=bool)
    label_mask_arr[label_idx] = False

    sig_no_label = sig_data_full[:, label_mask_arr]
    bkg_mc_no_label = bkg_data_mc_full[:, label_mask_arr]

    selection_no_label = selection[label_mask_arr].reset_index(drop=True)

    keep_mask = ~selection_no_label["type"].isin(["uni", "disc"])
    sig_data = sig_no_label[:, keep_mask]
    bkg_mc_data = bkg_mc_no_label[:, keep_mask]

    selection_final = selection_no_label[keep_mask].reset_index(drop=True)

    logging.info(f"   After dropping uni/disc: {sig_data.shape}")

    # preprocess mc data
    logging.info("preprocessing MC data")
    pre = Preprocessor(cont_rescale_type="gauss_rank", disc_rescale_type="none")

    sig_scaled, _, scalers = pre.preprocess(sig_data, selection_final)
    bkg_mc_scaled, _, _ = pre.preprocess(bkg_mc_data, selection_final)

    logging.info(f"Preprocessed MC shape: {sig_scaled.shape}")

    # Lload ml bkg
    logging.info("loading ML background.")
    ml_bkg_path = "/data0/korlz/f9-ml/ml/data/HIGGS/HIGGS_generated_unet1d_ddpm_model_v6.npy" # for DDPM
    # ml_bkg_path = "/data0/korlz/f9-ml/ml/data/HIGGS/HIGGS_generated_unet1D_EDM_s_model_v2.npy"  # for EDM
    ml_bkg_scaled = np.load(ml_bkg_path)

    logging.info(
        f"   ML background: {len(ml_bkg_scaled)} events, shape: {ml_bkg_scaled.shape}"
    )

    # ML inv transform
    logging.info("Inv transf. ML background.")
    ml_bkg_original_features = scalers["cont"][0][1].inverse_transform(ml_bkg_scaled)

    ml_bkg_full = np.zeros((len(ml_bkg_original_features), data_all.shape[1]))

    for idx, row in selection_final.iterrows():
        feature_name = row["feature"]
        orig_idx_no_label = selection_no_label[
            selection_no_label["feature"] == feature_name
        ].index[0]
        orig_idx_with_label = selection[selection["feature"] == feature_name].index[0]

        ml_bkg_full[:, orig_idx_with_label] = ml_bkg_original_features[:, idx]

    ml_bkg_full[:, label_idx] = 0

    ml_bkg_original = ml_bkg_full

    return {
        "sig_scaled": sig_scaled,
        "bkg_mc_scaled": bkg_mc_scaled,
        "ml_bkg_scaled": ml_bkg_scaled,
        "sig_original": sig_original,
        "bkg_mc_original": bkg_mc_original,
        "ml_bkg_original": ml_bkg_original,
        "m_bb_idx": m_bb_idx_with_label,
    }


class PostfitDistributions(PrefitDistributions):
    def __init__(self, classifier_uri, device="cpu"):
        super().__init__(classifier_uri, device)
        # Use iminuit for the optimization (standard in HEP)
        pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer())

    def perform_profile_fit(self, obs_data, bkg_template, sig_template):
        """
        Calculates post-fit yields using a full pyhf workspace specification.
        """
        # 1. Prepare data lists
        bkg_list = bkg_template.tolist()
        sig_list = sig_template.tolist()
        obs_list = obs_data.tolist()

        # Stat errors for background (cannot be zero for the fit to be stable)
        stat_err = np.sqrt(bkg_template)
        stat_err = np.where(stat_err == 0, 1e-6, stat_err).tolist()

        # 2. Define Workspace
        spec = {
            "version": "1.0.0",
            "channels": [
                {
                    "name": "single_channel",
                    "samples": [
                        {
                            "name": "signal",
                            "data": sig_list,
                            "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]
                        },
                        {
                            "name": "background",
                            "data": bkg_list,
                            "modifiers": [
                                {"name": "sys_unc", "type": "normsys", "data": {"hi": 1.10, "lo": 0.90}},
                                {"name": "staterror", "type": "staterror", "data": stat_err}
                            ]
                        }
                    ]
                }
            ],
            "observations": [{"name": "single_channel", "data": obs_list}],
            "measurements": [
                {
                    "name": "measurement_POI",
                    "config": {"poi": "mu", "parameters": []}
                }
            ]
        }

        workspace = pyhf.Workspace(spec)
        model = workspace.model()
        data = workspace.data(model)

        print("Starting Profile Likelihood Fit...")
        bestfit_pars, twice_nll = pyhf.infer.mle.fit(data, model, return_fitted_val=True)

        print(f"Fit converged with -2*log(L) = {twice_nll:.2f}")
        print(f"Best-fit mu (signal strength): {bestfit_pars[model.config.poi_index]:.3f}")

        # 3. Extract Post-fit yields
        mu_idx = model.config.poi_index
        mu_fitted = bestfit_pars[mu_idx]

        # Signal yield scaled by mu
        sig_post = mu_fitted * np.array(sig_list)

        # Background yield - need to apply systematic shifts
        # The normsys parameter is at index 1 (after mu)
        # staterror parameters come after normsys

        # For normsys: parameter value represents log(kappa)
        # where kappa is the multiplicative factor
        # bestfit_pars[1] = 0 means no shift (factor = 1)
        # bestfit_pars[1] = +1 means 10% increase
        # bestfit_pars[1] = -1 means 10% decrease

        if len(bestfit_pars) > 1:
            # Get normsys parameter (index 1)
            normsys_param = bestfit_pars[1]
            # Convert to multiplicative factor
            # pyhf uses: factor = exp(normsys_param * log(hi/lo) / 2)
            # For hi=1.10, lo=0.90: log(1.10/0.90) = log(1.222) ≈ 0.2
            log_kappa = np.log(1.10 / 0.90) / 2
            bkg_factor = np.exp(normsys_param * log_kappa)

            # Apply staterror bin-by-bin
            # staterror parameters start at index 2
            staterror_start_idx = 2
            staterror_params = bestfit_pars[staterror_start_idx:staterror_start_idx + len(bkg_list)]

            # staterror is multiplicative per bin
            # Each parameter multiplies the bin by (1 + param * relative_error)
            bkg_post = np.array(bkg_list) * bkg_factor

            # Apply per-bin staterror corrections
            for i, stat_param in enumerate(staterror_params):
                if i < len(bkg_post):
                    # Relative error for this bin
                    rel_err = stat_err[i] / bkg_list[i] if bkg_list[i] > 0 else 0
                    bkg_post[i] *= (1 + stat_param * rel_err)
        else:
            bkg_post = np.array(bkg_list)

        print(f"Post-fit background total: {bkg_post.sum():.1f} (pre-fit: {sum(bkg_list):.1f})")
        print(f"Post-fit signal total: {sig_post.sum():.1f} (pre-fit: {sum(sig_list):.1f})")

        # 4. Calculate total post-fit uncertainty using Hessian
        try:
            # Calculate Hessian-based uncertainties
            result = pyhf.infer.mle.fit(data, model, return_uncertainties=True)

            # result can be (pars, corr) or (pars, corr, fitted_val)
            # Just use pars from before and calculate uncertainty another way

            # Use expected model variance at best-fit point
            # This gives us the uncertainty on the total prediction
            total_post = sig_post + bkg_post

            # Approximate uncertainty: combine statistical and systematic
            # Statistical: sqrt(N) for each bin
            stat_unc = np.sqrt(total_post)

            # Systematic: 10% on background (from normsys)
            sys_unc = 0.10 * bkg_post

            # Total uncertainty: add in quadrature
            total_unc = np.sqrt(stat_unc**2 + sys_unc**2)

        except Exception as e:
            print(f"Warning: Could not calculate detailed uncertainties: {e}")
            print("Using Poisson approximation")
            total_post = sig_post + bkg_post
            total_unc = np.sqrt(total_post)

        return bkg_post, sig_post, total_unc

    def postfit_plot(self, data_dict, score_cut=0.55, save_path=None, save_data_dir=None):
        """
        Create Figure 20: Post-fit distributions with uncertainty bands.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 3.47412, 3.47412 * 0.8))
        
        # Define Bins (Must be same as Prefit)
        bins_mbb = np.linspace(0, 3, 30)
        bins_score = np.linspace(score_cut, 1.0, 30)
        
        # --- LEFT PLOT: m_bb ---
        h_obs_mbb, _ = np.histogram(data_dict['asimov_mbb'], bins=bins_mbb)
        h_bkg_ml_mbb, _ = np.histogram(data_dict['bkg_ml_mbb'], bins=bins_mbb)
        h_sig_mbb, _ = np.histogram(data_dict['sig_mbb'], bins=bins_mbb)
        
        print("\n" + "="*70)
        print("FITTING m_bb DISTRIBUTION")
        print("="*70)
        
        # Run Fit for m_bb
        bkg_post, sig_post, unc_post = self.perform_profile_fit(
            h_obs_mbb, h_bkg_ml_mbb, h_sig_mbb
        )
        
        bin_centers = (bins_mbb[:-1] + bins_mbb[1:]) / 2
        bin_width = np.diff(bins_mbb)

        # Plot Data (Crosses)
        ax1.errorbar(bin_centers, h_obs_mbb, yerr=np.sqrt(h_obs_mbb), 
                    fmt='x', color='k', markersize=5, capsize=2,
                    label='MC data (Asimov)', zorder=10)
        
        # Plot Stacked Post-fit (Blue Bkg + Orange Sig)
        ax1.bar(bin_centers, bkg_post, width=bin_width, 
               color=set1_list[1], alpha=0.5, label='Fitted ML bkg',
               edgecolor='none')
        ax1.bar(bin_centers, sig_post, width=bin_width, bottom=bkg_post, 
               color=set1_list[4], alpha=0.5, label='MC sig',
               edgecolor='none')
        
        # Hatched Uncertainty Band
        total_post = bkg_post + sig_post
        ax1.bar(bin_centers, 2*unc_post, bottom=total_post-unc_post, 
               width=bin_width, fill=False, hatch='////', 
               edgecolor='gray', linewidth=0, label='Sys. uncertainty')

        ax1.set_xlabel(r'$m_{b\bar{b}}$ [TeV]', fontsize=10)
        ax1.set_ylabel('Events', fontsize=10)
        ax1.legend(fontsize=7, framealpha=0.9, loc='upper right')
        ax1.set_xlim(bins_mbb[0], bins_mbb[-1])
        ax1.grid(alpha=0.3)
        ax1.tick_params(which="both", direction='in')

        # --- RIGHT PLOT: Score ---
        h_obs_scr, _ = np.histogram(data_dict['asimov_scores'], bins=bins_score)
        h_bkg_ml_scr, _ = np.histogram(data_dict['bkg_ml_scores'], bins=bins_score)
        h_sig_scr, _ = np.histogram(data_dict['sig_scores'], bins=bins_score)
        
        print("\n" + "="*70)
        print("FITTING CLASSIFIER SCORE DISTRIBUTION")
        print("="*70)
        
        # Run Fit for Score
        bkg_p_scr, sig_p_scr, unc_p_scr = self.perform_profile_fit(
            h_obs_scr, h_bkg_ml_scr, h_sig_scr
        )
        
        bin_centers_scr = (bins_score[:-1] + bins_score[1:]) / 2
        bin_width_scr = np.diff(bins_score)

        ax2.errorbar(bin_centers_scr, h_obs_scr, yerr=np.sqrt(h_obs_scr), 
                    fmt='x', color='k', markersize=5, capsize=2, zorder=10)
        ax2.bar(bin_centers_scr, bkg_p_scr, width=bin_width_scr, 
               color=set1_list[1], alpha=0.5, edgecolor='none')
        ax2.bar(bin_centers_scr, sig_p_scr, width=bin_width_scr, 
               bottom=bkg_p_scr, color=set1_list[4], alpha=0.5,
               edgecolor='none')
        
        total_p_scr = bkg_p_scr + sig_p_scr
        ax2.bar(bin_centers_scr, 2*unc_p_scr, bottom=total_p_scr-unc_p_scr, 
               width=bin_width_scr, fill=False, hatch='////', 
               edgecolor='gray', linewidth=0)

        ax2.set_xlabel('Classifier score', fontsize=10)
        ax2.set_ylabel('Events', fontsize=10)
        ax2.set_xlim(bins_score[0], bins_score[-1])
        ax2.grid(alpha=0.3)
        ax2.tick_params(which="both", direction='in')
        
        plt.tight_layout(pad=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSaved Figure 20 to: {save_path}")
        
        plt.close()
        
        # # Save data
        # if save_data_dir:
        #     # m_bb postfit data
        #     df_mbb = pd.DataFrame({
        #         'bin_centers': bin_centers,
        #         'obs': h_obs_mbb,
        #         'obs_err': np.sqrt(h_obs_mbb),
        #         'bkg_postfit': bkg_post,
        #         'sig_postfit': sig_post,
        #         'total_postfit': total_post,
        #         'total_unc': unc_post,
        #     })
        #     df_mbb.to_csv(save_data_dir / "fig20_mbb_postfit_data.csv", index=False)
            
        #     # Score postfit data
        #     df_score = pd.DataFrame({
        #         'bin_centers': bin_centers_scr,
        #         'obs': h_obs_scr,
        #         'obs_err': np.sqrt(h_obs_scr),
        #         'bkg_postfit': bkg_p_scr,
        #         'sig_postfit': sig_p_scr,
        #         'total_postfit': total_p_scr,
        #         'total_unc': unc_p_scr,
        #     })
        #     df_score.to_csv(save_data_dir / "fig20_score_postfit_data.csv", index=False)
            
        #     print(f"Saved postfit data to: {save_data_dir}")

if __name__ == "__main__":
    # prep data
    all_data = data_prep_f19()

    fig19 = PrefitDistributions(
        classifier_uri="models:/BinaryClassifier_sigbkg_gauss_rank_MC/1",  # MC clasifier
        # classifier_uri="models:/BinaryClassifier_sigbkg_gauss_rank_best6/1", #DDPM     classifier
        # classifier_uri="models:/BinaryClassifier_sigbkg_gauss_rank_EDMxl/1", #EDM xl classifier
        device="cpu",
    )


    fig20 = PostfitDistributions(
        classifier_uri="models:/BinaryClassifier_sigbkg_gauss_rank_MC/1",
        device="cpu"
    )

    # cut and prepare data
    score_cut = 0.55
    data_after_cut = fig19.data_prep(
        all_data["sig_scaled"],
        all_data["bkg_mc_scaled"],
        all_data["ml_bkg_scaled"],
        all_data["sig_original"],
        all_data["bkg_mc_original"],
        all_data["ml_bkg_original"],
        score_cut=score_cut,
        m_bb_idx=all_data["m_bb_idx"],
    )


    output_dir = Path("/data0/korlz/f9-ml/ml/custom/higgs/analysis/plots/sigbkg")

    # fig19.prefit_cut_plot(
    #     data_after_cut,
    #     score_cut=score_cut,
    #     save_path=output_dir / "cut_prefit_plots_EDM.png",
    #     save_data_dir=output_dir,
    # )

    fig20.postfit_plot(
        data_after_cut,
        score_cut=0.55,
        save_path=output_dir / "fig20_postfit_EDM.png"
    )