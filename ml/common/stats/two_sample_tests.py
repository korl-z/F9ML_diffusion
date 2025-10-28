import copy

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import chi2, ks_2samp, kstwo

from ml.common.utils.plot_utils import add_data_mc_ratio


def chisquare_2samp(h_A, h_B):
    c_1, c_2 = np.sqrt(np.sum(h_B) / np.sum(h_A)), np.sqrt(np.sum(h_A) / np.sum(h_B))
    chi2 = np.sum((c_1 * h_A - c_2 * h_B) ** 2 / (h_A + h_B))
    return chi2


def make_histograms(A, B, n_bins, bin_range=None):
    n_features = A.shape[1]

    combined_sample = np.concatenate([A, B])

    histograms_A, histograms_B = [], []
    for feature in range(n_features):
        bin_edges = np.histogram_bin_edges(combined_sample[:, feature], bins=n_bins, range=bin_range)

        h_A, _ = np.histogram(A[:, feature], bins=bin_edges)
        h_B, _ = np.histogram(B[:, feature], bins=bin_edges)

        idx_A, idx_B = np.where(h_A == 0)[0], np.where(h_B == 0)[0]
        zero_bins = list(set(idx_A) & set(idx_B))
        h_A, h_B = np.delete(h_A, zero_bins), np.delete(h_B, zero_bins)

        histograms_A.append(h_A)
        histograms_B.append(h_B)

    return histograms_A, histograms_B


def chi2_twosample_test(A, B, n_bins="auto", alpha=0.05, return_histograms=False, bin_range=None, do_copy=True):
    assert A.shape[1] == B.shape[1]

    if do_copy:
        A, B = copy.deepcopy(A), copy.deepcopy(B)

    if torch.is_tensor(A):
        A = A.cpu().numpy()

    if torch.is_tensor(B):
        B = B.cpu().numpy()

    histograms_A, histograms_B = make_histograms(A, B, n_bins, bin_range=bin_range)

    test_results = []
    for h_A, h_B in zip(histograms_A, histograms_B):
        test_result = chisquare_2samp(h_A, h_B)
        critical_value = chi2.ppf(1 - alpha, len(h_A) - 1)
        p_value = 1 - chi2.cdf(test_result, len(h_A) - 1)
        test_results.append({"chi2": test_result, "crit": critical_value, "p": p_value})

    if return_histograms:
        return parse_test_results(test_results), (histograms_A, histograms_B)
    else:
        return parse_test_results(test_results)


def ks_twosample_test(A, B, alpha=0.05, do_copy=True):
    assert A.shape[1] == B.shape[1]

    if do_copy:
        A, B = copy.deepcopy(A), copy.deepcopy(B)

    if torch.is_tensor(A):
        A = A.cpu().numpy()

    if torch.is_tensor(B):
        B = B.cpu().numpy()

    N, M = len(A), len(B)

    test_results = []
    for a, b in zip(A.T, B.T):
        test_result = ks_2samp(a, b)
        critical_value = kstwo.ppf(1 - alpha, N * M // (N + M))
        test_results.append({"ks": test_result[0], "crit": critical_value, "p": test_result[1]})

    return parse_test_results(test_results)


def parse_test_results(results):
    parsed = np.zeros((len(results), 3))
    statistic = list(set(results[0].keys()) ^ {"crit", "p"})[0]
    for i, res_dct in enumerate(results):
        parsed[i, 0] = res_dct[statistic]
        parsed[i, 1] = res_dct["crit"]
        parsed[i, 2] = res_dct["p"]

    df = pd.DataFrame(parsed, columns=[statistic, "crit", "p"])
    return df

features_list = [
    "lepton pT",
    "lepton eta",
    "missing energy",
    "jet1 pT",
    "jet1 eta",
    "jet2 pT",
    "jet2 eta",
    "jet3 pT",
    "jet3 eta",
    "jet4 pT",
    "jet4 eta",
    "m jj",
    "m jjj",
    "m lv",
    "m jlv",
    "m bb",
    "m wbb",
    "m wwbb",
]

def two_sample_plot(
    A,
    B,
    axs,
    n_bins=50,
    label=None,
    labels=features_list,
    log_scale=False,
    bin_range=None,
    xlim=None,
    ylim=None,
    titles=None,
    combine_for_bins=False,
    do_copy=True,
    **kwargs,
):
    assert A.shape[1] == B.shape[1], "A and B must have the same number of features!"

    A = A.cpu().numpy() if torch.is_tensor(A) else copy.deepcopy(A)
    B = B.cpu().numpy() if torch.is_tensor(B) else copy.deepcopy(B)
    
    n_features = A.shape[1]

    for feature in range(n_features):
        ax = axs[feature]

        # combined_sample = np.concatenate([A[:, feature], B[:, feature]])
        current_bin_range = bin_range[feature] if bin_range and len(bin_range) > feature else None
        # bin_edges = np.histogram_bin_edges(combined_sample, bins=n_bins, range=current_bin_range)
        bin_edges = np.histogram_bin_edges(A[:, feature], bins=n_bins, range=current_bin_range)

        sns.histplot(A[:, feature], bins=bin_edges, ax=ax, stat="density", color="gray", alpha=0.3, label="MC")
        ax.hist(B[:, feature], bins=bin_edges, density=True, histtype="step", lw=1.0, color="blue", label="ML")
        
        ax.set_yscale("log")
        ax.legend()
        if labels and len(labels) > feature:
            ax.set_xlabel(labels[feature])

        real_counts, _ = np.histogram(A[:, feature], bins=bin_edges)
        gen_counts, _ = np.histogram(B[:, feature], bins=bin_edges)

        sum_real = real_counts.sum()
        sum_gen = gen_counts.sum()

        if sum_gen == 0:
            gen_counts_scaled = np.full_like(gen_counts, 1e-8, dtype=float)
        else:
            scale_factor = float(sum_real) / float(sum_gen)
            gen_counts_scaled = gen_counts.astype(float) * scale_factor

        real_yerr = np.sqrt(real_counts.astype(float))
        gen_yerr = np.sqrt(gen_counts.astype(float)) * (scale_factor if sum_gen != 0 else 1.0)
        
        real_counts_safe = real_counts.astype(float).copy()
        zero_mask = real_counts_safe == 0
        if np.any(zero_mask):
            real_counts_safe[zero_mask] = 1e-8
            real_yerr[zero_mask] = 1e-8

        add_data_mc_ratio(
            ax=ax,
            bin_edges=bin_edges,
            data_hist=gen_counts_scaled.astype(float),
            data_yerr=gen_yerr.astype(float),
            mc_hists=real_counts_safe[None, :].astype(float),
            mc_yerrs=real_yerr[None, :].astype(float),
            ylim=(0.5, 1.5),
            lower_ylabel="ML / MC",
        )
    return axs
