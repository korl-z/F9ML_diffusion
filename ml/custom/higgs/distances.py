import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

from ml.common.stats.distances import Distances
from ml.common.utils.loggers import setup_logger
from ml.common.utils.picklers import mkdir
from ml.common.utils.plot_utils import set_size, style_setup
from ml.custom.higgs.analysis.utils import (
    LABELS_MAP,
    MODEL_MAP,
    equalize_counts_to_ref,
    get_model,
    get_scaler,
    rescale_back,
    run_chainer,
    sample_from_models,
)


class DistancesTest:
    distances = {
        "kl_divergence": "KL divergence",
        "hellinger_distance": "Hellinger distance",
        "chi2_distance": r"$\chi^2$ distance",
        "wasserstein_distance": "Wasserstein distance",
    }

    def __init__(
        self,
        select_models,
        cont_rescale_type,
        N,
        resample,
        save_dir="/data0/korlz/f9-ml/ml/custom/higgs/analysis/plots/distances",
        versions=None,
    ):
        self.select_models = select_models
        self.cont_rescale_type = cont_rescale_type
        self.N = N
        self.resample = resample
        self.save_dir = save_dir

        if versions is None:
            self.versions = [-1] * len(select_models)
        else:
            self.versions = versions

        mkdir(save_dir)

        self.samples = self.setup()

        self.d_results = None

    def setup(self):
        ref_data, selection, _ = run_chainer(
            n_data=-1,
            return_data=True,
            cont_rescale_type="gauss_rank",
            use_hold=True,
        )

        label_idx = selection[selection["feature"] == "label"].index

        # analysis on background data, so remove label
        bkg_mask = (ref_data[:, label_idx] == 0).flatten()
        bkg_ref_data = ref_data[bkg_mask]

        label_mask = np.ones(bkg_ref_data.shape[1], dtype=bool)
        label_mask[label_idx] = False
        bkg_ref_data = bkg_ref_data[:, label_mask]

        # filter models
        model_dct = {
            f"{select_model}{ver}": get_model(select_model, ver=ver).eval()
            for select_model, ver in zip(self.select_models, self.versions)
        }

        for k, v in model_dct.items():
            if "-1" in k:
                model_dct[k.replace("-1", "")] = v

        # fetch scalers
        scalers_dct = {
            f"{select_model}{ver}": get_scaler(select_model, ver=ver)
            for select_model, ver in zip(self.select_models, self.versions)
        }

        for k, v in scalers_dct.items():
            if "-1" in k:
                scalers_dct[k.replace("-1", "")] = v

        samples = sample_from_models(model_dct, self.N, ver=self.versions, chunks=10, resample=self.resample)

        # need twice as much data for baseline
        bkg_ref_data = bkg_ref_data[: int(4 * self.N), :]

        # rescale back
        samples = rescale_back(samples, scalers_dct, selection)

        first_scaler = list(scalers_dct.values())[0]

        # Create a dict for rescale_back function
        ref_samples_dict = {"ref": [bkg_ref_data]}
        ref_samples_rescaled = rescale_back(ref_samples_dict, {"ref": first_scaler}, selection)
        bkg_ref_data = ref_samples_rescaled["ref"][0]

        # add reference data to samples
        samples["ref"] = [bkg_ref_data]

        # make sure all samples have the same number of events
        samples = equalize_counts_to_ref(samples)

        # CHECK: Are values in physical range?
        print("\n=== Value ranges after rescale ===")
        print(f"Ref min/max: {bkg_ref_data.min():.3f} / {bkg_ref_data.max():.3f}")
        for key, val in samples.items():
            if key != 'ref':
                print(f"{key} min/max: {val[0].min():.3f} / {val[0].max():.3f}")

        return samples

    def _reduce_distances_dct(self, d_results):
        for model_name, distances in d_results.items():
            for d_name, d_values in distances.items():
                d_results[model_name][d_name] = [np.mean(d_values, axis=0), np.std(d_values, axis=0)]
        return d_results

    def _get_baseline(self, x, n_bins, **kwargs):
        n, distances = len(x), {d_name: [] for d_name in self.distances.keys()}

        # no resampling for baseline
        # split data in half and calculate distances -> "baseline"
        for d_name in distances.keys():
            obj = Distances(
                x[: n // 2, :],      # First half
                x[n // 2 :, :],      # Second half
                mean_reduction=False,
                reduce=True,
                density_kwargs={"n_bins": n_bins},
                **kwargs,
            )
            func = getattr(obj, d_name)
            distances[d_name].append(func())

        return distances

    def get_model_distances(self, n_bins=50, **kwargs):

        logging.info("[red]Will set P to be reference data and Q to be generated data.[/red]")

        x = self.samples["ref"][0]

        n, d_results = len(x), dict()
        for model_name in tqdm(self.samples.keys(), leave=False, desc="Calculating distances"):
            if model_name == "ref":
                continue

            y = self.samples[model_name]

            distances = {d_name: [] for d_name in self.distances.keys()}

            for d_name in distances.keys():
                for y_i in y:  # iterate over resampled data
                    obj = Distances(
                        x[: n // 2, :], 
                        y_i,             # Generated samples
                        mean_reduction=False,
                        reduce=True,
                        density_kwargs={"n_bins": n_bins},
                        **kwargs,
                    )
                    func = getattr(obj, d_name)
                    distances[d_name].append(func())

            d_results[model_name] = distances

        d_results["baseline"] = self._get_baseline(x, n_bins=n_bins, **kwargs)

        # get mean and std
        self.d_results = self._reduce_distances_dct(d_results)

        return self.d_results

    def plot_model_distances(self, log_scale=True, **kwargs):
        if self.d_results is None:
            self.get_model_distances(**kwargs)

        model_names = list(self.d_results.keys())

        model_map = copy.deepcopy(MODEL_MAP)
        model_map["baseline"] = "Baseline"

        for distance_name in self.distances.keys():
            mean_base, _ = self.d_results["baseline"][distance_name]

            csv_data = {"feature": list(LABELS_MAP.values()), "baseline": mean_base}

            for i, model_name in enumerate(model_names):
                if model_name == "baseline":
                    continue

                mean, std = self.d_results[model_name][distance_name]

                idx = np.argsort(np.abs(mean - mean_base))
                mean, std = mean[idx], std[idx]

                csv_data[f"{model_map[model_name]}_mean"] = mean
                csv_data[f"{model_map[model_name]}_std"] = std

                plt.plot(mean, label=model_map[model_name], color=f"C{i}", zorder=0, ls="-")
                plt.fill_between(
                    np.arange(len(mean)),
                    mean - std,
                    mean + std,
                    alpha=0.3,
                )
                plt.xticks(np.arange(len(mean)), np.array(list(LABELS_MAP.values()), dtype=object)[idx], rotation=90)

            plt.plot(mean_base[idx], label="baseline", color="k", zorder=10, ls="--")

            if log_scale:
                plt.yscale("log")

            csv_data["feature"] = np.array(list(LABELS_MAP.values()), dtype=object)[idx]
            csv_data["baseline"] = mean_base[idx]
            df = pd.DataFrame(csv_data)
            csv_path = f"{self.save_dir}/{distance_name}_data.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"[blue]Saved distance data to {csv_path}[/blue]")

            logging.info(f"[green]Saving distance plot for {distance_name}.[/green]")


            plt.legend(ncol=4)
            plt.xlabel("Feature")
            plt.ylabel(self.distances[distance_name])
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/{distance_name}.pdf")
            plt.close()


if __name__ == "__main__":
    setup_logger()
    set_size()
    style_setup()

    select_models = [
        "unet1d_ddpm_model",
        "unet1d_ddpm_model",

        "unet1d_VP_model",
        "MPtinyunet_VE_model",
        "unet1d_EDM2_model",
        "unet1d_EDMnoEMA_model",

        # "unet1d_EDMraw_model",
        # "unet1d_EDMraw_model",
        
        # "unet1dconv_EDMsimple_model",
        "unet1dconv_VP_model",
    ]
    versions = [
                2, 6,
                3, 3, 1, 1, #v1 EDM noEMA
                # 1, 2, #no. 2 is EDMsimple
                # 1, 
                1 #1dconv models
                ]

    distances_test = DistancesTest(
        select_models=select_models,
        versions=versions,
        cont_rescale_type="gauss_rank",
        N=1000000,
        resample=1,
    )

    distances_test.plot_model_distances(log_scale=True, n_bins=50)
