import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

# name mapping
distances = {
    "kl_divergence": "KL divergence",
    "hellinger_distance": "Hellinger distance",
    "chi2_distance": r"$\chi^2$ distance",
    "wasserstein_distance": "Wasserstein distance",
}

data_dir = Path(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\distances")

fig, axs = plt.subplots(2, 2, figsize=(2 * 3.47412, 2 * 3.47412 * 1))
axs = axs.flatten()

for ax_idx, (distance_key, distance_label) in enumerate(distances.items()):
    ax = axs[ax_idx]

    csv_path = data_dir / f"{distance_key}_data2.csv"
    df = pd.read_csv(csv_path)

    features = df["feature"].values
    baseline = df["baseline"].values

    model_columns = [col for col in df.columns if col not in ["feature", "baseline"]]

    model_names = list(
        set([col.replace("_mean", "").replace("_std", "") for col in model_columns])
    )
    model_names = sorted(model_names)

    color_idx = 0
    for model_name in model_names:
        mean_col = f"{model_name}_mean"
        std_col = f"{model_name}_std"

        if mean_col in df.columns:
            mean = df[mean_col].values
            std = df[std_col].values if std_col in df.columns else np.zeros_like(mean)

            ax.plot(
                mean,
                label=model_name,
                color=set1_list[color_idx % len(set1_list)],
                zorder=0,
                ls="-",
                lw=1.5,
            )
            ax.fill_between(
                np.arange(len(mean)),
                mean - std,
                mean + std,
                alpha=0.2,
                color=set1_list[color_idx % len(set1_list)],
            )
            color_idx += 1

    ax.plot(baseline, label="Baseline", color="k", zorder=10, ls="--", lw=1.5)

    ax.set_yscale("log")

    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=8)

    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel(distance_label, fontsize=10)

    handles, labels = axs[0].get_legend_handles_labels()

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

data_dirout = r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\plots"
output_path = data_dir / "distances_combined2.png"
plt.savefig(output_path, bbox_inches="tight", dpi=300)
print(f"Plot saved to: {output_path}")

plt.show()
