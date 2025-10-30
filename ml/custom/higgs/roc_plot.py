import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import auc

plt.rcParams.update(
    {"text.usetex": True, "font.family": "Helvetica", "font.size": 8}
)
roc_data_list = {
        # "UNet DDPM S":  r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_unet1d_ddpm_model_c2st_gen_model_all_v2.csv",
        # "UNet DDPM XL": r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_unet1d_ddpm_model_c2st_gen_model_all_v6.csv",
        # "EDM no EMA": r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_unet1d_EDMnoEMA_model_c2st_gen_model_all_v1.csv",
        # "EDM2":       r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_unet1d_EDM2_model_c2st_gen_model_all_v1.csv",
        # "EDM simple": r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_unet1dconv_EDMsimple_model_c2st_gen_model_all_v1.csv",
        # "EDM raw":    r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_unet1d_EDMraw_model_c2st_gen_model_all_v1.csv",
        # "EDM simple": r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_unet1d_EDMraw_model_c2st_gen_model_all_v2.csv",
        # "VP":         r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_unet1d_VP_model_c2st_gen_model_all_v3.csv",
        # "VE":         r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_MPtinyunet_VE_model_c2st_gen_model_all_v3.csv",
        # "VP conv":    r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\data\c2st\roc_data_BinaryClassifier_unet1dconv_VP_model_c2st_gen_model_all_v1.csv",
    }
    
set1_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#f781bf", "#999999"]
auc_values = {}
r = 1
fig, ax = plt.subplots(figsize=(r * 3.47412, r * 1.1 * 3.47412))

for idx, (model_name, csv_path) in enumerate(roc_data_list.items()):
    df = pd.read_csv(csv_path)
    
    roc_auc = auc(df['fpr'], df['tpr'])
    auc_values[model_name] = roc_auc
    
    plt.plot(df['fpr'], df['tpr'], 
             color=set1_list[idx % len(set1_list)],
             linewidth=2.5,
             label=f'{model_name} (AUC: {roc_auc:.4f})',
             alpha=0.8)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Baseline', alpha=0.5)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_dir = Path(r"C:\Users\Uporabnik\Documents\IJS-F9\korlz\ppt\plots")
output_path = output_dir / "roc_curves_comparison_VPVE.pdf"
plt.savefig(output_path, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

plt.show()


