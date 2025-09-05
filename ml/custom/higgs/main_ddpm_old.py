import matplotlib.pyplot as plt
import torch
import seaborn as sns
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from korlz.ml.common.data_utils.data_modules import HiggsDataset
from ml.common.data_utils.syn_datacreator import create_custom_multidim_dataset
from ml.diffusion.ddpm.diffusers import DiffuserDDPMeps
from ml.diffusion.ddpm.model import (
    NoisePredictorUNet,
)
from ml.diffusion.ddpm.losses import LossDDPMNoise
from ml.diffusion.ddpm.samplers import SamplerNoise
from ml.common.utils.utils import EMA
from ml.common.utils.loggers import timeit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 512

D_syn = 3
datasetraw = create_custom_multidim_dataset(
    n_samples=20000, n_features=D_syn, label_random=True, signal_frac=0.5, seed=123
)
print("dataset shape:", datasetraw.shape) 
print(datasetraw[:5])

features_syn = [f"f{i}" for i in range(D_syn)]

# X_syn, labels_syn = datasetraw[:, :-1], datasetraw[:, -1][:, None]  #full data
train_data_syn, val_data_syn = train_test_split(datasetraw, train_size=0.8)

print(train_data_syn.shape, val_data_syn.shape)
train_dataset_syn = HiggsDataset(train_data_syn)
val_dataset_syn = HiggsDataset(val_data_syn)

batches_train = DataLoader(
    train_dataset_syn, num_workers=0, batch_size=batch_size, shuffle=True
)
batches_val = DataLoader(
    train_dataset_syn, num_workers=0, batch_size=batch_size, shuffle=True
)

X_syn, labels_syn = next(iter(batches_train))


TIME_STEPS = 1000
num_epochs = 20

diffuser2 = DiffuserDDPMeps(timesteps=TIME_STEPS, scheduler="cosine", device=device)
model2 = NoisePredictorUNet(
    data_dim=D_syn, base_dim=128, depth=2, time_emb_dim=32, timesteps=TIME_STEPS
)
loss_fn2 = LossDDPMNoise(model2, diffuser2)
sampler2 = SamplerNoise(model2, diffuser2)


@timeit(unit="min")
def train2(
    model, diffuser, train_dataloader, val_dataloader, device, max_epochs=10, lr=1e-3
):
    """
    Train a noise-predicting DDPM model (eps prediction).
    - model: NoisePredictorMLP
    - diffuser: DiffuserDDPMeps (schedule helper) already configured with timesteps/betas
    - train_dataloader, val_dataloader: yield (features, labels) but labels are unused here
    """
    model.to(device)
    diffuser.to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = LossDDPMNoise(model, diffuser)  # expects x0 when called
    ema = EMA(model, decay=0.9999, device=device, start_step=0)

    epoch_train_loss = []
    epoch_val_loss = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = []
        loop = tqdm(
            train_dataloader, leave=False, desc=f"Epoch {epoch+1}/{max_epochs} training"
        )
        for features_batch, labels_batch in loop:
            features_batch = features_batch.to(device)
            optimizer.zero_grad()
            loss = loss_func(features_batch)
            loss.backward()
            optimizer.step()
            ema.update(model)

            train_loss.append(loss.item())
            loop.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        epoch_avg = float(np.mean(train_loss)) if len(train_loss) > 0 else 0.0
        epoch_train_loss.append(epoch_avg)
        print(f"Epoch {epoch+1}/{max_epochs} avg_loss={epoch_avg:.4f}")


        model.eval()
        val_loss = []
        loop_val = tqdm(
            val_dataloader, leave=False, desc=f"Epoch {epoch+1}/{max_epochs} validation"
        )

        with torch.no_grad():
            for features_batch_val, labels_batch_val in loop_val:
                features_batch_val = features_batch_val.to(device)

                loss_val = loss_func(features_batch_val)
                val_loss.append(loss_val.item())
                loop_val.set_postfix({"val_batch_loss": f"{loss_val.item():.4f}"})

        epoch_avg_val = float(np.mean(val_loss)) if len(val_loss) > 0 else 0.0
        epoch_val_loss.append(epoch_avg_val)
        print(f"Epoch {epoch+1}/{max_epochs} avg_loss_val={epoch_avg_val:.4f}")

    return model, epoch_train_loss, epoch_val_loss


if __name__ == "__main__":
    model2_trained, epoch_train_loss2, epoch_val_loss2 = train2(
        model2, diffuser2, batches_train, batches_val, device, max_epochs=20
    )


plt.plot(epoch_train_loss2, label="train_loss")
plt.plot(epoch_val_loss2, label="val_loss")
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()


sample_count = 1024

print("generating samples")
sampler = SamplerNoise(model2_trained, diffuser2)
ema = EMA(model2_trained, decay=0.9999, device=device, start_step=0)


ema.apply_to(model2_trained)
gen_samples = sampler.sample(
    batch_size=sample_count,
    device=next(model2_trained.parameters()).device,
    mean_only=False,
)
ema.restore(model2_trained)

if isinstance(gen_samples, torch.Tensor):
    gen_samples = gen_samples.cpu().numpy()


real, real_l = datasetraw[:, :-1], datasetraw[:, -1][:, None]
scaler = preprocessing.StandardScaler().fit(real)
real = scaler.transform(real)

if isinstance(real, torch.Tensor):
    real = real.cpu().numpy()

D_syn = real.shape[1]
features_list = globals().get("features_syn", [f"f{i}" for i in range(D_syn)])

fig, axes = plt.subplots(D_syn, 1, figsize=(6, 2 * D_syn), sharex=False)
if D_syn == 1:
    axes = [axes]

for idx, ax in enumerate(axes):
    combined = np.concatenate([real[:, idx], gen_samples[:, idx]])
    bins = np.histogram_bin_edges(real[:, idx], bins=40)

    sns.histplot(
        real[:, idx],
        bins=bins,
        ax=ax,
        stat="density",
        color="C0",
        alpha=0.6,
        label="real",
    )
    ax.hist(
        gen_samples[:, idx],
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        color="gray",
        label="gen",
    )

    ax.set_xlabel(features_list[idx])
    ax.set_ylabel("density")
    ax.legend()

plt.tight_layout()
plt.show()
