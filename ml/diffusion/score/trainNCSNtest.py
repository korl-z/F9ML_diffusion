import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from ml.diffusion.score.model import RefineNet
from ml.diffusion.score.score_matching import score_matching_loss, linear_noise_scale
from ml.diffusion.score.langevin_dynamics import sample

from ml.common.data_utils.data_modules import HiggsDataset
from ml.common.data_utils.syn_datacreator import create_custom_multidim_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

print("Creating dummy Higgs dataset...")
N_FEATURES = 2

datasetraw = create_custom_multidim_dataset(n_samples=10000, n_features=N_FEATURES, label_random=True, signal_frac=0.5, seed=123)
print("dataset shape:", datasetraw.shape)   
print(datasetraw[:5])

# Split the 3-feature dataset
train_data_syn, val_data_syn = train_test_split(datasetraw, train_size=0.8)

train_dataset_syn = HiggsDataset(train_data_syn)
val_dataset_syn = HiggsDataset(val_data_syn)

batch_size = 512
batches_train = DataLoader(train_dataset_syn, num_workers=0, batch_size=batch_size, shuffle=True)
batches_val = DataLoader(val_dataset_syn, num_workers=0, batch_size=batch_size, shuffle=False)

# print(next(iter(batches_train)))

print(f"\n Data loaded. Using {N_FEATURES} features.")
print(f"Train shape: {train_data_syn.shape}, Val shape: {val_data_syn.shape}")
print("-" * 30)

# --- 2. Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
EPOCHS = 5
LEARNING_RATE = 1e-4
N_NOISE_SCALES = 10

IMG_SHAPE = (1, 2, 1) 
assert np.prod(IMG_SHAPE) == N_FEATURES, "Image shape must match number of features"

noise_scales = linear_noise_scale(length=N_NOISE_SCALES).to(device)

# CHANGED: Initialize the model with a smaller architecture suitable for 3 features.
model = RefineNet(
    in_channels=IMG_SHAPE[0],
    hidden_channels=(16, 32, 64, 128), # Reduced channel sizes
    n_noise_scale=N_NOISE_SCALES
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Model and Optimizer Initialized.")
print("New Image Shape:", IMG_SHAPE)
print("-" * 30)

# --- 4. Training Loop (No changes needed here) ---
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    
    for step, (features, _) in enumerate(batches_train):
        optimizer.zero_grad()
        
        features = features.to(device)
        
        reshaped_features = features.view(-1, *IMG_SHAPE) # -> (batch_size, 1, 3, 1)

        loss = score_matching_loss(model, reshaped_features, noise_scales)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(batches_train)

    # --- 5. Validation Loop ---
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for step, (features, _) in enumerate(batches_val):
            features = features.to(device)
            reshaped_features = features.view(-1, *IMG_SHAPE)
            
            loss = score_matching_loss(model, reshaped_features, noise_scales)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(batches_val)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

print("\nTraining finished.")
print("-" * 30)

# --- 6. Sampling (Data Generation) (No changes needed here) ---
print("Generating new data samples using annealed Langevin dynamics...")

sample_count = 2000   

# sample_shape = (sample_count, *IMG_SHAPE)

# generated_samples = sample(model, sample_shape, noise_scales, device, n_steps=100)

# # Reshape back to the original tabular format
# generated_data = generated_samples.view(sample_count, N_FEATURES)




sample_shape = (sample_count, *IMG_SHAPE)
samples_2d = sample(model, sample_shape, noise_scales, device, n_steps=100)

samples = samples_2d.view(sample_count, N_FEATURES)

print(f"Successfully generated {sample_count} samples.")
print("Shape of generated data (original format):", samples.shape)
print("Example of a generated data point:\n", samples[0].cpu().numpy())

if isinstance(samples, torch.Tensor):
    samples = samples.cpu().numpy()

#real data (not in batches)
real, real_l = datasetraw[:, :-1], datasetraw[:, -1][:, None]

scaler = preprocessing.StandardScaler().fit(real)

real = scaler.transform(real)

if isinstance(real, torch.Tensor):
    real = real.cpu().numpy()

features_list = globals().get("features_syn", [f"f{i}" for i in range(N_FEATURES)])

# Plot per-feature histograms aligned
fig, axes = plt.subplots(N_FEATURES, 1, figsize=(6, 2 * N_FEATURES), sharex=False)
if N_FEATURES == 1:
    axes = [axes]

for idx, ax in enumerate(axes):
    # compute bins from combined data for alignment
    combined = np.concatenate([real[:, idx], samples[:, idx]])
    bins = np.histogram_bin_edges(real[:, idx], bins=40)

    # real filled hist
    sns.histplot(real[:, idx], bins=bins, ax=ax, stat="density", color='C0', alpha=0.6, label='real')

    # generated outline histogram
    ax.hist(samples[:, idx], bins=bins, density=True, histtype='step', lw=1.8, color='gray', label='gen')

    ax.set_xlabel(features_list[idx])
    ax.set_ylabel("density")
    ax.legend()

plt.tight_layout()
plt.show()

