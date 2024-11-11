import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = np.load('data/data_white_noise.npy')
min_val = data.min()
max_val = data.max()
#data = 2 * (data - min_val) / (max_val - min_val) - 1

# Step 1: Fit PCA to the data
pca = PCA(n_components=2)
pca.fit(data)

# Step 2: Generate samples in the latent space (standard normal samples)
latent_samples = np.random.normal(0, 1, (len(x), 2))* np.sqrt(pca.explained_variance_)  # White noise in 2D

# Step 3: Transform the latent samples using PCA components
generated_samples = pca.inverse_transform(latent_samples)

# Plotting original vs. generated samples
plt.figure(figsize=(10, 5))
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Original Data', alpha=0.5)
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='red', label='PCA Generated Samples', alpha=0.5)
# Add arrows for principal components as thin guiding lines
origin = np.mean(data, axis=0)  # Center point for the arrows (mean of data)
for i, (component, variance) in enumerate(zip(pca.components_, pca.explained_variance_)):
    # Scale component vector by a factor to make arrows more visible
    arrow_length = np.sqrt(variance) * 3  # Scale up for visual clarity
    plt.plot(
        [origin[0], origin[0] + component[0] * arrow_length],
        [origin[1], origin[1] + component[1] * arrow_length],
        color="black", linestyle="--", linewidth=4,
        label=f"Principal Direction {i+1}" if i == 0 else ""
    )
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Original Data vs. PCA Generated Samples")
plt.savefig("PCA_normal.png")
