import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import model_lib

data = np.load('data/data_white_noise.npy')
data=data
data_tensor = torch.tensor(data, dtype=torch.float32)
min_val = data_tensor.min()
max_val = data_tensor.max()
#data_tensor = 2 * (data_tensor - min_val) / (max_val - min_val) - 1


# Initialize VAE, optimizer, and loss function
latent_dim = 2
vae = model_lib.ShallowVAE(input_dim=2, hidden_dim=16, latent_dim=latent_dim)
loss_function = model_lib.loss_function()
# Training loop
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

epochs = 5000
batch_size = 64
data_loader = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        reconstructed_batch, mean, log_var = vae(batch)
        loss = loss_function(reconstructed_batch, batch, mean, log_var)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_tensor):.4f}")

# Sampling from the trained VAE
with torch.no_grad():
    latent_samples = torch.randn(2000, latent_dim)  # sample from standard normal in latent space
    generated_samples = vae.decode(latent_samples).numpy()

# Plotting the original and generated samples for visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.3)
plt.title("Original Data")
plt.subplot(1, 2, 2)
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.3, color="green")
plt.title("Generated Samples from VAE")
plt.savefig("images/VAE_output.png")