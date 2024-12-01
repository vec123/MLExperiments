import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import model_lib

torch.cuda.empty_cache()

data = np.load('data/single_circle_pattern.npy')
data = np.load('data/data_spirals_1_turn.npy')
data = np.load('data/data_sine_const_variance.npy')
data = np.load('data/data_concentric_circles.npy')
#data = np.load('data/.npy')
#data = np.load('data/data_cov.npy')
random_indices = np.random.choice(data.shape[0], 500, replace=False)
#data_subset = data[random_indices]
#data = data_subset
data_tensor = torch.tensor(data, dtype=torch.float32)
#min_val = data_tensor.min()
#max_val = data_tensor.max()
mean_val = data_tensor.mean(dim=0, keepdim=True)
std_val = data_tensor.std(dim=0, keepdim=True)
data_tensor = (data_tensor - mean_val) / std_val
#data_tensor = 2 * (data_tensor - min_val) / (max_val - min_val) - 1


# Initialize VAE, optimizer, and loss function
#-------VAE
input_dim = 2
hidden_dim = 32
latent_dim = 2
#vae = model_lib.ShallowVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

#-------VAELLS
num_anchors = 6
num_operators = 4
#vae = model_lib.TransportOperatorVAE(input_dim, hidden_dim, latent_dim, num_anchors, num_operators)

#-------GSWVAE
vae = model_lib.GSWVAE(input_dim, hidden_dim, latent_dim, projection_type="polynomial", degree=3, num_hidden=2, hidden_size=64)
# Training loop
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

epochs = 5000
batch_size = 64
data_loader = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        reconstructed_batch, mean, log_var,z = vae(batch)
        #loss = model_lib.loss_function(reconstructed_batch, batch, mean, log_var)
         #loss = model_lib.vaells_loss(reconstructed_batch, batch, mean, log_var, vae.anchors, vae.transport_operators, z)
                
        latent_target =  model_lib.get_latent_target(batch_size, latent_dim, target="gaussian")  # Change target here
        loss = model_lib.gsw_vae_loss(reconstructed_batch, batch, z, latent_target, vae, gamma=1e-3)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_tensor):.4f}")


# Sampling from the trained VAE
with torch.no_grad():
    latent_samples = torch.randn(4000, latent_dim)  # sample from standard normal in latent space
    generated_samples = vae.decode(latent_samples).numpy()


# Plotting the original and generated samples for visualization

plt.figure(figsize=(10, 5))
plt.scatter(data_tensor[:, 0], data_tensor[:, 1], alpha=0.3, color="red")
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.3, color="blue")
plt.title("VAE Generated Samples ")
plt.savefig("images/GSWVAE_output_test.png")
