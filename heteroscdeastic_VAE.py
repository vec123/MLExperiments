import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load data
data = np.load('data/data_continuous_covariance.npy')
x_data, y_data = data[:, 0], data[:, 1]

# Prepare dataset
train_data = torch.tensor(np.stack((x_data, y_data), axis=1), dtype=torch.float32)
train_loader = DataLoader(TensorDataset(train_data), batch_size=128, shuffle=True)

# Define VAE components
latent_dim = 2  # Adjust based on complexity
hidden_dim = 64  # Number of neurons in hidden layers

# Encoder model
class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=hidden_dim, latent_dim=latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var

# Decoder model
class Decoder(nn.Module):
    def __init__(self, latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = torch.relu(self.fc4(h))
        return self.fc_out(h)
    
class HeteroscedasticDecoder(nn.Module):
    def __init__(self, latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=2):
        super(HeteroscedasticDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_out_log_var = nn.Linear(hidden_dim, output_dim)  # Log variance output

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        h = torch.relu(self.fc4(h))
        mean = self.fc_out_mean(h)
        log_var = self.fc_out_log_var(h)  # Log variance for heteroscedasticity
        return mean, log_var
    
# VAE model
class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var

class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed_mean, x_reconstructed_log_var = self.decoder(z)
        return x_reconstructed_mean, x_reconstructed_log_var, z_mean, z_log_var
    
# Define loss function
def vae_loss(x, x_reconstructed, z_mean, z_log_var):
    reconstruction_loss = nn.MSELoss()(x_reconstructed, x)
    kl_divergence = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return reconstruction_loss + kl_divergence

def heteroscedastic_vae_loss(x, x_reconstructed_mean, x_reconstructed_log_var, z_mean, z_log_var):
    # Ensure the tensors are appropriately shaped and handle both mean and variance separately
    reconstruction_loss = 0.5 * torch.mean(
        torch.exp(-x_reconstructed_log_var) * (x - x_reconstructed_mean) ** 2 + x_reconstructed_log_var
    )
    kl_divergence = -0.001 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return reconstruction_loss + kl_divergence

# Instantiate models and optimizer
encoder = Encoder()
decoder = Decoder()
vae = VariationalAutoencoder(encoder, decoder)
vae = VariationalAutoencoder(encoder, HeteroscedasticDecoder())
optimizer = optim.Adam(vae.parameters(), lr=0.0001)

# Training loop
epochs = 5000
for epoch in range(epochs):
    for batch in train_loader:
        x = batch[0]
        optimizer.zero_grad()
        #x_reconstructed, z_mean, z_log_var = vae(x)
        #loss = vae_loss(x, x_reconstructed, z_mean, z_log_var)
        x_reconstructed_mean, x_reconstructed_log_var, z_mean, z_log_var = vae(x)
        loss = heteroscedastic_vae_loss(x, x_reconstructed_mean, x_reconstructed_log_var, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Sampling from the learned distribution
def sample_from_distribution(num_samples=100):
    with torch.no_grad():
        z_samples = torch.randn(num_samples, latent_dim)
        generated_samples = vae.decoder(z_samples)
    return generated_samples.numpy()

def sample_from_hetero_distribution(num_samples=100):
    with torch.no_grad():
        # Step 1: Sample from the latent space (standard normal distribution)
        z_samples = torch.randn(num_samples, latent_dim)
        
        # Step 2: Decode to get mean and log variance of the reconstructed outputs
        generated_means, generated_log_vars = vae.decoder(z_samples)
        
        # Step 3: Sample from the Gaussian distribution defined by the predicted mean and variance
        std_devs = torch.exp(0.5 * generated_log_vars)  # Convert log variance to standard deviation
        generated_samples = generated_means + std_devs * torch.randn_like(generated_means)
    
    return generated_samples.numpy()
# Generate and save samples

samples = sample_from_hetero_distribution(num_samples=1000)
#np.save('images/generated_samples_pytorch.npy', samples)

import matplotlib.pyplot as plt
# Plotting the sampled points
plt.figure(figsize=(10, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, color='blue')
plt.title('Sampled Points from Heteroscedastic VAE')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.xlim(-150, 150)  # Adjust limits as needed
plt.ylim(-150, 150)  # Adjust limits as needed
plt.grid()
plt.savefig('images/sampled_points_plot.png')
plt.close()
print("Sampled points plot saved to 'data/sampled_points_plot.png'")

print("Generated samples saved to 'data/generated_samples_pytorch.npy'")
