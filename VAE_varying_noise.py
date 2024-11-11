import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic data with a specific covariance matrix
# 10. Data Distributed Along a y axis continous
# Generate y-values uniformly along the y-axis
y_values = np.random.uniform(-100, 100, 2000)
y_values.sort()  
# Initialize arrays to store x-values
x_values = np.zeros_like(y_values)
noise_values = []
for i, y in enumerate(y_values):
    # Define a continuously varying covariance for x based on y
    #variance_x = 5 + (y + 100) * 0.95  # Map y in [-100, 100] to a variance in [5, 195]
    variance_x = 5 + 800 * (0.5 + 0.5 * np.sin(y * np.pi /50))  # Scales sine wave to [5, 50]
    covariance_matrix = [[variance_x, 0], [0, 0.1]]  # Keep variance for y small
    noise = np.random.multivariate_normal([0, 0], covariance_matrix)
    noise_values.append(noise)
    x_values[i] = noise[0]
    
data_continuous_covariance = np.vstack((x_values, y_values)).T
data_continuous_covariance = np.array(data_continuous_covariance, dtype=np.float32)
data_min = data_continuous_covariance.min(axis=0)
data_max = data_continuous_covariance.max(axis=0)

# Scale the data to [-1, 1]
data_continuous_covariance = 2 * (data_continuous_covariance - data_min) / (data_max - data_min) - 1

# 11. Map to circle
theta = (y_values - y_values.min()) / (y_values.max() - y_values.min()) * 2 * np.pi  # Map y in [-100, 100] to [0, 2*pi]
print("len theta:", len(theta))
radius = 200
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
noise_circle = np.zeros((len(theta), 2))
for i in range(len(theta)):
    # Determine the normal direction at this point (perpendicular to the tangent)
    normal_direction = np.array([np.cos(theta[i]), np.sin(theta[i])])
    noise_circle[i] = noise_values[i][0] * normal_direction 
data_circle_transformed = np.vstack((circle_x, circle_y)).T + noise_circle
data_circle_transformed = np.array(data_circle_transformed, dtype=np.float32)

# Find the minimum and maximum values of the data
data_min = data_circle_transformed.min(axis=0)
data_max = data_circle_transformed.max(axis=0)

# Scale the data to [-1, 1]
data_circle_scaled = 2 * (data_circle_transformed - data_min) / (data_max - data_min) - 1


data_tensor = torch.tensor(data_continuous_covariance)
# Define VAE model in PyTorch
class VAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, latent_dim=2):
        super(VAE, self).__init__()
        # Encoder
        self.fc1_in = nn.Linear(input_dim, hidden_dim)
        self.fc2_in = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_in = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc1_out = nn.Linear(latent_dim, hidden_dim)
        self.fc2_out = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_out = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1_in(x))
        h = torch.relu(self.fc2_in(h))
        h = torch.relu(self.fc3_in(h))
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = torch.relu(self.fc1_out(z))
        h = torch.relu(self.fc2_out(h))
        h = torch.relu(self.fc3_out(h))
        return self.fc4_out(h)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var

# Initialize VAE, optimizer, and loss function
latent_dim = 5
vae = VAE(input_dim=2, hidden_dim=16, latent_dim=latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Loss function
def loss_function(reconstructed_x, x, mean, log_var):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction="sum")
    kl_divergence = -0.1 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

# Training loop
epochs = 10000
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
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_tensor):.4f}")

# Sampling from the trained VAE
with torch.no_grad():
    latent_samples = torch.randn(500, latent_dim)  # sample from standard normal in latent space
    generated_samples = vae.decode(latent_samples).numpy()

# Plotting the original and generated samples for visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data_continuous_covariance[:, 0], data_continuous_covariance[:, 1], alpha=0.3)
plt.title("Original Data")
plt.subplot(1, 2, 2)
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.3, color="orange")
plt.title("Generated Samples from VAE")
plt.savefig("images/VAE_varying_noise.png")