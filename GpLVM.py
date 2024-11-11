import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

# Load data
data = np.load('data/data_sine_const_variance.npy')
x_values = data[:, 0]
y_values = data[:, 1]
X_train = torch.tensor(np.column_stack((x_values, y_values)), dtype=torch.float32)

# Set the latent space dimension and initialize latent variables
latent_dim = 1  # Define latent dimension (reduction to 1D)
latent_vars = torch.randn(X_train.shape[0], latent_dim, requires_grad=True)  # Initialize random latent positions

# Define RBF kernel in PyTorch
def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    """Compute the RBF (Gaussian) kernel between X1 and X2."""
    sqdist = torch.cdist(X1, X2, p=2) ** 2
    return variance * torch.exp(-0.5 * sqdist / length_scale ** 2)

# Optimization setup
optimizer = optim.Adam([latent_vars], lr=0.01)
num_iters = 1000

# Training loop
for i in range(num_iters):
    # Compute the kernel matrix for latent variables
    K = rbf_kernel(latent_vars, latent_vars, length_scale=1.0, variance=1.0)
    K += 1e-5 * torch.eye(K.shape[0])
    # Add a small noise term for numerical stability
    K += 1e-6 * torch.eye(K.shape[0])

    # Predict using the GP kernel and calculate loss
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(X_train, L)
    reconstructed = K @ alpha
    
    # Mean squared error between reconstructed and original data
    loss = F.mse_loss(reconstructed, X_train)

    # Backpropagate and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss.item()}")

# Visualize learned latent space and reconstruction
latent_np = latent_vars.detach().numpy()
x_test = np.linspace(latent_np.min(), latent_np.max(), 100).reshape(-1, latent_dim)
x_test_torch = torch.tensor(x_test, dtype=torch.float32)
K_test = rbf_kernel(x_test_torch, latent_vars, length_scale=1.0, variance=1.0)
y_test = (K_test @ alpha).detach().numpy()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(latent_np, x_values, color='grey', label='Latent Representation of X')
plt.plot(y_test[:, 0],y_test[:, 1] , 'r-', label='GPLVM Reconstruction')
plt.title('GPLVM: Learned Latent Space and Reconstructed Data')
plt.xlabel('Latent Dimension')
plt.ylabel('Observed Data')
plt.legend()
plt.savefig("images/GPLVM_output.png")
