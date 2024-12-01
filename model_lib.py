import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter


#--------------Loss functions
def loss_function(reconstructed_x, x, mean, log_var):
    reconstruction_loss = 1*nn.functional.mse_loss(reconstructed_x, x, reduction="mean")
    kl_divergence = -0.001 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence / x.size(0)

# Define the MoG-VAE loss function
def mog_vae_loss(reconstructed_x, x, mean, log_var, z, model, kl_weight=1):
    # Reconstruction loss
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction="sum")

    # KL divergence with Mixture of Gaussians prior
    kl_divergence = model.mog_kl_divergence(z, mean, log_var)
    return reconstruction_loss + kl_weight * kl_divergence


# Loss Function with Transport Operators
def vaells_loss(reconstructed_x, x, mean, log_var, anchors, transport_operators, z, gamma=1e-3, eta=1e-2):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction="mean")
    
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    # Transport operator penalty
    operator_loss = eta * sum(torch.norm(op, p="fro") for op in transport_operators)
    
    # Anchor-based prior (encourages paths through manifold)
    anchor_loss = gamma * torch.mean(torch.norm(z.unsqueeze(1) - anchors, dim=-1))
    
    return reconstruction_loss + kl_divergence + operator_loss + anchor_loss


# Max-GSW Loss Function
def wasserstein_vae_loss(reconstructed_x, x, z, target_distribution, vae, gamma=1e-3):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction="mean")
    wasserstein_loss = vae.wasserstein_distance(z, target_distribution)
    return reconstruction_loss + gamma * wasserstein_loss

# Loss function for Max-GSW VAE
def gsw_vae_loss(reconstructed_x, x, z, target_distribution, vae, gamma=1e-3):
    """
    Computes the loss for Max-GSW VAE.
    Args:
        reconstructed_x: Reconstructed samples.
        x: Input samples.
        z: Latent space samples.
        target_distribution: Target latent distribution (e.g., standard Gaussian).
        vae: Max-GSW VAE model.
        gamma: Weight for the Max-GSW term.
    """
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction="mean")
    gsw_loss = vae.gsw_distance(z, target_distribution)
    return reconstruction_loss + gamma * gsw_loss

# Loss function for Max-GSW VAE
def max_gsw_vae_loss(reconstructed_x, x, z, target_distribution, vae, gamma=1e-3):
    """
    Computes the loss for Max-GSW VAE.
    Args:
        reconstructed_x: Reconstructed samples.
        x: Input samples.
        z: Latent space samples.
        target_distribution: Target latent distribution (e.g., standard Gaussian).
        vae: Max-GSW VAE model.
        gamma: Weight for the Max-GSW term.
    """
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction="mean")
    gsw_loss = vae.max_gsw_distance(z, target_distribution)
    return reconstruction_loss + gamma * gsw_loss




#--------------Models
class VDeepVAE(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=16, latent_dim=2):
        super(VDeepVAE, self).__init__()
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
        self.fc4_out = nn.Linear(hidden_dim, hidden_dim)
        self.fc5_out = nn.Linear(hidden_dim, hidden_dim)
        self.fc6_out = nn.Linear(hidden_dim, hidden_dim)
        self.fc7_out = nn.Linear(hidden_dim, hidden_dim)
        self.fc8_out = nn.Linear(hidden_dim, hidden_dim)
        self.fc9_out = nn.Linear(hidden_dim, hidden_dim)
        self.fc10_out = nn.Linear(hidden_dim, input_dim)

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
        h = torch.relu(self.fc4_out(h))
        h = torch.relu(self.fc5_out(h))
        h = torch.relu(self.fc6_out(h))
        h = torch.relu(self.fc7_out(h))
        h = torch.relu(self.fc8_out(h))
        h = torch.relu(self.fc9_out(h))
        return self.fc10_out(h)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var
    
class DeepVAE(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=16, latent_dim=2):
        super(DeepVAE, self).__init__()
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
    
class ShallowVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, latent_dim=2):
        super(ShallowVAE, self).__init__()
        # Encoder
        self.fc1_in = nn.Linear(input_dim, hidden_dim)
        self.fc2_in = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc1_out = nn.Linear(latent_dim, hidden_dim)
        self.fc2_out = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1_in(x))
        h = torch.relu(self.fc2_in(h))
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
        return self.fc3_out(h)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var
    
class MoGVAE(nn.Module):
    
    def __init__(self, input_dim=2, hidden_dim=16, latent_dim=2, n_components=3):
        super(MoGVAE, self).__init__()
        self.n_components = n_components
        self.latent_dim = latent_dim

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

        # Mixture of Gaussians Prior Parameters
        self.mixture_weights = nn.Parameter(torch.ones(n_components) / n_components)
        self.mixture_means = nn.Parameter(torch.randn(n_components, latent_dim))
        self.mixture_log_vars = nn.Parameter(torch.zeros(n_components, latent_dim))

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

    def mog_kl_divergence(self, z, mean, log_var):
        # KL divergence between q(z|x) and MoG prior
        z = z.unsqueeze(1)  # (batch_size, 1, latent_dim)
        prior_means = self.mixture_means.unsqueeze(0)  # (1, n_components, latent_dim)
        prior_log_vars = self.mixture_log_vars.unsqueeze(0)  # (1, n_components, latent_dim)
        
        # Compute log-probabilities for each component
        log_weights = torch.log_softmax(self.mixture_weights, dim=0)
        log_probs = -0.5 * (
            torch.sum(prior_log_vars, dim=2)
            + torch.sum((z - prior_means) ** 2 / torch.exp(prior_log_vars), dim=2)
        )
        log_probs += log_weights  # Incorporate mixture weights

        # Compute log-sum-exp for stabilization
        log_qz = torch.logsumexp(log_probs, dim=1)
        log_pz = -0.5 * torch.sum(log_var + 1 - mean.pow(2) - log_var.exp())
        
        return log_pz - torch.sum(log_qz)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var, z
    
class TransportOperatorVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_anchors, num_operators):
        super(TransportOperatorVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_anchors = num_anchors
        self.num_operators = num_operators

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # Output both mean and log-variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Transport Operators and Anchor Points
        self.anchors = nn.Parameter(torch.randn(num_anchors, latent_dim))
        self.transport_operators = nn.Parameter(torch.randn(num_operators, latent_dim, latent_dim))

    def encode(self, x):
        encoded = self.encoder(x)
        mean, log_var = torch.chunk(encoded, 2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var, z

    def transport_operator_path(self, z, anchor_idx):
        """
        Computes transformation paths using transport operators and anchor points.
        """
        anchor = self.anchors[anchor_idx]
        transformations = [torch.matrix_exp(op) for op in self.transport_operators]
        path = sum([transform @ anchor for transform in transformations])
        return path

class GSWVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, projection_type="linear", degree=3, num_hidden=2, hidden_size=64, num_slices=10):
        """
        GSW VAE with configurable slicing projection.
        Args:
            input_dim (int): Dimensionality of input data.
            hidden_dim (int): Hidden layer size for encoder/decoder.
            latent_dim (int): Latent space dimensionality.
            projection_type (str): Type of projection ("linear", "polynomial", "nn").
            degree (int): Degree for polynomial projection.
            num_hidden (int): Number of hidden layers for NN projection.
            hidden_size (int): Hidden layer size for NN projection.
            num_slices (int): Number of slices for the GSW distance.
        """
        super(GSWVAE, self).__init__()
        self.latent_dim = latent_dim
        self.projection_type = projection_type
        self.degree = degree
        self.num_slices = num_slices

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # Mean and log variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Projection parameters
        if projection_type == "linear":
            self.theta = Parameter(torch.randn(num_slices, latent_dim))  # Multiple slicing directions
        elif projection_type == "polynomial":
            self.theta = Parameter(torch.randn(num_slices, degree, latent_dim))
        elif projection_type == "nn":
            self.projection_nn = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_size),
                    nn.ReLU(),
                    *[layer for _ in range(num_hidden) for layer in [nn.Linear(hidden_size, hidden_size), nn.ReLU()]],
                    nn.Linear(hidden_size, 1)
                ) for _ in range(num_slices)
            ])
        else:
            raise ValueError("Unsupported projection type. Choose from 'linear', 'polynomial', or 'nn'.")

    def encode(self, x):
        encoded = self.encoder(x)
        mean, log_var = torch.chunk(encoded, 2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var, z

    def gsw_projection(self, z, slice_idx):
        """Compute the projection for a specific slice."""
        if self.projection_type == "linear":
            return z @ self.theta[slice_idx]
        elif self.projection_type == "polynomial":
            poly_proj = torch.stack([z.pow(i) @ self.theta[slice_idx, i - 1] for i in range(1, self.degree + 1)], dim=-1)
            return poly_proj.sum(dim=-1)
        elif self.projection_type == "nn":
            return self.projection_nn[slice_idx](z).squeeze(-1)
        else:
            raise ValueError("Unsupported projection type.")

    def gsw_distance(self, z1, z2):
        """
        Compute the GSW distance by averaging over all slices.
        """
        total_distance = 0.0
        for slice_idx in range(self.num_slices):
            # Projection for current slice
            z1_proj = self.gsw_projection(z1, slice_idx)
            z2_proj = self.gsw_projection(z2, slice_idx)

            # Sorting projections
            z1_proj_sorted, _ = torch.sort(z1_proj, dim=0)
            z2_proj_sorted, _ = torch.sort(z2_proj, dim=0)

            # Wasserstein distance between sorted projections
            total_distance += torch.mean(torch.abs(z1_proj_sorted - z2_proj_sorted))

        return total_distance / self.num_slices
    

class MaxGSWVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, projection_type="linear", degree=3, num_hidden=2, hidden_size=64):
        """
        Max-GSW VAE with configurable slicing projection.
        Args:
            input_dim (int): Dimensionality of input data.
            hidden_dim (int): Hidden layer size for encoder/decoder.
            latent_dim (int): Latent space dimensionality.
            projection_type (str): Type of projection ("linear", "polynomial", "nn").
            degree (int): Degree for polynomial projection.
            num_hidden (int): Number of hidden layers for NN projection.
            hidden_size (int): Hidden layer size for NN projection.
        """
        super(MaxGSWVAE, self).__init__()
        self.latent_dim = latent_dim
        self.projection_type = projection_type
        self.degree = degree

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # Mean and log variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Projection parameters
        if projection_type == "linear":
            self.theta = Parameter(torch.randn(latent_dim))
        elif projection_type == "polynomial":
            self.theta = Parameter(torch.randn(degree, latent_dim))
        elif projection_type == "nn":
            self.projection_nn = nn.Sequential(
                nn.Linear(latent_dim, hidden_size),
                nn.ReLU(),
                *[layer for _ in range(num_hidden) for layer in [nn.Linear(hidden_size, hidden_size), nn.ReLU()]],
                nn.Linear(hidden_size, 1)
            )
        else:
            raise ValueError("Unsupported projection type. Choose from 'linear', 'polynomial', or 'nn'.")

    def encode(self, x):
        encoded = self.encoder(x)
        mean, log_var = torch.chunk(encoded, 2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var, z

    def max_gsw_projection(self, z):
        """Computes the projection based on the specified type."""
        if self.projection_type == "linear":
            return z @ self.theta
        elif self.projection_type == "polynomial":
            poly_proj = torch.stack([z.pow(i) @ self.theta[i] for i in range(1, self.degree + 1)], dim=-1)
            return poly_proj.sum(dim=-1)
        elif self.projection_type == "nn":
            return self.projection_nn(z)
        else:
            raise ValueError("Unsupported projection type.")

    def max_gsw_distance(self, z1, z2):
        """
        Compute the Max-GSW distance by optimizing the slicing projection.
        """
        # Projection of z1 and z2
        z1_proj = self.max_gsw_projection(z1)
        z2_proj = self.max_gsw_projection(z2)

        # Sorting projections
        z1_proj_sorted, _ = torch.sort(z1_proj, dim=0)
        z2_proj_sorted, _ = torch.sort(z2_proj, dim=0)

        # Wasserstein distance between sorted projections
        return torch.mean(torch.abs(z1_proj_sorted - z2_proj_sorted))
    
# Gaussian Isotropic Target
def gaussian_isotropic_target(batch_size, latent_dim):
    return torch.randn(batch_size, latent_dim)

# Circle Target
def circle_target(batch_size, radius=1.0, noise=0.05):
    angles = 2 * np.pi * np.random.rand(batch_size)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    data = np.vstack((x, y)).T
    data += noise * np.random.randn(*data.shape)
    return torch.tensor(data, dtype=torch.float32)

# Spiral Target
def spiral_target(batch_size, noise=0.05):
    t = np.linspace(0, 4 * np.pi, batch_size)
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.vstack((x, y)).T
    data += noise * np.random.randn(*data.shape)
    return torch.tensor(data, dtype=torch.float32)

# Choose Latent Space Target: Gaussian Isotropic, Circle, Spiral
def get_latent_target(batch_size, latent_dim, target="gaussian"):
    if target == "gaussian":
        return gaussian_isotropic_target(batch_size, latent_dim)
    elif target == "circle":
        return circle_target(batch_size, radius=1.0)
    elif target == "spiral":
        return spiral_target(batch_size)
    else:
        raise ValueError("Invalid latent space target. Choose 'gaussian', 'circle', or 'spiral'.")

