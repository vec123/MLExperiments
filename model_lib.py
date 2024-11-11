import torch
import torch.nn as nn



#--------------Loss functions
def loss_function(reconstructed_x, x, mean, log_var):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction="mean")
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence / x.size(0)

# Define the MoG-VAE loss function
def mog_vae_loss(reconstructed_x, x, mean, log_var, z, model, kl_weight=1):
    # Reconstruction loss
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction="sum")

    # KL divergence with Mixture of Gaussians prior
    kl_divergence = model.mog_kl_divergence(z, mean, log_var)
    return reconstruction_loss + kl_weight * kl_divergence



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
        return torch.tanh(self.fc3_out(h))

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