import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.feature_dim = 64 * 7 * 7
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding = 1),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.feature_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.view(x.size(0), -1);
        x = F.relu(self.fc(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.feature_dim = 64 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feature_dim)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # (32,14,14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # (1,28,28)
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 7, 7)
        x = self.deconv(z)
        return x

class VAE(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.Encoder = Encoder(input_channels, hidden_dim, latent_dim)
        self.Decoder = Decoder(latent_dim, hidden_dim)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.Encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.Decoder(z)
        return x_recon, mu, logvar
    def loss_function(self, x_recon, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return (recon_loss + kl_loss) / x.size(0)

