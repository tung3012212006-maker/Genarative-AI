import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.modules):
    def __init__(self, latent_dim = 20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding = 1),
            nn.RELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding = 1),
            nn.RELU()
        )
        self.fc = nn.linear(64*7*7, 256)
        self.mu = nn.linear(256, latent_dim)
        self.logvar = nn.linear(256, latent_dim)

    def forward(self, inputs):
        x = self.conv(x)
        x = x.view(x.size(0), -1);
        x = F.relu(self.fc(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class Decoder(torch.nn.module):
    def __init__(self, latent_dim = 20):
        super().__init__()
        self.fc = nn.linear(latent_dim, 64 * 7 * 7)
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

class VAE(nn.module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.Encoder = Encoder(latent_dim)
        self.Decoder = Decoder(latent_dim)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.Encoder.forward(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.Decoder(z)
        return x_recon, mu, logvar
def loss_function(x_recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    
    return recon_loss + kl_loss

