import torch

from models.vae import VAE
from plot_latent_manifold import plot_latent_manifold
from sampling import sampling

INPUT_CHANEL = 1
HIDDEN_DIM = 400
LATENT_DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(input_channels = INPUT_CHANEL, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
model.load_state_dict(torch.load("./checkpoints/VAE.pth", weights_only=True))
model.to(DEVICE)
sampling(model=model, device=DEVICE, grid=4)