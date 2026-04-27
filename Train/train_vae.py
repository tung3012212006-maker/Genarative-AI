<<<<<<< HEAD

=======
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from models.vae import VAE
from data.mnist import get_dataloader
from Train.train_function import train

INPUT_CHANEL = 1
HIDDEN_DIM = 400
LATENT_DIM = 2
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
os.makedirs("./checkpoints", exist_ok=True)
MODEL_PATH = "./checkpoints/VAE.pth"

model = VAE(input_channels = INPUT_CHANEL, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
train_loader, _ = get_dataloader(batch_size=BATCH_SIZE)

train(model=model, epochs=EPOCHS, optimizer=optimizer, device=DEVICE, train_loader=train_loader, model_path=MODEL_PATH)
>>>>>>> c6c239e (update)
