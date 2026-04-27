import torch
import matplotlib.pyplot as plt

def sampling(model, device, grid = 4):
    num_samples = grid **2
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        generated_imgs = model.decode(z)
        generated_imgs = generated_imgs.cpu()

        plt.figure(figsize=(8, 8))
        for i in range(num_samples):
            plt.subplot(grid, grid, i + 1)
            plt.imshow(generated_imgs[i].squeeze(), cmap='gray')
            plt.axis('off')
        
        plt.suptitle("Generated Images (z ~ N(0,1))")
        plt.show()
