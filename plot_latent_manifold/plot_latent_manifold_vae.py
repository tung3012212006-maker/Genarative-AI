import torch
import matplotlib.pyplot as plt

def plot_latent_manifold(model, device, n=20, digit_size=28):
    model.eval()
    figure = torch.zeros((digit_size * n, digit_size * n))
    
    # Create a grid of coordinates from -3 to +3 standard deviations
    grid_x = torch.linspace(-3, 3, n)
    grid_y = torch.linspace(-3, 3, n)

    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]]).to(device)
                x_decoded = model.decode(z_sample)
                digit = x_decoded.view(digit_size, digit_size)
                
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(8, 8))
    plt.imshow(figure.cpu(), cmap='Greys_r')
    plt.axis('off')
    plt.title("Latent Space Manifold")
    plt.show()
    # plt.savefig("Latent Space Manifold", dpi=600)