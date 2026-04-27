from torchvision import datasets, transforms
from torch.utils.data import  DataLoader

def get_dataloader(batch_size):
    # 1. Define the transformations (Convert to Tensor)
    # We don't normalize to mean=0, std=1 here because for a VAE
    # we usually want inputs in [0, 1] range to match the Sigmoid output.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 2. Download the Training and Test datasets
    # 'root' is where the data will be stored.
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 3. Create Data Loaders (for batching)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader