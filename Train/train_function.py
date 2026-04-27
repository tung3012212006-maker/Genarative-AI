
import torch
from tqdm import tqdm

def train(model, epochs, optimizer, device, train_loader, model_path):
    model.train()
    for epoch in range(epochs):
        train_loss = 0

        train_loop = tqdm(train_loader,
                        desc=f"Epoch {epoch + 1}/{epochs} [Training]",
                        leave=False)
        for batch_idx, (data, _) in enumerate(train_loop):
            data = data.to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(train_loader)
        print(f'Epoch: {epoch+1} | Average Loss: {avg_loss:.4f}')

    print("Training complete!")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved in: {model_path}")
