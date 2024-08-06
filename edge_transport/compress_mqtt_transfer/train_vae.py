import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import multiprocessing

"""
This script trains a Variational Autoencoder (VAE) for image compression using PyTorch.
The VAE is trained on a dataset of blood cell images and can be used for efficient image transfer or storage.
"""

# VAE parameters
input_dim = 256
latent_dim = 6144
batch_size = 32
num_epochs = 50
learning_rate = 1e-4

# Path to your folder of training images
train_data_path = "archive/dataset2-master/dataset2-master/images/TRAIN/"
# Train images were downloaded from Kaggle:
# https://www.kaggle.com/datasets/paultimothymooney/blood-cells?resource=download

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model for image compression.

    This VAE uses convolutional layers in the encoder and transposed convolutional layers in the decoder.
    It's designed to work with color images of size 256x256 pixels.
    """

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode the input into the latent space."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode the latent representation back to image space."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    Computes the VAE loss function.

    Args:
    recon_x (Tensor): The reconstructed input
    x (Tensor): The original input
    mu (Tensor): The mean of the latent Gaussian
    logvar (Tensor): The log variance of the latent Gaussian

    Returns:
    Tensor: The total loss (reconstruction loss + KL divergence)
    """
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, train_loader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
    model (nn.Module): The VAE model
    train_loader (DataLoader): DataLoader for the training data
    optimizer (Optimizer): The optimizer for the model
    device (torch.device): The device to train on
    """
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Average loss: {train_loss / len(train_loader.dataset):.4f}')

def main():
    """
    Main function to prepare data, initialize model, and run the training loop.
    """
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((input_dim, input_dim)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize the VAE and optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Main training process
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        train(model, train_loader, optimizer, device)

    # Save the trained model
    torch.save(model.state_dict(), 'vae_weights.pth')
    print("Training completed. Model weights saved as 'vae_weights.pth'")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Add this line
    main()