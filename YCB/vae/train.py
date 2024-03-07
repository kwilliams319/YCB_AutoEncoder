import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from VAE import ConvVAE


# Load the dataset
batch_size = 16
root_dir = 'YCB/ycb_processing/padded_data'
ycb_dataset = ImageFolder(root=root_dir, transform=ToTensor())
train_loader = DataLoader(dataset=ycb_dataset, batch_size=batch_size, shuffle=True)

# Initialize the VAE
vae = ConvVAE().to('cuda')

# Load the results of the last training run
vae.load_state_dict(torch.load('YCB/vae/vae_model.pth'))

# Define loss function
mse_loss = nn.MSELoss(reduction='sum')

# Define optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    total_loss = 0
    for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()

        images = images.to('cuda')

        # Forward pass
        recon_images, mu, logvar = vae(images)

        # Compute reconstruction loss
        reconstruction_loss = mse_loss(recon_images, images)

        # Compute KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Compute total correlation loss in a single line
        # delete this term to revert from disentangled VAE to vanilla VAE
        tc_loss = ((torch.exp(0.5 * logvar).mean(dim=0) - 1).sum())

        # Total loss
        loss = reconstruction_loss + 50*kl_divergence + 50*tc_loss

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss / len(train_loader.dataset)}")
    # Save the model
    torch.save(vae.state_dict(), 'YCB/vae/d_vae_model.pth')

# # Save the model
# torch.save(vae.state_dict(), 'YCB/vae/vae_model.pth')
