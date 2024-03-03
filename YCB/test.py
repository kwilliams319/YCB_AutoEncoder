import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, RandomSampler
import matplotlib.pyplot as plt
import numpy as np
from AutoEncoder import ConvAutoencoder  # Assuming autoencoder_conv.py contains the ConvAutoencoder class definition

# Load the YCB RGB dataset
# root_dir = 'YCB/ycb_data/less_data'
root_dir = 'YCB/padded_data'
ycb_dataset = ImageFolder(root=root_dir, transform=transforms.ToTensor())

# Initialize the autoencoder
autoencoder = ConvAutoencoder()
autoencoder.load_state_dict(torch.load('YCB/autoencoder_lin.pth'))  # Load trained weights

# Set the model to evaluation mode
autoencoder.eval()

# Function to display original image and autoencoder output side by side
def display_images(originals, reconstructed):
    num_images = originals.size(0)
    fig, axes = plt.subplots(nrows=1, ncols=num_images*2, figsize=(15*num_images, 5))
    for i in range(num_images):
        axes[2*i].imshow(np.transpose(originals[i].numpy(), (1, 2, 0)))
        axes[2*i].set_title(f'Original Image {i+1}')
        axes[2*i].axis('off')
        axes[2*i+1].imshow(np.transpose(reconstructed[i].detach().numpy(), (1, 2, 0)))
        axes[2*i+1].set_title(f'Autoencoder Output {i+1}')
        axes[2*i+1].axis('off')
    plt.show()

# Create a random sampler
sampler = RandomSampler(ycb_dataset, replacement=True, num_samples=3)

# Create a data loader using the random sampler
data_loader = DataLoader(ycb_dataset, batch_size=3, sampler=sampler)

# Test the autoencoder on three sets of random samples from the dataset
for original_images, _ in data_loader:
    # Pass the original images through the autoencoder
    reconstructed_images = autoencoder(original_images)

    # Display original and reconstructed images
    display_images(original_images, reconstructed_images)
