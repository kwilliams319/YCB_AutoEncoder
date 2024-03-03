import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from AutoEncoder import ConvAutoencoder  # Assuming autoencoder_conv.py contains the ConvAutoencoder class definition
from tqdm import tqdm
# from AlexNet import AlexNet

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Hyperparameters
batch_size = 16
learning_rate = 0.0001
num_epochs = 3000

# Initialize the convolutional autoencoder
autoencoder = ConvAutoencoder().to('cuda')
# autoencoder.load_state_dict(torch.load('YCB/autoencoder.pth')) 

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Load the YCB RGB dataset
# root_dir = 'YCB/ycb_data/less_data'  # Updated path
root_dir = 'YCB/padded_data'
ycb_dataset = ImageFolder(root=root_dir, transform=ToTensor())

# Create a data loader
train_loader = DataLoader(dataset=ycb_dataset, batch_size=batch_size, shuffle=True)#, pin_memory=True)

# Train the autoencoder
for epoch in range(num_epochs):
    epoch_loss = 0
    for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

        images = images.to('cuda')

        # Forward pass
        outputs = autoencoder(images)
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.8f}')

    # Save the model checkpoint
    torch.save(autoencoder.state_dict(), f'YCB/autoencoder_lin.pth')