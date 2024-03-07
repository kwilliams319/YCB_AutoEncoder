import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from gan import Generator, Discriminator

# Define your Generator and Discriminator classes here (same as before)

# Initialize the generator and discriminator
latent_dim = 32
generator = Generator(latent_dim).to('cuda')
discriminator = Discriminator().to('cuda')

# generator.load_state_dict(torch.load('YCB/gan/generator.pth'))
# discriminator.load_state_dict(torch.load('YCB/gan/discriminator.pth'))

# Define loss function and optimizers
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

# Load the dataset
batch_size = 16
root_dir = 'YCB/ycb_processing/padded_data'
ycb_dataset = ImageFolder(root=root_dir, transform=ToTensor())
train_loader = DataLoader(dataset=ycb_dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move images to CUDA
        images = images.to('cuda')

        # Generate fake images
        z = torch.randn(images.size(0), latent_dim).to('cuda')
        fake_images = generator(z)

        # Train Discriminator
        d_optimizer.zero_grad()
        real_outputs = discriminator(images)
        fake_outputs = discriminator(fake_images.detach())
        d_loss = criterion(real_outputs, torch.ones_like(real_outputs)) \
               + criterion(fake_outputs, torch.zeros_like(fake_outputs))
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, torch.ones_like(fake_outputs))
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    # Save the models
    torch.save(generator.state_dict(), 'YCB/gan/generator.pth')
    torch.save(discriminator.state_dict(), 'YCB/gan/discriminator.pth')
