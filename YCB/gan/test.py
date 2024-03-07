import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from gan import Generator

# Initialize the generator
latent_dim = 32
generator = Generator(latent_dim).to('cuda')

# Load the trained generator weights
generator.load_state_dict(torch.load('YCB/gan/generator.pth'))

# Generate images
num_images = 12
z = torch.randn(num_images, latent_dim).to('cuda')
generated_images = generator(z)

# Plot the images
grid = make_grid(generated_images.cpu().detach(), nrow=int(num_images ** 0.5), normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.show()