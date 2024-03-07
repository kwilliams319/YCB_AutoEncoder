import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256 * 85 * 29),  # Linear layer
            nn.ReLU(True),
            nn.Unflatten(1, (256, 85, 29)),  # Reshape to 256x85x29
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 169x58
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 337x115
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 674x230
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1347x459
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)[:, :, 7:-6, 3:-2]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 674x230
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 337x115
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 169x58
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 85x29
            nn.ReLU(True),
            nn.Flatten(),  # Flatten the output
            nn.Linear(256 * 85 * 29, 128),  # Linear layer
            nn.ReLU(True),
            nn.Linear(128, 1),  # Additional linear layer for discriminator
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)#.view(-1, 1)


# # Test the GAN
# latent_dim = 32
# batch_size = 2
# generator = Generator(latent_dim)
# discriminator = Discriminator()

# # Generate fake images
# fake_input = torch.randn(batch_size, latent_dim)
# fake_images = generator(fake_input)

# # Discriminate fake images
# fake_outputs = discriminator(fake_images)
# pass
# # Calculate generator loss (not in this snippet)
# # Calculate discriminator loss (not in this snippet)
