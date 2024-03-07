import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 674x230
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 337x115
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 169x58
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 85x29
            nn.ReLU(True),
            nn.Flatten(),  # Flatten the output
            nn.Linear(256 * 85 * 29, 128),  # Linear layer for mean
            nn.ReLU(True),
            nn.Linear(128, latent_dim * 2)  # Linear layer for mean and logvar
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),  # Linear layer for input
            nn.ReLU(True),
            nn.Linear(128, 256 * 85 * 29),  # Linear layer for decoding
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
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode input into mean and log variance
        enc_output = self.encoder(x)
        mu, logvar = enc_output[:, :self.latent_dim], enc_output[:, self.latent_dim:]

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decode latent representation
        decoded = self.decoder(z)[:, :, 7:-6, 3:-2]
        return decoded, mu, logvar

# # Test the VAE
# batch_size = 2
# channels = 3
# height = 1347
# width = 459
# input_tensor = torch.randn(batch_size, channels, height, width)

# # Instantiate the VAE
# vae = ConvVAE()

# # Pass the input tensor through the VAE
# output_tensor, mu, logvar = vae(input_tensor)
