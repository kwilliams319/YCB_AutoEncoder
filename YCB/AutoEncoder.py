import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 674x230
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 337x115
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 169x58
        #     nn.ReLU(True),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 85x29
        #     nn.ReLU(True)
        # )
        
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 169x58
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 337x115
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 674x230
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1347x459
        #     nn.Sigmoid()
        # )

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
            nn.Linear(256 * 85 * 29, 32),  # Linear layer
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 256 * 85 * 29),  # Linear layer
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
        # print(f"x: {x.shape}")
        # encoded = self.encoder(x)
        # print(f"enc: {encoded.shape}")
        decoded = self.decoder(self.encoder(x))[:, :, 7:-6, 3:-2]
        # decoded = decoded[:, :, 7:-6, 3:-2]
        # print(f"dec: {decoded.shape}")
        # return self.decoder(self.encoder(x))
        return decoded   
    

# batch_size = 2
# channels = 3
# height = 1347
# width = 459
# input_tensor = torch.randn(batch_size, channels, height, width)

# # Instantiate the autoencoder
# autoencoder = ConvAutoencoder()

# # Pass the input tensor through the autoencoder
# output_tensor = autoencoder(input_tensor)