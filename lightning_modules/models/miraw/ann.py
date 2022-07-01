import torch.nn as nn

from lightning_modules.models.miraw.autoencoder import Autoencoder


class ANN(nn.Module):
    def __init__(self, encoder: Autoencoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier(encoded)
