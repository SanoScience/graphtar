import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(120, 200),
            nn.ReLU(),
            nn.Linear(200, 175),
            nn.ReLU(),
            nn.Linear(175, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 75),
            nn.ReLU(),
            nn.Linear(75, 150),
            nn.ReLU(),
            nn.Linear(150, 175),
            nn.ReLU(),
            nn.Linear(175, 200),
            nn.ReLU(),
            nn.Linear(200, 120),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
