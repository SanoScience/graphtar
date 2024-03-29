import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 350),
            nn.ReLU(),
            nn.Linear(350, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 150),
            nn.ReLU(),
            nn.Linear(150, 300),
            nn.ReLU(),
            nn.Linear(300, 350),
            nn.ReLU(),
            nn.Linear(350, 400),
            nn.ReLU(),
            nn.Linear(400, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
