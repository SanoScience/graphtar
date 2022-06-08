import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Linear(25, 2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier(encoded)
