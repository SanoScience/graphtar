import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(input_size, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.dropout(encoded)
        return self.decoder(encoded)
