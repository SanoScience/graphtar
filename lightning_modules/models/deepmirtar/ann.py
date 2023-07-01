import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, pretrained_sda_path: str):
        super().__init__()
        self.sda = torch.load(pretrained_sda_path)
        self.classifier = nn.Linear(
            list(self.sda[-1].children())[0][-1].out_features, 1
        )

    def forward(self, x):
        for da in self.sda:
            x = da.encoder(x)
        return self.classifier(x)
