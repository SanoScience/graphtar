import torch.nn as nn


class MitarNet(nn.Module):
    def __init__(self, n_embeddings: int):
        super().__init__()
        self.embedding = nn.Embedding(n_embeddings, 5)
        self.conv = nn.Sequential(
            nn.Conv1d(5, 320, 12),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.birnn = nn.LSTM(169, 32, bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(20480, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv(x.permute(0, 2, 1))
        x = self.birnn(x)[0]
        return self.classifier(x)
