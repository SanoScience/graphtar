import torch
import torch.nn as nn


class MitarNet(nn.Module):
    def __init__(self, n_embeddings: int, input_size: int):
        super().__init__()
        kernel_size = 12
        n_filters = 320
        lstm_hidden_size = 32
        self.embedding_dim = 5
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * n_embeddings, input_size * n_embeddings),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(self.embedding_dim, n_filters, kernel_size),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.birnn = nn.LSTM(n_filters, lstm_hidden_size, bidirectional=True,
                             batch_first=True)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear((input_size - (kernel_size - 1)) // 2 * 2 * lstm_hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = torch.reshape(x, (x.shape[0], self.embedding_dim, x.shape[1] // self.embedding_dim))
        x = self.conv(x)
        x = self.birnn(x.permute(0, 2, 1))[0]
        return self.classifier(x)
