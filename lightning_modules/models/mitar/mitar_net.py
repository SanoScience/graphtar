import torch.nn as nn


class MitarNet(nn.Module):
    def __init__(self, n_embeddings: int):
        super().__init__()
        kernel_size = 12
        embedding_dim = 5
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, 320, kernel_size),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.birnn = nn.LSTM((n_embeddings * embedding_dim - kernel_size + 1) // 2, 32, bidirectional=True,
                             batch_first=True)

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
