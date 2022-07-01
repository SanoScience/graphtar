import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from lightning_modules.models.miraw.ann import ANN
from lightning_modules.models.miraw.autoencoder import Autoencoder


class AnnLM(pl.LightningModule):
    def __init__(self, x_key, y_key, encoder: Autoencoder):
        super().__init__()
        self.x_key = x_key
        self.y_key = y_key
        self.model = ANN(encoder)

    def forward(self, x):
        with torch.no_grad():
            x = self.model.encoder(x)
        x = self.model.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
