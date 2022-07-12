import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lightning_modules.models.miraw.autoencoder import Autoencoder


class AutoencoderLM(pl.LightningModule):
    def __init__(self, input_size: int, x_key, y_key, lr: float):
        super().__init__()
        self.x_key = x_key
        self.y_key = y_key
        self.lr = lr
        self.model = Autoencoder(input_size)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, torch.flatten(x, 1, -1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self(x)
        loss = F.mse_loss(y_hat, torch.flatten(x, 1, -1))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-8)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}]
