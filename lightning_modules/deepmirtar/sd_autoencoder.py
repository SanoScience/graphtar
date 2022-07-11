import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lightning_modules.models.deepmirtar.denoising_autoencoder import DenoisingAutoencoder


class SdAutoencoderLM(pl.LightningModule):
    def __init__(self, initial_da: DenoisingAutoencoder, x_key: str,
                 y_key: str, lr: float):
        super().__init__()
        self.sda = nn.ModuleList([initial_da])
        self.x_key = x_key
        self.y_key = y_key
        self.lr = lr

    def forward(self, x):
        x = self.sda[-1](x)
        return x

    def get_intermediate_x(self, x):
        with torch.no_grad():
            if len(self.sda) == 1:
                return torch.flatten(x, 1, -1)
            else:
                for module in self.sda[:-1]:
                    x = module.encoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        x_intermediary = self.get_intermediate_x(x)
        y_hat = self.forward(x_intermediary)
        loss = F.mse_loss(y_hat, x_intermediary)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        x_intermediary = self.get_intermediate_x(x)
        y_hat = self.forward(x_intermediary)
        loss = F.mse_loss(y_hat, x_intermediary)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-8)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}]

    def append_module(self, module: nn.Module):
        self.sda.append(module)
