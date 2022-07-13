import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score

from lightning_modules.models.miraw.ann import ANN
from lightning_modules.models.miraw.autoencoder import Autoencoder


class AnnLM(pl.LightningModule):
    def __init__(self, x_key, y_key, lr: float, encoder: Autoencoder):
        super().__init__()
        self.x_key = x_key
        self.y_key = y_key
        self.lr = lr
        self.model = ANN(encoder)
        self.accuracy = Accuracy()
        self.f1 = F1Score(1)

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
        self.accuracy(y_hat, y.int())
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss)
        self.accuracy(y_hat, y.int())
        self.log("val_acc", self.accuracy)
        self.f1(y_hat, y.int())
        self.log("val_f1", self.f1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-8)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}]
