import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

from lightning_modules.models.mitar.mitar_net import MitarNet


class MitarNetLM(pl.LightningModule):
    def __init__(self, n_embeddings: int, input_size: int, lr: float, x_key: str, y_key: str):
        super().__init__()
        self.model = MitarNet(n_embeddings=n_embeddings, input_size=input_size)
        self.x_key = x_key
        self.y_key = y_key
        self.lr = lr
        self.accuracy = Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = torch.squeeze(batch[self.x_key]), batch[self.y_key]
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        self.accuracy(y_hat, y.int())
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = torch.squeeze(batch[self.x_key]), batch[self.y_key]
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss)
        self.accuracy(y_hat, y.int())
        self.log("val_acc", self.accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-8)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}]
