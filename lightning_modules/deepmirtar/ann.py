import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

from lightning_modules.models.deepmirtar.ann import ANN


class AnnLM(pl.LightningModule):
    def __init__(self, pretrained_sda_path: str, x_key: str, y_key: str, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = ANN(pretrained_sda_path)
        self.x_key = x_key
        self.y_key = y_key
        self.lr = lr
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        self.accuracy(y_hat, y.int())
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss)
        self.accuracy(y_hat, y.int())
        self.log("val_acc", self.accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}]
