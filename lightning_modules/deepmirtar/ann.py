import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from lightning_modules.models.deepmirtar.ann import ANN


class AnnLM(pl.LightningModule):
    def __init__(self, pretrained_sda_path: str, x_key: str, y_key: str):
        super().__init__()
        self.model = ANN(pretrained_sda_path)
        self.x_key = x_key
        self.y_key = y_key

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
