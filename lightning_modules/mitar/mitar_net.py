import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from lightning_modules.models.mitar.mitar_net import MitarNet


class MitarNetLM(pl.LightningModule):
    def __init__(self, n_embeddings: int, input_size: int, x_key: str, y_key: str):
        super().__init__()
        self.model = MitarNet(n_embeddings=n_embeddings, input_size=input_size)
        self.x_key = x_key
        self.y_key = y_key

    def training_step(self, batch, batch_idx):
        x, y = torch.squeeze(batch[self.x_key]), batch[self.y_key]
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = torch.squeeze(batch[self.x_key]), batch[self.y_key]
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
