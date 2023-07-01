from typing import Tuple, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torchmetrics import Accuracy, F1Score

from lightning_modules.models.graphtar.gnn import GNN, LayerType, GlobalPoolingType


class GnnLM(pl.LightningModule):
    def __init__(
        self,
        layer_type: LayerType,
        graph_layer_sizes: List[Tuple[int, int]],
        global_pooling: GlobalPoolingType,
        hidden_layer_sizes: List[Tuple[int, int]],
        dropout_rate: float,
        lr: float,
    ):
        super().__init__()
        self.model = GNN(
            layer_type,
            graph_layer_sizes,
            global_pooling,
            hidden_layer_sizes,
            dropout_rate,
        )
        self.lr = lr
        self.accuracy = Accuracy()
        self.f1 = F1Score(1)

    def forward(self, data: Data):
        return self.model(data.x, data.edge_index, data.batch)

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0])
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(y_hat), batch[0].y)
        self.log("train_loss", loss)
        self.accuracy(torch.squeeze(y_hat), batch[0].y.int())
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0])
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(y_hat), batch[0].y)
        self.log("val_loss", loss)
        self.accuracy(torch.squeeze(y_hat), batch[0].y.int())
        self.log("val_acc", self.accuracy)
        self.f1(torch.squeeze(y_hat), batch[0].y.int())
        self.log("val_f1", self.f1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=0.1, patience=10, min_lr=1e-8
        )
        return [optimizer], [
            {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}
        ]
