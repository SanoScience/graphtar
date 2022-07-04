from typing import Tuple, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from lightning_modules.models.graphtar.gnn import GNN, LayerType, GlobalPoolingType


class GnnLM(pl.LightningModule):
    def __init__(self, layer_type: LayerType, graph_layer_sizes: List[Tuple[int, int]],
                 global_pooling: GlobalPoolingType, hidden_layer_sizes: List[Tuple[int, int]], dropout_rate: float):
        super().__init__()
        self.model = GNN(layer_type, graph_layer_sizes, global_pooling, hidden_layer_sizes, dropout_rate)

    def forward(self, data: Data):
        return self.model(data.x, data.edge_index, data.batch)

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0])
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(y_hat), batch[0].y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch[0])
        loss = F.binary_cross_entropy_with_logits(torch.squeeze(y_hat), batch[0].y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
