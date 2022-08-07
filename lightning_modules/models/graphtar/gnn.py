from enum import Enum
from functools import partial
from typing import Tuple, List

import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn.glob.glob import global_add_pool, global_max_pool, global_mean_pool


class LayerType(Enum):
    GCN = GCNConv
    GRAPHSAGE = SAGEConv
    GAT = GATConv


class GlobalPoolingType(Enum):
    MEAN = partial(global_mean_pool)
    ADD = partial(global_add_pool)
    MAX = partial(global_max_pool)


class GNN(nn.Module):
    def __init__(self, layer_type: LayerType, graph_layer_sizes: List[Tuple[int, int]],
                 global_pooling: GlobalPoolingType, hidden_layer_sizes: List[Tuple[int, int]], dropout_rate: float):
        super().__init__()

        self.graph_layers = ModuleList([layer_type.value(sizes[0], sizes[1]) for sizes in graph_layer_sizes])
        self.global_pooling_fn = global_pooling
        self.dropout_rate = dropout_rate
        self.classifier = nn.Sequential(
            *[self.get_classifier_unit(size) for size in hidden_layer_sizes],
            nn.Linear(hidden_layer_sizes[-1][1], 1)
        )

    def forward(self, x, edge_index, batch):
        for layer in self.graph_layers:
            x = layer(x, edge_index)
            x = x.relu()
        x = self.global_pooling_fn.value(x, batch)
        x = self.classifier(x)
        return x

    def get_classifier_unit(self, hidden_layer_size: Tuple[int, int]):
        return nn.Sequential(
            nn.Linear(hidden_layer_size[0], hidden_layer_size[1]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
