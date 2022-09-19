import numpy as np
import torch
from torch_geometric.data import Data

from data_modules.datasets.interaction_dataset import InteractionDataset


class InteractionGraphDataset(InteractionDataset):
    def __init__(self, csv_file_path: str, transform=None):
        super().__init__(csv_file_path, transform)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        data_objects = []
        for mirna, target, label in zip(sample[self.mirna_key], sample[self.target_key], sample[self.label_key]):
            edge_index, edge_features = self.get_edges(mirna, target)
            data_object = Data(
                x=torch.as_tensor(torch.cat((mirna, target), dim=0), dtype=torch.float),
                edge_index=edge_index,
                y=torch.as_tensor(label, dtype=torch.float)
            )
            data_objects.append(data_object)
        return data_objects

    def get_edges(self, mirna, target):
        mirna_inter_edges = self.get_inter_edges(mirna)
        mirna_inter_edges_features = np.tile([1, 0], (mirna_inter_edges.shape[1], 1))
        target_inter_edges = self.get_inter_edges(target) + len(mirna)
        target_inter_edges_features = np.tile([1, 0], (target_inter_edges.shape[1], 1))
        shorter_len = len(mirna) if len(mirna) < len(target) else len(target)
        cross_edges = np.array((np.arange(shorter_len), np.arange(shorter_len) + shorter_len))
        cross_edges = np.concatenate((cross_edges, cross_edges[[1, 0], :]), axis=1)
        cross_edges_features = np.tile([0, 1], (cross_edges.shape[1], 1))
        edges = np.concatenate((mirna_inter_edges, target_inter_edges, cross_edges), axis=1)
        edge_features = np.concatenate((mirna_inter_edges_features, target_inter_edges_features, cross_edges_features),
                                       axis=0)
        return torch.as_tensor(edges, dtype=torch.long), torch.as_tensor(edge_features, dtype=torch.long)

    def get_inter_edges(self, sequence):
        inter_edges = np.array((np.arange(len(sequence))[:-1], np.arange(len(sequence))[:-1] + 1))
        inter_edges = np.concatenate((inter_edges, inter_edges[[1, 0], :]), axis=1)
        return inter_edges
