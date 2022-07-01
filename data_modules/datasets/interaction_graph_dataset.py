import numpy as np
import torch
from torch_geometric.data import Data

from data_modules.datasets.interaction_dataset import InteractionDataset


class InteractionGraphDataset(InteractionDataset):
    def __init__(self, csv_file_path: str, transform=None):
        super().__init__(csv_file_path, transform)

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        data_objects = []
        for mirna, target, label in zip(sample[self.mirna_key], sample[self.target_key], sample[self.label_key]):
            data_object = Data(
                x=torch.cat((mirna, target), dim=1),
                edge_index=self.get_edges(mirna, target),
                y=label
            )
            data_objects.append(data_object)
        return data_objects

    def get_edges(self, mirna, target):
        mirna_inter_edges = np.array((np.arange(len(mirna))[:-1], np.arange(len(mirna))[:-1] + 1))
        mirna_inter_edges = np.concatenate((mirna_inter_edges, mirna_inter_edges[[1, 0], :]), axis=1)
        target_inter_edges = mirna_inter_edges + len(mirna)
        shorter_len = len(mirna) if len(mirna) > len(target) else len(target)
        cross_edges = np.array((np.arange(shorter_len), np.arange(shorter_len) + shorter_len))
        cross_edges = np.concatenate((cross_edges, cross_edges[[1, 0], :]), axis=1)
        edges = np.concatenate((mirna_inter_edges, target_inter_edges, cross_edges), axis=1)
        return edges
