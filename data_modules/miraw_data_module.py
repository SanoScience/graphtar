import math
from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from data_modules.datasets.interaction_dataset import InteractionDataset
from data_modules.datasets.transforms.merge import Merge
from data_modules.datasets.transforms.pad import Pad
from data_modules.datasets.transforms.to_one_hot import ToOneHot
from data_modules.datasets.transforms.to_tensor import ToTensor


class InteractionDataModule(pl.LightningDataModule):

    def __init__(self, csv_file_path: str, batch_size: int, train_val_ratio: Tuple[float, float]):
        super().__init__()
        self.csv_file_path = csv_file_path
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.x_key = "merged"
        self.y_key = "label"
        self.transform = transforms.Compose([
            Pad(40, "target", "N"),
            Pad(30, "mirna", "N"),
            Merge(merged_key="merged", keys_to_merge=("mirna", "target")),
            ToOneHot({'A': 0, 'U': 1, 'T': 1, 'G': 2, 'C': 3, 'N': 4}, self.x_key),
            ToTensor(keys=(self.x_key, self.y_key))
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            interaction_full = InteractionDataset(self.csv_file_path, transform=self.transform)
            train_len = math.floor(self.train_val_ratio[0] * len(interaction_full))
            self.train_data, self.val_data = random_split(interaction_full,
                                                          [train_len, len(interaction_full) - train_len])
        if stage == "test":
            raise NotImplementedError("Test stage not implemented.")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def get_batch_keys(self):
        return self.x_key, self.y_key
