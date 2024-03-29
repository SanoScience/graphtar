import json
import math
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader

from data_modules.configs.dataset_config import DatasetConfigDecoder, DatasetConfig
from data_modules.datasets.interaction_dataset import InteractionDataset


class InteractionDataModule(pl.LightningDataModule):
    def __init__(self, transforms_config: str, batch_size: int, data_split_seed: int):
        super().__init__()
        with open(transforms_config, "r") as f:
            self.dataset_config = DatasetConfig(
                **json.load(f, object_hook=DatasetConfigDecoder.object_hook)
            )
        self.batch_size = batch_size
        self.data_split_seed = data_split_seed

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage == "test":
            interaction_full = InteractionDataset(
                self.dataset_config.csv_path, transform=self.dataset_config.transform
            )
            self.initialize_train_val_splits(interaction_full)

    def initialize_train_val_splits(self, interaction_full):
        train_len = math.floor(
            self.dataset_config.train_val_ratio[0] * len(interaction_full)
        )
        self.train_data, val_test_data = random_split(
            interaction_full,
            [train_len, len(interaction_full) - train_len],
            generator=torch.Generator().manual_seed(self.data_split_seed),
        )
        val_len = math.floor(
            self.dataset_config.train_val_ratio[1] * len(interaction_full)
        )
        self.val_data, self.test_data = random_split(
            val_test_data,
            [val_len, len(val_test_data) - val_len],
            generator=torch.Generator().manual_seed(self.data_split_seed),
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def get_batch_keys(self):
        return self.dataset_config.x_key, self.dataset_config.y_key
