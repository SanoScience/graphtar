from typing import Optional

from torch_geometric.loader import DataLoader
from torchvision.transforms import Compose

from data_modules.datasets.interaction_graph_dataset import InteractionGraphDataset
from data_modules.interaction_data_module import InteractionDataModule


class GraphInteractionDataModule(InteractionDataModule):
    def __init__(self, transforms_config: str, batch_size: int, data_split_seed: int, transform: Compose = None):
        super().__init__(transforms_config, batch_size, data_split_seed)
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage == "test":
            if not self.transform:
                interaction_full = InteractionGraphDataset(self.dataset_config.csv_path,
                                                           transform=self.dataset_config.transform)
            else:
                interaction_full = InteractionGraphDataset(self.dataset_config.csv_path,
                                                           transform=self.transform)
            self.initialize_train_val_splits(interaction_full)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
