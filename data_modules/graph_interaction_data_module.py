from typing import Optional

from torch_geometric.loader import DataLoader

from data_modules.datasets.interaction_graph_dataset import InteractionGraphDataset
from data_modules.interaction_data_module import InteractionDataModule


class GraphInteractionDataModule(InteractionDataModule):
    def __init__(self, transforms_config: str):
        super().__init__(transforms_config)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            interaction_full = InteractionGraphDataset(self.dataset_config['csv_path'],
                                                       transform=self.dataset_config['transform'])
            self.initialize_train_val_splits(interaction_full)
        if stage == "test":
            raise NotImplementedError("Test stage not implemented")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.dataset_config['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.dataset_config['batch_size'])
