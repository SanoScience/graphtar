import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(self, csv_file_path: str, transform=None):
        self.data_df = pd.read_csv(csv_file_path, index_col=0)
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.data_df.iloc[idx, [1, 3, 4]].to_numpy()
        sample = {'mirna': item[:, 0], 'target': item[:, 1], 'label': item[:, 2]}
        if self.transform:
            sample = self.transform(sample)
        return sample
