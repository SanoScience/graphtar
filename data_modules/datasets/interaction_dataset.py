import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(self, csv_file_path: str, transform=None):
        self.data_df = pd.read_csv(csv_file_path, index_col=0)
        self.transform = transform
        self.mirna_key = 'mirna'
        self.target_key = 'target'
        self.label_key = 'label'

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            mirna = self.data_df.iloc[idx, 1].to_numpy()
            target = self.data_df.iloc[idx, 3].to_numpy()
            label = self.data_df.iloc[idx, 4].to_numpy()
        else:
            mirna = np.expand_dims(np.array(self.data_df.iloc[idx, 1]), axis=0)
            target = np.expand_dims(np.array(self.data_df.iloc[idx, 3]), axis=0)
            label = np.expand_dims(np.array(self.data_df.iloc[idx, 4]), axis=0)
        sample = {self.mirna_key: mirna, self.target_key: target, self.label_key: label}
        if self.transform:
            sample = self.transform(sample)
        return sample
