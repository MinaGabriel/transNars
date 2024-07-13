from torch.utils.data import Dataset, DataLoader
from data.Loader import Loader
import numpy as np
import torch


class TestDataset(Dataset):
    def __init__(self, data, loader: Loader):
        self.batch_h = data["batch_h"]
        self.batch_t = data["batch_t"]
        self.batch_r = data["batch_r"]
        self.loader = loader
        self.batch_size = loader.batch_size
        self.num_batches = (len(self.batch_h) // self.batch_size) + (1 if len(self.batch_h) % self.batch_size != 0 else 0)


    def __len__(self):
        # Number of batches
        return self.num_batches

    def __getitem__(self, idx):
        # Calculate start and end indices for the batch
        start_idx = idx * self.loader.batch_size
        end_idx = start_idx + self.loader.batch_size

        # Retrieve the batch data
        batch_h = torch.tensor(self.batch_h[start_idx:end_idx], dtype=torch.int64)
        batch_t = torch.tensor(self.batch_t[start_idx:end_idx], dtype=torch.int64)
        batch_r = torch.tensor(self.batch_r[start_idx:end_idx], dtype=torch.int64)
        return batch_h, batch_t, batch_r
