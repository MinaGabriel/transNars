from torch.utils.data import Dataset, DataLoader
from data.Loader import Loader
import numpy as np
import torch


class ValidDataset(Dataset):
    def __init__(self, data):
        self.batch_h = data["batch_h"]
        self.batch_t = data["batch_t"]
        self.batch_r = data["batch_r"]

    def __len__(self):
        # Number of samples
        return len(self.batch_h)

    def __getitem__(self, idx):
        # Retrieve the sample data
        batch_h = torch.tensor([self.batch_h[idx]], dtype=torch.int64)
        batch_t = torch.tensor([self.batch_t[idx]], dtype=torch.int64)
        batch_r = torch.tensor([self.batch_r[idx]], dtype=torch.int64)
        return batch_h, batch_t, batch_r
