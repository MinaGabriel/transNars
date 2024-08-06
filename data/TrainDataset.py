from torch.utils.data import Dataset, DataLoader
from data.Loader import Loader
import numpy as np
import torch


class TrainDataset(Dataset):
    def __init__(self, data, loader: Loader):
        self.batch_h = data["batch_h"]
        self.batch_t = data["batch_t"]
        self.batch_r = data["batch_r"]
        self.batch_y = data["batch_y"]
        self.neg_ratio = loader.config['neg_ratio']
        self.batch_size = loader.config['batch_size']
        self.num_batches = (len(self.batch_y) + self.batch_size * (1 + self.neg_ratio) - 1) // (self.batch_size * (1 + self.neg_ratio))


    def __len__(self):
        # Number of batches
        return self.num_batches

    def __getitem__(self, idx):
        start = idx * self.batch_size * (1 + self.neg_ratio)
        end = start + self.batch_size * (1 + self.neg_ratio)

        batch_h = self.batch_h[start:end]
        batch_t = self.batch_t[start:end]
        batch_r = self.batch_r[start:end]
        batch_y = self.batch_y[start:end]

        # Split into positive and negative samples
        pos_indices = np.where(batch_y == 1)[0]
        neg_indices = np.where(batch_y == -1)[0]

        pos_h = torch.tensor(batch_h[pos_indices], dtype=torch.int64)
        pos_t = torch.tensor(batch_t[pos_indices], dtype=torch.int64)
        pos_r = torch.tensor(batch_r[pos_indices], dtype=torch.int64)
        pos_y = torch.tensor(batch_y[pos_indices], dtype=torch.float32)

        neg_h = torch.tensor(batch_h[neg_indices], dtype=torch.int64)
        neg_t = torch.tensor(batch_t[neg_indices], dtype=torch.int64)
        neg_r = torch.tensor(batch_r[neg_indices], dtype=torch.int64)
        neg_y = torch.tensor(batch_y[neg_indices], dtype=torch.float32)

        return (pos_h, pos_t, pos_r, pos_y), (neg_h, neg_t, neg_r, neg_y)
