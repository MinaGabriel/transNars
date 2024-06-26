from torch.utils.data import Dataset, DataLoader
from data.TripletsDataset import TripletsDataset
import numexpr as ne
import numpy as np
import torch
import os
from data.utils import *
import logging


class NegativeDataset(Dataset):
    def __init__(self, dataset: TripletsDataset, filename):
        self.neg_sam_file_path = os.path.join(dataset.dataset_dir, filename)
        self.triplets = dataset.load_triplets_ids(self.neg_sam_file_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.triplets[idx]
        return torch.tensor(sample, dtype=torch.long)
