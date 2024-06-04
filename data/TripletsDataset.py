import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from data.utils import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

import os

# Set the maximum number of threads for NumExpr
os.environ["NUMEXPR_MAX_THREADS"] = "20"  # or any number you prefer

import numexpr as ne
print(f"NumExpr will use {ne.nthreads} threads")

class TripletsDataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.name = os.path.basename(dataset_dir)
        self.train_file_path = os.path.join(dataset_dir, 'train2id.txt')
        self.valid_file_path = os.path.join(dataset_dir, 'valid2id.txt')
        self.test_file_path = os.path.join(dataset_dir, 'test2id.txt')
        self.entity_map = generate_dictionary(os.path.join(dataset_dir, 'entity2id.txt'))
        self.relation_map = generate_dictionary(os.path.join(dataset_dir, 'relation2id.txt'))
        self.num_train = int(open(self.train_file_path).readline().strip())
        self.num_valid = int(open(self.valid_file_path).readline().strip())
        self.num_test = int(open(self.test_file_path).readline().strip())
        self.num_entities = len(self.entity_map)
        self.num_relations = len(self.relation_map)
        self.train_dataset, self.valid_dataset, self.test_dataset = self.load_datasets()

        self.every_relationships = self.build_relationships()
        
    def load_triplets_ids(self, file_path: str) -> np.ndarray:
        head = []
        relation = []
        tail = []

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    # the benchmark dataset is in the form of <subject> <object> <relation>
                    h, t, r = parts
                    head.append(h)
                    relation.append(r)
                    tail.append(t)
                else:
                    print(f"Warning: Line does not contain exactly three elements: {
                        line.strip()} in {file_path}")

        # Convert lists to numpy arrays of integers
        head = np.array(head, dtype=np.int64)
        relation = np.array(relation, dtype=np.int64)
        tail = np.array(tail, dtype=np.int64)

        # Stack arrays vertically to form a single array of triplets
        triplets_array = np.column_stack((head, relation, tail))

        return triplets_array


    def load_datasets(self):
        return TrainDataset(self), ValidDataset(self), TestDataset(self)

    def build_relationships(self):
        x = {}
        for dataset in [self.train_dataset, self.valid_dataset, self.test_dataset]:
            for idx in range(len(dataset)):
                triplet = dataset[idx]
                h, r, t = triplet.tolist()
                if (h, r) not in x:
                    x[(h, r)] = []
                x[(h, r)].append(t)
        return x

class TrainDataset(DataLoader):
    def __init__(self, dataset: TripletsDataset):
        self.triplets = dataset.load_triplets_ids(dataset.train_file_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.triplets[idx]
        return torch.tensor(sample, dtype=torch.long)


class ValidDataset(DataLoader):
    def __init__(self, dataset: TripletsDataset):
        self.triplets = dataset.load_triplets_ids(dataset.valid_file_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.triplets[idx]
        return torch.tensor(sample, dtype=torch.long)


class TestDataset(DataLoader):
    def __init__(self, dataset: TripletsDataset):
        self.triplets = dataset.load_triplets_ids(dataset.test_file_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.triplets[idx]
        return torch.tensor(sample, dtype=torch.long)
