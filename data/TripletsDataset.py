import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from data.utils import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class TripletsDataset(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.name = os.path.basename(dataset_dir)
        self.train_file_path = os.path.join(dataset_dir, f'train2id.txt')
        self.valid_file_path = os.path.join(dataset_dir, f'valid2id.txt')
        self.test_file_path = os.path.join(dataset_dir, f'test2id.txt') 
        self.entity_dictionary = generate_dictionary(
            os.path.join(dataset_dir, 'entity2id.txt'))
        self.relation_dictionary = generate_dictionary(
            os.path.join(dataset_dir, 'relation2id.txt'))
        self.num_train = int(open(self.train_file_path).readline().strip())
        self.num_valid = int(open(self.valid_file_path).readline().strip())
        self.num_test = int(open(self.test_file_path).readline().strip())
        self.num_entities = len(self.entity_dictionary)
        self.num_relations = len(self.relation_dictionary)
        
        
        self.train_dataset = TrainDataset(self)
        self.valid_dataset = ValidDataset(self)



class TrainDataset(DataLoader):
    def __init__(self, dataset: TripletsDataset):
        self.triplets = load_triplets_ids(dataset.train_file_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.triplets[idx]
        return torch.tensor(sample, dtype=torch.long)

class ValidDataset(DataLoader):
    def __init__(self, dataset: TripletsDataset):
        self.triplets = load_triplets_ids(dataset.valid_file_path)
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.triplets[idx]
        return torch.tensor(sample, dtype=torch.long)
        