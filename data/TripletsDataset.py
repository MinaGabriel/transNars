import numexpr as ne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from data.utils import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Set the maximum number of threads for NumExpr
os.environ["NUMEXPR_MAX_THREADS"] = "20"  # or any number you prefer

print(f"NumExpr will use {ne.nthreads} threads")


class TripletsDataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.name = os.path.basename(dataset_dir)
        self.train_file_path = os.path.join(dataset_dir, 'train2id.txt')
        self.valid_file_path = os.path.join(dataset_dir, 'valid2id.txt')
        self.test_file_path = os.path.join(dataset_dir, 'test2id.txt')

        self.entity_map = generate_dictionary(
            os.path.join(dataset_dir, 'entity2id.txt'))
        self.relation_map = generate_dictionary(
            os.path.join(dataset_dir, 'relation2id.txt'))
        self.all_entity_ids = self.get_all_entity_ids()
        self.all_relation_ids = self.get_all_relation_ids()
        self.reverse_entity_map = {v: k for k, v in self.entity_map.items()}
        self.reverse_relation_map = {
            v: k for k, v in self.relation_map.items()}
        self.num_train = int(open(self.train_file_path).readline().strip())
        self.num_valid = int(open(self.valid_file_path).readline().strip())
        self.num_test = int(open(self.test_file_path).readline().strip())

        self.num_entities = len(self.entity_map)
        self.num_relations = len(self.relation_map)
        self.train_dataset, self.valid_dataset, self.test_dataset = self.load_datasets()
        self.all_possible_hs, self.all_possible_ts = self.get_observed_triples()

    def load_triplets_ids(self, file_path: str) -> np.ndarray:
        head = []
        relation = []
        tail = []

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    h, t, r = parts
                    head.append(h)
                    relation.append(r)
                    tail.append(t)
                else:
                    print(f"Warning: Line does not contain exactly three elements: {line.strip()} in {file_path}")

        head = np.array(head, dtype=np.int64)
        relation = np.array(relation, dtype=np.int64)
        tail = np.array(tail, dtype=np.int64)

        triplets_array = np.column_stack((head, relation, tail))

        return triplets_array

    def load_datasets(self):
        return TrainDataset(self), ValidDataset(self), TestDataset(self)

    def get_observed_triples(self):
        all_possible_hs = defaultdict(dict)
        all_possible_ts = defaultdict(dict)

        # Combine train, valid, and test datasets
        train = torch.as_tensor(self.train_dataset.triplets, dtype=torch.int32)
        valid = torch.as_tensor(self.valid_dataset.triplets, dtype=torch.int32)
        test = torch.as_tensor(self.test_dataset.triplets, dtype=torch.int32)
        all_triples = torch.cat((train, valid, test))
        X = all_triples.detach().clone()

        # Iterate through all triples and populate the dictionaries
        for triple in range(X.shape[0]):
            h, r, t = X[triple][0].item(
            ), X[triple][1].item(), X[triple][2].item()

            try:
                all_possible_hs[t][r].append(h)
            except KeyError:
                all_possible_hs[t][r] = [h]
            try:
                all_possible_ts[h][r].append(t)
            except KeyError:
                all_possible_ts[h][r] = [t]
        # Convert defaultdicts to dicts before returning
        return dict(all_possible_hs), dict(all_possible_ts)

    def get_all_entity_ids(self):
        return list(self.entity_map.values())

    def get_all_relation_ids(self):
        return list(self.relation_map.values())


class TrainDataset(Dataset):
    def __init__(self, dataset: TripletsDataset):
        self.triplets = dataset.load_triplets_ids(dataset.train_file_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.triplets[idx]
        return torch.tensor(sample, dtype=torch.long)


class ValidDataset(Dataset):
    def __init__(self, dataset: TripletsDataset):
        self.triplets = dataset.load_triplets_ids(dataset.valid_file_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.triplets[idx]
        return torch.tensor(sample, dtype=torch.long)


class TestDataset(Dataset):
    def __init__(self, dataset: TripletsDataset):
        self.triplets = dataset.load_triplets_ids(dataset.test_file_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.triplets[idx]
        return torch.tensor(sample, dtype=torch.long)


if __name__ == "__main__":
    dataset = TripletsDataset('datasets/FB15K237')
    print(f"All triplets tensor shape: {dataset.all_triplets_tensor.shape}")
    print(f"All triplets tensor device: {dataset.all_triplets_tensor.device}")
