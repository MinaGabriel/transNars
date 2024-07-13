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


class Loader(object):
    def __init__(self, dataset_dir, neg_sample,
                 epoch, lr, model, dim, neg_ratio, batch_size, device):
        self.dataset_dir = dataset_dir
        self.neg_sample = neg_sample
        self.epoch = epoch
        self.lr = lr
        self.model_name = model
        self.embedding_dimension = dim
        self.neg_ratio = neg_ratio
        self.batch_size = batch_size
        self.device = device

        # Normalize the path to remove any trailing slashes
        normalized_path = os.path.normpath(dataset_dir)
        # Extract the basename
        self.name = os.path.basename(normalized_path)

        self.train_file_path = os.path.join(dataset_dir, 'train2id.txt')
        self.valid_file_path = os.path.join(dataset_dir, 'valid2id.txt')
        self.test_file_path = os.path.join(dataset_dir, 'test2id.txt')
        self.entity_file_path = os.path.join(dataset_dir, 'entity2id.txt')
        self.relation_file_path = os.path.join(dataset_dir, 'relation2id.txt')

        self.num_train = int(open(self.train_file_path).readline().strip())
        self.num_valid = int(open(self.valid_file_path).readline().strip())
        self.num_test = int(open(self.test_file_path).readline().strip())

        self.num_entities = int(open(self.entity_file_path).readline().strip())
        self.num_relations = int(
            open(self.relation_file_path).readline().strip())
