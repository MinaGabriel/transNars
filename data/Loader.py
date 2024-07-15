import argparse
import numexpr as ne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from data.utils import check_folder_and_files  # Assuming check_folder_and_files is a function in data.utils
import logging
 
class Loader:
    def __init__(self, dataset_dir, neg_sample, epoch, lr, model, dim, neg_ratio, batch_size,max_threads, device, validation_rate):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.dataset_dir = dataset_dir
        self.neg_sample = neg_sample
        self.epoch = epoch
        self.lr = lr
        self.model_name = model
        self.embedding_dimension = dim
        self.neg_ratio = neg_ratio
        self.batch_size = batch_size
        self.device = device
        self.max_threads = max_threads
        self.validation_rate = validation_rate
        # Log the configuration
        self.logger.info("Starting training with the following parameters:")
        self.logger.info(f"Dataset directory: {dataset_dir}")
        self.logger.info(f"Negative Sample: {neg_sample}")
        self.logger.info(f"Epochs: {epoch}")
        self.logger.info(f"Learning rate (λ): {lr}")
        self.logger.info(f"Model: {model}")
        self.logger.info(f"Embedding dimension (κ): {dim}")
        self.logger.info(f"Negative sampling ratio: {neg_ratio}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Device: {device}")

        # Check if folder and the required files exist
        if not check_folder_and_files(dataset_dir):
            self.logger.error("Dataset directory or required files are missing.")
            raise FileNotFoundError("Dataset directory or required files are missing.")

        # Normalize the path to remove any trailing slashes
        normalized_path = os.path.normpath(dataset_dir)
        # Extract the basename
        self.name = os.path.basename(normalized_path)

        self.train_file_path = os.path.join(dataset_dir, 'train2id.txt')
        self.valid_file_path = os.path.join(dataset_dir, 'valid2id.txt')
        self.test_file_path = os.path.join(dataset_dir, 'test2id.txt')
        self.entity_file_path = os.path.join(dataset_dir, 'entity2id.txt')
        self.relation_file_path = os.path.join(dataset_dir, 'relation2id.txt')

        self.num_train = self.read_first_line(self.train_file_path)
        self.num_valid = self.read_first_line(self.valid_file_path)
        self.num_test = self.read_first_line(self.test_file_path)

        self.num_entities = self.read_first_line(self.entity_file_path)
        self.num_relations = self.read_first_line(self.relation_file_path)
    

    @staticmethod
    def read_first_line(file_path):
        with open(file_path) as file:
            return int(file.readline().strip())

    @classmethod
    def config(cls, config):
        return cls(
            config['dataset_dir'],
            config['neg_sample'],
            config['epoch'],
            config['lr'],
            config['model'],
            config['dim'],
            config['neg_ratio'],
            config['batch_size'],
            config['max_threads'],
            config['device'], 
            config['validation_rate']
            
        )
