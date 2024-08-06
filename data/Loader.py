import argparse
import numexpr as ne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Assuming check_folder_and_files is a function in data.utils
from data.utils import check_folder_and_files
import logging
import yaml


class Loader:
    def __init__(self, config_file):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        self.logger = logging.getLogger(__name__)
        with open(f"./models/{config_file}", "r") as file:
            self.config = yaml.safe_load(file)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset_dir = self.config["dataset"]

        # Check if folder and the required files exist
        if not check_folder_and_files(dataset_dir):
            self.logger.error("Dataset directory or required files are missing.")
            raise FileNotFoundError("Dataset directory or required files are missing.")

        # Normalize the path to remove any trailing slashes
        normalized_path = os.path.normpath(dataset_dir)
        # Extract the basename
        self.name = os.path.basename(normalized_path)

        self.train_file_path = os.path.join(dataset_dir, "train2id.txt")
        self.valid_file_path = os.path.join(dataset_dir, "valid2id.txt")
        self.test_file_path = os.path.join(dataset_dir, "test2id.txt")
        self.entity_file_path = os.path.join(dataset_dir, "entity2id.txt")
        self.relation_file_path = os.path.join(dataset_dir, "relation2id.txt")

        self.num_train = self.read_first_line(self.train_file_path)
        self.num_valid = self.read_first_line(self.valid_file_path)
        self.num_test = self.read_first_line(self.test_file_path)

        self.num_entities = self.read_first_line(self.entity_file_path)
        self.num_relations = self.read_first_line(self.relation_file_path)

    @staticmethod
    def read_first_line(file_path):
        with open(file_path) as file:
            return int(file.readline().strip())
