
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from os import path
from collections import defaultdict
import yaml
import numpy as np



import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class TripletsDataset(Dataset):
    def __init__(self, dataset_dir):
        self.entities_txt = self.get_text(os.path.join(dataset_dir, 'entities.txt'))
        self.relation_txt = self.get_text(os.path.join(dataset_dir, 'relations.txt'))
        # Generate Entity and Relation IDs
        self.entity_ids = self.generate_id(os.path.join(dataset_dir, 'entities.txt'))
        self.relation_ids = self.generate_id(os.path.join(dataset_dir, 'relations.txt'))
    
        # Generate Test Train Validation mappings (E_str R_str E_str) --> (E_int R_int E_int) type is numpy 
        self.train_triplets_ids = self.convert_triplets_to_ids(os.path.join(dataset_dir,'train.txt'),self.entity_ids, self.relation_ids) 
        self.test_triplets_ids = self.convert_triplets_to_ids(os.path.join(dataset_dir,'test.txt'),self.entity_ids, self.relation_ids) 
        self.valid_triplets_ids = self.convert_triplets_to_ids(os.path.join(dataset_dir,'valid.txt'),self.entity_ids, self.relation_ids) 
        
        self.config = {
            'name': dataset_dir,
            'num_entities': len(self.entity_ids),
            'num_relations': len(self.relation_ids)
        }
    
    def get_dataloader(self, split, batch_size):
        # TODO: set the shuffle to true to see if this will improve training.
        if split == 'train':
            return DataLoader(self.train_triplets_ids, batch_size, shuffle=False) 
        elif split == 'test':
            return DataLoader(self.test_triplets_ids, batch_size, shuffle=False) 
        elif split == 'valid':
            return DataLoader(self.valid_triplets_ids, batch_size, shuffle=False) 
    def get_text(self, file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    def generate_id(self, file_path):
        with open(file_path, "r") as f:
            mapping = defaultdict(lambda: len(mapping))
            for line in f:
                val = line.strip()
                mapping[val]
        return mapping
    
    def convert_triplets_to_ids(self, file_path: str, entity_ids: Dict[str, int], relation_ids: Dict[str, int]) -> np.ndarray:
        subject_ids = []
        predicate_ids_list = []
        object_ids = []

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    subject, relation, object = parts
                    if subject in entity_ids and object in entity_ids and relation in relation_ids:
                        subject_ids.append(entity_ids[subject])
                        predicate_ids_list.append(relation_ids[relation])
                        object_ids.append(entity_ids[object])
                    else:
                        print(f"Warning: Skipping invalid triplet {line.strip()} in {file_path}")
                else:
                    print(f"Warning: Line does not contain exactly three elements: {line.strip()} in {file_path}")

        # Convert lists to numpy arrays
        subject_ids = np.array(subject_ids)
        predicate_ids = np.array(predicate_ids_list)
        object_ids = np.array(object_ids)

        # Stack arrays vertically to form a single array of triplets
        triplets_array = np.column_stack((subject_ids, predicate_ids, object_ids))

        return triplets_array

        

def check_folder_and_files(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        return f"Folder '{folder_path}' does not exist."

    # List of expected files
    required_files = ['entities.txt', 'relations.txt',
                      'test.txt', 'train.txt', 'valid.txt']

    # Check for the presence of each file
    missing_files = [file for file in required_files if not os.path.isfile(
        os.path.join(folder_path, file))]

    if missing_files:
        print(f"Missing files in '{folder_path}': " + ", ".join(missing_files))
        return False
    else:
        print(f"All required files are present in '{folder_path}'.")
        return True
    

def embedding_visualization(embedding_before, embedding_after, dataset:TripletsDataset):
    pass
    


