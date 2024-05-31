
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from os import path
from collections import defaultdict
import yaml
import numpy as np
import re


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

  

def check_folder_and_files(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        return f"Folder '{folder_path}' does not exist."

    # List of expected files
    required_files = ['entity2id.txt', 'relation2id.txt',
                      'test2id.txt', 'train2id.txt', 'valid2id.txt']

    # Check for the presence of each file
    missing_files = [file for file in required_files if not os.path.isfile(
        os.path.join(folder_path, file))]

    if missing_files:
        print(f"Missing files in '{folder_path}': " + ", ".join(missing_files))
        return False
    else:
        print(f"All required files are present in '{folder_path}'.")
        return True
    

"""
takes a file_path and returns a dictionary of {"entity_name": entity_id} or {"relation_name": relation_id}
"""
def generate_dictionary(file_path: str) -> dict:
    dictionary = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Use regex to split based on any sequence of whitespace
            parts = re.split(r'\s+', line.strip())
            if len(parts) == 2:
                value, id = parts
                try:
                    id = int(id)
                    dictionary[value] = id
                except ValueError:
                    print(f"Warning: ID is not an integer in line: {line.strip()}")
            else:
                print(f"Warning: Line does not contain exactly two elements: {line.strip()}")
    return dictionary

def load_triplets_ids(file_path: str) -> np.ndarray:
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
