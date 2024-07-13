
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
    

