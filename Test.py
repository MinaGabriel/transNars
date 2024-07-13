from data.TestDataLoader import TestDataLoader
import argparse
from datetime import datetime
from tqdm import tqdm 
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.autograd import Variable
from data.utils import check_folder_and_files
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from sklearn.utils import shuffle
import logging
from TransE import TransE
import matplotlib.pyplot as plt
from data.Loader import * 
from torch.utils.data import DataLoader
from data.TrainDataLoader import TrainDataLoader
from data.Loader import Loader
import ctypes
from ctypes import c_char_p, POINTER, c_int64, c_float 

# Set random seeds for reproducibility
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Test: 
    def __init__(self, dataset: TestDataLoader):
        self.dataset = dataset.data 
        self.loader = dataset.loader
        self.device = dataset.loader.device 
        self.model = TransE(self.loader.num_entities, self.loader.num_relations, self.loader.embedding_dimension, self.device)
        self.model.load_state_dict(torch.load(f'models/TransE_{self.loader.name}.pt', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.lib = ctypes.CDLL("./Base.so")
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64]  
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64]    

    def test_head(self, e, r, t):
        # FIXME: but mode head or tail 
        scores = self.model(e, r, t)
        return scores

    def test_tail(self, h, r, e):
        # FIXME: but mode head or tail 
        scores = self.model(h, r, e)
        return scores

    def run_link_prediction(self):
        num_entities = self.loader.num_entities
        for _, (batch_h, batch_t, batch_r ) in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            batch_h = batch_h.to(self.device, non_blocking=True)  # Move data to GPU
            batch_r = batch_r.to(self.device, non_blocking=True)
            batch_t = batch_t.to(self.device, non_blocking=True)
            
            
            h = batch_h.clone().detach()
            r = batch_r.clone().detach()
            t = batch_t.clone().detach()
            
            heads = h.repeat_interleave(num_entities)
            relations = r.repeat_interleave(num_entities)
            tails = t.repeat_interleave(num_entities)
            
            entities = torch.arange(num_entities, device=self.device)
            entities = entities.repeat(len(h))

            scores_h = self.test_head(entities, relations, tails)
            score = scores_h.squeeze().detach().cpu().numpy()
            self.lib.testHead(score.__array_interface__["data"][0], _)

            scores_t = self.test_tail(heads, relations, entities)
            score = scores_t.squeeze().detach().cpu().numpy()
            self.lib.testTail(score.__array_interface__["data"][0], _)

        self.lib.test_link_prediction()
            
            

            


# Example usage
if __name__ == "__main__":
    loader = Loader("./datasets/FB15K237/", "c", 100,
                    0.01, "TransE", 200, 25, 1, "cuda:0")
    print(loader.name)
    test_dataloader = TestDataLoader(loader)
  
    tester = Test(test_dataloader)
    tester.run_link_prediction()
    