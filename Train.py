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
from data.ValidDataLoader import ValidDataLoader
# Set random seeds for reproducibility
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Train:
    def __init__(self, dataset: TrainDataLoader):
        self.dataset = dataset.data
        self.lr = dataset.loader.lr
        self.loader = dataset.loader
        self.device = torch.device(
            dataset.loader.device if torch.cuda.is_available() else 'cpu')
        self.batch_size = dataset.loader.batch_size
        self.gamma = 5.0
        self.lambda_reg = 2.0
        self.weight_decay = 0
        self.neg_ratio = dataset.loader.neg_ratio
        self.epoch = dataset.loader.epoch
        self.best_loss = float('inf')
        self.best_model_path = f"models/{dataset.loader.model_name}_{dataset.loader.name}.pt"
        if dataset.loader.model_name == 'TransE':
            self.model = TransE(dataset.loader.num_entities,
                                dataset.loader.num_relations, dataset.loader.embedding_dimension, dataset.loader.device)

        self.model.to(self.device)
        
        # get validation data
        self.test_dataloader = ValidDataLoader(self.loader)
        
    def validation(self):
        return 
        for _, batch in enumerate(self.test_dataloader.data): 
            print(_)
            

    def start(self):
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        training_range = tqdm(range(self.epoch))
        epoch_losses = []
        validation_losses = []

        for epoch in training_range:
            total_loss = 0.0
            num_batches = 0

            for _, batch in enumerate(self.dataset): 
                optimizer.zero_grad()
                (pos_h, pos_t, pos_r, pos_y), (neg_h, neg_t, neg_r, neg_y) = batch


                positive_scores = self.model(pos_h, pos_r, pos_t)
                negative_scores = self.model(neg_h, neg_r, neg_t)

                training_loss = self.model.pairwise_hinge_loss(
                    positive_scores, negative_scores, self.gamma)

                
                
                heads = torch.vstack((pos_h.view(-1,1), neg_h.view(-1,1))).to( self.device)
                relations = torch.vstack((pos_r.view(-1,1), neg_r.view(-1,1))).to( self.device)
                tails = torch.vstack((pos_t.view(-1,1), neg_t.view(-1,1))).to( self.device)
                reg_loss = self.model.regularization_loss(heads, relations, tails)

                total_batch_loss = training_loss + self.lambda_reg * reg_loss
                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()
                num_batches += 1
            
            self.validation()

            average_loss = total_loss / num_batches
            epoch_losses.append(average_loss)
 
            training_range.set_description(
                f"Epoch {epoch + 1} | Avg Training Loss: {average_loss:.4f}")

            if average_loss < self.best_loss:
                self.best_loss = average_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"New best model saved with training loss {
                            average_loss:.4f}")
 
