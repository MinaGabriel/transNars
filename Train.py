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
        self.patience = 5
        self.best_model_path = f"models/{dataset.loader.model_name}_{dataset.loader.name}.pt"
        if dataset.loader.model_name == 'TransE':
            self.model = TransE(dataset.loader.num_entities,
                                dataset.loader.num_relations, dataset.loader.embedding_dimension, dataset.loader.device)

        self.model.to(self.device)
        
        # get validation data
        self.valid_dataset = ValidDataLoader(self.loader).data
        
    def validation(self):
        # Using the MRR for validation
        self.model.eval()
        raw_ranks_h, raw_ranks_t = [], []
        num_entities = self.loader.num_entities
        with torch.no_grad():
            for _, (batch_h, batch_t, batch_r) in enumerate(self.valid_dataset):
                batch_h = batch_h.to(self.device, non_blocking=True)  # Move data to GPU
                batch_r = batch_r.to(self.device, non_blocking=True)
                batch_t = batch_t.to(self.device, non_blocking=True)
                
                h = batch_h.clone().detach()
                r = batch_r.clone().detach()
                t = batch_t.clone().detach()
                
                # Compute scores for head entities
                heads = h.repeat_interleave(num_entities)
                relations = r.repeat_interleave(num_entities)
                tails = t.repeat_interleave(num_entities)
                
                entities = torch.arange(num_entities, device=self.device)
                entities = entities.repeat(len(h))
                
                # Scores for head entities
                scores_h = self.model(entities, relations, tails)
                scores_h = scores_h.view(len(h), num_entities)
                sorted_scores_h, sorted_indices_h = torch.sort(scores_h, descending=False)
                true_head_indices = h.view(-1, 1)
                ranks_h = (sorted_indices_h == true_head_indices).nonzero(as_tuple=False)[:, 1] + 1  # 1-based rank
                raw_ranks_h.extend(ranks_h.cpu().numpy())

                # Compute scores for tail entities
                heads = h.repeat_interleave(num_entities)
                relations = r.repeat_interleave(num_entities)
                tails = t.repeat_interleave(num_entities)
                
                # Scores for tail entities
                scores_t = self.model(heads, relations, entities)
                scores_t = scores_t.view(len(t), num_entities)
                sorted_scores_t, sorted_indices_t = torch.sort(scores_t, descending=False)
                true_tail_indices = t.view(-1, 1)
                ranks_t = (sorted_indices_t == true_tail_indices).nonzero(as_tuple=False)[:, 1] + 1  # 1-based rank
                raw_ranks_t.extend(ranks_t.cpu().numpy())

        # Compute mean ranks
        mean_rank_h = sum(raw_ranks_h) / len(raw_ranks_h)
        mean_rank_t = sum(raw_ranks_t) / len(raw_ranks_t)
        mean_rank = (mean_rank_h + mean_rank_t) / 2
        return mean_rank


    def start(self):
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        training_range = tqdm(range(self.epoch))
        epoch_losses = []
        validation_losses = []
        mean_rank = float('inf')

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
            
            average_loss = total_loss / num_batches
            epoch_losses.append(average_loss)
 
            if (epoch != 0 and epoch % self.loader.validation_rate == 0):
                mean_rank = self.validation()
                validation_losses.append(mean_rank)
                
                if mean_rank < self.best_loss:
                    self.best_loss = mean_rank
                    torch.save(self.model.state_dict(), self.best_model_path) 
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.patience:
                        logger.info(f"Early stopping after {epoch} epochs")
                        break 
                    
            training_range.set_description(
                f"Epoch: {epoch + 1} | Loss: {average_loss:.3f} | μ-rank:{mean_rank:.1f} | Best-μ-rank: {self.best_loss:.1f}")
 
