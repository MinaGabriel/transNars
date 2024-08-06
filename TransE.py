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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import ctypes
from ctypes import c_char_p, POINTER, c_int64, c_float
from datetime import datetime
import logging
from data.utils import check_folder_and_files
from data.Loader import *
import torch
from data.TrainDataLoader import *
from torch.utils.data import DataLoader
from data.TestDataLoader import *
from data.ValidDataLoader import *

# Set random seeds for reproducibility
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

config = None


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, device):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim).to(self.device)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails, mode='head_batch'):
        head_embeddings = self.entity_embeddings(heads.to(self.device))
        relation_embeddings = self.relation_embeddings(
            relations.to(self.device))
        tail_embeddings = self.entity_embeddings(tails.to(self.device))

        scores = self._calculate_score(
            head_embeddings, relation_embeddings, tail_embeddings, mode)
        return scores.view(-1, 1)

    def _calculate_score(self, h, r, t, mode):
        h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        # NOTE: using L1 norm p=1. 
        return torch.norm(score, p=1, dim=-1)

    def pairwise_hinge_loss(self, positive_scores, negative_scores, gamma):
        negative_scores = negative_scores.view(-1, len(positive_scores)).permute(1, 0)
        
        loss = (torch.max(positive_scores - negative_scores + gamma, torch.tensor(0.0))).mean()

        #loss = (torch.max(positive_scores - negative_scores, -torch.tensor(gamma))).mean() + torch.tensor(gamma)
        return loss

    def regularization_loss(self, heads, relations, tails):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        regularization = (torch.mean(head_embeddings ** 2) +
                          torch.mean(tail_embeddings ** 2) +
                          torch.mean(relation_embeddings ** 2)) / 3
        return regularization


class Train:
    def __init__(self, dataset: TrainDataLoader):
        global config
        config = dataset.loader.config
        self.dataset = dataset.data
        self.loader = dataset.loader
        self.device = dataset.loader.device
        self.early_stop_counter = 0
        self.saved_on_epoch = 0
        self.best_loss = float('inf')
        self.best_model_path = f"./models/trained/{config['model']}_{dataset.loader.name}.pt" 
        self.model = TransE(self.loader.num_entities, self.loader.num_relations,
                            config['dim'], self.device).to(self.device)
        self.valid_dataset = ValidDataLoader(self.loader).data

    def validation(self):
        self.model.eval()
        raw_ranks_h, raw_ranks_t = [], []
        num_entities = self.loader.num_entities
        with torch.no_grad():
            for batch_h, batch_t, batch_r in self.valid_dataset:
                batch_h, batch_t, batch_r = batch_h.to(self.device), batch_t.to(
                    self.device), batch_r.to(self.device)

                heads = batch_h.repeat_interleave(num_entities)
                relations = batch_r.repeat_interleave(num_entities)
                tails = batch_t.repeat_interleave(num_entities)

                entities = torch.arange(
                    num_entities, device=self.device).repeat(len(batch_h))

                scores_h = self.model(entities, relations, tails).view(
                    len(batch_h), num_entities)
                ranks_h = (torch.argsort(scores_h, dim=1) ==
                           batch_h.view(-1, 1)).nonzero(as_tuple=False)[:, 1] + 1
                raw_ranks_h.extend(ranks_h.cpu().numpy())

                scores_t = self.model(heads, relations, entities).view(
                    len(batch_t), num_entities)
                ranks_t = (torch.argsort(scores_t, dim=1) ==
                           batch_t.view(-1, 1)).nonzero(as_tuple=False)[:, 1] + 1
                raw_ranks_t.extend(ranks_t.cpu().numpy())

        mean_rank_h = np.mean(raw_ranks_h)
        mean_rank_t = np.mean(raw_ranks_t)
        return (mean_rank_h + mean_rank_t) / 2

    def start(self):
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        training_range = tqdm(range(config['epoch']))
        epoch_losses = []
        validation_losses = []
        mean_rank = float('inf')

        for epoch in training_range:
            total_loss = 0.0
            num_batches = 0

            for batch in self.dataset:
                optimizer.zero_grad()
                (pos_h, pos_t, pos_r, pos_y), (neg_h, neg_t, neg_r, neg_y) = batch

                pos_h, pos_t, pos_r = pos_h.to(self.device), pos_t.to(
                    self.device), pos_r.to(self.device)
                neg_h, neg_t, neg_r = neg_h.to(self.device), neg_t.to(
                    self.device), neg_r.to(self.device)

                positive_scores = self.model(pos_h, pos_r, pos_t)
                negative_scores = self.model(neg_h, neg_r, neg_t)

                training_loss = self.model.pairwise_hinge_loss(
                    positive_scores, negative_scores, config['margin'])

                heads = torch.vstack(
                    (pos_h.view(-1, 1), neg_h.view(-1, 1))).to(self.device)
                relations = torch.vstack(
                    (pos_r.view(-1, 1), neg_r.view(-1, 1))).to(self.device)
                tails = torch.vstack(
                    (pos_t.view(-1, 1), neg_t.view(-1, 1))).to(self.device)
                reg_loss = self.model.regularization_loss(
                    heads, relations, tails)

                total_batch_loss = training_loss + 2.0 * reg_loss
                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            
            epoch_losses.append(average_loss)

            if epoch != 0 and epoch % config['validation_rate'] - 1 == 0:
                mean_rank = self.validation()
                validation_losses.append(mean_rank)

                if mean_rank < self.best_loss:
                    self.best_loss = mean_rank
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.saved_on_epoch = epoch
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= config['patience']:
                        logger.info(f"Early stopping after {epoch} epochs")
                        break
            
            training_range.set_description(
                f"Epoch: {epoch + 1} | Loss: {average_loss:.3f} | μ-rank: {self.best_loss:.1f} | Stop: {self.early_stop_counter}/{config['patience']}")
        print(epoch_losses)
        print(validation_losses)
        
        plot_loss_and_mean_rank(epoch_losses, validation_losses, 
                                config['validation_rate'] ,self.saved_on_epoch, f'{config['model']}_{self.loader.name}')

class Test:
    def __init__(self, dataset: TestDataLoader):
        global config
        config = dataset.loader.config
        self.dataset = dataset.data
        self.loader = dataset.loader
        self.device = dataset.loader.device
        self.model = TransE(self.loader.num_entities,
                            self.loader.num_relations, config['dim'], self.device)
        self.model.load_state_dict(torch.load(
            f'./models/trained/{config["model"]}_{self.loader.name}.pt', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.lib = ctypes.CDLL("./Base.so")
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64]

    def test_head(self, e, r, t, mode):
        scores = self.model(e, r, t, mode)
        return scores

    def test_tail(self, h, r, e, mode):
        scores = self.model(h, r, e, mode)
        return scores

    def run_link_prediction(self):
        num_entities = self.loader.num_entities
        for _, (batch_h, batch_t, batch_r) in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            batch_h, batch_r, batch_t = batch_h.to(self.device), batch_r.to(
                self.device), batch_t.to(self.device)

            heads = batch_h.repeat_interleave(num_entities)
            relations = batch_r.repeat_interleave(num_entities)
            tails = batch_t.repeat_interleave(num_entities)

            entities = torch.arange(
                num_entities, device=self.device).repeat(len(batch_h))

            scores_h = self.test_head(entities, relations, tails, "head_batch")
            score = scores_h.squeeze().detach().cpu().numpy()
            self.lib.testHead(score.__array_interface__["data"][0], _)

            scores_t = self.test_tail(heads, relations, entities, "tail_batch")
            score = scores_t.squeeze().detach().cpu().numpy()
            self.lib.testTail(score.__array_interface__["data"][0], _)

        self.lib.test_link_prediction()


if __name__ == "__main__":
    model_name = "FB15K237"
    loader = Loader(f"TransE_{model_name}.yaml")
    # read train data.
    # train_dataloader = TrainDataLoader(loader)

    # # #Initialize and start training
    #trainer = Train(train_dataloader)
    #trainer.start()

    test_dataloader = TestDataLoader(loader)

    tester = Test(test_dataloader)
    tester.run_link_prediction()
