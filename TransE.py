
from tqdm import tqdm
from negative_sampling import *
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.autograd import Variable
from data.utils import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from sklearn.utils import shuffle
import logging


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, device):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device
        

        self.entity_embeddings = nn.Embedding(
            num_entities, embedding_dim).to(self.device)
        self.relation_embeddings = nn.Embedding(
            num_relations, embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        

    def forward(self, triplets, mode='head_batch'):
        heads = triplets[:, 0].to(self.device)
        relations = triplets[:, 1].to(self.device)
        tails = triplets[:, 2].to(self.device)

        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

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
        score = torch.norm(score, p=1, dim=-1)
        return score

    def pairwise_hinge_loss(self, positive_scores, negative_scores, gamma):
        negative_scores = negative_scores.view(-1, len(positive_scores)).permute(1,0)
        # criterion = nn.MarginRankingLoss(margin=gamma)
        # target = torch.ones_like(negative_scores)
        # loss = criterion(positive_scores, negative_scores, target)
        
        # Example will take neg [68024 X 1] and convert it to [25 X 2720]
        # permute to switch the columns and the rows
        # 

        loss = (torch.max(positive_scores - negative_scores, -
                torch.tensor(gamma))).mean() + torch.tensor(gamma)
        # print(f"Positive scores: {positive_scores.mean().item()}, Negative scores: {negative_scores.mean().item()}, Loss: {loss.item()}")
        return loss

    def regularization_loss(self, heads, relations, tails):
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        regularization = (torch.mean(head_embeddings ** 2) +
                          torch.mean(tail_embeddings ** 2) +
                          torch.mean(relation_embeddings ** 2)) / 3
        return regularization
