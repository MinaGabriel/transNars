
from negative_sampling import *
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.autograd import Variable

from utils import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from sklearn.utils import shuffle
torch.manual_seed(7)
random.seed(7)
torch.manual_seed(7)
np.random.seed(7)


class Train:
    def __init__(self, dataset: TripletsDataset, model_name: str, lr: float, embedding_dimension: int, epoch: int):
        self.dataset = dataset
        self.lr = lr
        self.epoch = epoch
        if model_name == 'TransE':
            self.model = TransE(dataset.config['num_entities'],
                                dataset.config['num_relations'], embedding_dimension)

    def start(self):
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            # FIXME: can be good for research  λ
            # FIXME: why is it zero
            weight_decay=0.0  # Weight decay (L2 regularization)
        )

        # print all embedding for testing 
        print(self.model.entity_embeddings.weight)
        for epoch in range(self.epoch):
            print(f'epoch:{epoch}')
            # FIXME: I am doing one batch
            for _, positive_batch in enumerate(self.dataset.get_dataloader(split='train', batch_size=3)):
                
                # get negative batch: random_sampling_r
                negative_batch = NegativeSampling.random_negative_sampling_r(
                    positive_batch, self.dataset)
                # convert torch from numpy for training
                negative_batch = torch.from_numpy(negative_batch)

                optimizer.zero_grad()

                positive_scores = self.model.forward(positive_batch)
                negative_scores = self.model.forward(negative_batch)

                training_loss = self.model.pairwise_hinge_loss(
                    positive_scores, negative_scores, gamma=1.0)

                print(f"Batch {_+1}: and training loss: {training_loss}")
                
                # This line computes the gradient of the loss with respect to all the parameters
                training_loss.backward()
                optimizer.step()
                
        # print all embedding for testing 
        print(self.model.entity_embeddings.weight)

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.num_entities = num_entities  # Number of entities in the dataset
        self.num_relations = num_relations  # Number of relations in the dataset
        self.embedding_dim = embedding_dim  # Dimension of the embedding space
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.distance_type = 'l2'  # Type of distance to use for score calculation

        # Initialize entity and relation embeddings
        self.entity_embeddings = nn.Embedding(
            num_entities, embedding_dim).to(self.device)
        self.relation_embeddings = nn.Embedding(
            num_relations, embedding_dim).to(self.device)
        # TODO: WHY INIT THE WEIGHTS?
        # FIXME: Can LLM generate better results if using Embeddings (BERT)?
        nn.init.xavier_uniform_(self.entity_embeddings.weight, gain=1)
        nn.init.xavier_uniform_(self.relation_embeddings.weight, gain=1)

    def forward(self, triplets):
        # Extract head, relation, and tail indices from triplets
        # head: size 2048 -> tensor([  700, 10234,  6447,  ..., 10827, 13234,  2825],
        # Assuming triplets is already a PyTorch tensor
        heads = triplets[:, 0].clone().detach().long().to(self.device)
        relations = triplets[:, 1].clone().detach().long().to(self.device)
        tails = triplets[:, 2].clone().detach().long().to(self.device)
        
        
    
        # Fetch embeddings for heads, relations, and tails
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        
        assert torch.all(torch.isfinite(head_embeddings)), "NaN or Inf in head embeddings"
        assert torch.all(torch.isfinite(relation_embeddings)), "NaN or Inf in relation embeddings"
        assert torch.all(torch.isfinite(tail_embeddings)), "NaN or Inf in tail embeddings"
        
        # Calculate the score using the embedding vectors
        scores = self._calculate_score(
            head_embeddings, relation_embeddings, tail_embeddings)
        return scores.view(-1, 1)

    def _calculate_score(self, head_embeddings, relation_embeddings, tail_embeddings):
        # Calculate the TransE score as negative distance
        score = (head_embeddings + relation_embeddings) - tail_embeddings
        # TODO: need to improve it is all missed up
        if self.distance_type == 'l1':
            score = torch.norm(score, p=1, dim=-1)
        else:  # Default to l2 distance
            score = torch.sqrt(torch.sum(score ** 2, dim=1))
        return -score

    def pairwise_hinge_loss(self, positive_scores, negative_scores, gamma):
        # Calculate pairwise hinge loss using MarginRankingLoss
        criterion = nn.MarginRankingLoss(margin=gamma)

        # assume + and - are of same size
        # ratio 1:1 thus all 1s
        target = torch.ones_like(positive_scores)

        # Calculate loss
        loss = criterion(positive_scores, negative_scores, target)

        return loss

    def regularization_loss(self, heads, relations, tails):
        # Calculate regularization loss as the mean of squared norms of embeddings
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        regularization = (torch.mean(head_embeddings ** 2) +
                          torch.mean(tail_embeddings ** 2) +
                          torch.mean(relation_embeddings ** 2)) / 3
        return regularization
