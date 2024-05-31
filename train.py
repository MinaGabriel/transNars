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


# Set random seeds for reproducibility
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Train:
    def __init__(self, dataset: TripletsDataset, model_name: str, lr: float, embedding_dimension: int, epoch: int):
        self.dataset = dataset
        self.lr = lr
        self.epoch = epoch
        if model_name == 'TransE':
            self.model = TransE(dataset.num_entities,
                                dataset.num_relations, embedding_dimension)

    def start(self):
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=0)

        training_range = tqdm(range(self.epoch))
        for epoch in training_range:
            total_loss = 0.0
            num_batches = 0
            for _, positive_batch in enumerate(DataLoader(self.dataset.train_dataset, batch_size=256, shuffle=True)):
                negative_batch = NegativeSampling.random_negative_sampling_r(
                    positive_batch, self.dataset)
                negative_batch = torch.from_numpy(
                    negative_batch).to(self.model.device)

                optimizer.zero_grad()

                positive_scores = self.model(
                    positive_batch.to(self.model.device))
                negative_scores = self.model(negative_batch)

                training_loss = self.model.pairwise_hinge_loss(
                    positive_scores, negative_scores, gamma=5.0)
                training_loss.backward()
                optimizer.step()

                total_loss += training_loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            training_range.set_description(
                "Epoch %d | Avg Loss: %f" % (epoch, average_loss * 100))

        self.validate()
    def validate(self):
        self.model.eval()
        
        filter_mrr_h, filter_mrr_t = [], []
        raw_mrr_h, raw_mrr_t = [], []

        hits_at_k = {1: [0, 0], 3: [0, 0], 5: [0, 0], 10: [0, 0]}

        valid_loader = DataLoader(self.dataset.valid_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for valid_triple in tqdm(valid_loader):
                valid_triple = valid_triple.squeeze().to(self.model.device)
                h, r, t = valid_triple[0], valid_triple[1], valid_triple[2]

                heads = h.repeat(self.dataset.num_entities)
                relations = r.repeat(self.dataset.num_entities)
                tails = t.repeat(self.dataset.num_entities)
                entities = torch.arange(self.dataset.num_entities, device=self.model.device)

                # Predict tails
                triplets_tail = torch.stack((heads, relations, entities), dim=1)
                tails_predictions = self.model(triplets_tail).squeeze()

                # Predict heads
                triplets_head = torch.stack((entities, relations, tails), dim=1)
                heads_predictions = self.model(triplets_head).squeeze()

                # Calculate raw ranks
                raw_rank_h = (torch.argsort(heads_predictions, descending=True) == h).nonzero(as_tuple=True)[0].item() + 1
                raw_rank_t = (torch.argsort(tails_predictions, descending=True) == t).nonzero(as_tuple=True)[0].item() + 1

                raw_mrr_h.append(1.0 / raw_rank_h)
                raw_mrr_t.append(1.0 / raw_rank_t)

                # Calculate filtered ranks
                valid_heads = self.dataset.train_triples[:, 0]
                valid_tails = self.dataset.train_triples[:, 2]

                # Filter valid heads
                for valid_head in valid_heads:
                    if valid_head != h:
                        heads_predictions[valid_head] = -float('inf')

                # Filter valid tails
                for valid_tail in valid_tails:
                    if valid_tail != t:
                        tails_predictions[valid_tail] = -float('inf')

                filter_rank_h = (torch.argsort(heads_predictions, descending=True) == h).nonzero(as_tuple=True)[0].item() + 1
                filter_rank_t = (torch.argsort(tails_predictions, descending=True) == t).nonzero(as_tuple=True)[0].item() + 1

                filter_mrr_h.append(1.0 / filter_rank_h)
                filter_mrr_t.append(1.0 / filter_rank_t)

                # Calculate Hits@K
                indices_head = torch.argsort(heads_predictions, descending=True)
                indices_tail = torch.argsort(tails_predictions, descending=True)

                for k in hits_at_k.keys():
                    hits_at_k[k][0] += (indices_head[:k] == h).sum().item()
                    hits_at_k[k][1] += (indices_tail[:k] == t).sum().item()

        filter_mrr = (np.mean(filter_mrr_h) + np.mean(filter_mrr_t)) / 2
        raw_mrr = (np.mean(raw_mrr_h) + np.mean(raw_mrr_t)) / 2

        for k in hits_at_k.keys():
            hits_at_k[k] = (hits_at_k[k][0] + hits_at_k[k][1]) / (2 * len(valid_loader))

        print(f'Filtered MRR: {filter_mrr:.6f}')
        print(f'Filtered Hits@1: {hits_at_k[1]:.6f}')
        print(f'Filtered Hits@3: {hits_at_k[3]:.6f}')
        print(f'Filtered Hits@5: {hits_at_k[5]:.6f}')
        print(f'Filtered Hits@10: {hits_at_k[10]:.6f}')

        print(f'Raw MRR: {raw_mrr:.6f}')
        print(f'Raw Hits@1: {hits_at_k[1]:.6f}')
        print(f'Raw Hits@3: {hits_at_k[3]:.6f}')
        print(f'Raw Hits@5: {hits_at_k[5]:.6f}')
        print(f'Raw Hits@10: {hits_at_k[10]:.6f}')



class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.distance_type = 'l2'

        self.entity_embeddings = nn.Embedding(
            num_entities, embedding_dim).to(self.device)
        self.relation_embeddings = nn.Embedding(
            num_relations, embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, triplets):
        heads = triplets[:, 0].to(self.device)
        relations = triplets[:, 1].to(self.device)
        tails = triplets[:, 2].to(self.device)

        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)

        scores = self._calculate_score(
            head_embeddings, relation_embeddings, tail_embeddings)
        return scores.view(-1, 1)

    def _calculate_score(self, h, r, t):
        h = F.normalize(h, 2, -1)
        r = F.normalize(r, 2, -1)
        t = F.normalize(t, 2, -1)
        score = (h + r) - t
        score = torch.norm(score, p=1, dim=-1)
        return score

    def pairwise_hinge_loss(self, positive_scores, negative_scores, gamma):
        # criterion = nn.MarginRankingLoss(margin=gamma)
        # target = torch.ones_like(positive_scores)
        # loss = criterion(positive_scores, negative_scores, target)
        loss = (torch.max(positive_scores - negative_scores, - torch.tensor(gamma))).mean() + torch.tensor(gamma)
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
