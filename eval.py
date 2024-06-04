from data.TripletsDataset import TripletsDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class Eval:
    def __init__(self, dataset: TripletsDataset, model, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model = model
        self.model.eval()
        self.seen_triplets = dataset.every_relationships  # Dictionary of seen triplets

    def filter_predictions(self, predictions, h, r, t):
        if (h.item(), r.item()) in self.seen_triplets:
            valid_tails = self.seen_triplets[(h.item(), r.item())]
            for valid_tail in valid_tails:
                if valid_tail != t.item():
                    predictions[valid_tail] = -float('inf')
        return predictions

    def validate(self):
        filter_mrr_h, filter_mrr_t = [], []
        raw_mrr_h, raw_mrr_t = [], []

        hits_at_k = {1: [0, 0], 3: [0, 0], 5: [0, 0], 10: [0, 0]}

        valid_loader = DataLoader(self.dataset.valid_dataset, self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(self.model.device)
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]

                all_heads_predictions = []
                all_tails_predictions = []

                for h, r, t in zip(h_batch, r_batch, t_batch):
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

                    # Filter valid heads and tails
                    heads_predictions = self.filter_predictions(heads_predictions, h, r, t)
                    tails_predictions = self.filter_predictions(tails_predictions, h, r, t)

                    all_heads_predictions.append(heads_predictions)
                    all_tails_predictions.append(tails_predictions)

                all_heads_predictions = torch.stack(all_heads_predictions)
                all_tails_predictions = torch.stack(all_tails_predictions)

                for idx, (h, r, t) in enumerate(zip(h_batch, r_batch, t_batch)):
                    heads_predictions = all_heads_predictions[idx]
                    tails_predictions = all_tails_predictions[idx]

                    # Calculate raw ranks
                    raw_rank_h = (torch.argsort(heads_predictions, descending=True) == h).nonzero(as_tuple=True)[0].item() + 1
                    raw_rank_t = (torch.argsort(tails_predictions, descending=True) == t).nonzero(as_tuple=True)[0].item() + 1

                    raw_mrr_h.append(1.0 / raw_rank_h)
                    raw_mrr_t.append(1.0 / raw_rank_t)

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
            hits_at_k[k] = (hits_at_k[k][0] + hits_at_k[k][1]) / (2 * len(valid_loader.dataset))

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

        print("Validating...")
 