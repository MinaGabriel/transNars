import torch
from tqdm import tqdm
from TransE import TransE 
from data.TripletsDataset import TripletsDataset
import numpy as np
from prettytable import PrettyTable



class Tester(object):
    def __init__(self, name, embedding_dimension, device):
        self.name = name
        self.device = device
        self.embedding_dimension = embedding_dimension
        self.dataset = TripletsDataset(f'datasets/{name}')  
        self.model = TransE(self.dataset.num_entities, self.dataset.num_relations, embedding_dimension, 'cuda:0')
        self.model.load_state_dict(torch.load(f'models/TransE_{name}.pt', map_location=self.model.device))
        self.model.eval() 
        self.size = len(self.dataset.test_dataset.triplets)
        self.table = PrettyTable()
        
    def test_head(self, e, r, t):
        triplets_head = torch.stack((e, r, t), dim=1)
        return self.model(triplets_head).squeeze()
    
    def test_tail(self, h, r, e):
        triplets_tail = torch.stack((h, r, e), dim=1)
        return self.model(triplets_tail).squeeze()
    
    def run_link_prediction(self):
        self.table.field_names = ["Metric", "MRR", "MR", "hit@10", "hit@3", "hit@1"]
        raw_mrr_h, raw_mrr_t = [], []
        raw_ranks_h, raw_ranks_t = [], []
        hits_at_k = {1: [0, 0], 3: [0, 0], 5: [0, 0], 10: [0, 0]}

        for _, data in tqdm(enumerate(self.dataset.test_dataset.triplets), total=self.size):
            h = torch.tensor(data[0], device=self.model.device)
            r = torch.tensor(data[1], device=self.model.device)
            t = torch.tensor(data[2], device=self.model.device)
            heads = h.repeat(self.dataset.num_entities)
            relations = r.repeat(self.dataset.num_entities)
            tails = t.repeat(self.dataset.num_entities)
            entities = torch.arange(self.dataset.num_entities, device=self.model.device)
            
            # Head prediction
            scores_h = self.test_head(entities, relations, tails)
            sorted_scores_h, sorted_indices_h = torch.sort(scores_h, descending=True)
            rank_h = (sorted_indices_h == h).nonzero(as_tuple=True)[0].item() + 1
            raw_mrr_h.append(1.0 / rank_h)
            raw_ranks_h.append(rank_h)
            
            # Tail prediction
            scores_t = self.test_tail(heads, relations, entities)
            sorted_scores_t, sorted_indices_t = torch.sort(scores_t, descending=True)
            rank_t = (sorted_indices_t == t).nonzero(as_tuple=True)[0].item() + 1
            raw_mrr_t.append(1.0 / rank_t)
            raw_ranks_t.append(rank_t)
            
            for k in hits_at_k.keys():
                hits_at_k[k][0] += (sorted_indices_h[:k] == h).sum().item()
                hits_at_k[k][1] += (sorted_indices_t[:k] == t).sum().item()
        
        self.table.add_row(["Head (Raw)", np.mean(raw_mrr_h), np.mean(raw_ranks_h), 
                            hits_at_k[10][0]/self.size, 
                            hits_at_k[3][0]/self.size, 
                            hits_at_k[1][0]/self.size])
        
        self.table.add_row(["Tail (Raw)", np.mean(raw_mrr_t), np.mean(raw_ranks_t), 
                            hits_at_k[10][1]/self.size, 
                            hits_at_k[3][1]/self.size, 
                            hits_at_k[1][1]/self.size])
        
        raw_mrr = (np.mean(raw_mrr_h) + np.mean(raw_mrr_t)) / 2
        raw_mr = (np.mean(raw_ranks_h) + np.mean(raw_ranks_t)) / 2
        
        hits_1 = (hits_at_k[1][0] + hits_at_k[1][1]) / (2 * self.size)
        hits_3 = (hits_at_k[3][0] + hits_at_k[3][1]) / (2 * self.size)
        hits_10 = (hits_at_k[10][0] + hits_at_k[10][1]) / (2 * self.size)
        
        self.table.add_row(["Average (Raw)", raw_mrr, raw_mr, hits_10, hits_3, hits_1])
        
        print(self.table)

if __name__ == "__main__":
    Tester('FB15K237', 200, device='cuda:0').run_link_prediction()