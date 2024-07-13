import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from TransE import TransE
from data.Loader import TripletsDataset
import numpy as np
from prettytable import PrettyTable
import logging
import time
from torch.autograd import Variable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tester(object):
    def __init__(self, name, embedding_dimension, device):
        self.name = name
        self.device = device
        self.embedding_dimension = embedding_dimension
        self.dataset = TripletsDataset(f'datasets/{name}')
        self.model = TransE(self.dataset.num_entities, self.dataset.num_relations, embedding_dimension, self.device)
        self.model.load_state_dict(torch.load(f'models/TransE_{name}.pt', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.size = len(self.dataset.test_dataset.triplets)
        self.table = PrettyTable()
        
    def filter_scores(self, triplets, scores):
        # Checking if each row in x is in t
        is_in_t = [(row == triplets).all(dim=1).any().item() for row in self.all_triplets_tensor]
    def test_head(self, e, r, t):
        triplets_head = torch.stack((e, r, t), dim=1).to(self.device)
        scores = self.model(triplets_head).squeeze() 
        return triplets_head, scores

    def test_tail(self, h, r, e):
        triplets_tail = torch.stack((h, r, e), dim=1).to(self.device)
        scores = self.model(triplets_tail).squeeze()
        return triplets_tail, scores

    def run_link_prediction(self):
        self.table.field_names = ["Metric", "MRR", "MR", "hit@10", "hit@3", "hit@1"]
        raw_mrr_h, raw_mrr_t = [], []
        raw_ranks_h, raw_ranks_t = [], []
        hits_at_k = {1: [0, 0], 3: [0, 0], 5: [0, 0], 10: [0, 0]}

        data_loader = DataLoader(
            self.dataset.test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=20,
            pin_memory=True,
            persistent_workers=True
        )

        num_entities = self.dataset.num_entities

        # Start timing
        start_time = time.time()

        # Iterate through the DataLoader and move data to GPU
        for _, data in tqdm(enumerate(data_loader), total=self.size // data_loader.batch_size):
            data = data.to(self.device, non_blocking=True)  # Move data to GPU

            h = data[:, 0].clone().detach()
            r = data[:, 1].clone().detach()
            t = data[:, 2].clone().detach()

            heads = h.repeat_interleave(num_entities)
            relations = r.repeat_interleave(num_entities)
            tails = t.repeat_interleave(num_entities)

            entities = torch.arange(num_entities, device=self.device)
            entities = entities.repeat(len(h))

            # Head prediction
            triplets_h, scores_h = self.test_head(entities, relations, tails) 
            scores_h = scores_h * -1
            sorted_scores_h, sorted_indices_h = torch.sort(scores_h.view(len(h), num_entities), descending=True)
            
            
            # Vectorized ranking
            h_expanded = h.unsqueeze(1).expand_as(sorted_indices_h)
            rank_h = (sorted_indices_h == h_expanded).nonzero(as_tuple=True)[1] + 1  # Get the ranks
            raw_mrr_h.extend((1.0 / rank_h).cpu().numpy())
            raw_ranks_h.extend(rank_h.cpu().numpy())

            # Vectorized hits calculation
            for k in hits_at_k.keys():
                hits_at_k[k][0] += (sorted_indices_h[:, :k] == h_expanded[:, :k]).sum().item()

            # Tail prediction
            triplets_tail, scores_t = self.test_tail(heads, relations, entities) 
            scores_t = scores_t * -1
            sorted_scores_t, sorted_indices_t = torch.sort(scores_t.view(len(t), num_entities), descending=True)

            # Vectorized ranking
            t_expanded = t.unsqueeze(1).expand_as(sorted_indices_t)
            rank_t = (sorted_indices_t == t_expanded).nonzero(as_tuple=True)[1] + 1  # Get the ranks
            raw_mrr_t.extend((1.0 / rank_t).cpu().numpy())
            raw_ranks_t.extend(rank_t.cpu().numpy())

            # Vectorized hits calculation
            for k in hits_at_k.keys():
                hits_at_k[k][1] += (sorted_indices_t[:, :k] == t_expanded[:, :k]).sum().item()
                
                
                
                
            # filtered ranks calculation
            
            

        # Stop timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        self.table.add_row(["Head (Raw)", f'{np.mean(raw_mrr_h):.6f}', f'{np.mean(raw_ranks_h):.6f}',
                            f'{hits_at_k[10][0]/self.size:.6f}',
                            f'{hits_at_k[3][0]/self.size:.6f}',
                            f'{hits_at_k[1][0]/self.size:.6f}'])

        self.table.add_row(["Tail (Raw)", f'{np.mean(raw_mrr_t):.6f}', f'{np.mean(raw_ranks_t):.6f}',
                            f'{hits_at_k[10][1]/self.size:.6f}',
                            f'{hits_at_k[3][1]/self.size:.6f}',
                            f'{hits_at_k[1][1]/self.size:.6f}'])

        raw_mrr = (np.mean(raw_mrr_h) + np.mean(raw_mrr_t)) / 2
        raw_mr = (np.mean(raw_ranks_h) + np.mean(raw_ranks_t)) / 2

        hits_1 = (hits_at_k[1][0] + hits_at_k[1][1]) / (2 * self.size)
        hits_3 = (hits_at_k[3][0] + hits_at_k[3][1]) / (2 * self.size)
        hits_10 = (hits_at_k[10][0] + hits_at_k[10][1]) / (2 * self.size)

        self.table.add_row(["Average (Raw)     ", f'{raw_mrr:.6f}', f'{raw_mr:.6f}', f'{
                           hits_10:.6f}', f'{hits_3:.6f}', f'{hits_1:.6f}'])


        print(self.table)
        logger.info(f"Total run time: {int(minutes)} minutes and {int(seconds)} seconds")

if __name__ == "__main__":
    Tester('FB15K237', 200, device='cuda:0').run_link_prediction()
