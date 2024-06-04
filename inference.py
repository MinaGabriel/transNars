from TransE import TransE
import argparse
from datetime import datetime
from data.utils import *
from train import Train
from data.TripletsDataset import *


dataset = TripletsDataset('datasets/WN11')
model = TransE(dataset.num_entities, dataset.num_relations, 200)

model.load_state_dict(torch.load(
    'TransE_WN11.pt', map_location=model.device))
model.eval()


head = '__american_arborvitae_1'
relation = '_member_holonym'

head_tensor = torch.tensor([dataset.entity_map[head]] * dataset.num_entities)
relation_tensor = torch.tensor(
    [dataset.relation_map[relation]] * dataset.num_entities)
tail_tensor = torch.tensor(list(dataset.entity_map.values()))

triplets = torch.stack([head_tensor, relation_tensor, tail_tensor], dim=1)

scores = model(triplets).detach().cpu().numpy()

tail_scores = {tail: score for tail, score in zip(dataset.entity_map.keys(), scores) if tail != head}

best_tail = min(tail_scores, key=tail_scores.get)
best_score = tail_scores[best_tail]

print(f'best match {best_tail}, with score {best_score}')

sorted_tail_scores = sorted(tail_scores.items(), key=lambda item: item[1])

print(sorted_tail_scores[1:10])