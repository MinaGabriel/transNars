from TransE import TransE
import torch
from prettytable import PrettyTable
from data.Loader import TripletsDataset

name = 'nations'
dataset = TripletsDataset(f'datasets/{name}')
model = TransE(dataset.num_entities, dataset.num_relations, 50, 'cuda:0')

model.load_state_dict(torch.load(
    f'models/TransE_{name}.pt', map_location=model.device))
model.eval()

head = 'usa'
relation = 'ngo'

head_tensor = torch.tensor([dataset.entity_map[head]] * dataset.num_entities)
relation_tensor = torch.tensor(
    [dataset.relation_map[relation]] * dataset.num_entities)
tail_tensor = torch.tensor(list(dataset.entity_map.values()))

triplets = torch.stack([head_tensor, relation_tensor, tail_tensor], dim=1)

scores = model(triplets).detach().cpu().numpy()

tail_scores = {tail: score.item() for tail, score in zip(dataset.entity_map.keys(), scores) if tail != head}

best_tail = min(tail_scores, key=tail_scores.get)
best_score = tail_scores[best_tail]

print(f'Best match: {best_tail}, with score: {best_score:.4f}')

# Sort and display the top 10 results
sorted_tail_scores = sorted(tail_scores.items(), key=lambda item: item[1])

# Create a PrettyTable for displaying the results
table = PrettyTable()
table.field_names = ["Rank", "Tail Entity", "Score"]

for rank, (tail, score) in enumerate(sorted_tail_scores[:10], start=1):
    table.add_row([rank, tail, f"{score:.4f}"])

print("Top 10 Predictions:")
print(table)
