import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
import argparse
import json
import argparse
import yaml
from tqdm import tqdm
from torch.nn.init import xavier_normal_, xavier_uniform_
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import os
import random
from sklearn.utils import shuffle
import logging
from data.Loader import Loader
from data.TrainDataLoader import TrainDataLoader
from data.TestDataLoader import TestDataLoader
from data.ValidDataLoader import ValidDataLoader
from data.utils import *

# Set random seeds for reproducibility
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

config = None


class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = nn.Embedding(num_entities, 200, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, 200, padding_idx=0)
        self.loss = nn.BCELoss()
        self.emb_dim1 = 20
        self.emb_dim2 = 10

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(0.2),
            nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(9728, 200), nn.Dropout(0.3), nn.BatchNorm1d(200), nn.ReLU()
        )

        self.register_parameter("b", Parameter(torch.zeros(num_entities)))

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        x = self.conv_block(stacked_inputs)
        x = x.view(x.shape[0], -1)
        x = self.fc_block(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred


class Train:
    def __init__(self, loader):
        global config
        self.loader = loader
        config = self.loader.config
        train_dataset = CustomTrainDataset(self.loader)
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        self.best_model_path = (
            f"./models/trained/{config['model']}_{self.loader.name}.pt"
        )

        self.valid_dataset = ValidDataLoader(self.loader).data
        # Initialize and move model to device
        self.model = ConvE(self.loader.num_entities, self.loader.num_relations)
        self.model.init()
        self.model.to(self.loader.device)

        self.early_stop_counter = 0
        self.saved_on_epoch = 0

    def start(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # Initialize variables to track the best model
        best_mrr = 0
        best_epoch = 0
        best_model_state = None

        epoch_losses = []
        validation_losses = []

        training_range = tqdm(range(config["epoch"]))

        # Training loop
        for epoch in training_range:
            self.model.train()
            total_loss = 0
            num_batches = 0
            for batch in self.train_dataloader:
                e1_tensor, rel_tensor, e2_multi_tensor = batch
                optimizer.zero_grad()
                pred = self.model.forward(e1_tensor, rel_tensor)
                # Compute loss
                loss = self.model.loss(pred, e2_multi_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            average_loss = (total_loss / num_batches) * 100

            epoch_losses.append(average_loss)

            self.model.eval()
            with torch.no_grad():
                if epoch > 0 and epoch % config["validation_rate"] == 0:
                    raw_ranks_t = []

                    for batch_h, batch_t, batch_r in self.valid_dataset:
                        batch_h, batch_t, batch_r = (
                            batch_h.to(self.loader.device),
                            batch_t.to(self.loader.device),
                            batch_r.to(self.loader.device),
                        )

                        # Tail prediction (normal)
                        scores_t = self.model.forward(batch_h, batch_r)
                        targets_t = batch_t.view(-1, 1)  # Reshape for broadcasting

                        # Calculate the ranks
                        _, indices = torch.sort(scores_t, descending=True, dim=1)
                        ranks = (indices == targets_t).nonzero(as_tuple=False)[
                            :, 1
                        ] + 1  # Extract the rank
                        raw_ranks_t.extend(ranks.tolist())

                    ranks_t = torch.FloatTensor(raw_ranks_t)
                    mr = ranks_t.mean().item()
                    mrr = (1 / ranks_t).mean().item()
                    hits_at_1 = (ranks_t <= 1).float().mean().item()
                    hits_at_3 = (ranks_t <= 3).float().mean().item()
                    hits_at_10 = (ranks_t <= 10).float().mean().item()

                    metrics = {
                        "MR": mr,
                        "MRR": mrr,
                        "Hit@10": hits_at_10,
                        "Hit@3": hits_at_3,
                        "Hit@1": hits_at_1,
                    }

                    df_metrics = pd.DataFrame([metrics])
                    print(df_metrics.to_string(index=False))

                    # Save the model if it achieves a new best MRR
                    if mrr > best_mrr:
                        best_mrr = mrr
                        best_epoch = epoch
                        best_model_state = self.model.state_dict()
                        torch.save(best_model_state, self.best_model_path)
                        print(
                            f"New best model saved with MRR: {best_mrr} at epoch {best_epoch}"
                        )
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                    if self.early_stop_counter >= config["patience"]:
                        logger.info(f"Early stopping after {epoch} epochs")
                        break
            training_range.set_description(
                f"Epoch: {epoch + 1} | Loss: {average_loss:.3f} | Stop: {self.early_stop_counter}/{config['patience']}"
            )

        print(f"Best model achieved at epoch {best_epoch} with MRR: {best_mrr}")


class CustomTrainDataset(Dataset):
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.y_multihot_template = torch.zeros(
            self.loader.num_entities, dtype=torch.float
        ).to(self.device)
        self.e1_list, self.rel_list, self.e2_multi_list = self._prepare_data()

    def _prepare_data(self):
        print("Preparing")
        train_dataloader = TrainDataLoader(self.loader)
        batch_h, batch_r, batch_t, start_idx, end_idx = (
            train_dataloader.get_sorted_train_by_head()
        )

        e1_list, rel_list, e2_multi_list = [], [], []
        for i in tqdm(range(self.loader.num_entities), desc="Loading data"):
            start = start_idx[i]
            end = end_idx[i]
            if start != -1 and end != -1:
                tail = []
                for j in range(start, end + 1):
                    tail.append(batch_t[j])
                    if j + 1 < len(batch_r) and (batch_r[j + 1] != batch_r[j]):
                        e1_list.append(batch_h[j])
                        rel_list.append(batch_r[j])
                        e2_multi_list.append(tail)
                        tail = []
                if tail:
                    e1_list.append(batch_h[start])
                    rel_list.append(batch_r[start])
                    e2_multi_list.append(tail)
        return e1_list, rel_list, e2_multi_list

    def __len__(self):
        return len(self.e1_list)

    def __getitem__(self, idx):
        e1 = self.e1_list[idx]
        rel = self.rel_list[idx]
        e2_multi = self.e2_multi_list[idx]

        e1_tensor = torch.tensor(e1, dtype=torch.long, device=self.device)
        rel_tensor = torch.tensor(rel, dtype=torch.long, device=self.device)
        e2_multi_tensor = torch.tensor(e2_multi, dtype=torch.long, device=self.device)

        y_multihot = self.y_multihot_template.clone()
        y_multihot.scatter_(0, e2_multi_tensor, 1)

        y_smooth = ((1 - 0.1) * y_multihot) + (0.1 / self.loader.num_entities)

        return e1_tensor, rel_tensor, y_smooth


def main():
    # Load your data
    Train(Loader("ConvE_FB15K237.yaml")).start()


if __name__ == "__main__":
    main()
