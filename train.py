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
from TransE import TransE
from eval import Eval

# Set random seeds for reproducibility
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Train:
    def __init__(self, dataset: TripletsDataset, model_name: str, lr: float, embedding_dimension: int, epoch: int):
        self.dataset = dataset
        self.lr = lr
        self.batch_size = 256
        self.gamma = 6.0
        self.lambda_reg = 0.01
        self.epoch = epoch
        self.best_loss = float('inf')
        self.best_model_path = f"{model_name}_{dataset.name}.pt"
        if model_name == 'TransE':
            self.model = TransE(dataset.num_entities,
                                dataset.num_relations, embedding_dimension)
        self.model.cuda()

    def start(self):
        self.model.train()
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # Use Adam optimizer

        training_range = tqdm(range(self.epoch))
        epoch_losses = []
        
        
        for epoch in training_range:
            total_loss = 0.0
            num_batches = 0
            for _, positive_batch in enumerate(DataLoader(self.dataset.train_dataset, batch_size=self.batch_size, shuffle=True)):
                negative_batch = NegativeSampling.random_negative_sampling_r(
                    positive_batch, self.dataset)
                negative_batch = torch.from_numpy(negative_batch).to(self.model.device)

                optimizer.zero_grad()

                positive_scores = self.model(positive_batch.to(self.model.device))
                negative_scores = self.model(negative_batch)

                training_loss = self.model.pairwise_hinge_loss(positive_scores, negative_scores, gamma=6.0)
                
                # Extract heads, relations, and tails from positive_batch for regularization loss
                heads = positive_batch[:, 0].to(self.model.device)
                relations = positive_batch[:, 1].to(self.model.device)
                tails = positive_batch[:, 2].to(self.model.device)
                reg_loss = self.model.regularization_loss(heads, relations, tails)

                total_batch_loss = training_loss + self.lambda_reg * reg_loss
                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            epoch_losses.append(average_loss)
            training_range.set_description("Epoch %d | Avg Loss: %f" % (epoch, average_loss))
            
            if average_loss < self.best_loss:
                self.best_loss = average_loss
                torch.save(self.model.state_dict(), self.best_model_path) 
            
        # Evaluate the model on the validation set 
        # eval = Eval(self.dataset, self.model, self.batch_size)
        # eval.validate()
        self.report_training(epoch_losses)

        
    def report_training(self, epoch_losses):
        logger.info("Training completed")
        logger.info(f"Total epochs: {self.epoch}")
        logger.info(f"Final average loss: {epoch_losses[-1]}")
        logger.info(f"Best model saved at: {self.best_model_path} with loss: {self.best_loss}")
 
