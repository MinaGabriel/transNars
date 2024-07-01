import argparse
from datetime import datetime
from tqdm import tqdm
from negative_sampling import NegativeSampling
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.autograd import Variable
from data.utils import check_folder_and_files
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
import matplotlib.pyplot as plt
from data.TripletsDataset import TripletsDataset
from data.NegativeDataset import NegativeDataset
from torch.utils.data import DataLoader

# Set random seeds for reproducibility
torch.manual_seed(7)
random.seed(7)
np.random.seed(7)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class Train:
    def __init__(self, dataset: TripletsDataset, negative_dataset: NegativeDataset, model_name: str, lr: float,
                 embedding_dimension: int, epoch: int, batch_size=256, device='cpu', neg_ratio=25):
        self.dataset = dataset
        self.negative_dataset = negative_dataset
        self.lr = lr
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.gamma = 6.0
        self.lambda_reg = 0.1
        self.weight_decay = 0
        self.neg_ratio = neg_ratio
        self.epoch = epoch
        self.best_loss = float('inf')
        self.best_model_path = f"models/{model_name}_{dataset.name}.pt"
        if model_name == 'TransE':
            self.model = TransE(dataset.num_entities,
                                dataset.num_relations, embedding_dimension, device)
        self.model.to(self.device)

    def start(self):
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        training_range = tqdm(range(self.epoch))
        epoch_losses = []
        validation_losses = []

        for epoch in training_range:
            total_loss = 0.0
            num_batches = 0

            for _, positive_batch in enumerate(DataLoader(self.dataset.train_dataset, 
                                                          batch_size=self.batch_size, shuffle=True, num_workers=4)):
                start_neg = num_batches * self.batch_size * self.neg_ratio
                end_neg = (num_batches + 1) * self.batch_size * self.neg_ratio
                negative_batch = self.negative_dataset[start_neg: end_neg]

                optimizer.zero_grad()

                positive_scores = self.model(positive_batch.to(self.device))
                negative_scores = self.model(negative_batch.to(self.device))

                training_loss = self.model.pairwise_hinge_loss(
                    positive_scores, negative_scores, self.gamma)

                stacked_batch = torch.vstack((positive_batch, negative_batch)).to(
                    self.device)
                
                heads = stacked_batch[:, 0].to(self.device)
                relations = stacked_batch[:, 1].to(self.device)
                tails = stacked_batch[:, 2].to(self.device)
                reg_loss = self.model.regularization_loss(
                    heads, relations, tails)

                total_batch_loss = training_loss + self.lambda_reg * reg_loss
                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            epoch_losses.append(average_loss)

            # Validate the model
            validation_loss = self.validate()
            validation_losses.append(validation_loss)

            training_range.set_description(
                f"Epoch {epoch + 1} | Avg Training Loss: {average_loss:.4f} | Validation Loss: {validation_loss:.4f}")

            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"New best model saved with validation loss {
                            validation_loss:.4f}")

        self.report_training(epoch_losses, validation_losses)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for _, positive_batch in enumerate(DataLoader(self.dataset.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)):
                negative_batch = NegativeSampling.random_negative_sampling_r(
                    positive_batch, self.dataset, self.neg_ratio)
                negative_batch = torch.from_numpy(
                    negative_batch).to(self.device)

                positive_scores = self.model(positive_batch.to(self.device))
                negative_scores = self.model(negative_batch)

                validation_loss = self.model.pairwise_hinge_loss(
                    positive_scores, negative_scores, self.gamma)

                heads = positive_batch[:, 0].to(self.device)
                relations = positive_batch[:, 1].to(self.device)
                tails = positive_batch[:, 2].to(self.device)
                reg_loss = self.model.regularization_loss(
                    heads, relations, tails)

                total_batch_loss = validation_loss + self.lambda_reg * reg_loss
                total_loss += total_batch_loss.item()
                num_batches += 1

        average_loss = total_loss / num_batches
        return average_loss

    def report_training(self, epoch_losses, validation_losses):
        logger.info("Training completed")
        logger.info(f"Total epochs: {self.epoch}")
        logger.info(f"Final training loss: {epoch_losses[-1]:.4f}")
        logger.info(f"Final validation loss: {validation_losses[-1]:.4f}")
        logger.info(f"Best model saved at: {
                    self.best_model_path} with loss: {self.best_loss:.4f}")

        epochs = range(1, self.epoch + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, epoch_losses, label='Training Loss')
        plt.plot(epochs, validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs. Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(
            os.getcwd(), 'training_vs_validation_loss.png'))
        plt.close()
