
from tqdm import tqdm 
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

class ConvE(nn.Module):
    def __init__(self, config, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = config.embedding_dim
        self.device = config.device
        
        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout)
        self.feature_map_dropout = torch.nn.Dropout(config.feature_map_dropout)
        self.hidden_layer_dropout = torch.nn.Dropout(config.hidden_layer_dropout)
        
        self.entity_embeddings = nn.Embedding(num_entities, self.embedding_dim).to(self.device)
        self.relation_embeddings = nn.Embedding(num_relations, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        self.loss = torch.nn.BCELoss()
        
        
        
        
        
        