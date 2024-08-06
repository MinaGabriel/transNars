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
from ConvE import ConvE
import ctypes
from ctypes import c_char_p, POINTER, c_int64, c_float

loader = Loader("ConvE_FB15K237.yaml")
test_dataloader = TestDataLoader(loader)
dataset = test_dataloader.data
model = ConvE(loader.num_entities, loader.num_relations)

model.load_state_dict(
    torch.load(
        f'./models/trained/{loader.config["model"]}_{loader.name}.pt',
        map_location=loader.device,
    )
)
model.to(loader.device)
model.eval()
lib = ctypes.CDLL("./Base.so")
lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64]
lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64]
num_entities = loader.num_entities

for _, (batch_h, batch_t, batch_r) in tqdm(enumerate(dataset), total=len(dataset)):
    batch_h, batch_r, batch_t = (
        batch_h.to(loader.device),
        batch_r.to(loader.device),
        batch_t.to(loader.device),
    )

    scores_h = model.forward(batch_t, batch_r)
    scores_h = scores_h.permute(1, 0)
    # inverse because the higher scores are similar to shortest distances
    inverted_scores = torch.max(scores_h) - scores_h + torch.min(scores_h)
    score = inverted_scores.detach().cpu().numpy()
    lib.testHead(score.__array_interface__["data"][0], _)

    scores_t = model.forward(batch_h, batch_r)
    scores_t = scores_t.permute(1, 0)
    # inverse because the higher scores are similar to shortest distances
    inverted_scores = torch.max(scores_t) - scores_t + torch.min(scores_t)
    score = inverted_scores.detach().cpu().numpy()
    lib.testTail(score.__array_interface__["data"][0], _)


lib.test_link_prediction()
