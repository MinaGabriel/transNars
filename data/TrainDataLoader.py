import numpy as np
import torch
import os
import logging
import ctypes
import time
from torch.utils.data import Dataset, DataLoader
from data.Loader import Loader
from data.utils import *
from torch import Tensor
from ctypes import c_char_p, POINTER, c_int64, c_float
from data.TrainDataset import TrainDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainDataLoader:
    def __init__(self, loader: Loader):
        self.lib = ctypes.CDLL("./Base.so")
        self.lib.trainDataLoader.argtypes = [
            c_char_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.lib.getSortedTrainByHead.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]

        self.loader = loader
        self.read()

    def read(self):
        batch_h = np.zeros(
            self.loader.num_train * (1 + self.loader.config["neg_ratio"]),
            dtype=np.int64,
        )
        batch_t = np.zeros(
            self.loader.num_train * (1 + self.loader.config["neg_ratio"]),
            dtype=np.int64,
        )
        batch_r = np.zeros(
            self.loader.num_train * (1 + self.loader.config["neg_ratio"]),
            dtype=np.int64,
        )
        batch_y = np.zeros(
            self.loader.num_train * (1 + self.loader.config["neg_ratio"]),
            dtype=np.float32,
        )
        batch_h_addr = batch_h.__array_interface__["data"][0]
        batch_t_addr = batch_t.__array_interface__["data"][0]
        batch_r_addr = batch_r.__array_interface__["data"][0]
        batch_y_addr = batch_y.__array_interface__["data"][0]
        self.lib.trainDataLoader(
            ctypes.create_string_buffer(
                self.loader.config["dataset"].encode(),
                len(self.loader.config["dataset"]) * 2,
            ),
            batch_h_addr,
            batch_t_addr,
            batch_r_addr,
            batch_y_addr,
            self.loader.config["neg_ratio"],
            self.loader.config["max_threads"],
        )
        data = {
            "batch_h": batch_h,
            "batch_r": batch_r,
            "batch_t": batch_t,
            "batch_y": batch_y,
        }
        custom_dataset = TrainDataset(data, self.loader)
        # batch size is 1, data already in batches array.
        self.data = DataLoader(custom_dataset, batch_size=1, shuffle=True)

    def get_sorted_train_by_head(self):
        """
        get sorted list of training by head -> relation -> tail
        """
        batch_h = np.zeros(self.loader.num_train, dtype=np.int64)
        batch_t = np.zeros(self.loader.num_train, dtype=np.int64)
        batch_r = np.zeros(self.loader.num_train, dtype=np.int64)

        start_idx = np.zeros(self.loader.num_entities, dtype=np.int64)
        end_idx = np.zeros(self.loader.num_entities, dtype=np.int64)

        batch_h_addr = batch_h.__array_interface__["data"][0]
        batch_t_addr = batch_t.__array_interface__["data"][0]
        batch_r_addr = batch_r.__array_interface__["data"][0]
        start_idx_addr = start_idx.__array_interface__["data"][0]
        end_idx_addr = end_idx.__array_interface__["data"][0]

        self.lib.getSortedTrainByHead(
            batch_h_addr, batch_t_addr, batch_r_addr, start_idx_addr, end_idx_addr
        )

        return batch_h, batch_r, batch_t, start_idx, end_idx
