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
from data.ValidDataset import ValidDataset


class ValidDataLoader(object):
    def __init__(self, loader: Loader):
        self.lib = ctypes.CDLL("./Base.so")
        self.lib.validDataLoader.argtypes = [
            c_char_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.loader = loader
        self.read()

    def read(self):
        batch_h = np.zeros(self.loader.num_valid, dtype=np.int64)
        batch_t = np.zeros(self.loader.num_valid, dtype=np.int64)
        batch_r = np.zeros(self.loader.num_valid, dtype=np.int64)
        batch_h_addr = batch_h.__array_interface__["data"][0]
        batch_t_addr = batch_t.__array_interface__["data"][0]
        batch_r_addr = batch_r.__array_interface__["data"][0]
        self.lib.validDataLoader(
            ctypes.create_string_buffer(
                self.loader.config["dataset"].encode(),
                len(self.loader.config["dataset"]) * 2,
            ),
            batch_h_addr,
            batch_t_addr,
            batch_r_addr,
        )
        data = {"batch_h": batch_h, "batch_r": batch_r, "batch_t": batch_t}
        custom_dataset = ValidDataset(data)
        # NOTE: batch size is 1, validate 1 data sample at a time...
        self.data = DataLoader(custom_dataset, batch_size=8, shuffle=False)
