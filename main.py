import argparse
from datetime import datetime
import logging
from data.utils import check_folder_and_files
from Train import Train
from data.Loader import *
from data.TrainDataLoader import *
import torch
from torch.utils.data import DataLoader
from data.TestDataLoader import *
from Test import *


def print_cuda_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            device_name = torch.cuda.get_device_name(i)
            print(f"CUDA Device {i}: {device_name}")
    else:
        print("No CUDA devices available. Running on CPU.")


def main():
    
    print_cuda_devices()
    
    with open('./models/TransE_WN18.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_config = config['Train']
    
    #NOTE: 
    config = {
        'dataset_dir': train_config['dataset'],
        'neg_sample': train_config['neg_sample'],
        'epoch': train_config['epoch'],
        'lr': train_config['lr'],
        'model': train_config['model'],
        'dim': train_config['dim'],
        'neg_ratio': train_config['neg_ratio'],
        'batch_size': train_config['batch_size'],
        'max_threads': train_config['max_threads'],
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'validation_rate': train_config['validation_rate']
    }


    loader = Loader.config(config)
    # read train data.
    train_dataloader = TrainDataLoader(loader) 
        
    # #Initialize and start training
    trainer = Train(train_dataloader)
    trainer.start()

    test_dataloader = TestDataLoader(loader)
  
    tester = Test(test_dataloader)
    tester.run_link_prediction()
    
    
if __name__ == '__main__':
    main()
