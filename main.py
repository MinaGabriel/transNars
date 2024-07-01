import argparse
from datetime import datetime
import logging
from data.utils import check_folder_and_files
from Train import Train
from data.TripletsDataset import TripletsDataset
from data.NegativeDataset import NegativeDataset 
import torch 
from torch.utils.data import DataLoader
def get_parameter():
    parser = argparse.ArgumentParser()
    # Expected 5 files inside the dataset directory:
    # 1. entity2id.txt 2. relation2id.txt 3. train2id.txt 4. test2id.txt 5. valid2id.txt
    parser.add_argument('-dataset', default='datasets/FB15K237', type=str, help='Dataset directory')
    parser.add_argument('-epoch', default=100, type=int, help="Number of epochs")
    parser.add_argument('-lr', default=0.01, type=float, help="Learning rate")
    parser.add_argument('-model', default="TransE", type=str, help="Knowledge graph embedding model")
    parser.add_argument('-dim', default=200, type=int, help="Embedding dimension")
    parser.add_argument('-neg_sample', default="c", type=str, help="Negative samples algorithm")
    parser.add_argument('-neg_ratio', default=25, type=int, help="Negative sampling ratio")
    parser.add_argument('-batch_size', default=512, type=int, help="Batch size")
    parser.add_argument('-device', default="cuda:0" if torch.cuda.is_available() else "cpu", type=str, help="Device to use (cpu|cuda:0)")

    args = parser.parse_args()
    return args

def print_cuda_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            device_name = torch.cuda.get_device_name(i)
            print(f"CUDA Device {i}: {device_name}")
    else:
        print("No CUDA devices available. Running on CPU.")

def main():
    args = get_parameter()
    print_cuda_devices()

    dataset_dir = args.dataset
    neg_sample = args.neg_sample
    epoch = args.epoch
    lr = args.lr
    model_name = args.model
    dim = args.dim
    neg_ratio = args.neg_ratio
    batch_size = args.batch_size
    device = args.device

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training with the following parameters:")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Negative Sample: {neg_sample}")
    logger.info(f"Epochs: {epoch}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Embedding dimension: {dim}")
    logger.info(f"Negative sampling ratio: {neg_ratio}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    
    # Check if folder and the required files exist
    if not check_folder_and_files(dataset_dir):
        logger.error("Dataset directory or required files are missing.")
        return

    # Load dataset
    dataset = TripletsDataset(dataset_dir)
    negative_dataset = NegativeDataset(dataset, f'neg_sam_{neg_sample}.txt')
    
    # Initialize and start training
    trainer = Train(dataset,negative_dataset, model_name, lr, dim, epoch, batch_size, device, neg_ratio)
    trainer.start()

if __name__ == '__main__':
    main()
