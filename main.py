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
def get_parameter():
    parser = argparse.ArgumentParser()
    # Expected 5 files inside the dataset directory:
    # 1. entity2id.txt 2. relation2id.txt 3. train2id.txt 4. test2id.txt 5. valid2id.txt
    parser.add_argument('-dataset', default='./datasets/benchmarks/WN18/', type=str, help='Dataset directory')
    parser.add_argument('-epoch', default=100, type=int, help="Number of epochs")
    parser.add_argument('-lr', default=0.01, type=float, help="λ: Learning rate")
    parser.add_argument('-model', default="TransE", type=str, help="Knowledge graph embedding model")
    parser.add_argument('-dim', default=50, type=int, help="K: Embedding dimension")
    parser.add_argument('-neg_sample', default="c", type=str, help="Negative samples algorithm")
    parser.add_argument('-neg_ratio', default=25, type=int, help="Negative sampling ratio")
    parser.add_argument('-batch_size', default=75, type=int, help="Batch size")
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
    model = args.model
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
    logger.info(f"Learning rate (λ): {lr}")
    logger.info(f"Model: {model}")
    logger.info(f"Embedding dimension (κ): {dim}")
    logger.info(f"Negative sampling ratio: {neg_ratio}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")

    # Check if folder and the required files exist
    if not check_folder_and_files(dataset_dir):
        logger.error("Dataset directory or required files are missing.")
        return

    loader = Loader(dataset_dir, neg_sample,
                    epoch, lr, model, dim, neg_ratio, 
                    batch_size, device)

    # read train data.
    train_dataloader = TrainDataLoader(loader) 
        
    # #Initialize and start training
    trainer = Train(train_dataloader)
    #trainer.start()

    test_dataloader = TestDataLoader(loader)
  
    tester = Test(test_dataloader)
    tester.run_link_prediction()
    
    print('Done with training')
    
if __name__ == '__main__':
    main()
