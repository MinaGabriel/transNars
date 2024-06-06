import argparse
from datetime import datetime
from data.utils import *
from Train import Train
from data.TripletsDataset import *
from Tester import Tester

def get_parameter():
    parser = argparse.ArgumentParser()
    # expected 5 files inside
    # 1. entity2id.txt 2. relation2id.txt 3. train2id.txt 4.test2id.txt 5.valid2id.txt
    parser.add_argument('-dataset', default='datasets/Countries', type=str, help='Dataset')
    parser.add_argument('-epoch', default=100, type=int, help="")
    parser.add_argument('-lr', default=0.001, type=float, help="learning rate")
    parser.add_argument('-model', default="TransE", type=str, help="kG embedding model")
    parser.add_argument('-dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=25, type=int, help="neg ratio")
    parser.add_argument('-batch_size', default=256, type=int, help="batch size")
    parser.add_argument('-device', default="cuda:0", type=str, help="(cpu|cuda:0)")
    
    return parser.parse_args()


def main():
    args = get_parameter()

    dataset_dir = args.dataset
    epoch = args.epoch
    lr = args.lr
    model_name = args.model
    dim = args.dim
    neg_ratio = args.neg_ratio
    batch_size = args.batch_size
    device = args.device
    
    # check if folder and the 5 files exists
    if (not check_folder_and_files(dataset_dir)):
        return

    dataset = TripletsDataset(dataset_dir)

    model = Train(dataset, model_name, lr, dim, epoch, batch_size,device, neg_ratio)
    model.start()


if __name__ == '__main__':
    main()
