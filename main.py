import argparse
from datetime import datetime
from data.utils import *
from train import Train
from data.TripletsDataset import *

def get_parameter():
    parser = argparse.ArgumentParser()
    # expected 5 files inside
    # 1. entity2id.txt 2. relation2id.txt 3. train2id.txt 4.test2id.txt 5.valid2id.txt   
    parser.add_argument('-dataset', default='datasets/FB15K237', type=str,
                        help='Dataset name, must be the same as folder name')
 
    

    return parser.parse_args()


def main():
    args = get_parameter()

    dataset_dir = args.dataset 
    # check if folder and the 5 files exists
    if (not check_folder_and_files(dataset_dir)):
        return
    
    
    
    dataset = TripletsDataset(dataset_dir) 
    
    model = Train(dataset, model_name='TransE', lr=0.001,
                  embedding_dimension=200, epoch=1)
    model.start()


if __name__ == '__main__':
    main()
