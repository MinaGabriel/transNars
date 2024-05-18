import argparse
from datetime import datetime
from utils import *
from train import Train


def get_parameter():
    parser = argparse.ArgumentParser()
    # expected 5 files inside
    # 1. entities.txt 2. relations.txt 3. test.txt 4. train.txt 5. valid.txt:
    parser.add_argument('-dataset', default='my_toy', type=str,
                        help='Dataset name, must be the same as folder name')
    return parser.parse_args()


def main():
    args = get_parameter()

    dataset_dir = args.dataset
    # check if folder and the 5 files exists
    if (not check_folder_and_files(dataset_dir)):
        return

    dataset = TripletsDataset(dataset_dir)
    model = Train(dataset, model_name='TransE', lr=0.01, embedding_dimension=2, epoch=5)
    model.start()


if __name__ == '__main__':
    main()
