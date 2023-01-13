import numpy as np
from torch.utils.data import random_split
from src.data.make_dataset import ASLDataset


def main():
    onehotencoded = True
    train_test_split = 0.8

    train_set = ASLDataset(data_folder="/home/anna/Dokumente/MLops/asl_alphabet_classification/data/processed", train=True, onehotencoded=onehotencoded)
    print(train_set.classes)
    print(train_set.imgs.shape)
    print(train_set.labels.shape)

    test_set = ASLDataset(data_folder="/home/anna/Dokumente/MLops/asl_alphabet_classification/data/processed", train=False, onehotencoded=onehotencoded)
    print(test_set.imgs.shape)

    # Split dataset in train and validation set
    train_set_size = int(len(train_set) * train_test_split)
    valid_set_size = len(train_set) - train_set_size
    train_set, val_set = random_split(train_set, [train_set_size, valid_set_size])
    print(type(val_set))

if __name__ == '__main__':
    main()