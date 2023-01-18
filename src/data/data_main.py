from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split

from src.data.make_dataset import ASLDataset


def main() -> None:
    """Testing dataset generation"""
    onehotencoded = True
    train_test_split = 0.8

    root_path = Path()
    # print(root_path)

    train_set = ASLDataset(
        data_folder=root_path / "data/processed", train=True, onehotencoded=onehotencoded
    )
    print(f"Classes: {train_set.classes}")
    print(f"Train images shape: {train_set.imgs.shape}")
    print(f"Train labels shape: {train_set.labels.shape}")

    test_set = ASLDataset(
        data_folder=root_path / "data/processed", train=False, onehotencoded=onehotencoded
    )
    print(f"Test images shape: {test_set.imgs.shape}")

    # Split dataset in train and validation set
    train_set_size = int(len(train_set) * train_test_split)
    valid_set_size = len(train_set) - train_set_size

    print(f"Train set size: {train_set_size}")
    print(f"Valid test size: {valid_set_size}")
    train_set, val_set = random_split(train_set, [train_set_size, valid_set_size])
    print(type(val_set))

    # Show image in plot
    img, label = train_set[8]
    img = torch.swapaxes(img, 2, 0)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
