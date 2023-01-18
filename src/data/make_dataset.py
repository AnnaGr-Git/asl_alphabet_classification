# -*- coding: utf-8 -*-
import glob
import logging
import ntpath
import os
import pathlib
import typing
from zipfile import ZipFile

import click
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ASLDataset(Dataset):
    """
    A class to load the MNIST dataset

    Attributes
    ----------
    data_folder : str
        path of directory where the processed data is stored
    images : torch.tensor (num_samples x image_size[0] x image_size[1])
        images of the dataset, stored in one tensor
    labels : torch.tensor
        labels of each sample (num_samples x 1
        )

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(
        self,
        data_folder: typing.Union[str, pathlib.Path] = "/data/processed",
        train: bool = True,
        img_file: str = "images.pt",
        label_file: str = "labels.npy",
        onehotencoded: bool = True,
    ) -> None:
        if train:
            dir = "train/"
        else:
            dir = "test/"
        self.root_dir = os.path.join(data_folder, dir)
        self.img_file = img_file
        self.label_file = label_file
        self.imgs = self.load_images()
        self.labels, self.classes = self.load_labels(onehotencoded)

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def load_images(self) -> torch.Tensor:
        return torch.load(os.path.join(self.root_dir, self.img_file))

    def load_labels(self, onehotencoded: bool) -> typing.Tuple[torch.Tensor, dict]:
        labels = np.load(os.path.join(self.root_dir, self.label_file))

        classes = np.unique(labels)

        class_dict = {}
        idx = 0
        for cl in classes:
            class_dict[cl] = idx
            idx += 1

        encoded = []
        encoded.extend([class_dict[label] for label in labels])
        encoded = torch.tensor(encoded, dtype=torch.int64)

        if onehotencoded:
            encoded = torch.nn.functional.one_hot(encoded)

        # print(type(encoded))
        # print(type(class_dict))

        return encoded, class_dict

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        return (self.imgs[idx].float(), self.labels[idx].float())


@click.group()
def cli() -> None:
    pass


def preprocess(
    num_samples: int,
    img_size: int,
    input_filepath: str,
    output_filepath: str,
    interim_filepath: str,
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making project data set from raw data")

    # Get path of zip
    print(input_filepath)
    file_name = glob.glob(os.path.join(input_filepath, "*.zip"))[0]
    print(file_name)

    # Define intermediate path to train and test directory
    training_folder = os.path.join(interim_filepath, "asl_alphabet_train/asl_alphabet_train/")
    test_folder = os.path.join(interim_filepath, "asl_alphabet_test/asl_alphabet_test/")

    if not (os.path.exists(training_folder) and os.path.exists(test_folder)):
        # Extract zip file
        z = ZipFile(file_name, "r")
        print(f"Extract data of zip folder in {interim_filepath}...")
        z.extractall(path=interim_filepath)

    # Get all training files
    convert_tensor = transforms.ToTensor()

    class_names = []
    for file in os.listdir(training_folder):
        class_names.append(file)

    count = 0
    train_labels = []
    for c in class_names:
        print(f"Get training data of class {c}")
        path = os.path.join(training_folder, c)
        # Get all images
        trainig_files = np.array(glob.glob(os.path.join(path, "*.jpg")))
        for p in trainig_files[:num_samples]:
            # Read image
            img = Image.open(p)

            # Resize image
            if img_size is not None:
                img = img.resize((img_size, img_size))

            # Convert PIL img to tensor
            img_n = convert_tensor(img)

            # Normalize image
            # mean, std = img_t.mean([1, 2]), img_t.std([1, 2])
            # transform_norm = transforms.Normalize(mean, std)
            # img_n = transform_norm(img_t)

            img_n = img_n.unsqueeze(0)

            # Add to images tensor
            if count == 0:
                train_images = img_n
            else:
                train_images = torch.concat((train_images, img_n), dim=0)
            count += 1

            # Add label to list
            train_labels.append(c)

    print(f"Shape Train images: {train_images.shape}")
    print(f"Shape Train labels: {np.shape(train_labels)}")

    # Get all test files
    test_files = glob.glob(os.path.join(test_folder, "*.jpg"))
    count = 0
    test_labels = []
    for file in test_files:
        # Get class
        head, tail = ntpath.split(file)
        idx = tail.find("_")
        label = tail[:idx]
        test_labels.append(label)

        # Read image
        img = Image.open(file)

        # Resize image
        if img_size is not None:
            img = img.resize((img_size, img_size))

        # Convert PIL img to tensor
        img_n = convert_tensor(img)

        # Normalize image
        # mean, std = img_t.mean([1, 2]), img_t.std([1, 2])
        # transform_norm = transforms.Normalize(mean, std)
        # img_n = transform_norm(img_t)

        img_n = img_n.unsqueeze(0)

        # Add to images tensor
        if count == 0:
            test_images = img_n
        else:
            test_images = torch.concat((test_images, img_n), dim=0)
        count += 1

    print(f"Shape Test images: {test_images.shape}")
    print(f"Shape Test labels: {np.shape(test_labels)}")

    # Save data in files
    trainpath = os.path.join(output_filepath, "train/")
    if not os.path.isdir(trainpath):
        os.makedirs(trainpath)
    testpath = os.path.join(output_filepath, "test/")
    if not os.path.isdir(testpath):
        os.makedirs(testpath)

    torch.save(train_images, os.path.join(trainpath, "images.pt"))
    np.save(os.path.join(trainpath, "labels.npy"), np.array(train_labels))

    torch.save(test_images, os.path.join(testpath, "images.pt"))
    np.save(os.path.join(testpath, "labels.npy"), np.array(test_labels))


@click.command()
@click.option("--num_samples", default=5, help="Number of training samples per class")
@click.option(
    "--img_size",
    default=192,
    help="Size that image should be resized to. For no resizing, pass None.",
)
@click.option("--input_filepath", default="data/raw", help="Filepath where raw data is located.")
@click.option(
    "--output_filepath", default="data/processed", help="Filepath where raw data is located."
)
@click.option(
    "--interim_filepath", default="data/interim", help="Filepath where intermediate data is saved."
)
def preprocess_command(num_samples, img_size, input_filepath, output_filepath, interim_filepath):
    preprocess(num_samples, img_size, input_filepath, output_filepath, interim_filepath)


cli.add_command(preprocess_command)

if __name__ == "__main__":
    cli()
