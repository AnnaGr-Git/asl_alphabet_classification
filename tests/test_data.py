from src.data.make_dataset import ASLDataset, preprocess
from pathlib import Path
import os
import torch
import random
import numpy as np

def test_data():
    project_dir = Path().parent.absolute()
    data_dir = os.path.join(project_dir, "data/")

    classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O",
               "P","Q","R","S","T","U","V","W","X","Y","Z","space","del","nothing"]

    onehotencoded = True
    N_samples = 5
    N_classes = len(classes)
    img_size = 192
    N_train = N_samples*N_classes
    N_test = N_classes

    # Create dummy data
    dummy_train_images = torch.rand(N_train,3,img_size,img_size)
    dummy_train_labels = np.array(classes)
    dummy_train_labels = np.concatenate((dummy_train_labels, np.array(random.choices(classes, k=N_train-len(classes)))))
    dummy_test_images = torch.rand(N_test,3,img_size,img_size)
    dummy_test_labels = np.array(classes)

    # Save data in files
    trainpath = "tests/dummydata/train/"
    if not os.path.isdir(trainpath):
        os.makedirs(trainpath)
    testpath = "tests/dummydata/test/"
    if not os.path.isdir(testpath):
        os.makedirs(testpath)

    torch.save(dummy_train_images, os.path.join(trainpath, "images.pt"))
    np.save(os.path.join(trainpath, "labels.npy"), np.array(dummy_train_labels))
    torch.save(dummy_test_images, os.path.join(testpath, "images.pt"))
    np.save(os.path.join(testpath, "labels.npy"), np.array(dummy_test_labels))
    
    # Labels are one-hot-encoded
    trainset = ASLDataset(data_folder="tests/dummydata/", train=True, onehotencoded=onehotencoded)
    testset = ASLDataset(data_folder="tests/dummydata/", train=False, onehotencoded=onehotencoded)
    assert len(trainset) == N_train
    assert len(testset) == N_test
    assert trainset.imgs.shape == torch.Size([N_train, 3, img_size, img_size])
    assert trainset.labels.shape == torch.Size([N_train, N_classes])

    # Labels are encoded labels (int)
    onehotencoded = False
    trainset = ASLDataset(data_folder="tests/dummydata/", train=True, onehotencoded=onehotencoded)
    assert trainset.labels.shape == torch.Size([N_train])
