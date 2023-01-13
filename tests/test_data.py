from src.data.make_dataset import ASLDataset, preprocess
from pathlib import Path
import os
import torch

def test_data():
    project_dir = Path().parent.absolute()
    data_dir = os.path.join(project_dir, "data/")
    
    onehotencoded = True
    N_samples = 5
    N_classes = 29
    img_size = 192
    preprocess(num_samples=N_samples,
               img_size=img_size,
               input_filepath=os.path.join(data_dir,"raw/"),
               output_filepath=os.path.join(data_dir, "processed/"),
               interim_filepath=os.path.join(data_dir, "interim/"))
    
    N_train = N_samples*N_classes
    N_test = 28

    print("Done preprocessing.")
    
    # Labels are one-hot-encoded
    trainset = ASLDataset(data_folder=os.path.join(data_dir, "processed/"), train=True, onehotencoded=onehotencoded)
    testset = ASLDataset(data_folder=os.path.join(data_dir, "processed/"), train=False, onehotencoded=onehotencoded)
    assert len(trainset) == N_train
    assert len(testset) == N_test
    assert trainset.imgs.shape == torch.Size([N_train, 3, img_size, img_size])
    assert trainset.labels.shape == torch.Size([N_train, N_classes])

    # Labels are encoded labels (int)
    onehotencoded = False
    trainset = ASLDataset(data_folder=os.path.join(data_dir, "processed/"), train=True, onehotencoded=onehotencoded)
    assert trainset.labels.shape == torch.Size([N_train])
