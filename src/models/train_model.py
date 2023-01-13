from pathlib import Path
from pytorch_lightning import Trainer
from src.models.model import MyAwesomeModel
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.data.make_dataset import ASLDataset

model = MyAwesomeModel()  # this is our LightningModule
#trainer = Trainer(accelerator='gpu', devices=1, max_epochs=10, log_every_n_steps=1)
trainer = Trainer(max_epochs=10, log_every_n_steps=1)
#trainer = Trainer(max_epochs=10, log_every_n_steps=1)
#trainer = Trainer(accelerator="mps", devices=1)
# Selects the accelerator


root_path = Path()
# print(root_path)

train_set = ASLDataset(data_folder=root_path / "data/processed", train=True, onehotencoded=True)
# # Split dataset in train and validation set
train_set_size = int(len(train_set) * 0.7)
valid_set_size = len(train_set) - train_set_size
train_set, val_set = random_split(train_set, [train_set_size, valid_set_size])

test_set = ASLDataset(data_folder=root_path / "data/processed", train=False, onehotencoded=True)


train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_set, batch_size=64)
# test_loader = DataLoader(test_set, batch_size=64)

trainer.fit(model, train_dataloaders=train_loader
# , val_dataloaders=val_loader
)
