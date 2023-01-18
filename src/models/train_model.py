import hydra
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
import os
from src.data.make_dataset import ASLDataset
from src.models.model import MyAwesomeModel
from pathlib import Path

root_path = Path(".").absolute()#.parent.absolute()
print(f"Root Path outside: {root_path}")


@hydra.main(version_base=None, config_path=os.path.join(root_path, "config/"), config_name='default_config.yaml')
def main(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    hparams = config.experiment

    model = MyAwesomeModel(lr=hparams["learning_rate"])  # this is our LightningModule
    trainer = Trainer(max_epochs=hparams["epochs"], log_every_n_steps=1)

    train_set = ASLDataset(data_folder=root_path / hparams["dataset_path"], train=True, onehotencoded=hparams["onehotencoded"])
    # # Split dataset in train and validation set
    train_set_size = int(len(train_set) * hparams["trainsize"])
    valid_set_size = len(train_set) - train_set_size
    train_set, val_set = random_split(train_set, [train_set_size, valid_set_size])

    test_set = ASLDataset(data_folder=root_path / hparams["dataset_path"], train=False, onehotencoded=hparams["onehotencoded"])


    train_loader = DataLoader(train_set, batch_size=hparams["batch_size"], shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_set, batch_size=64)
    # test_loader = DataLoader(test_set, batch_size=64)

    trainer.fit(
        model,
        train_dataloaders=train_loader
        # , val_dataloaders=val_loader
    )

if __name__ == "__main__":

    main()
