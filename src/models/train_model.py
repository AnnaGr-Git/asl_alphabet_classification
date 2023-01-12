from pytorch_lightning import Trainer
from src.models.model import MyAwesomeModel

model = MyAwesomeModel()  # this is our LightningModule
trainer = Trainer()
trainer.fit(model)
