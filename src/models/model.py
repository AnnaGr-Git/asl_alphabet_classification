from typing import Any
from pytorch_lightning import LightningModule
import timm
from torch import nn, optim


class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model('resnet18', pretrained=True, num_classes=24)

        # freeze all layers except last
        for name, param in self.m.named_parameters():
            if 'fc.weight' not in name and 'fc.bias' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.m(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=1e-2)

if __name__ == "__main__":
    m = MyAwesomeModel()