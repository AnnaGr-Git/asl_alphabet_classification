from typing import Any

import timm
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim


class MyAwesomeModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.m = timm.create_model("resnet18", pretrained=True, num_classes=29)

        # freeze all layers except last
        for name, param in self.m.named_parameters():
            if "fc.weight" not in name and "fc.bias" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.criterium = nn.CrossEntropyLoss()
        # for name, param in self.m.named_parameters():
        #     print(f"{name}: {param.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=1e-2)


if __name__ == "__main__":
    m = MyAwesomeModel()

    dummy_data = torch.rand([64, 3, 128, 128])

    with torch.no_grad():
        out = m(dummy_data)

    print(out.shape)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    print(probabilities.shape)
