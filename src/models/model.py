from typing import Any

import timm
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchvision import transforms
import matplotlib.pyplot as plt


class MyAwesomeModel(LightningModule):
    def __init__(self, lr:float=1e-2) -> None:
        super().__init__()
        self.m = timm.create_model("resnet18", pretrained=True, num_classes=29)
        self.lr = lr
        self.data_cfg = timm.data.resolve_data_config(self.m.pretrained_cfg)
        #self.data_transform = timm.data.create_transform(**self.data_cfg)
        self.data_transform = transforms.Compose([
            transforms.Resize(size=256, interpolation=transforms.functional.InterpolationMode.BILINEAR, max_size=None, antialias=None),
            transforms.CenterCrop(size=(224, 224)),
            transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])

        # print(f"Data Transform: {self.data_transform}")

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
        # Transform data
        image_tensor = self.data_transform(x)
        return self.m(image_tensor)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    m = MyAwesomeModel()

    dummy_data = torch.rand([64, 3, 128, 128])

    with torch.no_grad():
        out = m(dummy_data)

    # print(out.shape)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    print(probabilities.shape)
