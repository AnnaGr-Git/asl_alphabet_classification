import io
from typing import Any

import matplotlib.pyplot as plt

import timm
import torch
import torchmetrics
from PIL import Image
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchvision import transforms


class MyAwesomeModel(LightningModule):
    def __init__(self, lr: float = 1e-2) -> None:

        """Model for ASL classification"""

        super().__init__()
        self.num_classes = 29
        self.m = timm.create_model("resnet18", pretrained=True, num_classes=self.num_classes)
        self.lr = lr

        self.data_cfg = timm.data.resolve_data_config(self.m.pretrained_cfg)
        # self.data_transform = timm.data.create_transform(**self.data_cfg)
        self.data_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=256,
                    interpolation=transforms.functional.InterpolationMode.BILINEAR,
                    max_size=None,
                    antialias=None,
                ),
                transforms.CenterCrop(size=(224, 224)),
                transforms.Normalize(
                    mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                    std=torch.tensor([0.2290, 0.2240, 0.2250]),
                ),
            ]
        )

        # print(f"Data Transform: {self.data_transform}")

        # freeze all layers except last
        for name, param in self.m.named_parameters():
            if "fc.weight" not in name and "fc.bias" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform data and run inference"""
        image_tensor = self.data_transform(x)
        return self.m(image_tensor)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """Run model training"""
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        # Logging
        acc = (target.argmax(dim=-1) == preds.argmax(dim=-1)).float().mean()
        self.log("training_loss", loss)
        self.log("training_acc", acc)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, Any]:
        """Run model validation"""
        data, target = batch
        preds = self.forward(data)
        loss = self.criterium(preds, target)
        # on_epoch=True by default in `validation_step`,
        # so it is not necessary to specify
        acc = (target.argmax(dim=-1) == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc)
        return {"loss": loss, "preds": preds, "target": target}

    def validation_epoch_end(self, outs: list[dict[str, Any]]) -> None:
        """Run at the end of the epoch and save the confusion matrix
        each forward pass, thus leading to wrong accumulation. In practice do the following:"""
        tb = self.logger.experiment  # noqa

        outputs = torch.cat([tmp["preds"] for tmp in outs])
        labels = torch.cat([tmp["target"] for tmp in outs])

        confusion = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        ).to(outputs.get_device())
        confusion(outputs.argmax(dim=-1), labels.argmax(dim=-1))
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=list(range(self.num_classes)),
            columns=list(range(self.num_classes)),
        )

        fig, ax = plt.subplots(figsize=(20, 10))
        fig.subplots_adjust(left=0.05, right=0.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d", ax=ax)
        ax.legend(
            list(range(self.num_classes)),
            list(range(self.num_classes)),
            handler_map={int: IntHandler()},
            loc="upper left",
            bbox_to_anchor=(1.2, 1),
        )
        buf = io.BytesIO()

        plt.savefig(buf, format="jpeg", bbox_inches="tight")
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        self.logger.log_image(key="val_confusion_matrix", images=[im])
        plt.clf()

    def configure_optimizers(self) -> Any:
        """Configure optimizer"""
        return optim.Adam(self.parameters(), lr=self.lr)


class IntHandler:
    """Helper class to build the confusion matrix"""

    def legend_artist(self, legend: str, orig_handle: Any, fontsize: int, handlebox: Any) -> str:
        """Helper function for building the confusion matrix"""
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


if __name__ == "__main__":
    m = MyAwesomeModel()

    dummy_data = torch.rand([64, 3, 128, 128])

    with torch.no_grad():
        out = m(dummy_data)

    # print(out.shape)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    print(probabilities.shape)
