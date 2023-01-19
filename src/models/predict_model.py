from pathlib import Path
from PIL import Image
import click
import timm
from torchvision import transforms


# import helper
import matplotlib.pyplot as plt

# import numpy as np
import torch

from src.data.make_dataset import ASLDataset
from src.models.model import MyAwesomeModel


@click.command()
@click.argument("img_path")
@click.option(
    "--checkpoint",
    default="models/latest/model.ckpt",
    help="Checkpoint file",
)
@click.option("--show_image", default=False, help="Show matplotlib image", is_flag=True)
def main(img_path: Path, checkpoint: Path, show_image: bool) -> None:
    # print("Evaluating until hitting the ceiling")

    root_path = Path()
    # print(root_path.absolute() / "data/processed")
    test_set = ASLDataset(data_folder=root_path / "data/processed", train=False, onehotencoded=True)
    #img, y = test_set[int(i)]

    model = MyAwesomeModel.load_from_checkpoint(checkpoint)
    # disable randomness, dropout, etc...
    model.eval()

    # Get image from path
    img = Image.open(img_path)
    # Convert PIL img to tensor
    convert_tensor = transforms.ToTensor()
    img_t = convert_tensor(img)

    x = img_t.view(1, *img_t.shape)

    # print(x.shape)

    # predict with the model
    y_hat = model(x)

    # ps = torch.exp(y_hat)
    _, pred_class = y_hat.topk(1, dim=1)

    pred_y = int(pred_class[0, 0])
    #true_y = int([i for i, _ in enumerate(y) if y[i] == 1][0])
    # click.echo(f"predict y item: {pred_y}")
    # click.echo(f"true y item: {true_y}")
    # print([i for i in test_set.classes.keys() if test_set.classes[i] == pred_y])
    click.echo(
        f"predicted letter: {[i for i in test_set.classes.keys() if test_set.classes[i] == pred_y][0]}"
    )
    # click.echo(f"true y letter: {[i for i in test_set.classes.keys() if test_set.classes[i] == true_y][0]}")

    # Show image in plot

    if show_image:
        #img_show = img.permute(1, 2, 0)
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    main()
