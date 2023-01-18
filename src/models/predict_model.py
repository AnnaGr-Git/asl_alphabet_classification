from pathlib import Path

import click

# import helper
import matplotlib.pyplot as plt

# import numpy as np
import torch

from src.data.make_dataset import ASLDataset
from src.models.model import MyAwesomeModel


@click.command()
@click.argument("i")
@click.option(
    "--checkpoint",
    default="lightning_logs/version_0/checkpoints/epoch=9-step=20.ckpt",
    help="Checkpoint file",
)
def main(i: int, checkpoint: Path) -> None:
    print("Evaluating until hitting the ceiling")

    root_path = Path()
    print(root_path.absolute() / "data/processed")

    test_set = ASLDataset(data_folder=root_path / "data/processed", train=False, onehotencoded=True)

    x, y = test_set[int(i)]
    x = x.view(1, *x.shape)

    model = MyAwesomeModel.load_from_checkpoint(checkpoint)

    # disable randomness, dropout, etc...
    model.eval()

    print(x.shape)

    # predict with the model
    y_hat = model(x)

    # ps = torch.exp(y_hat)
    _, top_class = y_hat.topk(1, dim=1)
    click.echo(int(top_class[0, 0]))
    # Plot the image and probabilities
    img: torch.Tensor = y_hat.detach()
    # helper.view_classify(img, ps)

    # fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    axisshift = img.resize_(3, 192, 192).reshape(192, 192, 3).numpy().squeeze()
    plt.imshow(axisshift)
    plt.show()


if __name__ == "__main__":
    main()
