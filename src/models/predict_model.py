from pathlib import Path

import click
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

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
    """Main function to predict the asl letter given an image path"""
    # print("Evaluating until hitting the ceiling")

    classes = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "J": 9,
        "K": 10,
        "L": 11,
        "M": 12,
        "N": 13,
        "O": 14,
        "P": 15,
        "Q": 16,
        "R": 17,
        "S": 18,
        "T": 19,
        "U": 20,
        "V": 21,
        "W": 22,
        "X": 23,
        "Y": 24,
        "Z": 25,
        "del": 26,
        "nothing": 27,
        "space": 28,
    }
    # img, y = test_set[int(i)]

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
    # true_y = int([i for i, _ in enumerate(y) if y[i] == 1][0])
    # click.echo(f"predict y item: {pred_y}")
    # click.echo(f"true y item: {true_y}")
    # print([i for i in test_set.classes.keys() if test_set.classes[i] == pred_y])

    click.echo(
        f"predicted letter: \
            {[i for i in classes.keys() if classes[i] == pred_y][0]}"
    )
    # click.echo(f"true y letter: {[i for i in test_set.classes.keys()
    # if test_set.classes[i] == true_y][0]}")

    # Show image in plot

    if show_image:
        # img_show = img.permute(1, 2, 0)
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    main()
