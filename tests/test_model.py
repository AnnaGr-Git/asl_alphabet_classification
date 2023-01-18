import torch

from src.models.model import MyAwesomeModel


def test_model():
    """Check if model output has the correct shape"""
    m = MyAwesomeModel()

    dummy_data = torch.rand([64, 3, 128, 128])

    with torch.no_grad():
        out = m(dummy_data)
    assert out.shape == torch.Size([64, 29])
