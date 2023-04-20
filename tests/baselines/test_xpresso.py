import torch

from deep_bac.baselines import Xpresso


def test_xpresso():
    batch_size = 16
    seq_length = 2560

    x = torch.randn(batch_size, 4, seq_length)

    model = Xpresso()
    out = model(x)
    assert out.shape == (batch_size, 1)
