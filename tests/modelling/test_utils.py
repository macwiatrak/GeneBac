import torch

from deep_bac.modelling.utils import remove_ignore_index


def test_remove_ignore_index():
    loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([1, 2, 3, -100])
    loss = remove_ignore_index(loss, labels)
    assert torch.allclose(loss, torch.tensor([1.0, 2.0, 3.0]))
