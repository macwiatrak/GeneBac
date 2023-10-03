import torch

from genebac.baselines.expression.simple_cnn import SimpleCNN
from genebac.modelling.modules.utils import count_parameters


def test_simple_cnn():
    batch_size = 16
    seq_length = 2560

    x = torch.randn(batch_size, 4, seq_length)

    model = SimpleCNN()
    out = model(x)
    n_params = count_parameters(model)
    print("Number of trainable model parameters: ", n_params)
    assert out.shape == (batch_size, 1)
