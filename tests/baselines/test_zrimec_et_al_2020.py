import torch

from deep_bac.baselines import ZrimecEtAlModel
from deep_bac.modelling.modules.utils import count_parameters


def test_zrimec_et_al_2020():
    batch_size = 16
    seq_length = 2560

    x = torch.randn(batch_size, 4, seq_length)

    model = ZrimecEtAlModel()
    out = model(x)
    n_params = count_parameters(model)
    print("Number of trainable model parameters: ", n_params)
    assert out.shape == (batch_size, 1)
