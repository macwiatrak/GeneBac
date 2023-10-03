import torch

from genebac.modelling.modules.gene_bac_encoder import GeneBacEncoder
from genebac.modelling.modules.utils import count_parameters


def test_gene_bac_encoder():
    batch_size = 1
    seq_length = 2560
    in_channels = 4
    n_filters_init = 128
    n_bottleneck_layer = 64

    x = torch.rand(batch_size, in_channels, seq_length)
    model = GeneBacEncoder(
        n_bottleneck_layer=n_bottleneck_layer,
        n_filters_init=n_filters_init,
        batch_norm=True,
    )
    print("Nr of params:", count_parameters(model))
    out = model(x)
    assert out.shape == (batch_size, n_bottleneck_layer)
