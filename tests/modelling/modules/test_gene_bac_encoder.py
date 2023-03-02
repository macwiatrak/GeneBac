import torch

from deep_bac.modelling.modules.gene_bac_encoder import GeneBacEncoder
from deep_bac.modelling.modules.utils import count_parameters


def test_gene_bac_encoder():
    batch_size = 1
    seq_length = 2048
    in_channels = 4
    n_filters_init = 256
    n_repeat_blocks_tower = 5
    filters_mult = 1.122
    n_filters_pre_bottleneck = 227
    n_bottleneck_layer = 64

    x = torch.rand(batch_size, in_channels, seq_length)
    model = GeneBacEncoder(
        input_dim=in_channels,
        n_filters_init=n_filters_init,
        n_repeat_blocks_tower=n_repeat_blocks_tower,
        filters_mult=filters_mult,
        n_filters_pre_bottleneck=n_filters_pre_bottleneck,
        n_bottleneck_layer=n_bottleneck_layer,
        batch_norm=True,
    )
    print("Nr of params:", count_parameters(model))
    out = model(x)
    assert out.shape == (batch_size, n_bottleneck_layer)
