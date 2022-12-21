import torch

from deep_bac.modelling.modules.scBasset_encoder import scBassetEncoder


def test_scBasset_encoder():
    batch_size = 1
    seq_length = 1344
    in_channels = 4
    n_filters_init = 288
    n_repeat_blocks_tower = 5
    filters_mult = 1.122
    n_filters_pre_bottleneck = 256
    n_bottleneck_layer = 32

    x = torch.rand(batch_size, in_channels, seq_length)
    scBasset_encoder = scBassetEncoder(
        input_dim=in_channels,
        n_filters_init=n_filters_init,
        n_repeat_blocks_tower=n_repeat_blocks_tower,
        filters_mult=filters_mult,
        n_filters_pre_bottleneck=n_filters_pre_bottleneck,
        n_bottleneck_layer=n_bottleneck_layer,
        batch_norm=True,
    )
    out = scBasset_encoder(x)
    assert out.shape == (batch_size, n_bottleneck_layer)