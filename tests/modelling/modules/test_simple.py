import torch

from deep_bac.modelling.modules.simple_conv import SimpleConvEncoder


def test_simple_encoder():
    batch_size = 1
    seq_length = 1344
    in_channels = 4
    n_filters_init = 256
    n_filters_pre_bottleneck = 64
    n_bottleneck_layer = 128

    x = torch.rand(batch_size, in_channels, seq_length)
    model = SimpleConvEncoder(
        input_dim=in_channels,
        n_filters_init=n_filters_init,
        n_filters_pre_bottleneck=n_filters_pre_bottleneck,
        n_bottleneck_layer=n_bottleneck_layer,
        batch_norm=True,
    )
    out = model(x)
    assert out.shape == (batch_size, n_bottleneck_layer)