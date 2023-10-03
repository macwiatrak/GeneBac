import torch

from genebac.modelling.modules.layers import ConvLayer, DenseLayer


def test_conv_layer():
    batch_size = 8
    seq_length = 128
    in_channels = 4
    out_channels = 18
    kernel_size = 17
    pool_size = 3

    x = torch.rand(batch_size, in_channels, seq_length)
    conv_layer = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        pool_size=pool_size,
        batch_norm=True,
    )
    out = conv_layer(x)
    assert out.shape == (batch_size, out_channels, seq_length // pool_size)


def test_dense_layer():
    batch_size = 8
    in_features = 128
    out_features = 18

    x = torch.rand(batch_size, in_features)
    dense_layer = DenseLayer(
        in_features=in_features,
        out_features=out_features,
        batch_norm=True,
    )
    out = dense_layer(x)
    assert out.shape == (batch_size, out_features)
