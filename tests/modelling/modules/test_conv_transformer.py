import torch

from deep_bac.modelling.modules.conv_transformer import ConvTransformerEncoder


def test_conv_transformer():
    batch_size = 2
    n_genes = 3
    seq_length = 2048
    in_channels = 4
    n_filters = 256
    n_bottleneck_layer = 128

    x = torch.rand(batch_size, n_genes, in_channels, seq_length)
    model = ConvTransformerEncoder(
        n_bottleneck_layer=n_bottleneck_layer,
        batch_norm=True,
        dropout=0.0,
        n_filters=n_filters,
        n_transformer_layers=1,
    )
    out = model(x)
    assert out.shape == (batch_size, n_genes, n_bottleneck_layer)