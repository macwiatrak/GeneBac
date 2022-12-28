import torch

from deep_bac.modelling.modules.enformer_like_encoder import EnformerLikeEncoder


def test_enformer_like_encoder():
    batch_size = 1
    seq_length = 2048
    in_channels = 4
    num_downsamples = 3
    num_divisible_by = 128
    n_bottleneck_layer = 128

    x = torch.rand(batch_size, in_channels, seq_length)
    model = EnformerLikeEncoder(
        input_dim=seq_length,
        num_downsamples=num_downsamples,
        num_divisible_by=num_divisible_by,
    )
    out = model(x)
    assert out.shape == (batch_size, n_bottleneck_layer)
