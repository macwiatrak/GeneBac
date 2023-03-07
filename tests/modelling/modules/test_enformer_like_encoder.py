import torch

from deep_bac.modelling.modules.enformer_like_encoder import EnformerLikeEncoder
from deep_bac.modelling.modules.utils import count_parameters


def test_enformer_like_encoder():
    batch_size = 2
    seq_length = 2560
    in_channels = 4
    n_bottleneck_layer = 64

    x = torch.rand(batch_size, in_channels, seq_length)
    model = EnformerLikeEncoder(
        input_dim=in_channels,
        n_filters_init=256,
        n_repeat_blocks_tower=6,
    )
    out = model(x)
    print("Nr of params:", count_parameters(model))
    assert out.shape == (batch_size, n_bottleneck_layer)
