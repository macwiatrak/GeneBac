import torch
import torch.nn as nn

from deep_bac.modelling.modules.layers import (
    ConvLayer,
    DenseLayer,
    _round,
)


class GeneBacEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        n_filters_init: int = 128,
        n_repeat_blocks_tower: int = 6,
        filters_mult: float = 1.122,
        n_bottleneck_layer: int = 64,
        batch_norm: bool = True,
    ):
        super().__init__()

        self.stem = ConvLayer(
            in_channels=input_dim,
            out_channels=n_filters_init,
            kernel_size=17,
            pool_size=3,
            batch_norm=batch_norm,
        )

        tower_layers = []
        curr_n_filters = n_filters_init
        for i in range(n_repeat_blocks_tower):
            tower_layers.append(
                ConvLayer(
                    in_channels=curr_n_filters,
                    out_channels=_round(curr_n_filters * filters_mult),
                    kernel_size=6,
                    pool_size=2,
                    batch_norm=batch_norm,
                )
            )
            curr_n_filters = _round(curr_n_filters * filters_mult)
        self.tower = nn.Sequential(*tower_layers)

        seq_depth = 16
        self.bottleneck = DenseLayer(
            in_features=curr_n_filters * seq_depth,
            out_features=n_bottleneck_layer,
            use_bias=True,
            batch_norm=False,
            dropout=0.0,  # we apply dropout in the main model
            activation_fn=nn.Identity(),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.tower(x)
        # flatten the input
        x = x.view(x.shape[0], -1)
        x = self.bottleneck(x)
        return x
