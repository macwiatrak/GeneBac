import torch
import torch.nn as nn

from deep_bac.modelling.modules.layers import ConvLayer, DenseLayer


class scBassetEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 4,
            n_filters_init: int = 288,
            n_repeat_blocks_tower: int = 6,
            filters_mult: float = 1.122,
            n_filters_pre_bottleneck: int = 256,
            n_bottleneck_layer: int = 32,
            batch_norm: bool = True,
            dropout: float = 0.2,
    ):
        super().__init__()

        self.stem = ConvLayer(
            in_channels=input_dim,
            out_channels=n_filters_init,
            kernel_size=17,
            pool_size=3,
        )

        tower_layers = []
        curr_n_filters = n_filters_init
        for i in range(n_repeat_blocks_tower):
            tower_layers.append(
                ConvLayer(
                    in_channels=curr_n_filters,
                    out_channels=int(curr_n_filters * filters_mult),
                    kernel_size=5,
                    pool_size=2,
                )
            )
            curr_n_filters = int(curr_n_filters*filters_mult)
        self.tower = nn.Sequential(*tower_layers)

        self.pre_bottleneck = ConvLayer(
            in_channels=curr_n_filters,
            out_channels=n_filters_pre_bottleneck,
            kernel_size=1,
        )
        self.bottleneck = DenseLayer(
            in_features=n_filters_pre_bottleneck,
            out_features=n_bottleneck_layer,
            use_bias=True,
            batch_norm=False,
            dropout=0.0,
            activation_fn=nn.Identity(),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.pre_bottleneck(x)
        x = self.bottleneck(x)
        return x
