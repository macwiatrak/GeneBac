import torch
from torch import nn

from deep_bac.modelling.modules.layers import (
    Residual,
    AttentionPool,
    ConvLayer,
    exponential_linspace_int,
    DenseLayer,
)


class EnformerLikeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        seq_length: int = 2048,
        n_filters_init: int = 256,
        n_repeat_blocks_tower: int = 6,
        n_bottleneck_layer: int = 64,
        batch_norm: bool = True,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            ConvLayer(
                in_channels=input_dim,
                out_channels=n_filters_init,
                kernel_size=15,
                batch_norm=batch_norm,
            ),
            Residual(
                ConvLayer(
                    in_channels=n_filters_init,
                    out_channels=n_filters_init,
                    kernel_size=1,
                    batch_norm=batch_norm,
                )
            ),
            AttentionPool(n_filters_init, pool_size=3),
        )

        filter_list = exponential_linspace_int(
            start=n_filters_init, end=seq_length // 3, num=n_repeat_blocks_tower
        )

        conv_layers = []
        for in_channels, out_channels in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                nn.Sequential(
                    ConvLayer(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=5,
                        batch_norm=batch_norm,
                    ),
                    Residual(
                        ConvLayer(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            batch_norm=batch_norm,
                        )
                    ),
                    AttentionPool(out_channels, pool_size=2),
                )
            )

        self.conv_tower = nn.Sequential(*conv_layers)

        self.pre_bottleneck = nn.Sequential(
            ConvLayer(
                in_channels=out_channels,
                out_channels=out_channels // 2,
                kernel_size=5,
                batch_norm=batch_norm,
            ),
            Residual(
                ConvLayer(
                    in_channels=out_channels // 2,
                    out_channels=out_channels // 2,
                    kernel_size=1,
                    batch_norm=batch_norm,
                )
            ),
            AttentionPool(out_channels // 2, pool_size=2),
        )

        seq_depth = 16
        self.bottleneck = DenseLayer(
            in_features=(out_channels // 2) * seq_depth,
            out_features=n_bottleneck_layer,
            use_bias=True,
            batch_norm=False,
            dropout=0.0,
            activation_fn=nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.conv_tower(x)
        x = self.pre_bottleneck(x)
        # flatten the input
        x = x.view(x.shape[0], -1)
        x = self.bottleneck(x)
        return x
