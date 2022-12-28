import torch
from torch import nn

from deep_bac.modelling.modules.layers import Residual, AttentionPool, ConvLayer, exponential_linspace_int


class EnformerLikeEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            input_channels: int = 4,
            num_downsamples: int = 3,
            batch_norm: bool = True,
            num_divisible_by: int = 128,
    ):
        super().__init__()

        quarter_dim = input_dim // 4
        self.stem = nn.Sequential(
            ConvLayer(
                in_channels=input_channels,
                out_channels=quarter_dim,
                kernel_size=17,
                batch_norm=batch_norm,
                dropout=0.
            ),
            Residual(ConvLayer(
                in_channels=quarter_dim,
                out_channels=quarter_dim,
                kernel_size=1,
                batch_norm=batch_norm,
            )),
            AttentionPool(quarter_dim, pool_size=2),
        )

        filter_list = exponential_linspace_int(
            start=quarter_dim,
            end=input_dim, num=num_downsamples - 1, divisible_by=num_divisible_by)
        filter_list = [quarter_dim, *filter_list]

        conv_layers = []
        for in_channels, out_channels in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(nn.Sequential(
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    batch_norm=batch_norm,
                ),
                Residual(ConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    batch_norm=batch_norm,
                )),
                AttentionPool(out_channels, pool_size=2),
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.tower(x)
        return x
