import torch
from torch import nn

from deep_bac.modelling.modules.layers import DenseLayer, ConvLayer


class SimpleConvEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 4,
            n_filters_init: int = 256,
            n_filters_pre_bottleneck: int = 64,
            n_bottleneck_layer: int = 128,
            batch_norm: bool = True,
            dropout: float = 0.2,
    ):
        super().__init__()

        seq_depth = 29
        self.stem = ConvLayer(
            in_channels=input_dim,
            out_channels=n_filters_init,
            kernel_size=17,
            pool_size=3,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.pre_bottleneck = ConvLayer(
            in_channels=n_filters_init,
            out_channels=n_filters_pre_bottleneck,
            kernel_size=17,
            pool_size=15,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.bottleneck = DenseLayer(
            in_features=n_filters_pre_bottleneck * seq_depth,
            out_features=n_bottleneck_layer,
            use_bias=True,
            batch_norm=False,
            dropout=0.0,
            activation_fn=nn.Identity(),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.pre_bottleneck(x)
        # flatten the input
        x = x.view(x.shape[0], -1)
        x = self.bottleneck(x)
        return x
