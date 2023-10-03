import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from genebac.modelling.modules.layers import ConvLayer, DenseLayer


class ConvTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_bottleneck_layer: int = 128,
        n_filters: int = 256,
        kernel_size: int = 24,
        pool_size: int = 8,
        batch_norm: bool = True,
        dropout: float = 0.0,
        n_transformer_layers: int = 1,
        n_transformer_heads: int = 8,
    ):
        super().__init__()
        self.n_botlleneck_layer = n_bottleneck_layer

        self.conv_layer = ConvLayer(
            in_channels=4,
            out_channels=n_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        self.rearrange_fn = Rearrange("b n d -> b d n")
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=n_filters,
            nhead=n_transformer_heads,
            dim_feedforward=n_filters * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=n_transformer_layers
        )
        self.dense_layer = DenseLayer(
            in_features=2 * n_filters,
            out_features=n_bottleneck_layer,
            dropout=0.2,
            activation_fn=nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        # pass through layers
        x = self.conv_layer(x)
        x = self.rearrange_fn(x)
        x = self.transformer(x)
        x = F.dropout(x)
        # concatenate the max and mean pooling
        x = torch.cat([x.max(dim=1)[0], x.mean(dim=1)], dim=1)
        x = self.dense_layer(x)
        return x
