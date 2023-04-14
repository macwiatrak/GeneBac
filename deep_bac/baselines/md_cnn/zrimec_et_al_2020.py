import torch
from torch import nn

from deep_bac.modelling.modules.layers import ConvLayer
from deep_bac.modelling.modules.utils import Flatten


class ZrimecEtAlModel(nn.Module):
    """Implemantation of Zrimet et al. (2020)"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        self.conv_layers = nn.Sequential(
            *[
                ConvLayer(
                    in_channels=4,
                    out_channels=128,
                    kernel_size=40,
                    batch_norm=True,
                    activation_fn=nn.ReLU(),
                    dropout=dropout,
                    pool_size=2,
                ),
                ConvLayer(
                    in_channels=128,
                    out_channels=32,
                    kernel_size=30,
                    batch_norm=True,
                    activation_fn=nn.ReLU(),
                    dropout=dropout,
                ),
                ConvLayer(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=30,
                    batch_norm=True,
                    activation_fn=nn.ReLU(),
                    dropout=dropout,
                    dilation=4,
                ),
            ]
        )

        self.dense_layers = nn.Sequential(
            *[
                Flatten(),
                nn.Linear(76480, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x
