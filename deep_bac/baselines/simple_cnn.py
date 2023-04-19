import torch
from torch import nn

from deep_bac.modelling.modules.layers import ConvLayer
from deep_bac.modelling.modules.utils import Flatten


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            *[
                ConvLayer(
                    in_channels=4,
                    out_channels=128,
                    batch_norm=True,
                    pool_size=3,
                    kernel_size=17,
                ),
                ConvLayer(
                    in_channels=128,
                    out_channels=64,
                    batch_norm=True,
                    pool_size=3,
                    kernel_size=6,
                ),
                ConvLayer(
                    in_channels=64,
                    out_channels=32,
                    batch_norm=True,
                    pool_size=3,
                    kernel_size=6,
                    dropout=0.2,
                ),
            ]
        )

        self.dense_layers = nn.Sequential(
            *[
                Flatten(),
                nn.Linear(95 * 32, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x
