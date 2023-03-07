import torch
from torch import nn

from deep_bac.modelling.modules.layers import ConvLayer, DenseLayer, _round
from deep_bac.modelling.modules.utils import Flatten


class MDCNN(nn.Module):
    def __init__(self, seq_length: int, input_dim: int = 4, n_output: int = 14):
        super().__init__()
        self.conv_tower = nn.Sequential(
            *[
                ConvLayer(
                    in_channels=input_dim,
                    out_channels=64,
                    kernel_size=12,
                    activation_fn=nn.ReLU(),
                    batch_norm=False,
                    pool_size=None,
                ),
                ConvLayer(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=12,
                    activation_fn=nn.ReLU(),
                    pool_size=3,
                    batch_norm=False,
                ),
                ConvLayer(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=3,
                    activation_fn=nn.ReLU(),
                    pool_size=None,
                    batch_norm=False,
                ),
                ConvLayer(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    activation_fn=nn.ReLU(),
                    pool_size=3,
                    batch_norm=False,
                ),
            ]
        )

        self.dense_layers = nn.Sequential(
            *[
                Flatten(),
                DenseLayer(
                    in_features=int(int(seq_length / 3) / 3) * 32,
                    out_features=256,
                    activation_fn=nn.ReLU(),
                    dropout=0.0,
                    batch_norm=False,
                ),
                DenseLayer(
                    in_features=256,
                    out_features=256,
                    activation_fn=nn.ReLU(),
                    dropout=0.0,
                    batch_norm=False,
                ),
                nn.Linear(in_features=256, out_features=n_output),
            ]
        )
        # max gene length of the MD-CNN genes is 4051

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_tower(x)
        return self.dense_layers(x)
