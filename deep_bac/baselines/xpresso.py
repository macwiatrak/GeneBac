import torch
from torch import nn

from deep_bac.modelling.modules.utils import Flatten


class Xpresso(nn.Module):
    """Implemantation of Agarwal and Shendure (2020)"""

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            *[
                nn.Conv1d(
                    in_channels=4,
                    out_channels=128,
                    kernel_size=6,
                ),
                nn.MaxPool1d(30),
                nn.Conv1d(
                    in_channels=128,
                    out_channels=32,
                    kernel_size=9,
                ),
                nn.MaxPool1d(10),
                Flatten(),
                nn.Linear(224, 64),
                nn.ReLU(),
                nn.Dropout(0.00099),
                nn.Linear(64, 2),
                nn.ReLU(),
                nn.Dropout(0.01546),
                nn.Linear(2, 1),
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
