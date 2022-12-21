from typing import Callable

import torch
from torch import nn


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            pool_size: int = None,
            batch_norm: bool = True,
            dropout: float = 0.2,
            activation_fn: Callable = nn.GELU()
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        self.pool = nn.MaxPool1d(pool_size) if pool_size is not None else nn.Identity()
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.activation_fn(x)
        return x


class DenseLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_bias: bool = True,
            batch_norm: bool = True,
            dropout: float = 0.2,
            activation_fn: Callable = nn.GELU()
    ):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features, bias=use_bias)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dense(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation_fn(x)
        return x
