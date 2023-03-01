import math
from typing import Callable

import numpy as np
import torch

from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = None,
        batch_norm: bool = True,
        dropout: float = 0.0,
        activation_fn: Callable = nn.GELU(),
        attention_pooling: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.batch_norm = (
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        )
        if pool_size:
            self.pool = (
                AttentionPool(out_channels, pool_size)
                if attention_pooling
                else nn.MaxPool1d(pool_size)
            )
        else:
            self.pool = nn.Identity()

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        layer_norm: bool = True,
        batch_norm: bool = False,
        dropout: float = 0.1,
        activation_fn: Callable = nn.ReLU(),
    ):
        super().__init__()
        if layer_norm and batch_norm:
            batch_norm = False
            raise Warning(
                "LayerNorm and BatchNorm both used in the dense layer, "
                "defaulting to LayerNorm only"
            )
        self.dense = nn.Linear(in_features, out_features, bias=use_bias)
        self.layer_norm = (
            nn.LayerNorm(out_features, elementwise_affine=False)
            if layer_norm
            else nn.Identity()
        )
        self.batch_norm = (
            nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        )
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.batch_norm(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x


def _round(x):
    return int(np.round(x))


def make_conv_tower(
    n_repeat_blocks_tower: int,
    in_channels: int,
    filters_mult: float,
    kernel_size: int,
    pool_size: int,
    dropout: float,
    batch_norm: bool,
) -> nn.Sequential:
    tower_layers = []
    curr_n_filters = in_channels
    for i in range(n_repeat_blocks_tower):
        tower_layers.append(
            ConvLayer(
                in_channels=_round(curr_n_filters),
                out_channels=_round(curr_n_filters * filters_mult),
                kernel_size=kernel_size,
                pool_size=pool_size,
                dropout=dropout,
                batch_norm=batch_norm,
            )
        )
        curr_n_filters *= filters_mult
    return nn.Sequential(*tower_layers)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class AttentionPool(nn.Module):
    def __init__(self, dim: int, pool_size: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)


def exponential_linspace_int(
    start: int, end: int, num: int, divisible_by: int = 1
):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]
