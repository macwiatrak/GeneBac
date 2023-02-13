import math

import torch
from torch import nn


class IdentityPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, tss_indexes=None):
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim: int, n_genes: int, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(n_genes, dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, tss_indexes: torch.Tensor):
        x = x + self.pe[tss_indexes]
        return self.dropout(x)


class FixedPositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 3978):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        # TODO: simplify it
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(
        self, x: torch.Tensor, tss_indexes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # TODO: make it [batch_size, seq_len, embedding_dim]
        x = x + self.pe[tss_indexes]
        return self.dropout(x)


class FixedGeneExpressionPositionalEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        start_coef: float = 3.65,
        scale: float = 10e-5,
        tss_div: int = 1000,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.start_coef = start_coef
        self.scale = scale
        self.tss_div = tss_div

        self.min_range = math.log(start_coef) / math.log(2.0)
        coef_linspace = (
            -torch.linspace(-start_coef, -self.min_range, dim) * scale / 1000
        )
        self.register_buffer("pe", coef_linspace)

    def forward(
        self, x: torch.Tensor, tss_indexes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embedding_dim]
        """
        # convert it to KBs
        scaled_tss_indexes = tss_indexes / self.tss_div
        vals = scaled_tss_indexes.abs().unsqueeze(
            1
        ) * self.coef_linspace.unsqueeze(0)
        pe = torch.ones_like(vals) - vals
        x = x + pe
        return x
