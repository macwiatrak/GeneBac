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


class SinCosFixedPositionalEncoding(nn.Module):
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


class ExpGammaFixedPositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 3978):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        positions = torch.arange(max_len).unsqueeze(1)
        exp_pos_features = get_positional_features_exponential(
            positions=positions,
            seq_len=max_len,
            dim=dim // 2,
        )
        gamma_pos_features = get_positional_features_gamma(
            positions=positions,
            dim=dim // 2,
            seq_len=max_len,
        )
        pe = torch.cat([exp_pos_features, gamma_pos_features], dim=-1)
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


# relative positional encoding functions
def get_positional_features_exponential(
    positions: torch.Tensor, dim: int, seq_len: int, min_half_life: float = 3.0
):
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(min_half_life, max_range, dim).unsqueeze(0)
    return torch.exp(-math.log(2.0) / half_life * positions.abs())


def gamma_pdf(x: torch.Tensor, concentration: torch.Tensor, rate: torch.Tensor):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(
        rate
    )
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(
    positions: torch.Tensor,
    dim: int,
    seq_len: int,
    stddev=None,
    start_mean=None,
    eps=1e-8,
):
    if stddev is None:
        stddev = seq_len / (2 * dim)

    if start_mean is None:
        start_mean = seq_len / dim

    mean = torch.linspace(
        start_mean, seq_len, dim, device=positions.device
    ).unsqueeze(0)
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = gamma_pdf(positions.float().abs(), concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs
