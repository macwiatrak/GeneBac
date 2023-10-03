from typing import Tuple

import torch
from torch import nn

from genebac.data_preprocessing.utils import shift_seq


class StochasticShift(nn.Module):
    """
    Stochastically shift a one hot encoded DNA sequence.
    adapted from https://github.com/calico/scBasset/blob/main/scbasset/basenji_utils.py
    """

    def __init__(self, shift_max: int = 3, pad: float = 0.0):
        super().__init__()
        self.shift_max = shift_max
        self.pad = pad

    def forward(self, seq_1hot: torch.Tensor, training: bool = False):
        if not training:
            return seq_1hot
        shifts = torch.randint(
            low=-self.shift_max,
            high=self.shift_max + 1,
            size=(seq_1hot.shape[0],),  # first dim is the batch dim
        )
        return torch.stack(
            [
                shift_seq(seq, shift, pad=self.pad)
                for seq, shift in zip(seq_1hot, shifts)
            ]
        )


class StochasticReverseComplement(nn.Module):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self, prob: float = 0.5):
        super(StochasticReverseComplement, self).__init__()
        self.prob = prob

    def forward(
        self, seq: torch.Tensor, training: bool = False
    ) -> Tuple[torch.Tensor, torch.tensor]:
        if not training:
            return seq, torch.zeros(seq.shape[0], dtype=torch.bool)
        batch_size = seq.shape[0]
        probs = torch.empty(batch_size).uniform_(0, 1)
        rc_seq = torch.flip(seq, dims=[1, 2])
        is_rc = probs < self.prob
        return torch.where(is_rc, rc_seq, seq), is_rc


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
