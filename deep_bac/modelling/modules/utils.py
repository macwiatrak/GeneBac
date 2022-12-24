from typing import Tuple

import torch
from torch import nn


class StochasticShift(nn.Module):
    """
    Stochastically shift a one hot encoded DNA sequence.
    adapted from https://github.com/calico/scBasset/blob/main/scbasset/basenji_utils.py
    """

    def __init__(self, shift_max: int = 3, pad: float = 0.):
        super().__init__()
        self.shift_max = shift_max
        self.pad = pad

    def forward(
            self,
            seq_1hot: torch.Tensor,
            training: bool = False
    ):
        if not training:
            return seq_1hot
        shifts = torch.randint(
            low=-self.shift_max,
            high=self.shift_max + 1,
            size=(seq_1hot.shape[0], ),  # first dim is the batch dim
        )
        return torch.stack([
            shift_seq(seq, shift, pad=self.pad) for seq, shift in zip(seq_1hot, shifts)
        ])


def shift_seq(
        seq: torch.Tensor,
        shift: torch.tensor,
        pad: float = 0.
):
    """Shift a sequence left or right by shift_amount.
    adapted from https://github.com/calico/scBasset/blob/main/scbasset/basenji_utils.py
    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (torch.tensor)
    pad: value to fill the padding (float)
    """

    # if no shift return the sequence
    if shift == 0:
        return seq

    # create the padding
    pad = pad * torch.ones_like((seq[:, :shift.abs()]))

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift]
        # cat to the left along the sequence axis
        return torch.cat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:]
        # cat to the right along the sequence axis
        return torch.cat([sliced_seq, pad], axis=1)

    if shift > 0:  # if shift is positive shift_right
        return _shift_right(seq)
    # if shift is negative shift_left
    return _shift_left(seq)


class StochasticReverseComplement(nn.Module):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self, prob: float = 0.5):
        super(StochasticReverseComplement, self).__init__()
        self.prob = prob

    def forward(self, seq: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, torch.tensor]:
        if not training:
            return seq, torch.zeros(seq.shape[0], dtype=torch.bool)
        batch_size = seq.shape[0]
        probs = torch.empty(batch_size).uniform_(0, 1)
        rc_seq = torch.flip(seq, dims=[1, 2])
        is_rc = probs < self.prob
        return torch.where(is_rc, rc_seq, seq), is_rc
