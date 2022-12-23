import torch
from torch import nn


class StochasticShift(nn.Module):
    """Stochastically shift a one hot encoded DNA sequence."""

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
            size=seq_1hot.shape[0],  # first dim is the batch dim
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
    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if len(seq.shape) != 3:
        raise ValueError("input sequence should be rank 3")

    if shift == 0:
        return seq

    pad = pad * torch.ones_like(seq[:, 0: shift.abs(), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return torch.cat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return torch.cat([sliced_seq, pad], axis=1)

    if shift > 0:
        return _shift_right(seq)
    return _shift_left(seq)

