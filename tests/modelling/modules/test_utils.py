import torch

from deep_bac.modelling.modules.utils import StochasticShift, shift_seq


def test_stochastic_shift():
    batch_size = 8
    seq_length = 128
    seq = torch.ones(8, 4, seq_length)
    shift = StochasticShift(
        shift_max=3,
        pad=0.,
    )
    out = shift(seq, training=True)
    assert out.shape == (batch_size, 4, seq_length)


def test_shift_seq():
    seq_length = 128
    seq = torch.ones(4, seq_length)
    shift = torch.tensor(3)
    out = shift_seq(seq, shift, pad=0.)
    assert out.shape == (4, seq_length)
    assert out[:, :shift].sum() == 0.

    seq_length = 128
    seq = torch.ones(4, seq_length)
    shift = torch.tensor(2)
    out = shift_seq(seq, shift, pad=0.)
    assert out.shape == (4, seq_length)
    assert out[:, :shift].sum() == 0.

    seq_length = 128
    seq = torch.ones(4, seq_length)
    shift = torch.tensor(0)
    out = shift_seq(seq, shift, pad=0.)
    assert out.shape == (4, seq_length)
    assert out[:, :shift].sum() == 0.

    seq_length = 128
    seq = torch.ones(4, seq_length)
    shift = torch.tensor(-3)
    out = shift_seq(seq, shift, pad=0.)
    assert out.shape == (4, seq_length)
    assert out[:, shift:].sum() == 0.
