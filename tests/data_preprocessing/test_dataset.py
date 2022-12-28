import torch

from deep_bac.data_preprocessing.dataset import seq_to_one_hot, pad_one_hot_seq


def test_seq_to_one_hot():
    seq = "ATCGATG"
    one_hot = seq_to_one_hot(seq)
    assert one_hot.shape == (len(seq), 4)
    assert one_hot.sum() == len(seq)


def test_pad_one_hot_seq():
    seq = "ATCGATG"
    max_length = 10
    to_pad = max_length - len(seq)
    pad_value = 0.25

    one_hot = seq_to_one_hot(seq)
    one_hot_padded = pad_one_hot_seq(one_hot, max_length, pad_value)
    assert one_hot_padded.shape == (max_length, 4)
    assert one_hot_padded.sum() == max_length
    assert torch.all(
        one_hot_padded[len(seq):].eq(torch.full((to_pad, 4), pad_value)))
