import torch

from baselines.md_cnn.md_cnn import MDCNN


def test_md_cnn():
    batch_size = 2
    seq_length = 4051
    in_channels = 4
    n_output = 14
    n_genes = 4

    x = torch.rand(batch_size, n_genes, in_channels, seq_length)
    batch_size, n_genes, n_channels, seq_length = x.shape
    # reshape the input to allow the convolutional layer to work
    x = x.view(batch_size, n_channels, n_genes * seq_length)

    model = MDCNN(seq_length=n_genes * seq_length, n_output=n_output)
    out = model(x)
    assert out.shape == (batch_size, n_output)
