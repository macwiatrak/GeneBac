import torch

from genebac.modelling.modules.graph_transformer import GraphTransformer


def test_graph_transformer():
    batch_size = 2
    n_genes = 3
    dim = 128
    n_output = 5

    x = torch.rand(batch_size, n_genes, dim)
    model = GraphTransformer(
        dim=dim,
        n_output=n_output,
        n_layers=4,
        n_heads=4,
    )
    out = model(x)
    assert out.shape == (batch_size, n_output)
