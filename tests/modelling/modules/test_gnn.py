import torch

from deep_bac.modelling.modules.gnn import GNNModel


def test_gnn_encoder_steps():
    batch_size = 2
    n_genes = 4
    gene_dim = 64
    n_heads = 2
    n_edges = 10
    edge_features_dim = 6

    x = torch.rand(batch_size, n_genes, gene_dim)
    edge_indices = torch.randint(0, n_genes, (2, n_edges))
    edge_features = torch.randn(n_edges, edge_features_dim)

    model = GNNModel(
        input_dim=gene_dim,
        hidden_dim=gene_dim,
        output_dim=gene_dim,
        n_layers=2,
        layer_type="GAT",
        edge_indices=edge_indices,
        edge_features=edge_features,
        n_heads=n_heads,
    )

    out = model(x)
    assert out.shape == (batch_size, gene_dim)
