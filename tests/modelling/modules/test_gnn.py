import torch
from torch import nn

from deep_bac.modelling.modules.gnn import GNNModel, get_edge_data
from deep_bac.utils import DRUG_SPECIFIC_GENES_DICT


def test_gnn_encoder_forward():
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


def test_gnn_model():
    selected_gens = DRUG_SPECIFIC_GENES_DICT["cryptic"]
    gene_to_idx = {gene: idx for idx, gene in enumerate(selected_gens)}
    batch_size = 16
    n_genes = len(selected_gens)
    gene_dim = 64
    n_heads = 2
    n_output = 4

    x = torch.rand(batch_size, n_genes, gene_dim)
    edge_indices, edge_features = get_edge_data(
        edge_file_path="/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/ppi_interactions_string.tsv",
        gene_to_idx=gene_to_idx,
    )
    decoder = nn.Linear(gene_dim, n_output)
    labels = torch.ones(batch_size, n_output)

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

    logits = decoder(out)
    assert logits.shape == (batch_size, n_output)

    loss = nn.BCEWithLogitsLoss()(logits, labels)
    assert loss > 0.0
