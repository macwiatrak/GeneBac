from typing import Literal, List, Tuple, Dict

import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATv2Conv, MessagePassing

from deep_bac.modelling.modules.layers import DenseLayer
from deep_bac.modelling.modules.utils import Flatten

STRINGDB_EDGE_FEATURES = [
    "neighborhood_on_chromosome",
    "gene_fusion",
    "phylogenetic_cooccurrence",
    "homology",
    "coexpression",
    "experimentally_determined_interaction",
    "database_annotated",
    "automated_textmining",
    "combined_score",
]


def batch_edge_index(
    edge_index: torch.Tensor, n_batches: int, n_nodes: int
) -> torch.Tensor:
    """Add batch indices to edge_index

    Args:
        edge_index (torch.Tensor): Edge indices
        n_batches (int): Number of batches
        n_nodes (int): Number of nodes
    Returns:
        torch.Tensor: Edge indices with batch indices
    """
    edge_indices = [
        edge_index + batch_idx * n_nodes for batch_idx in range(n_batches)
    ]
    return torch.cat(edge_indices, dim=1)


def get_edge_data(
    edge_file_path: str,
    gene_to_idx: Dict[str, int],
    edge_feature_list: List[str] = STRINGDB_EDGE_FEATURES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_df = pd.read_csv(edge_file_path, sep="\t").reset_index()
    edge_tensor = torch.tensor(
        [
            [gene_to_idx[gene] for gene in edge_df["node1"].tolist()],
            [gene_to_idx[gene] for gene in edge_df["node2"].tolist()],
        ],
        dtype=torch.long,
    )

    edge_features = torch.tensor(
        edge_df[edge_feature_list].values, dtype=torch.float32
    )
    return edge_tensor, edge_features


class GNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_genes: int,
        n_layers: int = 2,
        n_heads: int = 2,
        layer_type: Literal["GCN", "GAT"] = "GAT",
        edge_indices: torch.Tensor = None,
        edge_features: torch.Tensor = None,
        dropout_rate: float = 0.2,
        activation_fn: nn.Module = nn.ReLU(inplace=False),
    ):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dropout_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        self.layer_type = layer_type
        self.same_edge_indices = edge_indices
        self.same_edge_features = edge_features

        gnn_layer = GCNConv if layer_type == "GCN" else GATv2Conv
        edge_dim = edge_features.shape[1] if edge_features is not None else None
        kwargs = {"heads": n_heads, "edge_dim": edge_dim}

        layers = []
        in_channels, out_channels = input_dim, hidden_dim
        for l_idx in range(n_layers - 1):
            layers += [
                gnn_layer(
                    in_channels=in_channels, out_channels=out_channels, **kwargs
                ),
                nn.ReLU(inplace=False),
                nn.Dropout(dropout_rate),
            ]
            in_channels = hidden_dim

        kwargs["heads"] = 1
        layers += [
            gnn_layer(
                in_channels=in_channels, out_channels=output_dim, **kwargs
            )
        ]
        self.layers = nn.ModuleList(layers)
        self.dense = nn.Sequential(
            Flatten(),
            DenseLayer(
                in_features=output_dim * n_genes,
                out_features=output_dim,
                layer_norm=False,
                batch_norm=False,
                activation_fn=nn.ReLU(inplace=False),
            ),
        )

        if layer_type == "GCN" and self.same_edge_features is not None:
            # select combined edge score as edge weight for the GCN model
            self.same_edge_features = self.same_edge_features[:, -1]

        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = activation_fn

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_features: torch.Tensor = None,
    ):
        """
        Inputs:
            node_features - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        edge_index = (
            edge_index if edge_index is not None else self.same_edge_indices
        )
        edge_features = (
            edge_features
            if edge_features is not None
            else self.same_edge_features
        )

        bs, n_nodes, dim = node_features.shape
        node_features = node_features.view(bs * n_nodes, dim)
        edge_index = batch_edge_index(edge_index, bs, n_nodes).to(
            node_features.device
        )

        if len(edge_features.shape) == 1:
            edge_features = edge_features.repeat(bs).to(node_features.device)
        else:
            edge_features = edge_features.repeat(bs, 1).to(node_features.device)

        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, MessagePassing):
                x = l(node_features, edge_index, edge_features)
            else:
                x = l(node_features)
        x = x.view(bs, n_nodes, -1)
        x = self.dense(self.dropout(self.activation_fn(x)))
        return x
