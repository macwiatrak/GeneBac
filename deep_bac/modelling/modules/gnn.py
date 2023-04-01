from typing import Literal

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATv2Conv


def batch_edge_index(edge_index: torch.Tensor, n_batches: int) -> torch.Tensor:
    """Add batch indices to edge_index

    Args:
        edge_index (torch.Tensor): Edge indices
        n_batches (int): Number of batches

    Returns:
        torch.Tensor: Edge indices with batch indices
    """
    batch = torch.arange(n_batches).repeat_interleave(edge_index.size(1))
    return torch.cat([batch.unsqueeze(0), edge_index], dim=0)


class GNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        layer_type: Literal["GCN", "GAT"] = "GAT",
        edge_indices: torch.Tensor = None,
        edge_features: torch.Tensor = None,
        dropout_rate: float = 0.2,
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
        kwargs = {"heads": 2} if layer_type == "GAT" else {}

        layers = []
        in_channels, out_channels = input_dim, hidden_dim
        for l_idx in range(n_layers - 1):
            layers += [
                gnn_layer(
                    in_channels=in_channels, out_channels=out_channels, **kwargs
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            in_channels = hidden_dim
        layers += [
            gnn_layer(
                in_channels=in_channels, out_channels=output_dim, **kwargs
            )
        ]
        self.layers = nn.Sequential(*layers)

        if layer_type == "GCN" and self.same_edge_features is not None:
            # select combined edge score as edge weight for the GCN model
            self.same_edge_features = self.same_edge_features[:, -1]

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
        # TODO: flatten it properly
        node_features = node_features.view(bs * n_nodes, dim)
        # TODO: batch edge indexes
        edge_index = batch_edge_index(edge_index, bs)
        # TODO: batch edge features
        edge_features = edge_features.repeat(bs, 1, 1)

        x = self.layers(node_features, edge_index, edge_features)
        # TODO: check this works
        x = x.view(bs, n_nodes, -1)
        # TODO: check this works
        x = x.mean(dim=1)
        return x
