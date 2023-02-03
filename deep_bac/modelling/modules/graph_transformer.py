import torch
from torch import nn

from deep_bac.modelling.modules.layers import DenseLayer
from deep_bac.modelling.modules.utils import Flatten


class GraphTransformer(nn.Module):
    def __init__(
        self,
        n_gene_bottleneck_layer: int,
        n_output: int,
        n_genes: int,
        n_layers: int = 1,
        n_heads: int = 2,
    ):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=n_gene_bottleneck_layer,
            nhead=n_heads,
            dim_feedforward=n_gene_bottleneck_layer * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            self.layer, num_layers=n_layers
        )
        self.dropout = nn.Dropout(0.2)
        self.decoder = nn.Sequential(
            Flatten(),
            DenseLayer(
                in_features=n_gene_bottleneck_layer * n_genes,
                out_features=n_gene_bottleneck_layer,
            ),
            nn.Linear(
                in_features=n_gene_bottleneck_layer,
                out_features=n_output,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        logits = self.decoder(x)
        return logits
