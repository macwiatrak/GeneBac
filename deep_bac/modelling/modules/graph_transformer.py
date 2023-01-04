import torch
from torch import nn


class GraphTransformer(nn.Module):
    def __init__(
            self,
            dim: int,
            n_output: int,
            n_layers: int = 4,
            n_heads: int = 8,
    ):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.layer, num_layers=n_layers)
        self.dropout = nn.Dropout(0.2)
        self.decoder = nn.Linear(dim * 2, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        x = self.dropout(x)
        # mean and max pooling
        x = torch.cat([x.max(dim=1)[0], x.mean(dim=1)], dim=1)
        logits = self.decoder(x)
        return logits
