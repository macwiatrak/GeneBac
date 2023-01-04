from typing import Literal

from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class DeepBacConfig:
    gene_encoder_type: Literal["conv_transformer"] = "conv_transformer"
    graph_model_type: Literal["transformer"] = "transformer"
    regression: bool = False
    n_gene_bottleneck_layer: int = 256
    n_init_filters: int = 256
    n_transformer_heads: int = 8
    n_graph_layers: int = 4
    n_output: int = 3
