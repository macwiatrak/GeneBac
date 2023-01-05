from typing import Literal, Optional

from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class DeepBacConfig:
    gene_encoder_type: Literal["conv_transformer"] = "conv_transformer"
    graph_model_type: Literal["transformer"] = "transformer"
    regression: bool = False
    lr: float = 0.001
    batch_size: int = 16
    n_gene_bottleneck_layer: int = 256
    n_init_filters: int = 256
    n_transformer_heads: int = 8
    n_graph_layers: int = 4
    n_output: int = 14  # nr of drugs in the cryptic dataset
    max_epochs: int = 100
    train_set_len: Optional[int] = None
    warmup_proportion: float = 0.1
    random_state: int = 42
