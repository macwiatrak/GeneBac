from typing import Literal, Optional

from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class DeepBacConfig:
    gene_encoder_type: Literal[
        "conv_transformer", "scbasset"
    ] = "conv_transformer"
    graph_model_type: Literal["transformer"] = "transformer"
    regression: bool = False
    lr: float = 0.001
    batch_size: int = 1
    n_gene_bottleneck_layer: int = 128
    n_init_filters: int = 256
    n_transformer_heads: int = 4
    n_graph_layers: int = 2
    n_output: int = 14  # nr of drugs in the cryptic dataset
    max_epochs: int = 100
    train_set_len: Optional[int] = None
    warmup_proportion: float = 0.1
    random_state: int = 42
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    accumulate_grad_batches: int = 1
    monitor_metric = Literal[
        "val_loss", "val_auroc", "val_r2", "val_spearmann"
    ] = "val_loss"
