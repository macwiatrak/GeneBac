from typing import Literal, Optional, Dict

from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class DeepGeneBacConfig:
    gene_encoder_type: Literal[
        "conv_transformer", "gene_bac", "MD-CNN", "enformer_like", "xpresso"
    ] = "gene_bac"
    graph_model_type: Literal["transformer", "dense", "GAT", "GCN"] = "dense"
    regression: bool = False
    lr: float = 0.001
    batch_size: int = 1
    n_gene_bottleneck_layer: int = 64
    n_init_filters: int = 128
    n_heads: int = 2
    n_graph_layers: int = 2
    n_output: int = 14  # nr of drugs in the cryptic dataset
    max_gene_length: int = 2560
    max_epochs: int = 100
    train_set_len: Optional[int] = None
    warmup_proportion: float = 0.1
    random_state: int = 42
    dropout_rate: float = 0.1
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    accumulate_grad_batches: int = 1
    monitor_metric: Literal[
        "val_loss",
        "val_auroc",
        "val_f1",
        "val_gmean_spec_sens",
        "val_r2",
        "val_spearman",
        "train_loss",
        "train_gmean_spec_sens",
        "train_r2",
    ] = "train_gmean_spec_sens"
    n_highly_variable_genes: int = 500
    pos_encoder_type: Literal["learnable", "fixed"] = None
    input_dir: str = None
    gene_to_idx: Dict[str, int] = None
