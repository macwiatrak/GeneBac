from typing import Literal

from tap import Tap

INPUT_DIR = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data/"


class DeepGeneBacArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str = INPUT_DIR
    output_dir: str = "/tmp/cryptic-model-output/"
    n_highly_variable_genes: int = 500
    # model arguments
    batch_size: int = 1
    gene_encoder_type: Literal[
        "conv_transformer", "gene_bac", "MD-CNN", "enformer_like"
    ] = "gene_bac"
    graph_model_type: Literal["transformer", "dense"] = "dense"
    regression: bool = False
    use_drug_idx: int = None
    lr: float = 0.001
    n_gene_bottleneck_layer: int = 64
    n_init_filters: int = 256
    n_transformer_heads: int = 2
    n_graph_layers: int = 1
    n_output: int = 14  # nr of drugs in the cryptic dataset
    # data loader arguments
    max_gene_length: int = 2560
    shift_max: int = 3
    pad_value: float = 0.25
    reverse_complement_prob: float = 0.0
    num_workers: int = None
    # trainer arguments
    max_epochs: int = 150
    early_stopping_patience: int = 100
    test: bool = False
    test_after_train: bool = False
    ckpt_path: str = None
    random_state: int = 42
    warmup_proportion: float = 0.1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    monitor_metric: Literal[
        "val_loss",
        "val_auroc",
        "val_f1",
        "val_gmean_spec_sens",
        "val_r2",
        "val_spearman",
    ] = "val_loss"
    use_drug_specific_genes: Literal[
        "INH", "Walker", "MD-CNN", "cryptic"
    ] = "cryptic"
    pos_encoder_type: Literal["learnable", "fixed"] = "fixed"
    resume_from_ckpt_path: str = None
