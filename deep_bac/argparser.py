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
        "gene_bac",
        "MD-CNN",
        "xpresso",
        "zrimec_et_al_2020",
        "simple_cnn",
    ] = "gene_bac"
    graph_model_type: Literal["transformer", "dense", "GAT", "GCN"] = "GAT"
    regression: bool = False
    use_drug_idx: int = None
    lr: float = 0.0001
    n_gene_bottleneck_layer: int = 64
    n_init_filters: int = 128
    n_heads: int = 2
    n_graph_layers: int = 2
    n_output: int = 14  # nr of drugs in the cryptic dataset
    # data loader arguments
    max_gene_length: int = 2560
    shift_max: int = 3
    pad_value: float = 0.25
    reverse_complement_prob: float = 0.0
    num_workers: int = None
    # trainer arguments
    max_epochs: int = 100
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
        "val_r2_high",
        "val_spearman",
        "train_loss",
        "train_gmean_spec_sens",
        "train_r2",
    ] = "train_gmean_spec_sens"
    use_drug_specific_genes: Literal[
        "cryptic",
        "PA_GWAS_top_3",
        "PA_GWAS_top_5",
    ] = "cryptic"
    pos_encoder_type: Literal["learnable", "fixed"] = "fixed"
    resume_from_ckpt_path: str = None
    fold_idx: int = None
    gene_encoder_ckpt_path: str = None
