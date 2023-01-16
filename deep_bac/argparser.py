import os
from typing import Literal

from tap import Tap

INPUT_DIR = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/"


class TrainArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)
    # file paths for loading data
    input_df_file_path: str = os.path.join(INPUT_DIR, "processed-genome-per-strain", "agg_variants.parquet")
    output_dir: str = "/tmp/cryptic-model-output/"
    reference_gene_seqs_dict_path: str = os.path.join(INPUT_DIR, 'reference_gene_seqs.json')
    phenotype_df_file_path: str = os.path.join(INPUT_DIR, "phenotype_labels_with_binary_labels.parquet")
    train_val_test_split_indices_file_path: str = os.path.join(INPUT_DIR, "train_val_test_split_unq_ids.json")
    n_highly_variable_genes: int = 500
    # model arguments
    batch_size: int = 1
    gene_encoder_type: Literal["conv_transformer"] = "conv_transformer"
    graph_model_type: Literal["transformer"] = "transformer"
    regression: bool = False
    lr: float = 1e-3
    n_gene_bottleneck_layer: int = 128
    n_init_filters: int = 256
    n_transformer_heads: int = 8
    n_graph_layers: int = 4
    n_output: int = 14  # nr of drugs in the cryptic dataset
    # data loader arguments
    max_gene_length: int = 2048
    shift_max: int = 3
    pad_value: float = 0.25
    reverse_complement_prob: float = 0.5
    num_workers: int = 8
    # trainer arguments
    max_epochs: int = 100
    early_stopping_patience: int = 10
    test: bool = False
    ckpt_path: str = None
    random_state: int = 42
    warmup_proportion: float = 0.1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    accelerator: Literal["cpu", "dp", "ddp"] = "cpu"
