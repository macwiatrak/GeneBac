from typing import Literal

from tap import Tap


class TrainArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)
    input_df_file_path: str
    output_dir: str
    reference_gene_seqs_dict_path: str
    phenotype_df_file_path: str
    train_val_test_split_indices_file_path: str
    selected_genes_file_path: str
    # model arguments
    batch_size: int = 8
    gene_encoder_type: Literal["conv_transformer"] = "conv_transformer"
    graph_model_type: Literal["transformer"] = "transformer"
    regression: bool = False
    lr: float = 1e-3
    n_gene_bottleneck_layer: int = 256
    n_init_filters: int = 256
    n_transformer_heads: int = 8
    n_graph_layers: int = 4
    n_output: int = 14  # nr of drugs in the cryptic dataset
    max_gene_length: int = 2048
    shift_max: int = 3
    pad_value: float = 0.25
    reverse_complement_prob: float = 0.5
    num_workers: int = 8
    num_epochs: int = 100
    test: bool = False
    ckpt_path: str = None
    random_state: int = 42
    warmup_proportion: float = 0.1
