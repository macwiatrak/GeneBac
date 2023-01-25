from typing import Literal

from tap import Tap

INPUT_DIR = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data/"


class TrainArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str = INPUT_DIR
    output_dir: str = "/tmp/cryptic-model-output/"
    n_highly_variable_genes: int = 500
    # model arguments
    batch_size: int = 1
    gene_encoder_type: Literal["conv_transformer", "scbasset"] = "scbasset"
    graph_model_type: Literal["transformer"] = "transformer"
    regression: bool = False
    lr: float = 1e-3
    n_gene_bottleneck_layer: int = 64
    n_init_filters: int = 256
    n_transformer_heads: int = 8
    n_graph_layers: int = 4
    n_output: int = 14  # nr of drugs in the cryptic dataset
    # data loader arguments
    max_gene_length: int = 2048
    shift_max: int = 3
    pad_value: float = 0.25
    reverse_complement_prob: float = 0.5
    num_workers: int = None
    # trainer arguments
    max_epochs: int = 100
    early_stopping_patience: int = 10
    test: bool = False
    ckpt_path: str = None
    random_state: int = 42
    warmup_proportion: float = 0.1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
