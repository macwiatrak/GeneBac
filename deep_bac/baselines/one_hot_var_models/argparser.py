import os
from typing import Literal

from tap import Tap

INPUT_DIR = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data/"


class OneHotModelArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    df_unq_ids_labels_file_path: str = os.path.join(
        INPUT_DIR, "phenotype_labels_with_binary_labels.parquet"
    )
    train_test_split_unq_ids_file_path: str = os.path.join(
        INPUT_DIR, "train_test_cv_split_unq_ids.json"
    )
    variant_matrix_input_dir: str = "/tmp/variant-matrix-cryptic-loci-updated/"
    output_dir: str = "/tmp/"
    # model hyperparameters
    max_iter: int = 500
    penalty: Literal["l1", "l2", "elasticnet"] = "l1"
    random_state: int = 42
    exclude_vars_not_in_train: bool = False
    regression: bool = False
