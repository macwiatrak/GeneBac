from typing import Literal

from tap import Tap


class OneHotModelsArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    train_test_split_unq_ids_file_path: str
    df_unq_ids_labels_file_path: str
    variant_matrix_input_dir: str = "/tmp/var-matrix/"
    output_file_path: str = "/tmp/lr_model_output.csv"
    # model hyperparameters
    max_iter: int = 1000
    penalty: Literal["l1", "l2", "elasticnet"] = "l2"
    random_state: int = 42
