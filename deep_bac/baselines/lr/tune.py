import logging
import os
from typing import Dict, List, Literal, Tuple, Any

import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from deep_bac.baselines.lr.data_reader import (
    get_var_matrix_data,
)
from deep_bac.baselines.lr.data_types import DataVarMatrices

logging.basicConfig(level=logging.INFO)

INPUT_DIR = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data/"


def tune(
    data_matrices: DataVarMatrices,
    model: LogisticRegression,
    parameters: Dict[str, List],
) -> Dict[str, Any]:
    # TODO: make a function to score based on gmean spec sens
    clf = GridSearchCV(model, parameters, cv=5, scoring="f1", n_jobs=-1)
    clf.fit(
        data_matrices.train_var_matrix,
        data_matrices.train_labels,
    )
    # return best hyperparameters
    return clf.best_params_


def run_grid_search_cv(
    drug_idx: int,
    train_test_split_unq_ids_file_path: str,
    variant_matrix_input_dir: str,
    df_unq_ids_labels: pd.DataFrame,
    params: Dict[str, List],
    max_iter: int,
    penalty: Literal["l1", "l2", "elasticnet"] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    seed_everything(random_state)

    data = get_var_matrix_data(
        drug_idx=drug_idx,
        variant_matrix_input_dir=variant_matrix_input_dir,
        train_test_split_unq_ids_file_path=train_test_split_unq_ids_file_path,
        df_unq_ids_labels=df_unq_ids_labels,
    )

    model = LogisticRegression(
        max_iter=max_iter,
        penalty=penalty,
        random_state=random_state,
        tol=0.001,
    )

    best_params, _ = tune(
        data_matrices=data,
        model=model,
        parameters=params,
    )
    print(best_params)
    return best_params


def main():

    best_params = run_grid_search_cv(
        drug_idx=0,
        train_test_split_unq_ids_file_path=os.path.join(
            INPUT_DIR, "train_test_cv_split_unq_ids.json"
        ),
        variant_matrix_input_dir="/tmp/var-matrix/",
        df_unq_ids_labels=pd.read_parquet(
            os.path.join(
                INPUT_DIR, "phenotype_labels_with_binary_labels.parquet"
            )
        ),
        params={
            "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
            "class_weight": [None, "balanced"],
        },
        penalty=None,
        max_iter=1000,
        random_state=42,
    )
    print(best_params)


if __name__ == "__main__":
    main()
