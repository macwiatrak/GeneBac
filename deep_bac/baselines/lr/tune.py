import json
import logging
import os
from typing import Dict, List, Literal

import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from deep_bac.baselines.lr.data_reader import (
    split_train_val_test,
    get_drug_var_matrices,
)

logging.basicConfig(level=logging.INFO)

INPUT_DIR = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data/"


def tune(
    drug_idx: int,
    train_test_split_unq_ids_file_path: str,
    variant_matrix: csr_matrix,
    unq_id_to_idx: Dict[str, int],
    df_unq_ids_labels: pd.DataFrame,
    model: LogisticRegression,
    parameters: Dict[str, List],
):
    data_matrices = split_train_val_test(
        train_test_split_unq_ids_file_path=train_test_split_unq_ids_file_path,
        variant_matrix=variant_matrix,
        unq_id_to_idx=unq_id_to_idx,
        df_unq_ids_labels=df_unq_ids_labels,
    )

    data_matrices = get_drug_var_matrices(
        drug_idx=drug_idx,
        data=data_matrices,
    )

    # do grid search for best hyperparameters
    # TODO: make a function to score based on gmean spec sens
    clf = GridSearchCV(model, parameters, cv=5, scoring="f1_score")
    clf.fit(data_matrices.train_var_matrix, data_matrices.train_labels)
    # return best hyperparameters
    return clf.best_params_


def run(
    drug_idx: int,
    train_test_split_unq_ids_file_path: str,
    variant_matrix_input_dir: str,
    df_unq_ids_labels: pd.DataFrame,
    params: Dict[str, List],
    max_iter: int,
    penalty: Literal["l1", "l2", "elasticnet"] = None,
):
    variant_matrix = load_npz(
        os.path.join(variant_matrix_input_dir, "var_matrix.npz")
    )
    with open(
        os.path.join(variant_matrix_input_dir, "unq_id_to_idx.json"), "r"
    ) as f:
        unq_id_to_idx = json.load(f)

    model = LogisticRegression(
        max_iter=max_iter,
        penalty=penalty,
    )
    best_params = tune(
        drug_idx=drug_idx,
        train_test_split_unq_ids_file_path=train_test_split_unq_ids_file_path,
        variant_matrix=variant_matrix,
        unq_id_to_idx=unq_id_to_idx,
        df_unq_ids_labels=df_unq_ids_labels,
        model=model,
        parameters=params,
    )
    return best_params


def main():

    best_params = run(
        drug_idx=0,
        train_test_split_unq_ids_file_path=os.path.join(
            INPUT_DIR, "train_test_cv_split_unq_ids.json"
        ),
        variant_matrix_input_dir="",
        df_unq_ids_labels=pd.read_parquet(
            os.path.join(
                INPUT_DIR, "phenotype_labels_with_binary_labels.parquet"
            )
        ),
        params={
            "C": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5],
            "class_weight": [None, "balanced"],
        },
        penalty=None,
        max_iter=1000,
    )
    print(best_params)
