import logging
import os
from typing import Dict, List, Literal

import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.linear_model import LogisticRegression

from deep_bac.baselines.lr.data_reader import (
    split_train_val_test,
    get_drug_var_matrices,
    get_var_matrix_data,
)
from deep_bac.baselines.lr.tune import tune, INPUT_DIR

logging.basicConfig(level=logging.INFO)


def run(
    drug_idx: int,
    train_test_split_unq_ids_file_path: str,
    variant_matrix_input_dir: str,
    df_unq_ids_labels: pd.DataFrame,
    params: Dict[str, List],
    max_iter: int,
    penalty: Literal["l1", "l2", "elasticnet"] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    seed_everything(random_state)
    data_matrices = get_var_matrix_data(
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
    logging.info(f"Using logistic regression with {penalty} penalty")

    best_params = tune(
        data_matrices=data_matrices,
        model=model,
        parameters=params,
    )
    logging.info(f"Best params: {best_params}")

    # tune and get the best model
    best_model = LogisticRegression(
        max_iter=max_iter,
        penalty=penalty,
        random_state=random_state,
        tol=0.001,
        **best_params,
    )

    # fit the best model
    best_model.fit(data_matrices.train_var_matrix, data_matrices.train_labels)
    train_pred = best_model.predict_proba(data_matrices.train_var_matrix)
    # TODO: get optimal threshold
    # threshold = get_optimal_threshold(train_pred, data_matrices.train_labels)
    threshold = 0.5

    test_pred = best_model.predict(data_matrices.test_var_matrix)
    # metrics = compute_metrics(test_pred, data_matrices.test_labels, threshold)
    return {}


def main():
    run(
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
            "C": [1.0],
            # "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
            # "class_weight": [None, "balanced"],
        },
        max_iter=1000,
        penalty="l2",
        random_state=42,
    )


if __name__ == "__main__":
    main()
