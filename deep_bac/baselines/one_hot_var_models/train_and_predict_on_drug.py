import logging
import os
from typing import Dict, List, Literal

import pandas as pd
import torch
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.linear_model import LogisticRegression, ElasticNet

from deep_bac.baselines.one_hot_var_models.data_reader import (
    get_var_matrix_data,
)
from deep_bac.baselines.one_hot_var_models.tune import tune, INPUT_DIR
from deep_bac.modelling.metrics import (
    choose_best_spec_sens_threshold,
    binary_cls_metrics,
)

logging.basicConfig(level=logging.INFO)


def train_and_predict(
    drug_idx: int,
    train_test_split_unq_ids_file_path: str,
    variant_matrix_input_dir: str,
    df_unq_ids_labels: pd.DataFrame,
    params: Dict[str, List],
    max_iter: int,
    penalty: Literal["l1", "l2", "elasticnet"] = "l1",
    random_state: int = 42,
    exclude_vars_not_in_train: bool = False,
) -> Dict[str, float]:
    seed_everything(random_state)
    data_matrices = get_var_matrix_data(
        drug_idx=drug_idx,
        variant_matrix_input_dir=variant_matrix_input_dir,
        train_test_split_unq_ids_file_path=train_test_split_unq_ids_file_path,
        df_unq_ids_labels=df_unq_ids_labels,
        exclude_vars_not_in_train=exclude_vars_not_in_train,
    )

    if penalty == "l1" or penalty == "l2":
        model = LogisticRegression(
            max_iter=max_iter,
            penalty=penalty,
            random_state=random_state,
            tol=0.001,
            solver="liblinear",  # supports both l1 and l2
        )
    else:
        model = ElasticNet(
            max_iter=max_iter,
            random_state=random_state,
            tol=0.001,
        )
    logging.info(f"Using logistic regression with {penalty} penalty")

    best_params = tune(
        data_matrices=data_matrices,
        model=model,
        parameters=params,
        penalty=penalty,
    )

    # tune and get the best model
    if penalty == "l1" or penalty == "l2":
        best_model = LogisticRegression(
            max_iter=max_iter,
            penalty=penalty,
            random_state=random_state,
            tol=0.001,
            solver="liblinear",  # supports all penalties
            **best_params,
        )
    else:
        best_model = ElasticNet(
            max_iter=max_iter,
            random_state=random_state,
            tol=0.001,
            **best_params,
        )

    logging.info(f"Fitting and computing metrics using the best model.")
    # fit the best model
    best_model.fit(data_matrices.train_var_matrix, data_matrices.train_labels)

    if penalty == "elasticnet":
        train_pred = best_model.predict(data_matrices.train_var_matrix)
    else:
        train_pred = best_model.predict_proba(data_matrices.train_var_matrix)[
            :, 1
        ]

    # get optimal thresholds using the train set
    thresh, _, _, _ = choose_best_spec_sens_threshold(
        logits=torch.tensor(train_pred),
        labels=torch.tensor(data_matrices.train_labels),
    )

    # predict on the test set
    if penalty == "elasticnet":
        test_pred = best_model.predict(data_matrices.test_var_matrix)
    else:
        test_pred = best_model.predict_proba(data_matrices.test_var_matrix)[
            :, 1
        ]

    # compute the metrics using the test set
    metrics, thresh = binary_cls_metrics(
        logits=torch.tensor(test_pred),
        labels=torch.tensor(data_matrices.test_labels),
        thresh=thresh,
    )
    metrics = {k: v.item() for k, v in metrics.items()}
    metrics["threshold"] = thresh
    # add best params to save them
    metrics.update(best_params)
    return metrics


def main():
    _ = train_and_predict(
        drug_idx=0,
        train_test_split_unq_ids_file_path=os.path.join(
            INPUT_DIR, "train_test_cv_split_unq_ids.json"
        ),
        variant_matrix_input_dir="/tmp/variant-matrix-specific-loci/",
        df_unq_ids_labels=pd.read_parquet(
            os.path.join(
                INPUT_DIR, "phenotype_labels_with_binary_labels.parquet"
            )
        ),
        params={
            "alpha": [0.5, 1.0],
            # "C": [1.0],
            # "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        },
        max_iter=100,
        penalty="elasticnet",
        random_state=42,
        exclude_vars_not_in_train=True,
    )


if __name__ == "__main__":
    main()
