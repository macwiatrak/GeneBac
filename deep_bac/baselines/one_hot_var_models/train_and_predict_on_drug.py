import logging
import os
from typing import Dict, List, Literal

import pandas as pd
import torch
from pytorch_lightning.utilities.seed import seed_everything

from deep_bac.baselines.one_hot_var_models.data_reader import (
    get_var_matrix_data,
)
from deep_bac.baselines.one_hot_var_models.tune import tune, INPUT_DIR
from deep_bac.baselines.one_hot_var_models.utils import get_model, get_preds
from deep_bac.modelling.metrics import (
    choose_best_spec_sens_threshold,
    binary_cls_metrics,
    get_regression_metrics,
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
    regression: bool = False,
) -> Dict[str, float]:
    seed_everything(random_state)
    data_matrices = get_var_matrix_data(
        drug_idx=drug_idx,
        variant_matrix_input_dir=variant_matrix_input_dir,
        train_test_split_unq_ids_file_path=train_test_split_unq_ids_file_path,
        df_unq_ids_labels=df_unq_ids_labels,
        exclude_vars_not_in_train=exclude_vars_not_in_train,
        regression=regression,
    )
    model = get_model(
        penalty=penalty,
        max_iter=max_iter,
        random_state=random_state,
        regression=regression,
    )

    logging.info(f"Using model with {penalty} penalty")

    best_params = tune(
        data_matrices=data_matrices,
        model=model,
        parameters=params,
        penalty=penalty,
    )

    best_model = get_model(
        penalty=penalty,
        max_iter=max_iter,
        random_state=random_state,
        regression=regression,
        best_params=best_params,
    )

    logging.info(f"Fitting and computing metrics using the best model.")
    # fit the best model
    best_model.fit(data_matrices.train_var_matrix, data_matrices.train_labels)

    train_pred = get_preds(
        best_model, data_matrices.train_var_matrix, penalty, regression
    )

    if not regression:
        # get optimal thresholds using the train set
        thresh, _, _, _ = choose_best_spec_sens_threshold(
            logits=torch.tensor(train_pred),
            labels=torch.tensor(data_matrices.train_labels),
        )

    # predict on the test set
    test_pred = get_preds(
        model, data_matrices.test_var_matrix, penalty, regression
    )

    if regression:
        metrics = get_regression_metrics(
            logits=torch.tensor(test_pred),
            labels=torch.tensor(data_matrices.test_labels),
        )
    else:
        # compute the metrics using the test set
        metrics, thresh = binary_cls_metrics(
            logits=torch.tensor(test_pred),
            labels=torch.tensor(data_matrices.test_labels),
            thresh=thresh,
        )
        metrics["threshold"] = torch.tensor(thresh)
    metrics = {k: v.item() for k, v in metrics.items()}
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
        regression=False,
    )


if __name__ == "__main__":
    main()
