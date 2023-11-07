import logging
import os
from typing import Dict, List, Literal, Any

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV

from deep_bac.baselines.abr.one_hot_var_models.data_reader import (
    get_var_matrix_data,
)
from deep_bac.baselines.abr.one_hot_var_models.data_types import DataVarMatrices
from deep_bac.modelling.metrics import choose_best_spec_sens_threshold

logging.basicConfig(level=logging.INFO)

INPUT_DIR = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data/"


def gmean_spec_sens_score_fn(y: np.ndarray, y_pred: np.ndarray) -> float:
    thresh, max_gmean, spec, sens = choose_best_spec_sens_threshold(
        logits=torch.tensor(y_pred),
        labels=torch.tensor(y),
    )
    return max_gmean.item()


def tune(
    data_matrices: DataVarMatrices,
    model: LogisticRegression,
    parameters: Dict[str, List],
    penalty: Literal["l1", "l2", "elasticnet"] = "l1",
    regression: bool = False,
    n_folds: int = 5,
) -> Dict[str, Any]:
    score_fn = gmean_spec_sens_score_fn if not regression else r2_score
    scorer = make_scorer(
        score_fn,
        greater_is_better=True,
        needs_proba=False if penalty == "elasticnet" or regression else True,
    )
    logging.info(f"Starting the tuning with nr of folds: {n_folds}")
    clf = GridSearchCV(model, parameters, cv=n_folds, scoring=scorer, n_jobs=-1)
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
    exclude_vars_not_in_train: bool = False,
    regression: bool = False,
) -> Dict[str, Any]:
    seed_everything(random_state)

    data = get_var_matrix_data(
        drug_idx=drug_idx,
        variant_matrix_input_dir=variant_matrix_input_dir,
        train_test_split_unq_ids_file_path=train_test_split_unq_ids_file_path,
        df_unq_ids_labels=df_unq_ids_labels,
        exclude_vars_not_in_train=exclude_vars_not_in_train,
        regression=regression,
    )

    solver = "saga" if penalty == "elasticnet" else "liblinear"
    model = LogisticRegression(
        max_iter=max_iter,
        penalty=penalty,
        random_state=random_state,
        tol=0.001,
        solver=solver,
    )

    best_params = tune(
        data_matrices=data,
        model=model,
        parameters=params,
        regression=regression,
    )
    logging.info(f"Best params: {best_params}")
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
        params={},
        penalty=None,
        max_iter=100,
        random_state=42,
    )
    print(best_params)


if __name__ == "__main__":
    main()
