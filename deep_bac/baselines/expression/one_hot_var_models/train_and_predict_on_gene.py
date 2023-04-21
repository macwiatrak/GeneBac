import logging
from typing import Literal, Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from deep_bac.baselines.abr.one_hot_var_models.utils import get_model, get_preds
from deep_bac.baselines.expression.one_hot_var_models.data_reader import (
    get_gene_matrix_data,
)
from deep_bac.baselines.expression.one_hot_var_models.data_types import (
    ExpressionDataVarMatrices,
)
from deep_bac.modelling.metrics import get_regression_metrics

logging.basicConfig(level=logging.INFO)


def tune(
    data_matrices: ExpressionDataVarMatrices,
    model: Optional[Lasso],
    parameters: Dict[str, List],
) -> Dict[str, Any]:
    scorer = make_scorer(
        r2_score,
        greater_is_better=True,
        needs_proba=False,  # if penalty == "elasticnet" or regression else True,
    )

    x = np.concatenate([data_matrices.train_x, data_matrices.val_x])
    y = np.concatenate([data_matrices.train_x, data_matrices.val_y])
    val_indices = np.concatenate(
        [
            np.ones(len(data_matrices.train_x)) * -1,
            np.zeros(len(data_matrices.val_x)),
        ]
    )
    ps = PredefinedSplit(val_indices)

    logging.info(f"Starting the tuning with a predefined split")
    clf = GridSearchCV(model, parameters, cv=ps, scoring=scorer, n_jobs=-1)
    clf.fit(x, y)
    # return best hyperparameters
    return clf.best_params_


def train_and_predict(
    gene: str,
    train_vars_df: pd.DataFrame,
    val_vars_df: pd.DataFrame,
    test_vars_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    exclude_vars_not_in_train: bool,
    params: Dict[str, List],
    max_iter: int,
    penalty: Literal["l1", "l2", "elasticnet"] = "l1",
    random_state: int = 42,
):
    seed_everything(random_state)
    data_matrices = get_gene_matrix_data(
        gene=gene,
        train_vars_df=train_vars_df,
        val_vars_df=val_vars_df,
        test_vars_df=test_vars_df,
        expression_df=expression_df,
        exclude_vars_not_in_train=exclude_vars_not_in_train,
    )
    model = get_model(
        penalty=penalty,
        max_iter=max_iter,
        random_state=random_state,
        regression=True,
    )

    logging.info(f"Using model with {penalty} penalty")

    best_params = tune(
        data_matrices=data_matrices,
        model=model,
        parameters=params,
    )

    best_model = get_model(
        penalty=penalty,
        max_iter=max_iter,
        random_state=random_state,
        best_params=best_params,
    )

    logging.info(f"Fitting and computing metrics using the best model.")
    # fit the best model
    best_model.fit(data_matrices.train_var_matrix, data_matrices.train_labels)

    test_pred = get_preds(
        model=best_model,
        X=data_matrices.test_x,
        penalty=penalty,
        regression=True,
    )

    metrics = get_regression_metrics(
        logits=torch.tensor(test_pred),
        labels=torch.tensor(data_matrices.test_y),
    )
    # add best params to save them
    metrics.update(best_params)
    return metrics
