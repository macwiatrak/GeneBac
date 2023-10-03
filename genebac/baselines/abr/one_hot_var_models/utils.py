from collections import defaultdict
from typing import Dict, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LogisticRegression,
    ElasticNet,
    Lasso,
    Ridge,
)

from genebac.modelling.metrics import MTB_DRUG_TO_DRUG_CLASS

MTB_DRUG_TO_IDX = {
    "MXF": 0,
    "BDQ": 1,
    "KAN": 2,
    "CFZ": 3,
    "AMI": 4,
    "DLM": 6,
    "RFB": 7,
    "LZD": 8,
    "EMB": 9,
    "LEV": 10,
    "ETH": 11,
    "INH": 12,
    "RIF": 13,
}  # removed "PAS": 5 as it has not enough labels


def get_model(
    penalty: Literal["l1", "l2", "elasticnet"] = "l1",
    random_state: int = 42,
    max_iter: int = 500,
    regression: bool = False,
    best_params: Dict = None,
):
    if best_params is None:
        best_params = {}

    if penalty == "elasticnet":
        return ElasticNet(
            max_iter=max_iter,
            random_state=random_state,
            tol=0.001,
            **best_params,
        )

    if not regression:
        return LogisticRegression(
            max_iter=max_iter,
            penalty=penalty,
            random_state=random_state,
            tol=0.001,
            solver="liblinear",  # supports both l1 and l2
            **best_params,
        )

    if penalty == "l1":
        return Lasso(
            max_iter=max_iter,
            random_state=random_state,
            tol=0.001,
            **best_params,
        )

    if penalty == "l2":
        return Ridge(
            max_iter=max_iter,
            random_state=random_state,
            tol=0.001,
            **best_params,
        )


def get_preds(
    model,
    X: np.ndarray,
    penalty: Literal["l1", "l2", "elasticnet"] = "l1",
    regression: bool = False,
) -> np.ndarray:
    if penalty == "elasticnet" or regression:
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def dict_metrics_to_df(
    metrics: Dict[str, float],
    drug: str,
    split: Literal["train", "val", "test"] = "test",
    drug_to_drug_class: Dict[str, str] = MTB_DRUG_TO_DRUG_CLASS,
) -> pd.DataFrame:
    """Converts the metrics dictionary to a dataframe."""
    output = defaultdict(list)
    for metric, val in metrics.items():
        output["value"].append(val)
        output["metric"].append(metric)
        output["drug"].append(drug)
        output["split"].append(split)
        output["drug_class"] = drug_to_drug_class[drug]
    return pd.DataFrame(output)
