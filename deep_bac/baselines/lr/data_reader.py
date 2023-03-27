import json
from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from deep_bac.baselines.lr.data_types import DataVarMatrices


def split_train_val_test(
    train_test_split_unq_ids_file_path: str,
    variant_matrix: csr_matrix,
    unq_id_to_idx: Dict[str, int],
    df_unq_ids_labels: pd.DataFrame,
) -> DataVarMatrices:
    with open(train_test_split_unq_ids_file_path, "r") as f:
        train_test_split_unq_ids = json.load(f)

    train_unq_ids = train_test_split_unq_ids["train"]
    test_unq_ids = train_test_split_unq_ids["test"]

    train_var_matrix = np.stack(
        [variant_matrix[unq_id_to_idx[unq_id]] for unq_id in train_unq_ids]
    )
    test_var_matrix = np.stack(
        [variant_matrix[unq_id_to_idx[unq_id]] for unq_id in test_unq_ids]
    )

    train_labels = np.stack(
        [df_unq_ids_labels.loc[unq_id] for unq_id in train_unq_ids]
    )
    test_labels = np.stack(
        [df_unq_ids_labels.loc[unq_id] for unq_id in test_unq_ids]
    )

    return DataVarMatrices(
        train_var_matrix=train_var_matrix,
        test_var_matrix=test_var_matrix,
        train_labels=train_labels,
        test_labels=test_labels,
    )


def get_drug_var_matrices(
    drug_idx: int,
    data: DataVarMatrices,
):
    train_drug_indices = np.where(data.train_labels[:, drug_idx] != -100.0)[0]
    test_drug_indices = np.where(data.train_labels[:, drug_idx] != -100.0)[0]

    train_var_matrix = data.train_var_matrix[train_drug_indices]
    train_labels = data.train_labels[train_drug_indices, drug_idx]

    test_var_matrix = data.test_var_matrix[test_drug_indices, drug_idx]
    test_labels = data.test_labels[test_drug_indices, drug_idx]
    return DataVarMatrices(
        train_var_matrix=train_var_matrix,
        test_var_matrix=test_var_matrix,
        train_labels=train_labels,
        test_labels=test_labels,
    )
