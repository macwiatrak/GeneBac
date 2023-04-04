import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, load_npz
from torch.utils.data import DataLoader, TensorDataset

from deep_bac.baselines.one_hot_var_models.data_types import (
    DataVarMatrices,
    OneHotVarDataReaderOutput,
)


def get_train_eval_var_data(
    train_unq_ids: List[str],
    eval_unq_ids: List[str],
    variant_matrix: csr_matrix,
    unq_id_to_idx: Dict[str, int],
    df_unq_ids_labels: pd.DataFrame,
    exclude_vars_not_in_train: bool = False,
) -> DataVarMatrices:

    zeros_vector = csr_matrix((1, variant_matrix.shape[1]), dtype=np.int8)

    train_var_matrix = []
    for unq_id in train_unq_ids:
        var_vector = (
            variant_matrix[unq_id_to_idx[unq_id]]
            if unq_id in unq_id_to_idx
            else zeros_vector
        )
        train_var_matrix.append(var_vector.toarray())
    train_var_matrix = np.concatenate(train_var_matrix)

    eval_var_matrix = []
    for unq_id in eval_unq_ids:
        var_vector = (
            variant_matrix[unq_id_to_idx[unq_id]]
            if unq_id in unq_id_to_idx
            else zeros_vector
        )
        eval_var_matrix.append(var_vector.toarray())
    test_var_matrix = np.concatenate(eval_var_matrix)

    if exclude_vars_not_in_train:
        vars_in_train = np.where(train_var_matrix.sum(axis=0) > 0)[0]
        train_var_matrix = train_var_matrix[:, vars_in_train]
        test_var_matrix = test_var_matrix[:, vars_in_train]

    train_labels = np.stack(
        [
            df_unq_ids_labels.loc[unq_id]["BINARY_LABELS"]
            for unq_id in train_unq_ids
        ]
    )
    test_labels = np.stack(
        [
            df_unq_ids_labels.loc[unq_id]["BINARY_LABELS"]
            for unq_id in eval_unq_ids
        ]
    )

    return DataVarMatrices(
        train_var_matrix=train_var_matrix,
        eval_var_matrix=test_var_matrix,
        train_labels=train_labels,
        eval_labels=test_labels,
    )


def get_drug_var_matrices(
    drug_idx: int,
    data: DataVarMatrices,
):
    train_drug_indices = np.where(data.train_labels[:, drug_idx] != -100.0)[0]
    test_drug_indices = np.where(data.eval_labels[:, drug_idx] != -100.0)[0]

    train_var_matrix = data.train_var_matrix[train_drug_indices]
    train_labels = data.train_labels[train_drug_indices, drug_idx]

    test_var_matrix = data.eval_var_matrix[test_drug_indices]
    test_labels = data.eval_labels[test_drug_indices, drug_idx]
    return DataVarMatrices(
        train_var_matrix=train_var_matrix,
        eval_var_matrix=test_var_matrix,
        train_labels=train_labels,
        eval_labels=test_labels,
    )


def get_var_matrix_data(
    drug_idx: int,
    variant_matrix_input_dir: str,
    train_test_split_unq_ids_file_path: str,
    df_unq_ids_labels: pd.DataFrame,
    exclude_vars_not_in_train: bool = False,
) -> DataVarMatrices:

    variant_matrix = load_npz(
        os.path.join(variant_matrix_input_dir, "var_matrix.npz")
    )
    with open(
        os.path.join(variant_matrix_input_dir, "unique_id_to_idx.json"), "r"
    ) as f:
        unq_id_to_idx = json.load(f)

    data = get_train_eval_var_data(
        train_test_split_unq_ids_file_path,
        variant_matrix,
        unq_id_to_idx,
        df_unq_ids_labels,
        exclude_vars_not_in_train,
    )
    data = get_drug_var_matrices(drug_idx, data)
    return data


def get_one_hot_data(
    variant_matrix_input_dir: str,
    df_unq_ids_labels_file_path: str,
    train_test_split_unq_ids_file_path: str = None,
    fold_idx: int = None,
    exclude_vars_not_in_train: bool = False,
    batch_size: int = 128,
    num_workers: int = 0,
):

    df_unq_ids_labels = pd.read_parquet(df_unq_ids_labels_file_path)

    variant_matrix = load_npz(
        os.path.join(variant_matrix_input_dir, "var_matrix.npz")
    )
    with open(
        os.path.join(variant_matrix_input_dir, "unique_id_to_idx.json"), "r"
    ) as f:
        unq_id_to_idx = json.load(f)

    with open(train_test_split_unq_ids_file_path, "r") as f:
        train_test_split_unq_ids = json.load(f)

    if fold_idx is None:
        train_unq_ids = train_test_split_unq_ids["train"]
        eval_unq_ids = train_test_split_unq_ids["test"]
    else:
        train_unq_ids = train_test_split_unq_ids[f"train_fold_{fold_idx}"]
        eval_unq_ids = train_test_split_unq_ids[f"val_fold_{fold_idx}"]

    data = get_train_eval_var_data(
        train_unq_ids,
        eval_unq_ids,
        variant_matrix,
        unq_id_to_idx,
        df_unq_ids_labels,
        exclude_vars_not_in_train,
    )
    if fold_idx is None:
        train_var_matrix = torch.tensor(
            data.train_var_matrix, dtype=torch.float32
        )
        train_labels = torch.tensor(data.train_labels)
        train_dl = DataLoader(
            TensorDataset(train_var_matrix, train_labels),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        eval_var_matrix = torch.tensor(
            data.eval_var_matrix, dtype=torch.float32
        )
        eval_labels = torch.tensor(data.eval_labels)
        test_dl = DataLoader(
            TensorDataset(eval_var_matrix, eval_labels),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        return OneHotVarDataReaderOutput(
            train_dl=train_dl,
            test_dl=test_dl,
            train_var_matrix=train_var_matrix,
            train_labels=train_labels,
        )

    train_var_matrix = torch.tensor(data.train_var_matrix, dtype=torch.float32)
    train_labels = torch.tensor(data.train_labels)

    train_dl = DataLoader(
        TensorDataset(train_var_matrix, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_var_matrix = torch.tensor(data.eval_var_matrix, dtype=torch.float32)
    val_labels = torch.tensor(data.eval_labels)
    val_dl = DataLoader(
        TensorDataset(val_var_matrix, val_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return OneHotVarDataReaderOutput(
        train_dl=train_dl,
        val_dl=val_dl,
        train_var_matrix=train_var_matrix,
        train_labels=train_labels,
    )
