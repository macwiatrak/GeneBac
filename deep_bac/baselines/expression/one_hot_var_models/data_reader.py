import json
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from deep_bac.baselines.expression.one_hot_var_models.data_types import (
    ExpressionDataVarMatrices,
)


def split_train_val_test(
    train_test_split_strain_ids_file_path: str,
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(train_test_split_strain_ids_file_path, "r") as f:
        train_test_split_strain_ids = json.load(f)

    train_strain_ids = train_test_split_strain_ids["train"]
    val_strain_ids = train_test_split_strain_ids["val"]
    test_strain_ids = train_test_split_strain_ids["test"]

    train = df[train_strain_ids]
    val = df[val_strain_ids]
    test = df[test_strain_ids]

    return train, val, test


def get_gene_matrix_data(
    gene: str,
    train_vars_df: pd.DataFrame,
    val_vars_df: pd.DataFrame,
    test_vars_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    exclude_vars_not_in_train: bool,
):
    train_x = np.stack(train_vars_df.loc[gene].values)
    val_x = np.stack(val_vars_df.loc[gene].values)
    test_x = np.stack(test_vars_df.loc[gene].values)

    if exclude_vars_not_in_train:
        vars_in_train = np.where(train_x.sum(axis=0) > 0)[0]
        if len(vars_in_train) == 0:
            train_x = np.zeros((train_x.shape[0], 1))
            val_x = np.zeros((val_x.shape[0], 1))
            test_x = np.zeros((test_x.shape[0], 1))
        else:
            train_x = train_x[:, vars_in_train]
            val_x = val_x[:, vars_in_train]
            test_x = test_x[:, vars_in_train]

    gene_expression = expression_df.loc[gene]
    train_y = np.array([gene_expression[s] for s in train_vars_df.columns])
    val_y = np.array([gene_expression[s] for s in val_vars_df.columns])
    test_y = np.array([gene_expression[s] for s in test_vars_df.columns])

    return dict(
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
    )


def process_one_hot_expression_data(
    vars_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    train_test_split_strain_ids_file_path: str,
    exclude_vars_not_in_train: bool,
):
    train, val, test = split_train_val_test(
        train_test_split_strain_ids_file_path=train_test_split_strain_ids_file_path,
        df=vars_df,
    )
    output = defaultdict(dict)
    for gene in tqdm(vars_df.index.tolist()):
        output[gene] = get_gene_matrix_data(
            gene=gene,
            train_vars_df=train,
            val_vars_df=val,
            test_vars_df=test,
            expression_df=expression_df,
            exclude_vars_not_in_train=exclude_vars_not_in_train,
        )

    max_vars = max([output[g]["train_x"].shape[-1] for g in output.keys()])

    for gene, vars_dict in tqdm(output.items()):
        n_samples, n_vars = vars_dict["train_x"].shape
        train_pad = np.zeros((n_samples, max_vars - n_vars))
        output[gene]["train_x"] = np.concatenate(
            [vars_dict["train_x"], train_pad], axis=1
        )

        n_samples, n_vars = vars_dict["val_x"].shape
        val_pad = np.zeros((n_samples, max_vars - n_vars))
        output[gene]["val_x"] = np.concatenate(
            [vars_dict["val_x"], val_pad], axis=1
        )

        n_samples, n_vars = vars_dict["test_x"].shape
        test_pad = np.zeros((n_samples, max_vars - n_vars))
        output[gene]["test_x"] = np.concatenate(
            [vars_dict["test_x"], test_pad], axis=1
        )

    train_dict = defaultdict(list)
    val_dict = defaultdict(list)
    test_dict = defaultdict(list)

    for gene, vars_dict in tqdm(output.items()):
        train_dict["x"] += [item for item in vars_dict["train_x"]]
        train_dict["y"] += [item for item in vars_dict["train_y"]]
        train_dict["gene"] += [gene] * vars_dict["train_x"].shape[0]
        train_dict["strain"] += train.columns.tolist()

        val_dict["x"] += [item for item in vars_dict["val_x"]]
        val_dict["y"] += [item for item in vars_dict["val_y"]]
        val_dict["gene"] += [gene] * vars_dict["val_x"].shape[0]
        val_dict["strain"] += val.columns.tolist()

        test_dict["x"] += [item for item in vars_dict["test_x"]]
        test_dict["y"] += [item for item in vars_dict["test_y"]]
        test_dict["gene"] += [gene] * vars_dict["test_x"].shape[0]
        test_dict["strain"] += test.columns.tolist()

    train_out = pd.DataFrame(train_dict)
    val_out = pd.DataFrame(val_dict)
    test_out = pd.DataFrame(test_dict)
    return train_out, val_out, test_out
