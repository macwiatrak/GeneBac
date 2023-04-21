import json
from typing import Tuple

import numpy as np
import pandas as pd

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
    # train, val, test = split_train_val_test(
    #     train_test_split_strain_ids_file_path=train_test_split_strain_ids_file_path,
    #     df=vars_df,
    # )
    # TODO check this works
    train_x = train_vars_df.loc[gene].values
    val_x = val_vars_df.loc[gene].values
    test_x = test_vars_df.loc[gene].values

    if exclude_vars_not_in_train:
        # TODO check this works
        vars_in_train = np.where(train_x.sum(axis=0) > 0)[0]
        train_x = train_x[:, vars_in_train]
        val_x = val_x[:, vars_in_train]
        test_x = test_x[:, vars_in_train]

    gene_expression = expression_df.loc[gene]
    train_y = np.array([gene_expression[s] for s in train_vars_df.columns])
    val_y = np.array([gene_expression[s] for s in val_vars_df.columns])
    test_y = np.array([gene_expression[s] for s in test_vars_df.columns])

    return ExpressionDataVarMatrices(
        train_x=train_x,
        val_x=val_x,
        test_x=test_x,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
    )
