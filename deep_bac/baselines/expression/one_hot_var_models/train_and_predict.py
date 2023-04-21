import json
import logging
import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from deep_bac.baselines.abr.one_hot_var_models.argparser import (
    OneHotModelArgumentParser,
)
from deep_bac.baselines.abr.one_hot_var_models.train_and_predict import (
    get_tuning_params,
)
from deep_bac.baselines.expression.one_hot_var_models.data_reader import (
    split_train_val_test,
)
from deep_bac.baselines.expression.one_hot_var_models.train_and_predict_on_gene import (
    train_and_predict,
)
from deep_bac.modelling.metrics import get_regression_metrics


def run(
    output_dir: str,
    input_dir: str,
    train_test_split_file_path: str,
    max_iter: int = 500,
    penalty: Literal["l1", "l2", "elasticnet"] = "l1",
    random_state: int = 42,
    exclude_vars_not_in_train: bool = False,
):
    vars_df = pd.read_parquet(
        os.path.join(input_dir, "variants_per_gene.parquet")
    )
    expression_df = pd.read_parquet(
        os.path.join(input_dir, "gene_expression_vals.parquet")
    )

    train, val, test = split_train_val_test(train_test_split_file_path, vars_df)

    params = get_tuning_params(penalty, regression=True)
    output_metrics = []
    output_preds = []
    output_y = []
    for gene in tqdm(vars_df.index.tolist()):
        logging.info(f"Tuning and predicting for {gene} gene.")
        test_metrics, preds, y = train_and_predict(
            gene=gene,
            train_vars_df=train,
            val_vars_df=val,
            test_vars_df=test,
            expression_df=expression_df,
            exclude_vars_not_in_train=exclude_vars_not_in_train,
            max_iter=max_iter,
            penalty=penalty,
            random_state=random_state,
            params=params,
        )
        logging.info(f"Test metrics for {gene} gene: {test_metrics}")
        output_metrics.append(test_metrics)
        output_preds.append(preds)
        output_y.append(y)

    output_preds = torch.tensor(np.stack(output_preds))
    output_y = torch.tensor(np.stack(output_y))
    micro_metrics = get_regression_metrics(
        logits=output_preds.view(-1),
        labels=output_y.view(-1),
    )
    micro_metrics = {f"micro_{k}": v for k, v in micro_metrics.items()}
    logging.info(f"Micro metrics: {micro_metrics}")

    pd.DataFrame(output_metrics).to_csv(
        os.path.join(output_dir, f"test_results_{penalty}_{random_state}")
    )


def main(args):
    run(
        output_dir="/tmp/",  # args.output_dir,
        train_test_split_file_path="/tmp/train_val_test_split_by_strain.json",  # args.train_test_split_file_path,
        input_dir="/tmp/",  # args.variant_matrix_input_dir,
        max_iter=10,  # args.max_iter,
        penalty=args.penalty,
        random_state=args.random_state,
        exclude_vars_not_in_train=True,  # args.exclude_vars_not_in_train,
    )


if __name__ == "__main__":
    args = OneHotModelArgumentParser().parse_args()
    main(args)
