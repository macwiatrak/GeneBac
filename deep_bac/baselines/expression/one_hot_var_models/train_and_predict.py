import json
import logging
import os
from typing import Literal

import pandas as pd

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
    for gene in vars_df.index.tolist():
        logging.info(f"Tuning and predicting for {gene} gene.")
        test_metrics = train_and_predict(
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

    pd.DataFrame(output_metrics).to_csv(
        os.path.join(output_dir, f"test_results_{penalty}_{random_state}")
    )


def main(args):
    run(
        output_dir=args.output_dir,
        train_test_split_file_path=args.train_test_split_file_path,
        input_dir=args.variant_matrix_input_dir,
        max_iter=args.max_iter,
        penalty=args.penalty,
        random_state=args.random_state,
        exclude_vars_not_in_train=args.exclude_vars_not_in_train,
    )


if __name__ == "__main__":
    args = OneHotModelArgumentParser().parse_args()
    main(args)
