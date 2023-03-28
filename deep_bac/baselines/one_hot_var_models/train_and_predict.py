import logging
from typing import Dict, Literal

import pandas as pd

from deep_bac.baselines.one_hot_var_models.argparser import (
    OneHotModelArgumentParser,
)
from deep_bac.baselines.one_hot_var_models.train_and_predict_on_drug import (
    train_and_predict,
)
from deep_bac.baselines.one_hot_var_models.utils import (
    dict_metrics_to_df,
    DRUG_TO_IDX,
)

logging.basicConfig(level=logging.INFO)


def run(
    output_dir: str,
    drug_to_idx: Dict[str, int],
    train_test_split_unq_ids_file_path: str,
    variant_matrix_input_dir: str,
    df_unq_ids_labels_file_path: str,
    max_iter: int = 1000,
    penalty: Literal["l1", "l2", "elasticnet"] = "l2",
    random_state: int = 42,
    exclude_vars_not_in_train: bool = False,
):
    df_unq_ids_labels = pd.read_parquet(df_unq_ids_labels_file_path)
    params = {
        "C": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "class_weight": [None, "balanced"],
    }
    # add l1 ratio if using elasticnet
    if penalty == "elasticnet":
        params["l1_ratio"] = [0.0, 0.25, 0.5, 0.75, 1.0]

    output_dfs = []
    for drug, drug_idx in drug_to_idx.items():
        logging.info(f"Tuning and predicting for {drug} drug")
        test_metrics = train_and_predict(
            drug_idx=drug_idx,
            train_test_split_unq_ids_file_path=train_test_split_unq_ids_file_path,
            variant_matrix_input_dir=variant_matrix_input_dir,
            df_unq_ids_labels=df_unq_ids_labels,
            params=params,
            max_iter=max_iter,
            random_state=random_state,
            penalty=penalty,
            exclude_vars_not_in_train=exclude_vars_not_in_train,
        )
        logging.info(f"Test metrics for {drug} drug: {test_metrics}")
        output_dfs.append(dict_metrics_to_df(test_metrics, drug, split="test"))
    pd.concat(output_dfs).to_csv(
        f"{output_dir}/test_results_{penalty}_{random_state}.csv"
    )


def main(args):
    run(
        output_dir=args.output_dir,
        drug_to_idx=DRUG_TO_IDX,
        train_test_split_unq_ids_file_path=args.train_test_split_unq_ids_file_path,
        variant_matrix_input_dir=args.variant_matrix_input_dir,
        df_unq_ids_labels_file_path=args.df_unq_ids_labels_file_path,
        max_iter=args.max_iter,
        penalty=args.penalty,
        random_state=args.random_state,
        exclude_vars_not_in_train=args.exclude_vars_not_in_train,
    )


if __name__ == "__main__":
    args = OneHotModelArgumentParser().parse_args()
    main(args)
