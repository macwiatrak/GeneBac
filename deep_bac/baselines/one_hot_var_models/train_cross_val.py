import json
import logging
import os
from typing import Dict, Literal

import numpy as np
import pandas as pd
import torch
from lightning_lite import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.sparse import load_npz
from torch.utils.data import DataLoader, TensorDataset

from deep_bac.baselines.one_hot_var_models.argparser import (
    OneHotModelArgumentParser,
)
from deep_bac.baselines.one_hot_var_models.data_reader import (
    get_train_eval_var_data,
    get_one_hot_data,
)
from deep_bac.baselines.one_hot_var_models.model import LinearModel

from deep_bac.baselines.one_hot_var_models.utils import (
    get_trainer_linear_model,
)
from deep_bac.modelling.metrics import compute_drug_thresholds

logging.basicConfig(level=logging.INFO)

N_FOLDS = 5


def get_tuning_params(penalty: Literal["l1", "l2", "elasticnet"] = "l2"):
    if penalty == "l2":
        return {"C": [0.01, 0.1, 0.5, 1.0, 5.0]}

    if penalty == "l1":
        return {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
    return {
        "l1_ratio": [0.001, 0.01, 0.1, 0.25, 0.5],
        "alpha": [0.001, 0.01, 0.1, 0.25, 0.5],
    }


def run(
    output_dir: str,
    train_test_split_unq_ids_file_path: str,
    variant_matrix_input_dir: str,
    df_unq_ids_labels_file_path: str,
    lr: float = 0.001,
    l1_lambda: float = 0.1,
    l2_lambda: float = 0.05,
    max_epochs: int = 500,
    random_state: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
    exclude_vars_not_in_train: bool = False,
    regression: bool = False,
):
    seed_everything(random_state)

    metrics_list = []
    for fold_idx in range(N_FOLDS):
        data = get_one_hot_data(
            variant_matrix_input_dir=variant_matrix_input_dir,
            df_unq_ids_labels_file_path=df_unq_ids_labels_file_path,
            train_test_split_unq_ids_file_path=train_test_split_unq_ids_file_path,
            fold_idx=fold_idx,
            exclude_vars_not_in_train=exclude_vars_not_in_train,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        logging.info(f"Starting training on fold {fold_idx}")

        model = LinearModel(
            input_dim=data.train_var_matrix.shape[1],
            output_dim=data.train_labels.shape[1],
            lr=lr,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            regression=regression,
        )
        trainer = get_trainer_linear_model(
            output_dir=output_dir, max_epochs=max_epochs
        )
        trainer.fit(
            model,
            train_dataloaders=data.train_dl,
            val_dataloaders=data.val_dl,
        )

        model = model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        model.drug_thresholds = compute_drug_thresholds(
            logits=model(torch.tensor(data.train_var_matrix)),
            labels=data.train_labels,
        )

        fold_metrics = trainer.test(
            model,
            dataloaders=data.val_dl,
        )
        # fold_metrics = trainer.checkpoint_callback.best_model_ep
        metrics_list.append(fold_metrics[0])

    # compute avg metrics across the folds
    avg_metrics = {
        k: np.mean([fold_metrics[k] for fold_metrics in metrics_list])
        for k in metrics_list[0].keys()
    }
    logging.info(
        f"\n\nAvg metrics for linear model with L1 lambda {l1_lambda} and L2 lambda {l2_lambda}: {avg_metrics}"
    )


def main(args):
    run(
        output_dir=args.output_dir,
        train_test_split_unq_ids_file_path=args.train_test_split_unq_ids_file_path,
        variant_matrix_input_dir=args.variant_matrix_input_dir,
        df_unq_ids_labels_file_path=args.df_unq_ids_labels_file_path,
        random_state=args.random_state,
        exclude_vars_not_in_train=args.exclude_vars_not_in_train,
        max_epochs=args.max_epochs,
        lr=args.lr,
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    args = OneHotModelArgumentParser().parse_args()
    main(args)
