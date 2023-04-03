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
    split_train_val_test,
)
from deep_bac.baselines.one_hot_var_models.model import LinearModel

from deep_bac.baselines.one_hot_var_models.utils import (
    DRUG_TO_IDX,
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
    l1_lambda: float = 0.05,
    l2_lambda: float = 0.05,
    max_epochs: int = 1000,
    random_state: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
    exclude_vars_not_in_train: bool = False,
    regression: bool = False,
):
    seed_everything(random_state)
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

    data = split_train_val_test(
        train_test_split_unq_ids_file_path,
        variant_matrix,
        unq_id_to_idx,
        df_unq_ids_labels,
        exclude_vars_not_in_train,
    )

    metrics_list = []
    for fold_idx in range(N_FOLDS):
        logging.info(f"Starting training on fold {fold_idx}")
        train_unique_ids = train_test_split_unq_ids[f"train_fold_{fold_idx}"]
        train_indices = np.array(
            [unq_id_to_idx[unq_id] for unq_id in train_unique_ids]
        )
        train_var_matrix = torch.tensor(data.train_var_matrix[train_indices])
        train_labels = torch.tensor(data.train_labels[train_indices])
        train_dl = DataLoader(
            TensorDataset(train_var_matrix, train_labels),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        val_unique_ids = train_test_split_unq_ids[f"val_fold_{fold_idx}"]
        train_indices = np.array(
            [unq_id_to_idx[unq_id] for unq_id in val_unique_ids]
        )
        val_var_matrix = torch.tensor(data.train_var_matrix[train_indices])
        val_labels = torch.tensor(data.train_labels[train_indices])
        val_dl = DataLoader(
            TensorDataset(val_var_matrix, val_labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model = LinearModel(
            input_dim=train_var_matrix.shape[1],
            output_dim=train_labels.shape[1],
            lr=lr,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            regression=regression,
        )
        trainer = Trainer(
            max_epochs=max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor="train_gmean_spec_sens",
                    patience=20,
                    mode="max",
                ),
                ModelCheckpoint(
                    dirpath=output_dir,
                    filename="{epoch:02d}-{train_gmean_spec_sens:.4f}",
                    monitor="train_gmean_spec_sens",
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                ),
                TQDMProgressBar(refresh_rate=100),
            ],
            logger=TensorBoardLogger(output_dir),
        )
        trainer.fit(
            model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
        )

        model = model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        model.drug_thresholds = compute_drug_thresholds(
            logits=model(torch.tensor(train_var_matrix)),
            labels=train_labels,
        )

        fold_metrics = trainer.test(
            model,
            dataloaders=val_dl,
        )
        metrics_list.append(fold_metrics)
    avg_metrics = {
        k: np.mean([fold_metrics[k] for fold_metrics in metrics_list])
        for k in metrics_list[0].keys()
    }
    logging.info(
        f"Avg metrics for linear model with L1 lambda {l1_lambda} and L2 lambda {l2_lambda}: {avg_metrics}"
    )


def main(args):
    run(
        output_dir=args.output_dir,
        train_test_split_unq_ids_file_path=args.train_test_split_unq_ids_file_path,
        variant_matrix_input_dir=args.variant_matrix_input_dir,
        df_unq_ids_labels_file_path=args.df_unq_ids_labels_file_path,
        random_state=args.random_state,
        exclude_vars_not_in_train=args.exclude_vars_not_in_train,
    )


if __name__ == "__main__":
    args = OneHotModelArgumentParser().parse_args()
    main(args)
