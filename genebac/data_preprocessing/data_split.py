import itertools
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification


INPUT_DIR = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data/"


def run(
    df_strain_ids_with_labels: pd.DataFrame,
    test_size: float = 0.2,
    n_folds: int = 5,
) -> Dict[str, List[str]]:
    drug_labels = np.stack(df_strain_ids_with_labels["BINARY_LABELS"].tolist())
    strain_ids = np.array(df_strain_ids_with_labels.index.tolist())
    strain_ids = np.reshape(strain_ids, (-1, 1))

    (
        strain_ids_train,
        drug_labels_train,
        strain_ids_test,
        drug_labels_test,
    ) = iterative_train_test_split(strain_ids, drug_labels, test_size=test_size)

    train_val_test_indices = {
        "train": list(itertools.chain(*strain_ids_train.tolist())),
        "test": list(itertools.chain(*strain_ids_test.tolist())),
    }

    k_fold = IterativeStratification(n_splits=n_folds, order=1)
    for idx, (train, test) in enumerate(
        k_fold.split(strain_ids_train, drug_labels_train)
    ):
        fold_train = list(itertools.chain(*strain_ids_train[train].tolist()))
        fold_val = list(itertools.chain(*strain_ids_train[test].tolist()))
        print(
            f"N items in train fold: {len(fold_train)}, val fold: {len(fold_val)}"
        )
        assert len(set(fold_train + fold_val)) == len(
            set(train_val_test_indices["train"])
        )

        train_val_test_indices[f"train_fold_{idx}"] = fold_train
        train_val_test_indices[f"val_fold_{idx}"] = fold_val

    return train_val_test_indices


def main():
    seed_everything(42)
    train_val_test_indices = run(
        df_strain_ids_with_labels=pd.read_parquet(
            INPUT_DIR, "phenotype_labels_with_binary_labels.parquet"
        ),
        test_size=0.2,
        n_folds=5,
    )
    with open(
        os.path.join(INPUT_DIR, "train_test_cv_split_unq_ids.json"), "w"
    ) as f:
        json.dump(train_val_test_indices, f)
