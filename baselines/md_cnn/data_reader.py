import json
import os
from typing import List

import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.md_cnn.dataset import MDCNNDataset
from deep_bac.data_preprocessing.data_reader import _collate_samples
from deep_bac.data_preprocessing.data_types import DataReaderOutput


def get_mdcnn_dataloader(
    batch_size: int,
    bac_loci_df_file_path: str,
    reference_loci_data_df: DataFrame,
    unique_ids: List[str] = None,
    phenotype_dataframe_file_path: str = None,
    max_gene_length: int = 2048,
    regression: bool = False,
    use_drug_idx: int = None,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.5,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = MDCNNDataset(
        unique_ids=unique_ids,
        bac_loci_df_file_path=bac_loci_df_file_path,
        reference_loci_data_df=reference_loci_data_df,
        phenotype_dataframe_file_path=phenotype_dataframe_file_path,
        max_loci_length=max_gene_length,
        regression=regression,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        use_drug_idx=use_drug_idx,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_samples,
    )
    return dataloader


def get_mdcnn_data(
    input_df_file_path: str,
    reference_loci_data_df_path: str,
    phenotype_df_file_path: str,
    train_val_test_split_indices_file_path: str,
    regression: bool = False,
    use_drug_idx: int = None,
    batch_size: int = 8,
    max_loci_length: int = 10147,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = 8,
    test: bool = False,
):
    reference_loci_data_df = pd.read_parquet(reference_loci_data_df_path)

    with open(train_val_test_split_indices_file_path, "r") as f:
        train_val_test_split_indices = json.load(f)
    train_unique_ids = train_val_test_split_indices["train"]
    val_unique_ids = train_val_test_split_indices["val"]
    test_unique_ids = train_val_test_split_indices["test"]

    train_dataloader = get_mdcnn_dataloader(
        batch_size=batch_size,
        unique_ids=train_unique_ids,
        bac_loci_df_file_path=input_df_file_path,
        reference_loci_data_df=reference_loci_data_df,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        max_gene_length=max_loci_length,
        regression=regression,
        use_drug_idx=use_drug_idx,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = get_mdcnn_dataloader(
        batch_size=batch_size,
        unique_ids=val_unique_ids,
        bac_loci_df_file_path=input_df_file_path,
        reference_loci_data_df=reference_loci_data_df,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        max_gene_length=max_loci_length,
        regression=regression,
        use_drug_idx=use_drug_idx,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if not test:
        return DataReaderOutput(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_set_len=len(train_unique_ids),
        )
    test_dataloader = get_mdcnn_dataloader(
        batch_size=batch_size,
        unique_ids=test_unique_ids,
        bac_loci_df_file_path=input_df_file_path,
        reference_loci_data_df=reference_loci_data_df,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        max_gene_length=max_loci_length,
        regression=regression,
        use_drug_idx=use_drug_idx,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return DataReaderOutput(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        train_set_len=len(train_unique_ids),
    )


def main():
    input_dir = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/processed-genome-per-strain-md-cnn/"
    data = get_mdcnn_data(
        input_df_file_path=os.path.join(
            input_dir, "processed_agg_variants_md_cnn.parquet"
        ),
        reference_loci_data_df_path=os.path.join(
            input_dir, "reference_loci_data.parquet"
        ),
        phenotype_df_file_path=os.path.join(
            input_dir, "phenotype_labels_with_binary_labels.parquet"
        ),
        train_val_test_split_indices_file_path=os.path.join(
            input_dir, "train_val_test_split_unq_ids.json"
        ),
        regression=False,
        use_drug_idx=None,
        batch_size=8,
        max_loci_length=10147,
        shift_max=0,
        pad_value=0.25,
        reverse_complement_prob=0.0,
        num_workers=8,
        test=True,
    )
    for _ in tqdm(data.val_dataloader):
        pass
    print("Val dataloader done")
    for _ in tqdm(data.test_dataloader):
        pass
    print("test dataloader done")

    for _ in tqdm(data.train_dataloader):
        pass
    print("train dataloader done")


if __name__ == "__main__":
    main()
