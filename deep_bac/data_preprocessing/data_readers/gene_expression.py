import os
from typing import Tuple

from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.data_preprocessing.data_readers.utils import _collate_samples
from deep_bac.data_preprocessing.data_types import (
    DataReaderOutput,
)
from deep_bac.data_preprocessing.datasets.gene_expression import GeneExprDataset


def get_gene_expr_dataloader(
    batch_size: int,
    bac_genes_df_file_path: str,
    max_gene_length: int = 2048,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.5,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, int]:
    dataset = GeneExprDataset(
        bac_genes_df_file_path=bac_genes_df_file_path,
        max_gene_length=max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_samples,
    )
    return dataloader, dataset.__len__()


def get_gene_expr_data(
    input_dir: str,
    batch_size: int = 512,
    max_gene_length: int = 2048,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.5,
    num_workers: int = 8,
    test: bool = False,
):

    train_dataloader, train_set_len = get_gene_expr_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path=os.path.join(input_dir, "train.parquet"),
        max_gene_length=max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader, _ = get_gene_expr_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path=os.path.join(input_dir, "val.parquet"),
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
            train_set_len=train_set_len,
        )
    test_dataloader, _ = get_gene_expr_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path=os.path.join(input_dir, "test.parquet"),
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
        train_set_len=train_set_len,
    )


def main():
    data = get_gene_expr_data(
        input_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/pseudomonas/",
        batch_size=512,
        max_gene_length=2048,
        shift_max=3,
        pad_value=0.25,
        reverse_complement_prob=0.5,
        num_workers=8,
        test=False,
    )
    for _ in tqdm(data.val_dataloader):
        pass
    print("Val done!")
    for _ in tqdm(data.train_dataloader):
        pass
    print("Train done!")


if __name__ == "__main__":
    main()
