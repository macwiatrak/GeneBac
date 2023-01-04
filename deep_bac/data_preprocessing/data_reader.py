import json
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.data_preprocessing.data_types import BacGenesInputSample, BatchBacGenesInputSample
from deep_bac.data_preprocessing.dataset import BacterialGenomeDataset


def _collate_samples(data: List[BacGenesInputSample]) -> BatchBacGenesInputSample:
    genes_tensor = torch.stack([sample.genes_tensor for sample in data])
    variants_in_gene = [sample.variants_in_gene for sample in data]

    if None not in variants_in_gene:
        variants_in_gene = torch.stack(variants_in_gene)

    labels = [sample.labels for sample in data]
    if None not in labels:
        labels = torch.stack(labels)

    unique_ids = [sample.unique_id for sample in data]
    return BatchBacGenesInputSample(
        genes_tensor=genes_tensor,
        variants_in_gene=variants_in_gene,
        labels=labels,
        unique_ids=unique_ids,
    )


def get_dataloader(
    batch_size: int,
    bac_genes_df_file_path: str,
    reference_gene_seqs_dict: Dict[str, str],
    phenotype_dataframe_file_path: str = None,
    max_gene_length: int = 2048,
    selected_genes: List = None,
    regression: bool = False,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.5,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = BacterialGenomeDataset(
        bac_genes_df_file_path=bac_genes_df_file_path,
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        phenotype_dataframe_file_path=phenotype_dataframe_file_path,
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        regression=regression,
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
    return dataloader


def get_data(
    input_df_file_path: str,
    reference_gene_seqs_dict_path: str,
    phenotype_df_file_path: str,
    train_val_test_split_indices_file_path: str,
    selected_genes_file_path: Optional[str] = None,
    batch_size: int = 8,
    max_gene_length: int = 2048,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.5,
    num_workers: int = 8,
):
    with open(reference_gene_seqs_dict_path, 'r') as f:
        reference_gene_seqs_dict = json.load(f)

    if selected_genes_file_path is not None:
        selected_genes = pd.read_csv(selected_genes_file_path)['Gene'].tolist()
    else:
        selected_genes = None

    with open(train_val_test_split_indices_file_path, 'r') as f:
        train_val_test_split_indices = json.load(f)
    train_indices = train_val_test_split_indices['train']
    val_indices = train_val_test_split_indices['val']
    test_indices = train_val_test_split_indices['test']

    train_dataloader = get_dataloader(
        batch_size=batch_size,
        indices=train_indices,
        bac_genes_df_file_path=input_df_file_path,
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_dataloader = get_dataloader(
        batch_size=batch_size,
        indices=val_indices,
        bac_genes_df_file_path=input_df_file_path,
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_dataloader = get_dataloader(
        batch_size=batch_size,
        indices=test_indices,
        bac_genes_df_file_path=input_df_file_path,
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


def main():
    input_dir = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/"
    with open(os.path.join(input_dir, 'reference_gene_seqs.json'), 'r') as f:
        reference_gene_seqs_dict = json.load(f)

    gene_variance_df = pd.read_csv(
        os.path.join(input_dir, 'unnormalised_variance_per_gene.csv'))
    selected_genes = gene_variance_df['Gene'].tolist()[:1000]

    max_gene_length = 2048
    dl = get_dataloader(
        batch_size=32,
        bac_genes_df_file_path=os.path.join(input_dir, "processed-genome-per-strain", "agg_variants.parquet"),
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        phenotype_dataframe_file_path=os.path.join(input_dir, "phenotype_labels_with_binary_labels.parquet"),
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        shift_max=3,
        pad_value=0.25,
        reverse_complement_prob=0.5,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=False,
    )

    for _ in tqdm(dl):
        pass
    print("Done!")


if __name__ == '__main__':
    main()
