import json
import os
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.data_preprocessing.data_types import BacGenesInputSample, BatchBacGenesInputSample
from deep_bac.data_preprocessing.dataset import BacterialGenomeDataset


def _collate_samples(data: List[BacGenesInputSample]) -> BatchBacGenesInputSample:
    genes_tensor = torch.stack([sample.genes_tensor for sample in data])
    variants_in_gene = torch.stack([sample.variants_in_gene for sample in data])

    labels = [sample.labels for sample in data]
    if None not in labels:
        labels = torch.stack([sample.labels for sample in data])
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
        phenotype_dataframe_file_path=os.path.join(input_dir, "phenotype_labels.parquet"),
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        shift_max=3,
        pad_value=0.25,
        reverse_complement_prob=0.5,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    for _ in tqdm(dl):
        pass
    print("Done!")


if __name__ == '__main__':
    main()
