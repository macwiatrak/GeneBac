import json
import os
from typing import Dict, List

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.data_preprocessing.data_readers.utils import _collate_samples
from deep_bac.data_preprocessing.data_types import DataReaderOutput
from deep_bac.data_preprocessing.datasets.gene_reg_proteins import (
    ProteinGeneRegDataset,
)


def get_gene_reg_prot_dataloader(
    batch_size: int,
    bac_genes_df_file_path: str,
    reference_gene_seqs_dict: Dict[str, str],
    unique_ids: List[str] = None,
    phenotype_dataframe_file_path: str = None,
    selected_genes: List = None,
    regression: bool = False,
    use_drug_idx: int = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = ProteinGeneRegDataset(
        unique_ids=unique_ids,
        bac_genes_df_file_path=bac_genes_df_file_path,
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        phenotype_dataframe_file_path=phenotype_dataframe_file_path,
        selected_genes=selected_genes,
        regression=regression,
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


def get_gene_reg_prot_data(
    input_df_file_path: str,
    reference_gene_seqs_dict_path: str,
    phenotype_df_file_path: str,
    train_val_test_split_indices_file_path: str,
    variance_per_gene_file_path: str,
    n_highly_variable_genes: int = 500,
    selected_genes: List = None,
    regression: bool = False,
    use_drug_idx: int = None,
    batch_size: int = 8,
    num_workers: int = 8,
    test: bool = False,
):
    with open(reference_gene_seqs_dict_path, "r") as f:
        reference_gene_seqs_dict = json.load(f)

    if selected_genes is None:
        selected_genes = pd.read_csv(variance_per_gene_file_path)[
            "Gene"
        ].tolist()[:n_highly_variable_genes]

    with open(train_val_test_split_indices_file_path, "r") as f:
        train_val_test_split_indices = json.load(f)
    train_unique_ids = train_val_test_split_indices["train"]
    val_unique_ids = train_val_test_split_indices["val"]
    test_unique_ids = train_val_test_split_indices["test"]

    train_dataloader = get_gene_reg_prot_dataloader(
        batch_size=batch_size,
        unique_ids=train_unique_ids,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        bac_genes_df_file_path=input_df_file_path,
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        selected_genes=selected_genes,
        regression=regression,
        use_drug_idx=use_drug_idx,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = get_gene_reg_prot_dataloader(
        batch_size=batch_size,
        unique_ids=val_unique_ids,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        bac_genes_df_file_path=input_df_file_path,
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        selected_genes=selected_genes,
        regression=regression,
        use_drug_idx=use_drug_idx,
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
    test_dataloader = get_gene_reg_prot_dataloader(
        batch_size=batch_size,
        unique_ids=test_unique_ids,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        bac_genes_df_file_path=input_df_file_path,
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        selected_genes=selected_genes,
        regression=regression,
        use_drug_idx=use_drug_idx,
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
    input_dir = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/"
    with open(os.path.join(input_dir, "reference_gene_seqs.json"), "r") as f:
        reference_gene_seqs_dict = json.load(f)

    gene_variance_df = pd.read_csv(
        os.path.join(input_dir, "unnormalised_variance_per_gene.csv")
    )
    selected_genes = gene_variance_df["Gene"].tolist()[:1000]

    max_gene_length = 2048
    dl = get_gene_reg_prot_dataloader(
        batch_size=32,
        unique_ids=None,
        bac_genes_df_file_path=os.path.join(
            input_dir, "processed-genome-per-strain", "agg_variants.parquet"
        ),
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        phenotype_dataframe_file_path=os.path.join(
            input_dir, "phenotype_labels_with_binary_labels.parquet"
        ),
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


if __name__ == "__main__":
    main()
