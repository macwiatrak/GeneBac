from typing import List

import pandas as pd
import torch

from deep_bac.data_preprocessing.data_reader import _collate_samples
from deep_bac.data_preprocessing.data_types import (
    BacInputSample,
    BatchBacInputSample,
)
from deep_bac.data_preprocessing.dataset import transform_dna_seq


def get_ref_batch(
    genes: List[str],
    reference_gene_data_df: pd.DataFrame,
    max_gene_length: int = 2560,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
) -> BacInputSample:
    genes_tensor = []
    tss_pos_genome = []
    for gene in genes:
        seq = reference_gene_data_df.loc[gene]["seq"]
        # process the sequence
        padded_one_hot_seq = transform_dna_seq(
            max_gene_length=max_gene_length,
            shift_max=shift_max,
            pad_value=pad_value,
            reverse_complement_prob=reverse_complement_prob,
            seq=seq,
        )
        # append to genes tensor
        genes_tensor.append(padded_one_hot_seq)
        tss_pos_genome.append(
            reference_gene_data_df.loc[gene]["tss_pos_genome"]
        )
    return BacInputSample(
        input_tensor=torch.stack(genes_tensor),
        tss_index=torch.tensor(tss_pos_genome, dtype=torch.long),
    )


def get_seq_with_variant(
    alt: str,
    ref: str,
    gene: str,
    start_index: int,
    end_index: int,
    prom_seq_len: int = 100,
) -> str:
    """
    Incorporate the variant into a reference gene sequence
    :param alt: the variant
    :param ref: the reference genome nucleotides at a variant site
    :param gene: the reference gene sequence
    :param start_index: the start index of the variant relative to start codon / TSS
    :param end_index: the end index of the variant relative to start codon / TSS
    """
    start_index = int(start_index + prom_seq_len)
    end_index = int(end_index + prom_seq_len)
    ref_gene = gene[start_index:end_index]
    if ref.lower() != ref_gene.lower():
        raise ValueError(
            "Reference genome from variant file does not match reference genome "
            "from reference_gene_data.parquet file"
        )
    seq = gene[:start_index] + alt.lower() + gene[end_index:]
    return seq


def batch_samples_w_variant(
    variant_df: pd.DataFrame,
    genes: List[str],
    reference_gene_data_df: pd.DataFrame,
    batch_size: int = 32,
    max_gene_length: int = 2560,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
) -> List[BatchBacInputSample]:
    output = []
    for idx, row in variant_df.iterrows():
        genes_tensor = []
        tss_pos_genome = []
        for gene in genes:
            if gene.lower() == row["gene"].lower():
                if "prom_gene_seq_w_variants" in row:
                    seq = row["prom_gene_seq_w_variants"]
                elif "alt" in row:
                    seq = get_seq_with_variant(
                        alt=row["alt"],
                        ref=row[
                            "ref"
                        ],  # for double-checking the ref is in-line with reference data
                        gene=reference_gene_data_df.loc[gene]["seq"],
                        start_index=row["start_index"],
                        end_index=row["end_index"],
                    )
                else:
                    raise ValueError(
                        "Variant file must contain either 'prom_gene_seq_w_variants' or 'alt' column"
                    )
            else:
                seq = reference_gene_data_df.loc[gene]["seq"]
            # process the sequence
            padded_one_hot_seq = transform_dna_seq(
                max_gene_length=max_gene_length,
                shift_max=shift_max,
                pad_value=pad_value,
                reverse_complement_prob=reverse_complement_prob,
                seq=seq,
            )
            # append to genes tensor
            genes_tensor.append(padded_one_hot_seq)
            tss_pos_genome.append(
                reference_gene_data_df.loc[gene]["tss_pos_genome"]
            )
        output.append(
            BacInputSample(
                input_tensor=torch.stack(genes_tensor),
                tss_index=torch.tensor(tss_pos_genome, dtype=torch.long),
            )
        )
    batched_data = [
        _collate_samples(output[i : i + batch_size])
        for i in range(0, len(output), batch_size)
    ]
    return batched_data
