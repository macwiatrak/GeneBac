from typing import List, Tuple

import pandas as pd
import torch

from deep_bac.data_preprocessing.data_reader import _collate_samples
from deep_bac.data_preprocessing.data_types import (
    BacInputSample,
    BatchBacInputSample,
)
from deep_bac.data_preprocessing.dataset import transform_dna_seq
from deep_bac.modelling.metrics import MTB_DRUG_TO_LABEL_IDX
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno


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
    gene: str,
    start_index: int,
    end_index: int,
    alt: str,
    ref: str = None,
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
    if ref is not None:
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
    prom_seq_len: int = 100,
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
                        ref=row.get(
                            "ref", None
                        ),  # for double-checking the ref is in-line with reference data
                        gene=reference_gene_data_df.loc[gene]["seq"],
                        start_index=row["start_index"],
                        end_index=row["end_index"],
                        prom_seq_len=prom_seq_len,
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


def compute_variant_effect_size(
    model: DeepBacGenePheno,
    reference_gene_data_df: pd.DataFrame,
    gene: str,
    variant: str,
    start_idx: int,
    end_idx: int,
    drug: str = None,  # if None, will return all drug scores
    prom_seq_len: int = 100,
):
    # get genes the model was trained on
    available_genes = list(model.config.gene_to_idx.keys())
    if gene not in available_genes:
        raise ValueError(
            f"Gene {gene} not available in model."
            f"Available genes are: {available_genes}"
        )

    ref_input_sample = get_ref_batch(
        genes=list(model.config.gene_to_idx.keys()),
        reference_gene_data_df=reference_gene_data_df,
        max_gene_length=model.config.max_gene_length,
        shift_max=0,
        pad_value=0.25,
        reverse_complement_prob=0.0,
    )

    # get ref variant scores
    with torch.no_grad():
        ref_scores = model(
            ref_input_sample.input_tensor.unsqueeze(0),
            ref_input_sample.tss_index.unsqueeze(0),
        )[0]
        ref_scores = (
            torch.sigmoid(ref_scores)
            if not model.config.regression
            else ref_scores
        )
    # get variant scores
    variants_df = pd.DataFrame(
        dict(
            gene=[gene],
            start_index=[start_idx],
            end_index=[end_idx],
            alt=[variant],
            drug=[drug],
        )
    )
    # batch data
    sample = batch_samples_w_variant(
        variant_df=variants_df,
        genes=list(model.config.gene_to_idx.keys()),
        reference_gene_data_df=reference_gene_data_df,
        max_gene_length=model.config.max_gene_length,
        shift_max=0,
        pad_value=0.25,
        reverse_complement_prob=0.0,
        batch_size=1,
        prom_seq_len=prom_seq_len,
    )[0]

    # get scores

    with torch.no_grad():
        scores = model(sample.input_tensor, sample.tss_indexes)
        scores = (
            torch.sigmoid(scores) if not model.config.regression else scores
        )
        scores -= ref_scores

    if drug is not None:
        if drug not in MTB_DRUG_TO_LABEL_IDX:
            raise ValueError(
                f"Drug {drug} not available. The model was not trained on this drug."
                f"Available drugs are: {list(MTB_DRUG_TO_LABEL_IDX.keys())}"
            )

        drug_idx = MTB_DRUG_TO_LABEL_IDX[drug]
        scores = scores[:, drug_idx]
    return scores
