from typing import Literal

import pandas as pd

from deep_bac.experiments.variant_scoring.variant_scoring import (
    get_ref_batch,
    batch_samples_w_variant,
)
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno
from deep_bac.utils import get_selected_genes


def run(
    ckpt_path: str,
    output_dir: str,
    variant_df_path: str,
    reference_gene_data_df_path: str,
    max_gene_length: int = 2560,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
):
    variant_df_path = pd.read_parquet(variant_df_path)
    reference_gene_data_df = pd.read_parquet(
        reference_gene_data_df_path
    ).set_index("gene")
    model = DeepBacGenePheno.load_from_checkpoint(ckpt_path)
    gene_to_idx = model.config.gene_to_idx

    ref_input_sample = get_ref_batch(
        genes=list(gene_to_idx.keys()),
        reference_gene_data_df=reference_gene_data_df,
        max_gene_length=max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
    )

    model = DeepBacGenePheno.load_from_checkpoint(ckpt_path)
    model.eval()

    # get ref variant scores
    ref_scores = model(
        ref_input_sample.input_tensor, ref_input_sample.tss_index
    )

    # get variant scores
    # batch data
    batches = batch_samples_w_variant(
        variant_df=variant_df_path,
        genes=list(gene_to_idx.keys()),
        reference_gene_data_df=reference_gene_data_df,
        max_gene_length=max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
    )

    var_scores = []
    for batch in batches:
        scores = model(batch.input_tensor, batch.tss_index)
        var_scores.append(scores)
