import json
import os
from typing import Dict, Optional, List

import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm
import logging

from deep_bac.data_preprocessing.run_variants_to_strains_genomes import (
    REF_GENOME_FILE_NAME,
    GENOME_ASSEMBLY_FILE_NAME,
    get_strain_w_phenotype_ids,
    PHENOTYPE_FILE_NAME,
    VARIANTS_FILE_NAME,
    agg_variants,
)
from deep_bac.data_preprocessing.utils import (
    read_ref_genome,
    read_genome_assembly,
    get_gene_data_dict,
    get_complement,
)
from deep_bac.data_preprocessing.variants_to_strains_genomes import (
    get_and_filter_variants_df,
    retrieve_variant_idxs,
)

logging.basicConfig(level=logging.INFO)
tqdm.pandas()


def apply_variants_to_a_gene(
    ref_gene_seq: str,
    strand: str,
    row: Dict[str, Optional[List]],
) -> Dict[str, float]:
    gene_seq_w_variants = ""

    len_ref_seq = len(ref_gene_seq)
    curr_index = 0
    len_change = 0  # for sanity assertions
    ref_correct = 0

    for ref, alt, nucleotide_nr, is_indel in zip(
        row["REF"], row["ALT"], row["NUCLEOTIDE_NUMBER"], row["IS_INDEL"]
    ):
        # get len of the ref to see where we stopped
        len_variant = len(ref)
        # get this for sanity assertions later
        len_change += len(alt) - len(ref)

        # get complement and reverse if strand is negative and is indel
        if strand == "-" and is_indel:
            alt = get_complement(alt)[::-1]

        vstart_idx, vend_idx = retrieve_variant_idxs(
            nucleotide_nr=int(nucleotide_nr),
            ref_len=len_variant,
            strand=strand,
            is_indel=is_indel,
        )

        vstart_idx = max(0, vstart_idx)
        vend_idx = vend_idx if vend_idx >= 0 else len_ref_seq

        gene_seq_w_variants += ref_gene_seq[curr_index:vstart_idx]
        gene_seq_w_variants += alt
        curr_index = vend_idx
        if ref == ref_gene_seq[vstart_idx:vend_idx]:
            ref_correct += 1

    gene_seq_w_variants += ref_gene_seq[curr_index:]
    ref_correct_ratio = ref_correct / len(row["REF"])
    gene_seq_w_variants = (
        gene_seq_w_variants.lower()
        .replace("o", "n")
        .replace("x", "")
        .replace("z", "n")
    )
    amino_acids_seq = Seq(gene_seq_w_variants).translate(
        to_stop=True, table="Bacterial"
    )

    return dict(
        amino_acids_seq_w_variants=str(amino_acids_seq),
        ref_correct_ratio=ref_correct_ratio,
    )


def postprocess_agg_variants(agg_variants_df: pd.DataFrame) -> pd.DataFrame:
    # remove redundant columns
    agg_variants_df = agg_variants_df.drop(
        columns=["REF", "ALT", "NUCLEOTIDE_NUMBER", "IS_INDEL"]
    )
    # clean up the columns
    agg_variants_df["amino_acids_seq_w_variants"] = agg_variants_df[
        "features"
    ].progress_apply(lambda x: x["amino_acids_seq_w_variants"])
    agg_variants_df["ref_correct_ratio"] = agg_variants_df[
        "features"
    ].progress_apply(lambda x: x["ref_correct_ratio"])
    # remove the features column as we don't need it anymore
    agg_variants_df = agg_variants_df.drop(columns=["features"])
    return agg_variants_df


def run(
    input_dir: str,
    output_dir: str,
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # read ref genome
    ref_genome = read_ref_genome(os.path.join(input_dir, REF_GENOME_FILE_NAME))

    # read genome assembly
    genome_assembly = read_genome_assembly(
        os.path.join(input_dir, GENOME_ASSEMBLY_FILE_NAME)
    )

    # get gene data dict with ref gene and promoter seqs for each gene
    # as well as info about the gene's strand
    gene_data_dict = get_gene_data_dict(
        ref_genome=ref_genome,
        genome_assembly=genome_assembly,
        prom_seq_len=0,
    )

    # get unique ids to use
    strain_w_phenotype_ids = get_strain_w_phenotype_ids(
        os.path.join(input_dir, PHENOTYPE_FILE_NAME)
    )

    logging.info("Reading variants")
    # get variants
    variants_df = get_and_filter_variants_df(
        file_path=os.path.join(input_dir, VARIANTS_FILE_NAME),
        unique_ids_to_use=strain_w_phenotype_ids,
        use_cds=True,
    )
    logging.info("Finished reading variants")

    agg_variants_df = agg_variants(variants_df=variants_df)
    del variants_df  # remove it to save memory
    logging.info("Finished aggregating variants")

    # apply variants
    agg_variants_df["features"] = agg_variants_df.progress_apply(
        lambda row: apply_variants_to_a_gene(
            ref_gene_seq=gene_data_dict[row["gene"]]["gene_seq"],
            strand=gene_data_dict[row["gene"]]["strand"],
            row=row,
        ),
        axis=1,
    )
    logging.info("Finished applying variants")

    agg_variants_df = postprocess_agg_variants(agg_variants_df)

    logging.info("Getting statistics about ref correct ratio")
    # logging some basic statistics about applied variants
    mean_ref_correct_ratio = agg_variants_df["ref_correct_ratio"].mean()
    std_ref_correct_ratio = agg_variants_df["ref_correct_ratio"].std()
    logging.info(
        f"Mean ref correct ratio: {mean_ref_correct_ratio}"
        f"Std ref correct ratio: {std_ref_correct_ratio}"
    )

    # save agg variants
    logging.info("Saving agg variants")
    agg_variants_df.to_parquet(os.path.join(output_dir, "agg_variants.parquet"))


def main():
    run(
        input_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/",
        output_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/processed-amino-acids-per-strain/",
    )


if __name__ == "__main__":
    main()
