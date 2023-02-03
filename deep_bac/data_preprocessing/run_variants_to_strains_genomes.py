import os
from typing import Set

import pandas as pd
from tqdm import tqdm
import logging

from deep_bac.data_preprocessing.utils import (
    read_ref_genome,
    read_genome_assembly,
    get_gene_data_dict,
)
from deep_bac.data_preprocessing.variants_to_strains_genomes import (
    get_and_filter_variants_df,
    apply_variants_to_a_gene,
)

REF_GENOME_FILE_NAME = "H37Rv_NC_000962.3_ref_genome.fna"
GENOME_ASSEMBLY_FILE_NAME = os.path.join("genome_assembly", "genomic.gff")
PHENOTYPE_FILE_NAME = "UKMYC_PHENOTYPES.csv.gz"
VARIANTS_FILE_NAME = "VARIANTS.csv.gz"

logging.basicConfig(level=logging.INFO)
tqdm.pandas()


def get_strain_w_phenotype_ids(file_path: str) -> Set[str]:
    df = pd.read_csv(file_path, compression="gzip")
    # filter low quality phenotypes
    df = df[~(df["PHENOTYPE_QUALITY"] == "LOW")]
    return set(df["UNIQUEID"].tolist())


def agg_variants(variants_df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        variants_df.sort_values(by=["NUCLEOTIDE_NUMBER"])
        .groupby(["UNIQUEID", "GENE"])
        .agg(list)
    )
    # add gene column with gene names
    genes = [i[-1] for i in agg_df.index]
    agg_df["gene"] = genes
    return agg_df


def postprocess_agg_variants(agg_variants_df: pd.DataFrame) -> pd.DataFrame:
    # remove redundant columns
    agg_variants_df = agg_variants_df.drop(
        columns=["REF", "ALT", "NUCLEOTIDE_NUMBER", "IS_INDEL"]
    )
    # clean up the columns
    agg_variants_df["prom_gene_seq_w_variants"] = agg_variants_df[
        "features"
    ].progress_apply(lambda x: x["prom_gene_seq_w_variants"])

    agg_variants_df["ref_correct_ratio"] = agg_variants_df[
        "features"
    ].progress_apply(lambda x: x["ref_correct_ratio"])

    agg_variants_df["n_nucleotide_change"] = agg_variants_df[
        "features"
    ].progress_apply(lambda x: x["n_nucleotide_change"])
    # remove the features column as we don't need it anymore
    agg_variants_df = agg_variants_df.drop(columns=["features"])
    return agg_variants_df


def run(
    input_dir: str,
    output_dir: str,
    prom_seq_len: int,
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
        prom_seq_len=prom_seq_len,
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
    )
    logging.info("Finished reading variants")

    agg_variants_df = agg_variants(variants_df=variants_df)
    del variants_df  # remove it to save memory
    logging.info("Finished aggregating variants")

    # apply variants
    agg_variants_df["features"] = agg_variants_df.progress_apply(
        lambda row: apply_variants_to_a_gene(
            ref_gene_seq=gene_data_dict[row["gene"]]["gene_seq"],
            ref_prom_seq=gene_data_dict[row["gene"]]["promoter_seq"],
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
        f"Mean ref correct ratio: {mean_ref_correct_ratio}\n"
        f"Std ref correct ratio: {std_ref_correct_ratio}"
    )

    # save agg variants
    logging.info("Saving agg variants")
    agg_variants_df.to_parquet(os.path.join(output_dir, "agg_variants.parquet"))


def main():
    run(
        input_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/",
        output_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/processed-genome-per-strain-1/",
        prom_seq_len=100,
    )


if __name__ == "__main__":
    main()
