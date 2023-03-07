import logging
import os
from collections import defaultdict
from functools import partial
from typing import Dict, Tuple, List, Set, Union, Optional

import pandas as pd

from baselines.md_cnn.utils import MD_CNN_GENOMIC_LOCI
from deep_bac.data_preprocessing.run_variants_to_strains_genomes import (
    REF_GENOME_FILE_NAME,
    get_strain_w_phenotype_ids,
    PHENOTYPE_FILE_NAME,
    VARIANTS_FILE_NAME,
)
from deep_bac.data_preprocessing.utils import (
    read_ref_genome,
    get_complement,
)
from deep_bac.data_preprocessing.variants_to_strains_genomes import (
    VARIANTS_COLS_TO_USE,
    retrieve_variant_idxs,
)


def get_genomic_loci_dict(
    ref_genome: str,
    genomic_loci: Dict[str, Tuple[int, int]],
) -> Dict[str, Dict]:
    out = defaultdict(dict)
    for loci_name, (start, end, strand) in genomic_loci.items():
        out[loci_name] = {
            "ref_seq": ref_genome[start - 1 : end - 1],
            "strand": strand,
            "len_seq": end - start,
            "start": start,
            "end": end,
        }
    return out


def var_in_loci(loci_dict: Dict, genome_index: int) -> Union[str, None]:
    for loci_name, (start, end, _) in loci_dict.items():
        if start <= genome_index < end:
            return loci_name
    return None


def get_loci_nucleotide_nr(row: Dict):
    loci = row["loci"]
    genome_index = row["GENOME_INDEX"]
    strand = MD_CNN_GENOMIC_LOCI[loci][2]
    if strand == "+":
        # add 1 for compatibility with 1-indexing
        return genome_index - MD_CNN_GENOMIC_LOCI[loci][0] + 1
    # add 1 for compatibility with 1-indexing
    return MD_CNN_GENOMIC_LOCI[loci][1] - genome_index + 1


def get_and_filter_variants_to_loci_df(
    file_path: str,
    unique_ids_to_use: Set[str],
    cols_to_use: List[str] = VARIANTS_COLS_TO_USE,
) -> pd.DataFrame:
    # filter cols
    df = pd.read_csv(
        file_path,
        usecols=cols_to_use,
        compression="gzip",
        error_bad_lines=False,
    )
    # filter unique ids
    df = df[df["UNIQUEID"].isin(unique_ids_to_use)]

    # only look at variants within specified loci
    var_in_loci_fn = partial(var_in_loci, MD_CNN_GENOMIC_LOCI)
    df["loci"] = df["GENOME_INDEX"].progress_apply(var_in_loci_fn)
    df = df[~df["loci"].isna()]

    df["loci_nucleotide_nr"] = df.progress_apply(
        lambda row: get_loci_nucleotide_nr(row), axis=1
    )
    return df


def agg_variants(variants_df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        variants_df.sort_values(by=["loci_nucleotide_nr"])
        .groupby(["UNIQUEID", "loci"])
        .agg(list)
    )
    # add gene column with gene names
    locis = [i[-1] for i in agg_df.index]
    agg_df["loci"] = locis
    return agg_df


def apply_variants_to_a_loci(
    ref_seq: str,
    strand: str,
    row: Dict[str, Optional[List]],
) -> Dict:
    seq_w_variants = ""

    len_ref_seq = len(ref_seq)
    curr_index = 0
    len_change = 0  # for sanity assertions
    ref_correct = 0
    n_nucleotide_change = 0

    for ref, alt, nucleotide_nr, is_indel in zip(
        row["REF"], row["ALT"], row["loci_nucleotide_nr"], row["IS_INDEL"]
    ):
        # get len of the ref to see where we stopped
        len_variant = len(ref)
        n_nucleotide_change += len_variant
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
        seq_w_variants += ref_seq[curr_index:vstart_idx]
        seq_w_variants += alt
        curr_index = vend_idx
        if ref == ref_seq[vstart_idx:vend_idx]:
            ref_correct += 1

    seq_w_variants += ref_seq[curr_index:]

    ref_correct_ratio = ref_correct / len(row["REF"])
    return dict(
        seq_w_variants=seq_w_variants,
        ref_correct_ratio=ref_correct_ratio,
        n_nucleotide_change=n_nucleotide_change,
    )


def postprocess_agg_variants(agg_variants_df: pd.DataFrame) -> pd.DataFrame:
    # remove redundant columns
    agg_variants_df = agg_variants_df.drop(
        columns=["REF", "ALT", "GENOME_INDEX", "IS_INDEL", "loci_nucleotide_nr"]
    )
    # clean up the columns
    agg_variants_df["seq_w_variants"] = agg_variants_df[
        "features"
    ].progress_apply(lambda x: x["seq_w_variants"])

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
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # read ref genome
    ref_genome = read_ref_genome(os.path.join(input_dir, REF_GENOME_FILE_NAME))

    # get unique ids to use
    strain_w_phenotype_ids = get_strain_w_phenotype_ids(
        os.path.join(input_dir, PHENOTYPE_FILE_NAME)
    )

    genomic_loci_dict = get_genomic_loci_dict(
        ref_genome=ref_genome,
        genomic_loci=MD_CNN_GENOMIC_LOCI,
    )

    logging.info("Reading variants")
    # get variants
    variants_df = get_and_filter_variants_to_loci_df(
        file_path=os.path.join(input_dir, VARIANTS_FILE_NAME),
        unique_ids_to_use=strain_w_phenotype_ids,
    )
    logging.info("Finished reading variants")

    agg_variants_df = agg_variants(variants_df=variants_df)
    del variants_df  # remove it to save memory
    logging.info("Finished aggregating variants")

    # apply variants
    agg_variants_df["features"] = agg_variants_df.progress_apply(
        lambda row: apply_variants_to_a_loci(
            ref_seq=genomic_loci_dict[row["loci"]]["ref_seq"],
            strand=genomic_loci_dict[row["loci"]]["strand"],
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
    agg_variants_df.to_parquet(
        os.path.join(output_dir, "agg_variants_md_cnn.parquet")
    )
