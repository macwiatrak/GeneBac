import logging
import os
from collections import defaultdict
from functools import partial
from typing import Dict, Tuple, List, Set, Union

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
)
from deep_bac.data_preprocessing.variants_to_strains_genomes import (
    VARIANTS_COLS_TO_USE,
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
        }
    return out


def var_in_loci(loci_dict: Dict, genome_index: int) -> Union[str, None]:
    for loci_name, (start, end, _) in loci_dict.items():
        if start <= genome_index < end:
            return loci_name
    return None


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
    df["loci"] = df["GENOME_INDEX"].apply(var_in_loci_fn)
    df = df[~df["loci"].isna()]

    # add strand info
    df["strand"] = df["loci"].apply(lambda x: MD_CNN_GENOMIC_LOCI[x][2])
    return df


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
