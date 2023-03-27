import json
import logging
import os
from collections import defaultdict
from typing import Literal, Dict, List, Tuple, Set

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix, save_npz

from deep_bac.data_preprocessing.run_variants_to_strains_genomes import (
    VARIANTS_FILE_NAME,
    get_strain_w_phenotype_ids,
    PHENOTYPE_FILE_NAME,
)
from deep_bac.utils import get_selected_genes

logging.basicConfig(level=logging.INFO)


VARIANTS_COLS_TO_USE = [
    "UNIQUEID",
    "GENOME_INDEX",
    "ALT",
    "GENE",
]


def get_and_filter_variants_df(
    file_path: str,
    unique_ids_to_use: Set[str],
    genes_to_use: Set[str] = None,
    cols_to_use: List[str] = VARIANTS_COLS_TO_USE,
) -> pd.DataFrame:
    # filter cols
    df = pd.read_csv(
        file_path,
        usecols=cols_to_use,
        compression="gzip",
        on_bad_lines=False,
    )
    # filter unique ids
    df = df[df["UNIQUEID"].isin(unique_ids_to_use)]
    if not genes_to_use:
        return df
    df = df[df["GENE"].isin(genes_to_use)]
    return df


def get_unqid_var_dict(df: DataFrame) -> Dict[str, List[Tuple[str, int, str]]]:
    out = defaultdict(list)
    for unqid, row in df.iterrows():
        for alt, gi, gene in zip(row["ALT"], row["GENOME_INDEX"], row["GENE"]):
            out[unqid].append((alt, gi, gene))
    return out


def get_var_to_idx(
    unq_id_var_dict: Dict[str, List[Tuple[str, int, str]]]
) -> Dict[Tuple[str, int, str], int]:
    out = []
    for _, items in unq_id_var_dict.items():
        out += items
    return {var: idx for idx, var in enumerate(set(out))}


def get_unqid_var_matrix(
    unqid_var_dict: Dict[str, List[Tuple[str, int, str]]],
    var_to_idx: Dict[Tuple[str, int, str], int],
):
    var_matrix = csr_matrix(
        (len(unqid_var_dict), len(var_to_idx)), dtype=np.int8
    ).tolil()
    for unqid_idx, (_, variants) in enumerate(unqid_var_dict.items()):
        for v in variants:
            var_matrix[unqid_idx, var_to_idx[v]] = 1
    var_matrix = var_matrix.tocsr()
    return var_matrix


def vars_to_matrix(df: DataFrame):
    unqid_var_dict = get_unqid_var_dict(df)
    var_to_idx = get_var_to_idx(unqid_var_dict)
    var_matrix = get_unqid_var_matrix(unqid_var_dict, var_to_idx)
    return var_matrix


def run(
    input_dir: str,
    output_dir: str,
    use_drug_specific_genes: Literal[
        "INH", "Walker", "MD-CNN", "cryptic"
    ] = None,
    train_test_split_unq_ids_file_path: str = None,
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # get unique ids to use
    strain_w_phenotype_ids = get_strain_w_phenotype_ids(
        os.path.join(input_dir, PHENOTYPE_FILE_NAME)
    )

    logging.info("Reading variants")
    # get variants
    variants_df = get_and_filter_variants_df(
        file_path=os.path.join(
            input_dir, VARIANTS_FILE_NAME
        ),  # "VARIANTS_SAMPLE.csv.gz"),
        unique_ids_to_use=strain_w_phenotype_ids,
        genes_to_use=get_selected_genes(use_drug_specific_genes),
    )
    logging.info("Finished reading variants")

    # agg the variants by unique id
    agg_variants_df = variants_df.groupby(["UNIQUEID"]).agg(list)
    del variants_df  # remove it to save memory
    logging.info("Finished aggregating variants")

    # Get vars per unique id / isolate
    unqid_var_dict = get_unqid_var_dict(agg_variants_df)
    # get var to idx dict
    var_to_idx = get_var_to_idx(unqid_var_dict)
    # get var matrix
    var_matrix = get_unqid_var_matrix(unqid_var_dict, var_to_idx)

    logging.info(f"Nr of unique strains and variants: {var_matrix.shape}")

    # save data
    logging.info("Saving data")
    with open(os.path.join(output_dir, "var_to_idx.json"), "w") as f:
        # revert the order as the key in a dict cannot be a tuple
        json.dump({idx: var for var, idx in var_to_idx.items()}, f)

    with open(os.path.join(output_dir, "unique_id_to_idx.json"), "w") as f:
        json.dump(
            {unqid: idx for idx, unqid in enumerate(unqid_var_dict.keys())}, f
        )

    save_npz(os.path.join(output_dir, "var_matrix.npz"), var_matrix)


def main():
    run(
        input_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/",
        output_dir="/tmp/var-matrix/",
        use_drug_specific_genes="cryptic",
    )


if __name__ == "__main__":
    main()
