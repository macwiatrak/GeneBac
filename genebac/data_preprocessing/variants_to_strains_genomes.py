from typing import List, Set, Dict, Optional

import pandas as pd

from genebac.data_preprocessing.utils import get_complement

VARIANTS_COLS_TO_USE = [
    "UNIQUEID",
    "REF",
    "ALT",
    "GENE",
    "NUCLEOTIDE_NUMBER",
    "IS_INDEL",
]


def get_and_filter_variants_df(
    file_path: str,
    unique_ids_to_use: Set[str],
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
    df = df[~df["GENE"].isna()]
    return df


def retrieve_idxs_pos_strand(nucleotide_nr: int, ref_len: int):
    if nucleotide_nr < 1:
        if ref_len == 1:
            return (nucleotide_nr, nucleotide_nr + 1)
        return (nucleotide_nr, nucleotide_nr + ref_len)
    return (nucleotide_nr - 1, nucleotide_nr - 1 + ref_len)


def retrieve_idxs_neg_strand(nucleotide_nr: int, ref_len: int, is_indel: bool):
    if nucleotide_nr < 1:
        if ref_len == 1:
            return nucleotide_nr, nucleotide_nr + 1
        return nucleotide_nr - ref_len + 1, nucleotide_nr + 1

    if is_indel:
        return nucleotide_nr - ref_len, nucleotide_nr
    return nucleotide_nr - 1, nucleotide_nr


def retrieve_variant_idxs(
    nucleotide_nr: int, ref_len: int, strand: str, is_indel: bool
):
    if strand == "+":
        return retrieve_idxs_pos_strand(
            nucleotide_nr=nucleotide_nr,
            ref_len=ref_len,
        )
    else:
        return retrieve_idxs_neg_strand(
            nucleotide_nr=nucleotide_nr,
            ref_len=ref_len,
            is_indel=is_indel,
        )


def apply_variants_to_a_gene(
    ref_gene_seq: str,
    ref_prom_seq: str,
    strand: str,
    row: Dict[str, Optional[List]],
) -> Dict:
    prom_gene_seq_w_variants = ""

    ref_seq = ref_prom_seq + ref_gene_seq
    len_ref_seq = len(ref_seq)
    len_prom = len(ref_prom_seq)
    curr_index = 0
    len_change = 0  # for sanity assertions
    ref_correct = 0
    n_nucleotide_change = 0

    for ref, alt, nucleotide_nr, is_indel in zip(
        row["REF"], row["ALT"], row["NUCLEOTIDE_NUMBER"], row["IS_INDEL"]
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
        vstart_idx += len_prom
        vend_idx += len_prom

        vstart_idx = max(0, vstart_idx)
        vend_idx = vend_idx if vend_idx >= 0 else len_ref_seq

        prom_gene_seq_w_variants += ref_seq[curr_index:vstart_idx]
        prom_gene_seq_w_variants += alt
        curr_index = vend_idx
        if ref == ref_seq[vstart_idx:vend_idx]:
            ref_correct += 1

    prom_gene_seq_w_variants += ref_seq[curr_index:]
    ref_correct_ratio = ref_correct / len(row["REF"])

    return dict(
        prom_gene_seq_w_variants=prom_gene_seq_w_variants,
        ref_correct_ratio=ref_correct_ratio,
        n_nucleotide_change=n_nucleotide_change,
    )
