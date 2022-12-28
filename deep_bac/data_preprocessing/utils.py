from collections import defaultdict
from typing import Dict

import gffpandas.gffpandas as gffpd
import pandas as pd
import torch

from pyfastx import Fastx


ONE_HOT_EMBED = torch.zeros(256, 4)
ONE_HOT_EMBED[ord("a")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("c")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("g")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
ONE_HOT_EMBED[ord("t")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
ONE_HOT_EMBED[ord("n")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("A")] = torch.Tensor([1.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("C")] = torch.Tensor([0.0, 1.0, 0.0, 0.0])
ONE_HOT_EMBED[ord("G")] = torch.Tensor([0.0, 0.0, 1.0, 0.0])
ONE_HOT_EMBED[ord("T")] = torch.Tensor([0.0, 0.0, 0.0, 1.0])
ONE_HOT_EMBED[ord("N")] = torch.Tensor([0.0, 0.0, 0.0, 0.0])
ONE_HOT_EMBED[ord(".")] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

REVERSE_COMPLEMENT_MAP = torch.Tensor([3, 2, 1, 0, 4]).long()


def read_genome_assembly(file_path: str):
    return gffpd.read_gff3(file_path).df


def read_ref_genome(file_path: str):
    fa = Fastx(file_path)
    name, seq, comment = next(fa)
    return seq


def get_gene_name(attr: str):
    return attr.split("Name=")[-1].split(";")[0]


def get_complement(seq):
    complement_dict = {'a': 't', 't': 'a', 'g': 'c', 'c': 'g'}
    out = ''
    for item in seq:
        out += complement_dict[item.lower()]
    assert len(out) == len(seq)
    return out


def get_gene_seq(ref_genome: str, row: Dict):
    start = row['start'] - 1
    end = row['end']

    gene_seq = ref_genome[start:end].lower()
    if row['strand'] == "-":
        gene_seq = get_complement(gene_seq)[::-1]
    return gene_seq


def get_promoter_seq(ref_genome: str, prom_seq_len: int, row: Dict):
    start = row['start'] - 1
    end = row['end']

    if row['strand'] == "-":
        if end + prom_seq_len > len(ref_genome):
            prom_seq = ref_genome[end:].lower()
        else:
            prom_seq = ref_genome[end:end + prom_seq_len].lower()
        prom_seq = get_complement(prom_seq)[::-1]
    else:
        if start - prom_seq_len < 0:
            prom_seq = ref_genome[:start].lower()
        else:
            prom_seq = ref_genome[start - 100:start].lower()
    return prom_seq


def convert_gene_data_df_to_dict(df: pd.DataFrame) -> Dict[str, Dict]:
    gene_dict = defaultdict(dict)

    for idx, row in df.iterrows():
        gene_name = row["GENE"]
        del row["GENE"]
        gene_dict[gene_name] = row
    return gene_dict


def get_gene_data_dict(ref_genome: str, genome_assembly: pd.DataFrame, prom_seq_len: int):
    gene_data_df = genome_assembly[genome_assembly['type'] == 'gene']  # include genes only

    # get gene name
    gene_data_df['GENE'] = gene_data_df['attributes'].apply(get_gene_name)
    gene_data_df = gene_data_df.drop(columns=['seq_id', 'source', 'attributes'])

    gene_data_df['gene_seq'] = gene_data_df.apply(
        lambda row: get_gene_seq(ref_genome=ref_genome, row=row),
        axis=1,
    )

    gene_data_df['promoter_seq'] = gene_data_df.apply(
        lambda row: get_promoter_seq(
            ref_genome=ref_genome,
            prom_seq_len=prom_seq_len,
            row=row,
        ),
        axis=1,
    )

    return convert_gene_data_df_to_dict(gene_data_df)
