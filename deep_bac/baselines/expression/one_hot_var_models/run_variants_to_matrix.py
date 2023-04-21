from collections import defaultdict
from functools import partial
import numpy as np
from tqdm import tqdm

import pandas as pd
import os
import gffpandas.gffpandas as gffpd

from pyfastx import Fastx

tqdm.pandas()


INPUT_DIR = "/Users/maciejwiatrak/Desktop/bacterial_genomics/pseudomonas/"


def read_ref_genome(file_path: str):
    fa = Fastx(file_path)
    name, seq, comment = next(fa)
    return seq


def read_genome_assembly(file_path: str):
    return gffpd.read_gff3(file_path).df


def get_gene_name(attr: str):
    return attr.split("Alias=")[-1].split(";")[0]


def get_rev_complement(seq):
    complement_dict = {"a": "t", "t": "a", "g": "c", "c": "g"}
    out = ""
    for item in seq:
        out += complement_dict[item.lower()]
    assert len(out) == len(seq)
    return out[::-1]


def get_ref_gene_seqs(ref_genome, gen_assembly, prom_length: int = 200):
    gen_assembly = gen_assembly[gen_assembly["type"] == "gene"]
    gen_assembly["gene_name"] = gen_assembly["attributes"].apply(get_gene_name)

    ref_gene_seq = defaultdict(dict)
    for idx, row in gen_assembly.iterrows():
        start = row["start"]
        end = row["end"]
        strand = row["strand"]

        gene_seq = ref_genome[start - 1 : end]

        if strand == "+":
            start_prom = max(start - 1 - prom_length, 0)
            prom_seq = ref_genome[start_prom : start - 1]
            prom_gene_seq = prom_seq + gene_seq
            start = start - len(prom_seq)
        else:
            prom_seq = ref_genome[end : end + prom_length]
            prom_gene_seq = gene_seq + prom_seq
            end = end + len(prom_seq)

        assert len(prom_seq) <= prom_length
        ref_gene_seq[row["gene_name"]] = {
            "prom_gene_seq": prom_gene_seq.lower(),
            "prom_seq_len": len(prom_seq),
            "start": start,
            "end": end,
            "strand": strand,
        }
    return ref_gene_seq


def get_pos(x):
    return int(x.split("_")[0])


def get_alt(x):
    return x.split("_")[1]


def get_ref_snp(ref_genome, x):
    return ref_genome[x - 1]


def get_ref_indel(x):
    return x.split("_")[1]


def process_df_vars(df, ref_genome):
    df["alt"] = df["pos"].apply(get_alt)
    df["pos"] = df["pos"].apply(get_pos)

    get_ref_fn = partial(get_ref_snp, ref_genome)
    df["ref"] = df["pos"].apply(get_ref_fn)
    df = df.drop(columns=["PAO1"])
    return df


def check_ref_correct_snp(ref_genome, row):
    if ref_genome[row["pos"] - 1].lower() == row["ref"].lower():
        return 1
    return 0


def check_ref_indel(ref_genome, row):
    start = row["pos"] - 1
    indel_len = len(row["ref"])

    ref_seq = ref_genome[start : start + indel_len]
    if ref_seq.lower() == row["ref"].lower():
        return 1
    return 0


def is_variant_in_gene(ref_gene_seqs, pos):

    for gene_name, gene_vals in ref_gene_seqs.items():
        in_prom = False
        in_gene = False

        start = gene_vals["start"]
        end = gene_vals["end"]
        prom_seq_len = gene_vals["prom_seq_len"]

        if start <= pos <= end:
            pos_in_gene = int(pos - start)

            if gene_vals["strand"] == "+" and start <= pos <= (
                start + prom_seq_len
            ):
                in_prom = True
            elif gene_vals["strand"] == "-" and end <= pos <= (
                end + prom_seq_len
            ):
                in_prom = True
            else:
                in_gene = True

            return dict(
                gene=gene_name,
                pos_in_gene=pos_in_gene,
                in_gene=in_gene,
                in_prom=in_prom,
            )

    return dict(
        gene=None,
        pos_in_gene=-1,
        in_gene=False,
        in_prom=False,
    )


def join_vars_and_indels(df_vars, df_indels):
    df_indels = df_indels.drop(columns=["feature_id"])
    df_indels["IS_SNP"] = False
    df_indels["IS_INDEL"] = True

    df_vars["IS_SNP"] = True
    df_vars["IS_INDEL"] = False

    return pd.concat([df_vars, df_indels])


def log1_transform(x):
    if isinstance(x, float):
        return np.log(1 + x)
    return np.log(1 + x[0])


def panaroo_to_gene_name(panaraoo_to_pa_orf, x):
    return panaraoo_to_pa_orf.get(x, None)


def run(
    input_dir: str,
    output_dir: str,
    promoter_len: int = 200,
):
    # process SNPs
    df_counts = pd.read_csv(
        os.path.join(input_dir, "counts_matrix.txt"), sep="\t"
    )
    df_gene2ref = pd.read_csv(
        os.path.join(input_dir, "gene2ref_PGD_refound.txt"), sep="\t"
    )
    df_vars = pd.read_csv(
        os.path.join(input_dir, "sample2variant_one_hot.txt"), sep="\t"
    )

    ref_genome = read_ref_genome(
        os.path.join(input_dir, "Pseudomonas_aeruginosa_PAO1_107.fna")
    )
    gen_assembly = read_genome_assembly(
        os.path.join(input_dir, "Pseudomonas_aeruginosa_PAO1_107.gff")
    )

    ref_gene_seqs_dict = get_ref_gene_seqs(
        ref_genome, gen_assembly, promoter_len
    )

    df_vars = process_df_vars(df_vars, ref_genome)
    df_vars["check_ref_correct"] = df_vars.apply(
        lambda row: check_ref_correct_snp(ref_genome, row), axis=1
    )
    ref_correct_snp = df_vars["check_ref_correct"].sum() / len(df_vars)
    print("SNP ref correct:", ref_correct_snp)

    is_variant_in_gene_fn = partial(is_variant_in_gene, ref_gene_seqs_dict)
    df_vars["var_gene_info"] = df_vars["pos"].progress_apply(
        is_variant_in_gene_fn
    )

    df_vars = pd.concat(
        [
            df_vars.drop(["var_gene_info"], axis=1),
            df_vars["var_gene_info"].apply(pd.Series),
        ],
        axis=1,
    )

    # remove mutations not in a gene or promoter
    df_vars_cds = df_vars[~df_vars["gene"].isna()]

    # process INDELs
    df_indels = pd.read_csv(
        os.path.join(input_dir, "indels_imputed.txt"), sep="\t"
    )

    df_indels["pos"] = df_indels["feature_id"].apply(get_pos)
    df_indels["alt"] = df_indels["feature_id"].apply(get_alt)
    df_indels["ref"] = df_indels["feature_id"].apply(get_ref_indel)

    df_indels["check_ref_correct"] = df_indels.apply(
        lambda row: check_ref_indel(ref_genome, row), axis=1
    )
    ref_correct_indel = df_indels["check_ref_correct"].sum() / len(df_indels)
    print("INDEL ref correct:", ref_correct_indel)

    df_indels["var_gene_info"] = df_indels["pos"].progress_apply(
        is_variant_in_gene_fn
    )

    df_indels = pd.concat(
        [
            df_indels.drop(["var_gene_info"], axis=1),
            df_indels["var_gene_info"].apply(pd.Series),
        ],
        axis=1,
    )
    # remove mutations not in a gene or promoter
    df_indels_cds = df_indels[~df_indels["gene"].isna()]

    df_vars_indels_cds = join_vars_and_indels(df_vars_cds, df_indels_cds)

    panaraoo_to_pa_orf = {
        key: val
        for key, val in zip(
            df_gene2ref["Panaroo"].tolist(), df_gene2ref["PA_ORF"].tolist()
        )
    }
    genes_w_expression = [
        panaraoo_to_pa_orf.get(item, None)
        for item in df_counts["feature_id"].tolist()
        if panaraoo_to_pa_orf.get(item, None)
    ]

    # remove genes for which we do not have expression
    df_vars_indels_cds_w_expression = df_vars_indels_cds[
        df_vars_indels_cds["gene"].isin(genes_w_expression)
    ]

    # sort by gene and position and then aggregate the variants by gene
    agg_variants = (
        df_vars_indels_cds_w_expression.sort_values(by=["gene", "pos"])
        .groupby("gene")
        .agg(list)
    )

    strains_with_counts = set(df_counts.columns) - {"feature_id", "gene_name"}
    agg_variants = agg_variants[strains_with_counts]

    # do intersection
    genes_to_add = list(
        set(genes_w_expression) - set(agg_variants.index.tolist())
    )
    genes_to_add_df = pd.DataFrame(
        {strain: [[0]] * len(genes_to_add) for strain in strains_with_counts},
        index=genes_to_add,
    )
    agg_variants = pd.concat([agg_variants, genes_to_add_df])

    panaroo_to_gene_name_fn = partial(panaroo_to_gene_name, panaraoo_to_pa_orf)
    df_counts["gene"] = df_counts["feature_id"].apply(panaroo_to_gene_name_fn)
    df_counts.index = df_counts["gene"]
    df_counts = df_counts[~df_counts["gene"].isna()]
    df_counts = df_counts.drop_duplicates(subset=["gene"])
    df_counts = df_counts[strains_with_counts]
    df_labels = df_counts.applymap(log1_transform)

    df_labels.to_parquet(
        os.path.join(output_dir, "gene_expression_vals.parquet")
    )
    agg_variants.to_parquet(os.path.join("variants_per_gene.parquet"))


def main():
    run(
        input_dir=INPUT_DIR,
        output_dir="/tmp/",
        promoter_len=200,
    )


if __name__ == "__main__":
    main()
