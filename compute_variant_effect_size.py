import pandas as pd

from deep_bac.experiments.variant_scoring.variant_scoring import (
    compute_variant_effect_size,
)
from deep_bac.utils import load_trained_pheno_model


def main():
    ckpt_path = "files/checkpoints/abr_mtb_regression.ckpt"
    gene_interactions_file_dir = "files/gene_interactions/mtb/"

    ref_gene_data_df = pd.read_parquet(
        "files/reference_gene_data_mtb.parquet"
    ).set_index("gene")

    model = load_trained_pheno_model(
        ckpt_path=ckpt_path,
        gene_interactions_file_dir=gene_interactions_file_dir,
    )

    # compute the variant effect size
    variant_effect_size = compute_variant_effect_size(
        model=model,
        reference_gene_data_df=ref_gene_data_df,
        gene="rpoB",
        variant="g",
        start_idx=1332,
        end_idx=1333,
        drug="RIF",  # if None, will return all drug scores
    )


if __name__ == "__main__":
    main()
