import os

import torch
from pytorch_lightning.utilities.seed import seed_everything

from genebac.experiments.variant_scoring.perform_ism import perform_ism
from genebac.experiments.variant_scoring.run_variant_scoring_who_cat import (
    VariantScoringArgumentParser,
)
from genebac.utils import DRUG_SPECIFIC_GENES_DICT


def run(
    ckpt_path: str,
    output_dir: str,
    reference_gene_data_df_path: str,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
):
    for idx, loci in enumerate(DRUG_SPECIFIC_GENES_DICT["cryptic"]):
        ism_scores = perform_ism(
            ckpt_path=ckpt_path,
            reference_gene_data_df_path=reference_gene_data_df_path,
            loci=loci,
            shift_max=shift_max,
            pad_value=pad_value,
            reverse_complement_prob=reverse_complement_prob,
            output_dir=output_dir,
            ism_region=None,  # do ISM on the whole gene
        )
        torch.save(
            ism_scores, os.path.join(output_dir, f"{loci}_{idx}_ism_scores.pt")
        )


def main(args):
    seed_everything(args.random_state)
    run(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        reference_gene_data_df_path=args.reference_gene_data_df_path,
        shift_max=args.shift_max,
        pad_value=args.pad_value,
        reverse_complement_prob=args.reverse_complement_prob,
    )


if __name__ == "__main__":
    args = VariantScoringArgumentParser().parse_args()
    main(args)
