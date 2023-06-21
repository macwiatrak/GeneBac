import os
from typing import Optional, Tuple

import pandas as pd
import torch
from pytorch_lightning.utilities.seed import seed_everything
from torch.nn.functional import one_hot
from tqdm import tqdm

from deep_bac.data_preprocessing.data_types import BacInputSample
from deep_bac.experiments.variant_scoring.run_variant_scoring_who_cat import (
    VariantScoringArgumentParser,
)
from deep_bac.experiments.variant_scoring.variant_scoring import get_ref_batch
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno


def compute_ism_scores(
    ref_scores: torch.Tensor,
    ref_input_sample: BacInputSample,
    model: Optional[DeepBacGenePheno],
    loci_idx: int,
    ism_region: Tuple[int, int] = None,
    device: str = "cpu",
    regression: bool = True,
) -> torch.Tensor:
    tss_indexes_batch = (
        ref_input_sample.tss_index.unsqueeze(0).repeat(4, 1).to(device)
    )
    # if ism_region is None, use the whole gene
    if ism_region is None:
        loci = ref_input_sample.input_tensor[loci_idx]
        loci_len = (loci.max(dim=0).values == 1.0).sum().item()
        ism_region = (0, loci_len)

    output = []
    # iterate through each position in the sequence
    for nucleotide_pos in tqdm(range(ism_region[0], ism_region[1])):
        nucleotide_pos_batch = []
        # iterate through each bp at each position
        for nucleotide in range(4):
            item = one_hot(torch.tensor(nucleotide), num_classes=4).type_as(
                ref_input_sample.input_tensor
            )
            mutated_input_tensor = ref_input_sample.input_tensor.clone()
            mutated_input_tensor[loci_idx, :, nucleotide_pos] = item
            nucleotide_pos_batch.append(mutated_input_tensor)

        nucleotide_pos_batch = torch.stack(nucleotide_pos_batch)
        with torch.no_grad():
            scores = model(
                nucleotide_pos_batch.to(device), tss_indexes_batch
            ).cpu()
            scores = torch.sigmoid(scores) if not regression else scores
            delta = scores - ref_scores
        output.append(delta)
    return torch.stack(output)


def perform_ism(
    ckpt_path: str,
    output_dir: str,
    reference_gene_data_df_path: str,
    loci: str,
    ism_region: Tuple[int, int] = None,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
) -> torch.Tensor:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reference_gene_data_df = pd.read_parquet(
        reference_gene_data_df_path
    ).set_index("gene")
    config = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"][
        "config"
    ]
    # config.input_dir = (
    #     "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data"
    # )
    model = DeepBacGenePheno.load_from_checkpoint(ckpt_path, config=config)
    model.eval()
    model.to(device)

    gene_to_idx = model.config.gene_to_idx

    loci_idx = gene_to_idx.get(loci, None)
    if loci_idx is None:
        raise ValueError(
            f"Input genomic loci {loci} not valid, please choose from:"
            f"{list(gene_to_idx.keys())}"
        )

    ref_input_sample = get_ref_batch(
        genes=list(gene_to_idx.keys()),
        reference_gene_data_df=reference_gene_data_df,
        max_gene_length=config.max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
    )

    # get ref variant scores
    with torch.no_grad():
        ref_scores = model(
            ref_input_sample.input_tensor.unsqueeze(0).to(device),
            ref_input_sample.tss_index.unsqueeze(0).to(device),
        ).cpu()
        ref_scores = (
            torch.sigmoid(ref_scores)
            if not model.config.regression
            else ref_scores
        )
    ism_scores = compute_ism_scores(
        model=model,
        ref_scores=ref_scores,
        ref_input_sample=ref_input_sample,
        loci_idx=loci_idx,
        ism_region=ism_region,
        device=device,
        regression=model.config.regression,
    )
    print("max ISM score value:", ism_scores.max())
    return ism_scores
    # torch.save(ism_scores, os.path.join(output_dir, "ism_scores.pt"))
    # torch.save(
    #     ref_input_sample.input_tensor, os.path.join(output_dir, "ref_tensor.pt")
    # )


def main(args):
    seed_everything(args.random_state)
    perform_ism(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        reference_gene_data_df_path=args.reference_gene_data_df_path,
        shift_max=args.shift_max,
        pad_value=args.pad_value,
        reverse_complement_prob=args.reverse_complement_prob,
        loci="rpoB",
        ism_region=(0, 50),
    )


if __name__ == "__main__":
    args = VariantScoringArgumentParser().parse_args()
    main(args)
