import logging
import os
from typing import Literal, List

import torch
from captum.attr import DeepLift
from pytorch_lightning.utilities.seed import seed_everything
from tap import Tap
from tqdm import tqdm

from deep_bac.data_preprocessing.data_reader import get_gene_pheno_data

from deep_bac.modelling.metrics import (
    MTB_DRUG_TO_LABEL_IDX,
    PA_DRUG_TO_LABEL_IDX,
)
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno
from deep_bac.utils import get_selected_genes, load_trained_pheno_model

logging.basicConfig(level=logging.INFO)


def get_drug_loci_imp(importances: List[torch.Tensor], normalise=False):
    mean_arr = []
    for drug_imp in importances:
        drug_imp = drug_imp[drug_imp.sum(dim=1) != 0]
        mean_drug_imp = drug_imp.mean(dim=0)

        if normalise:
            mean_drug_imp = (
                mean_drug_imp - mean_drug_imp.min()
            ) / mean_drug_imp.max()

        mean_arr.append(mean_drug_imp)
    return torch.stack(mean_arr)


def run(
    input_dir: str,
    ckpt_path: str,
    output_dir: str,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = 0,
    use_drug_specific_genes: Literal[
        "cryptic",
        "PA_small",
        "PA_medium",
    ] = "cryptic",
    compute_agg_importance: bool = True,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selected_genes = get_selected_genes(use_drug_specific_genes)
    logging.info(f"Selected genes: {selected_genes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"][
        "config"
    ]
    model = load_trained_pheno_model(
        ckpt_path=ckpt_path,
        gene_interactions_file_dir=input_dir,
    )
    model.to(device)

    data = get_gene_pheno_data(
        input_df_file_path=os.path.join(
            input_dir, "processed_agg_variants.parquet"
        ),
        reference_gene_data_df_path=os.path.join(
            input_dir, "reference_gene_data.parquet"
        ),
        phenotype_df_file_path=os.path.join(
            input_dir, "phenotype_labels_with_binary_labels.parquet"
        ),
        train_val_test_split_indices_file_path=os.path.join(
            input_dir, "train_test_cv_split_unq_ids.json"
        ),
        max_gene_length=config.max_gene_length,
        regression=config.regression,
        batch_size=config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        selected_genes=selected_genes,
        test=True,
    )
    logging.info("Finished loading data")

    deep_lift = DeepLift(model)

    agg_drug_loci_imps = []
    for drug, drug_idx in PA_DRUG_TO_LABEL_IDX.items():
        drug_loci_importance_abs_sum = []
        # skip PAS drug as we have too few samples anyway
        if drug == "PAS":
            continue

        for idx, batch in enumerate(tqdm(data.test_dataloader)):
            labels = batch.labels[:, drug_idx].cpu()
            label_mask = torch.where(
                labels != -100.0, torch.ones_like(labels), 0
            ).unsqueeze(-1)
            attrs = (
                deep_lift.attribute(
                    inputs=batch.input_tensor.to(device),
                    target=drug_idx,
                    additional_forward_args=batch.tss_indexes.to(device),
                    return_convergence_delta=False,
                )
                .detach()
                .cpu()
            )

            attrs_abs_sum = attrs.abs().sum(dim=-2).sum(dim=-1)
            attrs_abs_sum = attrs_abs_sum * label_mask
            drug_loci_importance_abs_sum.append(attrs_abs_sum)

        drug_loci_importance_abs_sum = torch.cat(drug_loci_importance_abs_sum)
        torch.save(
            drug_loci_importance_abs_sum,
            os.path.join(output_dir, f"{drug}_loci_importance_abs_sum.pt"),
        )
        logging.info(f"Finished processing and saving data for drug: {drug}")
        if compute_agg_importance:
            agg_drug_loci_imps.append(drug_loci_importance_abs_sum)

    agg_mean_loci_imps = get_drug_loci_imp(agg_drug_loci_imps, normalise=True)
    torch.save(agg_mean_loci_imps, "agg_mean_loci_importance.pt")


class ArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str
    ckpt_path: str
    output_dir: str
    shift_max: int = 0
    pad_value: float = 0.25
    reverse_complement_prob: float = 0.0
    random_state: int = 42


def main(args):
    seed_everything(args.random_state)
    run(
        input_dir=args.input_dir,
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        shift_max=args.shift_max,
        pad_value=args.pad_value,
        reverse_complement_prob=args.reverse_complement_prob,
    )


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
