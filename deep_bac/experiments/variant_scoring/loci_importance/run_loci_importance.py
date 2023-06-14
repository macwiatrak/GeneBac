import logging
import os
from collections import defaultdict

import torch
from captum.attr import DeepLift
from pytorch_lightning.utilities.seed import seed_everything
from tap import Tap
from tqdm import tqdm

from deep_bac.data_preprocessing.data_reader import get_gene_pheno_data

from deep_bac.modelling.metrics import DRUG_TO_LABEL_IDX
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno
from deep_bac.utils import get_selected_genes

logging.basicConfig(level=logging.INFO)


def run(
    input_dir: str,
    ckpt_path: str,
    output_dir: str,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = 4,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"][
        "config"
    ]
    config.input_dir = (
        "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data"
    )
    model = DeepBacGenePheno.load_from_checkpoint(ckpt_path, config=config)
    model.to(device)
    model.eval()

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
        variance_per_gene_file_path=os.path.join(
            input_dir, "unnormalised_variance_per_gene.csv"
        ),
        max_gene_length=config.max_gene_length,
        n_highly_variable_genes=config.n_highly_variable_genes,
        regression=config.regression,
        batch_size=config.batch_size * 2,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        selected_genes=get_selected_genes("cryptic"),
        test=True,
    )
    logging.info("Finished loading data")

    deep_lift = DeepLift(model)

    loci_importance = defaultdict(list)
    loci_importance_sum = defaultdict(list)
    for batch in tqdm(data.test_dataloader):
        for _, idx in DRUG_TO_LABEL_IDX.items():
            attrs = (
                deep_lift.attribute(
                    inputs=batch.input_tensor.to(device),
                    target=idx,
                    additional_forward_args=batch.tss_indexes.to(device),
                    return_convergence_delta=False,
                )
                .detach()
                .cpu()
            )

            # TODO: check the sum is correct here
            attrs_sum = attrs.sum(dim=1)
            attrs_sum = torch.where(
                batch.labels == -100.0, torch.zeros_like(attrs_sum), attrs_sum
            )
            loci_importance[idx].append(attrs_sum)

            attrs_abs_sum = attrs.abs().sum(dim=1)
            attrs_abs_sum = torch.where(
                batch.labels == -100.0,
                torch.zeros_like(attrs_abs_sum),
                attrs_abs_sum,
            )
            loci_importance_sum[idx].append(attrs_abs_sum)

    attrs_sum = torch.stack(
        [torch.cat(items) for _, items in loci_importance.items()]
    )
    torch.save(attrs_sum, os.path.join(output_dir, "loci_importance_sum.pt"))
    del attrs_sum

    attrs_abs_sum = torch.stack(
        [torch.cat(items) for _, items in loci_importance_sum.items()]
    )
    torch.save(
        attrs_abs_sum, os.path.join(output_dir, "loci_importance_abs_sum.pt")
    )


class ArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str = (
        "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data"
    )
    ckpt_path: str = "/Users/maciejwiatrak/Downloads/epoch=285-train_gmean_spec_sens=0.8652.ckpt"
    output_dir: str = "/tmp/var-scores/genebac/bin/"
    variant_df_path: str = (
        "/Users/maciejwiatrak/DeepBac/deep_bac/experiments/variant_scoring/"
        "data/vars_who_cat_for_deeplift_binary.parquet"
    )
    #     "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/"
    #     "data/who_cat_mutations_dna_seqs.parquet"
    # )
    reference_gene_data_df_path: str = (
        "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/"
        "data/reference_gene_data.parquet"
    )
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
