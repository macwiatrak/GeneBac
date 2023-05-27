import os

import pandas as pd
import torch
from captum.attr import DeepLiftShap, DeepLift
from pytorch_lightning.utilities.seed import seed_everything
from tap import Tap
from torch.nn.functional import one_hot
from tqdm import tqdm

from deep_bac.experiments.variant_scoring.variant_scoring import (
    get_ref_batch,
    batch_samples_w_variant,
)
from deep_bac.modelling.metrics import DRUG_TO_LABEL_IDX
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno


def run(
    ckpt_path: str,
    output_dir: str,
    variant_df_path: str,
    reference_gene_data_df_path: str,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_gene_data_df = pd.read_parquet(
        reference_gene_data_df_path
    ).set_index("gene")
    config = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"][
        "config"
    ]
    config.input_dir = (
        "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data"
    )
    model = DeepBacGenePheno.load_from_checkpoint(ckpt_path, config=config)
    model.to(device)
    model.eval()

    deep_lift = DeepLift(model)

    gene_to_idx = model.config.gene_to_idx

    ref_input_sample = get_ref_batch(
        genes=list(gene_to_idx.keys()),
        reference_gene_data_df=reference_gene_data_df,
        max_gene_length=config.max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
    )

    # get ref variant scores
    # with torch.no_grad():
    ref_scores = model(
        ref_input_sample.input_tensor.unsqueeze(0).to(device),
        ref_input_sample.tss_index.unsqueeze(0).to(device),
    )
    ref_scores = (
        (
            torch.sigmoid(ref_scores)
            if not model.config.regression
            else ref_scores
        )
        .detach()
        .cpu()
    )

    ref_attr_scores = []
    for _, idx in DRUG_TO_LABEL_IDX.items():
        attrs = deep_lift.attribute(
            ref_input_sample.input_tensor.unsqueeze(0).to(device),
            target=idx,
            additional_forward_args=ref_input_sample.tss_index.unsqueeze(0).to(
                device
            ),
            return_convergence_delta=False,
        )
        ref_attr_scores.append(attrs.detach().cpu())
    ref_attr_scores = torch.cat(ref_attr_scores, dim=0)
    torch.save(
        ref_attr_scores, os.path.join(output_dir, "ref_attributions_bin.pt")
    )

    # get variant scores
    variant_df = pd.read_parquet(variant_df_path)
    # batch data
    batches = batch_samples_w_variant(
        variant_df=variant_df,
        genes=list(gene_to_idx.keys()),
        reference_gene_data_df=reference_gene_data_df,
        max_gene_length=config.max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        batch_size=config.batch_size,
    )

    var_tensor = []
    var_scores = []
    var_attributions = []
    # with torch.no_grad():
    for batch in tqdm(batches):
        scores = model(
            batch.input_tensor.to(device), batch.tss_indexes.to(device)
        )
        scores = (
            torch.sigmoid(scores) if not model.config.regression else scores
        )
        var_scores.append(scores.detach().cpu() - ref_scores)

        attrs = deep_lift.attribute(
            inputs=batch.input_tensor.to(device),
            baselines=ref_input_sample.input_tensor.unsqueeze(0).to(device),
            target=batch.labels.to(device),
            additional_forward_args=batch.tss_indexes.to(device),
            return_convergence_delta=False,
        )
        var_attributions.append(attrs.detach().cpu())
        var_tensor.append(batch.input_tensor.detach().cpu())

    var_attributions = torch.cat(var_attributions, dim=0)
    var_tensor = torch.cat(var_tensor, dim=0)
    torch.save(
        var_attributions, os.path.join(output_dir, "var_attributions_bin.pt")
    )
    torch.save(var_tensor, os.path.join(output_dir, "var_tensor.pt"))

    # var_scores = [item.numpy() for item in torch.cat(var_scores, dim=0)]
    # variant_df["var_scores"] = var_scores
    # variant_df["var_score"] = variant_df.apply(
    #     lambda row: row["var_scores"][DRUG_TO_LABEL_IDX[row["drug"]]], axis=1
    # )
    # drop redundant column
    # variant_df = variant_df.drop(columns=["var_scores"])

    # variant_df.to_parquet(
    #     os.path.join(output_dir, "variants_who_cat_with_scores_binary_attrs.parquet")
    # )


class ArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
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
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        variant_df_path=args.variant_df_path,
        reference_gene_data_df_path=args.reference_gene_data_df_path,
        shift_max=args.shift_max,
        pad_value=args.pad_value,
        reverse_complement_prob=args.reverse_complement_prob,
    )


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
