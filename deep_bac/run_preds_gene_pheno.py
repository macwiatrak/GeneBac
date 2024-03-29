import json
import logging
import os
from collections import defaultdict
from typing import Literal

import pandas as pd
import torch
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.argparser import DeepGeneBacArgumentParser
from deep_bac.data_preprocessing.data_reader import get_gene_pheno_data
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno
from deep_bac.modelling.utils import get_drug_thresholds
from deep_bac.utils import get_selected_genes

logging.basicConfig(level=logging.INFO)


def collect_preds(
    model: DeepBacGenePheno, dataloader: DataLoader
) -> pd.DataFrame:
    out = defaultdict(list)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, mininterval=5)):
            logits = model(
                batch.input_tensor.to(model.device),
                batch.tss_indexes.to(model.device),
            )
            out["strain_id"] += batch.strain_ids  # one list
            out["logits"] += [
                item.cpu().numpy() for item in logits
            ]  # a list of numpy arrays
            # if strain_embeddings.nelement() != 0:
            #     out["embedding"] += [
            #         item.cpu().numpy() for item in strain_embeddings
            #     ]  # a list of numpy arrays
            out["labels"] += [
                item.cpu().numpy() for item in batch.labels
            ]  # a list of lists

    df = pd.DataFrame(out)
    return df


def run(
    ckpt_path: str,
    input_dir: str,
    output_dir: str,
    use_drug_idx: int = None,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = None,
    use_drug_specific_genes: Literal[
        "cryptic",
        "PA_small",
        "PA_medium",
    ] = "cryptic",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selected_genes = get_selected_genes(use_drug_specific_genes)
    logging.info(f"Selected genes: {selected_genes}")

    config = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"][
        "config"
    ]
    config.input_dir = (
        # "/Users/maciejwiatrak/Desktop/bacterial_genomics/pseudomonas/mic/"
        "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
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
            input_dir,
            "train_test_split_unq_ids_ablated_lineage3.json",  # "train_test_cv_split_unq_ids.json",
        ),
        max_gene_length=config.max_gene_length,
        regression=config.regression,
        use_drug_idx=use_drug_idx,
        batch_size=config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        selected_genes=selected_genes,
        test=True,
        pin_memory=False,
    )
    logging.info("Finished loading data")

    test_df = collect_preds(model, data.test_dataloader)
    test_df.to_parquet(os.path.join(output_dir, "test_preds.parquet"))
    logging.info("Finished collecting and saving test preds")
    del test_df

    with open(os.path.join(output_dir, "gene_to_idx.json"), "w") as f:
        json.dump(config.gene_to_idx, f)

    if not config.regression:
        drug_thresholds = get_drug_thresholds(model, data.train_dataloader)
        torch.save(
            drug_thresholds, os.path.join(output_dir, "drug_thresholds.pt")
        )

    train_df = collect_preds(model, data.train_dataloader)
    train_df.to_parquet(os.path.join(output_dir, "train_preds.parquet"))
    logging.info("Finished collecting and saving train preds")
    del train_df

    if data.val_dataloader is not None:
        val_df = collect_preds(model, data.val_dataloader)
        val_df.to_parquet(os.path.join(output_dir, "val_preds.parquet"))
        logging.info("Finished collecting and saving val preds")
        del val_df


def main(args):
    seed_everything(args.random_state)
    run(
        input_dir=args.input_dir,
        output_dir="/Users/maciejwiatrak/Downloads/ablation-lineage3-preds/",  # args.output_dir,
        shift_max=args.shift_max,
        pad_value=args.pad_value,
        reverse_complement_prob=args.reverse_complement_prob,
        num_workers=args.num_workers,
        ckpt_path="/Users/maciejwiatrak/Downloads/epoch=297-train_r2=0.4732.ckpt",  # args.ckpt_path,
        use_drug_idx=args.use_drug_idx,
        use_drug_specific_genes="cryptic",  # args.use_drug_specific_genes,
    )


if __name__ == "__main__":
    args = DeepGeneBacArgumentParser().parse_args()
    main(args)
