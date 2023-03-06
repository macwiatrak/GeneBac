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
from deep_bac.utils import get_selected_genes

logging.basicConfig(level=logging.INFO)


def collect_strain_reprs(model: DeepBacGenePheno, dataloader: DataLoader):
    out = defaultdict(list)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, mininterval=5)):
            logits, strain_embeddings = model(
                batch.input_tensor.to(model.device),
                batch.tss_indexes.to(model.device),
            )
            out["strain_id"] += batch.strain_ids  # one list
            out["logits"] += [
                item.cpu().numpy() for item in logits
            ]  # a list of numpy arrays
            out["embedding"] += [
                item.cpu().numpy() for item in strain_embeddings
            ]  # a list of numpy arrays
            out["labels"] += [batch.labels.cpu().tolist()]  # a list of lists
            if idx > 4:
                break

    df = pd.DataFrame(out)
    return df


def run(
    ckpt_path: str,
    input_dir: str,
    output_dir: str,
    n_highly_variable_genes: int = 500,
    use_drug_idx: int = None,
    max_gene_length: int = 2560,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = None,
    test: bool = True,
    use_drug_specific_genes: Literal["INH", "Walker", "MD-CNN"] = "MD-CNN",
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    model = DeepBacGenePheno.load_from_checkpoint(ckpt_path).to(device)
    model.to(device)
    model.eval()

    selected_genes = get_selected_genes(use_drug_specific_genes)
    logging.info(f"Selected genes: {selected_genes}")

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
            input_dir, "train_val_test_split_unq_ids.json"
        ),
        variance_per_gene_file_path=os.path.join(
            input_dir, "unnormalised_variance_per_gene.csv"
        ),
        max_gene_length=max_gene_length,
        n_highly_variable_genes=n_highly_variable_genes,
        regression=model.config.regression,
        use_drug_idx=use_drug_idx,
        batch_size=batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        selected_genes=selected_genes,
        test=test,
    )
    logging.info("Finished loading data")

    val_df = collect_strain_reprs(model, data.val_dataloader)
    val_df.to_parquet(
        os.path.join(output_dir, "val_strain_representations.parquet")
    )
    logging.info("Finished saving val data")

    if test:
        test_df = collect_strain_reprs(model, data.test_dataloader)
        test_df.to_parquet(
            os.path.join(output_dir, "test_strain_representations.parquet")
        )
        logging.info("Finished saving test data")

    train_df = collect_strain_reprs(model, data.train_dataloader)
    train_df.to_parquet(
        os.path.join(output_dir, "train_strain_representations.parquet")
    )
    logging.info("Finished saving train data")


def main(args):
    seed_everything(args.random_state)
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_highly_variable_genes=args.n_highly_variable_genes,
        max_gene_length=args.max_gene_length,
        shift_max=args.shift_max,
        pad_value=args.pad_value,
        reverse_complement_prob=args.reverse_complement_prob,
        num_workers=args.num_workers,
        test=args.test,
        ckpt_path=args.ckpt_path,
        use_drug_idx=args.use_drug_idx,
        use_drug_specific_genes=args.use_drug_specific_genes,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    args = DeepGeneBacArgumentParser().parse_args()
    main(args)
