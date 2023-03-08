import logging
import os
from collections import defaultdict

import pandas as pd
import torch
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.argparser import DeepGeneBacArgumentParser
from deep_bac.data_preprocessing.data_reader import get_gene_expr_data
from deep_bac.modelling.model_gene_expr import DeepBacGeneExpr

logging.basicConfig(level=logging.INFO)


def collect_gene_reprs(model: DeepBacGeneExpr, dataloader: DataLoader):
    out = defaultdict(list)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, mininterval=5)):
            logits, gene_embeddings = model(
                batch.input_tensor.to(model.device),
                batch.tss_indexes.to(model.device),
            )
            out["strain_id"] += batch.strain_ids
            out["logits"] += logits.view(-1).cpu().tolist()
            out["embedding"] += [item.cpu().numpy() for item in gene_embeddings]
            out["labels"] += batch.labels.view(-1).cpu().tolist()
            out["gene_name"] += batch.gene_names

    df = pd.DataFrame(out)
    return df


def run(
    ckpt_path: str,
    input_dir: str,
    output_dir: str,
    max_gene_length: int = 2560,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = None,
    test: bool = False,
    batch_size: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    model = DeepBacGeneExpr.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    data, most_variable_genes = get_gene_expr_data(
        input_dir=input_dir,
        max_gene_length=max_gene_length,
        batch_size=batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        test=test,
    )
    logging.info("Finished loading data")

    val_df = collect_gene_reprs(model, data.val_dataloader)
    logging.info("Finished collecting val data, saving it now...")
    val_df.to_parquet(
        os.path.join(output_dir, "val_gene_representations.parquet")
    )
    logging.info("Finished saving val data")

    if test:
        test_df = collect_gene_reprs(model, data.test_dataloader)
        logging.info("Finished collecting test data, saving it now...")
        test_df.to_parquet(
            os.path.join(output_dir, "test_gene_representations.parquet")
        )
        logging.info("Finished saving test data")


def main(args):
    seed_everything(args.random_state)
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_gene_length=args.max_gene_length,
        shift_max=args.shift_max,
        pad_value=args.pad_value,
        reverse_complement_prob=args.reverse_complement_prob,
        num_workers=args.num_workers,
        test=args.test,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    args = DeepGeneBacArgumentParser().parse_args()
    main(args)
