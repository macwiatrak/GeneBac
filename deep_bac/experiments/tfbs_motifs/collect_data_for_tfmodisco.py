import logging
import os
from collections import defaultdict
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift
from pytorch_lightning.utilities.seed import seed_everything
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.argparser import DeepGeneBacArgumentParser
from deep_bac.data_preprocessing.data_reader import get_gene_expr_data
from deep_bac.modelling.model_gene_expr import DeepBacGeneExpr

logging.basicConfig(level=logging.INFO)


def collect_tfmodisco_data(
    attr_model_fn: Callable, dataloader: DataLoader
) -> Tuple[csr_matrix, csr_matrix]:
    dna_tensor = []
    importance_scores = []
    for idx, batch in enumerate(tqdm(dataloader, mininterval=5)):
        attrs = attr_model_fn.attribute(
            batch.input_tensor.to(attr_model_fn.device),
            return_convergence_delta=False,
        )
        dna_tensor.append(batch.input_tensor)
        importance_scores.append(attrs.cpu())

    dna_tensor = np.stack(dna_tensor)
    dna_tensor = csr_matrix(dna_tensor)

    importance_scores = np.stack(importance_scores)
    importance_scores = csr_matrix(importance_scores)
    return dna_tensor, importance_scores


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
    attr_model_fn = DeepLift(model)

    data = get_gene_expr_data(
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

    dna_tensor, importance_scores_tensor = collect_tfmodisco_data(
        attr_model_fn, data.val_dataloader
    )
    logging.info("Finished collecting data on val, saving it now...")
    np.savez(os.path.join(output_dir, "dna_tensor.npz"), dna_tensor)
    np.savez(
        os.path.join(output_dir, "importance_scores_tensor.npz"),
        importance_scores_tensor,
    )
    logging.info("Finished saving val data")

    if test:
        dna_tensor, importance_scores_tensor = collect_tfmodisco_data(
            attr_model_fn, data.test_dataloader
        )
        logging.info("Finished collecting data on test, saving it now...")
        np.savez(os.path.join(output_dir, "dna_tensor.npz"), dna_tensor)
        np.savez(
            os.path.join(output_dir, "importance_scores_tensor.npz"),
            importance_scores_tensor,
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
