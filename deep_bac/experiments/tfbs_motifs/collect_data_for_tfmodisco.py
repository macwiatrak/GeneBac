import logging
import os
from typing import Callable, Tuple

import numpy as np
import torch
from captum.attr import DeepLift
from pytorch_lightning.utilities.seed import seed_everything
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.argparser import DeepGeneBacArgumentParser
from deep_bac.data_preprocessing.data_reader import get_gene_expr_data
from deep_bac.modelling.data_types import DeepGeneBacConfig
from deep_bac.modelling.model_gene_expr import DeepBacGeneExpr

logging.basicConfig(level=logging.INFO)


def collect_tfmodisco_data(
    attr_model_fn: Callable, dataloader: DataLoader, device: str
) -> Tuple[csr_matrix, csr_matrix]:
    dna_tensor = []
    importance_scores = []
    for idx, batch in enumerate(tqdm(dataloader, mininterval=5)):
        if idx > 100:
            break
        attrs = attr_model_fn.attribute(
            batch.input_tensor.to(device),
            additional_forward_args=batch.tss_indexes.to(device),
            return_convergence_delta=False,
        )
        dna_tensor.append(batch.input_tensor)
        importance_scores.append(attrs.detach().cpu())

    dna_tensor = torch.stack(dna_tensor).numpy()

    importance_scores = torch.stack(importance_scores).numpy()
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
    # model = DeepBacGeneExpr.load_from_checkpoint(ckpt_path)
    model = DeepBacGeneExpr(config=DeepGeneBacConfig())
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
        attr_model_fn,
        data.val_dataloader,
        device=device,
    )
    logging.info("Finished collecting data on val, saving it now...")
    np.save(os.path.join(output_dir, "dna_tensor.npy"), dna_tensor)
    np.save(
        os.path.join(output_dir, "importance_scores_tensor.npy"),
        importance_scores_tensor,
    )
    logging.info("Finished saving val data")

    if test:
        dna_tensor, importance_scores_tensor = collect_tfmodisco_data(
            attr_model_fn, data.test_dataloader, device=device
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
        # input_dir=args.input_dir,
        input_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/pseudomonas/prom-200-w-rev-comp/",
        # output_dir=args.output_dir,
        output_dir="/tmp/modisco-data/",
        max_gene_length=args.max_gene_length,
        shift_max=0,
        pad_value=args.pad_value,
        reverse_complement_prob=0.0,
        num_workers=args.num_workers,
        test=args.test,
        ckpt_path="/Users/maciejwiatrak/Downloads/epoch=28-val_r2=0.7911.ckpt",
        # batch_size=args.batch_size,
        batch_size=10,
    )


if __name__ == "__main__":
    args = DeepGeneBacArgumentParser().parse_args()
    main(args)
