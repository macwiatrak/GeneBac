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
from deep_bac.data_preprocessing.data_reader import get_gene_expr_dataloader
from deep_bac.modelling.model_gene_expr import DeepBacGeneExpr

logging.basicConfig(level=logging.INFO)


def collect_tfmodisco_data(
    attr_model_fn: Callable, dataloader: DataLoader, device: str
) -> csr_matrix:  # Tuple[csr_matrix, csr_matrix]:
    # dna_tensor = []
    importance_scores = []
    for idx, batch in enumerate(tqdm(dataloader, mininterval=5)):
        attrs = attr_model_fn.attribute(
            batch.input_tensor.to(device),
            additional_forward_args=batch.tss_indexes.to(device),
            return_convergence_delta=False,
        )
        # dna_tensor.append(batch.input_tensor)
        importance_scores.append(attrs.detach().cpu())

    # dna_tensor = torch.cat(dna_tensor, dim=0).numpy()
    importance_scores = torch.cat(importance_scores, dim=0).numpy()
    return importance_scores


def run(
    ckpt_path: str,
    input_dir: str,
    output_dir: str,
    max_gene_length: int = 2560,
    shift_max: int = 0,
    pad_value: float = 0.25,
    num_workers: int = None,
    batch_size: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    model = DeepBacGeneExpr.load_from_checkpoint(ckpt_path)
    # for testing
    # model = DeepBacGeneExpr(config=DeepGeneBacConfig())
    model.to(device)
    model.eval()
    attr_model_fn = DeepLift(model)

    dataloader, _ = get_gene_expr_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path=os.path.join(input_dir, "val.parquet"),
        max_gene_length=max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=0.0,  # set it to 0 during eval
        shuffle=False,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        pin_memory=True,
    )

    logging.info("Finished loading val data")

    importance_scores_tensor = collect_tfmodisco_data(
        attr_model_fn,
        dataloader,
        device=device,
    )
    logging.info("Finished collecting data on val, saving it now...")
    # np.save(os.path.join(output_dir, "val_dna_tensor.npy"), dna_tensor)
    np.save(
        os.path.join(output_dir, "val_importance_scores_tensor.npy"),
        importance_scores_tensor,
    )
    logging.info("Finished saving val data")
    del importance_scores_tensor

    dataloader, _ = get_gene_expr_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path=os.path.join(input_dir, "test.parquet"),
        max_gene_length=max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=0.0,  # set it to 0 during eval
        shuffle=False,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        pin_memory=True,
    )

    logging.info("Finished loading test data")

    importance_scores_tensor = collect_tfmodisco_data(
        attr_model_fn,
        dataloader,
        device=device,
    )
    logging.info("Finished collecting data on test, saving it now...")
    # np.save(os.path.join(output_dir, "test_dna_tensor.npy"), dna_tensor)
    np.save(
        os.path.join(output_dir, "test_importance_scores_tensor.npy"),
        importance_scores_tensor,
    )
    logging.info("Finished saving test data")
    # del dna_tensor
    del importance_scores_tensor


def main(args):
    seed_everything(args.random_state)
    run(
        input_dir=args.input_dir,
        # input_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/pseudomonas/prom-200-w-rev-comp/",
        output_dir=args.output_dir,
        # output_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/pseudomonas/modisco/val_data_sampled/",
        max_gene_length=args.max_gene_length,
        shift_max=0,
        pad_value=args.pad_value,
        num_workers=args.num_workers,
        ckpt_path=args.ckpt_path,
        # ckpt_path="/Users/maciejwiatrak/Downloads/epoch=80-val_r2=0.7983.ckpt",
        batch_size=args.batch_size,
        # batch_size=100,
    )


if __name__ == "__main__":
    args = DeepGeneBacArgumentParser().parse_args()
    main(args)
