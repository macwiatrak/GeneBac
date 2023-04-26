import logging
import os
from typing import Optional, Dict, Tuple

import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything

from deep_bac.argparser import DeepGeneBacArgumentParser
from deep_bac.data_preprocessing.data_reader import get_gene_expr_data
from deep_bac.data_preprocessing.utils import get_gene_std_expression
from deep_bac.modelling.data_types import DeepGeneBacConfig
from deep_bac.modelling.model_gene_expr import DeepBacGeneExpr
from deep_bac.modelling.trainer import get_trainer
from deep_bac.utils import get_gene_var_thresholds, GENE_STD_THRESHOLDS_DICT

logging.basicConfig(level=logging.INFO)


def run(
    config: DeepGeneBacConfig,
    input_dir: str,
    output_dir: str,
    max_gene_length: int = 2560,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = None,
    test: bool = False,
    ckpt_path: Optional[str] = None,
    gene_std_thresholds: Dict[
        str, Tuple[float, float]
    ] = GENE_STD_THRESHOLDS_DICT,
    test_after_train: bool = False,
    resume_from_ckpt_path: str = None,
):
    data = get_gene_expr_data(
        input_dir=input_dir,
        max_gene_length=max_gene_length,
        batch_size=config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        test=test,
    )
    logging.info("Finished loading data")

    config.train_set_len = data.train_set_len
    # this should always be true for gene expression prediction
    trainer = get_trainer(
        config,
        output_dir,
        resume_from_ckpt_path=resume_from_ckpt_path,
        refresh_rate=1000,
    )

    gene_std_dict = get_gene_std_expression(
        df=pd.read_parquet(os.path.join(input_dir, "val.parquet"))
    )
    gene_vars_w_thresholds = get_gene_var_thresholds(
        gene_std_dict=gene_std_dict,
        gene_std_thresholds=gene_std_thresholds,
    )
    model = DeepBacGeneExpr(
        config=config,
        gene_vars_w_thresholds=gene_vars_w_thresholds,
    )

    if test:
        return trainer.test(
            model,
            dataloaders=data.test_dataloader,
            ckpt_path=ckpt_path,
        )

    trainer.fit(
        model, data.train_dataloader, data.val_dataloader, ckpt_path=ckpt_path
    )

    if test_after_train:
        return trainer.test(
            model,
            dataloaders=data.test_dataloader,
            ckpt_path="best",
        )
    return None


def main(args):
    seed_everything(args.random_state)
    config = DeepGeneBacConfig.from_dict(args.as_dict())
    _ = run(
        config=config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_gene_length=args.max_gene_length,
        shift_max=args.shift_max,
        pad_value=args.pad_value,
        reverse_complement_prob=args.reverse_complement_prob,
        num_workers=args.num_workers,
        test=args.test,
        ckpt_path=args.ckpt_path,
        test_after_train=args.test_after_train,
        resume_from_ckpt_path=args.resume_from_ckpt_path,
    )


if __name__ == "__main__":
    args = DeepGeneBacArgumentParser().parse_args()
    main(args)
