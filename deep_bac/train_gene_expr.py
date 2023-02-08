import logging
import os
from typing import Optional, List

from pytorch_lightning.utilities.seed import seed_everything

from deep_bac.argparser import TrainArgumentParser
from deep_bac.data_preprocessing.data_reader import get_gene_expr_data
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.model_gene_expr import DeepBacGeneExpr
from deep_bac.modelling.trainer import get_trainer
from deep_bac.utils import get_gene_var_thresholds

logging.basicConfig(level=logging.INFO)


def run(
    config: DeepBacConfig,
    input_dir: str,
    output_dir: str,
    max_gene_length: int = 2048,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.5,
    num_workers: int = None,
    test: bool = False,
    ckpt_path: Optional[str] = None,
    gene_var_thresholds: List[float] = [0.05, 0.1, 0.25, 0.5],
):
    data, most_variable_genes = get_gene_expr_data(
        input_dir=input_dir,
        max_gene_length=max_gene_length,
        batch_size=config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
    )
    logging.info("Finished loading data")

    config.train_set_len = data.train_set_len
    # this should always be true for gene expression prediction
    trainer = get_trainer(config, output_dir)
    model = DeepBacGeneExpr(
        config=config,
        gene_vars_w_thresholds=get_gene_var_thresholds(
            most_variable_genes, gene_var_thresholds
        ),
    )

    results = None
    if test:
        results = trainer.test(
            model, dataloaders=data.test_dataloader, ckpt_path=ckpt_path
        )
    else:
        trainer.fit(model, data.train_dataloader, data.val_dataloader)
    # in the future we could save the results
    return results


def main(args):
    seed_everything(args.random_state)
    config = DeepBacConfig.from_dict(args.as_dict())
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
    )


if __name__ == "__main__":
    args = TrainArgumentParser().parse_args()
    main(args)
