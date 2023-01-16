import logging
import os
from typing import Optional

from pytorch_lightning.utilities.seed import seed_everything

from deep_bac.argparser import TrainArgumentParser
from deep_bac.data_preprocessing.data_reader import get_data
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.model import DeepBac
from deep_bac.modelling.modules.trainer import get_trainer


logging.basicConfig(level=logging.INFO)


def run(
        config: DeepBacConfig,
        input_dir: str,
        output_dir: str,
        n_highly_variable_genes: int = 500,
        max_gene_length: int = 2048,
        shift_max: int = 3,
        pad_value: float = 0.25,
        reverse_complement_prob: float = 0.5,
        num_workers: int = 8,
        test: bool = False,
        ckpt_path: Optional[str] = None,
):
    data = get_data(
        input_df_file_path=os.path.join(input_dir, "processed_agg_variants.parquet"),
        reference_gene_seqs_dict_path=os.path.join(input_dir, 'reference_gene_seqs.json'),
        phenotype_df_file_path=os.path.join(input_dir, "phenotype_labels_with_binary_labels.parquet"),
        train_val_test_split_indices_file_path=os.path.join(input_dir, "train_val_test_split_unq_ids.json"),
        variance_per_gene_file_path=os.path.join(input_dir, "unnormalised_variance_per_gene.csv"),
        max_gene_length=max_gene_length,
        n_highly_variable_genes=n_highly_variable_genes,
        batch_size=config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers,
    )
    logging.info("Finished loading data")

    config.train_set_len = data.train_set_len
    trainer = get_trainer(config, output_dir)
    model = DeepBac(config)

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
        n_highly_variable_genes=args.n_highly_variable_genes,
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
