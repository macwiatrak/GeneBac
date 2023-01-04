from typing import Optional

from pytorch_lightning.utilities.seed import seed_everything

from deep_bac.argparser import TrainArgumentParser
from deep_bac.data_preprocessing.data_reader import get_data
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.model import DeepBac
from deep_bac.modelling.modules.trainer import get_trainer
from deep_bac.utils import get_config


def run(
        config: DeepBacConfig,
        input_df_file_path: str,
        reference_gene_seqs_dict_path: str,
        phenotype_df_file_path: str,
        train_val_test_split_indices_file_path: str,
        selected_genes_file_path: Optional[str] = None,
        max_gene_length: int = 2048,
        shift_max: int = 3,
        pad_value: float = 0.25,
        reverse_complement_prob: float = 0.5,
        num_workers: int = 8,
        test: bool = False,
        ckpt_path: Optional[str] = None,
):
    data = get_data(
        input_df_file_path=input_df_file_path,
        reference_gene_seqs_dict_path=reference_gene_seqs_dict_path,
        phenotype_df_file_path=phenotype_df_file_path,
        train_val_test_split_indices_file_path=train_val_test_split_indices_file_path,
        max_gene_length=max_gene_length,
        selected_genes_file_path=selected_genes_file_path,
        batch_size=config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers,
    )
    config.train_set_len = len(data.train_dataloader) * config.batch_size
    trainer = get_trainer(config)
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
    config = get_config(args)
    _ = run(
        config=config,
        input_df_file_path=args.input_df_file_path,
        reference_gene_seqs_dict_path=args.reference_gene_seqs_dict_path,
        phenotype_df_file_path=args.phenotype_df_file_path,
        train_val_test_split_indices_file_path=args.train_val_test_split_indices_file_path,
        selected_genes_file_path=args.selected_genes_file_path,
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
