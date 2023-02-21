import logging
import os
from typing import Optional, Literal

from pytorch_lightning.utilities.seed import seed_everything

from deep_bac.argparser import DeepBacArgumentParser
from deep_bac.data_preprocessing.data_reader import get_gene_reg_data
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.model_gene_reg import DeepBacGeneReg
from deep_bac.modelling.trainer import get_trainer
from deep_bac.utils import get_selected_genes, format_and_write_results

logging.basicConfig(level=logging.INFO)


def run(
    config: DeepBacConfig,
    input_dir: str,
    output_dir: str,
    n_highly_variable_genes: int = 500,
    use_drug_idx: int = None,
    max_gene_length: int = 2048,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.5,
    num_workers: int = None,
    test: bool = False,
    ckpt_path: Optional[str] = None,
    use_drug_specific_genes: Literal["INH", "Walker", "MD-CNN"] = "MD-CNN",
    test_after_train: bool = False,
):
    selected_genes = get_selected_genes(use_drug_specific_genes)
    logging.info(f"Selected genes: {selected_genes}")

    data = get_gene_reg_data(
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
        regression=config.regression,
        use_drug_idx=use_drug_idx,
        batch_size=config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        selected_genes=selected_genes,
        test=any([test, test_after_train]),
    )
    logging.info("Finished loading data")

    config.train_set_len = data.train_set_len
    if use_drug_idx is not None:
        config.n_output = 1
    config.n_highly_variable_genes = (
        len(selected_genes) if selected_genes else n_highly_variable_genes
    )

    trainer = get_trainer(config, output_dir)
    model = DeepBacGeneReg(config)

    results = None
    if test and ckpt_path:
        results = trainer.test(
            model, dataloaders=data.test_dataloader, ckpt_path=ckpt_path
        )
    else:
        trainer.fit(model, data.train_dataloader, data.val_dataloader)

    if test_after_train:
        results = trainer.test(
            model,
            dataloaders=data.test_dataloader,
            ckpt_path="best",
        )
    # in the future we could save the results
    return results


def main(args):
    seed_everything(args.random_state)
    config = DeepBacConfig.from_dict(args.as_dict())
    results = run(
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
        use_drug_idx=args.use_drug_idx,
        use_drug_specific_genes=args.use_drug_specific_genes,
        test_after_train=args.test_after_train,
    )
    format_and_write_results(
        results=results,
        output_file_path=os.path.join(args.output_dir, "test_results.jsonl"),
    )


if __name__ == "__main__":
    args = DeepBacArgumentParser().parse_args()
    main(args)
