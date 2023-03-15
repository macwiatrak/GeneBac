import logging
import os
from typing import Optional, Literal

import torch
from pytorch_lightning.utilities.seed import seed_everything

from deep_bac.argparser import DeepGeneBacArgumentParser
from deep_bac.data_preprocessing.data_reader import get_gene_pheno_data
from deep_bac.modelling.data_types import DeepGeneBacConfig
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno
from deep_bac.modelling.trainer import get_trainer
from deep_bac.modelling.utils import get_drug_thresholds
from deep_bac.utils import get_selected_genes, format_and_write_results

logging.basicConfig(level=logging.INFO)


def run(
    config: DeepGeneBacConfig,
    input_dir: str,
    output_dir: str,
    n_highly_variable_genes: int = 500,
    use_drug_idx: int = None,
    max_gene_length: int = 2560,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.5,
    num_workers: int = None,
    test: bool = False,
    ckpt_path: Optional[str] = None,
    use_drug_specific_genes: Literal[
        "INH", "Walker", "MD-CNN", "cryptic"
    ] = "cryptic",
    test_after_train: bool = False,
    resume_from_ckpt_path: str = None,
):
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

    trainer = get_trainer(
        config, output_dir, resume_from_ckpt_path=resume_from_ckpt_path
    )
    model = DeepBacGenePheno(config)

    if test:
        model = model.load_from_checkpoint(ckpt_path)
        drug_thresholds = get_drug_thresholds(model, data.val_dataloader)
        model.drug_thresholds = drug_thresholds
        print("Val set results with thresholds tuned on val:")
        _ = trainer.test(
            model,
            dataloaders=data.val_dataloader,
        )
        print("Test set results with thresholds tuned on val:")
        _ = trainer.test(
            model,
            dataloaders=data.test_dataloader,
        )
        print("Test set results with 0.5 thresholds:")
        model.drug_thresholds = torch.ones(14) * 0.5
        _ = trainer.test(
            model,
            dataloaders=data.test_dataloader,
        )

    trainer.fit(model, data.train_dataloader, data.val_dataloader)

    if test_after_train:
        model = model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        drug_thresholds = get_drug_thresholds(model, data.val_dataloader)
        model.drug_thresholds = drug_thresholds
        return trainer.test(
            model,
            dataloaders=data.test_dataloader,
        )
    return None


def main(args):
    seed_everything(args.random_state)
    config = DeepGeneBacConfig.from_dict(args.as_dict())
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
        resume_from_ckpt_path=args.resume_from_ckpt_path,
    )
    format_and_write_results(
        results=results,
        output_file_path=os.path.join(args.output_dir, "test_results.jsonl"),
    )


if __name__ == "__main__":
    args = DeepGeneBacArgumentParser().parse_args()
    main(args)
