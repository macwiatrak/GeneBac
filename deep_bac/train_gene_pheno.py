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
    use_drug_idx: int = None,
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
    fold_idx: int = None,
    gene_encoder_ckpt_path: str = None,
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
            input_dir, "train_test_cv_split_unq_ids.json"
        ),
        variance_per_gene_file_path=os.path.join(
            input_dir, "unnormalised_variance_per_gene.csv"
        ),
        max_gene_length=config.max_gene_length,
        n_highly_variable_genes=config.n_highly_variable_genes,
        regression=config.regression,
        use_drug_idx=use_drug_idx,
        batch_size=config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        selected_genes=selected_genes,
        test=any([test, test_after_train]),
        fold_idx=fold_idx,
    )
    logging.info(f"Fold index: {fold_idx}")
    val_dataloader = data.val_dataloader if fold_idx is not None else []

    logging.info("Finished loading data")

    config.train_set_len = data.train_set_len
    if use_drug_idx is not None:
        config.n_output = 1
    config.n_highly_variable_genes = (
        len(selected_genes)
        if selected_genes
        else config.n_highly_variable_genes
    )
    config.gene_to_idx = data.gene_to_idx
    logging.info(f"Gene to idx: {config.gene_to_idx}")

    trainer = get_trainer(
        config, output_dir, resume_from_ckpt_path=resume_from_ckpt_path
    )
    model = DeepBacGenePheno(config)

    if gene_encoder_ckpt_path is not None:
        print("Loading gene encoder weights")
        gene_enc_sd = torch.load(gene_encoder_ckpt_path, map_location="cpu")[
            "state_dict"
        ]
        gene_encoder_sd = {
            k.lstrip("gene_encoder."): v
            for k, v in gene_enc_sd.items()
            if k.startswith("gene_encoder")
        }
        model.gene_encoder.load_state_dict(gene_encoder_sd)

    if test:
        model = DeepBacGenePheno.load_from_checkpoint(
            ckpt_path,
        )
        model.drug_thresholds = get_drug_thresholds(
            model, data.train_dataloader
        )
        return trainer.test(
            model,
            dataloaders=data.test_dataloader,
        )

    trainer.fit(model, data.train_dataloader, val_dataloaders=val_dataloader)

    if test_after_train:
        best_model = DeepBacGenePheno.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        best_model.drug_thresholds = get_drug_thresholds(
            best_model, data.train_dataloader
        )
        return trainer.test(
            best_model,
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
        fold_idx=args.fold_idx,
        gene_encoder_ckpt_path=args.gene_encoder_ckpt_path,
    )
    format_and_write_results(
        results=results,
        output_file_path=os.path.join(args.output_dir, "test_results.jsonl"),
    )


if __name__ == "__main__":
    args = DeepGeneBacArgumentParser().parse_args()
    main(args)
