import logging
import os
from typing import Literal


from deep_bac.data_preprocessing.data_reader import get_gene_pheno_data
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno
from deep_bac.utils import get_selected_genes
from experiments.latent_representations.utils import collect_strain_reprs

logging.basicConfig(level=logging.INFO)


def run(
    ckpt_path: str,
    input_dir: str,
    output_dir: str,
    n_highly_variable_genes: int = 500,
    use_drug_idx: int = None,
    max_gene_length: int = 2560,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = None,
    test: bool = True,
    use_drug_specific_genes: Literal["INH", "Walker", "MD-CNN"] = "MD-CNN",
):

    model = DeepBacGenePheno.load_from_checkpoint(ckpt_path)
    model.eval()

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
        regression=model.config.regression,
        use_drug_idx=use_drug_idx,
        batch_size=model.config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        selected_genes=selected_genes,
        test=test,
    )
    logging.info("Finished loading data")

    val_df = collect_strain_reprs(model, data.val_dataloader)
    val_df.to_parquet(
        os.path.join(output_dir, "val_strain_representations.parquet")
    )

    if test:
        test_df = collect_strain_reprs(model, data.test_dataloader)
        test_df.to_parquet(
            os.path.join(output_dir, "test_strain_representations.parquet")
        )
