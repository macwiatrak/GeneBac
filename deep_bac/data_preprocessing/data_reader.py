import json
import os
from typing import List, Tuple, Dict

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.data_preprocessing.data_types import (
    BacInputSample,
    BatchBacInputSample,
    DataReaderOutput,
)
from deep_bac.data_preprocessing.dataset import (
    BacGenomeGenePhenoDataset,
    BacGenomeGeneExprDataset,
)
from deep_bac.data_preprocessing.utils import get_gene_std_expression

VARIANCE_PER_GENE_FILE_PATH = (
    "/Users/maciejwiatrak/Desktop/bacterial_genomics/"
    "cryptic/unnormalised_variance_per_gene.csv"
)


def _collate_samples(data: List[BacInputSample]) -> BatchBacInputSample:
    genes_tensor = torch.stack([sample.input_tensor for sample in data])

    variants_in_gene = [sample.variants_in_gene for sample in data]
    if None not in variants_in_gene:
        variants_in_gene = torch.stack(variants_in_gene)

    labels = [sample.labels for sample in data]
    if None not in labels:
        labels = torch.stack(labels)

    tss_indexes = [sample.tss_index for sample in data]
    if None not in tss_indexes:
        tss_indexes = torch.stack(tss_indexes)

    gene_names = [sample.gene_name for sample in data]
    unique_ids = [sample.strain_id for sample in data]
    return BatchBacInputSample(
        input_tensor=genes_tensor,
        variants_in_gene=variants_in_gene,
        labels=labels,
        tss_indexes=tss_indexes,
        strain_ids=unique_ids,
        gene_names=gene_names,
    )


def get_gene_pheno_dataloader(
    batch_size: int,
    bac_genes_df_file_path: str,
    reference_gene_data_df: DataFrame,
    unique_ids: List[str] = None,
    phenotype_dataframe_file_path: str = None,
    max_gene_length: int = 2560,
    selected_genes: List = None,
    regression: bool = False,
    use_drug_idx: int = None,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Dict[str, int]]:
    dataset = BacGenomeGenePhenoDataset(
        unique_ids=unique_ids,
        bac_genes_df_file_path=bac_genes_df_file_path,
        reference_gene_data_df=reference_gene_data_df,
        phenotype_dataframe_file_path=phenotype_dataframe_file_path,
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        regression=regression,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        use_drug_idx=use_drug_idx,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_samples,
    )
    return dataloader, dataset.gene_to_idx


def get_gene_pheno_data(
    input_df_file_path: str,
    reference_gene_data_df_path: str,
    phenotype_df_file_path: str,
    train_val_test_split_indices_file_path: str,
    variance_per_gene_file_path: str,
    n_highly_variable_genes: int = 500,
    selected_genes: List = None,
    regression: bool = False,
    use_drug_idx: int = None,
    batch_size: int = 8,
    max_gene_length: int = 2560,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = 8,
    test: bool = False,
    fold_idx: int = None,
):
    reference_gene_data_df = pd.read_parquet(reference_gene_data_df_path)

    if selected_genes is None:
        selected_genes = pd.read_csv(variance_per_gene_file_path)[
            "Gene"
        ].tolist()[:n_highly_variable_genes]

    with open(train_val_test_split_indices_file_path, "r") as f:
        train_val_test_split_indices = json.load(f)

    if fold_idx is not None:
        train_unique_ids = train_val_test_split_indices[
            f"train_fold_{fold_idx}"
        ]
        val_unique_ids = train_val_test_split_indices[f"val_fold_{fold_idx}"]
    else:
        train_unique_ids = train_val_test_split_indices["train"]
        # we are not doing validation so make the list empty
        val_unique_ids = []

    test_unique_ids = train_val_test_split_indices["test"]

    train_dataloader, gene_to_id = get_gene_pheno_dataloader(
        batch_size=batch_size,
        unique_ids=train_unique_ids,
        bac_genes_df_file_path=input_df_file_path,
        reference_gene_data_df=reference_gene_data_df,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        regression=regression,
        use_drug_idx=use_drug_idx,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_dataloader, _ = get_gene_pheno_dataloader(
        batch_size=2
        * batch_size,  # double the batch size during eval for speed up
        unique_ids=val_unique_ids,
        bac_genes_df_file_path=input_df_file_path,
        reference_gene_data_df=reference_gene_data_df,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        regression=regression,
        use_drug_idx=use_drug_idx,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=0.0,  # set it to 0 during eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if not test:
        return DataReaderOutput(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_set_len=len(train_unique_ids),
            gene_to_idx=gene_to_id,
        )
    test_dataloader, _ = get_gene_pheno_dataloader(
        batch_size=2
        * batch_size,  # double the batch size during eval for speed up
        unique_ids=test_unique_ids,
        bac_genes_df_file_path=input_df_file_path,
        reference_gene_data_df=reference_gene_data_df,
        phenotype_dataframe_file_path=phenotype_df_file_path,
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        regression=regression,
        use_drug_idx=use_drug_idx,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=0.0,  # set it to 0 during eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return DataReaderOutput(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        train_set_len=len(train_unique_ids),
        gene_to_idx=gene_to_id,
    )


def get_gene_expr_dataloader(
    batch_size: int,
    bac_genes_df_file_path: str,
    max_gene_length: int = 2560,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, int]:
    dataset = BacGenomeGeneExprDataset(
        bac_genes_df_file_path=bac_genes_df_file_path,
        max_gene_length=max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_samples,
    )
    return dataloader, dataset.__len__()


def get_gene_expr_data(
    input_dir: str,
    batch_size: int = 512,
    max_gene_length: int = 2560,
    shift_max: int = 3,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = 8,
    test: bool = False,
) -> DataReaderOutput:

    train_dataloader, train_set_len = get_gene_expr_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path=os.path.join(input_dir, "train.parquet"),
        max_gene_length=max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader, _ = get_gene_expr_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path=os.path.join(input_dir, "val.parquet"),
        max_gene_length=max_gene_length,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=0.0,  # set it to 0 during eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if not test:
        return DataReaderOutput(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            train_set_len=train_set_len,
        )
    test_dataloader, _ = get_gene_expr_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path=os.path.join(input_dir, "test.parquet"),
        shift_max=shift_max,
        max_gene_length=max_gene_length,
        pad_value=pad_value,
        reverse_complement_prob=0.0,  # set it to 0 during eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return DataReaderOutput(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        train_set_len=train_set_len,
    )


def main():
    # input_dir = "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/"
    # with open(os.path.join(input_dir, "reference_gene_seqs.json"), "r") as f:
    #     reference_gene_seqs_dict = json.load(f)
    #
    # gene_variance_df = pd.read_csv(
    #     os.path.join(input_dir, "unnormalised_variance_per_gene.csv")
    # )
    # selected_genes = gene_variance_df["Gene"].tolist()[:1000]
    #
    # max_gene_length = 2560
    # dl = get_gene_reg_dataloader(
    #     batch_size=32,
    #     unique_ids=None,
    #     bac_genes_df_file_path=os.path.join(
    #         input_dir, "processed-genome-per-strain", "agg_variants.parquet"
    #     ),
    #     reference_gene_seqs_dict=reference_gene_seqs_dict,
    #     phenotype_dataframe_file_path=os.path.join(
    #         input_dir, "phenotype_labels_with_binary_labels.parquet"
    #     ),
    #     max_gene_length=max_gene_length,
    #     selected_genes=selected_genes,
    #     shift_max=3,
    #     pad_value=0.25,
    #     reverse_complement_prob=0.5,
    #     shuffle=True,
    #     num_workers=os.cpu_count(),
    #     pin_memory=False,
    # )
    #
    # for _ in tqdm(dl):
    #     pass
    # print("Done!")
    data, most_var_genes = get_gene_expr_data(
        input_dir="/Users/maciejwiatrak/Desktop/bacterial_genomics/pseudomonas/",
        batch_size=512,
        max_gene_length=2560,
        shift_max=3,
        pad_value=0.25,
        reverse_complement_prob=0.5,
        num_workers=8,
        test=False,
    )
    for _ in tqdm(data.val_dataloader):
        pass
    print("Val done!")
    for _ in tqdm(data.train_dataloader):
        pass
    print("Train done!")


if __name__ == "__main__":
    main()
