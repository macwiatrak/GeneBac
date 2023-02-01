import torch

from deep_bac.data_preprocessing.datasets.dataset_expression import (
    GeneExprDataset,
)
from deep_bac.data_preprocessing.datasets.dataset_dna import DnaGeneRegDataset


def test_bac_genome_gene_reg_dataset():
    selected_genes = ["PE1", "Rv1716", "Rv2000", "pepC", "pepD"]

    dataset_len = 6
    n_genes = len(selected_genes)
    n_nucleotides = 4
    max_gene_length = 1000
    reference_gene_seqs_dict = {gene: "atcgt" * 100 for gene in selected_genes}

    dataset = DnaGeneRegDataset(
        unique_ids=None,
        bac_genes_df_file_path="../test_data/sample_agg_variants.parquet",
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        phenotype_dataframe_file_path="../test_data/phenotype_labels_with_binary_labels.parquet",
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        regression=True,
    )
    data = [dataset[i] for i in range(dataset.__len__())]
    assert len(data) == 6
    assert torch.stack(
        [item.input_tensor for item in data]
    ).shape == torch.Size(
        [dataset_len, n_genes, n_nucleotides, max_gene_length]
    )

    item = data[0]
    assert item.variants_in_gene.tolist() == [1, 1, 1, 1, 1]
    assert int(item.labels.mean().item()) == -28
    assert item.strain_id == "site.01.subj.DR0011.lab.DR0011.iso.1"

    assert all(
        [torch.all(item.input_tensor.sum(dim=1) == 1.0).item() for item in data]
    )


def test_bac_genome_gene_expr_dataset():
    dataset_len = 1000
    n_nucleotides = 4
    max_gene_length = 2048

    dataset = GeneExprDataset(
        bac_genes_df_file_path="../test_data/sample_genes_with_variants_and_expression.parquet",
        max_gene_length=max_gene_length,
        shift_max=3,
        pad_value=0.25,
        reverse_complement_prob=0.5,
    )
    data = [dataset[i] for i in range(dataset.__len__())]
    assert len(data) == dataset_len
    assert torch.stack(
        [item.input_tensor for item in data]
    ).shape == torch.Size([dataset_len, n_nucleotides, max_gene_length])

    item = data[0]
    assert item.variants_in_gene == 1
    assert torch.allclose(item.labels, torch.tensor(2.3919))
    assert item.strain_id == "ZG302367"
    assert item.gene_name == "PA0001"

    assert all(
        [torch.all(item.input_tensor.sum(dim=0) == 1.0).item() for item in data]
    )
