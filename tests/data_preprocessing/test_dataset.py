import torch

from deep_bac.data_preprocessing.dataset import BacterialGenomeDataset


def test_dataset():
    selected_genes = [
        'PE1', 'Rv1716', 'Rv2000', 'pepC', 'pepD'
    ]

    dataset_len = 6
    n_genes = len(selected_genes)
    n_nucleotides = 4
    max_gene_length = 1000
    reference_gene_seqs_dict = {gene: "atcgt" * 100 for gene in selected_genes}

    dataset = BacterialGenomeDataset(
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
    assert torch.stack([item.genes_tensor for item in data]).shape == torch.Size(
        [dataset_len, n_genes, n_nucleotides, max_gene_length])

    item = data[0]
    assert item.variants_in_gene.tolist() == [1, 1, 1, 1, 1]
    assert int(item.labels.mean().item()) == -29
    assert item.unique_id == 'site.01.subj.DR0011.lab.DR0011.iso.1'

    assert all([torch.all(item.genes_tensor.sum(dim=1) == 1.).item() for item in data])
