from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from deep_bac.data_preprocessing.data_types import BacGenesInputSample
from deep_bac.data_preprocessing.utils import ONE_HOT_EMBED


class BacterialGenomeDataset(Dataset):
    def __init__(
            self,
            bac_genes_df_file_path: str,
            reference_gene_seqs_dict: Dict[str, str],
            phenotype_dataframe_file_path: str = None,
            max_gene_length: int = 2048,
            selected_genes: List = None,
            pad_value: float = 0.25,
    ):
        self.genes_df = pd.read_parquet(bac_genes_df_file_path)
        # get unique ids
        self.unique_ids = list(sorted(self.genes_df.index.levels[0]))

        self.id_to_labels = None
        if phenotype_dataframe_file_path is not None:
            self.phenotypes_df = pd.read_parquet(phenotype_dataframe_file_path)
            self.id_to_labels = dict(zip(self.phenotypes_df['UNIQUEID'], self.phenotypes_df['LOG2MIC']))

        self.reference_gene_seqs_dict = reference_gene_seqs_dict
        self.max_gene_length = max_gene_length
        self.pad_value = pad_value

        self.selected_genes = selected_genes if selected_genes is not None else list(reference_gene_seqs_dict.keys())
        self.gene_to_id = {gene: i for i, gene in enumerate(reference_gene_seqs_dict.keys()) if gene in selected_genes}

    def __getitem__(self, idx):
        unq_id = self.unique_ids[idx]
        unq_id_subset = self.genes_df.df.xs(unq_id, level='UNIQUEID')
        unq_id_genes = unq_id_subset['gene'].tolist()

        genes_tensor = []
        variants_in_gene = []
        for gene in self.gene_to_id.keys():
            if gene in unq_id_genes['gene']:
                idx = unq_id_genes.index(gene)
                seq = unq_id_subset.iloc[idx]['prom_gene_seq_w_variants']
                variants_in_gene.append(1)
            else:
                seq = self.reference_gene_seqs_dict[gene]
                variants_in_gene.append(0)
            one_hot_seq = seq_to_one_hot(seq[:self.max_gene_length])
            padded_one_hot_seq = pad_one_hot_seq(one_hot_seq, self.max_gene_length, self.pad_value)
            genes_tensor.append(padded_one_hot_seq)
        genes_tensor = torch.stack(genes_tensor)
        variants_in_gene = torch.tensor(variants_in_gene)

        labels = None
        if self.id_to_labels is not None:
            labels = torch.tensor(self.id_to_labels[self.genes_df.iloc[idx]['UNIQUEID']], dtype=torch.float32)

        return BacGenesInputSample(
            genes_tensor=genes_tensor,
            variants_in_gene=variants_in_gene,
            labels=labels if labels else None,
            unique_id=unq_id,
        )

    def __len__(self):
        return len(self.unique_ids)


def seq_to_one_hot(seq: str) -> torch.Tensor:
    seq_chrs = torch.tensor(list(map(ord, list(seq))), dtype=torch.long)
    return ONE_HOT_EMBED[seq_chrs]


def pad_one_hot_seq(one_hot_seq: torch.Tensor, max_length: int, pad_value: float = 0.25) -> torch.Tensor:
    seq_length, n_nucleotides = one_hot_seq.shape
    if seq_length == max_length:
        return one_hot_seq
    to_pad = max_length - seq_length
    return torch.cat([one_hot_seq, torch.full((to_pad, n_nucleotides), pad_value)], dim=0)
