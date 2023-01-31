import random
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from deep_bac.data_preprocessing.data_types import BacInputSample
from deep_bac.data_preprocessing.utils import (
    seq_to_one_hot,
    shift_seq,
    pad_one_hot_seq,
)


class DnaGeneRegDataset(Dataset):
    def __init__(
        self,
        bac_genes_df_file_path: str,
        reference_gene_seqs_dict: Dict[str, str],
        unique_ids: List[str] = None,
        phenotype_dataframe_file_path: str = None,
        max_gene_length: int = 2048,
        selected_genes: List = None,
        regression: bool = False,  # whether the task should be regression or binary classification
        shift_max: int = 3,
        pad_value: float = 0.25,
        reverse_complement_prob: float = 0.5,
        use_drug_idx: int = None,
    ):
        self.genes_df = pd.read_parquet(bac_genes_df_file_path)
        self.use_drug_idx = use_drug_idx
        # get unique ids
        self.unique_ids = unique_ids
        if not self.unique_ids:
            self.unique_ids = list(sorted(self.genes_df.index.levels[0]))

        self.id_to_labels_df = None
        self.label_column = "LOGMIC_LABELS" if regression else "BINARY_LABELS"
        if phenotype_dataframe_file_path is not None:
            # keep it a sa dataframe, not dict due to pytorch memory leakage issue
            self.id_to_labels_df = pd.read_parquet(
                phenotype_dataframe_file_path, columns=[self.label_column]
            )
        if use_drug_idx is not None:
            unq_ids = [
                idx
                for idx, row in self.id_to_labels_df.iterrows()
                if row[self.label_column][use_drug_idx] != -100.0
            ]
            self.unique_ids = [idx for idx in self.unique_ids if idx in unq_ids]

        self.max_gene_length = max_gene_length
        self.shift_max = shift_max
        self.pad_value = pad_value
        self.reverse_complement_prob = reverse_complement_prob

        self.selected_genes = (
            selected_genes
            if selected_genes is not None
            else list(reference_gene_seqs_dict.keys())
        )
        self.gene_to_id = {
            gene: i
            for i, gene in enumerate(reference_gene_seqs_dict.keys())
            if gene in selected_genes
        }

        # convert reference gene seqs to a dataframe to avoid memory leak
        # due to a pytorch issue with native python data structures
        # as it's a big dictionary
        self.reference_gene_seqs_df = pd.DataFrame(
            {
                "gene": list(self.gene_to_id.keys()),
                "seq": [
                    reference_gene_seqs_dict[gene]
                    for gene in self.gene_to_id.keys()
                ],
            }
        )

    def __getitem__(self, idx):
        unq_id = self.unique_ids[idx]
        unq_id_subset = self.genes_df.xs(unq_id, level="UNIQUEID")
        unq_id_genes = unq_id_subset["gene"].tolist()

        genes_tensor = []
        variants_in_gene = []
        for idx, gene in enumerate(self.gene_to_id.keys()):
            if gene in unq_id_genes:
                idx = unq_id_genes.index(gene)
                seq = unq_id_subset.iloc[idx]["prom_gene_seq_w_variants"]
                variants_in_gene.append(1)
            else:
                seq = self.reference_gene_seqs_df.iloc[idx]["seq"]
                variants_in_gene.append(0)
            # subset it to the max gene length
            one_hot_seq = seq_to_one_hot(seq[: self.max_gene_length])
            # stochastically do a reverse complement of the sequence
            if random.random() > self.reverse_complement_prob:
                # we can do reverse complement by flipping the one hot encoding
                # because of symmetricity in the one hot encoding of ACGT
                one_hot_seq = torch.flip(one_hot_seq, [0, 1])
            # randomly shift the DNA sequence
            random_shift = random.randint(
                -self.shift_max, self.shift_max
            )  # get random shift
            one_hot_seq = shift_seq(
                one_hot_seq, random_shift, self.pad_value
            )  # shift seq
            # pad one hot seq
            padded_one_hot_seq = pad_one_hot_seq(
                one_hot_seq, self.max_gene_length, self.pad_value
            )
            # transpose the one hot seq to make it channels first
            genes_tensor.append(padded_one_hot_seq.T)

        genes_tensor = torch.stack(genes_tensor)
        variants_in_gene = torch.tensor(variants_in_gene, dtype=torch.long)

        labels = None
        if self.id_to_labels_df is not None:
            if self.use_drug_idx:
                labels = torch.tensor(
                    self.id_to_labels_df.loc[unq_id][self.label_column][
                        self.use_drug_idx
                    ],
                    dtype=torch.float32,
                )
            else:
                labels = torch.tensor(
                    self.id_to_labels_df.loc[unq_id][self.label_column],
                    dtype=torch.float32,
                )

        return BacInputSample(
            input_tensor=genes_tensor,
            variants_in_gene=variants_in_gene,
            labels=labels,
            strain_id=unq_id,
        )

    def __len__(self):
        return len(self.unique_ids)