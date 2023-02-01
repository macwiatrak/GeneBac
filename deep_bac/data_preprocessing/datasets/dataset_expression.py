import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from deep_bac.data_preprocessing.data_types import BacInputSample
from deep_bac.data_preprocessing.utils import (
    seq_to_one_hot,
    shift_seq,
    pad_one_hot_seq,
)


class GeneExprDataset(Dataset):
    def __init__(
        self,
        bac_genes_df_file_path: str,
        max_gene_length: int = 2048,
        shift_max: int = 3,
        pad_value: float = 0.25,
        reverse_complement_prob: float = 0.5,
    ):
        self.df = pd.read_parquet(bac_genes_df_file_path)

        self.max_gene_length = max_gene_length
        self.shift_max = shift_max
        self.pad_value = pad_value
        self.reverse_complement_prob = reverse_complement_prob

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = row["prom_gene_seq_w_variants"]
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

        return BacInputSample(
            input_tensor=padded_one_hot_seq.T,
            labels=torch.tensor(row["expression_log1"], dtype=torch.float32),
            strain_id=row["strain"],
            gene_name=row["gene_name"],
            variants_in_gene=torch.tensor(1)
            if row["n_nucleotide_mutations"] > 0
            else torch.tensor(0),
        )

    def __len__(self):
        return len(self.df)
