import random
from typing import List

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset

from deep_bac.data_preprocessing.data_types import BacInputSample
from deep_bac.data_preprocessing.utils import (
    shift_seq,
    seq_to_one_hot,
    pad_one_hot_seq,
)


class MDCNNDataset(Dataset):
    def __init__(
        self,
        bac_loci_df_file_path: str,
        reference_loci_data_df: DataFrame,
        unique_ids: List[str] = None,
        phenotype_dataframe_file_path: str = None,
        max_loci_length: int = 10147,
        regression: bool = False,  # whether the task should be regression or binary classification
        shift_max: int = 0,
        pad_value: float = 0.25,
        reverse_complement_prob: float = 0.0,
        use_drug_idx: int = None,
    ):
        self.loci_df = pd.read_parquet(bac_loci_df_file_path)
        self.use_drug_idx = use_drug_idx
        self.reference_loci_data_df = reference_loci_data_df

        # get unique ids
        self.unique_ids = unique_ids
        if not self.unique_ids:
            self.unique_ids = list(sorted(self.loci_df.index.levels[0]))

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

        self.max_gene_length = max_loci_length
        self.shift_max = shift_max
        self.pad_value = pad_value
        self.reverse_complement_prob = reverse_complement_prob

        self.loci_to_id = {
            loci: i
            for i, loci in enumerate(reference_loci_data_df["loci"].tolist())
        }

    def __getitem__(self, idx):
        unq_id = self.unique_ids[idx]
        unq_id_subset = self.loci_df.xs(unq_id, level="UNIQUEID")
        unq_id_loci = unq_id_subset["loci"].tolist()

        loci_to_seq_dict = dict()
        for idx, loci in enumerate(self.loci_to_id.keys()):
            if loci in unq_id_loci:
                idx = unq_id_loci.index(loci)
                seq = unq_id_subset.iloc[idx]["seq_w_variants"]
            else:
                seq = self.reference_loci_data_df.iloc[idx]["seq"]
            loci_to_seq_dict[loci] = seq

        # dirty fix to make it directly comparable to results in MD-CNN paper
        loci_to_seq_dict["ethAR"] = (
            loci_to_seq_dict["ethA"] + loci_to_seq_dict["ethR"]
        )
        del loci_to_seq_dict["ethA"]
        del loci_to_seq_dict["ethR"]

        loci_tensor = []
        for loci_name, seq in loci_to_seq_dict.items():
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
            loci_tensor.append(padded_one_hot_seq.T)

        loci_tensor = torch.stack(loci_tensor)

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
            input_tensor=loci_tensor,
            labels=labels,
            strain_id=unq_id,
            gene_name=list(loci_to_seq_dict.keys()),
        )

    def __len__(self):
        return len(self.unique_ids)
