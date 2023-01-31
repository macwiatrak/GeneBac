import random
from functools import partial
from typing import Dict, List

import esm
import pandas as pd
import torch
from torch.utils.data import Dataset

from deep_bac.data_preprocessing.data_types import BacInputSample


def get_esm_embeddings(
    model, batch_converter, padding_idx: int, seqs: List[str]
) -> torch.Tensor:
    data = [(f"protein_{idx}", seq) for idx, seq in enumerate(seqs)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1 : tokens_len - 1].mean(0)
        )
    return torch.stack(sequence_representations)


class ProteinGeneRegDataset(Dataset):
    def __init__(
        self,
        bac_genes_df_file_path: str,
        reference_gene_seqs_dict: Dict[str, str],
        unique_ids: List[str] = None,
        phenotype_dataframe_file_path: str = None,
        selected_genes: List = None,
        regression: bool = False,  # whether the task should be regression or binary classification
        use_drug_idx: int = None,
    ):
        self.genes_df = pd.read_parquet(bac_genes_df_file_path)
        self.use_drug_idx = use_drug_idx
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.get_esm_embedding_fn = partial(
            get_esm_embeddings,
            self.model,
            batch_converter,
            self.alphabet.padding_idx,
        )
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
        self.reference_gene_aa_seqs_df = pd.DataFrame(
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

        protein_tensor = []
        variants_in_protein = []
        for idx, gene in enumerate(self.gene_to_id.keys()):
            if gene in unq_id_genes:
                idx = unq_id_genes.index(gene)
                seq = unq_id_subset.iloc[idx]["amino_acids_seq_w_variants"]
                variants_in_protein.append(1)
            else:
                seq = self.reference_gene_aa_seqs_df.iloc[idx]["seq"]
                variants_in_protein.append(0)
            protein_embedding = self.get_esm_embedding_fn(seq)
            protein_tensor.append(protein_embedding)

        protein_tensor = torch.cat(protein_tensor)
        variants_in_gene = torch.tensor(variants_in_protein, dtype=torch.long)

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
            input_tensor=protein_tensor,
            variants_in_gene=variants_in_gene,
            labels=labels,
            strain_id=unq_id,
        )

    def __len__(self):
        return len(self.unique_ids)
