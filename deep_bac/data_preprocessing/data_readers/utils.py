from typing import List

import torch

from deep_bac.data_preprocessing.data_types import (
    BacInputSample,
    BatchBacInputSample,
)

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

    unique_ids = [sample.strain_id for sample in data]
    return BatchBacInputSample(
        input_tensor=genes_tensor,
        variants_in_gene=variants_in_gene,
        labels=labels,
        strain_ids=unique_ids,
    )
