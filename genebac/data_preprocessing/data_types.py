from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
from torch.utils.data import DataLoader


@dataclass_json
@dataclass
class BacInputSample:
    input_tensor: torch.Tensor
    variants_in_gene: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    tss_index: Optional[torch.Tensor] = None
    strain_id: Optional[str] = None
    gene_name: Optional[str] = None


@dataclass_json
@dataclass
class BatchBacInputSample:
    input_tensor: torch.Tensor
    variants_in_gene: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    tss_indexes: Optional[torch.Tensor] = None
    strain_ids: List[str] = None
    gene_names: Optional[List[str]] = None


@dataclass_json
@dataclass
class DataReaderOutput:
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: Optional[DataLoader] = None
    train_set_len: Optional[int] = None
    gene_to_idx: Dict[str, int] = None
