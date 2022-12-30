from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Optional, List

import torch


@dataclass_json
@dataclass
class BacGenesInputSample:
    genes_tensor: torch.Tensor
    variants_in_gene: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    unique_id: Optional[str] = None


@dataclass_json
@dataclass
class BatchBacGenesInputSample:
    genes_tensor: torch.Tensor
    variants_in_gene: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    unique_ids: List[str] = None
