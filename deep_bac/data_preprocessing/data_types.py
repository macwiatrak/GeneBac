from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass_json
@dataclass
class BacGenesInputSample:
    genes_tensor: torch.Tensor
    variants_in_gene: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    sequence_id: Optional[str] = None
