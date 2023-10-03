from typing import List

import torch
from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class OneHotExpressionSample:
    x: torch.Tensor
    y: torch.Tensor = None
    gene: str = None
    strain: str = None


@dataclass_json
@dataclass
class OneHotExpressionBatch:
    x: torch.Tensor
    y: torch.Tensor = None
    gene_names: List[str] = None
    strain_ids: List[str] = None
