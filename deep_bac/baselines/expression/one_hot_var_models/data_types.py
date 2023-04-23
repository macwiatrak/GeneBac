from typing import List

import numpy as np
import torch
from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class ExpressionDataVarMatrices:
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray


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
