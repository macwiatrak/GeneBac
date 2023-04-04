import numpy as np
import torch
from dataclasses_json import dataclass_json
from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass_json
@dataclass
class DataVarMatrices:
    train_var_matrix: np.ndarray
    eval_var_matrix: np.ndarray
    train_labels: np.ndarray
    eval_labels: np.ndarray


@dataclass_json
@dataclass
class OneHotVarDataReaderOutput:
    train_dl: DataLoader
    val_dl: DataLoader = None
    test_dl: DataLoader = None
    train_var_matrix: torch.Tensor = None
    train_labels: torch.Tensor = None
