import numpy as np
from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class DataVarMatrices:
    train_var_matrix: np.ndarray
    test_var_matrix: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
