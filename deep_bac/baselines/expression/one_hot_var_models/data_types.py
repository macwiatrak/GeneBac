import numpy as np
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
