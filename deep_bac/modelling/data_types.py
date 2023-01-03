from typing import Literal

from dataclasses_json import dataclass_json
from dataclasses import dataclass


@dataclass_json
@dataclass
class DeepBacConfig:
    gene_encoder_type: Literal["conv_transformer"] = "conv_transformer"
    graph_model_type: Literal["transformer"] = "transformer"
    regression: bool = False
