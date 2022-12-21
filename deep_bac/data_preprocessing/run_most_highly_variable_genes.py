from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Parameters:
    lr: float = 0.001
    use_tf_features: bool = False
    # other parameters
