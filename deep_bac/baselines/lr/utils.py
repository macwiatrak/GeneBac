from collections import defaultdict
from typing import Dict, Literal

import pandas as pd

from deep_bac.utils import get_drug_line


def dict_metrics_to_df(
    metrics: Dict[str, float],
    drug: str,
    split: Literal["train", "val", "test"] = "test",
) -> pd.DataFrame:
    """Converts the metrics dictionary to a dataframe."""
    output = defaultdict(list)
    for metric, val in metrics.items():
        output["value"].append(val)
        output["metric"].append(metric)
        output["drug"].append(drug)
        output["split"].append(split)
        output["drug_class"] = get_drug_line(drug)
    return pd.DataFrame(output)
