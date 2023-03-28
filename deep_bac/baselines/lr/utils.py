from collections import defaultdict
from typing import Dict, Literal

import pandas as pd

from deep_bac.utils import get_drug_line


DRUG_TO_IDX = {
    "MXF": 0,
    "BDQ": 1,
    "KAN": 2,
    "CFZ": 3,
    "AMI": 4,
    "DLM": 6,
    "RFB": 7,
    "LZD": 8,
    "EMB": 9,
    "LEV": 10,
    "ETH": 11,
    "INH": 12,
    "RIF": 13,
}  # removed "PAS": 5 as it has not enough labels


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
