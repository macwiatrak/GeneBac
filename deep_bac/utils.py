import json
import os
from collections import defaultdict
from typing import Literal, Optional, Dict, List

from deep_bac.modelling.metrics import DRUG_TO_LABEL_IDX

DRUG_SPECIFIC_GENES_DICT = {
    "All": [
        "ahpC",
        "fabG1",
        "inhA",
        "katG",
        "ndh",
        "rpoB",
        "embA",
        "embB",
        "embC",
        "embR",
        "iniA",
        "iniC",
        "manB",
        "rmlD",
        "pncA",
        "rpsA",
        "gyrA",
        "gyrB",
        "rpsL",
        "gid",
        "rrs",
        "tlyA",
        "eis",
    ],
    "INH": [
        "katG",
        "proA",
        "ahpC",
        "fabG1",
        "rpoB",
        "inhA",
        "embB",
        "Rv1139c",
        "Rv1140",
        "Rv1158c",
        "rpsL",
        "Rv1219c",
        "ftsK",
        "Rv2749",
        "gid",
    ],
}


def get_selected_genes(use_drug_specific_genes: Literal["INH"] = None):
    if not use_drug_specific_genes:
        return None
    return DRUG_SPECIFIC_GENES_DICT[use_drug_specific_genes]


def write_results(
    results: List[Optional[Dict]] = None,
    output_file_path: str = None,
):
    if not results or not output_file_path:
        return

    if not isinstance(results, list):
        results = [results]

    if not os.path.isfile(output_file_path):
        with open(output_file_path, "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
    else:
        with open(output_file_path, "r") as f:
            existing_results = [json.loads(line) for line in f.readlines()]

        existing_results += results
        with open(output_file_path, "w") as f:
            for result in existing_results:
                f.write(json.dumps(result) + "\n")


def format_predictions(
    predictions: List[Dict],
    metrics_list: List[str],
    drug_to_idx_dict: Dict[str, int] = DRUG_TO_LABEL_IDX,
    split: str = "test",
):
    output = defaultdict(list)
    for preds in predictions:
        for metric in metrics_list:
            output["value"].append(preds[f"{split}_{metric}"])
            output["metric"].append(metric)
            output["drug"].append("All first and Second line drugs")
            for drug, idx in drug_to_idx_dict.items():
                output["drug"].append(drug)
                output["value"].append(preds[f"{split}_drug_{idx}_{metric}"])
                output["metric"].append(metric)

    output["split"] = [split] * len(output["drug"])
    return output
