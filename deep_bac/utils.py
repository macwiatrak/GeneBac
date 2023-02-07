import json
import os
from typing import Literal, Optional, Dict

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
    results: Optional[Dict] = None,
    output_file_path: str = None,
):
    if not results or not output_file_path:
        return
    if not os.path.isfile(output_file_path):
        with open(output_file_path, "w") as f:
            f.write(json.dumps(results) + "\n")
    else:
        with open(output_file_path, "r") as f:
            existing_results = [json.loads(line) for line in f.readlines()]

        existing_results.append(results)
        with open(output_file_path, "w") as f:
            for result in existing_results:
                f.write(json.dumps(result) + "\n")
