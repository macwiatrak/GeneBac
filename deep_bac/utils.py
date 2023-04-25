import json
import os
from collections import defaultdict
from typing import Literal, Optional, Dict, List, Tuple

from deep_bac.modelling.metrics import (
    DRUG_TO_LABEL_IDX,
    REGRESSION_METRICS,
    BINARY_CLS_METRICS,
    FIRST_LINE_DRUGS,
    SECOND_LINE_DRUGS,
    NEW_AND_REPURPOSED_DRUGS,
)

DRUG_SPECIFIC_GENES_DICT = {
    "Walker": [
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
    "MD-CNN": [
        "acpM",  # Isoniazid
        "kasA",  # Isoniazid
        # Rv3920c,
        # "gid",  # Streptomycin
        # "rpsA",  # Pyrazinamide
        # "PE_PGRS59",  # Pyrazinamide
        # "clpC1",  # Pyrazinamide
        "embC",  # Ethambutol
        "embA",  # Ethambutol
        "embB",  # Ethambutol
        "aftB",  # Ethambutol
        "ubiA",  # Ethambutol
        "mcr3",  # Streptomycin, Amikacin, Capreomycin, Kanamycin
        "rrs",  # Streptomycin, Amikacin, Capreomycin, Kanamycin
        "rrl",  # Streptomycin, Amikacin, Capreomycin, Kanamycin
        "rrf",  # Streptomycin, Amikacin, Capreomycin, Kanamycin
        "ethA",  # Ethionamide
        "ethR",  # Ethionamide
        "ahpC",  # Isoniazid
        # "tlyA",  # Capreomycin
        "Rv1907c",  # Isoniazid
        "katG",  # Isoniazid
        "furA",  # Isoniazid
        # "Rv1910c",
        # "rpsL",  # Streptomycin
        "rpoB",  # Rifampicin
        "rpoC",  # Rifampicin
        "Rv1482c",  # Isoniazid, Ethionamide
        "fabG1",  # Isoniazid, Ethionamide
        "inhA",  # Isoniazid, Ethionamide
        "eis",  # Kanamycin, Amikacin
        "Rv2417c",  # Ciprofloxacin, Levofloxacin, Moxifloxacin, Ofloxacin
        "gyrB",  # Ciprofloxacin, Levofloxacin, Moxifloxacin, Ofloxacin
        "gyrA",  # Ciprofloxacin, Levofloxacin, Moxifloxacin, Ofloxacin
        # "Rv3600c",  # Pyrazinamide
        # "panD",  # Pyrazinamide
        # "panC",  # Pyrazinamide
        # "Rv2042c",  # Pyrazinamide
        # "pncA",  # Pyrazinamide
        # "Rv2044c",  # Pyrazinamide
    ],
    # take 5 top loci for each drug
    "cryptic": [
        # First-line drugs
        "embB",  # EMB, RIF, LEV, MOX, RFB
        # "embA",  # EMB
        "rpoB",  # EMB, INH, RIF, AMI, ETH, KAN, LEV, MOX, RFB, BDQ, LZD
        "katG",  # EMB, INH, RIF, RFB
        # "pncA",  # EMB
        "ahpC",  # INH
        "fabG1",  # INH, ETH, CLF
        "inhA",  # INH, ETH
        "Rv1565c",  # RIF
        "guaA",  # RIF
        # Second-line drugs
        "rrs",  # AMI, KAN, LEV, MOX, BDQ
        "gyrA",  # AMI, ETH, KAN, LEV, MOX
        # "echA8",  # AMI
        # "Rv2896c",  # AMI
        "ethA",  # ETH, KAN
        "eis",  # KAN
        "gyrB",  # LEV, MOX
        # "rpoC",  # RFB
        # "Rv0810c",  # RFB,
        # New and repurposed druga
        "Rv0678",  # BDQ, CLF
        # "atpE",  # BDQ
        # "pgi",  # BDQ,
        "cyp142",  # CLF
        # "Rv3188",  # CLF,
        # "Rv3327",  # CLF
        "ddn",  # DLM
        "fadE22",  # DLM
        "fba",  # DLM
        # "Rv2180c",  # DLM
        # "gap",  # DLM
        "rplC",  # LZD
        "emrB",  # LZD
        # "Rv3552",  # LZD
        # "add",  # LZD
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


GENE_STD_THRESHOLDS_DICT = dict(
    high=(1e6, 0.7),
    medium=(0.7, 0.4),
    low=(0.4, 0.0),
)


def get_selected_genes(
    use_drug_specific_genes: Literal["INH", "Walker", "MD-CNN"] = "MD-CNN"
):
    if not use_drug_specific_genes:
        return None
    return DRUG_SPECIFIC_GENES_DICT[use_drug_specific_genes]


def get_drug_line(drug: str):
    if drug in FIRST_LINE_DRUGS:
        return "First"
    if drug in SECOND_LINE_DRUGS:
        return "Second"
    if drug in NEW_AND_REPURPOSED_DRUGS:
        return "New and repurposed"
    return None


def format_predictions(
    predictions: Dict,
    metrics_list: List[str],
    drug_to_idx_dict: Dict[str, int] = DRUG_TO_LABEL_IDX,
    split: Literal["train", "val", "test"] = "test",
):
    output = defaultdict(list)
    for metric in metrics_list:
        output["value"].append(predictions[f"{split}_{metric}"])
        output["metric"].append(metric)
        output["drug"].append("All first and Second line drugs")
        for drug, idx in drug_to_idx_dict.items():
            output["drug"].append(drug)
            output["value"].append(predictions[f"{split}_drug_{idx}_{metric}"])
            output["metric"].append(metric)

    output["split"] = [split] * len(output["drug"])
    output["drug_class"] = [get_drug_line(drug) for drug in output["drug"]]
    return output


def format_and_write_results(
    results: List[Optional[Dict]] = None,
    output_file_path: str = None,
    split: Literal["train", "val", "test"] = "test",
):
    if not results or not output_file_path:
        return

    if not isinstance(results, list):
        results = [results]
    results = [
        format_predictions(
            predictions=res,
            metrics_list=BINARY_CLS_METRICS
            if f"{split}_auroc" in res
            else REGRESSION_METRICS,
            drug_to_idx_dict=DRUG_TO_LABEL_IDX,
            split=split,
        )
        for res in results
    ]

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


def get_gene_var_thresholds(
    gene_std_dict: Dict[str, float],
    gene_std_thresholds: Dict[str, Tuple[float, float]],
) -> Dict[str, List[str]]:
    if not gene_std_thresholds or not gene_std_dict:
        return None
    output = defaultdict(list)
    for name, (high, low) in gene_std_thresholds.items():
        output[name] = [
            gene for gene, std in gene_std_dict.items() if low < std <= high
        ]
    return output
