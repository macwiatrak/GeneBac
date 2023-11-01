import json
import os
from collections import defaultdict
from typing import Literal, Optional, Dict, List, Tuple

import torch

from deep_bac.modelling.metrics import (
    MTB_DRUG_TO_LABEL_IDX,
    REGRESSION_METRICS,
    BINARY_CLS_METRICS,
    MTB_DRUG_TO_DRUG_CLASS,
)
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno

DRUG_SPECIFIC_GENES_DICT = {
    # take 5 top loci for each drug
    "cryptic": [
        # First-line drugs
        "embB",  # EMB, RIF, LEV, MOX, RFB
        "rpoB",  # EMB, INH, RIF, AMI, ETH, KAN, LEV, MOX, RFB, BDQ, LZD
        "katG",  # EMB, INH, RIF, RFB
        "ahpC",  # INH
        "fabG1",  # INH, ETH, CLF
        "inhA",  # INH, ETH
        "Rv1565c",  # RIF
        "guaA",  # RIF
        # Second-line drugs
        "rrs",  # AMI, KAN, LEV, MOX, BDQ
        "gyrA",  # AMI, ETH, KAN, LEV, MOX
        "ethA",  # ETH, KAN
        "eis",  # KAN
        "gyrB",  # LEV, MOX
        # New and repurposed druga
        "Rv0678",  # BDQ, CLF
        "cyp142",  # CLF
        "ddn",  # DLM
        "fadE22",  # DLM
        "fba",  # DLM
        "rplC",  # LZD
        "emrB",  # LZD
    ],
    "PA_small": [
        "PA0004",
        "PA0005",
        "PA0313",
        "PA0424",
        "PA0425",
        "PA0762",
        "PA0958",
        "PA1097",
        "PA1120",
        "PA2020",
        "PA2494",
        "PA3047",
        "PA3112",
        "PA3168",
        "PA3574",
        "PA4266",
        "PA4270",
        "PA4379",
        "PA4418",
        "PA4522",
        "PA4725",
        "PA4726",
        "PA4777",
        "PA4964",
    ],
    "PA_medium": [
        "PA0005",
        "PA0424",
        "PA1097",
        "PA1120",
        "PA2020",
        "PA3047",
        "PA3168",
        "PA3574",
        "PA4379",
        "PA4522",
        "PA4725",
        "PA4777",
        "PA4964",
    ],
}


GENE_STD_THRESHOLDS_DICT = dict(
    high=(1e6, 0.7),
    medium=(0.7, 0.4),
    low=(0.4, 0.0),
)


def get_selected_genes(
    use_drug_specific_genes: Literal[
        "cryptic",
        "PA_small",
        "PA_medium",
    ] = "cryptic",
):
    if not use_drug_specific_genes:
        return None
    return DRUG_SPECIFIC_GENES_DICT[use_drug_specific_genes]


def format_predictions(
    predictions: Dict,
    metrics_list: List[str],
    drug_to_idx_dict: Dict[str, int] = MTB_DRUG_TO_LABEL_IDX,
    drug_to_drug_class: Dict[str, str] = MTB_DRUG_TO_DRUG_CLASS,
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
    output["drug_class"] = [
        drug_to_drug_class.get(drug, None) for drug in output["drug"]
    ]
    return output


def write_results(
    results: List[Optional[Dict]] = None,
    output_file_path: str = None,
):
    if not results or not output_file_path:
        return

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
            drug_to_idx_dict=MTB_DRUG_TO_LABEL_IDX,
            split=split,
        )
        for res in results
    ]

    write_results(results, output_file_path)


def get_gene_var_thresholds(
    gene_std_dict: Dict[str, float],
    gene_std_thresholds: Dict[str, Tuple[float, float]],
) -> Dict[str, List[str]]:
    if not gene_std_thresholds or not gene_std_dict:
        return None
    output = defaultdict(list)
    for name, (high, low) in gene_std_thresholds.items():
        output[name] = [
            gene for gene, std in gene_std_dict.items() if low <= std < high
        ]
    return output


def fetch_gene_encoder_weights(ckpt_path: str) -> Dict[str, torch.Tensor]:
    gene_enc_sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    gene_encoder_sd = {
        k.lstrip("gene_encoder."): v
        for k, v in gene_enc_sd.items()
        if k.startswith("gene_encoder")
    }
    return gene_encoder_sd


def load_trained_pheno_model(
    ckpt_path: str,
    input_dir: str,
) -> DeepBacGenePheno:
    """This is a function which fixes the issue
    with loading a trained model with different path
    to the file with the gene interactions"""
    config = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"][
        "config"
    ]
    config.input_dir = input_dir
    model = DeepBacGenePheno.load_from_checkpoint(ckpt_path, config=config)
    return model
