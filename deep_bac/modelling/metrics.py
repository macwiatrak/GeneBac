from typing import List, Dict

import torch
from torchmetrics.functional import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    spearman_corrcoef,
    pearson_corrcoef,
    auroc,
    accuracy,
)
from torchmetrics.functional.classification import (
    binary_stat_scores,
    binary_f1_score,
)

BINARY_CLS_METRICS = ["accuracy", "f1", "auroc", "specificity", "sensitivity"]
REGRESSION_METRICS = ["pearson", "spearman", "mse", "mae", "r2"]

DRUG_TO_LABEL_IDX = {
    "MXF": 0,
    "BDQ": 1,
    "KAN": 2,
    "CFZ": 3,
    "AMI": 4,
    "PAS": 5,
    "DLM": 6,
    "RFB": 7,
    "LZD": 8,
    "EMB": 9,
    "LEV": 10,
    "ETH": 11,
    "INH": 12,
    "RIF": 13,
}

FIRST_LINE_DRUGS = ["INH", "RIF", "EMB"]

SECOND_LINE_DRUGS = ["AMI", "ETH", "KAN", "LEV", "MXF", "RFB"]


def get_regression_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Get metrics for regression task.
    Returns:
        dict of metrics
    """
    if len(logits) == 0:
        return {
            "pearson": torch.tensor(-100.0),
            "spearman": torch.tensor(-100.0),
            "mse": torch.tensor(-100.0),
            "mae": torch.tensor(-100.0),
            "r2": torch.tensor(-100.0),
        }

    return {
        "pearson": pearson_corrcoef(logits, labels),
        "spearman": spearman_corrcoef(logits, labels),
        "mse": mean_squared_error(logits, labels),
        "mae": mean_absolute_error(logits, labels),
        "r2": r2_score(logits, labels),
    }


def binary_cls_metrics(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> Dict[str, torch.Tensor]:
    """
    Get metrics for binary classification task.
    Returns:
        dict of metrics
    """
    tp, fp, tn, fn, sup = binary_stat_scores(
        logits, labels, ignore_index=ignore_index
    )
    if labels[labels != -100].sum() == 0:
        auroc_score = torch.tensor(-100.0)
    else:
        auroc_score = auroc(
            logits, labels, task="binary", ignore_index=ignore_index
        )
    return {
        "accuracy": accuracy(
            logits, labels, task="binary", ignore_index=ignore_index
        ),
        "auroc": auroc_score,
        "f1": binary_f1_score(logits, labels, ignore_index=ignore_index),
        "specificity": tn / (tn + fp),
        "sensitivity": tp / (tp + fn),
        # "tp": tp.type(torch.float32),
        # "fp": fp.type(torch.float32),
        # "tn": tn.type(torch.float32),
        # "fn": fn.type(torch.float32),
    }


def compute_agg_stats(
    outputs: List[Dict[str, torch.tensor]],
    regression: bool,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Compute aggregate statistics from model outputs.
    Args:
        outputs: list of model outputs
        regression: bool indicating if the task is regression or classification
        ignore_index: index to ignore in the labels
    Returns:
        dict of aggregate statistics
    """
    logits = torch.cat([x["logits"] for x in outputs]).squeeze(-1)
    labels = torch.cat([x["labels"] for x in outputs])
    loss = torch.stack([x["loss"] for x in outputs]).mean()
    metrics = {"loss": loss}
    if not regression:
        if len(labels.shape) == 1:
            bin_cls_metrics = binary_cls_metrics(logits, labels, ignore_index)
            metrics.update(bin_cls_metrics)
            return metrics
        drug_metrics = {}
        for drug_idx in range(labels.shape[1]):
            drug_labels = labels[:, drug_idx]
            drug_logits = logits[:, drug_idx]
            drug_m = binary_cls_metrics(drug_logits, drug_labels, ignore_index)
            drug_metrics.update(
                {f"drug_{drug_idx}_{k}": v for k, v in drug_m.items()}
            )
        for metric in BINARY_CLS_METRICS:
            metrics[f"{metric}"] = get_macro_metric(
                metrics_dict=drug_metrics,
                metric=metric,
                # use only first line drugs for macro metrics
                drug_idxs=[
                    DRUG_TO_LABEL_IDX[idx]
                    for idx in FIRST_LINE_DRUGS + SECOND_LINE_DRUGS
                ],
            )
        metrics.update(drug_metrics)
        return metrics

    if len(labels.shape) == 1:
        labels = labels.view(-1)
        logits = logits.view(-1)
        labels_wo_ignore = labels[labels != ignore_index]
        logits_wo_ignore = logits[labels != ignore_index]
        reg_metrics = get_regression_metrics(logits_wo_ignore, labels_wo_ignore)
        metrics.update(reg_metrics)
        return reg_metrics

    drug_metrics = {}
    for drug_idx in range(labels.shape[1]):
        drug_labels = labels[:, drug_idx]
        drug_logits = logits[:, drug_idx]
        drug_labels_wo_ignore = drug_labels[drug_labels != ignore_index]
        drug_logits_wo_ignore = drug_logits[drug_labels != ignore_index]
        drug_m = get_regression_metrics(
            drug_logits_wo_ignore, drug_labels_wo_ignore
        )
        drug_metrics.update(
            {f"drug_{drug_idx}_{k}": v for k, v in drug_m.items()}
        )
    for metric in REGRESSION_METRICS:
        metrics[f"{metric}"] = get_macro_metric(
            metrics_dict=drug_metrics,
            metric=metric,
            # use only first & second line drugs for macro metrics
            drug_idxs=[
                DRUG_TO_LABEL_IDX[idx]
                for idx in FIRST_LINE_DRUGS + SECOND_LINE_DRUGS
            ],
        )
    metrics.update(drug_metrics)
    return metrics


def get_macro_metric(
    metrics_dict: Dict[str, torch.Tensor],
    metric: str,
    drug_idxs: List[int],
):
    return torch.stack(
        [
            metrics_dict[f"drug_{drug_idx}_{metric}"]
            for drug_idx in drug_idxs
            if metrics_dict[f"drug_{drug_idx}_{metric}"] > -100.0
        ]
    ).mean()
