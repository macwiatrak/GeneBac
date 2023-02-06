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


def get_regression_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Get metrics for regression task.
    Returns:
        dict of metrics
    """
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
        "tp": tp.type(torch.float32),
        "fp": fp.type(torch.float32),
        "tn": tn.type(torch.float32),
        "fn": fn.type(torch.float32),
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
        bin_cls_metrics = binary_cls_metrics(logits, labels, ignore_index)
        metrics.update(bin_cls_metrics)
        if len(labels.shape) == 1:
            return metrics
        for drug_idx in range(labels.shape[1]):
            drug_labels = labels[:, drug_idx]
            drug_logits = logits[:, drug_idx]
            drug_metrics = binary_cls_metrics(
                drug_logits, drug_labels, ignore_index
            )
            metrics.update(
                {f"drug_{drug_idx}_{k}": v for k, v in drug_metrics.items()}
            )
        return metrics

    labels_wo_ignore = labels[labels != ignore_index]
    logits_wo_ignore = logits[labels != ignore_index]
    reg_metrics = get_regression_metrics(logits_wo_ignore, labels_wo_ignore)
    metrics.update(reg_metrics)
    return metrics
