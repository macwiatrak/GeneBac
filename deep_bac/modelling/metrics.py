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
    return {
        "accuracy": accuracy(
            logits, labels, task="binary", ignore_index=ignore_index
        ),
        "auroc": auroc(
            logits, labels, task="binary", ignore_index=ignore_index
        ),
        "f1": binary_f1_score(logits, labels, ignore_index=ignore_index),
        "specificity": tn / (tn + fp),
        "sensitivity": tp / (tp + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
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
        return metrics

    labels_wo_ignore = labels[labels != ignore_index]
    logits_wo_ignore = logits[labels != ignore_index]
    reg_metrics = get_regression_metrics(logits_wo_ignore, labels_wo_ignore)
    metrics.update(reg_metrics)
    return metrics
