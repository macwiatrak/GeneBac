from typing import List, Dict

import torch
from torchmetrics.functional import mean_squared_error, mean_absolute_error, r2_score, spearman_corrcoef, \
    pearson_corrcoef, auroc, accuracy


def compute_agg_stats(
        outputs: List[Dict[str, torch.tensor]],
        regression: bool,
        ignore_index: int = -100
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
    logits = torch.cat([x["logits"] for x in outputs])
    labels = torch.cat([x["labels"] for x in outputs])
    loss = torch.stack([x["loss"] for x in outputs]).mean()
    if not regression:
        return {
            "accuracy": accuracy(logits, labels, task="binary", ignore_index=ignore_index),
            "auroc": auroc(logits, labels, task="binary", ignore_index=ignore_index),
            "loss": loss,
        }

    labels_wo_ignore = labels[labels != ignore_index]
    logits_wo_ignore = logits[labels != ignore_index]
    return {
        "pearson": pearson_corrcoef(logits_wo_ignore, labels_wo_ignore),
        "spearman": spearman_corrcoef(logits_wo_ignore, labels_wo_ignore),
        "mse": mean_squared_error(logits_wo_ignore, labels_wo_ignore),
        "mae": mean_absolute_error(logits_wo_ignore, labels_wo_ignore),
        "r2": r2_score(logits_wo_ignore, labels_wo_ignore),
        "loss": loss,
    }
