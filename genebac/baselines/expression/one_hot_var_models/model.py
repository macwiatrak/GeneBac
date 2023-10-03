import itertools
from typing import List, Dict

import pytorch_lightning as pl
import torch
from torch import nn

from genebac.baselines.expression.one_hot_var_models.data_types import (
    OneHotExpressionBatch,
)
from genebac.modelling.metrics import (
    get_regression_metrics,
    get_stats_for_thresholds,
    get_macro_gene_expression_metrics,
    get_macro_thresh_metrics,
)


class LinRegGeneExpr(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        lr: float,
        l2_penalty: float,
        batch_size: int,  # for logging
        gene_vars_w_thresholds: Dict[str, List[str]] = None,
    ):
        super().__init__()
        self.lr = lr
        self.l2_penalty = l2_penalty
        self.gene_vars_w_thresholds = gene_vars_w_thresholds
        self.batch_size = batch_size

        self.layers = nn.Linear(input_dim, 1)

        self.loss_fn = nn.MSELoss(reduction="mean")

        self.save_hyperparameters(logger=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x).view(-1)

    def training_step(
        self, batch: OneHotExpressionBatch, batch_idx: int
    ) -> torch.Tensor:
        logits = self(batch.x)

        loss = self.loss_fn(logits, batch.y) + self.l2_reg()
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss

    def eval_step(self, batch: OneHotExpressionBatch):
        logits = self(batch.x)
        loss = self.loss_fn(logits, batch.y) + self.l2_reg()
        return dict(
            loss=loss,
            logits=logits,
            labels=batch.y,
            gene_names=batch.gene_names,
            strain_ids=batch.strain_ids,
        )

    def eval_epoch_end(
        self, outputs: List[Dict[str, torch.tensor]], data_split: str
    ) -> Dict[str, float]:

        logits = torch.cat([x["logits"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        gene_names = list(itertools.chain(*[x["gene_names"] for x in outputs]))
        strain_ids = list(itertools.chain(*[x["strain_ids"] for x in outputs]))
        agg_stats = get_regression_metrics(logits=logits, labels=labels)

        agg_macro_metrics, per_gene_metrics = get_macro_gene_expression_metrics(
            logits=logits,
            labels=labels,
            gene_names=gene_names,
            strain_ids=strain_ids,
        )
        agg_stats.update(agg_macro_metrics)

        if self.gene_vars_w_thresholds:
            macro_thresh_stats = get_macro_thresh_metrics(
                gene_vars_w_thresholds=self.gene_vars_w_thresholds,
                per_gene_metrics=per_gene_metrics,
            )
            agg_stats.update(macro_thresh_stats)

            thresh_stats = get_stats_for_thresholds(
                logits=logits,
                labels=labels,
                gene_names=gene_names,
                gene_vars_w_thresholds=self.gene_vars_w_thresholds,
            )
            agg_stats.update(thresh_stats)

        agg_stats["loss"] = torch.stack([x["loss"] for x in outputs]).mean()
        agg_stats = {f"{data_split}_{k}": v for k, v in agg_stats.items()}
        self.log_dict(
            agg_stats,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return agg_stats

    def validation_step(
        self, batch: OneHotExpressionBatch, batch_idx: int
    ) -> Dict[str, torch.tensor]:
        stats = self.eval_step(batch=batch)
        self.log(
            "val_loss",
            stats["loss"],
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return stats

    def test_step(
        self, batch: OneHotExpressionBatch, batch_idx: int
    ) -> Dict[str, torch.tensor]:
        stats = self.eval_step(batch=batch)
        self.log(
            "test_loss",
            stats["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        return stats

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.tensor]]
    ) -> Dict[str, float]:
        return self.eval_epoch_end(outputs=outputs, data_split="val")

    def test_epoch_end(
        self, outputs: List[Dict[str, torch.tensor]]
    ) -> Dict[str, float]:
        return self.eval_epoch_end(outputs=outputs, data_split="test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
        )
        return opt

    def l2_reg(self):
        l2_norm = self.layers.weight.pow(2).sum()
        return self.l2_penalty * l2_norm
