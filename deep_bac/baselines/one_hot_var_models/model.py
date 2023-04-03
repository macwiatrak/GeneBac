from typing import List, Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn

from deep_bac.modelling.metrics import compute_agg_stats
from deep_bac.modelling.utils import remove_ignore_index


class LinearModel(LightningModule):
    def __init__(
        self,
        input_dim: int,
        lr: float = 1e-3,
        l1_lambda: float = 0.05,
        l2_lambda: float = 0.05,
        drug_thresholds: torch.Tensor = None,
        regression: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.lr = lr
        self.drug_thresholds = drug_thresholds
        self.regression = regression

        self.linear = torch.nn.Linear(input_dim, 1)
        self.loss_fn = (
            nn.MSELoss(reduction="none")
            if self.regression
            else nn.BCEWithLogitsLoss(reduction="none")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def training_step(self, batch, batch_idx: int):
        x, labels = batch
        logits = self(x)
        # get loss with reduction="none" to compute loss per sample
        loss = self.loss_fn(logits.view(-1), labels.view(-1))
        # remove loss for samples with no label and compute mean
        loss = remove_ignore_index(loss, labels.view(-1))
        if (
            loss is None
        ):  # do this to prevent a failure in automatic optimisation in PL
            return None

        return dict(
            loss=loss,
            logits=logits.cpu(),
            labels=labels.cpu(),
        )

    def training_epoch_end(self, outputs: List[Dict[str, torch.tensor]]):
        agg_stats, thresholds = compute_agg_stats(
            outputs,
            regression=self.regression,
            thresholds=None,
        )
        self.drug_thresholds = thresholds
        agg_stats = {f"train_{k}": v for k, v in agg_stats.items()}
        self.log_dict(
            agg_stats,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def eval_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, labels = batch
        logits = self(x)
        # get loss with reduction="none" to compute loss per sample
        loss = (
            self.loss_fn(logits.view(-1), labels.view(-1))
            + self.l1_reg()
            + self.l2_reg()
        )
        # remove loss for samples with no label and compute mean
        loss = remove_ignore_index(loss, labels.view(-1))
        return dict(
            loss=loss.cpu(),
            logits=logits.cpu(),
            labels=labels.cpu(),
        )

    def eval_epoch_end(
        self, outputs: List[Dict[str, torch.tensor]], data_split: str
    ) -> Dict[str, float]:
        agg_stats, _ = compute_agg_stats(
            outputs, regression=self.regression, thresholds=self.drug_thresholds
        )
        agg_stats = {f"{data_split}_{k}": v for k, v in agg_stats.items()}
        self.log_dict(
            agg_stats,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return agg_stats

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.tensor]:
        stats = self.eval_step(batch=batch)
        self.log(
            "val_loss",
            stats["loss"],
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return stats

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.tensor]:
        stats = self.eval_step(batch=batch)
        self.log(
            "test_loss",
            stats["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def l1_reg(self):
        l1_norm = self.linear.weight.abs().sum(dim=-1)
        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.linear.weight.pow(2).sum(dim=-1)
        return self.l2_lambda * l2_norm
