import itertools
from typing import List, Dict

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from deep_bac.data_preprocessing.data_types import BatchBacInputSample
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.metrics import (
    get_regression_metrics,
    get_stats_for_thresholds,
)
from deep_bac.modelling.modules.positional_encodings import (
    FixedGeneExpressionPositionalEncoding,
)
from deep_bac.modelling.utils import (
    get_gene_encoder,
)


class DeepBacGeneExpr(pl.LightningModule):
    def __init__(
        self,
        config: DeepBacConfig,
        gene_vars_w_thresholds: Dict[float, List[str]] = None,
    ):
        super().__init__()
        self.config = config
        self.gene_vars_w_thresholds = gene_vars_w_thresholds

        self.gene_encoder = get_gene_encoder(config)
        self.decoder = nn.Linear(config.n_gene_bottleneck_layer, 1)
        self.pos_encoder = FixedGeneExpressionPositionalEncoding(
            dim=config.n_gene_bottleneck_layer
        )
        self.dropout = nn.Dropout(0.2)

        # get loss depending on whether we predict LOG2MIC or binary MIC
        self.loss_fn = nn.MSELoss(reduction="mean")

    def forward(
        self, batch_genes_tensor: torch.Tensor, tss_indexes: torch.Tensor = None
    ) -> torch.Tensor:
        # encode each gene
        gene_encodings = self.gene_encoder(batch_genes_tensor)
        if tss_indexes is not None:
            gene_encodings = self.pos_encoder(gene_encodings, tss_indexes)
        # pass the genes through the graph encoder
        logits = self.decoder(self.dropout(gene_encodings))
        return logits.view(-1)

    def training_step(
        self, batch: BatchBacInputSample, batch_idx: int
    ) -> torch.Tensor:
        logits = self(batch.input_tensor, batch.tss_indexes)
        # get loss with reduction="none" to compute loss per sample
        loss = self.loss_fn(logits, batch.labels) + 1e-8
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            batch_size=self.config.batch_size,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def eval_step(self, batch: BatchBacInputSample):
        logits = self(batch.input_tensor, batch.tss_indexes)
        loss = self.loss_fn(logits, batch.labels) + 1e-4
        return dict(
            loss=loss,
            logits=logits,
            labels=batch.labels,
            gene_names=batch.gene_names,
        )

    def eval_epoch_end(
        self, outputs: List[Dict[str, torch.tensor]], data_split: str
    ) -> Dict[str, float]:

        logits = torch.cat([x["logits"] for x in outputs]).squeeze(-1)
        labels = torch.cat([x["labels"] for x in outputs])
        agg_stats = get_regression_metrics(logits=logits, labels=labels)

        if self.gene_vars_w_thresholds:
            gene_names = list(
                itertools.chain(*[x["gene_names"] for x in outputs])
            )
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
        )
        return agg_stats

    def validation_step(
        self, batch: BatchBacInputSample, batch_idx: int
    ) -> Dict[str, torch.tensor]:
        stats = self.eval_step(batch=batch)
        self.log(
            "val_loss",
            stats["loss"],
            prog_bar=True,
            logger=True,
            batch_size=self.config.batch_size,
            on_step=False,
            on_epoch=True,
        )
        return stats

    def test_step(
        self, batch: BatchBacInputSample, batch_idx: int
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
        opt = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.config.lr,
        )
        if (
            self.config.train_set_len is None
            or self.config.warmup_proportion is None
        ):
            return opt
        scheduler = self.get_scheduler(opt)
        return [opt], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "reduce_on_plateau": False,
                "monitor": "train_loss",
            }
        ]

    def get_scheduler(self, optimizer: torch.optim.Optimizer):
        num_train_steps = (
            int(self.config.train_set_len / self.config.batch_size)
            * self.config.max_epochs
        )
        num_warmup_steps = int(num_train_steps * self.config.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        return scheduler
