from typing import List, Dict

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from deep_bac.data_preprocessing.data_types import BatchBacGenesInputSample
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.utils import remove_ignore_index, get_gene_encoder, get_graph_model


class DeepBac(pl.LightningModule):
    def __init__(
            self, config: DeepBacConfig,
    ):
        super().__init__()
        self.config = config
        self.gene_encoder = get_gene_encoder(config)
        self.graph_model = get_graph_model(config)

        # get loss depending on whether we predict LOG2MIC or binary MIC
        self.loss_fn = nn.MSELoss(reduction="none") if config.regression else nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, batch_genes_tensor: torch.Tensor) -> torch.Tensor:
        # encode each gene
        gene_encodings = self.gene_encoder(batch_genes_tensor)
        # pass the genes through the graph encoder
        logits = self.graph_model(gene_encodings)
        return logits

    def training_step(self, batch: BatchBacGenesInputSample, batch_idx: int) -> torch.Tensor:
        logits = self(batch.genes_tensor)
        # get loss with reduction="none" to compute loss per sample
        loss = self.loss_fn(logits, batch.labels)
        # remove loss for samples with no label and compute mean
        loss = remove_ignore_index(loss, batch.labels).mean()
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        return loss

    def eval_step(self, batch: BatchBacGenesInputSample):
        logits = self(batch.genes_tensor)
        loss = self.loss_fn(logits, batch.labels)
        # remove loss for samples with no label and compute mean
        loss = remove_ignore_index(loss, batch.labels).mean()
        return dict(
            loss=loss,
        )

    def eval_epoch_end(
            self, outputs: List[Dict[str, torch.tensor]], data_split: str
    ) -> Dict[str, float]:
        # agg_stats = compute_agg_stats(outputs)
        # agg_stats = {f"{data_split}_{k}": v for k, v in agg_stats.items()}
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        agg_stats = {f"{data_split}_loss": mean_loss}
        self.log_dict(
            agg_stats,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return agg_stats

    def validation_step(
            self, batch: BatchBacGenesInputSample, batch_idx: int
    ) -> Dict[str, torch.tensor]:
        stats = self.eval_step(batch=batch)
        self.log(
            "val_loss",
            stats["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return stats

    def test_step(
            self, batch: BatchBacGenesInputSample, batch_idx: int
    ) -> Dict[str, torch.tensor]:
        stats = self.eval_step(batch=batch)
        self.log(
            "test_loss",
            stats["loss"],
            on_step=True,
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
            lr=self.config.learning_rate,
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
                * self.config.num_train_epochs
        )
        num_warmup_steps = int(num_train_steps * self.config.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        return scheduler