from typing import List, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from deep_bac.data_preprocessing.data_types import BatchBacInputSample
from deep_bac.modelling.data_types import DeepGeneBacConfig
from deep_bac.modelling.metrics import compute_agg_stats
from deep_bac.modelling.utils import (
    remove_ignore_index,
    get_gene_encoder,
    get_genes_to_strain_model,
    get_pos_encoder,
)


class DeepBacGenePheno(pl.LightningModule):
    def __init__(
        self,
        config: DeepGeneBacConfig,
        drug_thresholds: torch.Tensor = None,
    ):
        super().__init__()
        self.config = config
        self.drug_thresholds = drug_thresholds

        self.model_type = config.gene_encoder_type
        self.n_bottleneck_layer = config.n_gene_bottleneck_layer
        self.gene_encoder = get_gene_encoder(config)
        self.graph_model = get_genes_to_strain_model(config)
        self.pos_encoder = get_pos_encoder(config)
        self.decoder = nn.Linear(
            in_features=config.n_gene_bottleneck_layer,
            out_features=config.n_output,
        )

        self.regression = config.regression
        # get loss depending on whether we predict LOG2MIC or binary MIC
        self.dropout = nn.Dropout(0.2)
        self.activation_fn = nn.ReLU()

        self.loss_fn = (
            nn.MSELoss(reduction="none")
            if self.regression
            else nn.BCEWithLogitsLoss(reduction="none")
        )

        self.save_hyperparameters(logger=False)

    def forward(
        self,
        batch_genes_tensor: torch.Tensor,
        tss_indexes: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, n_genes, in_channels, seq_length)
        batch_size, n_genes, n_channels, seq_length = batch_genes_tensor.shape

        # if using the MD-CNN for benchmarking
        if self.model_type == "MD-CNN":
            logits = self.gene_encoder(
                batch_genes_tensor.view(
                    batch_size,
                    n_channels,
                    n_genes * seq_length,
                )
            )
            # add dummy torch tensor for compatibility with other models
            return logits, torch.Tensor([])
        # reshape the input to allow the convolutional layer to work
        x = batch_genes_tensor.view(
            batch_size * n_genes, n_channels, seq_length
        )
        # encode each gene
        gene_encodings = self.gene_encoder(x)
        # reshape to out: (batch_size, n_genes, n_bottleneck_layer)
        gene_encodings = gene_encodings.view(
            batch_size, n_genes, self.n_bottleneck_layer
        )
        # add positional encodings
        if tss_indexes is not None:
            gene_encodings = self.pos_encoder(gene_encodings, tss_indexes)
        # pass the genes through the graph encoder
        strain_encodings = self.graph_model(
            self.dropout(self.activation_fn(gene_encodings))
        )
        logits = self.decoder(strain_encodings)
        return logits, strain_encodings

    def training_step(self, batch: BatchBacInputSample, batch_idx: int) -> Dict:
        logits, _ = self(batch.input_tensor, batch.tss_indexes)
        # get loss with reduction="none" to compute loss per sample
        loss = self.loss_fn(logits.view(-1), batch.labels.view(-1))
        # remove loss for samples with no label and compute mean
        loss = remove_ignore_index(loss, batch.labels.view(-1))
        return dict(
            loss=loss,
            logits=logits.cpu(),
            labels=batch.labels.cpu(),
        )

    def training_epoch_end(self, outputs: List[Dict[str, torch.tensor]]):
        if self.config.use_validation_set:
            return None
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

    def eval_step(self, batch: BatchBacInputSample):
        logits, _ = self(batch.input_tensor, batch.tss_indexes)
        loss = self.loss_fn(logits.view(-1), batch.labels.view(-1))
        # remove loss for samples with no label and compute mean
        loss = remove_ignore_index(loss, batch.labels.view(-1))
        return dict(
            loss=loss.cpu(),
            logits=logits.cpu(),
            labels=batch.labels.cpu(),
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
        self, batch: BatchBacInputSample, batch_idx: int
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
        if self.model_type == "MD-CNN":
            opt = torch.optim.Adam(
                [p for p in self.parameters() if p.requires_grad],
                lr=np.exp(-1.0 * 9),
            )
        else:
            opt = torch.optim.AdamW(
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
