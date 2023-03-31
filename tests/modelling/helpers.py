import argparse
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils.data import Dataset

from deep_bac.data_preprocessing.data_reader import _collate_samples
from deep_bac.data_preprocessing.data_types import BacInputSample


class TestBacGenomeGeneRegDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 100,
        n_genes: int = 5,
        n_classes: int = 4,
        seq_length: int = 1024,
        regression: bool = False,
    ):
        # (n_samples, n_genes, n_nucletoides, seq_length)
        self.dna_data = (
            F.one_hot(
                torch.randint(0, 4, (n_samples, n_genes, seq_length)),
                num_classes=4,
            )
            .transpose(-2, -1)
            .type(torch.float32)
        )
        self.tss_indexes = torch.randint(0, n_genes, (n_samples, n_genes))

        if regression:
            self.labels = torch.log2(
                torch.normal(
                    torch.zeros(n_samples, n_classes),
                    1.5 * torch.ones(n_samples, n_classes),
                ).abs()
            )
        else:
            self.labels = torch.empty(n_samples, n_classes).random_(2)

    def __len__(self):
        return len(self.dna_data)

    def __getitem__(self, idx):
        return BacInputSample(
            input_tensor=self.dna_data[idx],
            labels=self.labels[idx],
            tss_index=self.tss_indexes[idx],
        )


class TestBacGenomeGeneExprDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 100,
        seq_length: int = 1024,
        genome_length: int = 100000,
    ):
        # (n_samples, n_nucletoides, seq_length)
        self.data = (
            F.one_hot(
                torch.randint(0, 4, (n_samples, seq_length)),
                num_classes=4,
            )
            .transpose(-2, -1)
            .type(torch.float32)
        )

        self.labels = torch.log(
            1
            + torch.normal(
                torch.zeros(n_samples),
                1.5 * torch.ones(n_samples),
            ).abs()
        )
        self.tss_indexes = torch.randint(0, genome_length, (n_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return BacInputSample(
            input_tensor=self.data[idx],
            labels=self.labels[idx],
            tss_index=self.tss_indexes[idx],
            gene_name="test_gene" if idx % 2 == 0 else "test_gene_2",
            strain_id="strain_1" if idx % 2 == 0 else "strain_2",
        )


def get_test_gene_reg_dataloader(
    n_samples: int = 100,
    n_genes: int = 5,
    n_classes: int = 4,
    seq_length: int = 1024,
    regression: bool = False,
    batch_size: int = 10,
    num_workers: int = 0,
):
    dataset = TestBacGenomeGeneRegDataset(
        n_samples=n_samples,
        n_genes=n_genes,
        n_classes=n_classes,
        seq_length=seq_length,
        regression=regression,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_collate_samples,
    )


def get_test_gene_expr_dataloader(
    n_samples: int = 100,
    seq_length: int = 1024,
    batch_size: int = 10,
    num_workers: int = 4,
):
    dataset = TestBacGenomeGeneExprDataset(
        n_samples=n_samples,
        seq_length=seq_length,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_collate_samples,
    )


class BasicLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()
        self.train_logs = []
        self.val_logs = []

    @property
    def experiment(self) -> Any:
        return "Basic"

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ):
        if "train_loss" in metrics:
            self.train_logs.append(metrics)
        else:
            self.val_logs.append(metrics)

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        pass

    @property
    def name(self) -> str:
        return "BasicLogger"

    @property
    def version(self) -> Union[int, str]:
        return "basic"
