import argparse
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils.data import Dataset

from deep_bac.data_preprocessing.data_reader import _collate_samples
from deep_bac.data_preprocessing.data_types import BacGenesInputSample


class TestBacterialGenomeDataset(Dataset):
    def __init__(
            self,
            n_samples: int = 100,
            n_genes: int = 5,
            n_classes: int = 4,
            seq_length: int = 1024,
            regression: bool = False,
    ):
        # (n_samples, n_genes, seq_length, n_nucletoides)
        self.data = F.one_hot(
            torch.randint(0, 4, (n_samples, n_genes, seq_length)),
            num_classes=4).transpose(-2, -1).type(torch.float32)

        if regression:
            self.labels = torch.log2(
                torch.normal(torch.zeros(n_samples, n_classes), 1.5 * torch.ones(n_samples, n_classes)).abs())
        else:
            self.labels = torch.empty(n_samples, n_classes).random_(2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return BacGenesInputSample(
            genes_tensor=self.data[idx],
            labels=self.labels[idx],
        )


def get_test_dataloader(
        n_samples: int = 100,
        n_genes: int = 5,
        n_classes: int = 4,
        seq_length: int = 1024,
        regression: bool = False,
        batch_size: int = 10,
        num_workers: int = 0,
):
    dataset = TestBacterialGenomeDataset(
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
