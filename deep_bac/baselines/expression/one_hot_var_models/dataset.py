import torch
from pandas import DataFrame
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from deep_bac.baselines.expression.one_hot_var_models.data_types import (
    OneHotExpressionBatch,
    OneHotExpressionSample,
)


def _collate_samples(samples):
    x = torch.stack([sample.x for sample in samples])
    y = [sample.y for sample in samples]
    if None not in y:
        y = torch.stack(y)

    gene = [sample.gene for sample in samples]
    strain = [sample.strain for sample in samples]
    return OneHotExpressionBatch(x=x, y=y, gene_names=gene, strain_ids=strain)


def get_dataloader(
    df: DataFrame,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
):
    dataset = BacGenomeGeneExprDataset(df=df)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_samples,
    )


class BacGenomeGeneExprDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
    ):
        self.df = df
        self.gene_to_idx = {
            gene: idx for idx, gene in enumerate(self.df["gene"].unique())
        }

    def __getitem__(self, idx: int) -> OneHotExpressionSample:
        row = self.df.iloc[idx]
        gene_idx = self.gene_to_idx[row["gene"]]
        # TODO check this works
        x = torch.cat(
            [
                one_hot(
                    torch.tensor(gene_idx), num_classes=len(self.gene_to_idx)
                ),
                torch.tensor(row["x"], dtype=torch.float32),
            ],
            dim=0,
        )

        return OneHotExpressionSample(
            x=x,
            y=torch.tensor(row["y"], dtype=torch.float32),
            gene=row["gene"],
            strain=row["strain"],
        )

    def __len__(self):
        return len(self.df)
