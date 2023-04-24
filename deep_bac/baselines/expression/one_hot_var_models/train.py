import os

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from tap import Tap

from deep_bac.baselines.abr.one_hot_var_models.argparser import (
    OneHotModelArgumentParser,
)
from deep_bac.baselines.expression.one_hot_var_models.dataset import (
    get_dataloader,
)
from deep_bac.baselines.expression.one_hot_var_models.model import (
    OneHotGeneExpr,
)


def run(
    input_dir: str,
    output_dir: str,
    batch_size: int = 256,
    lr: float = 0.001,
    max_epochs: int = 500,
    early_stop_patience: int = 20,
    l2_penalty: float = 0.0,
    num_workers: int = 0,
    test: bool = False,
):
    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(input_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    train_dl = get_dataloader(
        train_df, batch_size, shuffle=True, num_workers=num_workers
    )
    val_dl = get_dataloader(
        val_df, batch_size, shuffle=False, num_workers=num_workers
    )
    test_dl = get_dataloader(
        test_df, batch_size, shuffle=False, num_workers=num_workers
    )

    n_vars = len(train_df.iloc[0]["x"])
    n_genes = train_df["gene"].nunique()
    model = OneHotGeneExpr(
        input_dim=n_vars + n_genes,
        lr=lr,
        l2_penalty=l2_penalty,
    )
    trainer = Trainer(
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(
                monitor="val_r2",
                patience=early_stop_patience,
                mode="max",
            ),
            TQDMProgressBar(refresh_rate=500),
        ],
    )
    trainer.fit(model, train_dl, val_dl)

    if test:
        return trainer.test(model, test_dl)
    return


class OneHotModelExpressionDataArgParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str = (
        "/Users/maciejwiatrak/Desktop/bacterial_genomics/pseudomonas/one-hot/"
    )
    output_dir: str = "/tmp/test-one-hot-expression/"
    lr: float = 0.001
    batch_size: int = 256
    max_epochs: int = 500
    early_stop_patience: int = 10
    l2_penalty: float = 0.0
    num_workers: int = 0
    test: bool = False


def main(args):
    test_results = run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        l2_penalty=args.l2_penalty,
        num_workers=args.num_workers,
        test=args.test,
    )


if __name__ == "__main__":
    args = OneHotModelExpressionDataArgParser().parse_args()
    main(args)
