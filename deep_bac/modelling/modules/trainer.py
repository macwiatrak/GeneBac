import torch.cuda
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from deep_bac.modelling.data_types import DeepBacConfig


def get_trainer(config: DeepBacConfig, output_dir: str) -> Trainer:
    if torch.cuda.is_available():
        devices = torch.cuda.device_count()
        accelerator = "gpu"
        strategy = "ddp" if devices > 1 else None
    else:
        devices = None
        accelerator = "cpu"
        strategy = None

    monitor_metric = (
        config.monitor_metric if config.monitor_metric else "val_loss"
    )
    mode = "min" if "loss" in config.monitor_metric else "max"
    """Get the trainer"""
    return Trainer(
        default_root_dir=output_dir,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=config.max_epochs,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[
            EarlyStopping(
                monitor=monitor_metric,
                patience=config.early_stopping_patience,
                mode=mode,
            ),
            ModelCheckpoint(
                monitor=monitor_metric,
                mode=mode,
                save_top_k=1,
                save_last=True,
            ),
            TQDMProgressBar(refresh_rate=200),
        ],
        logger=TensorBoardLogger(output_dir),
        accumulate_grad_batches=config.accumulate_grad_batches,
    )
