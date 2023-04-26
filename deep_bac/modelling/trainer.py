import signal

import torch.cuda
from lightning_lite.plugins.environments import SLURMEnvironment
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from deep_bac.modelling.data_types import DeepGeneBacConfig


def get_trainer(
    config: DeepGeneBacConfig,
    output_dir: str,
    resume_from_ckpt_path: str = None,
    refresh_rate: int = 200,
) -> Trainer:
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
    if config.monitor_metric in ["val_r2", "val_r2_high"]:
        filename = "{epoch:02d}-{val_r2:.4f}"
    elif config.monitor_metric == "val_gmean_spec_sens":
        filename = "{epoch:02d}-{val_gmean_spec_sens:.4f}"
    elif config.monitor_metric == "train_gmean_spec_sens":
        filename = "{epoch:02d}-{train_gmean_spec_sens:.4f}"
    else:
        filename = "{epoch:02d}-{val_loss:.4f}"

    """Get the trainer"""
    return Trainer(
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
                dirpath=output_dir,
                filename=filename,
                monitor=monitor_metric,
                mode=mode,
                save_top_k=1,
                save_last=True,
            ),
            TQDMProgressBar(refresh_rate=refresh_rate),
        ],
        logger=TensorBoardLogger(output_dir),
        accumulate_grad_batches=config.accumulate_grad_batches,
        resume_from_checkpoint=resume_from_ckpt_path,
        move_metrics_to_cpu=True,
        plugins=[
            SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGHUP)
        ],
    )
