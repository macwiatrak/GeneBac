import torch.cuda
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from deep_bac.modelling.data_types import DeepBacConfig


def get_trainer(config: DeepBacConfig, output_dir: str) -> Trainer:
    if torch.cuda.is_available() or config.accelerator == "mps":
        gpus = -1
    else:
        gpus = None
    """Get the trainer"""
    return Trainer(
        default_root_dir=output_dir,
        gpus=gpus,
        accelerator=config.accelerator,
        max_epochs=config.max_epochs,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
        ],
        logger=TensorBoardLogger(output_dir),
        accumulate_grad_batches=config.accumulate_grad_batches,
    )
