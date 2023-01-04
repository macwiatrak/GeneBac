from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from deep_bac.modelling.data_types import DeepBacConfig


def get_trainer(config: DeepBacConfig, output_dir: str) -> Trainer:
    """Get the trainer"""
    return Trainer(
        default_root_dir=output_dir,
        gpus=-1,
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
    )