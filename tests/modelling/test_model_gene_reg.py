import json
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from deep_bac.data_preprocessing.data_readers.gene_reg_dna import (
    get_gene_reg_dna_dataloader,
)
from deep_bac.data_preprocessing.data_types import BatchBacInputSample
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.model_gene_reg import DeepBacGeneReg
from deep_bac.modelling.modules.utils import count_parameters
from tests.modelling.helpers import get_test_gene_reg_dataloader, BasicLogger


def test_model_gene_reg_steps():
    batch_size = 2
    n_genes = 3
    seq_length = 2048
    in_channels = 4
    n_filters = 256
    n_bottleneck_layer = 64
    n_output = 5

    x = torch.rand(batch_size, n_genes, in_channels, seq_length)
    labels = torch.empty(batch_size, n_output).random_(2)

    config = DeepBacConfig(
        gene_encoder_type="scbasset",
        graph_model_type="transformer",
        regression=False,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        n_output=n_output,
    )

    model = DeepBacGeneReg(config)

    # test forward
    out = model(x)
    assert out.shape == (batch_size, n_output)

    # test training step
    out = model.training_step(
        batch=BatchBacInputSample(
            input_tensor=x,
            labels=labels,
        ),
        batch_idx=0,
    )
    assert out > 0.0

    # test eval step
    out = model.eval_step(
        batch=BatchBacInputSample(
            input_tensor=x,
            labels=labels,
        ),
    )
    assert out["loss"] > 0.0


def test_model_gene_reg_train_fake_data(tmpdir):
    n_samples = 100
    n_genes = 5
    n_classes = 4
    seq_length = 2048
    regression = False
    n_bottleneck_layer = 64
    n_filters = 256
    max_epochs = 20
    batch_size = 10

    config = DeepBacConfig(
        gene_encoder_type="scbasset",
        graph_model_type="dense",
        lr=0.001,
        batch_size=batch_size,
        regression=regression,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        n_output=n_classes,
        max_epochs=max_epochs,
        train_set_len=n_samples,
        n_graph_layers=2,
        n_transformer_heads=4,
        n_highly_variable_genes=n_genes,
    )

    dataloader = get_test_gene_reg_dataloader(
        n_samples=n_samples,
        n_genes=n_genes,
        n_classes=n_classes,
        seq_length=seq_length,
        regression=regression,
    )

    model = DeepBacGeneReg(config)
    n_params = count_parameters(model)
    print("Number of trainable model parameters: ", n_params)

    logger = BasicLogger()
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=max_epochs,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        logger=logger,
    )
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    assert logger.val_logs[-1]["val_loss"] < logger.val_logs[0]["val_loss"]
    assert (
        logger.train_logs[-1]["train_loss"] < logger.train_logs[0]["train_loss"]
    )


def test_model_gene_reg_train_real_data(tmpdir):
    n_classes = 14
    regression = False
    n_bottleneck_layer = 64
    n_filters = 256
    max_epochs = 3
    batch_size = 3
    max_gene_length = 2048
    selected_genes = ["PE1", "Rv1716", "Rv2000", "pepC", "pepD"]

    with open("../test_data/reference_gene_seqs.json", "r") as f:
        reference_gene_seqs_dict = json.load(f)

    config = DeepBacConfig(
        gene_encoder_type="scbasset",
        graph_model_type="dense",
        lr=0.001,
        batch_size=batch_size,
        regression=regression,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        n_output=n_classes,
        max_epochs=max_epochs,
        train_set_len=None,
        n_graph_layers=2,
        n_transformer_heads=4,
        n_highly_variable_genes=len(selected_genes),
    )

    dataloader = get_gene_reg_dna_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path="../test_data/sample_agg_variants.parquet",
        reference_gene_seqs_dict=reference_gene_seqs_dict,
        phenotype_dataframe_file_path="../test_data/phenotype_labels_with_binary_labels.parquet",
        max_gene_length=max_gene_length,
        selected_genes=selected_genes,
        regression=regression,
        shift_max=0,
        pad_value=0.25,
        reverse_complement_prob=0.0,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = DeepBacGeneReg(config)
    n_params = count_parameters(model)
    print("Number of trainable model parameters: ", n_params)
    monitor = "val_loss"
    logger = BasicLogger()
    trainer = pl.Trainer(
        default_root_dir=os.path.abspath(tmpdir),
        max_epochs=max_epochs,
        gradient_clip_val=1.0,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[
            TQDMProgressBar(refresh_rate=2),
            ModelCheckpoint(
                dirpath=os.path.abspath(tmpdir),
                filename="{epoch:02d}-{val_loss:.4f}",
                monitor=monitor,
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
        ],
    )
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    print(os.listdir(os.path.abspath(tmpdir)))
    assert logger.val_logs[-1]["val_loss"] < logger.val_logs[0]["val_loss"]
