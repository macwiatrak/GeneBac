import pandas as pd
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from deep_bac.data_preprocessing.data_reader import get_gene_pheno_dataloader
from deep_bac.data_preprocessing.data_types import BatchBacInputSample
from deep_bac.modelling.data_types import DeepGeneBacConfig
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno
from deep_bac.modelling.modules.utils import count_parameters
from deep_bac.modelling.utils import get_drug_thresholds
from tests.modelling.helpers import get_test_gene_reg_dataloader, BasicLogger


def test_model_gene_pheno_steps():
    batch_size = 2
    n_genes = 3
    seq_length = 2560
    in_channels = 4
    n_filters = 256
    n_bottleneck_layer = 64
    n_output = 5

    x = torch.rand(batch_size, n_genes, in_channels, seq_length)
    labels = torch.empty(batch_size, n_output).random_(2)
    tss_indexes = torch.randint(0, n_genes, (batch_size, n_genes))

    config = DeepGeneBacConfig(
        gene_encoder_type="gene_bac",
        graph_model_type="dense",
        regression=False,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        n_output=n_output,
        n_highly_variable_genes=n_genes,
    )

    model = DeepBacGenePheno(config)

    # test forward
    out = model(x, tss_indexes=tss_indexes)
    assert out.shape == (batch_size, n_output)

    # test training step
    out = model.training_step(
        batch=BatchBacInputSample(
            input_tensor=x, labels=labels, tss_indexes=tss_indexes
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


def test_model_gene_pheno_train_fake_data(tmpdir):
    n_samples = 100
    n_genes = 5
    n_classes = 14
    seq_length = 2560
    regression = False
    n_bottleneck_layer = 64
    max_epochs = 3
    batch_size = 10

    config = DeepGeneBacConfig(
        gene_encoder_type="gene_bac",
        graph_model_type="dense",
        lr=0.001,
        batch_size=batch_size,
        regression=regression,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_output=n_classes,
        max_epochs=max_epochs,
        train_set_len=n_samples,
        n_graph_layers=2,
        n_transformer_heads=4,
        n_highly_variable_genes=n_genes,
        max_gene_length=seq_length,
    )

    dataloader = get_test_gene_reg_dataloader(
        n_samples=n_samples,
        n_genes=n_genes,
        n_classes=n_classes,
        seq_length=seq_length,
        regression=regression,
    )

    model = DeepBacGenePheno(config)
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


def test_model_gene_pheno_train_real_data(tmpdir):
    n_classes = 14
    regression = False
    n_bottleneck_layer = 64
    n_filters = 256
    max_epochs = 20
    batch_size = 2
    max_gene_length = 2560
    selected_genes = ["PE1", "Rv1716", "Rv2000", "pepC", "pepD"]

    reference_gene_data_df = pd.read_parquet(
        "../test_data/reference_gene_data.parquet"
    )

    config = DeepGeneBacConfig(
        gene_encoder_type="gene_bac",
        graph_model_type="dense",
        pos_encoder_type="fixed",
        lr=0.001,
        batch_size=batch_size,
        regression=regression,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        n_output=n_classes,
        max_epochs=max_epochs,
        train_set_len=None,
        n_graph_layers=1,
        n_transformer_heads=2,
        n_highly_variable_genes=len(selected_genes),
        max_gene_length=max_gene_length,
    )

    dataloader = get_gene_pheno_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path="../test_data/sample_agg_variants.parquet",
        reference_gene_data_df=reference_gene_data_df,
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

    model = DeepBacGenePheno(config)
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
    assert logger.val_logs[-1]["val_loss"] < logger.val_logs[0]["val_loss"]


def test_model_gene_pheno_test_drug_thresh_real_data(tmpdir):
    n_classes = 14
    regression = False
    n_bottleneck_layer = 64
    n_filters = 256
    max_epochs = 3
    batch_size = 2
    max_gene_length = 2560
    selected_genes = ["PE1", "Rv1716", "Rv2000", "pepC", "pepD"]

    reference_gene_data_df = pd.read_parquet(
        "../test_data/reference_gene_data.parquet"
    )

    config = DeepGeneBacConfig(
        gene_encoder_type="gene_bac",
        graph_model_type="dense",
        pos_encoder_type="fixed",
        lr=0.001,
        batch_size=batch_size,
        regression=regression,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        n_output=n_classes,
        max_epochs=max_epochs,
        train_set_len=None,
        n_graph_layers=1,
        n_transformer_heads=2,
        n_highly_variable_genes=len(selected_genes),
        max_gene_length=max_gene_length,
    )

    dataloader = get_gene_pheno_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path="../test_data/sample_agg_variants.parquet",
        reference_gene_data_df=reference_gene_data_df,
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

    model = DeepBacGenePheno(config)
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

    model = DeepBacGenePheno.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    drug_thresholds = get_drug_thresholds(model, dataloader)
    model.drug_thresholds = drug_thresholds
    results = trainer.test(
        model,
        dataloaders=dataloader,
    )
    assert results is not None
