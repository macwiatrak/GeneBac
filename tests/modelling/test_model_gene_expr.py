import pandas as pd
import torch
import pytorch_lightning as pl

from deep_bac.data_preprocessing.data_reader import get_gene_expr_dataloader
from deep_bac.data_preprocessing.data_types import BatchBacInputSample
from deep_bac.data_preprocessing.utils import get_gene_std_expression
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.model_gene_expr import DeepBacGeneExpr
from deep_bac.modelling.modules.utils import count_parameters
from deep_bac.utils import get_gene_var_thresholds
from tests.modelling.helpers import BasicLogger, get_test_gene_expr_dataloader


def test_model_gene_expr_steps():
    batch_size = 2
    seq_length = 2048
    in_channels = 4
    n_filters = 256
    n_bottleneck_layer = 128
    n_output = 1

    x = torch.rand(batch_size, in_channels, seq_length)
    labels = torch.rand(batch_size, n_output)

    config = DeepBacConfig(
        gene_encoder_type="scbasset",
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        n_output=n_output,
    )

    model = DeepBacGeneExpr(config)

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


def test_model_gene_expr_train_fake_data(tmpdir):
    n_samples = 100
    seq_length = 2048
    n_bottleneck_layer = 32
    max_epochs = 20
    batch_size = 10

    config = DeepBacConfig(
        gene_encoder_type="scbasset",
        lr=0.0001,
        batch_size=batch_size,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        max_epochs=max_epochs,
        train_set_len=n_samples,
    )

    dataloader = get_test_gene_expr_dataloader(
        n_samples=n_samples,
        seq_length=seq_length,
    )

    model = DeepBacGeneExpr(config)
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


def test_model_gene_expr_train_real_data(tmpdir):
    n_bottleneck_layer = 32
    n_filters = 256
    max_epochs = 10
    batch_size = 20
    max_gene_length = 2048
    bac_genes_df_file_path = (
        "../test_data/sample_genes_with_variants_and_expression.parquet"
    )
    gene_var_thresholds = [0.1, 0.25, 0.5]

    config = DeepBacConfig(
        gene_encoder_type="scbasset",
        lr=0.001,
        batch_size=batch_size,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        max_epochs=max_epochs,
    )

    gene_std_dict = get_gene_std_expression(
        df=pd.read_parquet(bac_genes_df_file_path),
    )
    most_variable_genes = list(gene_std_dict.keys())

    dataloader, dataset_len = get_gene_expr_dataloader(
        batch_size=batch_size,
        bac_genes_df_file_path=bac_genes_df_file_path,
        max_gene_length=max_gene_length,
        shift_max=0,
        pad_value=0.25,
        reverse_complement_prob=0.0,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    config.train_set_len = dataset_len

    model = DeepBacGeneExpr(
        config=config,
        gene_vars_w_thresholds=get_gene_var_thresholds(
            most_variable_genes, gene_var_thresholds
        ),
    )
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
