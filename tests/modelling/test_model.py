import torch
import pytorch_lightning as pl

from deep_bac.data_preprocessing.data_types import BatchBacGenesInputSample
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.model import DeepBac
from deep_bac.modelling.modules.utils import count_parameters
from tests.modelling.helpers import get_test_dataloader, BasicLogger


def test_model_steps():
    batch_size = 2
    n_genes = 3
    seq_length = 2048
    in_channels = 4
    n_filters = 256
    n_bottleneck_layer = 128
    n_output = 5

    x = torch.rand(batch_size, n_genes, in_channels, seq_length)
    labels = torch.empty(batch_size, n_output).random_(2)

    config = DeepBacConfig(
        gene_encoder_type="conv_transformer",
        graph_model_type="transformer",
        regression=False,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        n_output=n_output,
    )

    model = DeepBac(config)

    # test forward
    out = model(x)
    assert out.shape == (batch_size, n_output)

    # test training step
    out = model.training_step(
        batch=BatchBacGenesInputSample(
            genes_tensor=x,
            labels=labels,
        ),
        batch_idx=0,
    )
    assert out > 0.

    # test eval step
    out = model.eval_step(
        batch=BatchBacGenesInputSample(
            genes_tensor=x,
            labels=labels,
        ),
    )
    assert out['loss'] > 0.


def test_model_train(tmpdir):
    n_samples = 100
    n_genes = 5
    n_classes = 4
    seq_length = 1024
    regression = False
    n_bottleneck_layer = 128
    n_filters = 256
    max_epochs = 20
    batch_size = 10

    config = DeepBacConfig(
        gene_encoder_type="conv_transformer",
        graph_model_type="transformer",
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
    )

    dataloader = get_test_dataloader(
        n_samples=n_samples,
        n_genes=n_genes,
        n_classes=n_classes,
        seq_length=seq_length,
        regression=regression,
    )

    model = DeepBac(config)
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
    assert logger.val_logs[-1]['val_loss'] < logger.val_logs[0]['val_loss']
    assert logger.train_logs[-1]['train_loss'] < logger.train_logs[0]['train_loss']
