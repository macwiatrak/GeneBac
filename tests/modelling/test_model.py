import torch

from deep_bac.data_preprocessing.data_types import BatchBacGenesInputSample
from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.model import DeepBac


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
