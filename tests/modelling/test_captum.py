import torch
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GradientShap,
    DeepLiftShap,
)

from deep_bac.modelling.data_types import DeepGeneBacConfig
from deep_bac.modelling.model_gene_expr import DeepBacGeneExpr
from deep_bac.modelling.model_gene_reg import DeepBacGeneReg


def test_model_gene_expr_captum():
    batch_size = 2
    seq_length = 2048
    in_channels = 4
    n_filters = 256
    n_bottleneck_layer = 64
    n_output = 1

    x = torch.rand(batch_size, in_channels, seq_length)
    baseline = torch.rand(batch_size, in_channels, seq_length)

    config = DeepGeneBacConfig(
        gene_encoder_type="gene_bac",
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_init_filters=n_filters,
        n_output=n_output,
    )

    model = DeepBacGeneExpr(config)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        x, baseline, return_convergence_delta=True
    )
    assert attributions.shape == x.shape
    assert delta.shape == (batch_size,)

    dl = DeepLift(model)
    attributions, delta = dl.attribute(
        x, baseline, return_convergence_delta=True
    )
    assert attributions.shape == x.shape
    assert delta.shape == (batch_size,)

    n_samples = 5
    gs = GradientShap(model)
    attributions, delta = gs.attribute(
        x, baseline, n_samples=n_samples, return_convergence_delta=True
    )
    assert attributions.shape == x.shape
    assert delta.shape == (batch_size * n_samples,)

    dls = DeepLiftShap(model)
    attributions, delta = dls.attribute(
        x, baseline, return_convergence_delta=True
    )
    assert attributions.shape == x.shape
    assert delta.shape == (batch_size * batch_size,)


def test_model_gene_reg_captum():
    batch_size = 2
    n_genes = 3
    seq_length = 2048
    in_channels = 4
    n_bottleneck_layer = 64
    n_output = 5

    x = torch.rand(batch_size, n_genes, in_channels, seq_length)
    baseline = torch.rand(batch_size, n_genes, in_channels, seq_length)
    labels = torch.empty(batch_size).random_(2).type(torch.int64)

    config = DeepGeneBacConfig(
        gene_encoder_type="gene_bac",
        graph_model_type="dense",
        regression=False,
        n_gene_bottleneck_layer=n_bottleneck_layer,
        n_output=n_output,
        n_highly_variable_genes=n_genes,
        pos_encoder_type="fixed",
    )

    model = DeepBacGeneReg(config)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        x, baseline, target=labels, return_convergence_delta=True
    )
    assert attributions[0].shape == x[0].shape
    assert attributions[1].shape == x[1].shape
    assert delta.shape == (batch_size,)

    dl = DeepLift(model)
    attributions, delta = dl.attribute(
        x, baseline, target=labels, return_convergence_delta=True
    )
    assert attributions[0].shape == x[0].shape
    assert attributions[1].shape == x[1].shape
    assert delta.shape == (batch_size,)

    n_samples = 5
    gs = GradientShap(model)
    attributions, delta = gs.attribute(
        x,
        baseline,
        target=labels,
        n_samples=n_samples,
        return_convergence_delta=True,
    )
    assert attributions[0].shape == x[0].shape
    assert attributions[1].shape == x[1].shape
    assert delta.shape == (batch_size * n_samples,)

    dls = DeepLiftShap(model)
    attributions, delta = dls.attribute(
        x, baseline, target=labels, return_convergence_delta=True
    )
    assert attributions[0].shape == x[0].shape
    assert attributions[1].shape == x[1].shape
    assert delta.shape == (batch_size * batch_size,)
