import torch

from deep_bac.modelling.data_types import DeepBacConfig
from deep_bac.modelling.modules.conv_transformer import ConvTransformerEncoder
from deep_bac.modelling.modules.graph_transformer import GraphTransformer
from deep_bac.modelling.modules.scBasset_encoder import scBassetEncoder


def remove_ignore_index(
    loss: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Remove the loss for the ignore index (-100)"""
    # Remove loss for ignore index
    loss = loss[labels != ignore_index]
    return loss


def get_gene_encoder(config: DeepBacConfig):
    """Get the gene encoder"""
    if config.gene_encoder_type == "conv_transformer":
        return ConvTransformerEncoder(
            n_bottleneck_layer=config.n_gene_bottleneck_layer,
            n_filters=config.n_init_filters,
            n_transformer_heads=config.n_transformer_heads,
        )

    if config.gene_encoder_type == "scbasset":
        return scBassetEncoder(
            n_filters_init=config.n_init_filters,
            n_bottleneck_layer=config.n_gene_bottleneck_layer,
        )
    raise ValueError(f"Unknown gene encoder type: {config.gene_encoder_type}")


def get_graph_model(config: DeepBacConfig):
    """Get the graph model"""
    if config.graph_model_type == "transformer":
        return GraphTransformer(
            dim=config.n_gene_bottleneck_layer,
            n_output=config.n_output,
            n_layers=config.n_graph_layers,
            n_heads=config.n_transformer_heads,
        )
    raise ValueError(f"Unknown graph model type: {config.graph_model_type}")
