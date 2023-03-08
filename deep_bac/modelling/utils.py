import torch
from torch import nn

from deep_bac.baselines.md_cnn.md_cnn import MDCNN
from deep_bac.modelling.data_types import DeepGeneBacConfig
from deep_bac.modelling.modules.conv_transformer import ConvTransformerEncoder
from deep_bac.modelling.modules.enformer_like_encoder import EnformerLikeEncoder
from deep_bac.modelling.modules.graph_transformer import GraphTransformer
from deep_bac.modelling.modules.layers import DenseLayer
from deep_bac.modelling.modules.positional_encodings import (
    IdentityPositionalEncoding,
    LearnablePositionalEncoding,
    FixedGeneExpressionPositionalEncoding,
)
from deep_bac.modelling.modules.gene_bac_encoder import GeneBacEncoder
from deep_bac.modelling.modules.utils import Flatten


def remove_ignore_index(
    loss: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Remove the loss for the ignore index (-100)"""
    # Remove loss for ignore index
    mask = torch.where(
        labels == ignore_index, torch.zeros_like(loss), torch.ones_like(loss)
    )
    loss = loss * mask
    return loss.sum() / mask.sum()


def get_gene_encoder(config: DeepGeneBacConfig):
    """Get the gene encoder"""
    if config.gene_encoder_type == "conv_transformer":
        return ConvTransformerEncoder(
            n_bottleneck_layer=config.n_gene_bottleneck_layer,
            n_filters=config.n_init_filters,
            n_transformer_heads=config.n_transformer_heads,
        )

    if config.gene_encoder_type == "gene_bac":
        return GeneBacEncoder(
            n_filters_init=config.n_init_filters,
            n_bottleneck_layer=config.n_gene_bottleneck_layer,
        )

    if config.gene_encoder_type == "enformer_like":
        return EnformerLikeEncoder(
            n_filters_init=config.n_init_filters,
            n_bottleneck_layer=config.n_gene_bottleneck_layer,
        )

    if config.gene_encoder_type == "MD-CNN":
        print("Max gene len:", config.max_gene_length)
        print("N highly var genes:", config.n_highly_variable_genes)
        return MDCNN(
            seq_length=config.n_highly_variable_genes * config.max_gene_length,
            n_output=config.n_output,
        )
    raise ValueError(f"Unknown gene encoder type: {config.gene_encoder_type}")


def get_genes_to_strain_model(config: DeepGeneBacConfig):
    """Get the graph model"""
    if config.gene_encoder_type == "MD-CNN":
        return None
    if config.graph_model_type == "transformer":
        return GraphTransformer(
            n_gene_bottleneck_layer=config.n_gene_bottleneck_layer,
            n_genes=config.n_highly_variable_genes,
            n_layers=config.n_graph_layers,
            n_heads=config.n_transformer_heads,
        )

    if config.graph_model_type == "dense":
        return nn.Sequential(
            Flatten(),
            DenseLayer(
                in_features=config.n_gene_bottleneck_layer
                * config.n_highly_variable_genes,
                out_features=config.n_gene_bottleneck_layer,
                layer_norm=True,
            ),
        )
    raise ValueError(f"Unknown graph model type: {config.graph_model_type}")


def get_pos_encoder(config: DeepGeneBacConfig):
    """Get the positional encoder"""
    if config.pos_encoder_type is None:
        return IdentityPositionalEncoding()
    if config.pos_encoder_type == "learnable":
        return LearnablePositionalEncoding(
            dim=config.n_gene_bottleneck_layer,
            n_genes=config.n_highly_variable_genes,
        )
    if config.pos_encoder_type == "fixed":
        return FixedGeneExpressionPositionalEncoding(
            dim=config.n_gene_bottleneck_layer,
        )
    raise ValueError(f"Unknown pos encoder type: {config.pos_encoder_type}")
