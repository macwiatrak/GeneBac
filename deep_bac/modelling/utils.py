import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_bac.baselines.abr.md_cnn import MDCNN
from deep_bac.modelling.data_types import DeepGeneBacConfig
from deep_bac.modelling.metrics import compute_drug_thresholds
from deep_bac.modelling.modules.gnn import GNNModel, get_edge_data
from deep_bac.modelling.modules.graph_transformer import GraphTransformer
from deep_bac.modelling.modules.layers import DenseLayer
from deep_bac.modelling.modules.positional_encodings import (
    IdentityPositionalEncoding,
    LearnablePositionalEncoding,
    FixedGeneExpressionPositionalEncoding,
)
from deep_bac.modelling.modules.gene_bac_encoder import GeneBacEncoder
from deep_bac.modelling.modules.utils import Flatten

GENE_INTERACTIONS_FILE_PATH = "ppi_interactions_string.tsv"

logging.basicConfig(level=logging.INFO)


def remove_ignore_index(
    loss: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Remove the loss for the ignore index (-100)"""
    # Remove loss for ignore index
    mask = torch.where(
        labels == ignore_index, torch.zeros_like(loss), torch.ones_like(loss)
    )
    loss = loss * mask
    loss = loss.sum() / mask.sum()
    loss = loss if not torch.isnan(loss) else None
    return loss


def get_gene_encoder(config: DeepGeneBacConfig):
    """Get the gene encoder"""

    if config.gene_encoder_type == "gene_bac":
        return GeneBacEncoder(
            n_filters_init=config.n_init_filters,
            n_bottleneck_layer=config.n_gene_bottleneck_layer,
        )

    if config.gene_encoder_type == "MD-CNN":
        return MDCNN(
            seq_length=config.n_genes * config.max_gene_length,
            n_output=config.n_output,
        )

    raise ValueError(f"Unknown gene encoder type: {config.gene_encoder_type}")


def get_genes_to_strain_model(
    config: DeepGeneBacConfig,
):
    """Get the graph model"""
    if config.gene_encoder_type == "MD-CNN":
        return None
    if config.graph_model_type == "transformer":
        return GraphTransformer(
            n_gene_bottleneck_layer=config.n_gene_bottleneck_layer,
            n_genes=config.n_genes,
            n_layers=config.n_graph_layers,
            n_heads=config.n_heads,
        )

    if config.graph_model_type == "GAT" or config.graph_model_type == "GCN":
        edge_indices, edge_features = get_edge_data(
            edge_file_path=os.path.join(
                config.input_dir, GENE_INTERACTIONS_FILE_PATH
            ),
            gene_to_idx=config.gene_to_idx,
        )
        return GNNModel(
            input_dim=config.n_gene_bottleneck_layer,
            hidden_dim=config.n_gene_bottleneck_layer,
            output_dim=config.n_gene_bottleneck_layer,
            n_genes=len(config.gene_to_idx),
            n_layers=config.n_graph_layers,
            n_heads=config.n_heads,
            layer_type=config.graph_model_type,
            edge_indices=edge_indices,
            edge_features=edge_features,
            dropout_rate=config.dropout_rate,
        )
    raise ValueError(f"Unknown graph model type: {config.graph_model_type}")


def get_pos_encoder(config: DeepGeneBacConfig):
    """Get the positional encoder"""
    if config.pos_encoder_type is None:
        return IdentityPositionalEncoding()
    if config.pos_encoder_type == "learnable":
        return LearnablePositionalEncoding(
            dim=config.n_gene_bottleneck_layer,
            n_genes=config.n_genes,
        )
    if config.pos_encoder_type == "fixed":
        return FixedGeneExpressionPositionalEncoding(
            dim=config.n_gene_bottleneck_layer,
            agg="sum",
        )
    raise ValueError(f"Unknown pos encoder type: {config.pos_encoder_type}")


def get_drug_thresholds(model, dataloader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    model.to(device)
    model.eval()
    logits_list = []
    labels_list = []
    logging.info(f"Calculating optimal thresholds.")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, mininterval=5)):
            logits = model(
                batch.input_tensor.to(model.device),
                batch.tss_indexes.to(model.device),
            )
            logits_list.append(logits.cpu())
            labels_list.append(batch.labels.cpu())
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    drug_thresholds = compute_drug_thresholds(logits, labels)
    return drug_thresholds
