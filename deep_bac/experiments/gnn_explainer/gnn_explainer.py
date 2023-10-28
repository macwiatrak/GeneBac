import logging
import os
from typing import Literal

import torch
from lightning_lite import seed_everything
from tap import Tap
from torch_geometric.explain import GNNExplainer, Explainer
from tqdm import tqdm

from deep_bac.data_preprocessing.data_reader import get_gene_pheno_data
from deep_bac.experiments.gnn_explainer.utils import (
    visualize_graph_via_networkx,
)
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno
from deep_bac.modelling.modules.gnn import batch_edge_index
from deep_bac.utils import get_selected_genes
from tests.modelling.helpers import get_test_gene_reg_dataloader


def run(
    input_dir: str,
    ckpt_path: str,
    output_dir: str,
    shift_max: int = 0,
    pad_value: float = 0.25,
    reverse_complement_prob: float = 0.0,
    num_workers: int = 0,
    use_drug_specific_genes: Literal[
        "cryptic",
        "PA_GWAS_top_3",
        "PA_GWAS_top_5",
    ] = "cryptic",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, "graphs")):
        os.makedirs(os.path.join(output_dir, "graphs"))

    selected_genes = get_selected_genes(use_drug_specific_genes)
    logging.info(f"Selected genes: {selected_genes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"][
        "config"
    ]
    # set the
    config.input_dir = (
        #     "/Users/maciejwiatrak/Desktop/bacterial_genomics/pseudomonas/mic/"
        "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data"
    )
    model = DeepBacGenePheno.load_from_checkpoint(ckpt_path, config=config)
    model.to(device)
    # model.eval()

    gnn_model = model.graph_model

    gnn_model.train()
    explainer = Explainer(
        model=gnn_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )
    edge_index = gnn_model.same_edge_indices

    data = get_gene_pheno_data(
        input_df_file_path=os.path.join(
            input_dir, "processed_agg_variants.parquet"
        ),
        reference_gene_data_df_path=os.path.join(
            input_dir, "reference_gene_data.parquet"
        ),
        phenotype_df_file_path=os.path.join(
            input_dir, "phenotype_labels_with_binary_labels.parquet"
        ),
        train_val_test_split_indices_file_path=os.path.join(
            input_dir, "train_test_cv_split_unq_ids.json"
        ),
        variance_per_gene_file_path=os.path.join(
            input_dir, "unnormalised_variance_per_gene.csv"
        ),
        max_gene_length=config.max_gene_length,
        n_highly_variable_genes=config.n_highly_variable_genes,
        regression=config.regression,
        batch_size=config.batch_size,
        shift_max=shift_max,
        pad_value=pad_value,
        reverse_complement_prob=reverse_complement_prob,
        num_workers=num_workers if num_workers is not None else os.cpu_count(),
        selected_genes=selected_genes,
        test=True,
    )
    logging.info("Finished loading data")

    # n_samples = 20
    # n_genes = 20
    # n_classes = 14
    # seq_length = 2560
    # regression = True
    # batch_size = 10

    # dataloader = get_test_gene_reg_dataloader(
    #     n_samples=n_samples,
    #     n_genes=n_genes,
    #     n_classes=n_classes,
    #     seq_length=seq_length,
    #     regression=regression,
    #     batch_size=batch_size,
    # )

    # batch = next(iter(dataloader))

    edge_mask_output = []
    node_mask_output = []
    idx = 0
    model.eval()

    for batch in tqdm(data.test_dataloader, mininterval=5):
        node_embeddings = model.get_gene_encodings(
            batch.input_tensor, batch.tss_indexes
        ).detach()

        for item in node_embeddings:
            explanation = explainer(item, edge_index)

            node_mask = explanation.node_stores[0]["node_mask"].squeeze(-1)
            node_mask_output.append(node_mask)

            edge_mask = explanation.edge_stores[0]["edge_mask"]
            edge_mask_output.append(edge_mask)

            edge_index = explanation.edge_stores[0]["edge_index"]

            visualize_graph_via_networkx(
                edge_index=edge_index,
                edge_weight=edge_mask,
                node_size_list=node_mask,
                labels=list(config.gene_to_idx.keys()),
                path=os.path.join(output_dir, "graphs", f"graph_{idx}.png"),
            )
            idx += 1
            # run it only on a subset of the test set
            # if idx > 500:
            #     break

    node_mask_output = torch.stack(node_mask_output)
    edge_mask_output = torch.stack(edge_mask_output)

    torch.save(
        node_mask_output, os.path.join(output_dir, "node_mask_output.pt")
    )
    torch.save(
        edge_mask_output, os.path.join(output_dir, "edge_mask_output.pt")
    )


class ArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str = (
        "/Users/maciejwiatrak/Desktop/bacterial_genomics/cryptic/data/"
    )
    ckpt_path: str = (
        "/Users/maciejwiatrak/Downloads/epoch=248-train_r2=0.4890_20810503.ckpt"
    )
    output_dir: str = "/tmp/gnn-explainer/"
    shift_max: int = 0
    pad_value: float = 0.25
    reverse_complement_prob: float = 0.0
    random_state: int = 42


def main(args):
    seed_everything(args.random_state)
    run(
        input_dir=args.input_dir,
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        shift_max=args.shift_max,
        pad_value=args.pad_value,
        reverse_complement_prob=args.reverse_complement_prob,
    )


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
