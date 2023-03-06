from collections import defaultdict

import pandas as pd
from torch.utils.data import DataLoader

from deep_bac.modelling.model_gene_expr import DeepBacGeneExpr
from deep_bac.modelling.model_gene_pheno import DeepBacGenePheno


def collect_strain_reprs(model: DeepBacGenePheno, dataloader: DataLoader):
    out = defaultdict(list)
    for batch in dataloader:
        logits, strain_embeddings = model(batch)
        out["strain_id"] += batch.strain_ids
        out["logits"] += [item.numpy() for item in logits]
        out["embedding"] += [item.numpy() for item in strain_embeddings]
        out["labels"] += batch.labels.view(-1).tolist()

    df = pd.DataFrame(out)
    return df


def collect_gene_reprs(model: DeepBacGeneExpr, dataloader: DataLoader):
    out = defaultdict(list)
    for batch in dataloader:
        logits, gene_embeddings = model(batch)
        out["strain_id"] += batch.strain_ids
        out["logits"] += [item.numpy() for item in logits]
        out["embedding"] += [item.numpy() for item in gene_embeddings]
        out["labels"] += batch.labels.view(-1).numpy()
        out["gene_name"] += batch.gene_names

    df = pd.DataFrame(out)
    return df
