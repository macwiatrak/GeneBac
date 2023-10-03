from typing import Tuple, List, Callable, Dict

import numpy as np
import torch

from genebac.data_preprocessing.utils import seq_to_one_hot, pad_one_hot_seq
from genebac.modelling.data_types import DeepGeneBacConfig
from genebac.modelling.model_gene_expr import DeepBacGeneExpr


def process_sample(
    max_seq_length: int = 2560,
    pad_value: float = 0.25,
    seq: str = None,
) -> torch.Tensor:
    one_hot_seq = seq_to_one_hot(seq[:max_seq_length])
    return pad_one_hot_seq(one_hot_seq, max_seq_length, pad_value).T


def batch_data(
    seq_data: List[Tuple[str, str]],
    max_seq_length: int = 2560,
    pad_value: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functiom which batches data for attribution scores
    :param seq_data: List of tuples where a single tuple is (alt_seq, ref_seq)
    :return:
    """
    alt = torch.stack(
        [
            process_sample(max_seq_length, pad_value, alt_seq)
            for (alt_seq, _) in seq_data
        ]
    )
    ref = torch.stack(
        [
            process_sample(max_seq_length, pad_value, ref_seq)
            for (_, ref_seq) in seq_data
        ]
    )
    return alt, ref


def compute_importance_scores(
    model: DeepBacGeneExpr,
    attribution_fn: Callable,
    alt_tensor: torch.Tensor,
    ref_tensor: torch.Tensor = None,
    use_baseline: bool = False,
) -> np.ndarray:
    attr_model_fn = attribution_fn(model)
    if use_baseline:
        attributions, delta = attr_model_fn.attribute(
            alt_tensor, ref_tensor, return_convergence_delta=True
        )
    else:
        attributions, delta = attr_model_fn.attribute(
            alt_tensor, return_convergence_delta=True
        )
    scores = attributions.sum(dim=1).unsqueeze(1) * alt_tensor
    return scores.detach().numpy()


def load_trained_model(ckpt_path: str) -> DeepBacGeneExpr:
    config = DeepGeneBacConfig(
        gene_encoder_type="gene_bac",
        graph_model_type="dense",
        n_gene_bottleneck_layer=64,
        n_output=1,
        random_state=42,
        max_gene_length=2560,
    )
    model = DeepBacGeneExpr.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=config,
    )
    model.eval()
    return model


def get_importance_scores(
    ckpt_path: str,
    seq_data: List[Tuple[str, str]],
    attribution_fns: List[Callable],
    max_seq_length: int = 2560,
    pad_value: float = 0.25,
    use_baseline: bool = False,
) -> Dict[str, np.ndarray]:
    model = load_trained_model(ckpt_path)
    alt_batch, ref_batch = batch_data(seq_data, max_seq_length, pad_value)

    output = dict()
    for attr_fn in attribution_fns:
        output[attr_fn.get_name()] = compute_importance_scores(
            model=model,
            attribution_fn=attr_fn,
            alt_tensor=alt_batch,
            ref_tensor=ref_batch,
            use_baseline=use_baseline,
        )
    return output
