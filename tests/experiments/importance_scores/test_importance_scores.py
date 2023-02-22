from captum.attr import DeepLift, IntegratedGradients

from experiments.importance_scores.importance_scores import (
    get_importance_scores,
)


def test_get_importance_scores():
    ckpt_path = "/Users/maciejwiatrak/Downloads/epoch=34-val_r2=0.7889.ckpt"
    seq_data = [
        ("acgttgcat", "gctatgcag"),
        ("gcggttctt", "gctatgcag"),
    ]
    attribution_fns = [DeepLift, IntegratedGradients]
    scores_dict = get_importance_scores(
        ckpt_path=ckpt_path,
        seq_data=seq_data,
        attribution_fns=attribution_fns,
        max_seq_length=2048,
        pad_value=0.25,
    )
    assert scores_dict["Deep Lift"].shape == (2, 4, 2048)
