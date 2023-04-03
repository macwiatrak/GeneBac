import torch

from deep_bac.baselines.one_hot_var_models.model import LinearModel


def test_model_steps():
    input_dim = 100
    batch_size = 10
    output_dim = 5
    lr = 0.01
    l1_lambda = 0.05
    l2_lambda = 0.05
    regression = False

    x = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, 2, (batch_size, output_dim))

    model = LinearModel(
        input_dim=input_dim,
        output_dim=output_dim,
        lr=lr,
        l1_lambda=l1_lambda,
        l2_lambda=l2_lambda,
        regression=regression,
    )
    out = model(x)
    assert out.shape == (batch_size, output_dim)

    out = model.training_step((x, labels), 0)
    assert out["loss"] > 0
    out["loss"].backward()

    out = model.eval_step((x, labels))
    assert out["loss"] > 0
