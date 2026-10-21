import torch

from Libs.models.networks.operational_twin import OperationalTwin


CONFIG = {"hidden_dim": 48, "posterior_dim": 12, "depth": 2, "head_count": 4, "mechanism_token_count": 4, "dropout_rate": 0.0, "minimum_scale": 0.0001}


def test_forward_uses_explicit_factors_and_learned_non_degenerate_uncertainty() -> None:
    torch.manual_seed(310031)
    model = OperationalTwin(feature_count=13, action_dim=59, config=CONFIG).eval()
    observation = torch.randn(2, 8, 13)
    mask = torch.zeros(2, 8, 13, dtype=torch.bool)
    mask[:, :4, :5] = True
    action = torch.randn(2, 59)
    output = model(observation, mask, action, torch.tensor([0, 2]))
    assert output["predictive_mean"].shape == observation.shape
    assert output["predictive_scale"].shape == observation.shape
    assert torch.isfinite(output["predictive_scale"]).all()
    assert torch.all(output["predictive_scale"] > 0)
    assert output["predictive_scale"].std() > 0
    assert output["posterior"].stacked().shape == (2, 5, 12)
    assert output["graph_scope"] == "observable_variable_adjacency_not_anatomical_graph"
    repeated = model(observation, mask, action, torch.tensor([0, 2]))
    assert torch.equal(output["predictive_mean"], repeated["predictive_mean"])


def test_forward_rejects_shape_mismatch() -> None:
    model = OperationalTwin(feature_count=13, action_dim=59, config=CONFIG)
    try:
        model(torch.zeros(1, 4, 13), torch.zeros(1, 4, 12, dtype=torch.bool), torch.zeros(1, 59), torch.zeros(1, dtype=torch.long))
    except ValueError as error:
        assert "align" in str(error)
    else:
        raise AssertionError("Shape mismatch was accepted")
