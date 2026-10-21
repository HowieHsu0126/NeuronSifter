"""Actual autocast arithmetic and gradients at the scientific group boundary."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
from Libs.utils.data.raw_data_registry import load_config


def particle_group(offset=0.):
    return {"features": torch.arange(128, dtype=torch.float32).reshape(32, 4)/128 + offset,
            "retained_log_likelihood": torch.linspace(-2., -.5, 32, dtype=torch.float64)}


def actual_loss(model, group):
    prediction = model(group["features"])
    # Float64 likelihood/normalization operations must remain float64 under AMP.
    log_weights = group["retained_log_likelihood"].log_softmax(0)
    weights = log_weights.exp()
    return (weights[:, None] * prediction.double().square()).sum(), prediction, weights


@pytest.mark.parametrize("enabled,configured_device,expected_dtype", [
    (True, "cpu", torch.bfloat16), (False, "cpu", torch.float32),
    (True, "cuda", torch.float32),
])
@pytest.mark.parametrize("backward", [True, False])
def test_scientific_group_uses_actual_autocast_and_preserves_accumulated_gradient(
        enabled, configured_device, expected_dtype, backward, monkeypatch):
    torch.manual_seed(87123)
    model = torch.nn.Linear(4, 3)
    reference = copy.deepcopy(model)
    model.train(backward)
    groups = [particle_group(), particle_group(.125)]
    policy = {"gradient_accumulation_steps": 2, "mixed_precision": {
        "enabled": enabled, "device_type": configured_device, "dtype": "bfloat16"}}
    observed, backward_context = [], []
    expected_enabled = enabled and configured_device == "cpu"
    def loss_call(current, group, config, device, *, learned_teacher=None):
        loss, prediction, weights = actual_loss(current, group)
        observed.append((prediction.dtype, weights.dtype, torch.is_autocast_enabled("cpu")))
        return loss, {"total_loss": float(loss.detach())}, {"expert_utilization": [1.]}
    def gradient_hook(gradient):
        backward_context.append(torch.is_autocast_enabled("cpu"))
        return gradient
    model.weight.register_hook(gradient_hook)
    monkeypatch.setattr(pretrain, "scientific_joint_group_loss", loss_call)
    rng = torch.random.get_rng_state().clone()
    if backward:
        metrics, status = pretrain.run_pretraining_step(model, {"joint_groups": groups}, {}, torch.device("cpu"), policy)
    else:
        with torch.no_grad():
            metrics, status = pretrain.scientific_group_batch(model, groups, {}, torch.device("cpu"), policy, backward=False)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert observed == [(expected_dtype, torch.float64, expected_enabled)] * 2
    assert not torch.is_autocast_enabled("cpu")
    expected_losses = []
    for group in groups:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=expected_enabled):
            expected, _, _ = actual_loss(reference, group)
        expected_losses.append(float(expected.detach()))
        if backward:
            (expected / (len(groups)*policy["gradient_accumulation_steps"])).backward()
    assert metrics["total_loss"] == pytest.approx(sum(expected_losses)/len(groups))
    assert len(status["source_groups"]) == 2
    if backward:
        assert backward_context == [False, False]
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            assert parameter.dtype == parameter.grad.dtype == torch.float32
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
    else:
        assert backward_context == [] and all(parameter.grad is None for parameter in model.parameters())


def test_scientific_fit_training_and_development_callers_use_policy(tmp_path, monkeypatch):
    class Groups(Dataset):
        family_group_counts = {"software_linear_fixture": 2}
        def __len__(self): return 2
        def __getitem__(self, index): return {"native_joint_group": particle_group(index/8)}
    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    training = config["screening_pipeline"]["screening_benchmark"]["training"]
    training["batch_size"] = 2
    training["memory_policy"].update(effective_batch_size=2, micro_batch_size=1, gradient_accumulation_steps=2)
    training["memory_policy"]["mixed_precision"].update(enabled=True, device_type="cpu", dtype="bfloat16")
    config["native_scientific_training"]["selection"].update(
        max_epochs=1, minimum_epochs=1, minimum_steps=1, max_steps=5, max_seconds=60)
    config["native_scientific_training"]["validation"]["interval_optimizer_steps"] = 1
    memory_policy = pretrain.training_memory_policy(config, False)
    model = torch.nn.Linear(4, 3)
    model.quantized_mddm = SimpleNamespace(training_sample_count=2, training_denoising_step_count=1,
        inference_sample_count=2, inference_denoising_step_count=1, sampler_seed=40)
    # The real single-process wrapper does not provide any other precision context.
    assert pretrain.wrap_fsdp_model(model, "single_process", {}, torch.device("cpu")) is model
    dataset = Groups()
    collate = lambda rows: {"joint_groups": [row["native_joint_group"] for row in rows]}
    monkeypatch.setattr(pretrain, "NativeJointGroupDataset", lambda *args: dataset)
    monkeypatch.setattr(pretrain, "data_loader_kwargs", lambda config: {"collate_fn": collate})
    calls, backward_context = [], []
    def loss_call(current, group, config, device, *, learned_teacher=None):
        loss, prediction, weights = actual_loss(current, group)
        calls.append({"training": current.training, "grad_enabled": torch.is_grad_enabled(),
                      "dtype": prediction.dtype, "weights_dtype": weights.dtype})
        return loss, {"total_loss": float(loss.detach())}, {"expert_utilization": [1.]}
    def gradient_hook(gradient):
        backward_context.append(torch.is_autocast_enabled("cpu"))
        return gradient
    model.weight.register_hook(gradient_hook)
    monkeypatch.setattr(pretrain, "scientific_joint_group_loss", loss_call)
    before = model.weight.detach().clone()
    result = pretrain.fit_scientific_native_groups(model, torch.optim.AdamW(model.parameters(), lr=1e-4),
        DataLoader(dataset, collate_fn=collate), config, torch.device("cpu"), memory_policy,
        tmp_path, "linear_amp_fixture", {"model_seed": 40})
    assert result["executed_steps"] == 1 and result["executed_micro_batches"] == 2
    assert result["selection"]["best"]["steps"] == 1
    assert result["first_microbatch_runtime"]["status"] == "COMPLETED"
    assert not torch.equal(before, model.weight)
    assert calls == [{"training": True, "grad_enabled": True, "dtype": torch.bfloat16, "weights_dtype": torch.float64}]*2 + [
        {"training": False, "grad_enabled": False, "dtype": torch.bfloat16, "weights_dtype": torch.float64}]*2
    assert backward_context == [False, False]


def test_scientific_group_requires_explicit_precision_policy():
    with pytest.raises(KeyError, match="mixed_precision"):
        pretrain.scientific_group_batch(torch.nn.Linear(4, 3), [particle_group()], {}, torch.device("cpu"),
                                       {"gradient_accumulation_steps": 1}, backward=False)
