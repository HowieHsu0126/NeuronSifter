from __future__ import annotations

import copy
import json
import os
import resource
import time

import pytest
import torch

from Libs.models.modules.field_rollout import FieldRolloutOperator
from Libs.models.modules.quantized_mddm_rollout_operator import (
    QuantizedMddmRolloutOperator, conditional_nonempty_mask, masked_categorical_objective,
)
from Libs.tests.test_screening_system_multiscale import neuronsifter_test_backbone, quantized_mddm_test_config, solver_test_kwargs
from Libs.utils.data.native_field_normalization import NativeFieldNormalization
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.neuronsifter_scale_registry import apply_neuronsifter_scale
from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config


def make_native_case(time_count=3, batch=1, actions=2):
    torch.manual_seed(23)
    kwargs = solver_test_kwargs()
    kwargs["rollout_update_policy"]["transition_backbone"] = neuronsifter_test_backbone()
    field = FieldRolloutOperator(
        latent_dim=6, regimen_feature_dim=7, hidden_dim=8, region_count=3,
        compartment_count=2, species_count=7, rollout_step_count=time_count,
        adjacency_matrix=[[1.0] * 6 for _ in range(6)],
        fast_state_indices=[0, 1, 5], slow_state_indices=[2, 3, 4, 6],
        **{k: v for k, v in kwargs.items() if k in ("rollout_update_policy", "solver_state_layout", "solver_config")},
    )
    native = field.neuronsifter_world_action_operator.eval()
    config = quantized_mddm_test_config()
    config["inverse_reconstruction_policy"]["return_sample_trajectories"] = True
    operator = QuantizedMddmRolloutOperator(config, 12, 7).eval()
    initial = torch.rand(batch, actions, 3, 2, 7, requires_grad=True)
    teacher = torch.rand(batch, actions, time_count, 3, 2, 7)
    shape = (batch, actions, time_count, 3, 2, 1)
    context = {
        "occupancy": torch.rand(shape, requires_grad=True),
        "support": torch.ones(shape),
        "physical_time": torch.arange(float(time_count)).reshape(1, 1, time_count, 1, 1, 1).expand(shape),
        "posterior_context": torch.rand(batch, actions, 3, 2, 12, requires_grad=True),
    }
    kwargs = dict(initial_state=initial, teacher_trajectory=teacher, teacher_confidence=torch.ones(batch, actions),
                  conditioning=context, native_operator=native, adjacency=field.node_adjacency, coordinates=field.node_coordinates)
    return operator, torch.zeros_like(teacher), kwargs


def test_fixed_initial_wavelet_odd_length_roundtrip():
    operator, carrier, kwargs = make_native_case(time_count=7)
    kwargs["native_operator"].tokenizer.wavelet_level_count = 3
    residual = kwargs["teacher_trajectory"] - kwargs["initial_state"].unsqueeze(2)
    packed, layout = operator.pack(residual, kwargs["native_operator"])
    assert sum(layout["sizes"]) > residual.shape[2]  # replicated odd boundaries
    assert torch.allclose(operator.unpack(packed, layout, kwargs["native_operator"]), residual, atol=3e-7)
    restored = operator.reconstruct(packed, kwargs["initial_state"], layout, kwargs["native_operator"])
    assert torch.equal(restored[:, :, 0], kwargs["initial_state"])
    assert torch.allclose(restored[:, :, 1:], kwargs["teacher_trajectory"][:, :, 1:], atol=3e-7)


def test_feature_graph_symmetry_does_not_mutate_transport_graph():
    _, _, kwargs = make_native_case()
    graph = torch.arange(36.).reshape(6, 6)
    before = graph.clone()
    tokenizer = kwargs["native_operator"].tokenizer
    _, eigenvalues = tokenizer.graph_basis(graph)
    symmetric = (graph + graph.T) / 2
    expected = torch.linalg.eigh(torch.diag(symmetric.sum(-1)) - symmetric)[0]
    assert torch.equal(graph, before)
    assert torch.allclose(eigenvalues, expected, atol=1e-5)


def test_exact_nonempty_distribution_and_importance_normalization():
    generator = torch.Generator().manual_seed(3407)
    draws = torch.stack([conditional_nonempty_mask(torch.tensor(0.5), 2, generator) for _ in range(6000)])
    assert draws.any(-1).all()
    for pattern in ((True, False), (False, True), (True, True)):
        assert abs((draws == torch.tensor(pattern)).all(-1).float().mean().item() - 1 / 3) < .025
    logits, ids = torch.zeros(2, 4), torch.zeros(2, dtype=torch.long)
    values = [masked_categorical_objective(logits, ids, torch.tensor(mask), torch.tensor(.5), torch.tensor(1.))
              for mask in ((True, False), (False, True), (True, True))]
    c = torch.log(torch.tensor(4.))
    assert torch.allclose(torch.stack(values), c * torch.tensor([.75, .75, 1.5]))
    assert torch.allclose(torch.stack(values).mean(), c)
    assert torch.allclose(masked_categorical_objective(logits[:1], ids[:1], torch.ones(1, dtype=torch.bool), torch.tensor(.13), torch.tensor(1.)), c)


def test_teacher_weight_gradients_and_target_independence():
    operator, carrier, kwargs = make_native_case()
    first = operator(carrier, True, **kwargs)
    first["rollout_auxiliary_loss"].backward()
    native = kwargs["native_operator"]
    for module in (operator.value_projection, operator.condition_projection, native.blocks[0].self_attention,
                   native.blocks[0].operator_splitting_expert_bank):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
    # AdaLN-Zero starts with zero attention gates; condition/gate and expert
    # gradients still train the native backbone from the first backward.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in native.blocks[0].parameters())
    assert kwargs["conditioning"]["occupancy"].grad.abs().sum() > 0
    assert kwargs["conditioning"]["posterior_context"].grad.abs().sum() > 0
    changed = dict(kwargs, teacher_trajectory=kwargs["teacher_trajectory"] + .2)
    second = operator(carrier + 10, True, **changed)
    assert torch.equal(first["sample_trajectories"], second["sample_trajectories"])
    assert not torch.allclose(first["rollout_auxiliary_loss"], second["rollout_auxiliary_loss"])
    zero = operator(carrier, True, **dict(kwargs, teacher_confidence=torch.zeros_like(kwargs["teacher_confidence"])))
    double = operator(carrier, True, **dict(kwargs, teacher_confidence=2 * kwargs["teacher_confidence"]))
    assert zero["rollout_auxiliary_loss"] == 0
    assert torch.allclose(double["rollout_auxiliary_loss"], 2 * first["rollout_auxiliary_loss"])


def test_reverse_support_frequencies_reveal_and_fixed_tokens():
    operator, _, _ = make_native_case()
    probability = torch.tensor([.1, .2, .3, .4])
    logits = probability.log().expand(30000, 4)
    samples = operator.categorical_draw(logits, torch.Generator().manual_seed(2))
    assert samples.min() >= 0 and samples.max() < 4
    assert torch.allclose(torch.bincount(samples, minlength=4) / 30000, probability, atol=.012)
    tokens = torch.full((30000,), operator.codebook_size, dtype=torch.long)
    out = operator.reverse_step(tokens, logits, 1., .7, torch.Generator().manual_seed(3))
    assert abs((out != operator.codebook_size).float().mean() - .3) < .015
    final = operator.reverse_step(out, logits, .7, 0., torch.Generator().manual_seed(4))
    assert not (final == operator.codebook_size).any()
    assert torch.equal(final[out != operator.codebook_size], out[out != operator.codebook_size])


def test_sampling_identity_chunk_and_merge():
    operator, carrier, kwargs = make_native_case()
    kwargs = {k: v for k, v in kwargs.items() if k not in ("teacher_trajectory", "teacher_confidence")}
    first = operator(carrier, False, **kwargs)
    operator.sample_chunk_size = 1
    operator.denoising_token_chunk_size = 10000
    second = operator(carrier, False, **kwargs)
    assert torch.equal(first["sample_trajectories"], second["sample_trajectories"])
    assert not torch.equal(first["sample_trajectories"][0], first["sample_trajectories"][1])
    pieces = []
    for ai in range(carrier.shape[1]):
        part = dict(kwargs, initial_state=kwargs["initial_state"][:, ai:ai+1],
                    conditioning={k: v[:, ai:ai+1] for k, v in kwargs["conditioning"].items()},
                    regimen_ids=torch.tensor([ai]))
        pieces.append(operator(carrier[:, ai:ai+1], False, **part)["sample_trajectories"])
    assert torch.equal(first["sample_trajectories"], torch.cat(pieces, dim=2))
    assert torch.equal(first["sample_trajectories"][:, :, :, 0], kwargs["initial_state"].unsqueeze(0).expand(3, -1, -1, -1, -1, -1))


@pytest.mark.parametrize("field", ["initial_state", "conditioning", "native_operator", "teacher_trajectory", "teacher_confidence"])
def test_missing_training_contract_rejected(field):
    operator, carrier, kwargs = make_native_case(actions=1)
    kwargs[field] = None
    with pytest.raises(ValueError):
        operator(carrier, True, **kwargs)


def test_malformed_context_and_self_teacher_rejected():
    operator, carrier, kwargs = make_native_case(actions=1)
    with pytest.raises(ValueError, match="own independent teacher"):
        operator(carrier, True, **dict(kwargs, teacher_trajectory=carrier))
    kwargs["conditioning"]["occupancy"] = kwargs["conditioning"]["occupancy"].detach() + 2
    with pytest.raises(ValueError, match="occupancy"):
        operator(carrier, True, **kwargs)


def test_registry_reference_and_profile_budget_precedence():
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    scaled = apply_neuronsifter_scale(config, "neuronsifter_large")
    backbone = scaled["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    assert [backbone[key] for key in ("operator_hidden_dim", "operator_depth", "attention_head_count", "mechanism_state_token_count")] == [696, 8, 12, 96]
    assert backbone["graph_spectral_tokenizer"]["graph_mode_count"] == 36
    assert backbone["temporal_wavelet_tokenizer"]["wavelet_level_count"] == 3
    assert len(backbone["mechanism_moe"]["expert_names"]) == len(backbone["operator_splitting"]["composition_order"]) == 10
    resolved = quantized_mddm_rollout_from_config(scaled, backbone)
    assert resolved["partial_mask_schedule"]["inference_denoising_step_count"] == 16
    assert resolved == scaled["neuronsifter_scale_selection"]["quantized_mddm_rollout"]
    backbone["quantized_mddm_rollout"]["runtime_overrides"] = {"sample_chunk_size": 1}
    assert quantized_mddm_rollout_from_config(scaled, backbone)["partial_mask_schedule"]["sample_chunk_size"] == 1
    backbone["quantized_mddm_rollout"]["residual_reference_policy"] = "causal_previous_state"
    with pytest.raises(ValueError, match="Conflicting"):
        quantized_mddm_rollout_from_config(scaled, backbone)


def test_declared_deterministic_ablation_remains_disabled():
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone = copy.deepcopy(backbone)
    backbone["quantized_mddm_rollout"]["enabled"] = False
    assert not quantized_mddm_rollout_from_config(config, backbone)["enabled"]


def test_real_pretrainer_teacher_loss_and_missing_store_binding():
    from Libs.exp.training.pretrain_microenvironment_world_model import (
        MicroenvironmentPretrainingModel, native_conditioning_array_names, node_coordinates, run_pretraining_step,
    )
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    backbone = benchmark["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone.update(neuronsifter_test_backbone())
    backbone["quantized_mddm_rollout"] = {"enabled": True, "registry_binding": "quantized_mddm_main_rollout"}
    cfg = config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]
    cfg["partial_mask_schedule"].update(training_sample_count=1, training_denoising_step_count=1)
    with pytest.raises(ValueError, match="conditioning_array_names"):
        native_conditioning_array_names(config)
    model = MicroenvironmentPretrainingModel(config, False)
    species = len(config["dynamics_ready_registry"]["species"])
    nodes = model.node_adjacency.shape[0]
    regions, compartments, times = 6, 6, 3
    assert nodes == regions * compartments
    teacher = torch.rand(1, times, regions, compartments, species)
    initial = torch.rand(1, regions, compartments, species)
    program_shape = (1, times, regions, compartments, 1)
    batch = {"trajectory": teacher, "initial_state": initial,
             "occupancy": torch.rand(program_shape), "support": torch.ones(program_shape),
             "physical_time": torch.arange(float(times)).reshape(1, times, 1, 1, 1).expand(program_shape),
             "posterior_context": torch.rand(1, regions, compartments, 12),
             "teacher_confidence": torch.ones(1), "teacher_loss_weight": torch.ones(1),
             "uncertainty": torch.ones(1, species)}
    state = teacher[:, :-1].reshape(1, times - 1, nodes, species)
    coordinates = node_coordinates(regions, compartments, torch.device("cpu")).expand(1, times - 1, -1, -1)
    with pytest.raises(ValueError, match="admitted conditioning batch"):
        model(state, coordinates)
    metrics, diagnostics = run_pretraining_step(model, batch, config, torch.device("cpu"),
        {"mixed_precision": {"enabled": False}, "gradient_accumulation_steps": 1})
    assert torch.isfinite(torch.tensor(metrics["total_loss"])) and metrics["native_masked_diffusion_loss"] > 0
    assert diagnostics["expert_utilization"]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.operator.parameters())


def test_screening_training_admitted_tensor_selection():
    from types import SimpleNamespace
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs
    model = SimpleNamespace(field_rollout_operator=SimpleNamespace(quantized_mddm_rollout_operator=object(), field_normalization=NativeFieldNormalization(None,1)))
    with pytest.raises(ValueError, match="admitted native_diffusion cache"):
        native_diffusion_training_kwargs(model, {"labels": [0, 1]}, [1], [1, 3], torch.device("cpu"))
    teacher = torch.arange(2 * 4 * 3.).reshape(2, 4, 3, 1, 1, 1)
    payload = {"labels": [0, 1], "native_diffusion": {"split_name": "train", "teacher_trajectory": teacher, "teacher_confidence": torch.ones(2, 4),
               "occupancy": torch.zeros_like(teacher), "support": torch.ones_like(teacher), "physical_time": teacher,
               "initial_state": torch.zeros(2, 4, 1, 1, 1), "posterior_context": torch.zeros(2, 4, 1, 1, 12),
               "sample_uids": ["s0", "s1"], "regimen_uids": ["r0", "r1", "r2", "r3"]}}
    kwargs = native_diffusion_training_kwargs(model, payload, [1], [1, 3], torch.device("cpu"))
    assert torch.equal(kwargs["teacher_trajectory"], teacher[1:2, [1, 3]])
    from Libs.utils.data.native_diffusion_data import stable_numeric_id
    assert kwargs["regimen_ids"].tolist() == [stable_numeric_id("r1"), stable_numeric_id("r3")]
    assert kwargs["sample_ids"].tolist() == [stable_numeric_id("s1")]


def test_serving_context_uid_alignment_and_missing_input():
    from types import SimpleNamespace
    from Libs.scripts.services.serve_screening_api import native_diffusion_serving_kwargs
    bundle = {"screening_model": SimpleNamespace(field_rollout_operator=SimpleNamespace(quantized_mddm_rollout_operator=object(), field_normalization=NativeFieldNormalization(None,1))),
              "regimen_feature_matrix": torch.zeros(2, 7), "regimen_index_by_uid": {"a": 0, "b": 1}}
    with pytest.raises(ValueError, match="compiled occupancy"):
        native_diffusion_serving_kwargs(bundle, {}, ["b"])
    context = {"regimen_uids": ["a", "b"], "occupancy": torch.rand(1, 2, 3, 1, 1, 1),
               "support": torch.ones(1, 2, 3, 1, 1, 1), "physical_time": torch.rand(1, 2, 3, 1, 1, 1)}
    selected = native_diffusion_serving_kwargs(bundle, {"native_diffusion_context": context}, ["b", "a"])
    assert torch.equal(selected["diffusion_context"]["occupancy"], context["occupancy"][:, [1, 0]])
    assert selected["regimen_ids"].tolist() == [1, 0]


def test_delegate_rejects_seed_label_without_effective_config(monkeypatch):
    from types import SimpleNamespace
    from Libs.utils.exp import paper_grade_entrypoint as entrypoint
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    monkeypatch.setattr(entrypoint, "parse_standard_args", lambda *args: SimpleNamespace(dry_run=False, seed=40))
    monkeypatch.setattr(entrypoint, "load_effective_config", lambda args: config)
    called = []
    monkeypatch.setattr(entrypoint.subprocess, "run", lambda *args, **kwargs: called.append(args))
    with pytest.raises(ValueError, match="effective YAML seed"):
        entrypoint.run_delegate_entrypoint("test", "world_model_pretraining", "Libs/exp/training/pretrain_microenvironment_world_model.py")
    assert called == []


def test_native_tensor_schema_matches_caller_contract():
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    schema = config["schemas"]["native_diffusion_training_batch"]
    assert set(schema["required_fields"]) == {"split_name", "teacher_trajectory", "teacher_confidence", "initial_state", "posterior_context", "occupancy", "support", "physical_time", "sample_uids", "regimen_uids"}
    assert schema["generated_rollout_is_teacher"] is False
    assert config["schemas"]["native_diffusion_serving_context"]["teacher_payload_allowed"] is False
    from Libs.models.modules.world_action_encoder import NATIVE_ACTION_IDENTITY_SCHEMA
    assert schema["action_identity_schema"] == NATIVE_ACTION_IDENTITY_SCHEMA
    assert schema["action_identity_axes"] == ["sample", "action", "digest_component"]
    assert schema["native_action_policies"] == ["occupancy_program", "drug_id", "target_id", "scalar_exposure"]
    from Libs.models.modules.world_action_encoder import NATIVE_TARGET_IDENTITY_SCHEMA
    assert schema["target_id_identity_schema"] == NATIVE_TARGET_IDENTITY_SCHEMA
    assert schema["target_id_identity_source"] == "program.action.target_set"
    serving = config["schemas"]["native_diffusion_serving_context"]
    assert serving["drug_id_required_fields"] == serving["target_id_required_fields"] == ["programs"]
    assert serving["target_id_identity_schema"] == NATIVE_TARGET_IDENTITY_SCHEMA
    assert serving["native_unimplemented_policies"] == []
    assert serving["program_validation"] == "canonical_compiler_and_explicit_grid_bindings"
    assert serving["sequential_initial_state"] == "preceding_final_field"
    assert serving["sequential_posterior_policy"] == "same_observation_anchor"


def test_native_screening_does_not_read_legacy_label_targets(monkeypatch):
    from Libs.exp.training import train_screening_system as runner
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    def forbidden(*args):
        raise AssertionError("Native training must not load condition-label targets.")
    monkeypatch.setattr(runner, "selected_regimen_simulation_tensors", forbidden)
    payload = {"rna": {"train": {"native_diffusion": {"teacher_trajectory": torch.zeros(2, 3, 7, 6, 6, 15)}}}}
    legacy, horizon = runner.screening_rollout_target_source(config, None, payload)
    assert legacy is None and horizon == 7
    with pytest.raises(ValueError, match="admitted native_diffusion"):
        runner.screening_rollout_target_source(config, None, {"rna": {"train": {}}})
    with pytest.raises(ValueError, match="axes"):
        runner.screening_rollout_target_source(config, None, {"rna": {"train": {"native_diffusion": {"teacher_trajectory": torch.zeros(3, 4)}}}})
    payload["ephys"] = {"train": {"native_diffusion": {"teacher_trajectory": torch.zeros(2, 3, 4, 6, 6, 15)}}}
    with pytest.raises(ValueError, match="horizons disagree"):
        runner.screening_rollout_target_source(config, None, payload)


def _pretraining_confidence_case():
    from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel
    torch.manual_seed(53)
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    backbone = benchmark["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone.update(neuronsifter_test_backbone())
    backbone["quantized_mddm_rollout"] = {"enabled": True, "registry_binding": "quantized_mddm_main_rollout"}
    config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"]["partial_mask_schedule"].update(
        training_sample_count=1, training_denoising_step_count=1)
    # Isolate every teacher-dependent objective from the separate router regularizers.
    benchmark["training"]["pretraining_objectives"]["router_regularization_weight"] = 0.0
    config["neuronsifter_scale_registry"]["distillation_losses"]["terms"]["expert_diagnostics"]["weight"] = 0.0
    model = MicroenvironmentPretrainingModel(config, False)
    species = len(config["dynamics_ready_registry"]["species"])
    shape = (3, 3, 6, 6, 1)
    batch = {
        "trajectory": torch.rand(3, 3, 6, 6, species),
        "initial_state": torch.rand(3, 6, 6, species),
        "occupancy": torch.rand(shape), "support": torch.ones(shape),
        "physical_time": torch.arange(3.).reshape(1, 3, 1, 1, 1).expand(shape),
        "posterior_context": torch.rand(3, 6, 6, 12),
        "teacher_confidence": torch.tensor([0., .25, 1.]),
        "teacher_loss_weight": torch.tensor([1., .5, .25]),
        "uncertainty": torch.tensor([1., 2., .5]).unsqueeze(1).expand(3, species),
        "sample_ids": torch.arange(3),
    }
    return config, model, batch


def test_pretraining_confidence_excludes_teacher_losses_and_gradients():
    from Libs.exp.training.pretrain_microenvironment_world_model import run_pretraining_step
    config, model, batch = _pretraining_confidence_case()
    memory = {"mixed_precision": {"enabled": False}, "gradient_accumulation_steps": 1}

    def evaluate(current):
        model.zero_grad(set_to_none=True)
        metrics, _ = run_pretraining_step(model, current, config, torch.device("cpu"), memory)
        gradients = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in model.parameters()])
        return metrics, gradients

    baseline, baseline_gradient = evaluate(batch)
    perturbed = {key: value.clone() for key, value in batch.items()}
    perturbed["trajectory"][0] += 7.0
    changed, changed_gradient = evaluate(perturbed)
    teacher_keys = [
        "masked_state_reconstruction_loss", "next_state_prediction_loss", "long_horizon_rollout_loss",
        "teacher_distillation_loss", "counterfactual_consistency_loss", "masked_species_group_prediction_loss",
        "masked_compartment_prediction_loss", "counterfactual_latent_delta_prediction_loss",
        "rank_order_distillation_loss", "uncertainty_calibration_distillation_loss", "tri_consistency_distillation_loss",
    ]
    for key in [*teacher_keys, "native_masked_diffusion_loss", "teacher_dependent_loss", "total_loss"]:
        assert changed[key] == pytest.approx(baseline[key], abs=1e-7), key
    assert torch.allclose(changed_gradient, baseline_gradient, atol=1e-7, rtol=1e-6)
    assert baseline_gradient.abs().sum() > 0
    assert torch.equal(batch["teacher_confidence"], torch.tensor([0., .25, 1.]))

    singles = []
    for index in (1, 2):
        current = {key: value[index:index + 1].clone() for key, value in batch.items()}
        singles.append(evaluate(current)[0])
    # Auxiliary means retain the existing normalized QC/family mean. Raw
    # confidence changes the ratio once: .25*.5 versus 1*.25, i.e. 1:2.
    for key in teacher_keys:
        assert baseline[key] == pytest.approx((singles[0][key] + 2 * singles[1][key]) / 3, rel=2e-5, abs=1e-7), key
    # Native losses retain the fixed batch denominator, not weight normalization.
    assert baseline["native_masked_diffusion_loss"] == pytest.approx(
        (singles[0]["native_masked_diffusion_loss"] + singles[1]["native_masked_diffusion_loss"]) / 3, rel=2e-5)
    unit = {key: value[1:2].clone() for key, value in batch.items()}
    unit["teacher_confidence"].fill_(1)
    unit["teacher_loss_weight"].fill_(1)
    unit["uncertainty"].fill_(1)
    unit_metrics, _ = evaluate(unit)
    assert singles[0]["native_masked_diffusion_loss"] == pytest.approx(
        .25 * .5 * .5 * unit_metrics["native_masked_diffusion_loss"], rel=2e-5)
    assert singles[0]["teacher_distillation_loss"] == pytest.approx(
        .5 * unit_metrics["teacher_distillation_loss"], rel=2e-5)

    zero = {key: value.clone() for key, value in batch.items()}
    zero["teacher_confidence"].zero_()
    zero_metrics, zero_gradient = evaluate(zero)
    for key in [*teacher_keys, "native_masked_diffusion_loss", "teacher_dependent_loss", "total_loss"]:
        assert zero_metrics[key] == 0, key
    assert torch.count_nonzero(zero_gradient) == 0


def test_zero_confidence_preserves_separate_router_regularization():
    from Libs.exp.training.pretrain_microenvironment_world_model import run_pretraining_step
    config, model, batch = _pretraining_confidence_case()
    batch["teacher_confidence"].zero_()
    config["screening_pipeline"]["screening_benchmark"]["training"]["pretraining_objectives"]["router_regularization_weight"] = .3
    config["neuronsifter_scale_registry"]["distillation_losses"]["terms"]["expert_diagnostics"]["weight"] = .2
    metrics, _ = run_pretraining_step(model, batch, config, torch.device("cpu"),
        {"mixed_precision": {"enabled": False}, "gradient_accumulation_steps": 1})
    assert metrics["teacher_dependent_loss"] == 0
    assert metrics["total_loss"] == pytest.approx(
        .3 * metrics["router_regularization_loss"] + .2 * metrics["expert_diagnostics_distillation_loss"], abs=1e-8)


def test_screening_confidence_reaches_real_training_and_validation_losses():
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.exp.training.train_screening_system import native_diffusion_training_kwargs, screening_teacher_losses
    torch.manual_seed(53)
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    solver = solver_test_kwargs()
    backbone = neuronsifter_test_backbone()
    backbone["quantized_mddm_rollout"] = quantized_mddm_test_config()
    solver["rollout_update_policy"]["transition_backbone"] = backbone
    model = ADScreeningSystem(
        patient_state_dim=4, target_feature_dim=3, regimen_feature_dim=7, hidden_dim=8, latent_dim=6,
        dropout_rate=0., compartment_count=2, region_count=3, species_count=7, rollout_step_count=3,
        tri_weight_by_component={name: 1. for name in ("calibration", "coverage", "pkpd_physics_consistency", "ranking_stability", "ood_score")},
        adjacency_matrix=[[1.] * 6 for _ in range(6)], region_prior_feature_dim=2,
        default_region_prior=[[0., 0.] for _ in range(3)], fast_state_indices=[0, 1, 5], slow_state_indices=[2, 3, 4, 6],
        **solver,
    )
    patient, regimens = torch.rand(2, 4), torch.rand(3, 7)
    shape = (2, 3, 3, 3, 2, 1)
    confidence = torch.tensor([[0., .25, .5], [1., 0., .75]])
    native = {
        "split_name": "train", "teacher_trajectory": torch.rand(2, 3, 3, 3, 2, 7), "teacher_confidence": confidence,
        "initial_state": torch.rand(2, 3, 3, 2, 7), "posterior_context": torch.rand(2, 3, 3, 2, 12),
        "sample_uids": ["s0", "s1"], "regimen_uids": ["r0", "r1", "r2"],
        "occupancy": torch.rand(shape), "support": torch.ones(shape),
        "physical_time": torch.arange(3.).reshape(1, 1, 3, 1, 1, 1).expand(shape),
    }
    masks = {key: torch.ones(3) for key in ("z_pk", "z_env", "z_ephys")}
    weights = config["screening_pipeline"]["screening_benchmark"]["loss_weights"]

    def evaluate(current, training):
        model.train(training)
        model.zero_grad(set_to_none=True)
        role = "train" if training else "validation"
        current = dict(current, split_name=role)
        kwargs = native_diffusion_training_kwargs(model, {"labels": [0, 1], "native_diffusion": current},
                                                 slice(None), slice(None), torch.device("cpu"), split_name=role)
        output = model.score_regimens(patient, regimens, **kwargs)
        mse, factor = screening_teacher_losses(output, kwargs["teacher_trajectory"], masks, config, kwargs, training=training)
        total = (weights["transition_supervision_loss"] * mse + weights["simulation_teacher_factor_loss"] * factor
                 + weights["quantized_mddm_rollout_loss"] * output["rollout_auxiliary_loss"])
        total.backward()
        gradient = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in model.parameters()])
        return torch.stack([mse, factor, output["rollout_auxiliary_loss"], total]).detach(), gradient

    perturbed = copy.deepcopy(native)
    perturbed["teacher_trajectory"][confidence == 0] += 9
    for training in (True, False):
        losses, gradients = evaluate(native, training)
        changed_losses, changed_gradients = evaluate(perturbed, training)
        assert torch.allclose(losses, changed_losses, atol=1e-7, rtol=1e-6)
        assert torch.allclose(gradients, changed_gradients, atol=1e-7, rtol=1e-6)
        assert gradients.abs().sum() > 0
        scaled = dict(native, teacher_confidence=2 * confidence)
        scaled_losses, scaled_gradients = evaluate(scaled, training)
        assert torch.allclose(scaled_losses, 2 * losses, atol=1e-7, rtol=1e-6)
        assert torch.allclose(scaled_gradients, 2 * gradients, atol=1e-7, rtol=1e-6)


def test_screening_teacher_mse_uses_action_weights_before_reduction():
    from Libs.exp.training.train_screening_system import screening_teacher_losses
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["family"] = "deterministic_compatibility"
    prediction = torch.zeros(2, 2, 1, 1, 1, 7, requires_grad=True)
    target = torch.tensor([[100., 2.], [3., 200.]]).reshape(2, 2, 1, 1, 1, 1).expand_as(prediction)
    confidence = torch.tensor([[0., .25], [.5, 0.]])
    output = {"field_training_reconstruction": prediction, "field_trajectory": prediction}
    masks = {key: torch.ones(2) for key in ("z_pk", "z_env", "z_ephys")}
    mse, factor = screening_teacher_losses(output, target, masks, config, {"teacher_confidence": confidence}, training=True)
    expected = (.25 * 4 + .5 * 9) / 4
    factor_config = config["screening_pipeline"]["screening_benchmark"]["foundation_training"]["simulation_teacher"]
    assert mse.item() == pytest.approx(expected)
    assert factor.item() == pytest.approx(expected * sum(factor_config[key + "_weight"] for key in masks) / 3)
    (mse + factor).backward()
    assert not prediction.grad[confidence == 0].any()
    assert prediction.grad[confidence > 0].abs().sum() > 0


@pytest.mark.parametrize("policy", ["drug_id", "target_id"])
def test_native_identity_pretraining_excludes_program_information(policy):
    from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel, node_coordinates
    config, _, batch = _pretraining_confidence_case()
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"] = policy
    model = MicroenvironmentPretrainingModel(config, False).eval()
    batch = {key: value[:1].clone() for key, value in batch.items()}
    batch["teacher_confidence"] = torch.ones(1)
    batch["action_identity"] = torch.tensor([[.1, -.2, .3, -.4]], requires_grad=True)
    coords = node_coordinates(6, 6, torch.device("cpu")).expand(1, 2, -1, -1)
    state = batch["trajectory"][:, :-1].flatten(2, 3)
    first = model(state, coords, batch)
    changed = dict(batch, occupancy=1 - batch["occupancy"], support=torch.zeros_like(batch["support"]))
    second = model(state, coords, changed)
    assert torch.equal(first["state_prediction"], second["state_prediction"])
    assert torch.equal(first["rollout_auxiliary_loss"], second["rollout_auxiliary_loss"])
    first["rollout_auxiliary_loss"].backward()
    assert batch["action_identity"].grad is not None
    assert batch["action_identity"].grad.abs().sum() > 0


def _screening_action_case(policy="drug_id", joint_config=None):
    from Libs.models.networks.screening_system import ADScreeningSystem
    from Libs.models.modules.world_action_encoder import native_action_identity_spec
    encoder, _ = native_action_identity_spec(policy)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        solver = solver_test_kwargs()
        if joint_config is not None:
            solver["world_state_posterior_config"] = joint_config
        solver["rollout_update_policy"]["action_representation"]["active_policy"] = policy
        action = solver["rollout_update_policy"]["action_representation"]
        action["feature_groups"]["target_identity"] = [1]
        action["policies"]["target_id"] = {"retained_feature_groups": ["target_identity"]}
        backbone = neuronsifter_test_backbone()
        backbone["diffusion_sampler"]["sampler_seed"] = 41
        backbone["quantized_mddm_rollout"] = quantized_mddm_test_config()
        backbone["quantized_mddm_rollout"]["partial_mask_schedule"]["sampler_seed"] = 41
        solver["rollout_update_policy"]["transition_backbone"] = backbone
        model = ADScreeningSystem(
            patient_state_dim=4, target_feature_dim=3, regimen_feature_dim=7, hidden_dim=8, latent_dim=6,
            dropout_rate=0., compartment_count=2, region_count=3, species_count=7, rollout_step_count=3,
            tri_weight_by_component={name: 1. for name in ("calibration", "coverage", "pkpd_physics_consistency", "ranking_stability", "ood_score")},
            adjacency_matrix=[[1.] * 6 for _ in range(6)], region_prior_feature_dim=2,
            default_region_prior=[[0., 0.] for _ in range(3)], fast_state_indices=[0, 1, 5], slow_state_indices=[2, 3, 4, 6],
            **solver,
        ).eval()
        patient, regimens = torch.rand(1, 4), torch.rand(2, 7)
        shape = (1, 2, 3, 3, 2, 1)
        context = {
            "initial_state": torch.rand(1, 1, 3, 2, 7).expand(-1, 2, -1, -1, -1),
            "posterior_context": torch.rand(1, 1, 3, 2, 12).expand(-1, 2, -1, -1, -1),
            "occupancy": torch.rand(shape, requires_grad=True), "support": torch.ones(shape),
            "physical_time": torch.arange(3.).reshape(1, 1, 3, 1, 1, 1).expand(shape),
            "action_identity": torch.tensor([[encoder({"agent_set": ["agent-a"], "target_set": ["target-a"]}),
                                               encoder({"agent_set": ["agent-b"], "target_set": ["target-b"]})]], requires_grad=True),
        }
        return model, patient, regimens, context


@pytest.mark.parametrize("explicit_posterior", [True, False])
@pytest.mark.parametrize("policy", ["drug_id", "target_id"])
def test_actual_screening_identity_isolates_logits_loss_samples_and_features(explicit_posterior, monkeypatch, policy):
    model, patient, regimens, context = _screening_action_case(policy)
    if not explicit_posterior:
        context = {key: value for key, value in context.items() if key not in {"initial_state", "posterior_context"}}
    teacher = torch.rand(1, 2, 3, 3, 2, 7)
    kwargs = dict(diffusion_context=context, teacher_trajectory=teacher, teacher_confidence=torch.ones(1, 2))
    sampler = model.field_rollout_operator.quantized_mddm_rollout_operator
    original_logits = sampler.logits
    captured = []
    def capture(*args, **kwargs):
        result = original_logits(*args, **kwargs)
        captured.append(result[0].detach().clone())
        return result
    monkeypatch.setattr(sampler, "logits", capture)
    first = model.score_regimens(patient, regimens, **kwargs)
    first_logits = torch.cat([value.flatten() for value in captured])
    captured.clear()
    changed_context = dict(context, occupancy=1-context["occupancy"], support=torch.flip(context["occupancy"], [2]).detach())
    second = model.score_regimens(patient, regimens + 10, **dict(kwargs, diffusion_context=changed_context))
    assert torch.equal(first_logits, torch.cat([value.flatten() for value in captured]))
    captured.clear()
    for key in ("field_training_reconstruction", "field_trajectory", "field_trajectory_samples", "rollout_auxiliary_loss"):
        assert torch.equal(first[key], second[key]), key
    first["rollout_auxiliary_loss"].backward()
    assert context["occupancy"].grad is None
    assert context["action_identity"].grad is not None and context["action_identity"].grad.abs().sum() > 0
    different_identity = dict(context, action_identity=-context["action_identity"])
    third = model.score_regimens(patient, regimens, **dict(kwargs, diffusion_context=different_identity))
    assert not torch.allclose(first_logits, torch.cat([value.flatten() for value in captured]))
    assert not torch.allclose(first["field_training_reconstruction"], third["field_training_reconstruction"])
    # Actual inference without a teacher has the same conditional sample law.
    model.eval()
    prediction = model.score_regimens(patient, regimens, diffusion_context=context)
    perturbed = model.score_regimens(patient, regimens + 10, diffusion_context=changed_context, regimen_chunk_size=1)
    assert torch.equal(prediction["field_trajectory_samples"], perturbed["field_trajectory_samples"])
    missing = {key: value for key, value in context.items() if key != "action_identity"}
    with pytest.raises(ValueError, match="action_identity"):
        model.score_regimens(patient, regimens, diffusion_context=missing)
    with pytest.raises(ValueError, match="support"):
        model.score_regimens(patient, regimens, diffusion_context=dict(context, support=2 * context["support"]))


@pytest.mark.parametrize("policy", ["drug_id", "target_id"])
def test_native_policy_models_have_identical_shapes_budgets_and_occupancy_sensitivity(policy):
    drug, _, _, _ = _screening_action_case(policy)
    occupancy, patient, regimens, context = _screening_action_case("occupancy_program")
    assert {key: value.shape for key, value in drug.named_parameters()} == {
        key: value.shape for key, value in occupancy.named_parameters()}
    parameter_count = sum(parameter.numel() for parameter in drug.parameters())
    assert parameter_count == sum(parameter.numel() for parameter in occupancy.parameters())
    print(f"Matched bounded screening parameter count per policy: {parameter_count}")
    drug_sampler = drug.field_rollout_operator.quantized_mddm_rollout_operator
    occupancy_sampler = occupancy.field_rollout_operator.quantized_mddm_rollout_operator
    for key in ("hidden_dim", "species_count", "codebook_size", "training_sample_count", "inference_sample_count",
                "sample_chunk_size", "denoising_token_chunk_size", "training_denoising_step_count",
                "inference_denoising_step_count", "sampler_seed"):
        assert getattr(drug_sampler, key) == getattr(occupancy_sampler, key)
    teacher = torch.rand(1, 2, 3, 3, 2, 7)
    first = occupancy.score_regimens(patient, regimens, diffusion_context=context,
                                    teacher_trajectory=teacher, teacher_confidence=torch.ones(1, 2))
    first["rollout_auxiliary_loss"].backward()
    assert context["occupancy"].grad.abs().sum() > 0
    assert context["action_identity"].grad is None
    changed = occupancy.score_regimens(patient, regimens, diffusion_context=dict(context, occupancy=1-context["occupancy"]),
                                      teacher_trajectory=teacher, teacher_confidence=torch.ones(1, 2))
    assert not torch.allclose(first["field_training_reconstruction"], changed["field_training_reconstruction"])


def test_native_target_policy_and_legacy_profile_remain_supported():
    policy = "target_id"
    from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel
    from Libs.utils.data.native_diffusion_data import native_enabled
    from Libs.utils.model.screening_model_config import rollout_update_policy_from_config
    config, _, _ = _pretraining_confidence_case()
    rollout = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]
    rollout["action_representation"]["active_policy"] = policy
    assert MicroenvironmentPretrainingModel(config, False).quantized_mddm.action_policy_name == policy
    assert native_enabled(config)
    assert QuantizedMddmRolloutOperator(quantized_mddm_test_config(), 12, 7, action_policy=policy).action_policy_name == policy
    with pytest.raises(ValueError, match="unimplemented"):
        QuantizedMddmRolloutOperator(quantized_mddm_test_config(), 12, 7, action_policy="unknown")
    rollout["transition_backbone"]["quantized_mddm_rollout"]["enabled"] = False
    assert not native_enabled(config)
    rollout["transition_backbone"]["mechanism_moe"]["router_loss_weight"] = 0.0
    assert rollout_update_policy_from_config(config)["action_representation"]["active_policy"] == policy
    assert MicroenvironmentPretrainingModel(config, False).quantized_mddm is None


@pytest.mark.skipif(
    os.environ.get("NEURONSIFTER_FULL_NATIVE_RUNTIME") != "1",
    reason="Opt-in full Large synthetic CPU runtime; requires an explicitly allocated memory/CPU budget.",
)
def test_full_native_reference_runtime(monkeypatch):
    """Full architecture diagnostic only: no dataset, optimizer or scientific result."""
    from Libs.exp.training.pretrain_microenvironment_world_model import (
        MicroenvironmentPretrainingModel, run_pretraining_step, validate_default_parameter_budget,
    )
    from Libs.utils.model.transolver_parameter_budget import validate_parameter_budget

    started = time.perf_counter()
    torch.manual_seed(23)
    config = apply_neuronsifter_scale(
        load_config("Libs/configs/config_stack.yaml"), "neuronsifter_large", "large_main_screening")
    backbone_config = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    assert [backbone_config[key] for key in (
        "operator_hidden_dim", "operator_depth", "attention_head_count", "mechanism_state_token_count")] == [696, 8, 12, 96]
    assert len(config["dynamics_ready_registry"]["species"]) == 15
    print(json.dumps({"phase": "constructing_full_native_reference", "torch_threads": torch.get_num_threads(),
                      "dtype": "float32", "scope": "synthetic software diagnostic"}), flush=True)
    model = MicroenvironmentPretrainingModel(config, use_smoke_dimensions=False).train()
    native, sampler = model.operator, model.quantized_mddm
    assert native.hidden_dim == 696 and native.depth == len(native.blocks) == 8
    assert native.mechanism_token_mixer.token_count == 96
    assert native.tokenizer.graph_mode_count == 36 and native.tokenizer.wavelet_level_count == 3
    assert sampler.hidden_dim == 696 and sampler.species_count == 15
    assert sampler.action_policy_name == "occupancy_program"
    assert sampler.training_sample_count == 2 and sampler.training_denoising_step_count == 4
    assert sampler.inference_sample_count == sampler.inference_denoising_step_count == 16
    resolved = quantized_mddm_rollout_from_config(config, backbone_config)
    assert resolved["training_objective"] == "absorbing_masked_categorical_nonempty_importance"
    assert sampler.codebook_size == resolved["residual_tokenization"]["codebook_size"]
    experts = backbone_config["mechanism_moe"]["expert_names"]
    order = backbone_config["operator_splitting"]["composition_order"]
    assert len(experts) == len(order) == 10 and set(experts) == set(order)
    for block in native.blocks:
        assert block.self_attention.embed_dim == block.cross_attention.embed_dim == 696
        assert block.self_attention.num_heads == block.cross_attention.num_heads == 12
        assert block.operator_splitting_expert_bank.expert_names == experts
        assert block.operator_splitting_expert_bank.composition_order == order
        assert set(block.operator_splitting_expert_bank.experts) == set(experts)
    assert all(parameter.dtype == torch.float32 and parameter.device.type == "cpu" for parameter in model.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    native_total = sum(parameter.numel() for parameter in native.parameters())
    analytic = validate_default_parameter_budget(config)
    budget = backbone_config["parameter_budget"]
    print(json.dumps({"phase": "constructed", "seconds": time.perf_counter() - started,
        "model_total_parameters": total, "model_trainable_parameters": trainable,
        "native_backbone_parameters": native_total,
        "nominal_trainable_parameters": config["neuronsifter_scale_selection"]["nominal_trainable_parameter_count"],
        "analytic_active_parameters": analytic, "parameter_budget": budget,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
    validate_parameter_budget(trainable, int(budget["active_parameter_lower_bound"]), int(budget["active_parameter_upper_bound"]))

    # Independent synthetic conditioning and target streams; none reads a registry.
    inputs = torch.Generator().manual_seed(101)
    targets = torch.Generator().manual_seed(303)
    shape = (1, 28, 6, 6, 1)
    initial = torch.rand(1, 6, 6, 15, generator=inputs, requires_grad=True)
    batch = {
        "trajectory": torch.rand(1, 28, 6, 6, 15, generator=targets),
        "initial_state": initial,
        "posterior_context": torch.rand(1, 6, 6, 696, generator=inputs, requires_grad=True),
        "occupancy": torch.rand(shape, generator=inputs, requires_grad=True),
        "support": torch.ones(shape),
        "physical_time": torch.linspace(0., 672., 28).reshape(1, 28, 1, 1, 1).expand(shape),
        "teacher_confidence": torch.ones(1), "teacher_loss_weight": torch.ones(1),
        "uncertainty": torch.full((1, 15), .25), "sample_ids": torch.tensor([0], dtype=torch.long),
    }
    residual = batch["trajectory"].unsqueeze(1) - initial.detach().unsqueeze(1).unsqueeze(2)
    packed, layout = sampler.pack(residual, native)
    assert packed.shape == (1, 1, 29, 6, 6, 15)
    assert layout == {"lengths": [28, 14, 7], "sizes": [4, 4, 7, 14]}
    assert torch.allclose(sampler.unpack(packed, layout, native), residual, atol=1e-6)

    observed, block_calls, exhausted_masks = {}, [0] * 8, []
    def observe_block(index):
        def record(module, args):
            assert args[0].shape == args[1].shape == (1, 1044, 696)
            assert args[2].shape == (1, 1140, 696)
            block_calls[index] += 1
            if index == 0:
                print(json.dumps({"phase": "native_denoiser_call", "call": block_calls[index],
                                  "seconds": time.perf_counter() - started,
                                  "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
        return record
    def request_samples(module, args, kwargs):
        assert args[0].shape == (1, 1, 28, 6, 6, 15) and args[1] is True
        # The public return flag exposes draws already computed by the normal call.
        return args, dict(kwargs, return_samples=True)
    def observe_samples(module, args, output):
        observed.update({key: value.detach() if torch.is_tensor(value) else {
            name: tensor.detach() for name, tensor in value.items()} for key, value in output.items()})
        print(json.dumps({"phase": "native_forward_complete", "seconds": time.perf_counter() - started,
                          "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)
    reverse_step = sampler.reverse_step
    def observe_reverse(*args, **kwargs):
        tokens = reverse_step(*args, **kwargs)
        if args[3] == 0:
            remaining = int((tokens == sampler.codebook_size).sum())
            assert remaining == 0
            exhausted_masks.append(remaining)
        return tokens
    monkeypatch.setattr(sampler, "reverse_step", observe_reverse)
    handles = [block.register_forward_pre_hook(observe_block(index)) for index, block in enumerate(native.blocks)]
    handles.extend([sampler.register_forward_pre_hook(request_samples, with_kwargs=True),
                    sampler.register_forward_hook(observe_samples)])
    forward_backward_started = time.perf_counter()
    try:
        metrics, router = run_pretraining_step(model, batch, config, torch.device("cpu"),
            {"mixed_precision": {"enabled": False}, "gradient_accumulation_steps": 1})
    finally:
        for handle in handles:
            handle.remove()
    forward_backward_seconds = time.perf_counter() - forward_backward_started
    assert block_calls == [9] * 8 and exhausted_masks == [0, 0]
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["native_masked_diffusion_loss"] > 0
    samples = observed["sample_trajectories"]
    assert samples.shape == (2, 1, 1, 28, 6, 6, 15)
    assert torch.equal(samples[:, :, :, 0], initial.detach().unsqueeze(0).unsqueeze(2).expand(2, -1, -1, -1, -1, -1))
    assert not torch.equal(samples[0], samples[1])
    for key in ("sample_trajectories", "training_reconstruction", "trajectory_mean", "trajectory_std",
                "trajectory_quantiles", "ranking_stability", "rollout_auxiliary_loss"):
        assert torch.isfinite(observed[key]).all(), key
    assert observed["trajectory_mean"].shape == observed["trajectory_std"].shape == (1, 1, 28, 6, 6, 15)
    assert observed["trajectory_quantiles"].shape == (len(sampler.quantile_levels), 1, 1, 28, 6, 6, 15)
    assert observed["trajectory_std"].max() > 0
    diagnostics = observed["sampler_diagnostics"]
    state_count = native.blocks[0].operator_splitting_expert_bank.state_name_count
    diagnostic_shapes = {
        "expert_utilization": (10,), "operator_splitting_expert_utilization": (10,),
        "operator_splitting_species_attribution": (10, state_count),
        "operator_splitting_coupling_matrix": (10, 10),
        "operator_splitting_score_norm_by_expert": (10,),
        "operator_splitting_uncertainty_by_expert": (10,),
        "operator_splitting_case_activation": (1, 1, 10),
    }
    for key, expected in diagnostic_shapes.items():
        assert diagnostics[key].shape == expected and torch.isfinite(diagnostics[key]).all(), key
    assert diagnostics["score_block_count"] == 8 and diagnostics["operator_splitting_composition_count"] == 10
    assert diagnostics["mddm_sample_count"] == 2 and diagnostics["mddm_denoising_step_count"] == 4
    assert diagnostics["mddm_final_mask_probability"] == 0
    assert len(router["expert_utilization"]) == 10
    gradient_elements, nonzero_gradient_elements, missing_gradients = 0, 0, []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing_gradients.append(name)
            continue
        assert torch.isfinite(parameter.grad).all(), name
        gradient_elements += parameter.numel()
        nonzero_gradient_elements += int(torch.count_nonzero(parameter.grad))
    for name in ("initial_state", "posterior_context", "occupancy"):
        assert batch[name].grad is not None and torch.isfinite(batch[name].grad).all()
        assert torch.count_nonzero(batch[name].grad) > 0, name
    for module in (sampler.value_projection, sampler.condition_projection, sampler.context_projection,
                   sampler.token_denoiser, native.blocks[0].self_adaln.condition_projection,
                   native.blocks[-1].operator_splitting_expert_bank):
        assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in module.parameters())
    for block in native.blocks:
        for expert in block.operator_splitting_expert_bank.experts.values():
            assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in expert.parameters())
    print(json.dumps({"phase": "full_native_reference_pass", "model_total_parameters": total,
        "native_backbone_parameters": native_total, "gradient_parameter_elements": gradient_elements,
        "nonzero_gradient_elements": nonzero_gradient_elements, "parameters_without_gradient": missing_gradients,
        "block_calls": block_calls, "field_tokens": 1044, "cross_tokens": 1140, "packed_times": 29,
        "species_channels": 15, "sample_shape": list(samples.shape), "ordered_experts": order,
        "diagnostic_shapes": {key: list(value) for key, value in diagnostic_shapes.items()},
        "losses": metrics, "final_mask_counts": exhausted_masks,
        "forward_backward_seconds": forward_backward_seconds, "total_seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "full architecture synthetic CPU software diagnostic; no optimizer or scientific qualification"}), flush=True)



@pytest.mark.parametrize("consumer", ["pretraining", "screening"])
def test_native_target_checkpoint_binds_policy_and_schema(tmp_path, consumer):
    import json
    from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel
    from Libs.models.modules.world_action_encoder import NATIVE_TARGET_IDENTITY_SCHEMA
    from Libs.models.modules.quantized_mddm_rollout_operator import validate_native_action_checkpoint
    from Libs.tests.test_scalar_exposure import write_and_read_checkpoint, resume_state
    def make_model(policy):
        if consumer == "screening":
            return _screening_action_case(policy)[0]
        config, _, _ = _pretraining_confidence_case()
        config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"] = policy
        return MicroenvironmentPretrainingModel(config, False)
    model = make_model("target_id")
    state = resume_state(write_and_read_checkpoint(model, consumer, tmp_path), consumer)
    key = next(name for name in state if name.endswith("action_conditioning_identity"))
    expected = {"action_policy": "target_id", "scalar_exposure_definition": None,
                "action_identity_schema": NATIVE_TARGET_IDENTITY_SCHEMA}
    assert json.loads(bytes(state[key].tolist())) == expected
    for strict in (True, False):
        result = model.load_state_dict(state, strict=strict)
        assert not result.missing_keys and not result.unexpected_keys
        for policy in ("drug_id", "occupancy_program"):
            other = make_model(policy)
            with pytest.raises(ValueError, match="identity"):
                other.load_state_dict(state, strict=strict)
            with pytest.raises(ValueError, match="identity"):
                model.load_state_dict(other.state_dict(), strict=strict)
    for mutation in ("missing", "schema", "policy"):
        invalid = dict(state)
        if mutation == "missing":
            invalid.pop(key)
        else:
            changed = dict(expected, **({"action_identity_schema": "canonical_agent_set_sha256_float4.v1"}
                                       if mutation == "schema" else {"action_policy": "drug_id"}))
            invalid[key] = torch.tensor(list(json.dumps(changed).encode()), dtype=torch.uint8)
        with pytest.raises(ValueError, match="identity"):
            validate_native_action_checkpoint(model, invalid)
        with pytest.raises(ValueError, match="identity"):
            model.load_state_dict(invalid, strict=False)


def test_target_profile_preserves_backbone_and_seed_boundary():
    from Libs.utils.exp.experiment_manifests import load_experiment_family, deep_merge
    from Libs.utils.data.native_diffusion_data import native_enabled
    config = load_config("Libs/configs/config_stack.yaml")
    profile = load_experiment_family("action_conditioning_target_identity", config)
    target = deep_merge(config, profile["config_overrides"])
    before = config["screening_pipeline"]["screening_benchmark"]
    after = target["screening_pipeline"]["screening_benchmark"]
    assert after["model_architecture"]["rollout_operator"]["action_representation"]["active_policy"] == "target_id"
    assert after["model_architecture"]["rollout_operator"]["transition_backbone"] == before["model_architecture"]["rollout_operator"]["transition_backbone"]
    assert after["model_architecture"]["world_state_posterior"] == before["model_architecture"]["world_state_posterior"]
    assert after["training"] == before["training"]
    assert profile["runtime"]["seed_values"] == [3407, 42, 26]
    assert native_enabled(target)
