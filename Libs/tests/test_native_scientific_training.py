"""Tiny native models on separate real-builder fixtures; no production slots."""
import copy

import pytest
import torch
from torch.utils.data import DataLoader

from Libs.exp.training import pretrain_microenvironment_world_model as pretrain
from Libs.tests.test_native_observable_source import build_disposable_observable_fixture
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.native_training_selection import training_checkpoint_path


def test_all_family_paired_training_and_immutable_selection(tmp_path, monkeypatch):
    _, _, _, config, _ = build_disposable_observable_fixture(tmp_path, monkeypatch)
    config["screening_pipeline"]["screening_benchmark"]["training"]["data_loader"]["num_workers"] = 0
    scientific = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_scientific_training"] = copy.deepcopy(scientific["native_scientific_training"])
    policy = config["native_scientific_training"]
    policy["selection"].update(max_epochs=3, minimum_epochs=1, minimum_steps=1, patience=2,
                                min_delta=1e6, max_steps=20, max_seconds=1000)
    policy["validation"].update(sample_count=2, denoising_step_count=1)
    policy["learning_rate_schedule"]["warmup_steps"] = 1
    posterior_config = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]
    posterior_config["joint_objectives"] = copy.deepcopy(
        scientific["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["joint_objectives"])
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone.update(operator_hidden_dim=12, operator_depth=1, attention_head_count=3, mechanism_state_token_count=6)
    config["native_diffusion_data"].update(posterior_context_width=12, source_family_uid=None)
    train = pretrain.NativeJointGroupDataset(config, "train")
    development = pretrain.NativeJointGroupDataset(config, "development")
    assert len(train) == len(development) == 3
    assert set(train.family_group_counts.values()) == {1}
    train_ids = {train[i]["native_joint_group"]["items"][0]["sample_uid"] for i in range(len(train))}
    dev_ids = {development[i]["native_joint_group"]["items"][0]["sample_uid"] for i in range(len(development))}
    assert not train_ids & dev_ids
    torch.manual_seed(973)
    model = pretrain.MicroenvironmentPretrainingModel(config, False)
    model.quantized_mddm.training_sample_count = 2
    model.quantized_mddm.training_denoising_step_count = 1
    model.quantized_mddm.inference_sample_count = 2
    model.quantized_mddm.inference_denoising_step_count = 1
    model.quantized_mddm.sampler_seed = 973
    group = train[0]["native_joint_group"]
    model.train()
    loss, metrics, status = pretrain.scientific_joint_group_loss(model, group, config, torch.device("cpu"))
    loss.backward()
    assert status["particle_count"] == 4 and status["paired_target_count"] == 2
    assert status["belief_evidence_status"]["held_sensing_block_count"] > 0
    assert model.world_state_posterior.joint_candidate_parameter_projection.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # Mutating future targets changes scoring, not the target-free predictions.
    captured = []
    def capture(_module, _args, output):
        captured.append(output["sample_trajectories"].detach().clone())
    hook = model.quantized_mddm.register_forward_hook(capture)
    model.eval()
    with torch.no_grad():
        _, first, _ = pretrain.scientific_joint_group_loss(model, group, config, torch.device("cpu"))
        reference = captured[:]
        captured.clear()
        changed = copy.deepcopy(group)
        for item in changed["items"]:
            item["trajectory"][1:] += .01 * item["quantity_support"][1:]
        _, second, _ = pretrain.scientific_joint_group_loss(model, changed, config, torch.device("cpu"))
    hook.remove()
    assert len(reference) == len(captured) == 4
    assert all(torch.equal(a, b) for a, b in zip(reference, captured))
    assert first["crps"] != second["crps"]
    loader = DataLoader(train, batch_size=1, collate_fn=pretrain.collate_native_sources)
    optimizer = pretrain.create_adamw_optimizer(model, learning_rate=1e-4, weight_decay=1e-4)
    output = tmp_path / "checkpoints"
    output.mkdir()
    result = pretrain.fit_scientific_native_groups(
        model, optimizer, loader, config, torch.device("cpu"),
        dict(config["screening_pipeline"]["screening_benchmark"]["training"]["memory_policy"], gradient_accumulation_steps=1), output, "disposable", {"model_seed":973})
    selection = result["selection"]
    assert result["train_group_count"] == result["development_group_count"] == 3
    assert result["executed_steps"] == 9
    assert selection["best"]["checkpoint_path"] != selection["last"]["checkpoint_path"]
    manifest_path = output / "microenvironment_pretraining_manifest.json"
    best = torch.load(training_checkpoint_path(manifest_path, selection, selection["best"]["checkpoint_path"]), weights_only=False)
    last = torch.load(training_checkpoint_path(manifest_path, selection, selection["last"]["checkpoint_path"]), weights_only=False)
    assert best["epoch"] == 1 and last["epoch"] == 3
    assert result["retained_checkpoint_count"] == 2
    assert not (output / "microenvironment_pretraining_disposable_epoch_2.pt").exists()
    assert len(selection["history"]) == 3
    assert all("checkpoint_path" not in row for row in selection["history"])
    assert not selection["sufficient_training"]
    assert any(not torch.equal(best["model_state_dict"][k], last["model_state_dict"][k])
               for k in best["model_state_dict"])
    with pytest.raises(FileExistsError, match="already exists"):
        pretrain.save_training_checkpoint(model, optimizer, output,
            "microenvironment_pretraining_disposable_epoch_1", 1, 3, "single_process", immutable=True)

def test_scientific_native_forward_and_belief_views_without_builder(tmp_path):
    from Libs.tests.test_joint_physical_parameters import named_case
    from Libs.utils.data.native_diffusion_data import bind_joint_programs, joint_particle_contexts
    from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support
    config, group, _, _, raw, _, _ = named_case(tmp_path, "electrophysiology_conductance_world")
    scientific = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_scientific_training"] = scientific["native_scientific_training"]
    # Physical coordinates are a diagnostic convention here, not a replacement
    # for the fitted transform required by the scientific group objective.
    config["native_diffusion_data"]["registry_path"] = None
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    backbone.update(operator_hidden_dim=12, operator_depth=1, attention_head_count=3, mechanism_state_token_count=6)
    config["native_diffusion_data"]["posterior_context_width"] = 12
    model = pretrain.MicroenvironmentPretrainingModel(config, False)
    assert model.scientific_training_enabled and model.effective_config is None
    from Libs.utils.data.native_field_normalization import NativeFieldNormalization
    model.field_normalization = NativeFieldNormalization(
        {"kind":"physical", "quantity_profile":observable_profile(config)}, 26)
    model.quantized_mddm.training_sample_count = 2
    model.quantized_mddm.training_denoising_step_count = 1
    model.train()
    posterior = model(None, None, joint_source=raw)
    programs = bind_joint_programs(posterior, raw)
    contexts = joint_particle_contexts(posterior, programs, programs["regimen_uids"])
    grid = family_grid(observable_profile(config), group["family_uid"])
    mask = torch.as_tensor(quantity_support(grid))
    target = posterior.support.initial_field[0, 0].expand(2, len(grid["time_hours"]), -1, -1, -1).clone()
    batch = {key: value[0] for key, value in contexts[0].items()}
    batch.update(trajectory=target, teacher_confidence=torch.ones(2),
                 normalization_id=["physical", "physical"], sample_ids=torch.arange(2))
    output = model(None, pretrain.node_coordinates(6, 6, torch.device("cpu")), diffusion_batch=batch)
    assert output["sample_trajectories"].shape == (2, 2, 1, len(grid["time_hours"]), 6, 6, 26)
    assert output["sample_log_probabilities"].shape == (2, 2, 1)
    assert output["sample_log_probabilities"].requires_grad
    assert torch.isfinite(output["sample_log_probabilities"]).all()
    original_source = copy.deepcopy(raw)
    held, nuisance, status = pretrain.controlled_belief_regularizers(model, raw, posterior, config)
    assert raw == original_source
    # Augmentation did not mutate or relax the authoritative program binding.
    from Libs.utils.data.native_diffusion_data import parse_joint_source
    parse_joint_source(raw)
    assert status["held_sensing_block_count"] > 0
    objective = output["rollout_auxiliary_loss"] + .001 * output["sample_log_probabilities"].mean() + held + nuisance
    objective.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.world_state_posterior.joint_candidate_parameter_projection.weight.grad.abs().sum() > 0
    model.eval()
    model.quantized_mddm.inference_sample_count = 2
    model.quantized_mddm.inference_denoising_step_count = 1
    with torch.no_grad():
        evaluation = model(None, pretrain.node_coordinates(6, 6, torch.device("cpu")), diffusion_batch=batch)
    assert evaluation["sample_trajectories"].shape == output["sample_trajectories"].shape
    assert not evaluation["sample_log_probabilities"].requires_grad
    assert evaluation["sample_log_probabilities"].grad_fn is None
    assert evaluation["sample_trajectories"].grad_fn is None


def test_categorical_step_checkpoint_replays_draw_and_gradient(tmp_path):
    from torch.utils.checkpoint import checkpoint
    # A controlled categorical logit fixture isolates exact RNG/gradient replay.
    from Libs.models.modules.quantized_mddm_rollout_operator import QuantizedMddmRolloutOperator
    operator = object.__new__(QuantizedMddmRolloutOperator)
    torch.nn.Module.__init__(operator)
    operator.codebook_size = 3
    parameter = torch.nn.Parameter(torch.tensor([.2, -.3, .1]))
    operator.logits = lambda values, *args: (parameter.expand(*values.shape, 3), {})
    values = torch.zeros(2, 2, 1)
    tokens = torch.full_like(values, 3, dtype=torch.long)
    generator = torch.Generator().manual_seed(8703)
    arguments = (tokens, values, tokens == 3, torch.tensor(1.), 0.,
                 generator.get_state(), None, {}, None, None, None)
    plain = operator.score_reverse_step(*arguments)
    plain[1].backward()
    plain_gradient = parameter.grad.clone()
    parameter.grad = None
    replayed = checkpoint(operator.score_reverse_step, *arguments, use_reentrant=False)
    replayed[1].backward()
    assert torch.equal(replayed[0], plain[0])
    assert torch.equal(replayed[3], plain[3])
    assert torch.equal(replayed[1], plain[1])
    assert torch.allclose(parameter.grad, plain_gradient)


@pytest.mark.parametrize("failure_phase", ["first_microbatch", "development"])
def test_failed_validation_preserves_best_and_latest_checkpoint(tmp_path, monkeypatch, failure_phase):
    import json
    from types import SimpleNamespace

    class Groups(torch.utils.data.Dataset):
        family_group_counts = {"disposable":1}
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return index

    config = load_config("Libs/configs/exp/manuscript_native_protocol.yaml")
    config["native_scientific_training"]["selection"].update(
        max_epochs=3, minimum_epochs=1, minimum_steps=1, max_seconds=60)
    dataset = Groups()
    collate = lambda rows: {"joint_groups":rows}
    monkeypatch.setattr(pretrain, "NativeJointGroupDataset", lambda *args: dataset)
    monkeypatch.setattr(pretrain, "data_loader_kwargs", lambda _: {"collate_fn":collate})
    model = torch.nn.Linear(1, 1)
    model.quantized_mddm = SimpleNamespace(training_sample_count=2, training_denoising_step_count=1,
        inference_sample_count=2, inference_denoising_step_count=1, sampler_seed=40)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    validations = 0

    def group_batch(model, groups, config, device, memory_policy, *, backward, learned_teacher=None):
        nonlocal validations
        if backward:
            if failure_phase == "first_microbatch":
                raise RuntimeError("disposable first microbatch failure")
            loss = model.weight.square().sum()
            loss.backward()
            return {"total_loss":float(loss.detach())}, {}
        validations += 1
        if validations == 2:
            raise RuntimeError("disposable development failure")
        return {"total_loss":1.0}, {}

    def step(model, optimizer, config):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    monkeypatch.setattr(pretrain, "scientific_group_batch", group_batch)
    monkeypatch.setattr(pretrain, "execute_optimizer_step", step)
    with pytest.raises(RuntimeError, match="disposable .* failure"):
        pretrain.fit_scientific_native_groups(
            model, optimizer, DataLoader(dataset, collate_fn=collate), config,
            torch.device("cpu"), {"gradient_accumulation_steps":1},
            tmp_path, "disposable", {"model_seed":40})
    payload = json.loads((tmp_path/"microenvironment_pretraining_manifest.json").read_text())
    assert payload["selection"]["terminal"] == "FAILED"
    if failure_phase == "first_microbatch":
        assert payload["first_microbatch_runtime"] == {"status":"NOT_COMPLETED"}
        assert payload["executed_micro_batches"] == payload["executed_steps"] == 0
        assert payload["retained_checkpoint_count"] == 0
        assert payload["selection"]["best"] is payload["selection"]["last"] is None
        assert payload["selection"]["history"] == []
        assert not list(tmp_path.glob("*.pt")) and validations == 0
        assert payload["failure"]["message"] == "disposable first microbatch failure"
        assert model.quantized_mddm.sampler_seed == 40
        return
    assert payload["first_microbatch_runtime"]["status"] == "COMPLETED"
    assert payload["first_microbatch_runtime"]["cuda_memory"] == {"status":"NOT_APPLICABLE"}
    assert payload["selection"]["best"]["epoch"] == 1
    assert payload["selection"]["last"]["epoch"] == 2
    assert payload["selection"]["last"]["development_loss"] is None
    assert payload["retained_checkpoint_count"] == 2
    for key in ("best", "last"):
        checkpoint = torch.load(training_checkpoint_path(
            tmp_path/"microenvironment_pretraining_manifest.json", payload["selection"],
            payload["selection"][key]["checkpoint_path"]), weights_only=False)
        assert checkpoint["epoch"] == payload["selection"][key]["epoch"]
    assert not payload["selection"]["sufficient_training"]
    assert model.quantized_mddm.sampler_seed == 40
    assert model.quantized_mddm.inference_sample_count == 2
