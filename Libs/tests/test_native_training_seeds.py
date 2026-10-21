from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.screening_model_config import rollout_update_policy_from_config
from Libs.utils.exp.experiment_manifests import load_experiment_family, resolve_training_seed, validate_training_seed_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ["dynamics_mainline_ad", "action_conditioning_drug_identity",
            "rollout_operator_parallel_experts", "action_conditioning_scalar_exposure",
            "rollout_operator_reversed_experts", "rollout_operator_interleaved_experts"]
SEEDS = [40, 41, 42, 43, 44]


@pytest.fixture
def base(tmp_path):
    config = load_config("Libs/configs/config_stack.yaml")
    config["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"] = "manuscript_native_five"
    config["logging_registry"].update(file_enabled=False, console_enabled=False, root_dir=str(tmp_path / "logs"))
    config["execution_environment"]["python_executable"] = sys.executable
    return config


def selected(config, seed=41):
    config = copy.deepcopy(config)
    config["selected_formal_seed"] = seed
    config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"] = seed
    return config


def cli_materialize(config, directory, profile, seed=None):
    directory.mkdir(parents=True, exist_ok=True)
    source, output = directory / "base.yaml", directory / "selected.yaml"
    source.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [sys.executable, str(ROOT / "Libs/exp/orchestration/prepare_experiment_config.py"),
               "--profile", profile, "--base-config", str(source), "--output-config", str(output)]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    result = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    return load_config(output), output


@pytest.mark.parametrize("profile", [*PROFILES, "action_conditioning_target_identity"])
def test_actual_cli_materializes_all_five_with_unrelated_seeds_unchanged(base, tmp_path, profile):
    assert load_experiment_family(profile, base)["runtime"]["seed_values"] == SEEDS
    untouched = {name: copy.deepcopy(base[name]) for name in
                 ("neuronsifter_scale_registry", "paper_grade_full_suite", "paper_grade_baseline_registry")}
    per_run = []
    for seed in SEEDS:
        config, _ = cli_materialize(base, tmp_path / str(seed), profile, seed)
        binding = resolve_training_seed(config, require_selected=True)
        if config["paper_experiments"][profile].get("expert_order_control", False):
            backbone = rollout_update_policy_from_config(config)["transition_backbone"]
            assert len(backbone["mechanism_moe"]["expert_names"]) == 10
            assert backbone["operator_splitting"]["composition_mode"] == "serial"
        assert binding["allowed_training_seeds"] == SEEDS
        assert binding["selected_formal_seed"] == binding["base_training_seed"] == binding["model_seed"] == seed
        for name, value in untouched.items():
            assert config[name] == value
        for name in ("shared_latent_model", "logistic_regression"):
            assert config["benchmark_registry"][name]["random_seed"] == base["benchmark_registry"][name]["random_seed"]
        assert config["screening_pipeline"]["screening_benchmark"]["formal_seed_values"] == [3407, 42, 26]
        comparable = copy.deepcopy(config)
        comparable.pop("selected_formal_seed")
        benchmark = comparable["screening_pipeline"]["screening_benchmark"]
        benchmark["training"]["random_seed"] = 0
        benchmark.pop("benchmark_name")
        per_run.append(comparable)
    assert all(value == per_run[0] for value in per_run[1:])


def test_unselected_prebuild_and_lawful_subset(base, tmp_path):
    config, _ = cli_materialize(base, tmp_path / "shared", PROFILES[0])
    assert "selected_formal_seed" not in config
    assert resolve_training_seed(config)["base_training_seed"] is None
    with pytest.raises(ValueError, match="materialized"):
        resolve_training_seed(config, require_selected=True)
    for seed in (41, 44):
        subset, _ = cli_materialize(base, tmp_path / str(seed), PROFILES[0], seed)
        assert resolve_training_seed(subset, require_selected=True)["allowed_training_seeds"] == SEEDS


@pytest.mark.parametrize("bad", [[40,41,42,43], [40,41,42,43,44,45], [40,41,42,43,43],
                                  [40,41,42,43,True], [40,41,42,43,"44"], [40,41,42,43,-1]])
def test_invalid_policy_lists_reject(base, bad):
    base["paper_grade_full_suite"]["formal_seed_values"] = bad
    with pytest.raises(ValueError, match="exact ordered integer"):
        load_experiment_family(PROFILES[0], base)


@pytest.mark.parametrize("mutation", ["suite", "baseline", "selected", "effective", "request", "unknown_policy"])
def test_inconsistent_bindings_reject(base, mutation):
    config = selected(base)
    request = None
    if mutation == "suite":
        config["paper_grade_full_suite"]["seeds"] = [44,43,42,41,40]
    elif mutation == "baseline":
        config["paper_grade_baseline_registry"]["seed_values"] = [40,41,42,43,45]
    elif mutation == "selected":
        config["selected_formal_seed"] = 39
    elif mutation == "effective":
        config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"] = 44
    elif mutation == "request":
        request = 44
    else:
        config["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"] = "unknown"
    with pytest.raises(ValueError, match="seed|declaration"):
        resolve_training_seed(config, request, require_selected=True)


def metadata_args(path, directory, seed=None):
    return argparse.Namespace(config=str(path), suite_config="fixture", seed=seed,
        task="world_model_pretraining", model_family="neuronsifter_mainline",
        experiment_family=PROFILES[0], output_dir=str(directory), run_id="",
        checkpoint="", freeze_manifest=str(directory / "absent-freeze.json"),
        dry_run=False, allow_pending=False)


def local_metadata_collectors(monkeypatch):
    from Libs.utils import exp_utils
    monkeypatch.setattr(exp_utils, "get_git_head", lambda: "fixture")
    monkeypatch.setattr(exp_utils, "compute_git_full_diff_hash", lambda: "fixture")
    monkeypatch.setattr(exp_utils, "_data_manifest_hash", lambda config: "fixture")
    monkeypatch.setattr(exp_utils, "_device_label", lambda: "cpu")


@pytest.mark.parametrize("seed", [41,44])
@pytest.mark.parametrize("explicit", [False,True])
def test_nonfirst_wrapper_delegate_and_actual_run_metadata(base, tmp_path, monkeypatch, seed, explicit):
    from Libs.utils.exp import paper_grade_entrypoint as entry
    from Libs.utils import exp_utils
    config = selected(base, seed)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    args = metadata_args(path, tmp_path / "output", seed if explicit else None)
    local_metadata_collectors(monkeypatch)
    monkeypatch.setattr(entry, "parse_standard_args", lambda *unused: args)
    calls = []
    monkeypatch.setattr(entry.subprocess, "run", lambda command, **kwargs: calls.append(command))
    entry.run_delegate_entrypoint("fixture", args.task, "Libs/exp/training/pretrain_microenvironment_world_model.py")
    assert len(calls) == 1 and calls[0][-2:] == ["--config", str(path)]
    metadata = json.loads((tmp_path / "output/run_metadata.json").read_text())
    assert metadata["seed"] == seed and metadata["run_id"].endswith("seed_" + str(seed))
    assert metadata["training_seed_binding"]["selected_formal_seed"] == seed
    args.output_dir = ""
    assert "seed_" + str(seed) in str(exp_utils.resolve_output_dir(args, config))
    direct = exp_utils.build_run_metadata(args, config)
    assert direct["seed"] == seed


@pytest.mark.parametrize("mutation", ["unselected", "mismatch", "invalid"])
def test_invalid_wrapper_rejects_before_delegate_or_metadata(base, tmp_path, monkeypatch, mutation):
    from Libs.utils.exp import paper_grade_entrypoint as entry
    config = base if mutation == "unselected" else selected(base)
    args = metadata_args(tmp_path / "config.yaml", tmp_path / "must-not-exist",
                         44 if mutation == "mismatch" else 39 if mutation == "invalid" else None)
    monkeypatch.setattr(entry, "parse_standard_args", lambda *unused: args)
    monkeypatch.setattr(entry, "load_effective_config", lambda unused: config)
    def forbidden(*args, **kwargs):
        raise AssertionError("delegate or successful output reached")
    monkeypatch.setattr(entry.subprocess, "run", forbidden)
    monkeypatch.setattr(entry, "write_entrypoint_outputs", forbidden)
    with pytest.raises(ValueError, match="seed"):
        entry.run_delegate_entrypoint("fixture", args.task, "Libs/exp/training/pretrain_microenvironment_world_model.py")
    assert not Path(args.output_dir).exists()


@pytest.mark.parametrize("consumer", ["pretrain", "screening"])
@pytest.mark.parametrize("mutation", ["unselected", "effective", "list"])
def test_native_entrypoints_reject_before_data_or_model(base, monkeypatch, consumer, mutation):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain, train_screening_system as screening
    module = pretrain if consumer == "pretrain" else screening
    config = base if mutation == "unselected" else selected(base)
    if mutation == "effective":
        config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"] = 44
    if mutation == "list":
        config["paper_grade_full_suite"]["formal_seed_values"] = [40,41,42,43]
    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(sys, "argv", ["fixture"])
    def forbidden(*args, **kwargs):
        raise AssertionError("data, model, or parameter-budget path reached")
    if consumer == "pretrain":
        monkeypatch.setattr(module, "apply_neuronsifter_scale", forbidden)
        monkeypatch.setattr(module, "build_pretraining_dataset", forbidden)
    else:
        monkeypatch.setattr(module, "benchmark_program_config", forbidden)
        monkeypatch.setattr(module, "assert_clean_action_ready_manifest", forbidden)
    with pytest.raises(ValueError, match="seed"):
        module.main()


@pytest.mark.parametrize("seed", [41,44])
def test_actual_pretrainer_seed_before_model_construction(base, monkeypatch, seed):
    from Libs.exp.training import pretrain_microenvironment_world_model as module
    config = selected(base, seed)
    monkeypatch.setattr(sys, "argv", ["fixture"])
    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "apply_neuronsifter_scale", lambda cfg, *args: cfg)
    monkeypatch.setattr(module, "validate_default_parameter_budget", lambda cfg: 0)
    monkeypatch.setattr(module, "build_pretraining_dataset", lambda *args, **kwargs: [None])
    monkeypatch.setattr(module, "training_memory_policy", lambda *args: {"micro_batch_size": 1})
    monkeypatch.setattr(module, "initialize_device_and_distributed", lambda *args: (torch.device("cpu"), "single_process"))
    monkeypatch.setattr(module, "data_loader_kwargs", lambda cfg: {})
    calls = []
    monkeypatch.setattr(module, "seed_everything", calls.append)
    class StopBeforeModel(Exception): pass
    monkeypatch.setattr(module, "MicroenvironmentPretrainingModel", lambda *args, **kwargs: (_ for _ in ()).throw(StopBeforeModel()))
    with pytest.raises(StopBeforeModel):
        module.main()
    assert calls == [seed]


def test_legacy_profile_and_non_native_materializer_remain_supported(base, tmp_path):
    from Libs.exp.orchestration.prepare_experiment_config import apply_seed_overrides
    base["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"] = "legacy_formal"
    for profile in PROFILES:
        assert load_experiment_family(profile, base)["runtime"]["seed_values"] == [3407,42,26]
    config = copy.deepcopy(base)
    apply_seed_overrides(config, {}, 123)
    assert config["benchmark_registry"]["shared_latent_model"]["random_seed"] == 123
    assert config["benchmark_registry"]["logistic_regression"]["random_seed"] == 123
    assert config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"] == 123


def scope_arguments(module, config, tmp_path):
    arguments = {name: None for name, parameter in inspect.signature(module.train_one_scope).parameters.items()
                 if parameter.default is inspect.Parameter.empty}
    arguments.update(config=config, config_path=tmp_path / "config.yaml", argv=[],
                     scope="folds", split_name="fixture_fold")
    return arguments


@pytest.mark.parametrize("seed", [41, 44])
def test_screening_actual_scoped_seed_before_cache(base, tmp_path, monkeypatch, seed):
    from Libs.exp.training import train_screening_system as module
    from Libs.utils.exp.experiment_tracking import scoped_seed
    config = selected(base, seed)
    binding = module.screening_training_seed_binding(config, "folds", "fixture_fold")
    assert binding["selected_formal_seed"] == binding["base_training_seed"] == seed
    assert binding["model_seed"] == scoped_seed(seed, "folds", "fixture_fold", "ad_screening_system")
    assert binding["model_seed"] not in SEEDS
    monkeypatch.setattr(module, "artifact_dir", lambda *args: tmp_path)
    monkeypatch.setattr(module, "get_pipeline_logger", lambda **kwargs: None)
    monkeypatch.setattr(module, "cache_enabled", lambda config: True)
    calls = []
    monkeypatch.setattr(module, "seed_everything", calls.append)
    class StopBeforeCache(Exception): pass
    def stop(*args, **kwargs):
        assert args[-1] == binding["model_seed"]
        raise StopBeforeCache
    monkeypatch.setattr(module, "load_scope_training_cache", stop)
    with pytest.raises(StopBeforeCache):
        module.train_one_scope(**scope_arguments(module, config, tmp_path))
    assert calls == [binding["model_seed"]]


@pytest.mark.parametrize("writer", ["pretrain", "pretrain_shard", "screening", "portable"])
def test_actual_checkpoint_writers_preserve_binding_without_training(base, tmp_path, monkeypatch, writer):
    from Libs.exp.training import pretrain_microenvironment_world_model as pretrain, train_screening_system as screening
    binding = (screening.screening_training_seed_binding(selected(base), "folds", "fixture_fold")
               if writer == "screening" else resolve_training_seed(selected(base), require_selected=True))
    model = SimpleNamespace(state_dict=lambda: {"opaque_model": "fixture"})
    optimizer = SimpleNamespace(state_dict=lambda: {"opaque_optimizer": "fixture"})
    if writer.startswith("pretrain"):
        monkeypatch.setattr(pretrain, "rank_local_model_shard", lambda model: model.state_dict())
        monkeypatch.setattr(pretrain, "distributed_rank", lambda: 0)
        strategy = "native_pytorch_fsdp" if writer == "pretrain_shard" else "single_process"
        path = pretrain.save_training_checkpoint(model, optimizer, tmp_path, "fixture", 1, 0, strategy,
                                                training_seed_binding=binding)
    elif writer == "screening":
        path = tmp_path / "fixture.pt"
        screening.save_checkpoint_state(path, model, optimizer, {"opaque_best": "fixture"}, 0.5,
                                        0, [], np.random.default_rng(1), 1, 0, training_seed_binding=binding)
    else:
        path = pretrain.save_portable_training_checkpoint(tmp_path / "fixture.pt",
                    {"model_state": model.state_dict(), "training_seed_binding": binding})
    payload = torch.load(path, map_location="cpu")
    validate_training_seed_checkpoint(payload, binding)
    assert payload["training_seed_binding"] == binding
    if writer == "screening":
        assert payload["best_training_seed_binding"] == binding


@pytest.mark.parametrize("location", ["current", "best", "completion", "final"])
@pytest.mark.parametrize("mutation", ["matching", "wrong_seed", "missing", "float_seed"])
def test_screening_reuse_checks_before_cache(base, tmp_path, monkeypatch, location, mutation):
    from Libs.exp.training import train_screening_system as module
    config = selected(base)
    binding = module.screening_training_seed_binding(config, "folds", "fixture_fold")
    state = {"training_seed_binding": copy.deepcopy(binding), "best_training_seed_binding": copy.deepcopy(binding),
             "model_state": {}, "best_state": {}}
    completion = {"status": "completed", "training_seed_binding": copy.deepcopy(binding)}
    final = {"model_state": {}, "training_seed_binding": copy.deepcopy(binding)}
    target = {"current": state, "best": state, "completion": completion, "final": final}[location]
    key = "best_training_seed_binding" if location == "best" else "training_seed_binding"
    if mutation == "missing":
        target.pop(key)
    elif mutation == "wrong_seed":
        target[key] = module.screening_training_seed_binding(selected(base, 44), "folds", "fixture_fold")
    elif mutation == "float_seed":
        target[key]["model_seed"] = float(target[key]["model_seed"])
    torch.save(state, tmp_path / "training_state_latest.pt")
    (tmp_path / "training_completion.json").write_text(json.dumps(completion))
    torch.save(final, tmp_path / "model_weights.pt")
    monkeypatch.setattr(module, "artifact_dir", lambda *args: tmp_path)
    monkeypatch.setattr(module, "get_pipeline_logger", lambda **kwargs: None)
    monkeypatch.setattr(module, "cache_enabled", lambda config: True)
    monkeypatch.setattr(module, "seed_everything", lambda seed: None)
    class StopBeforeCache(Exception): pass
    def stop(*args, **kwargs):
        raise StopBeforeCache
    monkeypatch.setattr(module, "load_scope_training_cache", stop)
    if mutation == "matching":
        current, completed = module.read_seed_bound_screening_checkpoints(tmp_path, binding)
        assert current == state and completed == final
        with pytest.raises(StopBeforeCache):
            module.train_one_scope(**scope_arguments(module, config, tmp_path))
    else:
        with pytest.raises(ValueError, match="checkpoint training seed binding"):
            module.train_one_scope(**scope_arguments(module, config, tmp_path))


@pytest.mark.parametrize("mutation", ["matching", "wrong_seed", "missing", "float_seed"])
def test_pretrainer_resume_checks_before_scale_or_data(base, tmp_path, monkeypatch, mutation):
    from Libs.exp.training import pretrain_microenvironment_world_model as module
    config = selected(base)
    binding = resolve_training_seed(config, require_selected=True)
    payload = {"training_seed_binding": copy.deepcopy(binding), "model_state_dict": {}, "optimizer_state_dict": {}}
    if mutation == "wrong_seed":
        payload["training_seed_binding"] = resolve_training_seed(selected(base, 44), require_selected=True)
    elif mutation == "missing":
        payload.pop("training_seed_binding")
    elif mutation == "float_seed":
        payload["training_seed_binding"]["base_training_seed"] = 41.0
    path = tmp_path / "resume.pt"
    torch.save(payload, path)
    monkeypatch.setattr(sys, "argv", ["fixture", "--resume-checkpoint", str(path)])
    monkeypatch.setattr(module, "load_config", lambda path: config)
    class StopBeforeScale(Exception): pass
    def stop(*args, **kwargs):
        raise StopBeforeScale
    monkeypatch.setattr(module, "apply_neuronsifter_scale", stop)
    with pytest.raises(StopBeforeScale if mutation == "matching" else ValueError):
        module.main()


@pytest.mark.parametrize("field", ["selected_formal_seed", "base_training_seed", "model_seed", "scope"])
def test_checkpoint_rejects_individually_changed_binding(base, field):
    binding = resolve_training_seed(selected(base), require_selected=True)
    changed = copy.deepcopy(binding)
    changed[field] = ["another_scope"] if field == "scope" else 44
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_training_seed_checkpoint({"training_seed_binding": changed}, binding)


def test_legacy_resume_compatibility_does_not_retag_prospective_checkpoint(base):
    prospective = resolve_training_seed(selected(base), require_selected=True)
    base["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"] = "legacy_formal"
    legacy = resolve_training_seed(base)
    validate_training_seed_checkpoint({"model_state": {}}, legacy)
    validate_training_seed_checkpoint({"training_seed_binding": legacy}, legacy)
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_training_seed_checkpoint({"training_seed_binding": prospective}, legacy)


@pytest.mark.parametrize("profile", [PROFILES[0], *PROFILES[-2:]])
@pytest.mark.parametrize("launcher", ["submit_profile_suite.sh", "submit_dynamics_profile_suite.sh"])
def test_actual_launcher_print_only_materializes_five_without_submission(base, tmp_path, launcher, profile):
    mini = tmp_path / "mini_repo"
    for relative in [f"Libs/exp/submission/{launcher}",
                     "Libs/exp/orchestration/prepare_experiment_config.py", "PROJECT_PREFIX"]:
        destination = mini / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "sinfo").write_text("#!/bin/sh\nexit 0\n")
    for command in ("sbatch", "srun", "squeue"):
        (binary / command).write_text('#!/bin/sh\nprintf forbidden > "$NO_SUBMIT_MARKER"\nexit 99\n')
    for path in binary.iterdir():
        path.chmod(0o755)
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    (fixture_root / "folds.tsv").write_text("fold_name\tdisease_program\nfixture_fold\tad_control\n")
    base["benchmark_registry"]["local_root"] = str(fixture_root)
    base["benchmark_registry"]["relative_paths"]["fold_catalog"] = "folds.tsv"
    path = tmp_path / "base.yaml"
    base["paper_experiments"][profile]["base_config_path"] = str(path)
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                       PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "Libs")]),
                       NO_SUBMIT_MARKER=str(tmp_path / "forbidden"))
    result = subprocess.run(["bash", str(mini / "Libs/exp/submission" / launcher), profile,
                             "--print-only", "--config", str(path)], cwd=mini,
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "forbidden").exists()
    config_root = mini / "Output/paper_configs/print_only" / profile
    sbatch_root = mini / "Output/logs/print_only" / profile
    scripts = [p.read_text() for p in sbatch_root.glob("*.sbatch")]
    assert scripts
    for seed in SEEDS:
        output = config_root / f"{profile}_seed_{seed}.yaml"
        actual = load_config(output)
        assert resolve_training_seed(actual, require_selected=True)["selected_formal_seed"] == seed
        assert any("--config " + str(output.relative_to(mini)) in text for text in scripts)
    assert len(list(config_root.glob("*_seed_*.yaml"))) == 5
    unselected = [load_config(p) for p in config_root.glob("*.yaml") if "_seed_" not in p.name]
    assert len(unselected) == (1 if launcher == "submit_dynamics_profile_suite.sh" else 0)
    assert all("selected_formal_seed" not in payload for payload in unselected)


@pytest.mark.parametrize("values", [[41,42,43], [40,41,42], [41,42,44], [43,42,41], [41,42,42], [41,42,True], [41,42,"43"]])
def test_active_three_seed_policy_exact_integer_order_and_checkpoint(base, values):
    config = copy.deepcopy(base)
    config["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"] = "manuscript_native_three"
    config["paper_grade_full_suite"]["formal_seed_values"] = values
    config["paper_grade_full_suite"]["seeds"] = values
    config["paper_grade_baseline_registry"]["seed_values"] = values
    config = selected(config, 41)
    if values != [41,42,43] or any(type(value) is not int for value in values):
        with pytest.raises(ValueError, match="exact ordered integer"):
            resolve_training_seed(config, require_selected=True)
        return
    binding = resolve_training_seed(config, require_selected=True)
    assert binding["allowed_training_seeds"] == [41,42,43]
    validate_training_seed_checkpoint({"training_seed_binding": binding}, binding)
    historical = resolve_training_seed(selected(base, 41), require_selected=True)
    with pytest.raises(ValueError, match="binding mismatch"):
        validate_training_seed_checkpoint({"training_seed_binding": historical}, binding)
    for seed in (40,44,True):
        with pytest.raises(ValueError, match="training seed"):
            resolve_training_seed(selected(config, seed), require_selected=True)


def test_active_three_seed_real_config_materialization_preserves_other_rng(base, tmp_path):
    config = copy.deepcopy(base)
    config["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"] = "manuscript_native_three"
    config["paper_grade_full_suite"].update(formal_seed_values=[41,42,43], seeds=[41,42,43])
    config["paper_grade_baseline_registry"]["seed_values"] = [41,42,43]
    for seed in (41,42,43):
        actual, _ = cli_materialize(config, tmp_path / str(seed), PROFILES[0], seed)
        assert resolve_training_seed(actual, require_selected=True)["model_seed"] == seed
        assert actual["screening_pipeline"]["screening_benchmark"]["formal_seed_values"] == [3407,42,26]

@pytest.mark.parametrize("seed_policy", ["manuscript_native_three", "legacy_formal"])
@pytest.mark.parametrize("seed", [41, 42, 43])
def test_screening_training_seed_policy_before_cache(base, tmp_path, monkeypatch, seed_policy, seed):
    from Libs.exp.training import train_screening_system as module
    from Libs.utils.exp.experiment_tracking import scoped_seed
    config = selected(base, seed)
    config["screening_pipeline"]["screening_benchmark"]["training"]["seed_policy"] = seed_policy
    if seed_policy == "manuscript_native_three":
        config["paper_grade_full_suite"].update(formal_seed_values=[41, 42, 43], seeds=[41, 42, 43])
        config["paper_grade_baseline_registry"]["seed_values"] = [41, 42, 43]
    binding = module.screening_training_seed_binding(config, "folds", "fixture_fold")
    expected = seed if seed_policy == "manuscript_native_three" else scoped_seed(seed, "folds", "fixture_fold", "ad_screening_system")
    assert binding["seed_policy"] == seed_policy
    assert binding["selected_formal_seed"] == binding["base_training_seed"] == seed
    assert binding["model_seed"] == expected
    assert binding["scope"] == ["folds", "fixture_fold", "ad_screening_system"]
    assert binding["allowed_training_seeds"] == ([41, 42, 43] if seed_policy == "manuscript_native_three" else [3407, 42, 26])
    monkeypatch.setattr(module, "artifact_dir", lambda *args: tmp_path)
    monkeypatch.setattr(module, "get_pipeline_logger", lambda **kwargs: None)
    monkeypatch.setattr(module, "cache_enabled", lambda config: True)
    calls = []
    monkeypatch.setattr(module, "seed_everything", calls.append)
    class StopBeforeCache(Exception): pass
    def stop(*args, **kwargs):
        assert args[-1] == expected
        raise StopBeforeCache
    monkeypatch.setattr(module, "load_scope_training_cache", stop)
    with pytest.raises(StopBeforeCache):
        module.train_one_scope(**scope_arguments(module, config, tmp_path))
    assert calls == [expected]
