from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from Libs.exp.evaluation.evaluate_rollout_prediction import parse_args, resolve_native_identity
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.exp.experiment_manifests import deep_merge, load_experiment_family
from Libs.utils.exp.paper_grade_entrypoint import run_identifier
from Libs.utils.exp_utils import resolve_output_dir

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "Libs/exp/evaluation/evaluate_rollout_prediction.py"
PROFILE_MODELS = [
    ("dynamics_mainline_ad", "neuronsifter_mainline"),
    ("action_conditioning_target_identity", "action_target_only_reference"),
    ("rollout_operator_reversed_experts", "neuronsifter_reversed_experts"),
    ("rollout_operator_interleaved_experts", "neuronsifter_interleaved_experts"),
]


@pytest.fixture(scope="module")
def canonical_config():
    return load_config("Libs/configs/config_stack.yaml")


def metadata_config(canonical, directory, profile):
    directory.mkdir(parents=True, exist_ok=True)
    family = load_experiment_family(profile, canonical)
    config = deep_merge(canonical, family.get("config_overrides", {}))
    config["selected_experiment_family"] = profile
    config["selected_formal_seed"] = 41
    config["screening_pipeline"]["screening_benchmark"]["training"].update(
        seed_policy="manuscript_native_five", random_seed=41)
    config["logging_registry"].update(file_enabled=False, console_enabled=False,
                                      root_dir=str(directory / "logs"))
    config["ai_ready_preprocessing"]["simulation_audit"]["local_root"] = str(directory / "absent-qc")
    registry = directory / "metadata-only-registry.json"
    registry.write_text(json.dumps({"fixture": "metadata_only_no_native_payload"}))
    config["native_diffusion_data"]["registry_path"] = str(registry)
    entry = config["paper_grade_full_suite"]["entrypoint_registry"]["rollout_prediction"]
    # Preserve the real default template and its placeholders inside disposable storage.
    entry["output_template"] = str(directory / entry["output_template"])
    return config


def invoke(config, directory, extra=()):
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    argv = ["--config", str(config_path), "--freeze-manifest", str(directory / "absent-freeze.json"), *extra]
    completed = subprocess.run([sys.executable, str(CLI), *argv],
                               cwd=ROOT, text=True, capture_output=True)
    return completed, argv


def assert_written_identity(directory, model, profile):
    output = directory / "Output/results/paper_grade_full_suite/manifests/rollout_prediction" / model / "seed_41"
    metadata = json.loads((output / "run_metadata.json").read_text())
    status = json.loads((output / "status.json").read_text())
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    assert metadata["model_family"] == status["model_family"] == manifest["model_key"] == model
    assert metadata["experiment_family"] == profile
    assert metadata["run_id"] == f"rollout_prediction_{model}_seed_41"
    assert metadata["seed"] == status["seed"] == manifest["seed"] == 41
    assert status["status"] == manifest["postrun_status"] == "dry_run"
    return output


def test_native_defaults_write_distinct_profile_namespaces(canonical_config, tmp_path):
    outputs = []
    for profile, model in PROFILE_MODELS:
        config = metadata_config(canonical_config, tmp_path, profile)
        result, argv = invoke(config, tmp_path, ["--execution-mode", "native", "--dry-run"])
        assert result.returncode == 0, result.stdout + result.stderr
        args = parse_args(argv)
        assert args.model_family is None
        resolve_native_identity(args, config)
        output = assert_written_identity(tmp_path, model, profile)
        assert resolve_output_dir(args, config) == output
        assert run_identifier(args, 41) == f"rollout_prediction_{model}_seed_41"
        outputs.append(output)
    assert len(set(outputs)) == len(PROFILE_MODELS)


@pytest.mark.parametrize("profile,wrong", [
    ("rollout_operator_reversed_experts", "neuronsifter_mainline"),
    ("action_conditioning_target_identity", "neuronsifter_mainline"),
    ("rollout_operator_interleaved_experts", "transolver_microenvironment_operator"),
])
def test_explicit_wrong_registered_native_label_rejects_before_inference_or_output(
    canonical_config, tmp_path, profile, wrong
):
    config = metadata_config(canonical_config, tmp_path, profile)
    assert wrong in config["paper_grade_full_suite"]["models"]
    # No dry-run: even an absent checkpoint and invalid native metadata must not
    # be consumed before the conflicting registered label is rejected.
    result, argv = invoke(config, tmp_path, ["--execution-mode", "native",
        "--model-family", wrong, "--checkpoint", str(tmp_path / "absent.pt")])
    assert parse_args(argv).model_family == wrong
    assert result.returncode != 0
    assert "Native model label" in result.stderr and "conflicts with profile" in result.stderr
    assert not (tmp_path / "Output").exists()


def test_explicit_matching_native_label_is_preserved(canonical_config, tmp_path):
    profile, model = PROFILE_MODELS[-1]
    config = metadata_config(canonical_config, tmp_path, profile)
    result, _ = invoke(config, tmp_path, ["--execution-mode", "native", "--dry-run",
        "--model-family", model, "--experiment-family", profile])
    assert result.returncode == 0, result.stdout + result.stderr
    assert_written_identity(tmp_path, model, profile)


def test_conflicting_experiment_label_rejects_before_output(canonical_config, tmp_path):
    config = metadata_config(canonical_config, tmp_path, PROFILE_MODELS[-1][0])
    result, _ = invoke(config, tmp_path, ["--execution-mode", "native", "--dry-run",
        "--experiment-family", PROFILE_MODELS[0][0]])
    assert result.returncode != 0
    assert "experiment label differs" in result.stderr
    assert not (tmp_path / "Output").exists()


@pytest.mark.parametrize("mutation", ["missing", "ambiguous"])
def test_native_identity_requires_unambiguous_existing_mapping(canonical_config, mutation):
    config = copy.deepcopy(canonical_config)
    profile = PROFILE_MODELS[-1][0]
    config["selected_experiment_family"] = profile
    rows = config["paper_grade_baseline_registry"]["baselines"]
    if mutation == "missing":
        config["paper_grade_baseline_registry"]["baselines"] = [
            row for row in rows if row.get("profile_name") != profile]
    else:
        config["paper_grade_full_suite"]["models"]["transolver_microenvironment_operator"]["profile_name"] = profile
    with pytest.raises(ValueError, match="one unambiguous existing"):
        resolve_native_identity(parse_args(["--execution-mode", "native"]), config)


@pytest.mark.parametrize("label", [None, "transolver_microenvironment_operator"])
def test_catalog_parser_and_writer_preserve_existing_labels(canonical_config, tmp_path, label):
    config = metadata_config(canonical_config, tmp_path, PROFILE_MODELS[-1][0])
    extra = ["--dry-run"] + ([] if label is None else ["--model-family", label])
    result, argv = invoke(config, tmp_path, extra)
    assert result.returncode == 0, result.stdout + result.stderr
    args = parse_args(argv)
    assert args.execution_mode == "catalog"
    expected = label or "neuronsifter_mainline"
    assert args.model_family == expected
    assert_written_identity(tmp_path, expected, "")



def test_target_reference_catalog_metadata_preserves_native_identity(canonical_config, monkeypatch):
    from types import SimpleNamespace
    from Libs.scripts.validators.validate_paper_grade_full_suite import baseline_schema_failures
    from Libs.utils.exp import paper_grade_entrypoint as entrypoint
    profile, identity = "action_conditioning_target_identity", "action_target_only_reference"
    row = next(row for row in canonical_config["paper_grade_baseline_registry"]["baselines"] if row["baseline_id"] == identity)
    assert row["reference_only"] is True
    assert row["main_table_runnable"] is row["appendix_runnable"] is False
    assert row["profile_name"] == profile and "pending" in row["reason_if_not_runnable"]
    failures = baseline_schema_failures(canonical_config)
    assert failures == []
    broken = copy.deepcopy(canonical_config)
    next(row for row in broken["paper_grade_baseline_registry"]["baselines"] if row["baseline_id"] == identity)["reference_only"] = False
    assert set(baseline_schema_failures(broken)) == set(failures) | {identity + ":non_reference_baseline_has_no_table"}
    print("BASELINE_SCHEMA_FAILURES", sorted(failures))
    config = deep_merge(canonical_config, load_experiment_family(profile, canonical_config)["config_overrides"])
    config["selected_experiment_family"] = profile
    args = SimpleNamespace(model_family=None, experiment_family=profile, dry_run=False)
    resolve_native_identity(args, config)
    assert args.model_family == identity
    emitted = []
    monkeypatch.setattr(entrypoint, "load_effective_config", lambda args: config)
    monkeypatch.setattr(entrypoint, "catalog_metric_payload", lambda *args: pytest.fail("Reference-only catalog entry read metrics"))
    def write(args, config, metrics, sources, status, **kwargs):
        emitted.append((args.model_family, metrics, sources, status, kwargs))
        return {}
    monkeypatch.setattr(entrypoint, "write_entrypoint_outputs", write)
    entrypoint.run_catalog_entrypoint("target metadata compatibility", "rollout_prediction", args=args)
    assert emitted == [(identity, {}, [], "reference_only", {"reference_only": True})]


# Catalog evidence classification must not disable these native comparisons.
CATALOG_NATIVE_COMPARISONS = [
    ("action_conditioning_scalar_exposure", "action_scalar_pk_exposure_reference", "native_diffusion"),
    ("dynamics_mainline_ad", "native_initial_state_persistence", "initial_state_persistence"),
    ("rollout_operator_parallel_experts", "neuronsifter_parallel_experts", "native_diffusion"),
    ("rollout_operator_reversed_experts", "neuronsifter_reversed_experts", "native_diffusion"),
    ("rollout_operator_interleaved_experts", "neuronsifter_interleaved_experts", "native_diffusion"),
]


def comparison_config(canonical, directory, profile, predictor):
    config = metadata_config(canonical, directory, profile)
    config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] = predictor
    return config


def fixture_argv(config, directory, extra):
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return [str(CLI), "--config", str(config_path),
            "--freeze-manifest", str(directory / "absent-freeze.json"), *extra]


@pytest.mark.parametrize("profile,model,predictor", CATALOG_NATIVE_COMPARISONS)
def test_reference_catalog_real_writer_skips_collection_and_removes_stale_metrics(
    canonical_config, tmp_path, monkeypatch, profile, model, predictor
):
    from Libs.exp.evaluation import evaluate_rollout_prediction as rollout
    from Libs.utils.exp import paper_grade_entrypoint as entrypoint

    config = comparison_config(canonical_config, tmp_path, profile, predictor)
    argv = fixture_argv(config, tmp_path, ["--model-family", model])
    output = resolve_output_dir(parse_args(argv[1:]), config)
    assert output.is_relative_to(tmp_path)
    output.mkdir(parents=True)
    # Only task-owned disposable output is replaced; no existing result is read.
    (output / "metrics.json").write_text(json.dumps({"fixture": "stale_unaccepted_metrics"}))
    monkeypatch.setattr(entrypoint, "catalog_metric_payload",
                        lambda *args: pytest.fail("Reference-only catalog collected metrics or sources"))
    monkeypatch.setattr(sys, "argv", argv)
    rollout.main()

    status = json.loads((output / "status.json").read_text())
    metadata = json.loads((output / "run_metadata.json").read_text())
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    assert status == json.loads((output / "reference_audit_status.json").read_text())
    assert status["status"] == "reference_only"
    assert status["metrics_materialized"] is False
    assert model in status["reason"]
    assert metadata["model_family"] == manifest["model_key"] == model
    assert metadata["reference_only"] is True
    assert metadata["postrun_status"] == manifest["postrun_status"] == "reference_only"
    assert metadata["source_artifacts"] == []
    assert metadata["seed"] == manifest["seed"] == 41
    assert metadata["run_id"] == f"rollout_prediction_{model}_seed_41"
    assert manifest["metrics_json"] == hashes["metrics_hash"] == ""
    assert hashes["source_hashes"] == {}
    assert not (output / "metrics.json").exists()


@pytest.mark.parametrize("profile,model,predictor", CATALOG_NATIVE_COMPARISONS)
def test_reference_classification_preserves_explicit_native_dispatch_and_namespaces(
    canonical_config, tmp_path, monkeypatch, profile, model, predictor
):
    from Libs.exp.evaluation import evaluate_rollout_prediction as rollout
    from Libs.exp.evaluation import evaluate_screening_system as evaluator

    config = comparison_config(canonical_config, tmp_path, profile, predictor)
    checkpoint = tmp_path / "routing-only.checkpoint"
    checkpoint.write_text("Disposable routing fixture; no trained model or numerical evidence.")
    calls = []

    def routing_only(effective, selected_checkpoint):
        assert effective["selected_experiment_family"] == profile
        assert effective["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] == predictor
        assert selected_checkpoint == checkpoint
        calls.append((profile, predictor))
        # Exercise the real writer; this is explicitly not scientific evaluation.
        return {"status": "complete", "metrics": {}, "source_status": "routing_fixture_only"}, []

    monkeypatch.setattr(evaluator, "evaluate_native_rollout_prediction", routing_only)
    monkeypatch.setattr(rollout, "run_catalog_entrypoint",
                        lambda *args, **kwargs: pytest.fail("Explicit native mode entered catalog dispatch"))
    default_output = tmp_path / "Output/results/paper_grade_full_suite/manifests/rollout_prediction" / model / "seed_41"
    default_id = f"rollout_prediction_{model}_seed_41"
    if predictor == "initial_state_persistence":
        default_output /= profile
        default_id += "_" + profile
    for explicit in (False, True):
        output = tmp_path / "explicit-output" if explicit else default_output
        run_id = "explicit-native-routing" if explicit else default_id
        extra = ["--execution-mode", "native", "--checkpoint", str(checkpoint)]
        if explicit:
            extra += ["--model-family", model, "--experiment-family", profile,
                      "--output-dir", str(output), "--run-id", run_id]
        monkeypatch.setattr(sys, "argv", fixture_argv(config, tmp_path, extra))
        rollout.main()
        metadata = json.loads((output / "run_metadata.json").read_text())
        manifest = json.loads((output / "artifact_manifest.json").read_text())
        status = json.loads((output / "status.json").read_text())
        payload = json.loads((output / "metrics.json").read_text())
        assert metadata["model_family"] == manifest["model_key"] == status["model_family"] == model
        assert metadata["experiment_family"] == profile
        assert metadata["run_id"] == run_id
        assert metadata["seed"] == manifest["seed"] == status["seed"] == 41
        assert metadata["reference_only"] is status["reference_only"] is False
        assert metadata["postrun_status"] == manifest["postrun_status"] == status["status"] == "materialized"
        assert metadata["source_artifacts"] == []
        assert payload == {"status": "complete", "metrics": {}, "source_status": "routing_fixture_only"}
        assert manifest["metrics_json"] == str(output / "metrics.json")
        assert manifest["checkpoint_hash"]
    assert calls == [(profile, predictor), (profile, predictor)]


@pytest.mark.parametrize("profile,model,predictor", CATALOG_NATIVE_COMPARISONS)
@pytest.mark.parametrize("rejection", ["missing_checkpoint", "conflicting_label", "unsupported_predictor"])
def test_reference_native_rejects_invalid_request_before_evaluator_or_output(
    canonical_config, tmp_path, monkeypatch, profile, model, predictor, rejection
):
    from Libs.exp.evaluation import evaluate_rollout_prediction as rollout
    from Libs.exp.evaluation import evaluate_screening_system as evaluator

    config = comparison_config(canonical_config, tmp_path, profile, predictor)
    extra = ["--execution-mode", "native"]
    if rejection == "missing_checkpoint":
        message = "requires one explicit training-seed checkpoint"
    else:
        extra += ["--checkpoint", str(tmp_path / "absent.checkpoint")]
        if rejection == "conflicting_label":
            extra += ["--model-family", "transolver_microenvironment_operator"]
            message = "Native model label.*conflicts with profile"
        else:
            config["paper_grade_full_suite"]["native_rollout_evaluation"]["predictor_kind"] = "unsupported_fixture"
            message = "Unsupported native evaluation predictor_kind"
    monkeypatch.setattr(evaluator, "evaluate_native_rollout_prediction",
                        lambda *args: pytest.fail("Invalid request reached native evaluator"))
    monkeypatch.setattr(sys, "argv", fixture_argv(config, tmp_path, extra))
    with pytest.raises(ValueError, match=message):
        rollout.main()
    assert not (tmp_path / "Output").exists()
