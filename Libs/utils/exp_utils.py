from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

import yaml

from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.data.raw_data_registry import ensure_directory
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = "Libs/configs/config_stack.yaml"
DEFAULT_SUITE_CONFIG_PATH = "Libs/configs/exp/paper_grade_full_suite.yaml"
DEFAULT_FREEZE_MANIFEST_PATH = "Output/audits/formal_experiment_freeze_manifest.json"
DEFAULT_OUTPUT_TEMPLATE = (
    "Output/results/paper_grade_full_suite/manifests/"
    "{task}/{model_family}/seed_{seed}"
)


def resolve_project_root() -> Path:
    """Return the NeuronSifter repository root."""
    return REPO_ROOT


def _artifact_path(path_value: str | Path) -> Path:
    """Resolve a repository-relative artifact path."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, object]:
    """Load a YAML mapping without applying the global config-stack warning policy."""
    yaml_path = _artifact_path(path)
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {yaml_path}")
    return payload


def load_config_stack(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load the repository config stack through the canonical loader."""
    return load_config(config_path)


def load_suite_config(suite_config_path: str | Path = DEFAULT_SUITE_CONFIG_PATH) -> dict[str, object]:
    """Load the paper-grade suite YAML and return its suite mapping when present."""
    payload = load_yaml(suite_config_path)
    suite_payload = payload.get("paper_grade_full_suite", payload)
    if not isinstance(suite_payload, dict):
        raise TypeError(f"Suite config root must be a mapping: {suite_config_path}")
    return suite_payload


def build_standard_experiment_arg_parser(description: str = "Run a paper-grade experiment.") -> argparse.ArgumentParser:
    """Build the standard paper-grade experiment CLI parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--suite-config", default=DEFAULT_SUITE_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--task", default="")
    parser.add_argument("--model-family", default="neuronsifter_mainline")
    parser.add_argument("--experiment-family", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--freeze-manifest", default=DEFAULT_FREEZE_MANIFEST_PATH)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-pending", action="store_true")
    return parser


def parse_standard_experiment_args(
    description: str = "Run a paper-grade experiment.",
    default_task: str = "",
) -> argparse.Namespace:
    """Parse standard experiment arguments and apply a default task when supplied."""
    parser = build_standard_experiment_arg_parser(description)
    parser.set_defaults(task=default_task)
    return parser.parse_args()


def _suite_from_config(config: Mapping[str, object], suite_config: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return the suite mapping from an explicit suite config or config stack."""
    if suite_config:
        return dict(suite_config)
    suite = config.get("paper_grade_full_suite", {})
    if not isinstance(suite, dict) or not suite:
        raise KeyError("Missing paper_grade_full_suite configuration.")
    return dict(suite)


def _suite_seeds(suite_config: Mapping[str, object]) -> list[int]:
    """Return YAML-declared paper-grade suite seeds."""
    seed_values = suite_config.get("seeds", suite_config.get("formal_seed_values", []))
    return [int(value) for value in seed_values]


def _resolved_seed(args: argparse.Namespace, suite_config: Mapping[str, object], config: Mapping[str, object] | None = None) -> int:
    """Resolve selected prospective runs and preserve legacy suite defaults."""
    binding = resolve_training_seed(dict(config or {}), args.seed, require_selected=True,
                                    legacy_seed_values=_suite_seeds(suite_config))
    if binding["base_training_seed"] is None:
        raise ValueError("Suite seed list is empty.")
    return int(binding["base_training_seed"])


def _run_id(args: argparse.Namespace, seed: int) -> str:
    """Return an explicit run id or a deterministic task/model/seed id."""
    if str(args.run_id).strip():
        return str(args.run_id)
    return f"{args.task}_{args.model_family}_seed_{seed}"


def resolve_output_dir(
    args: argparse.Namespace,
    config: Mapping[str, object],
    suite_config: Mapping[str, object] | None = None,
) -> Path:
    """Resolve the run output directory from CLI, suite registry, or default template."""
    suite = _suite_from_config(config, suite_config)
    seed = _resolved_seed(args, suite, config)
    if str(args.output_dir).strip():
        return _artifact_path(str(args.output_dir))
    registry = dict(suite.get("entrypoint_registry", {}))
    entry = dict(registry.get(str(args.task), {}))
    template = str(entry.get("output_template", DEFAULT_OUTPUT_TEMPLATE))
    experiment_family = str(args.experiment_family or args.model_family)
    return _artifact_path(
        template.format(
            task=str(args.task),
            task_name=str(args.task),
            model_family=str(args.model_family),
            experiment_family=experiment_family,
            seed=seed,
            run_id=_run_id(args, seed),
        )
    )


def make_run_dir(
    task: str,
    model_family: str,
    experiment_family: str,
    seed: int,
    run_id: str,
) -> Path:
    """Create and return the default task/model/seed run directory."""
    run_dir = _artifact_path(DEFAULT_OUTPUT_TEMPLATE.format(task=task, model_family=model_family, seed=seed))
    if experiment_family:
        run_dir = run_dir / experiment_family
    if run_id:
        run_dir = run_dir / run_id
    ensure_directory(run_dir)
    return run_dir


def load_freeze_manifest(path: str | Path = DEFAULT_FREEZE_MANIFEST_PATH) -> dict[str, object]:
    """Load the formal freeze manifest."""
    manifest_path = _artifact_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Freeze manifest must be a JSON object: {manifest_path}")
    return payload


def get_freeze_snapshot_signature(path: str | Path = DEFAULT_FREEZE_MANIFEST_PATH) -> dict[str, object]:
    """Return the freeze snapshot signature from a manifest path."""
    manifest = load_freeze_manifest(path)
    signature = manifest.get("snapshot_signature", {})
    if not isinstance(signature, dict):
        raise TypeError("Freeze manifest snapshot_signature must be a mapping.")
    return dict(signature)


def _command_stdout(args: Sequence[str]) -> str:
    """Run a repository-local command and return stdout."""
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    return completed.stdout


def get_git_head() -> str:
    """Return the current git HEAD SHA."""
    return _command_stdout(["git", "rev-parse", "HEAD"]).strip()


def get_git_dirty_file_list() -> list[str]:
    """Return dirty worktree entries from git porcelain output."""
    output = _command_stdout(["git", "status", "--porcelain=v1", "-uall"])
    return [line for line in output.splitlines() if line.strip()]


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA256 for an existing file."""
    file_path = _artifact_path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_json_sha256(obj: object) -> str:
    """Compute stable SHA256 for a JSON-serializable object."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_git_full_diff_hash() -> str:
    """Compute a deterministic hash over tracked diffs and untracked file content."""
    diff_payload = _command_stdout(["git", "diff", "--binary", "HEAD", "--"])
    status_entries = get_git_dirty_file_list()
    untracked_hashes: list[dict[str, str]] = []
    for entry in status_entries:
        if not entry.startswith("?? "):
            continue
        path = _artifact_path(entry[3:])
        if path.is_file():
            untracked_hashes.append({"path": entry[3:], "sha256": compute_file_sha256(path)})
    return compute_json_sha256(
        {
            "tracked_diff": diff_payload,
            "status_entries": status_entries,
            "untracked_file_hashes": sorted(untracked_hashes, key=lambda row: row["path"]),
        }
    )


def _torch_version() -> str:
    """Return the installed torch version, or an empty string when unavailable."""
    if importlib.util.find_spec("torch") is None:
        return ""
    return importlib.metadata.version("torch")


def _device_label() -> str:
    """Return cuda when torch can see CUDA, otherwise cpu."""
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    torch_module = importlib.import_module("torch")
    return "cuda" if bool(torch_module.cuda.is_available()) else "cpu"


def _data_manifest_hash(config: Mapping[str, object]) -> str:
    """Return the simulation QC catalog hash or a model-input registry hash."""
    audit_config = dict(dict(config.get("ai_ready_preprocessing", {})).get("simulation_audit", {}))
    local_root = str(audit_config.get("local_root", "")).strip()
    relative_paths = dict(audit_config.get("relative_paths", {}))
    qc_catalog = str(relative_paths.get("qc_catalog", "")).strip()
    if local_root and qc_catalog:
        candidate = _artifact_path(local_root) / qc_catalog
        if candidate.exists() and candidate.is_file():
            return compute_file_sha256(candidate)
    return compute_json_sha256(config.get("model_input_registry", {}))


def build_run_metadata(
    args: argparse.Namespace,
    config: Mapping[str, object],
    suite_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build standard run metadata for a paper-grade task."""
    suite = _suite_from_config(config, suite_config)
    seed = _resolved_seed(args, suite, config)
    timestamp = datetime.now(timezone.utc).isoformat()
    split_payload = {
        "task": str(args.task),
        "model_family": str(args.model_family),
        "experiment_family": str(args.experiment_family),
        "seed": seed,
        "holdouts": [
            dict(task).get("required_holdouts", [])
            for task in suite.get("tasks", [])
            if str(dict(task).get("task_name", "")) == str(args.task)
        ],
    }
    freeze_signature = {}
    freeze_path = str(args.freeze_manifest)
    if freeze_path.strip() and _artifact_path(freeze_path).exists():
        freeze_signature = get_freeze_snapshot_signature(freeze_path)
    return {
        "seed": seed,
        "training_seed_binding": resolve_training_seed(dict(config), args.seed, require_selected=True,
            legacy_seed_values=_suite_seeds(suite)) if config.get("screening_pipeline", {}).get("screening_benchmark", {}).get("training", {}).get("seed_policy") == "manuscript_native_five" else None,
        "task": str(args.task),
        "task_name": str(args.task),
        "model_family": str(args.model_family),
        "experiment_family": str(args.experiment_family),
        "config_path": str(args.config),
        "suite_config_path": str(args.suite_config),
        "freeze_manifest": freeze_path,
        "checkpoint": str(getattr(args, "checkpoint", "")),
        "run_id": _run_id(args, seed),
        "dry_run": bool(args.dry_run),
        "git_head": get_git_head(),
        "git_full_diff_hash": compute_git_full_diff_hash(),
        "resolved_config_sha256": compute_json_sha256(config),
        "freeze_snapshot_signature": freeze_signature,
        "data_manifest_hash": (compute_file_sha256(_artifact_path(config["native_diffusion_data"]["registry_path"]))
                               if getattr(args, "execution_mode", None) == "native" else _data_manifest_hash(config)),
        "split_hash": compute_json_sha256(split_payload),
        "start_time": timestamp,
        "end_time": timestamp,
        "hostname": socket.gethostname(),
        "device": _device_label(),
        "python_version": platform.python_version(),
        "torch_version": _torch_version(),
    }


def write_run_metadata(path: str | Path, metadata: Mapping[str, object]) -> None:
    """Write run metadata JSON."""
    write_json(_artifact_path(path), dict(metadata))


def write_status(path: str | Path, status: Mapping[str, object]) -> None:
    """Write a standard status JSON."""
    write_json(_artifact_path(path), dict(status))


def write_metrics(path: str | Path, metrics: Mapping[str, object]) -> None:
    """Write metrics JSON."""
    write_json(_artifact_path(path), dict(metrics))


def write_artifact_hashes(path: str | Path, artifact_paths: Iterable[str | Path]) -> None:
    """Write SHA256 hashes for existing artifact paths."""
    hashes = {}
    for artifact_path in artifact_paths:
        path_obj = _artifact_path(artifact_path)
        if path_obj.exists() and path_obj.is_file():
            hashes[path_obj.relative_to(REPO_ROOT).as_posix()] = compute_file_sha256(path_obj)
    write_json(_artifact_path(path), hashes)


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Python and torch when available."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if importlib.util.find_spec("torch") is not None:
        torch_module = importlib.import_module("torch")
        torch_module.manual_seed(seed)
        if bool(torch_module.cuda.is_available()):
            torch_module.cuda.manual_seed_all(seed)


def validate_seed(seed: int, allowed_seeds: Iterable[int]) -> None:
    """Validate that a seed belongs to the YAML-declared seed set."""
    allowed = [int(value) for value in allowed_seeds]
    if int(seed) not in allowed:
        raise ValueError(f"Seed {seed} is not in allowed suite seeds {allowed}.")


def validate_task_block(task: str, suite_config: Mapping[str, object]) -> None:
    """Validate that a task exists in suite task blocks and registry."""
    task_names = {str(dict(row).get("task_name", dict(row).get("task_id", ""))) for row in suite_config.get("tasks", [])}
    registry_names = set(str(key) for key in dict(suite_config.get("entrypoint_registry", {})))
    if task not in task_names or task not in registry_names:
        raise ValueError(f"Task {task} is not declared in the paper-grade suite.")


def validate_model_family(model_family: str, suite_config: Mapping[str, object]) -> None:
    """Validate that a model family exists in the suite model registry."""
    if model_family not in dict(suite_config.get("models", {})):
        raise ValueError(f"Model family {model_family} is not declared in the suite model registry.")


def validate_experiment_family(experiment_family: str, suite_config: Mapping[str, object]) -> None:
    """Validate that an experiment family is declared when it is not empty."""
    if not str(experiment_family).strip():
        return
    family_names = {str(value) for value in suite_config.get("experiment_families", [])}
    model_profiles = {
        str(dict(value).get("profile_name", ""))
        for value in dict(suite_config.get("models", {})).values()
    }
    if experiment_family not in family_names and experiment_family not in model_profiles:
        raise ValueError(f"Experiment family {experiment_family} is not declared in the suite.")


def compute_mean_std_ci(values: Sequence[float]) -> dict[str, object]:
    """Compute mean, sample std, normal-approximation 95% CI, and order statistics."""
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return {"count": 0, "values": [], "mean": None, "std": None, "ci95": None}
    mean_value = statistics.fmean(numeric_values)
    std_value = statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0
    ci95 = 1.96 * std_value / math.sqrt(len(numeric_values)) if numeric_values else 0.0
    sorted_values = sorted(numeric_values)
    return {
        "count": len(numeric_values),
        "values": numeric_values,
        "mean": mean_value,
        "std": std_value,
        "ci95": ci95,
        "ci95_low": mean_value - ci95,
        "ci95_high": mean_value + ci95,
        "best": max(numeric_values),
        "median": statistics.median(sorted_values),
        "worst": min(numeric_values),
    }


def _flatten_numeric_metrics(payload: Mapping[str, object], prefix: str = "") -> dict[str, float]:
    """Flatten nested numeric metric values into dotted keys."""
    rows: dict[str, float] = {}
    for key, value in payload.items():
        metric_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            rows[metric_key] = float(value)
            continue
        if isinstance(value, dict):
            rows.update(_flatten_numeric_metrics(value, metric_key))
    return rows


def aggregate_seed_metrics(metric_files: Iterable[str | Path]) -> dict[str, object]:
    """Aggregate numeric metrics across seed-level metrics JSON files."""
    grouped: dict[str, list[float]] = {}
    per_seed_values: dict[str, dict[str, float]] = {}
    for metric_file in metric_files:
        path = _artifact_path(metric_file)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Metric file must contain a JSON object: {path}")
        seed = str(payload.get("seed", path.parent.name))
        flat_metrics = _flatten_numeric_metrics(payload.get("metrics", payload))
        per_seed_values[seed] = flat_metrics
        for metric_name, value in flat_metrics.items():
            grouped.setdefault(metric_name, []).append(float(value))
    return {
        "metric_count": len(grouped),
        "seed_count": len(per_seed_values),
        "per_seed_values": per_seed_values,
        "metrics": {metric_name: compute_mean_std_ci(values) for metric_name, values in sorted(grouped.items())},
    }


def write_aggregate_metrics(path: str | Path, aggregate: Mapping[str, object]) -> None:
    """Write aggregate metrics JSON."""
    write_json(_artifact_path(path), dict(aggregate))


def scan_text_for_forbidden_claims(text: str, guardrails: Mapping[str, object]) -> list[str]:
    """Return forbidden claim fragments present in text."""
    fragments: list[str] = []
    rules = guardrails.get("forbidden_claims", guardrails.get("claim_prohibition_rules", []))
    for rule in rules:
        if isinstance(rule, str):
            candidates = [rule]
        else:
            candidates = [str(value) for value in dict(rule).get("forbidden_fragments", [])]
        for candidate in candidates:
            if candidate and candidate.lower() in text.lower():
                fragments.append(candidate)
    return sorted(set(fragments))


def assert_no_forbidden_claims(text: str, guardrails: Mapping[str, object]) -> None:
    """Raise when forbidden public claim fragments are present."""
    violations = scan_text_for_forbidden_claims(text, guardrails)
    if violations:
        raise ValueError(f"Forbidden claim fragments found: {violations}")


def fail_with_status(run_dir: str | Path, message: str, exception: Exception | None = None) -> None:
    """Write failed status in a run directory and raise RuntimeError."""
    output_dir = _artifact_path(run_dir)
    ensure_directory(output_dir)
    status = {
        "status": "failed",
        "message": message,
        "exception_type": type(exception).__name__ if exception is not None else "",
        "exception_message": str(exception) if exception is not None else "",
    }
    write_status(output_dir / "status.json", status)
    raise RuntimeError(message)


def mark_reference_only(run_dir: str | Path, reason: str) -> dict[str, object]:
    """Write reference-only status and audit files without metrics."""
    output_dir = _artifact_path(run_dir)
    ensure_directory(output_dir)
    payload = {
        "status": "reference_only",
        "reason": reason,
        "metrics_materialized": False,
    }
    write_status(output_dir / "status.json", payload)
    write_json(output_dir / "reference_audit_status.json", payload)
    return payload
