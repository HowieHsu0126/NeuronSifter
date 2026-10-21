from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, write_json
from Libs.utils.exp_utils import build_run_metadata
from Libs.utils.exp_utils import mark_reference_only
from Libs.utils.exp_utils import parse_standard_experiment_args
from Libs.utils.exp_utils import resolve_output_dir
from Libs.utils.exp_utils import validate_seed
from Libs.utils.exp_utils import write_status


REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def artifact_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_standard_args(description: str, default_task: str):
    return parse_standard_experiment_args(description, default_task)


def effective_config_path(args: argparse.Namespace) -> str:
    return str(args.config)


def load_effective_config(args: argparse.Namespace) -> dict[str, object]:
    return load_config(args.config)


def suite_config(config: dict[str, object]) -> dict[str, object]:
    payload = config.get("paper_grade_full_suite", {})
    if not isinstance(payload, dict) or not payload:
        raise KeyError("Missing paper_grade_full_suite config.")
    return payload


def first_suite_seed(config: dict[str, object]) -> int:
    seeds = [int(value) for value in suite_config(config)["formal_seed_values"]]
    if not seeds:
        raise ValueError("paper_grade_full_suite.formal_seed_values is empty.")
    return seeds[0]


def resolved_seed(args: argparse.Namespace, config: dict[str, object]) -> int:
    binding = resolve_training_seed(config, args.seed,
        require_selected=True,
        legacy_seed_values=suite_config(config)["formal_seed_values"])
    return int(binding["base_training_seed"])


def run_identifier(args: argparse.Namespace, seed: int) -> str:
    if str(args.run_id).strip():
        return str(args.run_id)
    return f"{args.task}_{args.model_family}_seed_{seed}"


def default_output_dir(args: argparse.Namespace, config: dict[str, object], seed: int) -> Path:
    return resolve_output_dir(args, config, suite_config(config))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: object) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {repo_relative(path): file_sha256(path) for path in paths if path.exists() and path.is_file()}


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def model_catalog_path(config: dict[str, object], model_family: str, seed: int) -> Path | None:
    models = dict(suite_config(config).get("models", {}))
    if model_family not in models:
        return None
    payload = dict(models[model_family])
    template = payload.get("result_catalog_template", "")
    if not str(template).strip():
        return None
    return artifact_path(str(template).format(seed=seed))


def summarize_catalog_metrics(rows: list[dict[str, str]], metric_names: set[str] | None = None) -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        metric_name = str(row.get("metric_name", ""))
        if metric_names is not None and metric_name not in metric_names:
            continue
        if not metric_name:
            continue
        grouped.setdefault(metric_name, []).append(float(row["metric_value"]))
    return {
        name: {
            "values": values,
            "mean": float(sum(values) / len(values)),
            "count": len(values),
        }
        for name, values in sorted(grouped.items())
        if values
    }


def catalog_metric_payload(
    args: argparse.Namespace,
    config: dict[str, object],
    metric_names: Iterable[str] | None = None,
) -> tuple[dict[str, object], list[Path], str]:
    seed = resolved_seed(args, config)
    catalog_path = model_catalog_path(config, str(args.model_family), seed)
    if catalog_path is None:
        return {"metrics": {}, "source_status": "no_catalog_template"}, [], "pending"
    if not catalog_path.exists():
        if args.allow_pending or args.dry_run:
            return {
                "metrics": {},
                "source_status": "catalog_missing",
                "missing_catalog": repo_relative(catalog_path),
            }, [catalog_path], "pending"
        raise FileNotFoundError(f"Result catalog is missing: {catalog_path}")
    metrics = summarize_catalog_metrics(read_tsv_rows(catalog_path), set(metric_names) if metric_names is not None else None)
    status = "materialized" if metrics else "pending"
    return {
        "metrics": metrics,
        "source_status": "catalog_available",
        "source_catalog": repo_relative(catalog_path),
    }, [catalog_path], status


def reference_only_baseline_ids(config: dict[str, object]) -> set[str]:
    registry = config.get("paper_grade_baseline_registry", {})
    return {
        str(row["baseline_id"])
        for row in registry.get("baselines", [])
        if bool(dict(row).get("reference_only", False))
    }


def write_entrypoint_outputs(
    args: argparse.Namespace,
    config: dict[str, object],
    metrics_payload: dict[str, object],
    source_paths: Iterable[Path],
    postrun_status: str,
    reference_only: bool = False,
) -> dict[str, object]:
    seed = resolved_seed(args, config)
    binding = resolve_training_seed(config, args.seed,
        require_selected=True,
        legacy_seed_values=suite_config(config)["formal_seed_values"])
    validate_seed(seed, binding["allowed_training_seeds"])
    output_dir = default_output_dir(args, config, seed)
    ensure_directory(output_dir)
    metadata = build_run_metadata(args, config, suite_config(config))
    metadata["reference_only"] = reference_only
    metadata["postrun_status"] = postrun_status
    metadata["source_artifacts"] = [repo_relative(path) for path in source_paths if path.exists()]
    run_metadata_path = output_dir / "run_metadata.json"
    write_json(run_metadata_path, metadata)
    metrics_path = output_dir / "metrics.json"
    status_path = output_dir / "status.json"
    status_payload = {
        "task_name": str(args.task),
        "model_family": str(args.model_family),
        "seed": seed,
        "status": postrun_status,
        "dry_run": bool(args.dry_run),
        "reference_only": reference_only,
    }
    if reference_only:
        if metrics_path.exists():
            metrics_path.unlink()
        mark_reference_only(output_dir, f"{args.model_family} is declared reference_only in paper_grade_baseline_registry.")
        metrics_json_value = ""
    else:
        write_json(metrics_path, metrics_payload)
        write_status(status_path, status_payload)
        metrics_json_value = repo_relative(metrics_path)
    hash_payload = {
        "config_hash": file_sha256(artifact_path(effective_config_path(args))),
        "freeze_manifest_hash": file_sha256(artifact_path(args.freeze_manifest)) if artifact_path(args.freeze_manifest).exists() else "",
        "source_hashes": existing_file_hashes(source_paths),
        "run_metadata_hash": file_sha256(run_metadata_path),
        "metrics_hash": file_sha256(metrics_path) if metrics_path.exists() else "",
        "status_hash": file_sha256(status_path) if status_path.exists() else "",
    }
    artifact_hashes_path = output_dir / "artifact_hashes.json"
    write_json(artifact_hashes_path, hash_payload)
    manifest = {
        "task_name": str(args.task),
        "model_key": str(args.model_family),
        "seed": seed,
        "config_hash": hash_payload["config_hash"],
        "split_hash": str(metadata["split_hash"]),
        "checkpoint_hash": file_sha256(artifact_path(args.checkpoint)) if str(args.checkpoint).strip() and artifact_path(args.checkpoint).exists() else "",
        "metrics_json": metrics_json_value,
        "stdout_path": "",
        "stderr_path": "",
        "artifact_hash": file_sha256(artifact_hashes_path),
        "runtime_metadata": repo_relative(run_metadata_path),
        "postrun_status": postrun_status,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "output_dir": repo_relative(output_dir),
        "run_metadata": repo_relative(run_metadata_path),
        "artifact_hashes": repo_relative(artifact_hashes_path),
        "artifact_manifest": repo_relative(manifest_path),
        "metrics_json": metrics_json_value,
        "postrun_status": postrun_status,
    }


def run_catalog_entrypoint(description: str, default_task: str, metric_names: Iterable[str] | None = None,
                           *, args: argparse.Namespace | None = None) -> None:
    args = parse_standard_args(description, default_task) if args is None else args
    config = load_effective_config(args)
    reference_only = str(args.model_family) in reference_only_baseline_ids(config)
    if reference_only:
        result = write_entrypoint_outputs(args, config, {}, [], "reference_only", reference_only=True)
        print(json.dumps(result, sort_keys=True))
        return
    if args.dry_run:
        payload = {"metrics": {}, "source_status": "dry_run"}
        result = write_entrypoint_outputs(args, config, payload, [], "dry_run")
        print(json.dumps(result, sort_keys=True))
        return
    payload, source_paths, status = catalog_metric_payload(args, config, metric_names)
    if status != "materialized" and not args.allow_pending:
        raise SystemExit(f"No materialized metrics for {args.task}:{args.model_family}.")
    result = write_entrypoint_outputs(args, config, payload, source_paths, status)
    print(json.dumps(result, sort_keys=True))


def run_delegate_entrypoint(
    description: str,
    default_task: str,
    delegate_path: str,
    delegate_args: list[str] | None = None,
) -> None:
    args = parse_standard_args(description, default_task)
    config = load_effective_config(args)
    binding = resolve_training_seed(config, args.seed, require_selected=True,
                                    legacy_seed_values=suite_config(config)["formal_seed_values"])
    if args.dry_run:
        result = write_entrypoint_outputs(args, config, {"metrics": {}, "source_status": "dry_run"}, [], "dry_run")
        print(json.dumps(result, sort_keys=True))
        return
    seed = binding["base_training_seed"]
    effective_seed = int(config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"])
    if seed != effective_seed:
        raise ValueError(f"Requested training seed {seed} differs from effective YAML seed {effective_seed}; select the matching experiment config before delegation.")
    python_bin = str(config["execution_environment"]["python_executable"])
    command = [python_bin, "-u", delegate_path, "--config", effective_config_path(args)]
    if str(args.checkpoint).strip():
        if delegate_path.endswith("pretrain_microenvironment_world_model.py"):
            command.extend(["--resume-checkpoint", str(args.checkpoint)])
        else:
            raise SystemExit(f"--checkpoint is not supported by delegate {delegate_path}.")
    command.extend(delegate_args or [])
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    result = write_entrypoint_outputs(args, config, {"metrics": {}, "source_status": "delegated"}, [], "materialized")
    print(json.dumps(result, sort_keys=True))
