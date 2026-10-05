from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from Libs.utils.data.raw_data_registry import write_json


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def log_duration_seconds(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    started_at = parse_timestamp(lines[0].split(" | ", 1)[0])
    ended_at = parse_timestamp(lines[-1].split(" | ", 1)[0])
    if started_at is None or ended_at is None:
        return None
    return float((ended_at - started_at).total_seconds())


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def classify_scale(
    command: str,
    training_history_length: int,
    sample_count: int,
    parameter_count: int,
    duration_seconds: float | None,
) -> str:
    checkpoint_window_command = "--training-window-ordinal" in command and "--training-window-count" in command
    if ("--max-epochs-override" in command or "--fold-name" in command) and not checkpoint_window_command:
        return "interface_smoke"
    if training_history_length <= 3:
        return "interface_smoke"
    if sample_count < 5000 or parameter_count < 10_000_000:
        return "light_scale_formal"
    if duration_seconds is not None and duration_seconds < 600.0:
        return "light_scale_formal"
    return "full_scale_formal"


def baseline_family_for_model(model_name: str) -> str:
    if model_name in {"aligned_gene_logistic_regression", "aligned_metabolite_logistic_regression"}:
        return "classical_linear"
    if model_name in {"shared_latent_dual_encoder_classifier", "tabular_jepa_auxiliary_encoder"}:
        return "internal_lightweight"
    if model_name == "ad_screening_system":
        return "action_conditioned_twin"
    return "unspecified"


def audit_row(
    benchmark_name: str,
    disease_program: str,
    model_name: str,
    artifact_root: Path,
    run_metadata: dict[str, Any],
    training_manifest: dict[str, Any],
    duration_seconds: float | None,
) -> dict[str, Any]:
    command = str(run_metadata.get("command", ""))
    metadata_context = run_metadata.get("extra_metadata", {})
    train_shapes = metadata_context.get("train_shape_by_modality", {})
    if not train_shapes:
        train_shapes = metadata_context.get("train_shape", {})
    sample_count = 0
    if isinstance(train_shapes, dict):
        for shape in train_shapes.values():
            if isinstance(shape, list) and shape:
                sample_count += safe_int(shape[0])
    elif isinstance(train_shapes, list) and train_shapes:
        sample_count = safe_int(train_shapes[0])
    parameter_count = 0
    model_kwargs = training_manifest.get("model_kwargs", {})
    feature_count_by_modality = training_manifest.get("feature_count_by_modality", {})
    if feature_count_by_modality:
        parameter_count = sum(safe_int(value) for value in feature_count_by_modality.values())
    parameter_count = max(parameter_count, safe_int(training_manifest.get("parameter_count", 0)))
    scale_classification = classify_scale(
        command,
        len(training_manifest.get("training_history", [])),
        sample_count,
        parameter_count,
        duration_seconds,
    )
    return {
        "benchmark_name": benchmark_name,
        "disease_program": disease_program,
        "model_name": model_name,
        "baseline_family": baseline_family_for_model(model_name),
        "artifact_root": str(artifact_root),
        "run_metadata_path": str(artifact_root / "run_metadata.json"),
        "training_manifest_path": str(artifact_root / "training_manifest.json"),
        "sample_count": sample_count,
        "parameter_count_proxy": parameter_count,
        "duration_seconds": duration_seconds,
        "scale_classification": scale_classification,
        "command": command,
        "model_kwargs": model_kwargs,
    }


def write_audit_report(output_path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {
        "row_count": len(rows),
        "rows": rows,
    }
    write_json(output_path, payload)
