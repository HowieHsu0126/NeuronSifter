"""Qualify simple C2 baseline adapters against the restricted current-clamp store.

This is a bounded software-readiness execution.  It uses train and validation
rows only, emits no final-holdout predictions, and does not start the formal
baseline challenge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json
from Libs.utils.evaluation.restricted_c2_baselines import (
    fit_linear_state_space,
    flatten_state_grid,
    gru_rollout,
    linear_state_space_rollout,
    mean_squared_error,
    persistence_rollout,
    prediction_sha256,
    train_gru_smoke,
)
from Libs.utils.model.dynamics_store import load_zarr_rows, read_zarr_array_metadata


DEFAULT_CONFIG = "Libs/configs/config_stack.yaml"
DEFAULT_MANIFEST_ROOT = "Input/manifests/c2"
DEFAULT_READINESS = "Output/audits/c2_domain_readiness.json"
DEFAULT_INTEGRITY = "Output/audits/c2/current_clamp/integrity_audit.json"
DEFAULT_OUTPUT = "Output/audits/c2/current_clamp/baseline_qualification.json"
SCHEMA_VERSION = "neuronsifter-restricted-c2-baseline-qualification"
ADAPTER_SOURCE = "Libs/utils/evaluation/restricted_c2_baselines.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path, source_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_binding(path: Path, source_root: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required C2 qualification input is missing: {path}")
    return {
        "path": _relative_or_absolute(path, source_root),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object payload at {path}")
    return payload


def _split_role(value: str) -> str:
    normalized = str(value).strip().lower()
    for role in ("train", "validation", "test"):
        if normalized == role or normalized.endswith(f"_{role}"):
            return role
    raise ValueError(f"Unsupported restricted C2 split role: {value}")


def _simulation_index(artifact_uri: str) -> int:
    prefix, marker, suffix = str(artifact_uri).partition("#simulation=")
    if not prefix or marker != "#simulation=" or not suffix.isdigit():
        raise ValueError(f"Restricted C2 artifact URI has no canonical simulation index: {artifact_uri}")
    return int(suffix)


def _store_path_from_rows(rows: list[dict[str, str]]) -> str:
    stores = {str(row["artifact_uri"]).split("#", 1)[0] for row in rows}
    if len(stores) != 1:
        raise ValueError(f"Restricted C2 qualification requires one trajectory store, found {sorted(stores)}")
    return next(iter(stores))


def _ordered_trajectory_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        index = _simulation_index(row["artifact_uri"])
        if index in by_index:
            raise ValueError(f"Restricted C2 trajectory store maps simulation {index} more than once.")
        by_index[index] = row
    if not by_index:
        raise ValueError("Restricted C2 trajectory manifest is empty.")
    return [by_index[index] for index in sorted(by_index)]


def _validate_manifest_alignment(
    trajectory_rows: list[dict[str, str]],
    action_rows: list[dict[str, str]],
    matched_control_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    metadata_shape: tuple[int, ...],
) -> dict[str, list[dict[str, str]]]:
    if len(metadata_shape) != 5:
        raise ValueError(f"Restricted C2 store rank must be five, got {metadata_shape}.")
    ordered = _ordered_trajectory_rows(trajectory_rows)
    if len(ordered) != metadata_shape[0]:
        raise ValueError(
            "Restricted C2 manifest/store simulation count mismatch: "
            f"{len(ordered)} manifest rows and {metadata_shape[0]} store rows."
        )
    action_by_uid = {str(row["action_uid"]): row for row in action_rows}
    if len(action_by_uid) != len(action_rows):
        raise ValueError("Restricted C2 action catalog contains duplicate action identifiers.")
    split_by_trajectory = {str(row["trajectory_uid"]): row for row in split_rows}
    if len(split_by_trajectory) != len(split_rows):
        raise ValueError("Restricted C2 split manifest contains duplicate trajectory identifiers.")
    control_by_pair = {str(row["pair_uid"]): row for row in matched_control_rows}
    if len(control_by_pair) != len(matched_control_rows):
        raise ValueError("Restricted C2 matched-control manifest contains duplicate pair identifiers.")
    by_role = {"train": [], "validation": [], "test": []}
    for row in ordered:
        action_uid = str(row["action_uid"])
        if action_uid not in action_by_uid:
            raise ValueError(f"Restricted C2 trajectory has no action catalog row: {action_uid}")
        trajectory_uid = str(row["trajectory_uid"])
        split_row = split_by_trajectory.get(trajectory_uid)
        if split_row is None:
            raise ValueError(f"Restricted C2 trajectory has no strict split row: {trajectory_uid}")
        manifest_role = _split_role(row["train_eval_role"])
        split_role = _split_role(split_row["split_role"])
        if manifest_role != split_role:
            raise ValueError(
                f"Restricted C2 trajectory split mismatch for {trajectory_uid}: {manifest_role} != {split_role}."
            )
        if str(split_row["pair_uid"]) != str(row["pair_uid"]):
            raise ValueError(f"Restricted C2 pair mismatch for {trajectory_uid}.")
        control_row = control_by_pair.get(str(row["pair_uid"]))
        if control_row is None:
            raise ValueError(f"Restricted C2 trajectory has no matched-control row: {trajectory_uid}")
        expected_actions = {
            str(control_row["intervention_action_uid"]),
            str(control_row["control_action_uid"]),
        }
        if str(row["action_uid"]) not in expected_actions:
            raise ValueError(f"Restricted C2 action is not bound to its matched-control pair: {trajectory_uid}")
        by_role[manifest_role].append(row)
    if not by_role["train"] or not by_role["validation"] or not by_role["test"]:
        raise ValueError("Restricted C2 qualification requires non-empty train, validation, and final-holdout splits.")
    return by_role


def _action_values(rows: list[dict[str, str]], action_rows: list[dict[str, str]]) -> np.ndarray:
    actions = {str(row["action_uid"]): row for row in action_rows}
    values: list[float] = []
    for row in rows:
        action = actions[str(row["action_uid"])]
        if str(action.get("dose_unit", "")).strip() != "nA":
            raise ValueError(f"Restricted C2 action {row['action_uid']} must use nA amplitude units.")
        values.append(float(action["dose_value"]))
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Restricted C2 action amplitudes contain NaN or Inf.")
    return array


def _adapter_record(
    baseline_uid: str,
    prediction: np.ndarray,
    validation_target: np.ndarray,
    *,
    implementation: str,
    details: dict[str, object],
) -> dict[str, object]:
    return {
        "baseline_uid": baseline_uid,
        "implementation_source": implementation,
        "qualification_status": "adapter_qualified",
        "validation_only": True,
        "prediction_shape": list(prediction.shape),
        "validation_mse": mean_squared_error(prediction, validation_target),
        "validation_prediction_sha256": prediction_sha256(prediction),
        "details": details,
    }


def build(
    *,
    config_path: Path,
    manifest_root: Path,
    readiness_path: Path,
    integrity_path: Path,
    output_path: Path,
    store_path: Path | None,
    source_root: Path,
) -> dict[str, object]:
    config = load_config(config_path)
    settings = config["dynamics_ready_registry"]["neuron_rxd_teacher"]["restricted_c2_baseline_qualification"]
    if settings.get("formal_challenge_started") is not False:
        raise ValueError("Restricted C2 qualification configuration must prohibit formal challenge execution.")
    if settings.get("test_prediction_materialized") is not False:
        raise ValueError("Restricted C2 qualification configuration must prohibit final-holdout prediction materialization.")

    action_path = manifest_root / "action_catalog.tsv"
    trajectory_path = manifest_root / "trajectory_manifest.tsv"
    matched_control_path = manifest_root / "matched_control_manifest.tsv"
    split_path = manifest_root / "strict_split_manifest.tsv"
    action_rows = read_tsv(action_path)
    trajectory_rows = read_tsv(trajectory_path)
    matched_control_rows = read_tsv(matched_control_path)
    split_rows = read_tsv(split_path)
    readiness = _load_json(readiness_path)
    integrity = _load_json(integrity_path)
    if readiness.get("status") != "simulator_grounded_c2_restricted":
        raise ValueError("Restricted C2 readiness scope is not the expected simulator-grounded domain.")
    if readiness.get("baseline_challenge_allowed") is not False:
        raise ValueError("Restricted C2 readiness unexpectedly permits the formal baseline challenge.")
    if integrity.get("status") != "passed" or integrity.get("strict_split_overlap") != 0:
        raise ValueError("Restricted C2 integrity audit is not clean.")

    declared_store = _store_path_from_rows(trajectory_rows)
    actual_store = store_path or _resolve(source_root, declared_store)
    if not actual_store.is_dir():
        raise FileNotFoundError(f"Restricted C2 trajectory store is missing: {actual_store}")
    metadata = read_zarr_array_metadata(actual_store, "trajectories")
    shape = tuple(int(value) for value in metadata["shape"])
    by_role = _validate_manifest_alignment(
        trajectory_rows,
        action_rows,
        matched_control_rows,
        split_rows,
        shape,
    )
    expected_store_hash = readiness.get("producer", {}).get("trajectory_store_sha256")
    if integrity.get("store_hash") != expected_store_hash or not expected_store_hash:
        raise ValueError("Restricted C2 readiness and integrity audits disagree on the trajectory store identity.")

    ordered_by_role = {
        role: sorted(rows, key=lambda row: _simulation_index(row["artifact_uri"]))
        for role, rows in by_role.items()
    }
    train_indices = np.asarray([_simulation_index(row["artifact_uri"]) for row in ordered_by_role["train"]], dtype=np.int64)
    validation_indices = np.asarray([_simulation_index(row["artifact_uri"]) for row in ordered_by_role["validation"]], dtype=np.int64)
    test_indices = np.asarray([_simulation_index(row["artifact_uri"]) for row in ordered_by_role["test"]], dtype=np.int64)
    train_trajectories = load_zarr_rows(actual_store, "trajectories", train_indices)
    validation_trajectories = load_zarr_rows(actual_store, "trajectories", validation_indices)
    train_flat, state_shape = flatten_state_grid(train_trajectories)
    validation_flat, validation_state_shape = flatten_state_grid(validation_trajectories)
    if state_shape != validation_state_shape:
        raise ValueError("Restricted C2 train and validation state-grid shapes differ.")
    train_actions = _action_values(ordered_by_role["train"], action_rows)
    validation_actions = _action_values(ordered_by_role["validation"], action_rows)
    time_count = int(train_flat.shape[1])

    persistence_prediction = persistence_rollout(validation_flat[:, 0, :], time_count)
    linear_settings = settings["linear_state_space"]
    linear_model = fit_linear_state_space(
        train_trajectories,
        train_actions,
        ridge_penalty=float(linear_settings["ridge_penalty"]),
    )
    linear_prediction = linear_state_space_rollout(
        linear_model,
        validation_flat[:, 0, :],
        validation_actions,
        time_count,
    )
    gru_settings = settings["gru"]
    gru_model = train_gru_smoke(
        train_trajectories,
        train_actions,
        hidden_size=int(gru_settings["hidden_size"]),
        epoch_count=int(gru_settings["epoch_count"]),
        learning_rate=float(gru_settings["learning_rate"]),
        seed=int(gru_settings["seed"]),
    )
    gru_prediction = gru_rollout(
        gru_model,
        validation_flat[:, 0, :],
        validation_actions,
        time_count,
    )

    adapters = [
        _adapter_record(
            "persistence",
            persistence_prediction,
            validation_flat,
            implementation=ADAPTER_SOURCE,
            details={"action_conditioning": "none", "training": "not_required"},
        ),
        _adapter_record(
            "linear_state_space",
            linear_prediction,
            validation_flat,
            implementation=ADAPTER_SOURCE,
            details={
                "action_conditioning": "constant_current_amplitude_nA",
                "ridge_penalty": float(linear_settings["ridge_penalty"]),
                "training_split": "train_only",
            },
        ),
        _adapter_record(
            "gru",
            gru_prediction,
            validation_flat,
            implementation=ADAPTER_SOURCE,
            details={
                "action_conditioning": "constant_current_amplitude_nA",
                "hidden_size": int(gru_settings["hidden_size"]),
                "epoch_count": int(gru_settings["epoch_count"]),
                "learning_rate": float(gru_settings["learning_rate"]),
                "seed": int(gru_settings["seed"]),
                "training_split": "train_only",
            },
        ),
    ]
    if any(adapter["qualification_status"] != "adapter_qualified" for adapter in adapters):
        raise ValueError("Restricted C2 adapter qualification did not complete for every simple baseline.")

    source_manifest_root = _resolve(source_root, DEFAULT_MANIFEST_ROOT)
    source_readiness = _resolve(source_root, DEFAULT_READINESS)
    source_integrity = _resolve(source_root, DEFAULT_INTEGRITY)
    source_config = _resolve(source_root, DEFAULT_CONFIG)
    bindings = {
        "config": _file_binding(source_config if source_config.is_file() else config_path, source_root),
        "action_catalog": _file_binding(
            source_manifest_root / "action_catalog.tsv" if (source_manifest_root / "action_catalog.tsv").is_file() else action_path,
            source_root,
        ),
        "trajectory_manifest": _file_binding(
            source_manifest_root / "trajectory_manifest.tsv" if (source_manifest_root / "trajectory_manifest.tsv").is_file() else trajectory_path,
            source_root,
        ),
        "matched_control_manifest": _file_binding(
            source_manifest_root / "matched_control_manifest.tsv" if (source_manifest_root / "matched_control_manifest.tsv").is_file() else matched_control_path,
            source_root,
        ),
        "strict_split_manifest": _file_binding(
            source_manifest_root / "strict_split_manifest.tsv" if (source_manifest_root / "strict_split_manifest.tsv").is_file() else split_path,
            source_root,
        ),
        "domain_readiness": _file_binding(source_readiness if source_readiness.is_file() else readiness_path, source_root),
        "integrity_audit": _file_binding(source_integrity if source_integrity.is_file() else integrity_path, source_root),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "created_at": _utc_now(),
        "qualification_scope": "train_validation_adapter_smoke_only",
        "claim_scope": "simulator_grounded_c2_restricted",
        "domain_uid": readiness.get("selected_domain"),
        "trajectory_store": {
            "declared_path": declared_store,
            "readback_location": "node_local_staging",
            "store_hash": expected_store_hash,
            "shape": list(shape),
            "chunks": list(metadata["chunks"]),
        },
        "split_access": {
            "train_trajectory_count": len(train_indices),
            "validation_trajectory_count": len(validation_indices),
            "final_holdout_trajectory_count_declared": len(test_indices),
            "final_holdout_indices_declared": test_indices.tolist(),
            "final_holdout_prediction_materialized": False,
            "final_holdout_metrics_computed": False,
            "validation_used_for_tuning": False,
        },
        "adapters": adapters,
        "formal_challenge_started": False,
        "challenge_thresholds_frozen": False,
        "headline_training_started": False,
        "controlled_access_count": 0,
        "execution": {
            "job_uid": os.environ.get("SLURM_JOB_ID", "manual"),
            "node": os.environ.get("SLURMD_NODENAME", "local"),
            "mode": "compute_node_adapter_qualification",
        },
        "input_bindings": bindings,
        "artifact_manifest": [
            ADAPTER_SOURCE,
            "Libs/scripts/builders/build_restricted_c2_baseline_qualification.py",
            "Libs/scripts/validators/validate_restricted_c2_baseline_qualification.py",
        ],
        "blockers": [],
    }


def _failure_payload(error: Exception) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "created_at": _utc_now(),
        "qualification_scope": "train_validation_adapter_smoke_only",
        "formal_challenge_started": False,
        "challenge_thresholds_frozen": False,
        "headline_training_started": False,
        "controlled_access_count": 0,
        "execution": {
            "job_uid": os.environ.get("SLURM_JOB_ID", "manual"),
            "node": os.environ.get("SLURMD_NODENAME", "local"),
            "mode": "compute_node_adapter_qualification",
        },
        "blockers": [{"code": "qualification_execution_failed", "message": str(error)}],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train-validation-only restricted C2 baseline adapter qualification.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--manifest-root", default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--readiness", default=DEFAULT_READINESS)
    parser.add_argument("--integrity", default=DEFAULT_INTEGRITY)
    parser.add_argument("--store-path", default="")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", default=REPO_ROOT.as_posix())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_path = _resolve(source_root, args.output)
    try:
        payload = build(
            config_path=_resolve(source_root, args.config),
            manifest_root=_resolve(source_root, args.manifest_root),
            readiness_path=_resolve(source_root, args.readiness),
            integrity_path=_resolve(source_root, args.integrity),
            output_path=output_path,
            store_path=Path(args.store_path).resolve() if args.store_path else None,
            source_root=source_root,
        )
    except (KeyError, TypeError, ValueError, OSError, np.linalg.LinAlgError) as error:
        write_json(output_path, _failure_payload(error))
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        raise SystemExit(1)
    write_json(output_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "domain_uid": payload["domain_uid"],
                "qualified_adapters": [adapter["baseline_uid"] for adapter in payload["adapters"]],
                "formal_challenge_started": payload["formal_challenge_started"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
