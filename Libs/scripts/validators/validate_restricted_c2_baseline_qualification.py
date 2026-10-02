"""Validate the claim boundary and provenance of restricted C2 adapter qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ARTIFACT = "Output/audits/c2/current_clamp/baseline_qualification.json"
DEFAULT_SCHEMA = "Libs/configs/schema/restricted_c2_baseline_qualification.schema.json"
DEFAULT_READINESS = "Output/audits/c2_domain_readiness.json"
DEFAULT_INTEGRITY = "Output/audits/c2/current_clamp/integrity_audit.json"
EXPECTED_ADAPTERS = {"persistence", "linear_state_space", "gru"}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object payload at {path}")
    return payload


def _safe_binding_path(root: Path, value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe qualification binding path: {value}")
    return root / relative


def validate_payload(payload: dict[str, Any], schema: dict[str, Any], source_root: Path) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path))
    for error in errors[:20]:
        blockers.append({"code": "schema_invalid", "message": error.message})
    if payload.get("status") != "passed":
        blockers.append({"code": "qualification_not_passed", "message": str(payload.get("status", "missing"))})
        return {"status": "failed", "blocker_count": len(blockers), "blockers": blockers}

    required_false = (
        "formal_challenge_started",
        "challenge_thresholds_frozen",
        "headline_training_started",
    )
    for key in required_false:
        if payload.get(key) is not False:
            blockers.append({"code": "claim_boundary_violation", "message": f"{key} must be false"})
    if payload.get("controlled_access_count") != 0:
        blockers.append({"code": "controlled_access_violation", "message": "controlled_access_count must be zero"})
    if payload.get("claim_scope") != "simulator_grounded_c2_restricted":
        blockers.append({"code": "claim_scope_invalid", "message": str(payload.get("claim_scope", "missing"))})

    split_access = payload.get("split_access")
    if not isinstance(split_access, dict):
        blockers.append({"code": "split_access_missing", "message": "split_access must be an object"})
    else:
        if int(split_access.get("train_trajectory_count", 0)) <= 0:
            blockers.append({"code": "train_split_empty", "message": "train split must be non-empty"})
        if int(split_access.get("validation_trajectory_count", 0)) <= 0:
            blockers.append({"code": "validation_split_empty", "message": "validation split must be non-empty"})
        if int(split_access.get("final_holdout_trajectory_count_declared", 0)) <= 0:
            blockers.append({"code": "final_holdout_missing", "message": "final holdout must be declared"})
        if split_access.get("final_holdout_prediction_materialized") is not False:
            blockers.append({"code": "final_holdout_prediction_written", "message": "qualification must not write final-holdout predictions"})
        if split_access.get("final_holdout_metrics_computed") is not False:
            blockers.append({"code": "final_holdout_metric_computed", "message": "qualification must not compute final-holdout metrics"})
        if split_access.get("validation_used_for_tuning") is not False:
            blockers.append({"code": "validation_tuning_unexpected", "message": "qualification must not tune on validation results"})

    adapters = payload.get("adapters")
    if not isinstance(adapters, list):
        blockers.append({"code": "adapters_missing", "message": "adapters must be a list"})
    else:
        seen = {str(row.get("baseline_uid", "")) for row in adapters if isinstance(row, dict)}
        if seen != EXPECTED_ADAPTERS:
            blockers.append({"code": "simple_adapter_set_invalid", "message": f"observed={sorted(seen)}"})
        for row in adapters:
            if not isinstance(row, dict):
                blockers.append({"code": "adapter_row_invalid", "message": "adapter row is not an object"})
                continue
            if row.get("qualification_status") != "adapter_qualified":
                blockers.append({"code": "adapter_not_qualified", "message": str(row.get("baseline_uid", ""))})
            if row.get("validation_only") is not True:
                blockers.append({"code": "adapter_not_validation_only", "message": str(row.get("baseline_uid", ""))})
            mse = row.get("validation_mse")
            if not isinstance(mse, (int, float)) or not math.isfinite(float(mse)) or float(mse) < 0.0:
                blockers.append({"code": "adapter_metric_invalid", "message": str(row.get("baseline_uid", ""))})
            digest = str(row.get("validation_prediction_sha256", ""))
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                blockers.append({"code": "adapter_prediction_hash_invalid", "message": str(row.get("baseline_uid", ""))})

    bindings = payload.get("input_bindings")
    expected_bindings = {
        "config",
        "action_catalog",
        "trajectory_manifest",
        "matched_control_manifest",
        "strict_split_manifest",
        "domain_readiness",
        "integrity_audit",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_bindings:
        blockers.append({"code": "input_bindings_invalid", "message": "qualification bindings are incomplete"})
    else:
        for name, binding in bindings.items():
            if not isinstance(binding, dict):
                blockers.append({"code": "input_binding_invalid", "message": name})
                continue
            try:
                path = _safe_binding_path(source_root, binding.get("path", ""))
            except ValueError as error:
                blockers.append({"code": "input_binding_path_invalid", "message": str(error)})
                continue
            if not path.is_file():
                blockers.append({"code": "input_binding_missing", "message": name})
            elif _sha256_file(path) != binding.get("sha256"):
                blockers.append({"code": "input_binding_hash_mismatch", "message": name})

    try:
        readiness = _json(source_root / DEFAULT_READINESS)
        integrity = _json(source_root / DEFAULT_INTEGRITY)
        expected_hash = readiness.get("producer", {}).get("trajectory_store_sha256")
        if payload.get("domain_uid") != readiness.get("selected_domain"):
            blockers.append({"code": "domain_identity_mismatch", "message": "domain_uid differs from readiness audit"})
        if payload.get("trajectory_store", {}).get("store_hash") != expected_hash:
            blockers.append({"code": "trajectory_store_hash_mismatch", "message": "qualification store hash differs from readiness"})
        if integrity.get("store_hash") != expected_hash or integrity.get("strict_split_overlap") != 0:
            blockers.append({"code": "integrity_audit_invalid", "message": "restricted C2 integrity audit is not clean"})
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        blockers.append({"code": "authority_audit_unavailable", "message": str(error)})

    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "qualified_adapter_count": len(adapters) if isinstance(adapters, list) else 0,
        "formal_challenge_started": payload.get("formal_challenge_started"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate restricted C2 baseline adapter qualification evidence.")
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--source-root", default=REPO_ROOT.as_posix())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    artifact = _resolve(source_root, args.artifact)
    schema_path = _resolve(source_root, args.schema)
    try:
        report = validate_payload(_json(artifact), _json(schema_path), source_root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        report = {
            "status": "failed",
            "blocker_count": 1,
            "blockers": [{"code": "validation_input_unavailable", "message": str(error)}],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
