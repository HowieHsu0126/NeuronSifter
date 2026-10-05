"""Independently validate Round-23 checkpoint hashes, payloads, and run links."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.train_restricted_c2_vnext import _sha256_file  # noqa: E402
from Libs.models.networks.foundation_state_encoder import (  # noqa: E402
    NeuronSifterFoundationStateEncoder,
)
from Libs.utils.data.raw_data_registry import write_json  # noqa: E402


C1_CHECKPOINT_SCHEMA_VERSION = "neuronsifter-round23-c1-checkpoint-v1"
C1_ROUNDTRIP_OUTPUT_KEYS = (
    "latent_mean",
    "latent_log_variance",
    "biological_unit_latent_mean",
    "reconstructed_values",
)
C1_ROUNDTRIP_RTOL = 1e-5
C1_ROUNDTRIP_ATOL = 1e-6


def _repo_path(value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else REPO_ROOT / path).resolve()
    resolved.relative_to(REPO_ROOT.resolve())
    return resolved


def _finite_model_state(payload: dict[str, Any]) -> bool:
    model = payload.get("model")
    if not isinstance(model, dict) or not model:
        return False
    for value in model.values():
        if not isinstance(value, torch.Tensor):
            return False
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
            value
        ).all():
            return False
    return True


def _c1_model_from_constructor(
    constructor: Mapping[str, Any],
) -> NeuronSifterFoundationStateEncoder:
    return NeuronSifterFoundationStateEncoder(
        feature_vocabulary_size=int(constructor["feature_vocabulary_size"]),
        cell_ontology_size=int(constructor["cell_ontology_size"]),
        region_ontology_size=int(constructor["region_ontology_size"]),
        source_class_count=int(constructor["source_class_count"]),
        family_class_count=int(constructor["family_class_count"]),
        action_feature_dim=int(constructor["action_feature_dim"]),
        program_count=int(constructor["program_count"]),
        program_vocab_status=str(constructor["program_vocab_status"]),
        profile=constructor["profile"],
        posterior=constructor["posterior"],
    )


def _validate_c1_prediction_roundtrip(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != C1_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("C1 checkpoint schema mismatch")
    if (
        payload.get("lineage_id") != "neuronsifter_vnext_dev_v1"
        or payload.get("domain") != "c1"
        or payload.get("evidence_role") != "development_only"
        or payload.get("publication_claim_eligible") is not False
        or payload.get("protected_test_access") != 0
    ):
        raise ValueError("C1 checkpoint evidence boundary mismatch")
    constructor = payload.get("model_constructor")
    probe = payload.get("roundtrip_probe")
    if not isinstance(constructor, Mapping) or not isinstance(probe, Mapping):
        raise ValueError("C1 checkpoint constructor or roundtrip probe is missing")
    if probe.get("sample_posterior") is not False:
        raise ValueError("C1 checkpoint roundtrip probe must be deterministic")
    inputs = probe.get("inputs")
    options = probe.get("forward_options")
    expected = probe.get("expected_outputs")
    if (
        not isinstance(inputs, Mapping)
        or not isinstance(options, Mapping)
        or not isinstance(expected, Mapping)
        or any(not isinstance(value, torch.Tensor) for value in inputs.values())
    ):
        raise ValueError("C1 checkpoint roundtrip probe payload is invalid")
    model = _c1_model_from_constructor(constructor)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_ontology_context=bool(options["use_ontology_context"]),
            use_action_context=bool(options["use_action_context"]),
            sample_posterior=False,
            nuisance_gradient_scale=float(options["nuisance_gradient_scale"]),
        )
    max_abs_error: dict[str, float] = {}
    matches = True
    finite = True
    for key in C1_ROUNDTRIP_OUTPUT_KEYS:
        expected_value = expected.get(key)
        actual_value = outputs.get(key)
        if not isinstance(expected_value, torch.Tensor) or not isinstance(
            actual_value, torch.Tensor
        ):
            raise ValueError(f"C1 checkpoint roundtrip output {key} is missing")
        expected_value = expected_value.cpu()
        actual_value = actual_value.detach().cpu()
        finite = finite and bool(
            torch.isfinite(expected_value).all() and torch.isfinite(actual_value).all()
        )
        if actual_value.shape != expected_value.shape:
            matches = False
            max_abs_error[key] = float("inf")
            continue
        difference = torch.abs(actual_value - expected_value)
        max_abs_error[key] = float(difference.max().item()) if difference.numel() else 0.0
        matches = matches and bool(
            torch.allclose(
                actual_value,
                expected_value,
                rtol=C1_ROUNDTRIP_RTOL,
                atol=C1_ROUNDTRIP_ATOL,
            )
        )
    finite_uncertainty = bool(
        torch.isfinite(outputs["latent_log_variance"]).all()
        and torch.isfinite(torch.exp(0.5 * outputs["latent_log_variance"])).all()
    )
    return {
        "prediction_roundtrip": bool(matches and finite and finite_uncertainty),
        "finite_outputs": bool(finite),
        "finite_uncertainty": finite_uncertainty,
        "output_keys": list(C1_ROUNDTRIP_OUTPUT_KEYS),
        "max_abs_error": max_abs_error,
        "rtol": C1_ROUNDTRIP_RTOL,
        "atol": C1_ROUNDTRIP_ATOL,
    }


def validate(*, stage_root: Path) -> dict[str, Any]:
    catalog_path = REPO_ROOT / "Output/audits/round23_development_checkpoint_catalog.json"
    training_path = REPO_ROOT / "Output/audits/round23_training_manifest.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if catalog.get("lineage_id") != "neuronsifter_vnext_dev_v1":
        raise ValueError("Round-23 checkpoint catalog lineage drift")
    if int(catalog.get("protected_test_access_count", -1)) != 0:
        raise ValueError("Round-23 checkpoint catalog opened protected-test access")
    stage_root.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    row_by_path: dict[str, dict[str, Any]] = {}
    run_rows = training.get("runs", [])
    run_by_checkpoint = {
        str(row["checkpoint_path"]): row
        for row in run_rows
        if row.get("checkpoint_path")
    }
    if len(run_by_checkpoint) != sum(
        bool(row.get("checkpoint_path")) for row in run_rows
    ):
        blockers.append("duplicate_training_checkpoint_path")
    for run in run_rows:
        if (
            run.get("evidence_role") != "development_only"
            or run.get("publication_claim_eligible") is not False
            or run.get("protected_test_access") != 0
        ):
            blockers.append(
                f"run_boundary_mismatch:{run.get('domain')}:{run.get('run_uid')}"
            )
        try:
            run_result_path = _repo_path(str(run.get("run_result_path", "")))
            if (
                not run_result_path.is_file()
                or _sha256_file(run_result_path) != run.get("run_result_sha256")
            ):
                blockers.append(
                    f"run_result_hash_mismatch:{run.get('domain')}:{run.get('run_uid')}"
                )
        except (ValueError, OSError):
            blockers.append(
                f"run_result_path_invalid:{run.get('domain')}:{run.get('run_uid')}"
            )
    for index, entry in enumerate(catalog.get("checkpoints", [])):
        relative_path = str(entry["path"])
        source_path = _repo_path(relative_path)
        row: dict[str, Any] = {
            "path": relative_path,
            "artifact_kind": entry["artifact_kind"],
            "expected_sha256": entry["sha256"],
            "source_exists": source_path.is_file(),
            "staged_sha256": None,
            "hash_matches": False,
            "payload_loadable": False,
            "payload_identity_matches": None,
            "finite_model_state": None,
            "run_checkpoint_sha_matches": None,
            "payload_boundary_matches": None,
            "c1_prediction_roundtrip": None,
            "c1_roundtrip_max_abs_error": None,
            "error": None,
        }
        if not source_path.is_file():
            row["error"] = "source_missing"
            blockers.append(f"missing:{relative_path}")
            rows.append(row)
            row_by_path[relative_path] = row
            continue
        staged_path = stage_root / f"{index:03d}_{source_path.name}"
        try:
            shutil.copy2(source_path, staged_path)
            observed_hash = _sha256_file(staged_path)
            row["staged_sha256"] = observed_hash
            row["hash_matches"] = observed_hash == str(entry["sha256"])
            if not row["hash_matches"]:
                blockers.append(f"hash_mismatch:{relative_path}")
            if source_path.suffix == ".pt":
                payload = torch.load(staged_path, map_location="cpu", weights_only=False)
                if not isinstance(payload, dict):
                    raise TypeError("checkpoint payload is not an object")
                row["payload_loadable"] = True
                row["finite_model_state"] = _finite_model_state(payload)
                if not row["finite_model_state"]:
                    blockers.append(f"nonfinite_or_invalid_model_state:{relative_path}")
                run = run_by_checkpoint.get(relative_path)
                if run is None:
                    row["payload_identity_matches"] = False
                    blockers.append(f"checkpoint_without_run:{relative_path}")
                else:
                    row["run_checkpoint_sha_matches"] = (
                        str(run.get("checkpoint_sha256")) == observed_hash
                    )
                    if not row["run_checkpoint_sha_matches"]:
                        blockers.append(f"run_checkpoint_hash_mismatch:{relative_path}")
                    row["payload_identity_matches"] = (
                        str(payload.get("model_scale")) == str(run["model_scale"])
                        and int(payload.get("seed", -1)) == int(run["seed"])
                    )
                    if not row["payload_identity_matches"]:
                        blockers.append(f"payload_identity_mismatch:{relative_path}")
                    if run.get("domain") == "c1":
                        row["payload_boundary_matches"] = (
                            payload.get("schema_version")
                            == C1_CHECKPOINT_SCHEMA_VERSION
                            and payload.get("lineage_id")
                            == "neuronsifter_vnext_dev_v1"
                            and payload.get("domain") == "c1"
                            and payload.get("evidence_role") == "development_only"
                            and payload.get("publication_claim_eligible") is False
                            and payload.get("protected_test_access") == 0
                            and payload.get("checkpoint_path") == relative_path
                        )
                        if not row["payload_boundary_matches"]:
                            blockers.append(f"c1_payload_boundary_mismatch:{relative_path}")
                        c1_roundtrip = _validate_c1_prediction_roundtrip(payload)
                        row["c1_prediction_roundtrip"] = c1_roundtrip[
                            "prediction_roundtrip"
                        ]
                        row["c1_roundtrip_max_abs_error"] = c1_roundtrip[
                            "max_abs_error"
                        ]
                        if not row["c1_prediction_roundtrip"]:
                            blockers.append(
                                f"c1_prediction_roundtrip_failed:{relative_path}"
                            )
                    elif run.get("domain") == "c2s" and (
                        payload.get("schema_version")
                        != "neuronsifter-round23-c2s-checkpoint-v1"
                        or payload.get("evidence_role") != "development_only"
                        or payload.get("publication_claim_eligible") is not False
                    ):
                        row["payload_boundary_matches"] = False
                        blockers.append(f"c2s_payload_boundary_mismatch:{relative_path}")
                    else:
                        row["payload_boundary_matches"] = True
            else:
                payload = json.loads(staged_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise TypeError("checkpoint metadata payload is not an object")
                row["payload_loadable"] = True
                row["payload_identity_matches"] = True
                if "publication_claim_eligible" in payload and payload.get(
                    "publication_claim_eligible"
                ) is not False:
                    blockers.append(f"metadata_publication_boundary_mismatch:{relative_path}")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}:{exc}"
            blockers.append(f"payload_error:{relative_path}:{type(exc).__name__}")
        finally:
            if staged_path.exists():
                staged_path.unlink()
        rows.append(row)
        row_by_path[relative_path] = row

    passed_runs = [
        row for row in training["runs"] if row.get("status") in {"passed", "completed"}
    ]
    for run in passed_runs:
        path = str(run.get("checkpoint_path") or "")
        validation = row_by_path.get(path)
        source_boundary_valid = (
            run.get("evidence_role") == "development_only"
            and run.get("publication_claim_eligible") is False
            and run.get("protected_test_access") == 0
            and run.get("finite_loss") is True
            and run.get("finite_uncertainty") is True
        )
        c1_roundtrip_valid = (
            run.get("domain") != "c1"
            or validation is not None
            and validation.get("c1_prediction_roundtrip") is True
        )
        if (
            not path
            or run.get("checkpoint_roundtrip") is not True
            or not source_boundary_valid
            or not c1_roundtrip_valid
            or validation is None
            or validation.get("hash_matches") is not True
            or validation.get("run_checkpoint_sha_matches") is not True
            or validation.get("payload_loadable") is not True
            or validation.get("payload_identity_matches") is not True
            or validation.get("finite_model_state") is not True
        ):
            blockers.append(f"passed_run_checkpoint_invalid:{run['domain']}:{run['run_uid']}")
    report = {
        "schema_version": "neuronsifter-round23-checkpoint-validation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not blockers else "failed",
        "lineage_id": "neuronsifter_vnext_dev_v1",
        "catalog_path": catalog_path.relative_to(REPO_ROOT).as_posix(),
        "catalog_sha256": _sha256_file(catalog_path),
        "training_manifest_path": training_path.relative_to(REPO_ROOT).as_posix(),
        "training_manifest_sha256": _sha256_file(training_path),
        "validation_host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node_local_staging_used": True,
        "catalog_entry_count": len(rows),
        "passed_run_count": len(passed_runs),
        "all_catalog_hashes_match": all(row["hash_matches"] for row in rows),
        "all_payloads_loadable": all(row["payload_loadable"] for row in rows),
        "all_c1_prediction_roundtrips": all(
            row.get("c1_prediction_roundtrip") is True
            for row in rows
            if row.get("artifact_kind") == "checkpoint"
            and row.get("path") in run_by_checkpoint
            and run_by_checkpoint[row["path"]].get("domain") == "c1"
        ),
        "all_passed_run_checkpoints_valid": not any(
            value.startswith("passed_run_checkpoint_invalid:") for value in blockers
        ),
        "rows": rows,
        "blockers": sorted(set(blockers)),
        "protected_test_access_count": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(REPO_ROOT / "Output/audits/round23_checkpoint_validation.json", report)
    if blockers:
        raise RuntimeError("Round-23 checkpoint validation failed: " + "; ".join(report["blockers"]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True)
    arguments = parser.parse_args()
    result = validate(stage_root=Path(arguments.stage_root).resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "catalog_entry_count": result["catalog_entry_count"],
                "passed_run_count": result["passed_run_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
