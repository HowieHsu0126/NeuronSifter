"""Validate the C1 G3 decision without treating blockers as empirical results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_c1_experiment_registries import build as build_registries  # noqa: E402
from Libs.utils.data.foundation_state_c1_capabilities import (  # noqa: E402
    derive_c1_headline_feasibility,
    load_c1_objective_capability_manifest,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    contract = config["foundation_state_c1"]
    execution = contract["execution"]
    preflight = _load(repo_root / execution["unified_preflight_path"])
    matrix = _load(repo_root / "Output/audits/c1_job_matrix.json")
    gate = _load(repo_root / contract["artifacts"]["gate_output_path"])
    tasks, _, _, _ = build_registries(repo_root)
    manifest = load_c1_objective_capability_manifest(config, repo_root, verify_live=True)
    capability_path = repo_root / contract["artifacts"]["objective_capability_manifest_path"]
    expected_feasibility = derive_c1_headline_feasibility(
        manifest,
        tasks,
        runtime_profile_id=str(execution["headline_runtime_profile_id"]),
        required_task_family_count=int(contract["acceptance"]["minimum_distinct_task_families"]),
        required_holdout_family_count=int(contract["acceptance"]["minimum_distinct_holdout_families"]),
    )
    blockers: list[dict[str, object]] = []
    if gate.get("corpus_hash") != preflight.get("corpus_identity_hash"):
        blockers.append({"code": "gate_corpus_hash_mismatch"})
    if gate.get("preflight_hash") != canonical_sha256(preflight):
        blockers.append({"code": "gate_preflight_hash_mismatch"})
    if gate.get("capability_manifest_hash") != sha256_file(capability_path):
        blockers.append({"code": "gate_capability_hash_mismatch"})
    if matrix.get("corpus_hash") != preflight.get("corpus_identity_hash"):
        blockers.append({"code": "matrix_corpus_hash_mismatch"})
    if matrix.get("preflight_hash") != canonical_sha256(preflight):
        blockers.append({"code": "matrix_preflight_hash_mismatch"})
    if matrix.get("headline_feasibility") != preflight.get("headline_feasibility"):
        blockers.append({"code": "matrix_headline_feasibility_mismatch"})
    if expected_feasibility != preflight.get("headline_feasibility"):
        blockers.append({"code": "live_headline_feasibility_mismatch"})
    if expected_feasibility["status"] == "unreachable" and gate.get("status") != "blocked_incomplete":
        blockers.append({"code": "unreachable_headline_coverage_not_blocked"})
    if expected_feasibility["status"] == "unreachable" and preflight.get("headline_training_allowed") is True:
        blockers.append({"code": "unreachable_headline_coverage_allowed"})
    if preflight.get("headline_training_allowed") is not True and any(
        row.get("submission_allowed") is True for row in matrix.get("jobs", [])
        if isinstance(row, dict)
    ):
        blockers.append({"code": "blocked_preflight_has_submit_ready_job"})
    if gate.get("claim_allowed") is not False:
        blockers.append({"code": "blocked_gate_claim_permission_drift"})

    schema_path = repo_root / execution["g3_gate_schema_path"]
    schema = _load(schema_path)
    Draft202012Validator.check_schema(schema)
    schema_errors = list(Draft202012Validator(schema).iter_errors(gate))
    blockers.extend({"code": "gate_schema_violation", "message": error.message} for error in schema_errors)
    report = {
        "schema_version": "neuronsifter-c1-g3-validation-v1",
        "status": "passed" if not blockers else "blocked",
        "g3_status": gate.get("status"),
        "integrity_blocker_count": len(blockers),
        "blockers": blockers,
        "claim_allowed": gate.get("claim_allowed"),
        "corpus_hash": preflight.get("corpus_identity_hash"),
        "preflight_hash": canonical_sha256(preflight),
        "capability_manifest_hash": sha256_file(capability_path),
        "provenance": {
            "preflight": execution["unified_preflight_path"],
            "matrix": "Output/audits/c1_job_matrix.json",
            "gate": contract["artifacts"]["gate_output_path"],
        },
    }
    return report


def main() -> None:
    report = validate(REPO_ROOT)
    write_json(REPO_ROOT / "Output/audits/c1_g3_validation.json", report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
