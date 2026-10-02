"""Audit current C1/C2 contracts after the rescue-identity migration.

The audit records the pre-migration configuration hashes supplied by the
Round-20M handoff, verifies the current public-corpus binding, and reports
missing active artifacts without substituting historical evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


CURRENT_RESCUE_UID = "public_foundation_corpus_rescue"
CURRENT_RESCUE_HASH = "3c9899d1704ce262801229d98fb5402f49159fb8959295d3b86600522b4ab7f1"
CURRENT_RESCUE_BUNDLE = "Output/audits/public_foundation_corpus_rescue_evidence_bundle.json"
CURRENT_RESCUE_VALIDATION = "Output/audits/public_foundation_corpus_rescue_evidence_validation.json"
SCHEMA_PATH = "Libs/configs/schema/active_contract_migration.schema.json"
DEFAULT_OUTPUT_PATH = "Output/audits/active_contract_migration.json"

CONTRACT_SPECS = (
    {
        "uid": "c1_foundation_rescue",
        "config_path": "Libs/configs/exp/foundation_state_c1_rescue.yaml",
        "load_path": "Libs/configs/config_stack_c1_rescue.yaml",
        "root_key": "foundation_state_c1",
        "pre_migration_config_sha256": "35944c66fc423ea4fbecfd330ac9d3c4d3822a25679324f732d3262bc7d06a58",
    },
    {
        "uid": "c2_structured_world_action",
        "config_path": "Libs/configs/exp/structured_world_action_c2.yaml",
        "load_path": "Libs/configs/config_stack.yaml",
        "root_key": "structured_world_action_c2",
        "pre_migration_config_sha256": "1f54df0e610c656ce57ea5be1c5f70d4196cc14fbe80df4ac0a31873ed8cb4e8",
    },
    {
        "uid": "c2_trajectory_action_registry",
        "config_path": "Libs/configs/exp/trajectory_action_registry_c2r.yaml",
        "load_path": "Libs/configs/config_stack.yaml",
        "root_key": "trajectory_action_registry_c2r",
        "pre_migration_config_sha256": "b0ba5700b1b8a24e0254ef7391ae8bbd0f46d3b8ad5a891ff3e548dd847d76b9",
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _check(name: str, expected: object, actual: object) -> dict[str, object]:
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
    }


def _artifact(
    repo_root: Path,
    name: str,
    relative_path: str,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    path = safe_repo_path(repo_root, relative_path)
    exists = path.is_file()
    digest = sha256_file(path) if exists else None
    return {
        "name": name,
        "path": relative_path,
        "exists": exists,
        "sha256": digest,
        "expected_sha256": expected_sha256,
        "hash_matches": exists if expected_sha256 is None else digest == expected_sha256,
    }


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for {name}")
    return value


def _c1_contract_record(repo_root: Path, spec: Mapping[str, str]) -> dict[str, object]:
    config = load_config(repo_root / spec["load_path"])
    contract = _require_mapping(config.get(spec["root_key"]), spec["root_key"])
    frozen = _require_mapping(contract.get("frozen_corpus"), "frozen_corpus")
    execution = _require_mapping(contract.get("execution"), "execution")
    artifacts = _require_mapping(contract.get("artifacts"), "artifacts")
    encoder = _require_mapping(contract.get("encoder"), "encoder")
    input_contract = _require_mapping(encoder.get("input_contract"), "input_contract")

    schema_path = str(contract["schema_path"])
    vocabulary_schema_path = "Libs/configs/schema/c1_rescue_vocabulary_manifest.schema.json"
    vocabulary_schema_hash: object = None
    vocabulary_schema = safe_repo_path(repo_root, vocabulary_schema_path)
    if vocabulary_schema.is_file():
        vocabulary_schema_hash = _load_json(vocabulary_schema).get("properties", {}).get(
            "corpus_hash", {}
        ).get("const")

    binding_checks = [
        _check("corpus_uid", CURRENT_RESCUE_UID, frozen.get("corpus_uid")),
        _check("corpus_identity_hash", CURRENT_RESCUE_HASH, frozen.get("identity_hash")),
        _check("evidence_bundle_path", CURRENT_RESCUE_BUNDLE, frozen.get("evidence_bundle_path")),
        _check("evidence_validation_path", CURRENT_RESCUE_VALIDATION, execution.get("evidence_validation_path")),
        _check("vocabulary_schema_corpus_hash", CURRENT_RESCUE_HASH, vocabulary_schema_hash),
    ]
    artifact_checks = [
        _artifact(repo_root, "contract_schema", schema_path),
        _artifact(
            repo_root,
            "vocabulary_manifest",
            str(input_contract["vocabulary_manifest_path"]),
            str(input_contract["vocabulary_manifest_sha256"]),
        ),
        _artifact(repo_root, "vocabulary_schema", vocabulary_schema_path),
        _artifact(repo_root, "unified_preflight", str(execution["unified_preflight_path"])),
        _artifact(repo_root, "corpus_preflight", str(artifacts["preflight_output_path"])),
        _artifact(
            repo_root,
            "objective_capability_manifest",
            str(artifacts["objective_capability_manifest_path"]),
            str(artifacts["objective_capability_manifest_sha256"]),
        ),
        _artifact(
            repo_root,
            "unified_preflight_schema",
            str(artifacts["unified_preflight_schema_path"]),
        ),
    ]
    config_path = safe_repo_path(repo_root, spec["config_path"])
    return {
        "uid": spec["uid"],
        "config_path": spec["config_path"],
        "pre_migration_config_sha256": spec["pre_migration_config_sha256"],
        "current_config_sha256": sha256_file(config_path),
        "resolved_contract_sha256": canonical_sha256(contract),
        "binding_checks": binding_checks,
        "artifact_checks": artifact_checks,
        "migration_binding_valid": all(check["passed"] for check in binding_checks),
    }


def _c2_world_action_record(repo_root: Path, spec: Mapping[str, str]) -> dict[str, object]:
    config = load_config(repo_root / spec["load_path"])
    contract = _require_mapping(config.get(spec["root_key"]), spec["root_key"])
    state_corpus = _require_mapping(contract.get("selected_state_corpus"), "selected_state_corpus")
    binding_checks = [
        _check("corpus_uid", CURRENT_RESCUE_UID, state_corpus.get("corpus_uid")),
        _check("corpus_identity_hash", CURRENT_RESCUE_HASH, state_corpus.get("identity_hash")),
        _check("evidence_bundle_path", CURRENT_RESCUE_BUNDLE, state_corpus.get("evidence_bundle_path")),
        _check("evidence_validation_path", CURRENT_RESCUE_VALIDATION, state_corpus.get("evidence_validation_path")),
    ]
    config_path = safe_repo_path(repo_root, spec["config_path"])
    return {
        "uid": spec["uid"],
        "config_path": spec["config_path"],
        "pre_migration_config_sha256": spec["pre_migration_config_sha256"],
        "current_config_sha256": sha256_file(config_path),
        "resolved_contract_sha256": canonical_sha256(contract),
        "binding_checks": binding_checks,
        "artifact_checks": [
            _artifact(repo_root, "c2_domain_readiness", "Output/audits/c2_domain_readiness.json"),
            _artifact(repo_root, "c2_integrity", "Output/audits/c2/current_clamp/integrity_audit.json"),
        ],
        "migration_binding_valid": all(check["passed"] for check in binding_checks),
    }


def _c2_registry_record(repo_root: Path, spec: Mapping[str, str]) -> dict[str, object]:
    config = load_config(repo_root / spec["load_path"])
    contract = _require_mapping(config.get(spec["root_key"]), spec["root_key"])
    immutable_inputs = contract.get("immutable_inputs")
    if not isinstance(immutable_inputs, list):
        raise TypeError("Expected immutable_inputs list")
    current_entries = [
        entry
        for entry in immutable_inputs
        if isinstance(entry, Mapping) and entry.get("corpus_uid") == CURRENT_RESCUE_UID
    ]
    current_entry = current_entries[0] if len(current_entries) == 1 else {}
    binding_checks = [
        _check("current_rescue_entry_count", 1, len(current_entries)),
        _check("corpus_identity_hash", CURRENT_RESCUE_HASH, current_entry.get("identity_hash")),
        _check("evidence_bundle_path", CURRENT_RESCUE_BUNDLE, current_entry.get("evidence_bundle_path")),
    ]
    config_path = safe_repo_path(repo_root, spec["config_path"])
    return {
        "uid": spec["uid"],
        "config_path": spec["config_path"],
        "pre_migration_config_sha256": spec["pre_migration_config_sha256"],
        "current_config_sha256": sha256_file(config_path),
        "resolved_contract_sha256": canonical_sha256(contract),
        "binding_checks": binding_checks,
        "artifact_checks": [
            _artifact(repo_root, "c2_domain_scorecard", "Output/audits/c2_domain_scorecard.tsv"),
            _artifact(repo_root, "c2_selected_domain", "Output/audits/c2_selected_domain.json"),
        ],
        "migration_binding_valid": all(check["passed"] for check in binding_checks),
    }


def _current_rescue(repo_root: Path) -> dict[str, object]:
    bundle = _load_json(safe_repo_path(repo_root, CURRENT_RESCUE_BUNDLE))
    validation = _load_json(safe_repo_path(repo_root, CURRENT_RESCUE_VALIDATION))
    return {
        "corpus_uid": bundle.get("corpus_uid"),
        "corpus_hash": bundle.get("corpus_hash"),
        "evidence_status": bundle.get("status"),
        "validation_status": validation.get("status"),
    }


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    records = [
        _c1_contract_record(repo_root, CONTRACT_SPECS[0]),
        _c2_world_action_record(repo_root, CONTRACT_SPECS[1]),
        _c2_registry_record(repo_root, CONTRACT_SPECS[2]),
    ]
    current_rescue = _current_rescue(repo_root)
    blockers: list[dict[str, str]] = []
    if current_rescue != {
        "corpus_uid": CURRENT_RESCUE_UID,
        "corpus_hash": CURRENT_RESCUE_HASH,
        "evidence_status": "valid_fail_closed",
        "validation_status": "valid_fail_closed",
    }:
        blockers.append(
            {
                "code": "current_rescue_evidence_invalid",
                "path": CURRENT_RESCUE_BUNDLE,
                "message": "The current rescue bundle or validation no longer matches its frozen public identity.",
            }
        )
    for record in records:
        for check in record["binding_checks"]:
            if not check["passed"]:
                blockers.append(
                    {
                        "code": "active_contract_binding_mismatch",
                        "path": str(record["config_path"]),
                        "message": str(check["name"]),
                    }
                )
        for artifact in record["artifact_checks"]:
            if not artifact["exists"]:
                blockers.append(
                    {
                        "code": "active_contract_artifact_unavailable",
                        "path": str(artifact["path"]),
                        "message": str(artifact["name"]),
                    }
                )
            elif not artifact["hash_matches"]:
                blockers.append(
                    {
                        "code": "active_contract_artifact_hash_mismatch",
                        "path": str(artifact["path"]),
                        "message": str(artifact["name"]),
                    }
                )
    migration_binding_valid = all(record["migration_binding_valid"] for record in records) and not any(
        blocker["code"] == "current_rescue_evidence_invalid" for blocker in blockers
    )
    execution_artifacts_ready = migration_binding_valid and not any(
        blocker["code"].startswith("active_contract_artifact_")
        or blocker["code"] == "active_contract_binding_mismatch"
        for blocker in blockers
    )
    return {
        "schema_version": "neuronsifter-active-contract-migration",
        "status": "verified" if execution_artifacts_ready else "blocked",
        "training_allowed": False,
        "current_rescue": current_rescue,
        "contracts": records,
        "migration_binding_valid": migration_binding_valid,
        "execution_artifacts_ready": execution_artifacts_ready,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "next_action": (
            "repair_current_c1_contract_artifacts_from_live_evidence"
            if not execution_artifacts_ready
            else "continue_claim_scoped_baseline_qualification"
        ),
    }


def validate_schema(payload: Mapping[str, object], repo_root: Path = REPO_ROOT) -> None:
    schema = _load_json(safe_repo_path(repo_root.resolve(), SCHEMA_PATH))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        raise ValueError(
            "Active contract migration schema violation: "
            + "; ".join(error.message for error in errors)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    payload = build(REPO_ROOT)
    validate_schema(payload, REPO_ROOT)
    write_json(safe_repo_path(REPO_ROOT, args.output), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "migration_binding_valid": payload["migration_binding_valid"],
                "execution_artifacts_ready": payload["execution_artifacts_ready"],
                "blocker_count": payload["blocker_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
