from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping

from Libs.utils.data.claim_gated_wording import build_claim_wording_audit, write_claim_wording_artifacts
from Libs.utils.data.g0_naming import build_naming_remediation_report
from Libs.utils.data.iclr_contract_union import (
    build_config_stack_contract_report,
    build_design_coverage_report,
    build_orchestrator_state_report,
    build_union_consistency_report,
    load_union_context,
    resolve_repo_path,
)
from Libs.utils.data.public_mainline_isolation import build_public_mainline_isolation_reports
from Libs.utils.data.raw_data_registry import load_config, write_json
from Libs.utils.data.trajectory_numeric_lineage import (
    build_numeric_lineage_audit,
    validate_numeric_lineage_report,
    write_numeric_lineage_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
IMMUTABLE_BLOCKER_IDS = (
    "G0-NAMING-LEGACY-RESIDUALS",
    "G0-PUBLIC-MAINLINE-CONTROLLED-INCLUDE",
    "G0-NUMERIC-LINEAGE-DRIFT",
    "G0-CLAIM-GATED-WORDING",
)
REQUIRED_CLAIM_IDS = ("C1", "C2", "C3", "C4", "C5")


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _command_config_path(config_path: str | Path) -> str:
    candidate = Path(config_path)
    if candidate.is_absolute() and candidate.is_relative_to(REPO_ROOT):
        return candidate.relative_to(REPO_ROOT).as_posix()
    return str(config_path)


def _audit_policy(config: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    union = _mapping(config.get("iclr_design_contract_union"))
    audit_contract = _mapping(union.get("audit_contract"))
    return union, audit_contract


def _expected_dedicated_schemas(audit_contract: Mapping[str, object]) -> dict[str, str]:
    schema_versions = _mapping(audit_contract.get("schema_versions"))
    return {
        "naming": str(schema_versions.get("g0_naming_remediation", "")),
        "public_mainline": str(schema_versions.get("g0_public_mainline_remediation", "")),
        "numeric_lineage": str(schema_versions.get("trajectory_numeric_lineage", "")),
        "claim_wording": str(schema_versions.get("g0_claim_wording_remediation", "")),
    }


def _artifact_record(path: Path, expected_schema: str) -> tuple[dict[str, object] | None, dict[str, object]]:
    relative_path = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)
    result: dict[str, object] = {
        "path": relative_path,
        "exists": path.is_file(),
        "sha256": _sha256_file(path),
        "expected_schema_version": expected_schema,
        "schema_version": None,
        "status": "missing",
        "blocker_count": None,
    }
    if not path.is_file():
        return None, result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        result["status"] = "invalid_json"
        result["error"] = str(error)
        return None, result
    if not isinstance(payload, dict):
        result["status"] = "invalid_shape"
        return None, result
    result["schema_version"] = payload.get("schema_version")
    result["blocker_count"] = payload.get("blocker_count")
    result["status"] = "pass" if payload.get("schema_version") == expected_schema else "schema_mismatch"
    return payload, result


def _structural_reports(config_path: str | Path) -> dict[str, dict[str, object]]:
    config, provenance = load_union_context(config_path)
    return {
        "config_stack": build_config_stack_contract_report(config_path),
        "design_coverage": build_design_coverage_report(config),
        "orchestrator_state": build_orchestrator_state_report(config),
        "contract_union": build_union_consistency_report(config, provenance, config_path),
    }


def _artifact_ownership_check(union: Mapping[str, object]) -> dict[str, object]:
    registry = [dict(row) for row in union.get("validator_registry", []) if isinstance(row, Mapping)]
    owners: dict[str, list[str]] = {}
    missing: list[str] = []
    for row in registry:
        validator_id = str(row.get("validator_id", ""))
        output_path = str(row.get("output_path", "")).strip()
        if not output_path:
            missing.append(validator_id)
            continue
        owners.setdefault(output_path, []).append(validator_id)
    duplicates = {
        path: validator_ids
        for path, validator_ids in owners.items()
        if len(validator_ids) != 1
    }
    return {
        "status": "pass" if not duplicates and not missing else "blocked",
        "duplicate_artifact_ownership_count": len(duplicates),
        "duplicate_artifact_owners": duplicates,
        "missing_output_path_validator_ids": missing,
        "registered_artifact_count": len(owners),
        "evidence": "iclr_design_contract_union.validator_registry",
    }


def _claim_mapping(union: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    claims_by_id: dict[str, dict[str, object]] = {}
    errors: list[dict[str, object]] = []
    for raw_claim in union.get("claims", []):
        if not isinstance(raw_claim, Mapping):
            errors.append({"code": "claim_mapping_invalid_row"})
            continue
        claim = dict(raw_claim)
        claim_id = str(claim.get("claim_id", ""))
        if claim_id in claims_by_id:
            errors.append({"code": "claim_mapping_duplicate_claim_id", "claim_id": claim_id})
            continue
        claims_by_id[claim_id] = claim

    expected_ids = set(REQUIRED_CLAIM_IDS)
    actual_ids = set(claims_by_id)
    if actual_ids != expected_ids:
        errors.append(
            {
                "code": "claim_mapping_required_ids_mismatch",
                "expected": list(REQUIRED_CLAIM_IDS),
                "actual": sorted(actual_ids),
            }
        )

    mapping: list[dict[str, object]] = []
    for claim_id in REQUIRED_CLAIM_IDS:
        claim = claims_by_id.get(claim_id)
        if claim is None:
            continue
        meaning = str(claim.get("meaning", "")).strip()
        result_status = str(claim.get("result_status", "")).strip()
        gate_status = str(claim.get("claim_gate_status", "")).strip()
        if not meaning or not result_status or not gate_status:
            errors.append({"code": "claim_mapping_missing_field", "claim_id": claim_id})
        mapping.append(
            {
                "claim_id": claim_id,
                "meaning": meaning,
                "status": result_status,
                "gate_status": gate_status,
            }
        )
    return mapping, errors


def _mandatory_input_resolutions(
    audit_contract: Mapping[str, object], union: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    output_paths = _mapping(audit_contract.get("output_paths"))
    registry = [dict(row) for row in union.get("validator_registry", []) if isinstance(row, Mapping)]
    resolutions: dict[str, dict[str, object]] = {}
    for resolution_id, raw_policy in _mapping(audit_contract.get("mandatory_input_resolutions")).items():
        policy = _mapping(raw_policy)
        required_path = str(policy.get("required_input_path", "")).strip()
        canonical_path = str(policy.get("canonical_replacement_path", "")).strip()
        expected_schema = str(policy.get("expected_schema_version", "")).strip()
        owner_validator_id = str(policy.get("owner_validator_id", "")).strip()
        legacy_policy = str(policy.get("legacy_artifact_policy", "")).strip()
        supersession_reason = str(policy.get("supersession_reason", "")).strip()
        canonical_payload, canonical_check = _artifact_record(resolve_repo_path(canonical_path), expected_schema)
        owner_rows = [
            row
            for row in registry
            if str(row.get("validator_id", "")) == owner_validator_id
            and str(row.get("output_path", "")) == canonical_path
        ]
        legacy_exists = bool(required_path) and resolve_repo_path(required_path).is_file()
        canonical_status = canonical_payload.get("status") if canonical_payload else None
        canonical_blocker_count = canonical_payload.get("blocker_count") if canonical_payload else None
        valid = (
            bool(required_path)
            and bool(canonical_path)
            and bool(expected_schema)
            and bool(owner_validator_id)
            and bool(supersession_reason)
            and canonical_path == str(output_paths.get("contract_union", ""))
            and canonical_check["status"] == "pass"
            and canonical_status == "pass"
            and canonical_blocker_count == 0
            and len(owner_rows) == 1
            and (legacy_policy != "must_be_absent" or not legacy_exists)
        )
        resolutions[str(resolution_id)] = {
            "status": "pass" if valid else "blocked",
            "required_input_path": required_path,
            "canonical_replacement_path": canonical_path,
            "expected_schema_version": expected_schema,
            "canonical_artifact": canonical_check,
            "canonical_status": canonical_status,
            "canonical_blocker_count": canonical_blocker_count,
            "owner_validator_id": owner_validator_id,
            "owner_binding_count": len(owner_rows),
            "legacy_artifact_policy": legacy_policy,
            "legacy_artifact_exists": legacy_exists,
            "supersession_reason": supersession_reason,
        }
    return resolutions


def _write_structural_reports(
    config: Mapping[str, object], reports: Mapping[str, Mapping[str, object]], config_path: str | Path
) -> None:
    _, audit_contract = _audit_policy(config)
    output_paths = _mapping(audit_contract.get("output_paths"))
    command_config_path = _command_config_path(config_path)
    producer_commands = {
        "config_stack": f"python Libs/scripts/validators/validate_config_stack_contract.py --config {command_config_path}",
        "design_coverage": f"python Libs/scripts/validators/validate_design_contract_coverage.py --config {command_config_path}",
        "orchestrator_state": f"python Libs/scripts/validators/validate_orchestrator_state_contract.py --config {command_config_path}",
        "contract_union": f"python Libs/scripts/validators/validate_iclr_design_contract_union.py --config {command_config_path}",
    }
    for key, report in reports.items():
        configured_path = output_paths.get(key)
        if configured_path:
            payload = deepcopy(dict(report))
            payload["producer_command"] = producer_commands[key]
            write_json(resolve_repo_path(configured_path), payload)


def _build_dedicated_reports(config_path: str | Path) -> dict[str, dict[str, object]]:
    config = load_config(config_path)
    _, audit_contract = _audit_policy(config)
    output_paths = _mapping(audit_contract.get("output_paths"))

    naming_inventory = resolve_repo_path(output_paths["g0_legacy_naming_inventory"])
    naming_report = build_naming_remediation_report(
        config,
        resolve_repo_path(str(config_path)),
        naming_inventory,
        naming_inventory,
        resolve_repo_path(output_paths["naming_residual"]),
        resolve_repo_path(output_paths["g0_naming_remediation"]),
        resolve_repo_path("Output/audits/g0_naming_remediation_report.md"),
    )

    graph_report, isolation_report, public_remediation = build_public_mainline_isolation_reports(config_path)
    public_paths = _mapping(_mapping(config.get("public_mainline_isolation")).get("output_paths"))
    write_json(resolve_repo_path(public_paths["config_graph"]), graph_report)
    write_json(resolve_repo_path(public_paths["isolation_audit"]), isolation_report)
    write_json(resolve_repo_path(public_paths["remediation_report"]), public_remediation)

    numeric_report = build_numeric_lineage_audit(config_path)
    numeric_blockers = validate_numeric_lineage_report(numeric_report, config_path)
    numeric_report["validator_blockers"] = numeric_blockers
    numeric_report["validator_blocker_count"] = len(numeric_blockers)
    write_numeric_lineage_artifacts(numeric_report, config_path)

    wording_report, wording_rows = build_claim_wording_audit(config_path)
    write_claim_wording_artifacts(wording_report, wording_rows, config_path)
    return {
        "naming": naming_report,
        "public_mainline": public_remediation,
        "numeric_lineage": numeric_report,
        "claim_wording": wording_report,
    }


def _disposition(
    blocker_id: str,
    report: Mapping[str, object],
    evidence_path: str,
) -> dict[str, object]:
    passed = False
    reason = ""
    metrics: dict[str, object] = {}
    if blocker_id == "G0-NAMING-LEGACY-RESIDUALS":
        metrics = {
            "forbidden_residual_count": report.get("forbidden_residual_count"),
            "forbidden_category_counts": report.get("forbidden_category_counts"),
        }
        passed = report.get("status") == "pass" and report.get("original_blocker_closed") is True and report.get("forbidden_residual_count") == 0
        reason = "Forbidden active naming residuals remain." if not passed else "Dedicated naming audit passed."
    elif blocker_id == "G0-PUBLIC-MAINLINE-CONTROLLED-INCLUDE":
        metrics = {
            "controlled_reachability_count": report.get("controlled_reachability_count"),
            "route_class_counts": report.get("route_class_counts"),
        }
        passed = report.get("status") == "pass" and report.get("original_blocker_closed") is True and report.get("controlled_reachability_count") == 0
        reason = "A public route reaches a controlled source." if not passed else "Dedicated public-mainline isolation audit passed."
    elif blocker_id == "G0-NUMERIC-LINEAGE-DRIFT":
        unresolved = [str(value) for value in report.get("unresolved_conflict_ids", [])]
        historical = _mapping(report.get("historical_snapshot"))
        independent_manifest = _mapping(historical.get("independent_manifest"))
        disposition = str(historical.get("disposition", ""))
        disposition_verified = historical.get("disposition_verified") is True
        metrics = {
            "unresolved_conflict_count": report.get("unresolved_conflict_count"),
            "unresolved_conflict_ids": unresolved,
            "independent_manifest_verified": independent_manifest.get("verified"),
            "historical_disposition": disposition,
            "historical_disposition_verified": disposition_verified,
            "canonical_metrics": [row.get("metric_name") for row in report.get("metrics", []) if isinstance(row, Mapping)],
        }
        passed = (
            report.get("status") == "pass"
            and not unresolved
            and report.get("unresolved_conflict_count") == 0
            and (
                independent_manifest.get("verified") is True
                or (
                    disposition == "abandoned_by_explicit_user_authorization"
                    and disposition_verified
                    and independent_manifest.get("verified") is False
                )
            )
        )
        reason = (
            "Numeric lineage is closed by explicit user-authorized abandonment of the unverifiable historical snapshot."
            if passed and disposition
            else "Numeric lineage has no unresolved conflict."
            if passed
            else "Numeric lineage remains fail-closed because its historical reconciliation evidence is incomplete."
        )
    elif blocker_id == "G0-CLAIM-GATED-WORDING":
        metrics = {"violation_count": report.get("violation_count"), "claim_statuses": report.get("claim_statuses")}
        passed = report.get("status") == "pass" and report.get("violation_count") == 0
        reason = "Pending public claims remain insufficiently qualified." if not passed else "Dedicated claim-wording audit passed."
    return {
        "blocker_id": blocker_id,
        "status": "closed" if passed else "blocked",
        "evidence_path": evidence_path,
        "reason": reason,
        "metrics": metrics,
    }


def build_g0_integrity_audit(config_path: str | Path = "Libs/configs/config_stack.yaml") -> dict[str, object]:
    config = load_config(config_path)
    union, audit_contract = _audit_policy(config)
    dedicated = _build_dedicated_reports(config_path)
    structural = _structural_reports(config_path)
    _write_structural_reports(config, structural, config_path)
    output_paths = _mapping(audit_contract.get("output_paths"))
    expected_schemas = _expected_dedicated_schemas(audit_contract)

    artifact_checks: dict[str, dict[str, object]] = {}
    expected_artifacts = {
        "naming": (output_paths["g0_naming_remediation"], expected_schemas["naming"]),
        "public_mainline": (output_paths["g0_public_mainline_remediation"], expected_schemas["public_mainline"]),
        "numeric_lineage": (output_paths["trajectory_numeric_lineage"], expected_schemas["numeric_lineage"]),
        "claim_wording": (output_paths["g0_claim_wording_remediation"], expected_schemas["claim_wording"]),
    }
    for name, (configured_path, expected_schema) in expected_artifacts.items():
        _, check = _artifact_record(resolve_repo_path(configured_path), str(expected_schema))
        artifact_checks[name] = check

    dispositions = [
        _disposition("G0-NAMING-LEGACY-RESIDUALS", dedicated["naming"], str(output_paths["g0_naming_remediation"])),
        _disposition("G0-PUBLIC-MAINLINE-CONTROLLED-INCLUDE", dedicated["public_mainline"], str(output_paths["g0_public_mainline_remediation"])),
        _disposition("G0-NUMERIC-LINEAGE-DRIFT", dedicated["numeric_lineage"], str(output_paths["trajectory_numeric_lineage"])),
        _disposition("G0-CLAIM-GATED-WORDING", dedicated["claim_wording"], str(output_paths["g0_claim_wording_remediation"])),
    ]
    remaining = [row["blocker_id"] for row in dispositions if row["status"] != "closed"]
    structural_checks = {
        name: {
            "status": report.get("status"),
            "blocker_count": report.get("blocker_count"),
            "blockers": report.get("blockers"),
        }
        for name, report in structural.items()
    }
    artifact_ownership = _artifact_ownership_check(union)
    claim_mapping, claim_mapping_errors = _claim_mapping(union)
    mandatory_input_resolutions = _mandatory_input_resolutions(audit_contract, union)
    structural_failures = [name for name, row in structural_checks.items() if row["status"] != "pass" or row["blocker_count"] != 0]
    invalid_artifacts = [name for name, row in artifact_checks.items() if row["status"] != "pass"]
    input_resolution_failures = [
        resolution_id
        for resolution_id, resolution in mandatory_input_resolutions.items()
        if resolution["status"] != "pass"
    ]
    integrity_blockers: list[dict[str, object]] = []
    for disposition in dispositions:
        if disposition["status"] != "closed":
            integrity_blockers.append(
                {
                    "blocker_id": disposition["blocker_id"],
                    "reason": disposition["reason"],
                    "evidence_path": disposition["evidence_path"],
                    "metrics": deepcopy(disposition["metrics"]),
                }
            )
    for name in structural_failures:
        integrity_blockers.append(
            {
                "blocker_id": "G0-STRUCTURAL-VALIDATION-FAILURE",
                "reason": "A mandatory structural validation report is not clean.",
                "validator": name,
                "evidence": structural_checks[name],
            }
        )
    for name in invalid_artifacts:
        integrity_blockers.append(
            {
                "blocker_id": "G0-ARTIFACT-SCHEMA-FAILURE",
                "reason": "A dedicated G0 artifact is missing or has the wrong schema version.",
                "artifact": name,
                "evidence": artifact_checks[name],
            }
        )
    if artifact_ownership["status"] != "pass":
        integrity_blockers.append(
            {
                "blocker_id": "G0-ARTIFACT-OWNERSHIP-FAILURE",
                "reason": "Validator artifact ownership is incomplete or duplicated.",
                "evidence": artifact_ownership,
            }
        )
    if claim_mapping_errors:
        integrity_blockers.append(
            {
                "blocker_id": "G0-CLAIM-MAPPING-FAILURE",
                "reason": "The immutable C1-C5 claim mapping is incomplete or ambiguous.",
                "evidence": claim_mapping_errors,
            }
        )
    for resolution_id in input_resolution_failures:
        integrity_blockers.append(
            {
                "blocker_id": "G0-MANDATORY-INPUT-RESOLUTION-FAILURE",
                "reason": "A mandatory G0 audit input lacks a valid schema-owned canonical replacement.",
                "resolution_id": resolution_id,
                "evidence": mandatory_input_resolutions[resolution_id],
            }
        )

    passed = not integrity_blockers
    return {
        "schema_version": str(_mapping(audit_contract.get("schema_versions")).get("g0_integrity", "neuronsifter-phase0-g0-integrity-audit-v1")),
        "status": "passed" if passed else "blocked",
        "owner": "ccf-integrity-auditor",
        "producer_command": f"python Libs/scripts/validators/validate_g0_integrity.py --config {config_path}",
        "git_commit": _git_commit(),
        "config_path": str(config_path),
        "config_hash": _sha256_file(resolve_repo_path(str(config_path))),
        "original_blocker_count": len(IMMUTABLE_BLOCKER_IDS),
        "immutable_blocker_ids": list(IMMUTABLE_BLOCKER_IDS),
        "blocker_dispositions": dispositions,
        "remaining_original_blocker_count": len(remaining),
        "remaining_original_blocker_ids": remaining,
        "blocker_count": len(integrity_blockers),
        "blockers": integrity_blockers,
        "dedicated_artifact_checks": artifact_checks,
        "structural_checks": structural_checks,
        "artifact_ownership": artifact_ownership,
        "mandatory_input_resolutions": mandatory_input_resolutions,
        "config_ownership": {
            "status": "pass" if structural_checks["config_stack"]["blocker_count"] == 0 else "blocked",
            "conflict_count": 0 if structural_checks["config_stack"]["blocker_count"] == 0 else None,
            "evidence": "config_stack",
        },
        "path_privacy": {
            "status": "pass" if dedicated["public_mainline"].get("controlled_reachability_count") == 0 else "blocked",
            "controlled_reachability_count": dedicated["public_mainline"].get("controlled_reachability_count"),
            "evidence": str(output_paths["g0_public_mainline_remediation"]),
        },
        "supersession": {
            "status": "pass" if structural_checks["contract_union"]["blocker_count"] == 0 else "blocked",
            "dependency_cycle_count": 0 if structural_checks["contract_union"]["blocker_count"] == 0 else None,
            "evidence": "contract_union",
        },
        "gate_recommendation": {
            "current_stage": "g0_complete" if passed else "g0_blocked",
            "g0_status": "passed" if passed else "blocked",
            "phase1_open": passed,
            "implementation_enabled": False,
            "next_owner": {"skill": "ccf-project-scaffolder"} if passed else None,
            "next_direction": {"id": "install-canonical-config-schema"} if passed else None,
        },
        "claim_mapping": claim_mapping,
    }


def write_g0_integrity_audit(report: Mapping[str, object], config_path: str | Path = "Libs/configs/config_stack.yaml") -> Path:
    config = load_config(config_path)
    _, audit_contract = _audit_policy(config)
    output_path = resolve_repo_path(_mapping(audit_contract.get("output_paths"))["g0_integrity"])
    write_json(output_path, report)
    return output_path
