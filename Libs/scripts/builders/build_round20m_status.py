"""Build progressive Round-20M materialization and readiness summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json
from Libs.scripts.validators.validate_c1_rescue_evidence import (
    validate_rescue_payload,
)
from Libs.scripts.validators.validate_active_contract_migration import (  # noqa: E402
    validate_payload as validate_active_contract_migration_payload,
)
from Libs.scripts.validators.validate_restricted_c2_baseline_qualification import (  # noqa: E402
    validate_payload as validate_restricted_c2_baseline_qualification_payload,
)


def _json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {} if default is None else default
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _junit_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "test_count": 0, "failures": 0, "errors": 1, "skips": 0}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {"status": "invalid", "test_count": 0, "failures": 0, "errors": 1, "skips": 0}
    cases = list(root.iter("testcase"))
    failures = sum(1 for case in cases if case.find("failure") is not None)
    errors = sum(1 for case in cases if case.find("error") is not None)
    skips = sum(1 for case in cases if case.find("skipped") is not None)
    return {
        "status": "passed" if cases and not failures and not errors and not skips else "not_passed",
        "test_count": len(cases),
        "failures": failures,
        "errors": errors,
        "skips": skips,
        "path": str(path),
    }


def _training_status(repo_root: Path) -> dict[str, Any]:
    """Separate development smoke runs from the prohibited headline run."""

    candidates = (
        repo_root / "Output/audits/c1_rescue_preflight.json",
        repo_root / "Output/audits/c1_unified_preflight.json",
    )
    development_started = False
    headline_started = False
    sources: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        payload = _json(path)
        sources.append(path.relative_to(repo_root).as_posix())
        development_started = development_started or payload.get("training_started") is True
        headline_started = headline_started or payload.get("headline_training_started") is True
    return {
        # The Round-20 safety alias concerns headline/formal training only.
        "training_started": headline_started,
        "headline_training_started": headline_started,
        "development_training_started": development_started,
        "status_sources": sources,
        "training_scope": "headline_or_formal_only",
    }


def _rescue_bundle_status(repo_root: Path) -> dict[str, Any]:
    bundle_path = repo_root / "Output/audits/public_foundation_corpus_rescue_evidence_bundle.json"
    if not bundle_path.is_file():
        return {"resolved": False, "status": "missing", "blockers": ["bundle_missing"]}
    bundle = _json(bundle_path)
    blockers: list[str] = []
    if bundle.get("corpus_uid") != "public_foundation_corpus_rescue":
        blockers.append("corpus_uid_mismatch")
    if bundle.get("status") != "valid_fail_closed":
        blockers.append("bundle_status_not_valid_fail_closed")
    if bundle.get("blockers") != []:
        blockers.append("bundle_blockers_present")
    evidence = bundle.get("evidence")
    required_evidence = {
        "source_qualification",
        "crosswalk_delta",
        "canonical_observation_blocks",
        "structured_action_catalog",
        "qualified_action_subset",
        "frozen_splits",
        "leakage_audit",
        "qc_ess_statistics",
        "storage_ledger",
        "controlled_isolation",
        "parent_v1_evidence_bundle",
    }
    if not isinstance(evidence, dict):
        blockers.append("evidence_map_missing")
    else:
        for key in sorted(required_evidence):
            reference = evidence.get(key)
            if not isinstance(reference, dict):
                blockers.append(f"evidence_reference_missing:{key}")
                continue
            relative = Path(str(reference.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                blockers.append(f"evidence_reference_unsafe:{key}")
                continue
            target = repo_root / relative
            if not target.is_file():
                blockers.append(f"evidence_reference_unavailable:{key}")
            elif key == "controlled_isolation":
                # The semantic validator binds the frozen C1 snapshot and only
                # accepts a pass-state, zero-reachability live supersession.
                continue
            elif _sha256_file(target) != str(reference.get("sha256", "")):
                blockers.append(f"evidence_reference_hash_mismatch:{key}")
    binding = bundle.get("c1_observation_binding")
    if not isinstance(binding, dict):
        blockers.append("observation_binding_missing")
    else:
        for key in ("manifest", "row_table"):
            reference = binding.get(key)
            if not isinstance(reference, dict):
                blockers.append(f"observation_binding_reference_missing:{key}")
                continue
            relative = Path(str(reference.get("path", "")))
            target = repo_root / relative
            if relative.is_absolute() or ".." in relative.parts or not target.is_file():
                blockers.append(f"observation_binding_reference_unavailable:{key}")
            elif _sha256_file(target) != str(reference.get("sha256", "")):
                blockers.append(f"observation_binding_hash_mismatch:{key}")
    validation_path = repo_root / "Output/audits/public_foundation_corpus_rescue_evidence_validation.json"
    validation = _json(validation_path)
    recomputed_validation: dict[str, Any] = {}
    if not validation_path.is_file():
        blockers.append("semantic_validation_missing")
    else:
        try:
            recomputed_validation = validate_rescue_payload(
                bundle,
                repo_root,
                verify_large_block_payloads=False,
            )
        except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
            blockers.append(f"semantic_validation_recompute_failed:{exc}")
        if recomputed_validation.get("status") != "valid_fail_closed":
            for row in recomputed_validation.get("blockers", []):
                if isinstance(row, dict):
                    blockers.append(
                        f"semantic_validation_recomputed:{row.get('code', 'unknown')}"
                    )
            if not recomputed_validation:
                blockers.append("semantic_validation_recompute_missing")
        if validation.get("status") != "valid_fail_closed":
            blockers.append("semantic_validation_not_valid_fail_closed")
        if validation.get("corpus_identity_hash") != bundle.get("corpus_hash"):
            blockers.append("semantic_validation_identity_mismatch")
        if validation.get("bundle_path") != "Output/audits/public_foundation_corpus_rescue_evidence_bundle.json":
            blockers.append("semantic_validation_bundle_path_mismatch")
        if validation.get("bundle_sha256") != _sha256_file(bundle_path):
            blockers.append("semantic_validation_bundle_hash_mismatch")
        migration = validation.get("semantic_migration")
        recomputed_migration = recomputed_validation.get("semantic_migration")
        required_migration_fields = (
            "status",
            "migration_type",
            "historical_bundle_path",
            "historical_bundle_sha256",
            "historical_corpus_hash",
            "current_corpus_hash",
            "same_parent_identity",
            "same_block_content_identity_multiset",
            "same_split_role_content_membership",
            "current_split_membership_hash",
            "historical_split_membership_hash",
            "historical_bundle_used_as_provenance_only",
        )
        if not isinstance(migration, dict) or not isinstance(recomputed_migration, dict):
            blockers.append("semantic_migration_binding_missing")
        else:
            missing_migration_fields = [
                field
                for field in required_migration_fields
                if field not in migration or field not in recomputed_migration
            ]
            if missing_migration_fields:
                blockers.append("semantic_migration_binding_missing")
            for field in required_migration_fields:
                if field in missing_migration_fields:
                    continue
                if migration.get(field) != recomputed_migration.get(field):
                    blockers.append(f"semantic_migration_binding_mismatch:{field}")
            if migration.get("current_corpus_hash") != bundle.get("corpus_hash"):
                blockers.append("semantic_migration_current_identity_unbound")
            if migration.get("historical_bundle_path") != "Output/audits/public_foundation_corpus_v2_evidence_bundle.json":
                blockers.append("semantic_migration_historical_reference_unbound")
            if migration.get("same_split_role_content_membership") is not True:
                blockers.append("semantic_migration_split_membership_unverified")
        if not isinstance(migration, dict) or migration.get("status") != "verified":
            blockers.append("semantic_migration_unverified")
    return {
        "resolved": not blockers,
        "status": bundle.get("status", "unknown"),
        "corpus_hash": bundle.get("corpus_hash"),
        "blockers": blockers,
        "path": str(bundle_path.relative_to(repo_root).as_posix()),
        "validation_path": str(validation_path.relative_to(repo_root).as_posix()),
        "validation_status": validation.get("status", "missing"),
    }


def _active_contract_migration_status(repo_root: Path) -> dict[str, Any]:
    audit_path = repo_root / "Output/audits/active_contract_migration.json"
    validation_path = repo_root / "Output/audits/active_contract_migration_validation.json"
    if not audit_path.is_file():
        return {
            "valid": False,
            "migration_binding_valid": False,
            "execution_artifacts_ready": False,
            "status": "missing",
            "blockers": ["active_contract_migration_audit_missing"],
        }
    payload = _json(audit_path)
    blockers: list[str] = []
    try:
        recomputed = validate_active_contract_migration_payload(payload, repo_root)
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        recomputed = {}
        blockers.append(f"active_contract_migration_recompute_failed:{exc}")
    if recomputed.get("status") != "valid_fail_closed":
        blockers.append("active_contract_migration_recompute_invalid")
    if not validation_path.is_file():
        blockers.append("active_contract_migration_validation_missing")
        validation = {}
    else:
        validation = _json(validation_path)
        if validation.get("status") != "valid_fail_closed":
            blockers.append("active_contract_migration_validation_invalid")
        if validation.get("audit_file_sha256") != _sha256_file(audit_path):
            blockers.append("active_contract_migration_validation_hash_mismatch")
        if validation.get("migration_binding_valid") != payload.get("migration_binding_valid"):
            blockers.append("active_contract_migration_binding_mismatch")
        if validation.get("execution_artifacts_ready") != payload.get("execution_artifacts_ready"):
            blockers.append("active_contract_migration_execution_state_mismatch")
    return {
        "valid": not blockers,
        "migration_binding_valid": payload.get("migration_binding_valid") is True,
        "execution_artifacts_ready": payload.get("execution_artifacts_ready") is True,
        "status": payload.get("status", "missing"),
        "blockers": blockers,
        "path": str(audit_path.relative_to(repo_root).as_posix()),
        "validation_path": str(validation_path.relative_to(repo_root).as_posix()),
        "validation_status": validation.get("status", "missing"),
    }


def _c2_integrity_status(repo_root: Path, readiness: dict[str, Any]) -> dict[str, Any]:
    path = repo_root / "Output/audits/c2/current_clamp/integrity_audit.json"
    audit = _json(path)
    producer = readiness.get("producer") if isinstance(readiness.get("producer"), dict) else {}
    expected_hash = producer.get("trajectory_store_sha256") if isinstance(producer, dict) else None
    valid = (
        audit.get("status") == "passed"
        and audit.get("store_hash") == expected_hash
        and audit.get("store_hash_matches_readiness") is True
        and audit.get("project_reader_status") == "passed"
        and audit.get("zarr_three_reader_status") == "passed"
        and audit.get("strict_split_overlap") == 0
    )
    return {
        "valid": valid,
        "status": audit.get("status", "missing"),
        "path": str(path.relative_to(repo_root).as_posix()),
        "store_hash": audit.get("store_hash"),
    }


def _restricted_c2_adapter_status(repo_root: Path) -> dict[str, Any]:
    artifact_path = repo_root / "Output/audits/c2/current_clamp/baseline_qualification.json"
    schema_path = repo_root / "Libs/configs/schema/restricted_c2_baseline_qualification.schema.json"
    if not artifact_path.is_file():
        return {
            "status": "missing",
            "qualified": False,
            "qualified_adapter_count": 0,
            "formal_challenge_started": False,
            "path": str(artifact_path.relative_to(repo_root).as_posix()),
            "blockers": ["adapter_qualification_artifact_missing"],
        }
    try:
        payload = _json(artifact_path)
        schema = _json(schema_path)
        validation = validate_restricted_c2_baseline_qualification_payload(payload, schema, repo_root)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "qualified": False,
            "qualified_adapter_count": 0,
            "formal_challenge_started": False,
            "path": str(artifact_path.relative_to(repo_root).as_posix()),
            "blockers": [f"adapter_qualification_validation_error:{exc}"],
        }
    return {
        "status": validation.get("status", "failed"),
        "qualified": validation.get("status") == "passed",
        "qualified_adapter_count": int(validation.get("qualified_adapter_count", 0)),
        "formal_challenge_started": payload.get("formal_challenge_started") is True,
        "path": str(artifact_path.relative_to(repo_root).as_posix()),
        "blockers": validation.get("blockers", []),
    }


def _family_status(repo_root: Path, family_uid: str) -> dict[str, Any]:
    root = repo_root / "Output/audits/allen/materialization" / family_uid
    processed_root = repo_root / "Input/processed/public_foundation/allen" / family_uid
    integrity = _json(root / "integrity.json")
    snapshot = _json(root / "snapshot_identity.json")
    snapshot_uid = str(snapshot.get("snapshot_uid", ""))
    processed = processed_root / snapshot_uid if snapshot_uid else processed_root / "__missing_snapshot__"
    parser = _json(processed / "parser_validation.json")
    canonicalization = _json(processed / "canonicalization_summary.json")
    qc = _json(processed / "qc_summary.json")
    consumer = _json(processed / "consumer_execution.json")
    downloaded = bool(integrity.get("real_payload_bytes")) and bool(integrity.get("expected_objects_accounted"))
    hash_verified = downloaded and bool(integrity.get("downloaded_objects_hash_verified")) and integrity.get("checksum_mismatch") == 0
    return {
        "family_uid": family_uid,
        "authorized": True,
        "downloaded": downloaded,
        "hash_verified": hash_verified,
        "parsed": hash_verified and parser.get("status") == "passed",
        "canonicalized": parser.get("status") == "passed" and canonicalization.get("status") == "passed" and int(canonicalization.get("canonical_record_count", 0)) > 0,
        "qc": canonicalization.get("status") == "passed" and qc.get("status") == "passed",
        "consumed": qc.get("status") == "passed" and consumer.get("status") == "completed",
        "integrity_path": str(root.relative_to(repo_root).as_posix()) if root.is_dir() else "",
    }


def build(repo_root: Path, output_root: Path) -> dict[str, Any]:
    training = _training_status(repo_root)
    graph = _tsv(repo_root / "Output/audits/allen_authority_graph.tsv")
    family_uids = sorted(row["family_uid"] for row in graph)
    family_states = [_family_status(repo_root, family_uid) for family_uid in family_uids]
    allen = {
        "configured": len(family_uids),
        "authorized": sum(state["authorized"] for state in family_states),
        "downloaded": sum(state["downloaded"] for state in family_states),
        "hash_verified": sum(state["hash_verified"] for state in family_states),
        "parsed": sum(state["parsed"] for state in family_states),
        "canonicalized": sum(state["canonicalized"] for state in family_states),
        "qc": sum(state["qc"] for state in family_states),
        "consumed": sum(state["consumed"] for state in family_states),
        "blocked": [state["family_uid"] for state in family_states if not state["downloaded"]],
        "families": family_states,
        **training,
        "evidence_fabric_sealed": False,
        "dt_core_evaluated": False,
    }
    c2 = _json(repo_root / "Output/audits/c2_domain_readiness.json")
    c2_integrity = _c2_integrity_status(repo_root, c2)
    c2_adapter_qualification = _restricted_c2_adapter_status(repo_root)
    c2_status = {
        "action_complete_domain": bool(c2.get("action_complete")),
        "matched_controls": bool(c2.get("matched_controls_complete")),
        "producer_qualified": bool(c2.get("qualified_producer")),
        "strict_split_overlap": c2.get("strict_split_overlap"),
        "status": c2.get("status", "missing_artifact"),
        "selected_domain": c2.get("selected_domain"),
        "next_action": c2.get("next_action", "recover_action_complete_public_domain"),
        "integrity": c2_integrity,
    }
    baseline_rows = _tsv(repo_root / "Output/audits/baseline_source_license_matrix.tsv")
    conditional = _json(repo_root / "Output/audits/baseline_conditional_activation.json")
    baseline_status = {
        "mandatory_source_license_decisions": len(baseline_rows),
        "mandatory_qualification_ready": sum(row.get("qualification_status") == "qualified" for row in baseline_rows),
        "restricted_c2_simple_adapter_qualification": c2_adapter_qualification,
        "conditional_decisions": int(conditional.get("conditional_count", 0)),
        "conditional_test_access_allowed": bool(conditional.get("test_access_allowed")),
        "simple_challenge_ready": False,
        "formal_challenge_started": False,
        **training,
    }
    rescue_bundle = _rescue_bundle_status(repo_root)
    active_contract_migration = _active_contract_migration_status(repo_root)
    focused = _junit_status(repo_root / "Output/audits/round20m_targeted_regression.xml")
    full_suite = _json(repo_root / "Output/audits/mainline_full_suite_execution.json")
    full_suite_green = (
        full_suite.get("full_suite_green") is True
        and int(full_suite.get("full_suite_failures", 1)) == 0
        and int(full_suite.get("full_suite_errors", 1)) == 0
        and int(full_suite.get("mandatory_skips", 1)) == 0
        and full_suite.get("terminal_execution") == "completed"
    )
    validation = {
        "current_rescue_path": rescue_bundle.get("path", "Output/audits/public_foundation_corpus_rescue_evidence_bundle.json"),
        "current_rescue_resolved": rescue_bundle["resolved"],
        "current_rescue": rescue_bundle,
        "active_contract_migration": active_contract_migration,
        "focused_authority_tests": focused["status"],
        "focused_test_summary": focused,
        "full_suite_green": full_suite_green,
        "full_suite_execution": full_suite,
        "controlled_access_count": 0,
        "evidence_fabric_sealed": False,
        "dt_core_evaluated": False,
        **training,
    }
    c2_ready = (
        c2_status["action_complete_domain"]
        and c2_status["matched_controls"]
        and c2_status["producer_qualified"]
        and c2_status["strict_split_overlap"] == 0
        and c2_integrity["valid"]
    )
    if allen["consumed"] < allen["configured"]:
        next_action = "complete_remaining_authorized_family_materialization"
    elif not c2_ready:
        next_action = "recover_action_complete_public_domain"
    elif not c2_adapter_qualification["qualified"]:
        next_action = "qualify_restricted_c2_simple_adapters"
    elif not active_contract_migration["migration_binding_valid"] or not active_contract_migration["execution_artifacts_ready"]:
        next_action = "repair_current_c1_contract_artifacts_from_live_evidence"
    elif not baseline_status["simple_challenge_ready"]:
        next_action = "qualify_external_baselines_before_c1_c2_challenge"
    elif not validation["current_rescue_resolved"] or not validation["full_suite_green"]:
        next_action = "run_exact_clean_full_suite_after_current_rescue_validation"
    else:
        next_action = "claim_scoped_training_authorization_review"
    gate_ok = (
        allen["authorized"] == 12 and allen["consumed"] == 12 and
        c2_ready and
        baseline_status["mandatory_source_license_decisions"] == 20 and baseline_status["conditional_decisions"] == 5 and baseline_status["simple_challenge_ready"] and
        validation["current_rescue_resolved"] and active_contract_migration["valid"] and active_contract_migration["execution_artifacts_ready"] and validation["focused_authority_tests"] == "passed" and validation["full_suite_green"] and
        validation["controlled_access_count"] == 0 and not validation["training_started"] and not validation["evidence_fabric_sealed"] and not validation["dt_core_evaluated"]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "round20m_allen_materialization_status.json", {"schema_version": "neuronsifter-round20m-allen-materialization-status", "status": "complete" if allen["consumed"] == 12 else "progressive", "allen": allen})
    write_json(output_root / "round20m_c2_recovery_status.json", {"schema_version": "neuronsifter-round20m-c2-recovery-status", "status": c2_status["status"], "c2": c2_status})
    write_json(output_root / "round20m_baseline_qualification_status.json", {"schema_version": "neuronsifter-round20m-baseline-qualification-status", "status": "ready" if baseline_status["simple_challenge_ready"] else "blocked", "baseline": baseline_status})
    write_json(output_root / "round20m_validation.json", {"schema_version": "neuronsifter-round20m-validation", "status": "blocked" if not validation["full_suite_green"] else "passed", "validation": validation})
    write_json(output_root / "round20m_gate.json", {"schema_version": "neuronsifter-round20m-gate", "gate": "round20_public_materialization_and_baseline_readiness_resolved" if gate_ok else "round20_public_materialization_and_baseline_readiness_blocked", "allen": allen, "c2": c2_status, "baseline": baseline_status, "validation": validation, "next_action": next_action})
    return {"allen": allen, "c2": c2_status, "baseline": baseline_status, "validation": validation, "gate_ok": gate_ok}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build progressive Round-20M status artifacts.")
    parser.add_argument("--output-root", default="Output/audits")
    args = parser.parse_args()
    result = build(REPO_ROOT, (REPO_ROOT / args.output_root).resolve())
    print(json.dumps({"gate_ok": result["gate_ok"], "allen": {key: result["allen"][key] for key in ("configured", "authorized", "downloaded", "hash_verified", "parsed", "canonicalized", "qc", "consumed")}}, sort_keys=True))


if __name__ == "__main__":
    main()
