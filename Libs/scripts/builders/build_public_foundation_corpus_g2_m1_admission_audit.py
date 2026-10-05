"""Audit G2-M1 candidates without promoting any source or observation block."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping
import uuid


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import relative_repo_path, safe_repo_path, sha256_file
from Libs.utils.data.public_foundation_roles import load_phase1a_ledger
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json


AUDIT_SCHEMA_VERSION = "neuronsifter-public-foundation-corpus-g2-m1-admission-audit-v1"
PRODUCER = "Libs/scripts/builders/build_public_foundation_corpus_g2_m1_admission_audit.py"
CORE_FAMILIES = (
    "cns_disease_aging_state_transcriptomics",
    "cns_cell_type_region_atlas_transcriptomics",
    "compound_dose_time_perturbation_transcriptomics",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checksum_index(repo_root: Path, source_id: str) -> dict[str, dict[str, str]]:
    path = safe_repo_path(repo_root, f"Input/raw/{source_id}/checksums/sha256.tsv")
    if not path.is_file():
        return {}
    return {str(row["relative_path"]): row for row in read_tsv(path)}


def _binding(
    repo_root: Path,
    source_id: str,
    relative_path: str,
    checksum_index: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    declaration = checksum_index.get(relative_path, {})
    path = safe_repo_path(repo_root, f"Input/raw/{source_id}/{relative_path}")
    declared_size = int(declaration.get("size_bytes") or 0)
    exists = path.is_file()
    current_size = path.stat().st_size if exists else None
    return {
        "path": f"Input/raw/{source_id}/{relative_path}",
        "declared_sha256": str(declaration.get("sha256") or "0" * 64),
        "declared_size_bytes": declared_size,
        "exists": exists,
        "current_size_bytes": current_size,
        "size_matches_declared": exists and current_size == declared_size,
        "content_checksum_reverified": False,
    }


def _candidate_status(candidate: Mapping[str, object]) -> str:
    kind = str(candidate["candidate_kind"])
    if kind == "external_candidate_not_materialized":
        return "not_materialized"
    if kind in {"metadata_without_native_expression_matrix", "aggregate_prior_without_cell_axis"}:
        return "not_viable"
    return "candidate_audited"


def _candidate_blockers(candidate: Mapping[str, object], decision: str, audit_status: str) -> list[str]:
    blockers = ["source_requalification_required", "mainline_role_not_granted"]
    kind = str(candidate["candidate_kind"])
    if audit_status == "not_materialized":
        blockers.extend(["candidate_not_materialized", "official_release_snapshot_missing"])
    if kind == "metadata_without_native_expression_matrix":
        blockers.extend(["native_expression_value_matrix_missing", "context_prior_role_only"])
    elif kind == "aggregate_prior_without_cell_axis":
        blockers.extend(["aggregate_prior_not_native_observation", "cell_or_sample_axis_missing"])
    elif kind == "raw_processed_matrix_with_sample_metadata":
        blockers.extend([
            "migration_value_format_not_canonical",
            "biological_unit_policy_not_frozen",
            "grade_ab_qc_not_completed",
            "independent_source_pair_not_closed",
        ])
    elif kind == "dose_time_signature_matrix_with_neural_cell_line_subset":
        blockers.extend([
            "weak_calibration_role_only",
            "cns_context_not_qualified",
            "dose_time_canonicalization_pending",
            "independent_source_pair_not_closed",
            "occupancy_provenance_missing",
        ])
    if decision == "formally_excluded":
        blockers.append("source_formally_excluded")
    return sorted(set(blockers))


def build(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], dict[str, object]]:
    repo_root = repo_root.resolve()
    config = load_config(safe_repo_path(repo_root, "Libs/configs/config_stack.yaml"))
    policy = config["public_foundation_corpus"]["g2_m1_admission_audit"]
    ledger_path = safe_repo_path(repo_root, str(policy["source_ledger_path"]))
    ledger = load_phase1a_ledger(ledger_path)
    candidates: list[dict[str, object]] = []
    for configured in policy["candidates"]:
        source_id = str(configured["source_id"])
        decision = ledger[source_id]["decision"]
        checksum_index = _checksum_index(repo_root, source_id)
        audit_status = _candidate_status(configured)
        candidates.append(
            {
                "candidate_id": configured["candidate_id"],
                "family_id": configured["family_id"],
                "source_id": source_id,
                "source_accession": configured["source_accession"],
                "publication_accession_group": configured["publication_accession_group"],
                "mirror_group": configured["mirror_group"],
                "phase1a_decision": decision,
                "candidate_kind": configured["candidate_kind"],
                "audit_status": audit_status,
                "admission_allowed": False,
                "value_evidence": [
                    _binding(repo_root, source_id, str(path), checksum_index)
                    for path in configured["value_evidence_paths"]
                ],
                "metadata_evidence": [
                    _binding(repo_root, source_id, str(path), checksum_index)
                    for path in configured["metadata_evidence_paths"]
                ],
                "local_value_format": configured["local_value_format"],
                "local_sample_count": int(configured["local_sample_count"]),
                "biological_unit_axis_status": configured["biological_unit_axis_status"],
                "disease_region_axis_status": configured["disease_region_axis_status"],
                "required_next_evidence": list(configured["required_next_evidence"]),
                "blocker_codes": _candidate_blockers(configured, decision, audit_status),
            }
        )

    family_summary: dict[str, dict[str, object]] = {}
    for family_id in CORE_FAMILIES:
        rows = [row for row in candidates if row["family_id"] == family_id]
        blocker_codes = sorted({code for row in rows for code in row["blocker_codes"]})
        family_summary[family_id] = {
            "candidate_count": len(rows),
            "candidate_audited_count": sum(row["audit_status"] == "candidate_audited" for row in rows),
            "admission_allowed_count": 0,
            "status": "not_passed",
            "blocker_codes": blocker_codes or ["candidate_missing"],
        }

    transition_sources = sorted({row["source_id"] for row in candidates})
    request = {
        "schema_version": "neuronsifter-public-foundation-corpus-source-requalification-request-v1",
        "producer": PRODUCER,
        "created_at": _now(),
        "request_uid": f"pfc-requalification-{uuid.uuid4().hex}",
        "status": "required_before_role_transition",
        "phase1a_ledger_immutable": True,
        "source_role_transition_count": 0,
        "frozen_ledger": {
            "path": relative_repo_path(repo_root, ledger_path),
            "sha256": sha256_file(ledger_path),
        },
        "delta_artifact": {
            "path": "Output/audits/source_qualification_delta.tsv",
            "status": "not_created_without_new_evidence",
        },
        "required_artifacts": list(policy["required_requalification_artifacts"]),
        "requests": [
            {
                "source_id": source_id,
                "current_decision": ledger[source_id]["decision"],
                "requested_decision": "qualified_for_mainline",
                "transition_authorized": False,
                "required_evidence": list(policy["required_requalification_artifacts"]),
            }
            for source_id in transition_sources
        ],
    }

    request_path = safe_repo_path(repo_root, str(policy["request_output_path"]))
    write_json(request_path, request)
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "producer": PRODUCER,
        "created_at": _now(),
        "audit_uid": f"pfc-g2-m1-{uuid.uuid4().hex}",
        "status": "candidate_audit_only",
        "scope": config["public_foundation_corpus"]["scope"],
        "source_role_transition_count": 0,
        "source_ledger_binding": {
            "path": relative_repo_path(repo_root, ledger_path),
            "sha256": sha256_file(ledger_path),
            "record_count": len(ledger),
        },
        "candidates": candidates,
        "family_summary": family_summary,
        "requalification_request_binding": {
            "path": relative_repo_path(repo_root, request_path),
            "sha256": sha256_file(request_path),
            "record_count": len(request["requests"]),
        },
        "blocker_codes": sorted({code for row in candidates for code in row["blocker_codes"]}),
    }
    return audit, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit G2-M1 public corpus candidates without admission.")
    parser.add_argument(
        "--output",
        default="Output/audits/public_foundation_corpus_g2_m1_admission_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit, _ = build(REPO_ROOT)
    write_json(safe_repo_path(REPO_ROOT, args.output), audit)
    print(json.dumps({"audit_uid": audit["audit_uid"], "status": audit["status"], "candidate_count": len(audit["candidates"])}, sort_keys=True))


if __name__ == "__main__":
    main()
