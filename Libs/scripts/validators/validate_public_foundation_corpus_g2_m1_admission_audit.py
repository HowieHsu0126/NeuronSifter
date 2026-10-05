"""Validate that G2-M1 audit output cannot promote a source or value block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

from jsonschema import Draft202012Validator, FormatChecker


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file
from Libs.utils.data.public_foundation_roles import load_phase1a_ledger
from Libs.utils.data.raw_data_registry import load_config, write_json
from Libs.scripts.builders.build_public_foundation_corpus_g2_m1_admission_audit import (
    _binding,
    _candidate_blockers,
    _candidate_status,
    _checksum_index,
)


DEFAULT_AUDIT = "Output/audits/public_foundation_corpus_g2_m1_admission_audit.json"
DEFAULT_OUTPUT = "Output/audits/public_foundation_corpus_g2_m1_admission_validation.json"


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _blocker(
    blockers: list[dict[str, str]],
    code: str,
    path: str,
    message: str,
) -> None:
    blockers.append({"code": code, "path": path, "message": message})


def _configured_evidence_bindings(
    repo_root: Path,
    configured: Mapping[str, object],
    evidence_kind: str,
) -> list[dict[str, object]]:
    source_id = str(configured["source_id"])
    checksum_index = _checksum_index(repo_root, source_id)
    configured_paths = configured[f"{evidence_kind}_paths"]
    if not isinstance(configured_paths, list):
        raise TypeError(f"{evidence_kind}_paths must be a list")
    return [
        _binding(repo_root, source_id, str(relative_path), checksum_index)
        for relative_path in configured_paths
    ]


def validate_payload(
    payload: object,
    *,
    repo_root: Path = REPO_ROOT,
    policy: Mapping[str, object],
    schema: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    for error in errors:
        blockers.append({"code": "json_schema_violation", "path": ".".join(map(str, error.absolute_path)) or "$", "message": error.message})
    if errors or not isinstance(payload, dict):
        return {"status": "blocked", "blocker_count": len(blockers), "blockers": blockers}

    ledger_path = safe_repo_path(repo_root, str(policy["source_ledger_path"]))
    ledger = load_phase1a_ledger(ledger_path)
    binding = payload["source_ledger_binding"]
    if (
        binding["path"] != str(policy["source_ledger_path"])
        or binding["sha256"] != sha256_file(ledger_path)
        or binding["record_count"] != len(ledger)
    ):
        _blocker(
            blockers,
            "frozen_ledger_binding_mismatch",
            "source_ledger_binding",
            "Audit is not bound to the current frozen Phase 1A ledger.",
        )
    if payload["source_role_transition_count"] != 0:
        _blocker(
            blockers,
            "source_role_transition_detected",
            "source_role_transition_count",
            "G2-M1 audit must not transition a source role.",
        )

    configured = {str(row["candidate_id"]): row for row in policy["candidates"]}
    candidates = payload["candidates"]
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    if (
        len(candidates) != len(configured)
        or len(candidate_ids) != len(set(candidate_ids))
        or set(candidate_ids) != set(configured)
    ):
        _blocker(
            blockers,
            "candidate_set_mismatch",
            "candidates",
            "Audit candidate set differs from frozen policy.",
        )
    for row in candidates:
        candidate_id = str(row["candidate_id"])
        expected = configured.get(candidate_id)
        if expected is None:
            continue
        source_id = str(row["source_id"])
        identity_fields = (
            "source_id",
            "family_id",
            "source_accession",
            "publication_accession_group",
            "mirror_group",
            "candidate_kind",
            "local_value_format",
            "local_sample_count",
            "biological_unit_axis_status",
            "disease_region_axis_status",
        )
        if any(row[field] != expected[field] for field in identity_fields):
            _blocker(
                blockers,
                "candidate_identity_mismatch",
                f"candidates.{candidate_id}",
                "Candidate identity or local evidence metadata differs from policy.",
            )
        if row["required_next_evidence"] != expected["required_next_evidence"]:
            _blocker(
                blockers,
                "candidate_required_evidence_mismatch",
                f"candidates.{candidate_id}.required_next_evidence",
                "Candidate requalification requirements differ from policy.",
            )
        frozen_decision = ledger.get(source_id, {}).get("decision")
        if frozen_decision is None:
            _blocker(
                blockers,
                "candidate_source_not_in_frozen_ledger",
                f"candidates.{candidate_id}.source_id",
                "Candidate source is absent from the frozen Phase 1A ledger.",
            )
            continue
        if row["phase1a_decision"] != frozen_decision:
            _blocker(
                blockers,
                "candidate_phase1a_decision_mismatch",
                f"candidates.{candidate_id}.phase1a_decision",
                "Candidate decision differs from frozen ledger.",
            )
        if row["admission_allowed"] is not False:
            _blocker(
                blockers,
                "candidate_admission_promoted",
                f"candidates.{candidate_id}.admission_allowed",
                "G2-M1 audit cannot authorize admission.",
            )
        expected_status = _candidate_status(expected)
        if row["audit_status"] != expected_status:
            _blocker(
                blockers,
                "candidate_status_derivation_mismatch",
                f"candidates.{candidate_id}.audit_status",
                "Candidate status is not derived from the frozen candidate kind.",
            )
        expected_blockers = _candidate_blockers(expected, frozen_decision, expected_status)
        if row["blocker_codes"] != expected_blockers:
            _blocker(
                blockers,
                "candidate_blocker_derivation_mismatch",
                f"candidates.{candidate_id}.blocker_codes",
                "Candidate blockers are not derived from its evidence class and frozen role.",
            )
        for evidence_kind in ("value_evidence", "metadata_evidence"):
            expected_bindings = _configured_evidence_bindings(
                repo_root,
                expected,
                evidence_kind,
            )
            if row[evidence_kind] != expected_bindings:
                _blocker(
                    blockers,
                    "candidate_evidence_binding_mismatch",
                    f"candidates.{candidate_id}.{evidence_kind}",
                    "Evidence bindings do not match the frozen policy and current local files.",
                )

    if set(payload["family_summary"]) != {
        "cns_disease_aging_state_transcriptomics",
        "cns_cell_type_region_atlas_transcriptomics",
        "compound_dose_time_perturbation_transcriptomics",
    }:
        _blocker(
            blockers,
            "family_summary_set_mismatch",
            "family_summary",
            "Audit must summarize every locked G2 transcriptomics family.",
        )
    for family_id, summary in payload["family_summary"].items():
        rows = [row for row in candidates if row["family_id"] == family_id]
        expected_family_blockers = sorted(
            {code for row in rows for code in row["blocker_codes"]}
        ) or ["candidate_missing"]
        if (
            summary["candidate_count"] != len(rows)
            or summary["candidate_audited_count"]
            != sum(row["audit_status"] == "candidate_audited" for row in rows)
            or summary["admission_allowed_count"] != 0
            or summary["status"] != "not_passed"
            or summary["blocker_codes"] != expected_family_blockers
        ):
            _blocker(
                blockers,
                "family_summary_derivation_mismatch",
                f"family_summary.{family_id}",
                "Family summary is not an audit-only derivation.",
            )
    request_binding = payload["requalification_request_binding"]
    request_path = safe_repo_path(repo_root, str(request_binding["path"]))
    requested_sources = sorted({str(row["source_id"]) for row in candidates})
    if (
        request_binding["path"] != str(policy["request_output_path"])
        or request_binding["record_count"] != len(requested_sources)
        or not request_path.is_file()
        or request_binding["sha256"] != sha256_file(request_path)
    ):
        _blocker(
            blockers,
            "requalification_request_binding_mismatch",
            "requalification_request_binding",
            "Requalification request binding is invalid.",
        )
    else:
        request = _load_json(request_path)
        if not isinstance(request, dict):
            _blocker(
                blockers,
                "requalification_request_transition_invalid",
                "requalification_request_binding",
                "Requalification request must be a JSON object.",
            )
        else:
            frozen_ledger = request.get("frozen_ledger")
            requests = request.get("requests")
            request_rows = requests if isinstance(requests, list) else []
            request_ids = [str(row.get("source_id", "")) for row in request_rows if isinstance(row, dict)]
            valid_request = (
                request.get("schema_version")
                == "neuronsifter-public-foundation-corpus-source-requalification-request-v1"
                and request.get("producer")
                == "Libs/scripts/builders/build_public_foundation_corpus_g2_m1_admission_audit.py"
                and request.get("status") == "required_before_role_transition"
                and request.get("phase1a_ledger_immutable") is True
                and request.get("source_role_transition_count") == 0
                and frozen_ledger == {
                    "path": str(policy["source_ledger_path"]),
                    "sha256": sha256_file(ledger_path),
                }
                and request.get("delta_artifact")
                == {
                    "path": "Output/audits/source_qualification_delta.tsv",
                    "status": "not_created_without_new_evidence",
                }
                and request.get("required_artifacts")
                == policy["required_requalification_artifacts"]
                and request_ids == requested_sources
                and len(request_rows) == len(requested_sources)
                and all(
                    isinstance(row, dict)
                    and row.get("current_decision") == ledger[str(row.get("source_id", ""))]["decision"]
                    and row.get("requested_decision") == "qualified_for_mainline"
                    and row.get("transition_authorized") is False
                    and row.get("required_evidence")
                    == policy["required_requalification_artifacts"]
                    for row in request_rows
                    if str(row.get("source_id", "")) in ledger
                )
                and all(str(row.get("source_id", "")) in ledger for row in request_rows if isinstance(row, dict))
            )
            if not valid_request:
                _blocker(
                    blockers,
                    "requalification_request_transition_invalid",
                    "requalification_request_binding",
                    "Request must preserve ledger immutability, frozen evidence requirements, and zero transitions.",
                )
    return {"status": "passed" if not blockers else "blocked", "blocker_count": len(blockers), "blockers": blockers}


def build_report(audit_path: str = DEFAULT_AUDIT, *, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    config = load_config(safe_repo_path(repo_root, "Libs/configs/config_stack.yaml"))
    policy = config["public_foundation_corpus"]["g2_m1_admission_audit"]
    schema = _load_json(safe_repo_path(repo_root, str(policy["schema_path"])))
    if not isinstance(schema, dict):
        raise TypeError("G2-M1 admission audit schema must be a JSON object")
    Draft202012Validator.check_schema(schema)
    payload = _load_json(safe_repo_path(repo_root, audit_path))
    return validate_payload(payload, repo_root=repo_root, policy=policy, schema=schema)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the G2-M1 candidate admission audit.")
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.audit)
    write_json(safe_repo_path(REPO_ROOT, args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
