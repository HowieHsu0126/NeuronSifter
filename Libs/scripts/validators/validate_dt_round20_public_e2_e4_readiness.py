#!/usr/bin/env python3
"""Materialize a fail-closed Round-20 Track-B evidence-readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RELATIVE = Path("Libs/configs/validation/round20_track_contracts.yaml")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def _artifact(repository: Path, relative: str) -> dict[str, Any]:
    path = repository / relative
    return {
        "path": relative,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _write(repository: Path, relative: str, payload: object) -> None:
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit(*, repository: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    track = config["round20_track_contracts"]["tracks"]["track_b_public_e2_e4_recovery"]
    blockers: list[dict[str, Any]] = []
    reused_authorities = []
    for item in track["reused_authorities"]:
        observed = _artifact(repository, str(item["path"]))
        reused_authorities.append({**observed, "expected_sha256": item["sha256"]})
        if not observed["exists"]:
            blockers.append({"code": "reused_authority_missing", "path": item["path"]})
        elif observed["sha256"] != item["sha256"]:
            blockers.append(
                {
                    "code": "reused_authority_hash_drift",
                    "path": item["path"],
                    "expected": item["sha256"],
                    "observed": observed["sha256"],
                }
            )

    e2_inputs = [
        "Output/evidence_fabric/round19/e2_candidate_catalog.parquet",
        "Output/evidence_fabric/round19/e2_admission_ledger.tsv",
        "Output/audits/dt_round19_e2_coverage.json",
        "Output/audits/dt_round19_e2_independence_audit.json",
    ]
    e4_inputs = [
        "Output/audits/dt_round19_intervention_producer_qualification.tsv",
        "Output/audits/dt_round19_e4_domain_selection.json",
        "Output/audits/dt_round19_e4_control_pair_audit.json",
        "Output/audits/dt_round19_e4_pilot_validation.json",
        "Output/evidence_fabric/round19/e4_intervention_trajectory_catalog.parquet",
    ]
    e2_inventory = [_artifact(repository, path) for path in e2_inputs]
    e4_inventory = [_artifact(repository, path) for path in e4_inputs]
    if not any(item["exists"] for item in e2_inventory):
        blockers.append({"code": "numeric_e2_artifacts_missing", "paths": e2_inputs})
    if not any(item["exists"] for item in e4_inventory):
        blockers.append({"code": "e4_producer_and_pilot_artifacts_missing", "paths": e4_inputs})
    blockers.extend(
        [
            {"code": "numeric_e2_minima_unverified", "required_fields": track["numeric_e2_required_fields"]},
            {"code": "intervention_action_semantics_unqualified"},
            {"code": "matched_control_pairs_not_executed"},
            {"code": "independent_validation_units_not_verified"},
        ]
    )

    e2_contract = {
        "schema_version": "neuronsifter-round20-e2-expansion-contract-v1",
        "status": "blocked_numeric_e2_not_verified",
        "priority_sources": track["priority_sources"],
        "required_fields": track["numeric_e2_required_fields"],
        "evidence_boundary": track["evidence_boundary"],
        "source_inventory": e2_inventory,
        "blockers": [item for item in blockers if "e2" in str(item.get("code", "")) or "numeric" in str(item.get("code", ""))],
    }
    e4_domain = {
        "schema_version": "neuronsifter-round20-e4-domain-selection-v1",
        "status": "blocked_no_evidence_aligned_domain_selected",
        "score_dimensions": track["e4_domain_score_dimensions"],
        "selected_domain": None,
        "scores": {dimension: "unresolved" for dimension in track["e4_domain_score_dimensions"]},
        "source_inventory": e4_inventory,
        "blockers": [
            {"code": "numeric_e2_minima_unverified"},
            {"code": "qualified_producer_missing"},
            {"code": "matched_control_pairs_not_executed"},
        ],
    }
    producer = {
        "schema_version": "neuronsifter-round20-e4-producer-qualification-v1",
        "status": "blocked_producer_not_qualified",
        "required_fields": track["producer_required_fields"],
        "producer_count": 0,
        "rows": [],
        "blockers": [{"code": "producer_qualification_artifact_missing"}, {"code": "action_semantics_incomplete"}],
    }
    controls = {
        "schema_version": "neuronsifter-round20-e4-control-pair-audit-v1",
        "status": "blocked_control_pairs_not_executed",
        "pilot_requirements": track["pilot_requirements"],
        "pair_count": 0,
        "pairs": [],
        "blockers": [{"code": "matched_control_pairs_not_executed"}, {"code": "intervention_control_pilot_execution_not_authorized"}],
    }
    readiness = {
        "schema_version": "neuronsifter-round20-public-e2-e4-training-readiness-v1",
        "artifact_uid": "dt_round20_public_e2_e4_training_readiness",
        "status": "blocked_insufficient_e4",
        "inherited_readiness": track["current_readiness"],
        "training_authorized": False,
        "training_started": False,
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
        "final_held_out_accessed": False,
        "e2": e2_contract,
        "e4_domain": e4_domain,
        "producer": producer,
        "control_pairs": controls,
        "reused_authorities": reused_authorities,
        "blockers": blockers,
        "next_action": "recover_independent_numeric_e2_and_qualify_an_action_complete_e4_producer_before_training_authorization",
    }
    outputs = {
        "e2": "Output/audits/dt_round20_e2_expansion_contract.json",
        "e4_domain": "Output/audits/dt_round20_e4_domain_selection.json",
        "producer": "Output/audits/dt_round20_e4_producer_qualification.json",
        "controls": "Output/audits/dt_round20_e4_control_pair_audit.json",
        "readiness": "Output/audits/dt_round20_public_e2_e4_training_readiness.json",
    }
    readiness["output_paths"] = outputs
    readiness["candidate_commit"] = _git(repository, "rev-parse", "HEAD")
    _write(repository, outputs["e2"], e2_contract)
    _write(repository, outputs["e4_domain"], e4_domain)
    _write(repository, outputs["producer"], producer)
    _write(repository, outputs["controls"], controls)
    _write(repository, outputs["readiness"], readiness)
    return readiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_RELATIVE)
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else repository / args.config
    report = audit(repository=repository, config_path=config_path.resolve())
    print(json.dumps({"status": report["status"], "blockers": len(report["blockers"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
