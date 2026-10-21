#!/usr/bin/env python3
"""Audit the Round-19A Allen inputs against the Round-20 Track-A contract.

This validator migrates metadata semantics only. It never retrieves payloads or
turns a historical source card into an admitted dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RELATIVE = Path("Libs/configs/validation/round20_track_contracts.yaml")
HISTORICAL_NEW_SCHEMAS = {
    "historical_contract": "neuronsifter-round20-track-a-historical-contract-v1",
    "historical_failed_gate": "neuronsifter-round20-historical-gate-evidence-v1",
    "historical_validation": "neuronsifter-round20-historical-validation-evidence-v1",
    "historical_handoff": "neuronsifter-round20-historical-handoff-evidence-v1",
    "historical_usage_inventory": "neuronsifter-round20-historical-usage-evidence-v1",
    "historical_source_cards": "neuronsifter-round20-source-card-migration-v1",
    "historical_failed_test_execution": "junit-xml-historical-evidence-v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load(path: Path) -> object:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".xml":
        return ET.parse(path).getroot()
    return {"byte_count": path.stat().st_size}


def _schema_identity(kind: str, payload: object) -> str:
    if isinstance(payload, Mapping):
        value = payload.get("schema_version")
        if isinstance(value, str) and value:
            return value
        keys = sorted(str(key) for key in payload)
        return f"mapping:{','.join(keys[:8])}"
    if isinstance(payload, ET.Element):
        return f"xml:{payload.tag}:{payload.attrib.get('tests', 'unknown')}"
    return f"{kind}:opaque"


def _projection(kind: str, payload: object) -> object:
    """Keep only the semantic fields needed for migration comparison."""
    if kind == "historical_source_cards" and isinstance(payload, Mapping):
        rows = []
        for card in payload.get("records", []):
            if not isinstance(card, Mapping):
                continue
            rows.append(
                {
                    "source_uid": card.get("source_uid"),
                    "release_version": card.get("release_version"),
                    "source_release_uid": card.get("source_release_uid"),
                    "modality": card.get("modality"),
                    "organism": card.get("organism"),
                    "train_eval_role": card.get("train_eval_role"),
                    "holdout_group": card.get("holdout_group"),
                    "materialization_status": card.get("materialization_status"),
                    "raw_hash_count": len(card.get("raw_hashes") or []),
                    "release_listing_count": len(card.get("release_listing_receipts") or []),
                }
            )
        return {"schema_version": payload.get("schema_version"), "records": rows}
    if isinstance(payload, ET.Element):
        suites = list(payload.iter("testsuite"))
        return {
            "tag": payload.tag,
            **{
                key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                for key in ("tests", "failures", "errors", "skipped")
            },
        }
    if isinstance(payload, Mapping):
        keys = ("schema_version", "artifact_kind", "status", "version", "source_families", "blockers")
        return {key: payload[key] for key in keys if key in payload}
    return payload


def audit(*, repository: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = config["round20_track_contracts"]
    track = contract["tracks"]["track_a_allen_full_landing"]
    blockers: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    cards_projection: Mapping[str, Any] | None = None

    for item in track["historical_inputs"]:
        kind = str(item["kind"])
        path = repository / str(item["path"])
        expected_hash = str(item["sha256"])
        if not path.is_file():
            blockers.append({"code": "historical_input_missing", "path": item["path"]})
            continue
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            blockers.append(
                {
                    "code": "historical_input_hash_drift",
                    "path": item["path"],
                    "expected": expected_hash,
                    "observed": observed_hash,
                }
            )
        payload = _load(path)
        projection = _projection(kind, payload)
        if kind == "historical_source_cards" and isinstance(projection, Mapping):
            cards_projection = projection
        decision = "migrated_compatible" if kind == "historical_source_cards" else "historical_only"
        semantic_diff = [
            "historical provenance remains immutable",
            "current release, terms, and payload identity are not inferred",
        ]
        if kind == "historical_source_cards":
            semantic_diff += [
                "source and observational roles map to the Round-20 family registry",
                "release identity, raw hashes, and admission remain unresolved",
            ]
        else:
            semantic_diff.append("artifact is retained as failure or provenance evidence, not a runtime input")
        records.append(
            {
                "path": str(item["path"]),
                "kind": kind,
                "historical_hash": expected_hash,
                "observed_hash": observed_hash,
                "historical_schema": _schema_identity(kind, payload),
                "new_schema": HISTORICAL_NEW_SCHEMAS[kind],
                "semantic_diff": semantic_diff,
                "migration_decision": decision,
                "new_hash": sha256_payload(projection),
                "new_hash_basis": "deterministic_semantic_projection",
            }
        )

    if cards_projection is None:
        blockers.append({"code": "source_cards_unavailable"})
    else:
        cards = cards_projection.get("records", [])
        family_uids = sorted(str(row.get("source_uid")) for row in cards)
        expected_uids = sorted(str(row["family_uid"]) for row in track["families"])
        if len(cards) != 12 or family_uids != expected_uids:
            blockers.append(
                {
                    "code": "source_family_mismatch",
                    "expected": expected_uids,
                    "observed": family_uids,
                }
            )
        if any(int(row.get("raw_hash_count", 0)) != 0 for row in cards):
            blockers.append({"code": "historical_payload_hash_present"})
        if any(row.get("source_release_uid") for row in cards):
            blockers.append({"code": "historical_release_identity_present"})

    gate_path = repository / "Output/audits/round20/historical_inputs/dt_round19a_allen_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "blocked_allen_license_or_terms":
        blockers.append({"code": "historical_gate_status_drift", "observed": gate.get("status")})

    junit_path = repository / "Output/audits/round20/historical_inputs/dt_round19a_allen_full_suite.junit.xml"
    root = ET.parse(junit_path).getroot()
    suites = list(root.iter("testsuite"))
    junit = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if junit != {"tests": 598, "failures": 17, "errors": 19, "skipped": 0}:
        blockers.append({"code": "historical_test_result_drift", "observed": junit})

    return {
        "schema_version": "neuronsifter-round20-allen-historical-migration-v1",
        "artifact_uid": "dt_round20_allen_historical_migration",
        "status": "passed_historical_migration_with_admission_blockers" if not blockers else "blocked_historical_migration",
        "candidate_commit": _git(repository, "rev-parse", "HEAD"),
        "source_contract_sha256": sha256_file(config_path),
        "historical_input_count": len(records),
        "records": records,
        "historical_failure_preservation": {
            "allen_gate": gate.get("status"),
            "junit": junit,
            "payload_families_materialized": 0,
            "training_started": False,
        },
        "admission_status": "blocked_pending_current_rights_release_and_payload_verification",
        "authorization": {
            "payload_download": False,
            "training": False,
            "evidence_fabric_sealing": False,
            "dt_core_evaluation": False,
        },
        "blockers": blockers,
        "next_action": "resolve_current_allen_terms_and_exact_releases_before_payload_retrieval",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_RELATIVE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else repository / args.config
    output = args.output or repository / "Output/audits/dt_round20_allen_historical_migration.json"
    output = output if output.is_absolute() else repository / output
    report = audit(repository=repository, config_path=config_path.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "blockers": len(report["blockers"]), "output": str(output)}, sort_keys=True))
    return 0 if report["status"].startswith("passed_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
