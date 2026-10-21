#!/usr/bin/env python3
"""Validate Round-19 public E2/E4 expansion and issue the canonical Gate."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.evidence_fabric_v0 import read_parquet_records  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json  # noqa: E402
from Libs.utils.data.round19_contracts import (  # noqa: E402
    READINESS_STATES,
    SCHEMA_VERSIONS,
    audit_independence,
    audit_split_leakage,
    validate_e2_admission,
    validate_e2_endpoint,
    validate_e4_producer,
    validate_frozen_gaps,
    validate_round19_invariants,
    validate_training_eligibility,
)
from Libs.utils.exp.orchestration_controller import (  # noqa: E402
    CANONICAL_CONTROLLER_UID,
    build_canonical_gate_decision,
    load_authority_contract,
)


EXPECTED_BASE = "dba20fb554c18d18ba53b0f1cd713dbe9279d743"
VALIDATOR_UID = "round19_public_e2_e4_training_readiness_validator"
STAGE_UID = "round_19_public_e2_e4_training_readiness"
ACCEPTED_STATE = "real_e2_e4_expansion_and_training_readiness_resolved"
AUTHORITY_PATH = "Libs/configs/validation/round11f_controller_authority.yaml"
CONTRACT_CONFIG_PATH = "Libs/configs/data/round19_public_e2_e4.yaml"
SCHEMA_REGISTRY_PATH = "Libs/configs/schema/schemas.yaml"
VALIDATION_PATH = "Output/audits/dt_round19_validation.json"
GATE_PATH = "Output/audits/dt_round19_gate.json"
HANDOFF_PATH = "Output/audits/dt_round20_handoff.json"
PREFULL_PATH = "Output/audits/dt_round19_prefull_validation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _block(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True, text=True).stdout.strip()


def _descends_from_base(repository: Path) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_BASE, "HEAD"], cwd=repository, check=False).returncode == 0


def _schema_registry(repository: Path) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    payload = yaml.safe_load((repository / SCHEMA_REGISTRY_PATH).read_text(encoding="utf-8"))
    raw = dict(payload["schemas"]["round19_public_e2_e4_schema_registry"])
    blockers: list[dict[str, object]] = []
    if set(raw) != set(SCHEMA_VERSIONS):
        blockers.append(_block("schema_registry_incomplete", "Round-19 schema registry is not exact.", missing=sorted(set(SCHEMA_VERSIONS) - set(raw)), unexpected=sorted(set(raw) - set(SCHEMA_VERSIONS))))
    for name, version in SCHEMA_VERSIONS.items():
        entry = dict(raw.get(name, {}))
        path = repository / str(entry.get("path", ""))
        if entry.get("schema_version") != version or not path.is_file():
            blockers.append(_block("schema_registry_entry_invalid", "Round-19 schema entry is invalid.", schema=name))
            continue
        schema = _load_json(path)
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:
            blockers.append(_block("schema_invalid", str(exc), schema=name))
    return raw, blockers


def _parse_jsonish(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def _schema_errors(repository: Path, schema_name: str, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    schema = _load_json(repository / "Libs/configs/schema" / f"round19_{schema_name}.schema.json")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    blockers: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        for error in validator.iter_errors(dict(row)):
            blockers.append(_block("schema_validation", error.message, schema=schema_name, row=index, path=[str(item) for item in error.path]))
    return blockers


def _paths(output_root: Path, config: Mapping[str, object]) -> dict[str, Path]:
    return {name: output_root / str(relative) for name, relative in dict(config["round19_public_e2_e4"]["output_paths"]).items()}


def validate_outputs(
    *, repository: Path, output_root: Path, contract_root: Path, round18_root: Path,
    round18i_root: Path,
    frozen_gap_ledger: Path, full_suite: Path | None, test_inventory: Path | None,
    require_final: bool,
) -> dict[str, object]:
    config = load_config(repository / CONTRACT_CONFIG_PATH)
    section = dict(config["round19_public_e2_e4"])
    paths = _paths(output_root, config)
    blockers: list[dict[str, object]] = []
    if not _descends_from_base(repository):
        blockers.append(_block("execution_identity", "Candidate does not descend from the accepted Round-18I main commit."))
    registry, schema_blockers = _schema_registry(repository)
    blockers.extend(schema_blockers)

    required_names = {
        "round18_diagnosis", "replay_failure_map", "e2_candidate_catalog", "e2_admission_ledger",
        "e2_independence_groups", "e2_coverage", "e2_independence_audit", "e2_duplicate_audit",
        "e2_leakage_audit", "e4_domain_selection", "producer_qualification", "e4_pilot_manifest",
        "e4_pilot_validation", "e4_control_pair_audit", "e4_trajectory_catalog", "e4_control_pair_catalog",
        "e4_admission_ledger", "e2_e4_alignment", "e4_coverage", "frozen_gap_resolution",
        "training_eligibility", "split_feasibility", "training_readiness",
    }
    missing = [name for name in sorted(required_names) if not paths[name].is_file()]
    if missing:
        blockers.append(_block("artifact_missing", "Required Round-19 artifacts are missing.", artifacts=missing))

    scientific = contract_root / "Output/audits/dt_round19_scientific_contract.json"
    readiness_contract = contract_root / "Output/audits/dt_round19_training_readiness_contract.json"
    freeze_audit = contract_root / "Output/audits/dt_round19_contract_freeze_audit.json"
    if not all(path.is_file() for path in (scientific, readiness_contract, freeze_audit)):
        blockers.append(_block("contract_freeze_missing", "Frozen Round-19 contracts are missing."))
    else:
        freeze = _load_json(freeze_audit)
        if freeze.get("status") != "frozen_before_discovery" or freeze.get("thresholds_mutable_after_source_observation") is not False:
            blockers.append(_block("contract_freeze_invalid", "Scientific thresholds were not prospectively frozen."))
        if freeze.get("scientific_contract_hash") != sha256_file(scientific) or freeze.get("training_readiness_contract_hash") != sha256_file(readiness_contract):
            blockers.append(_block("contract_hash_drift", "Frozen contract hashes changed after discovery."))

    parent_checks = (
        (round18_root, "dt_round18_gate.json", "status", "literature_held_out_sequential_replay_resolved"),
        (round18i_root, "dt_round18i_gate.json", "status", "literature_replay_mainline_integrated"),
        (round18i_root, "dt_round19_entry_gate.json", "status", "ready_for_real_e2_e4_evidence_expansion_and_training_readiness"),
    )
    for root, name, field, expected in parent_checks:
        path = root / "Output/audits" / name
        if not path.is_file() or _load_json(path).get(field) != expected:
            blockers.append(_block("parent_gate_invalid", "Accepted parent Gate is missing or changed.", artifact=name))

    if not missing:
        endpoints = read_parquet_records(paths["e2_candidate_catalog"])
        for row in endpoints:
            row["independence_groups"] = _parse_jsonish(row.get("independence_groups"))
        admissions = read_tsv(paths["e2_admission_ledger"])
        for row in admissions:
            row["reason_codes"] = _parse_jsonish(row.get("reason_codes"))
            row["empirical_observation"] = row.get("empirical_observation") == "True"
        blockers.extend(_schema_errors(repository, "e2_endpoint", endpoints))
        blockers.extend(_schema_errors(repository, "e2_admission", admissions))
        endpoint_by_uid = {str(row["endpoint_uid"]): row for row in endpoints}
        forbidden_source_tokens = tuple(str(item) for item in section["public_source_policy"]["rejected_routes"])
        for endpoint in endpoints:
            blockers.extend(validate_e2_endpoint(endpoint, forbidden_source_tokens=forbidden_source_tokens))
        for admission in admissions:
            endpoint = endpoint_by_uid.get(str(admission.get("endpoint_uid")), {})
            blockers.extend(validate_e2_admission(admission, endpoint))
        if any(row.get("decision") in {"admitted_e2_strong", "admitted_e2_weighted"} for row in admissions):
            blockers.append(_block("label_provenance", "No machine-readable strong or weighted E2 was available in this bounded execution."))

        groups = read_tsv(paths["e2_independence_groups"])
        independence = audit_independence(groups)
        if independence["status"] != "passed":
            blockers.extend(independence["blockers"])
        blockers.extend(audit_split_leakage(groups))
        leakage = _load_json(paths["e2_leakage_audit"])
        if leakage.get("final_held_out_access_count") != 0 or leakage.get("final_held_out_leakage_count") != 0:
            blockers.append(_block("final_held_out_access", "Final held-out evidence was accessed or leaked."))

        producer_rows = read_tsv(paths["producer_qualification"])
        for row in producer_rows:
            row["supported_state_variables"] = _parse_jsonish(row.get("supported_state_variables"))
            row["supported_action_semantics"] = _parse_jsonish(row.get("supported_action_semantics"))
            blockers.extend(validate_e4_producer(row))
        pilot = _load_json(paths["e4_pilot_manifest"])
        trajectories = read_parquet_records(paths["e4_trajectory_catalog"])
        pairs = read_parquet_records(paths["e4_control_pair_catalog"])
        if pilot.get("status") != "not_run_no_qualified_intervention_producer" or trajectories or pairs:
            blockers.append(_block("e4_qualification", "Blocked E4 producer must not generate a pilot trajectory or control pair."))

        parent_gaps = read_tsv(frozen_gap_ledger)
        gap_rows = read_tsv(paths["frozen_gap_resolution"])
        for row in gap_rows:
            row["evidence_refs"] = _parse_jsonish(row.get("evidence_refs"))
        blockers.extend(validate_frozen_gaps(gap_rows))
        if {row["gap_uid"] for row in gap_rows} != {row["gap_uid"] for row in parent_gaps}:
            blockers.append(_block("frozen_gap_lineage", "Round-19 gap UIDs do not exactly reconcile to the inherited ledger."))
        if any(row.get("status") != "frozen_unresolved" for row in gap_rows):
            blockers.append(_block("gap_fabrication", "Directional E2 and absent E4 cannot close an inherited empirical gap."))

        eligibility = read_tsv(paths["training_eligibility"])
        for row in eligibility:
            row["validation_only"] = row.get("validation_only") == "True"
        blockers.extend(validate_training_eligibility(eligibility))
        readiness = _load_json(paths["training_readiness"])
        if readiness.get("state") not in READINESS_STATES or readiness.get("training_authorized") is not False or readiness.get("training_started") is not False:
            blockers.append(_block("training_boundary", "Training readiness outcome or authority is invalid."))
        if readiness.get("thresholds_lowered_after_source_observation") is not False:
            blockers.append(_block("threshold_drift", "Readiness thresholds were lowered after source observation."))

        source_resolution = output_root / "Output/audits/dt_round19_public_source_resolution.json"
        if not source_resolution.is_file():
            blockers.append(_block("public_source_boundary", "Public source-resolution audit is missing."))
        else:
            source = _load_json(source_resolution)
            if source.get("strict_public_only_flag") is not True or source.get("controlled_source_resolution_count") != 0 or source.get("failed_attempts_preserved") is not True:
                blockers.append(_block("public_source_boundary", "Public-only source boundary or attempt preservation failed."))

    invariant_blockers = validate_round19_invariants(dict(section["invariants"]))
    blockers.extend(invariant_blockers)
    clean = _git(repository, "status", "--porcelain") == ""
    if require_final and not clean:
        blockers.append(_block("execution_identity", "Final Round-19 validation requires an exact-clean worktree."))

    full_value: dict[str, object] = {}
    inventory_value: dict[str, object] = {}
    if require_final:
        if full_suite is None or not full_suite.is_file():
            blockers.append(_block("full_suite", "Full-suite execution evidence is missing."))
        else:
            full_value = _load_json(full_suite)
            if not (full_value.get("terminal_state") == "COMPLETED" and full_value.get("exit_code") == "0:0" and int(full_value.get("passed", 0)) >= 587 and full_value.get("failures") == 0 and full_value.get("errors") == 0 and full_value.get("skipped") == 0):
                blockers.append(_block("full_suite", "Full CPU suite did not meet the accepted execution contract."))
            if full_value.get("commit") != _git(repository, "rev-parse", "HEAD"):
                blockers.append(_block("execution_identity", "Full suite was not run at the candidate commit."))
        if test_inventory is None or not test_inventory.is_file():
            blockers.append(_block("test_inventory", "Round-19 test inventory reconciliation is missing."))
        else:
            inventory_value = _load_json(test_inventory)
            if not (inventory_value.get("inherited_test_count") == 587 and inventory_value.get("unexplained_missing_tests") == 0 and inventory_value.get("mandatory_disappearance") == 0 and inventory_value.get("integrated_test_count") == full_value.get("passed")):
                blockers.append(_block("test_inventory", "The accepted 587-test membership was not completely retained."))

    ordered = sorted(blockers, key=lambda row: (str(row.get("code")), json.dumps(row, sort_keys=True)))
    input_paths = [path for name, path in paths.items() if name in required_names and path.is_file()]
    report = {
        "schema_version": "neuronsifter-dt-round19-validation-v1",
        "validator_uid": VALIDATOR_UID,
        "status": "passed" if not ordered else "blocked",
        "assessment": ACCEPTED_STATE if not ordered else "blocked_execution_verification",
        "blocker_count": len(ordered),
        "blockers": ordered,
        "candidate_commit": _git(repository, "rev-parse", "HEAD"),
        "candidate_tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "accepted_base_commit": EXPECTED_BASE,
        "dirty_status": "clean" if clean else "dirty",
        "schema_registry_count": 1,
        "canonical_controller_count": 1,
        "canonical_gate_writer_count": 1,
        "readiness": _load_json(paths["training_readiness"]) if paths["training_readiness"].is_file() else {},
        "full_suite": full_value,
        "test_inventory": inventory_value,
        "artifact_hashes": {path.relative_to(output_root).as_posix(): sha256_file(path) for path in input_paths},
        "invariants": dict(section["invariants"]),
        "frozen_gap_count_after_round19": 20,
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }
    return report


def build_gate(repository: Path, report: Mapping[str, object]) -> dict[str, object]:
    authority = load_authority_contract(repository / AUTHORITY_PATH)
    common = {
        "status": report["assessment"],
        "overall_gate_status": "accepted" if report["status"] == "passed" else "blocked",
        "training_readiness": dict(report.get("readiness", {})).get("state"),
        "training": {"authorized": False, "started": False},
        "evidence_fabric": {"sealed": False},
        "frozen_gap_count": 20,
        "dt_core": {"status": "not_evaluated"},
    }
    if report["status"] != "passed":
        return {
            "schema_version": "neuronsifter-dt-round19-blocked-gate-assessment-v1",
            "gate_uid": STAGE_UID,
            "stage_uid": STAGE_UID,
            "validator_uid": VALIDATOR_UID,
            "canonical_controller_uid": CANONICAL_CONTROLLER_UID,
            "decision": "blocked",
            "blockers": report["blockers"],
            **common,
        }
    return build_canonical_gate_decision(
        authority,
        gate_uid=STAGE_UID,
        stage_uid=STAGE_UID,
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid=VALIDATOR_UID,
        validator_report={"validator_uid": VALIDATOR_UID, "status": "passed", "assessment": ACCEPTED_STATE},
        input_artifact_hashes=dict(report["artifact_hashes"]),
        config_hash=sha256_file(repository / CONTRACT_CONFIG_PATH),
        commit=str(report["candidate_commit"]),
        decision="accepted",
        blockers=[],
        lineage_parent="round_18i_literature_replay_mainline_integration",
    ) | common


def build_handoff(report: Mapping[str, object], gate: Mapping[str, object]) -> dict[str, object]:
    if gate.get("status") != ACCEPTED_STATE:
        raise ValueError("Round-20 handoff requires the accepted Round-19 Gate")
    readiness = dict(report["readiness"])
    return {
        "schema_version": "neuronsifter-dt-round20-handoff-v1",
        "status": "ready_for_evidence_recovery" if str(readiness["state"]).startswith("blocked_") else "ready_to_propose_separately_authorized_restricted_tiny_model_experiment",
        "round19_gate": ACCEPTED_STATE,
        "round19_execution_commit": report["candidate_commit"],
        "training_readiness": readiness["state"],
        "remaining_requirements": [item["code"] for item in readiness.get("blockers", [])],
        "training": {"authorized": False, "started": False},
        "evidence_fabric": {"sealed": False},
        "frozen_gap_count": 20,
        "dt_core": {"status": "not_evaluated"},
        "world_action_training_requires_separate_gate": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--round18-root", type=Path, required=True)
    parser.add_argument("--round18i-root", type=Path, required=True)
    parser.add_argument("--frozen-gap-ledger", type=Path, required=True)
    parser.add_argument("--full-suite", type=Path)
    parser.add_argument("--test-inventory", type=Path)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    output_root = args.output_root.resolve()
    report = validate_outputs(
        repository=repository, output_root=output_root, contract_root=args.contract_root.resolve(),
        round18_root=args.round18_root.resolve(), round18i_root=args.round18i_root.resolve(),
        frozen_gap_ledger=args.frozen_gap_ledger.resolve(),
        full_suite=args.full_suite.resolve() if args.full_suite else None,
        test_inventory=args.test_inventory.resolve() if args.test_inventory else None,
        require_final=args.final,
    )
    if args.final:
        gate = build_gate(repository, report)
        write_json(output_root / VALIDATION_PATH, report)
        write_json(output_root / GATE_PATH, gate)
        if gate.get("status") == ACCEPTED_STATE:
            write_json(output_root / HANDOFF_PATH, build_handoff(report, gate))
    else:
        write_json(output_root / PREFULL_PATH, report)
    print(json.dumps({"status": report["status"], "assessment": report["assessment"], "blockers": report["blocker_count"]}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
