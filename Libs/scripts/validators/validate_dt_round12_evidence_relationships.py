"""Validate Round-12 typed evidence relationships and issue the canonical Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_dt_round12_evidence_relationships import AUTHORITY_PATH, OUTPUT_PATHS, SCHEMA_PATHS  # noqa: E402
from Libs.utils.data.evidence_relationships import (  # noqa: E402
    NODE_TYPES,
    RELATION_TYPES,
    SCHEMA_VERSIONS,
    frozen_gap_rows,
    round12_gate_status,
    validate_frozen_gap_preservation,
    validate_graph,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, sha256_of_file, write_json  # noqa: E402
from Libs.utils.exp.orchestration_controller import (  # noqa: E402
    CANONICAL_CONTROLLER_UID,
    build_canonical_gate_decision,
    git_commit,
    git_dirty_status,
    load_authority_contract,
    sha256_file,
    sha256_payload,
)


MANDATORY_NEGATIVE_TEST_NAMES = frozenset(
    {
        "test_cross_sectional_samples_cannot_be_longitudinal",
        "test_different_biological_units_cannot_be_same_unit",
        "test_static_endpoint_cannot_be_trajectory",
        "test_simulator_output_cannot_be_experimental_truth",
        "test_semantic_alignment_cannot_be_action_response",
        "test_hypothetical_scenario_cannot_be_empirical_evidence",
        "test_same_publication_is_insufficient_pairing_identity",
        "test_unknown_provenance_cannot_enter_strong_training",
        "test_unsupported_support_domain_cannot_generalize",
        "test_evidence_union_cannot_upgrade_eligibility",
        "test_conflict_cannot_disappear_during_union",
        "test_lineage_cycle_is_rejected",
        "test_duplicate_active_successor_is_rejected",
        "test_unknown_relationship_type_is_rejected",
        "test_missing_required_identity_axis_is_rejected",
    }
)
EXECUTION_IDENTITY_SCHEMA_VERSION = "neuronsifter-dt-round12-execution-identity-v1"
FULL_SUITE_EXECUTION_SCHEMA_VERSION = "neuronsifter-dt-round12-full-suite-execution-v1"
TEST_INVENTORY_SCHEMA_VERSION = "neuronsifter-dt-round12-test-inventory-diff-v1"


def _junit_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {"available": False, "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "testcase_names": []}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    testcase_names = sorted({str(case.get("name", "")) for case in root.iter("testcase")})
    return {
        "available": True,
        "path": path.name,
        "sha256": sha256_of_file(path),
        "tests": sum(int(suite.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.get("skipped", 0)) for suite in suites),
        "testcase_names": testcase_names,
    }


def _catalog_blockers(graph: dict[str, object], repository_root: Path) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    catalogs = {
        "nodes": ("nodes", "node_uid"),
        "relations": ("relations", "relation_uid"),
        "pairings": ("pairing_decisions", "pairing_uid"),
        "domains": ("support_domains", "support_domain_uid"),
        "eligibilities": ("eligibilities", "eligibility_uid"),
        "conflicts": ("conflicts", "conflict_uid"),
    }
    for output_key, (graph_key, uid_key) in catalogs.items():
        path = repository_root / OUTPUT_PATHS[output_key]
        if not path.is_file():
            blockers.append({"code": "missing_pilot_catalog", "message": "Required Round-12 pilot catalog is missing.", "path": OUTPUT_PATHS[output_key]})
            continue
        rows = read_tsv(path)
        expected = {str(row[uid_key]) for row in graph[graph_key]}
        actual = {str(row.get(uid_key, "")) for row in rows}
        if expected != actual or len(rows) != len(graph[graph_key]):
            blockers.append({"code": "pilot_catalog_drift", "message": "Pilot catalog does not exactly match the typed graph.", "path": OUTPUT_PATHS[output_key]})
    return blockers


def _schema_registry_blockers(repository_root: Path) -> list[dict[str, object]]:
    config = load_config(repository_root / "Libs/configs/config_stack.yaml")
    registry = config.get("schemas", {}).get("evidence_relationship_schema_registry", {})
    blockers: list[dict[str, object]] = []
    if set(registry) != {"evidence_node", "evidence_relation", "evidence_pairing_decision", "evidence_support_domain", "evidence_eligibility", "evidence_conflict", "evidence_relationship_graph"}:
        blockers.append({"code": "canonical_schema_registry_drift", "message": "Round-12 schemas are not exactly integrated into schemas.yaml."})
        return blockers
    expected_families = {
        "evidence_node": "node", "evidence_relation": "relation", "evidence_pairing_decision": "pairing",
        "evidence_support_domain": "support_domain", "evidence_eligibility": "eligibility",
        "evidence_conflict": "conflict", "evidence_relationship_graph": "graph",
    }
    for registry_key, family in expected_families.items():
        entry = registry[registry_key]
        if entry.get("schema_version") != SCHEMA_VERSIONS[family] or entry.get("path") != SCHEMA_PATHS[family]:
            blockers.append({"code": "canonical_schema_registry_binding", "message": "Schema registry binding is not canonical.", "schema_family": registry_key})
            continue
        schema_path = repository_root / entry["path"]
        if not schema_path.is_file():
            blockers.append({"code": "missing_registered_schema", "message": "Registered schema file is missing.", "path": entry["path"]})
            continue
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    return blockers


def _load_json_artifact(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _parent_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _execution_identity_blockers(
    *,
    path: Path | None,
    repository_root: Path,
    commit: str,
    bundle_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    identity = _load_json_artifact(path)
    if identity is None:
        return ([{"code": "execution_identity_artifact_missing", "message": "Round-12 requires a versioned exact-clean execution identity artifact."}], {})
    blockers: list[dict[str, object]] = []
    if identity.get("schema_version") != EXECUTION_IDENTITY_SCHEMA_VERSION:
        blockers.append({"code": "execution_identity_schema_mismatch", "message": "Execution identity schema version is not canonical."})
    expected_values = {
        "exact_git_commit": commit,
        "parent_commit": _parent_commit(repository_root),
        "dirty_status": "clean",
        "input_bundle_hash": bundle_sha256,
        "reproduction_relation": "exact_clean_commit",
    }
    for field, expected in expected_values.items():
        if identity.get(field) != expected:
            blockers.append({"code": "execution_identity_not_exact_clean", "message": "Execution identity does not match the exact clean worktree.", "field": field, "expected": expected, "actual": identity.get(field)})
    expected_hashes = {
        "config_hash": sha256_payload(load_config(repository_root / "Libs/configs/config_stack.yaml")),
        "schema_registry_hash": sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        "evidence_relationships_hash": sha256_file(repository_root / "Libs/utils/data/evidence_relationships.py"),
        "evidence_union_hash": sha256_file(repository_root / "Libs/utils/data/evidence_union.py"),
        "validator_hash": sha256_file(repository_root / "Libs/scripts/validators/validate_dt_round12_evidence_relationships.py"),
    }
    for field, expected in expected_hashes.items():
        if identity.get(field) != expected:
            blockers.append({"code": "execution_identity_hash_mismatch", "message": "Execution identity hash does not match the exact commit.", "field": field, "expected": expected, "actual": identity.get(field)})
    for field in ("test_manifest_hash", "environment_hash"):
        value = identity.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            blockers.append({"code": "execution_identity_hash_mismatch", "message": "Execution identity contains an invalid SHA-256.", "field": field})
    return blockers, identity


def _test_inventory_blockers(
    path: Path | None,
    full_summary: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    inventory = _load_json_artifact(path)
    if inventory is None:
        return ([{"code": "test_inventory_missing", "message": "Round-12 requires a diff against the accepted Round-11 test inventory."}], {})
    blockers: list[dict[str, object]] = []
    valid = (
        inventory.get("schema_version") == TEST_INVENTORY_SCHEMA_VERSION
        and inventory.get("round11_parent_test_count") == 367
        and inventory.get("round12_current_test_count") == full_summary.get("tests")
        and inventory.get("unexplained_missing_tests") == 0
        and inventory.get("mandatory_disappearance") == 0
        and inventory.get("status") == "complete"
    )
    common = inventory.get("common_tests")
    added = inventory.get("added_tests")
    removed = inventory.get("removed_tests")
    renamed = inventory.get("renamed_tests")
    superseded = inventory.get("superseded_tests")
    if not all(isinstance(value, list) for value in (common, added, removed, renamed, superseded)):
        valid = False
    elif len(common) + len(added) != inventory.get("round12_current_test_count"):
        valid = False
    elif len(common) + len(removed) != inventory.get("round11_parent_test_count"):
        valid = False
    if not valid:
        blockers.append({"code": "test_inventory_incomplete", "message": "Round-12 test inventory has missing, unexplained, or internally inconsistent testcase identities."})
    return blockers, inventory


def _full_suite_execution_blockers(
    *,
    path: Path | None,
    identity: dict[str, object],
    inventory_path: Path | None,
    full_summary: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    execution = _load_json_artifact(path)
    if execution is None:
        return ([{"code": "full_suite_execution_artifact_missing", "message": "Round-12 requires a terminal Slurm full-suite execution record."}], {})
    blockers: list[dict[str, object]] = []
    result = execution.get("structured_test_result")
    result = result if isinstance(result, dict) else {}
    terminal_valid = (
        execution.get("schema_version") == FULL_SUITE_EXECUTION_SCHEMA_VERSION
        and execution.get("terminal_state") == "COMPLETED"
        and execution.get("exit_code") == "0:0"
        and execution.get("pytest_exit_code") == 0
        and execution.get("node") == "cpu04"
        and execution.get("partition") == "cpu"
        and str(execution.get("job_name", "")).startswith("NrnSift_DT_R12_")
        and execution.get("mandatory_skipped") == 0
        and execution.get("core_xfail") == 0
        and result.get("tests") == full_summary.get("tests")
        and result.get("failures") == full_summary.get("failures")
        and result.get("errors") == full_summary.get("errors")
        and result.get("skipped") == full_summary.get("skipped")
    )
    if not terminal_valid:
        blockers.append({"code": "full_suite_terminal_evidence_invalid", "message": "Full-suite terminal, exit-code, node, or structured result evidence is incomplete."})
    identity_fields = {
        "exact_git_commit": "exact_git_commit",
        "input_bundle_hash": "input_bundle_hash",
        "environment_hash": "environment_hash",
        "config_hash": "config_hash",
        "schema_registry_hash": "schema_registry_hash",
    }
    if any(execution.get(execution_field) != identity.get(identity_field) for execution_field, identity_field in identity_fields.items()):
        blockers.append({"code": "full_suite_execution_identity_mismatch", "message": "Full-suite execution hashes do not match the exact-clean execution identity."})
    if inventory_path is None or not inventory_path.is_file() or execution.get("test_inventory_hash") != sha256_file(inventory_path):
        blockers.append({"code": "test_inventory_incomplete", "message": "Full-suite execution is not bound to the exact Round-12 test inventory."})
    streams = execution.get("streams")
    streams = streams if isinstance(streams, dict) else {}
    if not all(isinstance(streams.get(name), dict) and streams[name].get("sha256") for name in ("stdout", "stderr", "junit", "scontrol")):
        blockers.append({"code": "full_suite_terminal_evidence_invalid", "message": "Full-suite stream hashes are incomplete."})
    if execution.get("sacct_status") not in {"available", "explicit_slurmdbd_unavailable"}:
        blockers.append({"code": "full_suite_terminal_evidence_invalid", "message": "sacct evidence or an explicit SlurmDBD-unavailable status is required."})
    return blockers, execution


def validate_round12_outputs(
    *,
    execution_bundle_path: Path,
    expected_bundle_sha256: str,
    frozen_validation_path: Path,
    negative_junit_path: Path | None,
    full_suite_junit_path: Path | None,
    execution_identity_path: Path | None = None,
    full_suite_execution_path: Path | None = None,
    test_inventory_path: Path | None = None,
    repository_root: Path = REPO_ROOT,
) -> tuple[dict[str, object], dict[str, object]]:
    graph_path = repository_root / OUTPUT_PATHS["graph"]
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    blockers = list(validate_graph(graph)["blockers"])
    blockers.extend(_catalog_blockers(graph, repository_root))
    blockers.extend(_schema_registry_blockers(repository_root))

    observed_bundle_sha256 = sha256_of_file(execution_bundle_path)
    if observed_bundle_sha256 != expected_bundle_sha256 or graph.get("source_bundle", {}).get("sha256") != expected_bundle_sha256:
        blockers.append({"code": "execution_bundle_hash_mismatch", "message": "Round-12 graph does not bind the exact frozen Round-11 execution input bundle."})
    frozen_validation = json.loads(frozen_validation_path.read_text(encoding="utf-8"))
    mapped_gaps = read_tsv(repository_root / OUTPUT_PATHS["gaps"])
    config = load_config(repository_root / "Libs/configs/config_stack.yaml")
    source_adapters = config["digital_twin_evidence_fabric"]["source_adapters"]
    blockers.extend(validate_frozen_gap_preservation(frozen_validation, mapped_gaps, source_adapters))
    if mapped_gaps != [{key: str(value) for key, value in row.items()} for row in frozen_gap_rows(frozen_validation, source_adapters)]:
        blockers.append({"code": "frozen_gap_mapping_content_drift", "message": "Frozen gap map content differs from the authoritative adapter contracts."})

    negative_summary = _junit_summary(negative_junit_path)
    negative_names = set(negative_summary["testcase_names"])
    negative_verified = (
        negative_summary["available"]
        and negative_summary["failures"] == 0
        and negative_summary["errors"] == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(negative_names)
    )
    if not negative_verified:
        blockers.append({"code": "mandatory_negative_tests_not_verified", "message": "All 15 mandatory fail-closed negative tests require passing JUnit evidence.", "missing": sorted(MANDATORY_NEGATIVE_TEST_NAMES - negative_names)})

    full_summary = _junit_summary(full_suite_junit_path)
    junit_verified = (
        full_summary["available"]
        and full_summary["tests"] >= 367
        and full_summary["failures"] == 0
        and full_summary["errors"] == 0
        and full_summary["skipped"] == 0
    )
    commit = git_commit(repository_root)
    namespace_dirty_status = git_dirty_status(repository_root)
    identity_blockers, execution_identity = _execution_identity_blockers(
        path=execution_identity_path,
        repository_root=repository_root,
        commit=commit,
        bundle_sha256=observed_bundle_sha256,
    )
    blockers.extend(identity_blockers)
    if execution_identity_path is None and namespace_dirty_status != "clean":
        blockers.append({"code": "execution_identity_not_exact_clean", "message": "Round-12 formal Gate requires an exact clean Git execution identity."})
    inventory_blockers, test_inventory = _test_inventory_blockers(test_inventory_path, full_summary)
    blockers.extend(inventory_blockers)
    execution_blockers, full_suite_execution = _full_suite_execution_blockers(
        path=full_suite_execution_path,
        identity=execution_identity,
        inventory_path=test_inventory_path,
        full_summary=full_summary,
    )
    blockers.extend(execution_blockers)
    if not junit_verified:
        blockers.append({"code": "full_suite_not_verified", "message": "Round-12 requires a zero-failure full Libs/tests JUnit result with no mandatory disappearance."})
    blockers = sorted(blockers, key=lambda row: (str(row.get("code", "")), json.dumps(row, sort_keys=True)))
    gate_status = round12_gate_status(blockers)
    report = {
        "schema_version": SCHEMA_VERSIONS["validation"],
        "validator_uid": "round12_typed_evidence_relationship_validator",
        "status": "passed" if not blockers else "blocked",
        "gate_assessment": gate_status["status"],
        **{key: value for key, value in gate_status.items() if key != "status"},
        "canonical_node_type_count": len(NODE_TYPES),
        "canonical_relation_type_count": len(RELATION_TYPES),
        "graph_uid": graph.get("graph_uid"),
        "object_counts": validate_graph(graph)["object_counts"],
        "frozen_gap_source_count": 20,
        "frozen_gap_mapped_count": len(mapped_gaps),
        "frozen_gap_count_preserved": len(mapped_gaps) == 20,
        "mandatory_negative_case_count": len(MANDATORY_NEGATIVE_TEST_NAMES),
        "mandatory_negative_cases_verified": negative_verified,
        "negative_test_junit": negative_summary,
        "full_suite": full_summary,
        "full_suite_execution": full_suite_execution,
        "test_inventory": test_inventory,
        "execution_identity": execution_identity or {"exact_git_commit": commit, "dirty_status": namespace_dirty_status, "reproduction_relation": "not_exact_clean_commit"},
        "execution_namespace_git_status": namespace_dirty_status,
        "blockers": blockers,
        "blocker_count": len(blockers),
        "pilot_scope": "contract_validation_only_not_evidence_fabric",
        "evidence_fabric_sealed": False,
        "dt_core": "not_evaluated",
    }
    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    gate_input_hashes = {
        "execution_input_bundle": observed_bundle_sha256,
        "relationship_graph": sha256_file(graph_path),
        "frozen_gap_map": sha256_file(repository_root / OUTPUT_PATHS["gaps"]),
    }
    for name, path in (
        ("execution_identity", execution_identity_path),
        ("full_suite_execution", full_suite_execution_path),
        ("test_inventory", test_inventory_path),
        ("negative_junit", negative_junit_path),
        ("full_suite_junit", full_suite_junit_path),
    ):
        if path is not None and path.is_file():
            gate_input_hashes[name] = sha256_file(path)
    gate = build_canonical_gate_decision(
        authority,
        gate_uid="round_12_typed_evidence_relationships",
        stage_uid="round_12_typed_evidence_relationships",
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid="round12_typed_evidence_relationship_validator",
        validator_report={"validator_uid": "round12_typed_evidence_relationship_validator", "status": "passed", "assessment": report["gate_assessment"]},
        input_artifact_hashes=gate_input_hashes,
        config_hash=sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        commit=commit,
        decision="accepted" if not blockers else "blocked",
        blockers=blockers,
        lineage_parent="round_11g_targeted_repair_and_full_suite_closure",
    ) | {
        "round_uid": "round_12_typed_evidence_relationships",
        **gate_status,
        "validation_status": report["status"],
        "blocker_count": len(blockers),
        "frozen_gap_count": len(mapped_gaps),
        "execution_identity": report["execution_identity"],
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "round_11_closure": "accepted_unchanged",
    }
    return report, gate


def build_round13_handoff(graph: dict[str, object], gaps: list[dict[str, str]]) -> dict[str, object]:
    return {
        "schema_version": "neuronsifter-dt-round13-handoff-v1",
        "status": "ready_for_structured_world_action_contracts",
        "parent_gate": "typed_evidence_relationships_resolved",
        "canonical_evidence_node_vocabulary": sorted(NODE_TYPES),
        "canonical_relationship_vocabulary": sorted(RELATION_TYPES),
        "pairing_rules": graph["pairing_decisions"],
        "support_domain_rules": graph["support_domains"],
        "eligibility_matrix": graph["eligibilities"],
        "conflict_semantics": graph["conflicts"],
        "frozen_gap_ledger": gaps,
        "evidence_union_interaction_rules": {
            "eligibility": "boolean_intersection_only_no_upgrade",
            "pairing": "unanimous_exact_decision_only_otherwise_unsupported_pairing",
            "qc_and_evidence_class": "preserve_or_downgrade_never_upgrade",
            "typed_conflicts": "preserve_all_unique_conflicts_and_block_uid_payload_collisions",
        },
        "allowed_action_observation_consumers": ["round13_structured_world_action", "round13_information_action", "round13_observation_assimilation", "round13_hypothesis_contract"],
        "allowed_downstream_consumers": ["round13_structured_world_action", "round13_information_action", "round13_observation_assimilation", "round13_hypothesis_contract"],
        "round13_entry_conditions": [
            "round_12_gate_is_typed_evidence_relationships_resolved",
            "exact_clean_execution_identity_preserved",
            "round_12_full_suite_passed",
            "twenty_frozen_input_gaps_preserved",
            "evidence_fabric_remains_unsealed",
            "dt_core_remains_not_evaluated",
        ],
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Round-12 typed evidence relationship artifacts.")
    parser.add_argument("--execution-input-bundle", required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--frozen-validation", default="Output/audits/dt_evidence_fabric_validation.json")
    parser.add_argument("--negative-junit")
    parser.add_argument("--full-suite-junit")
    parser.add_argument("--execution-identity")
    parser.add_argument("--full-suite-execution")
    parser.add_argument("--test-inventory")
    args = parser.parse_args()
    report, gate = validate_round12_outputs(
        execution_bundle_path=Path(args.execution_input_bundle),
        expected_bundle_sha256=args.expected_bundle_sha256,
        frozen_validation_path=REPO_ROOT / args.frozen_validation,
        negative_junit_path=Path(args.negative_junit) if args.negative_junit else None,
        full_suite_junit_path=Path(args.full_suite_junit) if args.full_suite_junit else None,
        execution_identity_path=Path(args.execution_identity) if args.execution_identity else None,
        full_suite_execution_path=Path(args.full_suite_execution) if args.full_suite_execution else None,
        test_inventory_path=Path(args.test_inventory) if args.test_inventory else None,
    )
    write_json(REPO_ROOT / OUTPUT_PATHS["validation"], report)
    write_json(REPO_ROOT / OUTPUT_PATHS["gate"], gate)
    if gate["status"] == "typed_evidence_relationships_resolved":
        graph = json.loads((REPO_ROOT / OUTPUT_PATHS["graph"]).read_text(encoding="utf-8"))
        gaps = read_tsv(REPO_ROOT / OUTPUT_PATHS["gaps"])
        write_json(REPO_ROOT / "Output/audits/dt_round13_handoff.json", build_round13_handoff(graph, gaps))
    print(f"{gate['status']} blockers={report['blocker_count']} gaps={report['frozen_gap_mapped_count']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
