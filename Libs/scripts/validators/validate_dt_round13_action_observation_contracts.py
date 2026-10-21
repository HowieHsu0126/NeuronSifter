"""Validate Round-13 action/observation contracts and issue the canonical Gate."""

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

from Libs.scripts.builders.build_dt_round13_action_observation_contracts import (  # noqa: E402
    ASSIMILATION_VALIDATOR_UID,
    AUTHORITY_PATH,
    OUTPUT_PATHS,
    SCHEMA_PATHS,
    SUPPORTED_STATE_COMPONENTS,
)
from Libs.utils.data.action_observation_contracts import (  # noqa: E402
    SCHEMA_VERSIONS,
    WORLD_ACTION_TYPES,
    frozen_gap_action_observation_rows,
    round13_gate_status,
    validate_frozen_gap_action_observation_map,
    validate_round13_graph,
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


VALIDATION_PATH = "Output/audits/dt_round13_validation.json"
GATE_PATH = "Output/audits/dt_round13_gate.json"
ROUND14_HANDOFF_PATH = "Output/audits/dt_round14_handoff.json"
EXECUTION_IDENTITY_SCHEMA_VERSION = "neuronsifter-dt-round13-execution-identity-v1"
PREFULL_SCHEMA_VERSION = "neuronsifter-dt-round13-prefull-validation-v1"
FULL_SUITE_EXECUTION_SCHEMA_VERSION = "neuronsifter-dt-round13-full-suite-execution-v1"
TEST_INVENTORY_SCHEMA_VERSION = "neuronsifter-dt-round13-test-inventory-diff-v1"

MANDATORY_NEGATIVE_TEST_NAMES = frozenset(
    {
        "test_drug_id_treated_as_complete_action_rejected",
        "test_tool_query_treated_as_world_action_rejected",
        "test_tool_result_directly_updates_twin_rejected",
        "test_llm_proposal_directly_updates_twin_rejected",
        "test_missing_dose_unit_rejected",
        "test_unsupported_route_rejected",
        "test_unknown_parameter_provenance_rejected",
        "test_static_endpoint_authorizes_trajectory_update_rejected",
        "test_hypothetical_scenario_becomes_observation_rejected",
        "test_unknown_provenance_becomes_strong_observation_rejected",
        "test_qc_d_evidence_enters_assimilation_rejected",
        "test_cross_domain_observation_updates_state_rejected",
        "test_conflict_disappears_during_normalization_rejected",
        "test_evidence_union_upgrades_observation_eligibility_rejected",
        "test_noncanonical_controller_approves_assimilation_rejected",
        "test_unsupported_tool_silently_replaced_rejected",
        "test_arbitrary_shell_command_generated_by_action_rejected",
        "test_frozen_gap_reduced_without_evidence_rejected",
    }
)


def _load_json(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _junit_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {"available": False, "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "testcase_names": []}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    testcase_names = sorted(str(case.get("name", "")) for case in root.iter("testcase"))
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


def test_manifest_hash(repository_root: Path) -> str:
    entries = [
        {"path": path.relative_to(repository_root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted((repository_root / "Libs/tests").glob("test_*.py"))
    ]
    return sha256_payload(entries)


def _parent_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _catalog_blockers(graph: MappingLike, source_output_root: Path) -> list[dict[str, object]]:
    catalogs = {
        "world_actions": ("world_actions", "action_uid"),
        "information_actions": ("information_actions", "information_action_uid"),
        "observation_requests": ("observation_requests", "request_uid"),
        "tool_matches": ("tool_capability_matches", "match_uid"),
        "canonical_observations": ("canonical_observations", "observation_uid"),
        "assimilation_proposals": ("assimilation_proposals", "proposal_uid"),
        "assimilation_decisions": ("assimilation_decisions", "decision_uid"),
    }
    blockers: list[dict[str, object]] = []
    for output_key, (graph_key, uid_key) in catalogs.items():
        path = source_output_root / OUTPUT_PATHS[output_key]
        if not path.is_file():
            blockers.append({"code": "missing_pilot_catalog", "message": "Round-13 pilot catalog is missing.", "path": str(path)})
            continue
        rows = read_tsv(path)
        expected = {str(row[uid_key]) for row in graph[graph_key]}
        actual = {str(row.get(uid_key, "")) for row in rows}
        if actual != expected or len(rows) != len(graph[graph_key]):
            blockers.append({"code": "pilot_catalog_drift", "message": "Round-13 pilot catalog differs from the typed graph.", "path": str(path)})
    return blockers


def _schema_registry_blockers(repository_root: Path) -> list[dict[str, object]]:
    config = load_config(repository_root / "Libs/configs/config_stack.yaml")
    registry = config.get("schemas", {}).get("round13_action_observation_schema_registry", {})
    aliases = {**SCHEMA_PATHS, "round13_action_observation_graph": SCHEMA_PATHS["graph"]}
    aliases.pop("graph")
    blockers: list[dict[str, object]] = []
    if set(registry) != set(aliases):
        return [{"code": "canonical_schema_registry_drift", "message": "Round-13 schemas are not exactly registered."}]
    for family, path_value in aliases.items():
        version_family = "graph" if family == "round13_action_observation_graph" else family
        entry = registry[family]
        if entry.get("schema_version") != SCHEMA_VERSIONS[version_family] or entry.get("path") != path_value:
            blockers.append({"code": "canonical_schema_registry_binding", "message": "Round-13 schema binding is not canonical.", "family": family})
            continue
        schema_path = repository_root / path_value
        if not schema_path.is_file():
            blockers.append({"code": "missing_registered_schema", "message": "Registered Round-13 schema is missing.", "path": path_value})
            continue
        Draft202012Validator.check_schema(json.loads(schema_path.read_text(encoding="utf-8")))
    return blockers


def _execution_identity_blockers(
    path: Path | None,
    *,
    repository_root: Path,
    input_bundle_hash: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    identity = _load_json(path)
    if identity is None:
        return ([{"code": "execution_identity_not_exact_clean", "message": "Round-13 exact-clean execution identity is missing."}], {})
    commit = git_commit(repository_root)
    expected = {
        "schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
        "exact_git_commit": commit,
        "parent_commit": _parent_commit(repository_root),
        "dirty_status": "clean",
        "input_bundle_hash": input_bundle_hash,
        "reproduction_relation": "exact_clean_commit",
        "config_hash": sha256_payload(load_config(repository_root / "Libs/configs/config_stack.yaml")),
        "schema_registry_hash": sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        "action_observation_contracts_hash": sha256_file(repository_root / "Libs/utils/data/action_observation_contracts.py"),
        "evidence_relationships_hash": sha256_file(repository_root / "Libs/utils/data/evidence_relationships.py"),
        "evidence_union_hash": sha256_file(repository_root / "Libs/utils/data/evidence_union.py"),
        "builder_hash": sha256_file(repository_root / "Libs/scripts/builders/build_dt_round13_action_observation_contracts.py"),
        "validator_hash": sha256_file(repository_root / "Libs/scripts/validators/validate_dt_round13_action_observation_contracts.py"),
        "test_manifest_hash": test_manifest_hash(repository_root),
    }
    blockers: list[dict[str, object]] = []
    for field, value in expected.items():
        if identity.get(field) != value:
            blockers.append(
                {
                    "code": "execution_identity_not_exact_clean",
                    "message": "Execution identity differs from the exact clean commit.",
                    "field": field,
                    "expected": value,
                    "actual": identity.get(field),
                }
            )
    for field in ("environment_hash",):
        value = identity.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            blockers.append({"code": "execution_identity_not_exact_clean", "message": "Execution environment hash is invalid.", "field": field})
    return blockers, identity


def _prefull_blockers(
    path: Path | None,
    *,
    identity: dict[str, object],
    focused_summary: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    prefull = _load_json(path)
    if prefull is None:
        return ([{"code": "execution_identity_not_exact_clean", "message": "Round-13 pre-full validation artifact is missing."}], {})
    required_checks = {
        "py_compile",
        "ruff",
        "json_schema_validation",
        "canonical_schema_registry_validation",
        "git_diff_check",
        "round13_focused_negative_tests",
        "round11f_fail_closed_tests",
        "evidence_union_tests",
        "config_stack_tests",
        "schema_registry_tests",
        "round11_closure_invariants",
        "round13_pilot_builder",
        "round13_formal_validator_local_gate",
    }
    checks = prefull.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    valid = (
        prefull.get("schema_version") == PREFULL_SCHEMA_VERSION
        and prefull.get("status") == "passed"
        and prefull.get("exact_git_commit") == identity.get("exact_git_commit")
        and prefull.get("dirty_status_after") == "clean"
        and prefull.get("frozen_gap_count") == 20
        and prefull.get("canonical_controller_count") == 1
        and prefull.get("canonical_gate_writer_count") == 1
        and prefull.get("no_eligibility_upgrade") is True
        and prefull.get("no_conflict_loss") is True
        and prefull.get("no_fabricated_pairing") is True
        and required_checks.issubset(checks)
        and all(checks.get(name) == "passed" for name in required_checks)
        and prefull.get("focused_junit_hash") == focused_summary.get("sha256")
    )
    return ([] if valid else [{"code": "execution_identity_not_exact_clean", "message": "Pre-full validation evidence is incomplete or failed."}], prefull)


def _inventory_blockers(
    path: Path | None,
    *,
    full_summary: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    inventory = _load_json(path)
    if inventory is None:
        return ([{"code": "test_inventory_incomplete", "message": "Round-13 test inventory is missing."}], {})
    common = inventory.get("common_tests")
    added = inventory.get("added_tests")
    removed = inventory.get("removed_tests")
    renamed = inventory.get("renamed_tests")
    superseded = inventory.get("superseded_tests")
    valid = (
        inventory.get("schema_version") == TEST_INVENTORY_SCHEMA_VERSION
        and inventory.get("round12_parent_test_count") == 399
        and inventory.get("round13_current_test_count") == full_summary.get("tests")
        and inventory.get("unexplained_missing_tests") == 0
        and inventory.get("mandatory_disappearance") == 0
        and inventory.get("status") == "complete"
        and all(isinstance(value, list) for value in (common, added, removed, renamed, superseded))
    )
    if valid:
        valid = len(common) + len(added) == inventory.get("round13_current_test_count")
        valid = valid and len(common) + len(removed) == inventory.get("round12_parent_test_count")
    return ([] if valid else [{"code": "test_inventory_incomplete", "message": "Round-13 inventory is internally inconsistent."}], inventory)


def _full_execution_blockers(
    path: Path | None,
    *,
    identity: dict[str, object],
    inventory_path: Path | None,
    full_summary: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    execution = _load_json(path)
    if execution is None:
        return ([{"code": "full_suite_not_verified", "message": "Round-13 full-suite execution evidence is missing."}], {})
    result = execution.get("structured_test_result")
    result = result if isinstance(result, dict) else {}
    identity_fields = (
        "exact_git_commit",
        "input_bundle_hash",
        "environment_hash",
        "config_hash",
        "schema_registry_hash",
        "test_manifest_hash",
    )
    streams = execution.get("streams")
    streams = streams if isinstance(streams, dict) else {}
    valid = (
        execution.get("schema_version") == FULL_SUITE_EXECUTION_SCHEMA_VERSION
        and execution.get("terminal_state") == "COMPLETED"
        and execution.get("exit_code") == "0:0"
        and execution.get("pytest_exit_code") == 0
        and execution.get("node") == "cpu04"
        and execution.get("partition") == "cpu"
        and str(execution.get("job_name", "")).startswith("NrnSift_DT_R13_")
        and execution.get("mandatory_skipped") == 0
        and execution.get("core_xfail") == 0
        and result.get("tests") == full_summary.get("tests")
        and result.get("failures") == 0
        and result.get("errors") == 0
        and result.get("skipped") == 0
        and all(execution.get(field) == identity.get(field) for field in identity_fields)
        and inventory_path is not None
        and inventory_path.is_file()
        and execution.get("test_inventory_hash") == sha256_file(inventory_path)
        and all(isinstance(streams.get(name), dict) and streams[name].get("sha256") for name in ("stdout", "stderr", "junit", "scontrol"))
        and execution.get("sacct_status") in {"available", "explicit_slurmdbd_unavailable"}
    )
    return ([] if valid else [{"code": "full_suite_not_verified", "message": "Full-suite terminal or identity evidence is invalid."}], execution)


def validate_round13_outputs(
    *,
    handoff_path: Path,
    expected_handoff_sha256: str,
    round12_gate_path: Path,
    source_output_root: Path,
    focused_junit_path: Path | None,
    prefull_validation_path: Path | None,
    execution_identity_path: Path | None,
    full_suite_junit_path: Path | None,
    full_suite_execution_path: Path | None,
    test_inventory_path: Path | None,
    repository_root: Path = REPO_ROOT,
) -> tuple[dict[str, object], dict[str, object]]:
    handoff = _load_json(handoff_path)
    round12_gate = _load_json(round12_gate_path)
    if handoff is None or round12_gate is None:
        raise ValueError("Round-13 validation requires its accepted Round-12 inputs.")
    handoff_hash = sha256_file(handoff_path)
    round12_gate_hash = sha256_file(round12_gate_path)
    input_bundle_hash = sha256_payload({"round13_handoff": handoff_hash, "round12_gate": round12_gate_hash})
    graph_path = source_output_root / OUTPUT_PATHS["graph"]
    graph = _load_json(graph_path)
    if graph is None:
        raise ValueError("Round-13 action/observation graph is missing.")
    blockers: list[dict[str, object]] = []
    if handoff_hash != expected_handoff_sha256 or graph.get("source_handoff", {}).get("sha256") != expected_handoff_sha256:
        blockers.append({"code": "round13_handoff_hash_mismatch", "message": "Graph does not bind the exact authorized handoff."})
    if round12_gate.get("status") != "typed_evidence_relationships_resolved" or round12_gate.get("overall_gate_status") != "accepted":
        blockers.append({"code": "round12_parent_gate_drift", "message": "Round-12 accepted Gate has drifted."})

    graph_report = validate_round13_graph(
        graph,
        canonical_controller_uid=CANONICAL_CONTROLLER_UID,
        approved_validator_uid=ASSIMILATION_VALIDATOR_UID,
        supported_state_components=SUPPORTED_STATE_COMPONENTS,
    )
    blockers.extend(graph_report["blockers"])
    blockers.extend(_catalog_blockers(graph, source_output_root))
    blockers.extend(_schema_registry_blockers(repository_root))
    gap_path = source_output_root / OUTPUT_PATHS["gaps"]
    gap_rows = read_tsv(gap_path) if gap_path.is_file() else []
    blockers.extend(validate_frozen_gap_action_observation_map(handoff, gap_rows))
    if gap_rows != frozen_gap_action_observation_rows(handoff):
        blockers.append({"code": "frozen_gap_count_drift", "message": "Round-13 gap map content differs from its accepted handoff."})

    focused_summary = _junit_summary(focused_junit_path)
    focused_names = set(focused_summary["testcase_names"])
    negative_verified = (
        focused_summary["available"]
        and focused_summary["failures"] == 0
        and focused_summary["errors"] == 0
        and focused_summary["skipped"] == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(focused_names)
    )
    if not negative_verified:
        blockers.append(
            {
                "code": "mandatory_negative_tests_not_verified",
                "message": "All Round-13 mandatory negative cases require passing JUnit evidence.",
                "missing": sorted(MANDATORY_NEGATIVE_TEST_NAMES - focused_names),
            }
        )

    identity_blockers, identity = _execution_identity_blockers(
        execution_identity_path,
        repository_root=repository_root,
        input_bundle_hash=input_bundle_hash,
    )
    blockers.extend(identity_blockers)
    prefull_blockers, prefull = _prefull_blockers(prefull_validation_path, identity=identity, focused_summary=focused_summary)
    blockers.extend(prefull_blockers)
    full_summary = _junit_summary(full_suite_junit_path)
    inventory_blockers, inventory = _inventory_blockers(test_inventory_path, full_summary=full_summary)
    blockers.extend(inventory_blockers)
    execution_blockers, full_execution = _full_execution_blockers(
        full_suite_execution_path,
        identity=identity,
        inventory_path=test_inventory_path,
        full_summary=full_summary,
    )
    blockers.extend(execution_blockers)
    if not (
        full_summary["available"]
        and full_summary["tests"] >= 399
        and full_summary["failures"] == 0
        and full_summary["errors"] == 0
        and full_summary["skipped"] == 0
    ):
        blockers.append({"code": "full_suite_not_verified", "message": "Full Libs/tests JUnit has not passed without failures or skips."})
    blockers = sorted(
        {json.dumps(row, sort_keys=True): row for row in blockers}.values(),
        key=lambda row: (str(row.get("code", "")), json.dumps(row, sort_keys=True)),
    )
    status = round13_gate_status(blockers)
    report = {
        "schema_version": SCHEMA_VERSIONS["validation"],
        "validator_uid": ASSIMILATION_VALIDATOR_UID,
        "status": "passed" if not blockers else "blocked",
        "gate_assessment": status,
        "graph_uid": graph.get("graph_uid"),
        "object_counts": graph_report["object_counts"],
        "frozen_gap_source_count": 20,
        "frozen_gap_mapped_count": len(gap_rows),
        "frozen_gap_count_preserved": len(gap_rows) == 20,
        "mandatory_negative_case_count": len(MANDATORY_NEGATIVE_TEST_NAMES),
        "mandatory_negative_cases_verified": negative_verified,
        "focused_tests": focused_summary,
        "prefull_validation": prefull,
        "full_suite": full_summary,
        "full_suite_execution": full_execution,
        "test_inventory": inventory,
        "execution_identity": identity
        or {
            "exact_git_commit": git_commit(repository_root),
            "dirty_status": git_dirty_status(repository_root),
            "reproduction_relation": "not_exact_clean_commit",
        },
        "input_bundle_hash": input_bundle_hash,
        "blockers": blockers,
        "blocker_count": len(blockers),
        "pilot_scope": "contract_validation_only_not_evidence_fabric_materialization",
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "round_11_closure": "accepted_unchanged",
        "round_12_gate": "typed_evidence_relationships_resolved_unchanged",
    }

    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    input_hashes = {
        "round13_handoff": handoff_hash,
        "round12_gate": round12_gate_hash,
        "action_observation_graph": sha256_file(graph_path),
        "frozen_gap_map": sha256_file(gap_path),
    }
    for name, path in (
        ("focused_junit", focused_junit_path),
        ("prefull_validation", prefull_validation_path),
        ("execution_identity", execution_identity_path),
        ("full_suite_junit", full_suite_junit_path),
        ("full_suite_execution", full_suite_execution_path),
        ("test_inventory", test_inventory_path),
    ):
        if path is not None and path.is_file():
            input_hashes[name] = sha256_file(path)
    gate = build_canonical_gate_decision(
        authority,
        gate_uid="round_13_action_observation_contracts",
        stage_uid="round_13_action_observation_contracts",
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid=ASSIMILATION_VALIDATOR_UID,
        validator_report={"validator_uid": ASSIMILATION_VALIDATOR_UID, "status": "passed", "assessment": status},
        input_artifact_hashes=input_hashes,
        config_hash=sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        commit=git_commit(repository_root),
        decision="accepted" if not blockers else "blocked",
        blockers=blockers,
        lineage_parent="round_12_typed_evidence_relationships",
    ) | {
        "round_uid": "round_13_action_observation_contracts",
        "status": status,
        "overall_gate_status": "accepted" if not blockers else "blocked",
        "validation_status": report["status"],
        "blocker_count": len(blockers),
        "blocker_classes": sorted({status.removeprefix("blocked_")} if blockers else set()),
        "frozen_gap_count": len(gap_rows),
        "execution_identity": report["execution_identity"],
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "round_11_closure": "accepted_unchanged",
        "round_12_gate": "typed_evidence_relationships_resolved_unchanged",
    }
    return report, gate


def build_round14_handoff(graph: MappingLike, gap_rows: list[dict[str, str]], gate: MappingLike) -> dict[str, object]:
    return {
        "schema_version": "neuronsifter-dt-round14-handoff-v1",
        "status": "ready_for_evidence_fabric_materialization",
        "parent_gate": gate["status"],
        "execution_commit": gate["git_commit"],
        "action_vocabularies": sorted(WORLD_ACTION_TYPES),
        "action_compiler_inputs_outputs": {
            "input": ["WorldAction", "PK/PD context", "baseline state posterior", "graph/context"],
            "output": "ExposureOccupancyProgram",
            "fail_closed_statuses": ["insufficient_parameters", "unsupported_action", "invalid_action"],
        },
        "information_action_semantics": graph["invariants"],
        "observation_request_semantics": graph["observation_requests"],
        "tool_capability_matching": graph["tool_capability_matches"],
        "observation_qc": graph["observation_qc"],
        "assimilation_authority": {
            "controller_uid": CANONICAL_CONTROLLER_UID,
            "validator_uid": ASSIMILATION_VALIDATOR_UID,
            "proposal_state_write_authorized": False,
        },
        "support_domain_rules": graph["support_domains"],
        "conflict_propagation": graph["conflicts"],
        "frozen_gap_action_observation_map": gap_rows,
        "materialization_admission_requirements": [
            "exact_clean_round13_execution_identity",
            "round13_full_suite_completed_zero_failures",
            "typed_action_and_observation_schema_validation",
            "verified_provenance_and_nonrejected_qc",
            "support_domain_compatibility",
            "canonical_controller_assimilation_authority",
            "unresolved_conflict_retention",
            "twenty_frozen_gaps_preserved_until_real_evidence_resolution",
        ],
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }


MappingLike = dict[str, object]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Round-13 action and observation contract artifacts.")
    parser.add_argument("--round13-handoff", required=True)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--round12-gate", required=True)
    parser.add_argument("--source-output-root", default=str(REPO_ROOT))
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    parser.add_argument("--focused-junit")
    parser.add_argument("--prefull-validation")
    parser.add_argument("--execution-identity")
    parser.add_argument("--full-suite-junit")
    parser.add_argument("--full-suite-execution")
    parser.add_argument("--test-inventory")
    args = parser.parse_args()
    source_output_root = Path(args.source_output_root)
    output_root = Path(args.output_root)
    report, gate = validate_round13_outputs(
        handoff_path=Path(args.round13_handoff),
        expected_handoff_sha256=args.expected_handoff_sha256,
        round12_gate_path=Path(args.round12_gate),
        source_output_root=source_output_root,
        focused_junit_path=Path(args.focused_junit) if args.focused_junit else None,
        prefull_validation_path=Path(args.prefull_validation) if args.prefull_validation else None,
        execution_identity_path=Path(args.execution_identity) if args.execution_identity else None,
        full_suite_junit_path=Path(args.full_suite_junit) if args.full_suite_junit else None,
        full_suite_execution_path=Path(args.full_suite_execution) if args.full_suite_execution else None,
        test_inventory_path=Path(args.test_inventory) if args.test_inventory else None,
    )
    write_json(output_root / VALIDATION_PATH, report)
    write_json(output_root / GATE_PATH, gate)
    if gate["status"] == "action_observation_contracts_resolved":
        graph = _load_json(source_output_root / OUTPUT_PATHS["graph"])
        if graph is None:
            raise ValueError("Accepted Gate cannot build Round-14 handoff without its graph.")
        gap_rows = read_tsv(source_output_root / OUTPUT_PATHS["gaps"])
        write_json(output_root / ROUND14_HANDOFF_PATH, build_round14_handoff(graph, gap_rows, gate))
    print(f"{gate['status']} blockers={report['blocker_count']} gaps={report['frozen_gap_mapped_count']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
