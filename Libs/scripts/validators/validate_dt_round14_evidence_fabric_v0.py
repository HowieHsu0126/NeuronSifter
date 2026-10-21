"""Validate Evidence Fabric V0 and issue the canonical Round-14 Gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_dt_round14_evidence_fabric_v0 import ROUND14_VALIDATOR_UID  # noqa: E402
from Libs.utils.data.action_observation_contracts import validate_world_action  # noqa: E402
from Libs.utils.data.evidence_fabric_v0 import (  # noqa: E402
    ADMITTED_STATES,
    canonical_sha256,
    evidence_fabric_v0_config,
    round14_gate_status,
    sha256_file,
    sorted_blockers,
    validate_admission_record,
    validate_effective_sample_audit,
    validate_gap_resolution_evidence,
    validate_independence_grouping,
    validate_release_manifest,
    validate_typed_catalogs,
    validate_v0_config,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json  # noqa: E402
from Libs.utils.exp.orchestration_controller import (  # noqa: E402
    CANONICAL_CONTROLLER_UID,
    build_canonical_gate_decision,
    git_commit,
    git_dirty_status,
    load_authority_contract,
    sha256_payload,
)


AUTHORITY_PATH = "Libs/configs/validation/round11f_controller_authority.yaml"
VALIDATION_PATH = "Output/audits/dt_round14_validation.json"
GATE_PATH = "Output/audits/dt_round14_gate.json"
ROUND15_HANDOFF_PATH = "Output/audits/dt_round15_handoff.json"
EXECUTION_IDENTITY_SCHEMA_VERSION = "neuronsifter-dt-round14-execution-identity-v1"
PREFULL_SCHEMA_VERSION = "neuronsifter-dt-round14-prefull-validation-v1"
FULL_SUITE_EXECUTION_SCHEMA_VERSION = "neuronsifter-dt-round14-full-suite-execution-v1"
TEST_INVENTORY_SCHEMA_VERSION = "neuronsifter-dt-round14-test-inventory-diff-v1"

MANDATORY_NEGATIVE_TEST_NAMES = frozenset(
    {
        "test_unknown_source_admitted_as_grade_a_rejected",
        "test_missing_hash_admitted_rejected",
        "test_unresolved_license_admitted_rejected",
        "test_ambiguous_entity_mapping_silently_collapsed_rejected",
        "test_cross_sectional_evidence_promoted_to_longitudinal_rejected",
        "test_static_endpoint_promoted_to_trajectory_rejected",
        "test_legacy_incomplete_trajectory_promoted_to_e4_rejected",
        "test_simulator_output_promoted_to_experimental_truth_rejected_round14",
        "test_grade_c_numeric_ground_truth_rejected",
        "test_grade_s_headline_training_rejected",
        "test_hypothetical_scenario_admitted_as_observation_rejected_round14",
        "test_evidence_union_eligibility_upgrade_rejected_round14",
        "test_conflict_dropped_during_deduplication_rejected",
        "test_gap_closed_without_admission_evidence_rejected",
        "test_row_count_used_as_effective_sample_count_rejected",
        "test_duplicate_biological_units_split_as_independent_rejected",
    }
)


def _load_json(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _json_cell(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parquet_rows(path: Path) -> list[dict[str, object]]:
    return [{key: _json_cell(value) for key, value in row.items()} for row in pq.read_table(path).to_pylist()]


def _junit_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {"available": False, "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "testcase_names": []}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "available": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "tests": sum(int(suite.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.get("skipped", 0)) for suite in suites),
        "testcase_names": sorted(str(case.get("name", "")) for case in root.iter("testcase")),
    }


def test_manifest_hash(repository_root: Path) -> str:
    entries = [
        {"path": path.relative_to(repository_root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted((repository_root / "Libs/tests").glob("test_*.py"))
    ]
    return sha256_payload(entries)


def expected_execution_identity_fields(repository_root: Path) -> dict[str, object]:
    return {
        "schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
        "exact_git_commit": git_commit(repository_root),
        "parent_commit": subprocess.run(
            ["git", "rev-parse", "HEAD^"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "dirty_status": "clean",
        "config_hash": sha256_payload(load_config(repository_root / "Libs/configs/config_stack.yaml")),
        "schema_registry_hash": sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        "evidence_relationships_hash": sha256_file(repository_root / "Libs/utils/data/evidence_relationships.py"),
        "evidence_union_hash": sha256_file(repository_root / "Libs/utils/data/evidence_union.py"),
        "evidence_fabric_contracts_hash": sha256_file(repository_root / "Libs/utils/data/evidence_fabric_v0.py"),
        "builder_hash": sha256_file(
            repository_root / "Libs/scripts/builders/build_dt_round14_evidence_fabric_v0.py"
        ),
        "validator_hash": sha256_file(
            repository_root / "Libs/scripts/validators/validate_dt_round14_evidence_fabric_v0.py"
        ),
        "test_manifest_hash": test_manifest_hash(repository_root),
        "reproduction_relation": "exact_clean_commit",
        "execution_ref": "refs/heads/round14-execution",
        "index_lock_absent": True,
        "active_git_writer_absent": True,
    }


def _schema_blockers(config: dict[str, object], repository_root: Path) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    registry = config["schemas"].get("round14_evidence_fabric_schema_registry", {})
    if set(registry) != {"evidence_fabric_admission_record", "evidence_fabric_release_manifest"}:
        blockers.append({"code": "schema_registry_incomplete", "message": "Round-14 schemas are not canonically registered."})
        return blockers
    for name, entry in registry.items():
        path = repository_root / str(entry["path"])
        if not path.is_file():
            blockers.append({"code": "schema_registry_missing_path", "message": "Registered schema path is absent.", "schema": name})
            continue
        schema = json.loads(path.read_text(encoding="utf-8"))
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:  # jsonschema provides heterogeneous schema exceptions
            blockers.append({"code": "schema_registry_invalid", "message": str(exc), "schema": name})
        if schema.get("properties", {}).get("schema_version", {}).get("const") != entry.get("schema_version"):
            blockers.append({"code": "schema_registry_version_mismatch", "message": "Schema version differs from registry.", "schema": name})
    return blockers


def _artifact_paths(config: dict[str, object], source_output_root: Path) -> dict[str, Path]:
    paths = {
        name: source_output_root / str(path)
        for name, path in evidence_fabric_v0_config(config)["output_paths"].items()
    }
    paths.update(
        {
            "gap_ledger": source_output_root / "Output/audits/dt_round14_frozen_gap_resolution_ledger.tsv",
            "independence": source_output_root / "Output/audits/dt_round14_independence_group_catalog.tsv",
            "e0_validation": source_output_root / "Output/audits/dt_round14_e0_validation.json",
            "duplicate_audit": source_output_root / "Output/audits/dt_round14_duplicate_audit.json",
            "conflict_audit": source_output_root / "Output/audits/dt_round14_conflict_audit.json",
            "eligibility_audit": source_output_root / "Output/audits/dt_round14_eligibility_audit.json",
            "coverage": source_output_root / "Output/audits/dt_round14_evidence_fabric_coverage.json",
            "effective": source_output_root / "Output/audits/dt_round14_effective_sample_audit.json",
            "leakage": source_output_root / "Output/audits/dt_round14_leakage_readiness.json",
        }
    )
    return paths


def validate_materialized_release(
    *, config: dict[str, object], repository_root: Path, source_output_root: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    blockers = [*validate_v0_config(config), *_schema_blockers(config, repository_root)]
    paths = _artifact_paths(config, source_output_root)
    for name, path in paths.items():
        if not path.is_file():
            blockers.append({"code": "missing_materialization_artifact", "message": "Required Round-14 artifact is absent.", "artifact": name, "path": str(path)})
    if any(row["code"] == "missing_materialization_artifact" for row in blockers):
        return sorted_blockers(blockers), {"paths": {name: str(path) for name, path in paths.items()}}
    manifest = _load_json(paths["release_manifest"])
    assert manifest is not None
    blockers.extend(validate_release_manifest(manifest))
    release_schema = json.loads(
        (repository_root / "Libs/configs/schema/evidence_fabric_release_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for error in Draft202012Validator(release_schema).iter_errors(manifest):
        blockers.append(
            {
                "code": "release_manifest_schema_invalid",
                "message": error.message,
                "path": "/".join(str(part) for part in error.absolute_path),
            }
        )
    expected_hashes = {
        "source_manifest_hash": sha256_file(paths["source_manifest"]),
        "node_catalog_hash": sha256_file(paths["node_catalog"]),
        "relation_catalog_hash": sha256_file(paths["relation_catalog"]),
        "action_catalog_hash": sha256_file(paths["world_action_catalog"]),
        "observation_catalog_hash": sha256_file(paths["canonical_observation_catalog"]),
        "conflict_catalog_hash": sha256_file(paths["conflict_catalog"]),
        "gap_ledger_hash": sha256_file(paths["gap_ledger"]),
    }
    for field, digest in expected_hashes.items():
        if manifest.get(field) != digest:
            blockers.append({"code": "release_artifact_hash_mismatch", "message": "Release manifest artifact hash is stale.", "field": field})
    admissions = _parquet_rows(paths["admission_ledger"])
    rejection_rows = _parquet_rows(paths["rejection_ledger"])
    admission_schema = json.loads(
        (repository_root / "Libs/configs/schema/evidence_fabric_admission_record.schema.json").read_text(
            encoding="utf-8"
        )
    )
    admission_schema_validator = Draft202012Validator(admission_schema)
    for row in admissions:
        blockers.extend(validate_admission_record(row))
        for error in admission_schema_validator.iter_errors(row):
            blockers.append(
                {
                    "code": "admission_schema_invalid",
                    "message": error.message,
                    "candidate_uid": row.get("candidate_uid"),
                    "path": "/".join(str(part) for part in error.absolute_path),
                }
            )
    rejected_uids = {row["candidate_uid"] for row in admissions if row["terminal_state"] not in ADMITTED_STATES}
    if rejected_uids != {row["candidate_uid"] for row in rejection_rows}:
        blockers.append({"code": "rejection_ledger_incomplete", "message": "Rejected/quarantined candidates are not preserved exactly once."})
    e4 = [row for row in admissions if row["layer"] == "E4" and row["terminal_state"] in ADMITTED_STATES]
    if e4:
        blockers.append({"code": "e4_unverified_materialization", "message": "No pre-existing action-complete E4 trajectory was authorized for this release."})
    for row in admissions:
        eligibility = dict(row.get("eligibility", {}))
        if row["layer"] == "E7" and (
            row.get("empirical_truth") is True
            or any(eligibility.get(key) is True for key in (
                "eligible_for_state_pretraining", "eligible_for_action_endpoint_training",
                "eligible_for_trajectory_training", "eligible_for_calibration",
                "eligible_for_validation", "eligible_for_claim_support",
            ))
        ):
            blockers.append({"code": "hypothetical_boundary_violation", "message": "E7 entered an empirical role.", "candidate_uid": row["candidate_uid"]})
        if row["layer"] == "E3" and row["terminal_state"] in ADMITTED_STATES and row.get("legacy_role") not in {
            "legacy_dynamics_pretraining", "numerical_dynamics_only", "stress_only", "quarantined"
        }:
            blockers.append({"code": "legacy_role_unrestricted", "message": "E3 lacks a restricted legacy role."})
        if row["qc_grade"] == "C" and any(
            eligibility.get(key) is True for key in (
                "eligible_for_state_pretraining", "eligible_for_action_endpoint_training", "eligible_for_trajectory_training"
            )
        ):
            blockers.append({"code": "grade_c_strong_numeric_role", "message": "Grade C entered strong numeric supervision."})
        if row["qc_grade"] == "S" and any(
            eligibility.get(key) is True for key in (
                "eligible_for_state_pretraining", "eligible_for_action_endpoint_training", "eligible_for_trajectory_training",
                "eligible_for_claim_support",
            )
        ):
            blockers.append({"code": "grade_s_headline_role", "message": "Grade S entered a headline role."})
    action_rows = _parquet_rows(paths["world_action_catalog"])
    for row in action_rows:
        if row.get("admission_decision") in ADMITTED_STATES:
            action = {key: row.get(key) for key in row if key not in {"layer", "source_artifact", "content_hash", "admission_decision"}}
            if validate_world_action(action):
                blockers.append({"code": "admitted_action_contract_invalid", "message": "Admitted action fails Round-13 validation.", "action_uid": row.get("action_uid")})
    blockers.extend(
        validate_typed_catalogs(
            nodes=_parquet_rows(paths["node_catalog"]),
            relations=_parquet_rows(paths["relation_catalog"]),
            pairings=_parquet_rows(paths["pairing_catalog"]),
            support_domains=_parquet_rows(paths["support_domain_catalog"]),
            eligibilities=_parquet_rows(paths["eligibility_catalog"]),
            conflicts=_parquet_rows(paths["conflict_catalog"]),
        )
    )
    conflict_audit = _load_json(paths["conflict_audit"])
    if not conflict_audit or conflict_audit.get("typed_conflicts_preserved") is not True or conflict_audit.get("source_typed_conflict_count") != conflict_audit.get("preserved_typed_conflict_count"):
        blockers.append({"code": "conflict_loss", "message": "Typed conflicts were not preserved exactly."})
    eligibility_audit = _load_json(paths["eligibility_audit"])
    if not eligibility_audit or any(
        eligibility_audit.get(field) != 0
        for field in (
            "evidence_union_eligibility_upgrade_count", "evidence_union_qc_upgrade_count",
            "evidence_union_pairing_creation_count", "grade_c_strong_numeric_count",
            "hypothetical_empirical_truth_count", "legacy_headline_action_count",
        )
    ):
        blockers.append({"code": "eligibility_upgrade", "message": "Eligibility propagation audit is not fail-closed."})
    gaps = read_tsv(paths["gap_ledger"])
    blockers.extend(validate_gap_resolution_evidence(gaps, admissions))
    if len(gaps) != 20 or any(row.get("current_status") != "frozen_unresolved" or row.get("resolution_evidence") for row in gaps):
        blockers.append({"code": "gap_fabrication", "message": "All 20 gaps must remain explicitly unresolved without fabricated evidence."})
    independence = read_tsv(paths["independence"])
    blockers.extend(validate_independence_grouping(independence))
    leakage = _load_json(paths["leakage"])
    if not leakage or leakage.get("status") != "passed" or leakage.get("duplicate_biological_units_independent") is not False:
        blockers.append({"code": "independence_grouping_invalid", "message": "Independence grouping audit did not pass."})
    effective = _load_json(paths["effective"])
    blockers.extend(validate_effective_sample_audit(effective or {}))
    if not effective or effective.get("row_count_not_used_as_effective_sample_count") is not True:
        blockers.append({"code": "effective_sample_count_invalid", "message": "Row count was used as effective sample count."})
    e0 = _load_json(paths["e0_validation"])
    if not e0 or e0.get("status") != "passed" or e0.get("ambiguous_mapping_silently_collapsed_count") != 0:
        blockers.append({"code": "canonicalization_invalid", "message": "E0 canonicalization did not pass."})
    coverage = _load_json(paths["coverage"])
    required_coverage = {
        "sources",
        "publications_or_accessions",
        "accessions",
        "studies",
        "biological_units",
        "modalities",
        "organisms",
        "diseases",
        "regions",
        "compartments",
        "cell_types",
        "actions",
        "dose_route_schedule_combinations",
        "perturbation_endpoints_admitted",
        "trajectory_families",
        "simulator_families",
        "mechanistic_models",
        "tool_families",
        "qc_distribution",
        "eligibility_distribution",
        "conflict_distribution",
        "unresolved_gaps",
    }
    if (
        not coverage
        or coverage.get("status") != "materialized_unsealed"
        or coverage.get("sealed") is not False
        or required_coverage - set(coverage)
    ):
        blockers.append(
            {
                "code": "coverage_audit_incomplete",
                "message": "Evidence Fabric coverage dimensions are incomplete.",
                "missing": sorted(required_coverage - set(coverage or {})),
            }
        )
    summary = {
        "release_uid": manifest.get("release_uid"),
        "code_commit": manifest.get("code_commit"),
        "config_hash": manifest.get("config_hash"),
        "schema_registry_hash": manifest.get("schema_registry_hash"),
        "input_bundle_hash": manifest.get("input_bundle_hash"),
        "admission_count": len(admissions),
        "admitted_count": sum(1 for row in admissions if row["terminal_state"] in ADMITTED_STATES),
        "rejected_or_quarantined_count": len(rejected_uids),
        "layer_counts": dict(sorted(Counter(str(row["layer"]) for row in admissions).items())),
        "terminal_counts": dict(sorted(Counter(str(row["terminal_state"]) for row in admissions).items())),
        "gap_count": len(gaps),
        "e4_admitted_count": len(e4),
        "artifact_hashes": {name: sha256_file(path) for name, path in paths.items()},
    }
    return sorted_blockers(blockers), summary


def _execution_identity_blockers(path: Path | None, repository_root: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    value = _load_json(path)
    if value is None:
        return [{"code": "execution_identity_missing", "message": "Exact-clean execution identity is missing."}], None
    blockers: list[dict[str, object]] = []
    for field, expected in expected_execution_identity_fields(repository_root).items():
        if value.get(field) != expected:
            blockers.append(
                {
                    "code": "execution_identity_not_exact_clean",
                    "message": "Execution identity differs from the exact clean commit.",
                    "field": field,
                    "expected": expected,
                    "actual": value.get(field),
                }
            )
    environment_components = value.get("environment_components")
    if not isinstance(environment_components, dict) or value.get("environment_hash") != canonical_sha256(
        environment_components or {}
    ):
        blockers.append(
            {
                "code": "execution_environment_hash_mismatch",
                "message": "Execution environment hash is not content-addressed to its components.",
            }
        )
    if not isinstance(value.get("input_bundle_hash"), str) or len(str(value.get("input_bundle_hash"))) != 64:
        blockers.append(
            {
                "code": "execution_input_bundle_hash_invalid",
                "message": "Execution identity input bundle hash is absent or invalid.",
            }
        )
    return blockers, value


def _inventory_blockers(path: Path | None, full: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    value = _load_json(path)
    if value is None:
        return [{"code": "inventory_missing", "message": "Round-14 inventory reconciliation is missing."}], None
    blockers: list[dict[str, object]] = []
    if value.get("schema_version") != TEST_INVENTORY_SCHEMA_VERSION:
        blockers.append({"code": "inventory_schema_invalid", "message": "Round-14 inventory schema is invalid."})
    if value.get("round13_parent_test_count") != 437 or value.get("unexplained_missing_tests") != 0 or value.get("mandatory_disappearance") != 0:
        blockers.append({"code": "inventory_mandatory_disappearance", "message": "Mandatory parent tests disappeared."})
    if full.get("available") and value.get("round14_current_test_count") != full.get("tests"):
        blockers.append({"code": "inventory_count_mismatch", "message": "Inventory count differs from full JUnit."})
    return blockers, value


def _identity_materialization_blockers(
    identity: dict[str, object] | None, materialization: dict[str, object]
) -> list[dict[str, object]]:
    if identity is None:
        return []
    blockers: list[dict[str, object]] = []
    for identity_field, materialization_field in (
        ("exact_git_commit", "code_commit"),
        ("config_hash", "config_hash"),
        ("input_bundle_hash", "input_bundle_hash"),
        ("schema_registry_hash", "schema_registry_hash"),
    ):
        if identity.get(identity_field) != materialization.get(materialization_field):
            blockers.append(
                {
                    "code": "execution_materialization_identity_mismatch",
                    "message": "Materialized release does not match the exact execution identity.",
                    "field": identity_field,
                    "identity": identity.get(identity_field),
                    "materialization": materialization.get(materialization_field),
                }
            )
    return blockers


def build_prefull_report(
    *, materialization_blockers: list[dict[str, object]], materialization_summary: dict[str, object], focused: dict[str, object], identity: dict[str, object] | None, identity_blockers: list[dict[str, object]], execution_dirty_status: str
) -> dict[str, object]:
    focused_names = set(focused["testcase_names"])
    focused_ok = (
        focused["available"] and focused["failures"] == 0 and focused["errors"] == 0 and focused["skipped"] == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(focused_names)
    )
    blockers = [*materialization_blockers, *identity_blockers]
    blockers.extend(_identity_materialization_blockers(identity, materialization_summary))
    if not focused_ok:
        blockers.append({"code": "focused_negative_tests_not_verified", "message": "Mandatory Round-14 negative cases did not all pass.", "missing": sorted(MANDATORY_NEGATIVE_TEST_NAMES - focused_names)})
    if execution_dirty_status != "clean":
        blockers.append({"code": "execution_worktree_dirty", "message": "Execution worktree became dirty."})
    blockers = sorted_blockers(blockers)
    return {
        "schema_version": PREFULL_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "materialization": materialization_summary,
        "focused_tests": focused,
        "mandatory_negative_cases_verified": focused_ok,
        "execution_identity": identity,
        "execution_worktree_status": execution_dirty_status,
        "frozen_gap_count": materialization_summary.get("gap_count"),
        "evidence_fabric": {"materialization": "materialized_unsealed", "sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "blockers": blockers,
        "blocker_count": len(blockers),
    }


def validate_round14_outputs(
    *,
    config: dict[str, object],
    repository_root: Path,
    source_output_root: Path,
    focused_junit_path: Path | None,
    prefull_validation_path: Path | None,
    execution_identity_path: Path | None,
    full_suite_junit_path: Path | None,
    full_suite_execution_path: Path | None,
    test_inventory_path: Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    blockers, materialization = validate_materialized_release(
        config=config, repository_root=repository_root, source_output_root=source_output_root
    )
    identity_blockers, identity = _execution_identity_blockers(execution_identity_path, repository_root)
    blockers.extend(identity_blockers)
    blockers.extend(_identity_materialization_blockers(identity, materialization))
    focused = _junit_summary(focused_junit_path)
    focused_names = set(focused["testcase_names"])
    if not (
        focused["available"] and focused["failures"] == 0 and focused["errors"] == 0 and focused["skipped"] == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(focused_names)
    ):
        blockers.append({"code": "focused_negative_tests_not_verified", "message": "Mandatory focused negative tests are not verified."})
    prefull = _load_json(prefull_validation_path)
    if not prefull or prefull.get("schema_version") != PREFULL_SCHEMA_VERSION or prefull.get("status") != "passed":
        blockers.append({"code": "prefull_not_verified", "message": "Exact-clean pre-full Gate did not pass."})
    full = _junit_summary(full_suite_junit_path)
    if not (full["available"] and full["tests"] >= 437 and full["failures"] == 0 and full["errors"] == 0 and full["skipped"] == 0):
        blockers.append({"code": "full_suite_not_verified", "message": "Full Libs/tests suite is not verified."})
    inventory_blockers, inventory = _inventory_blockers(test_inventory_path, full)
    blockers.extend(inventory_blockers)
    full_execution = _load_json(full_suite_execution_path)
    if not full_execution or full_execution.get("schema_version") != FULL_SUITE_EXECUTION_SCHEMA_VERSION:
        blockers.append({"code": "execution_full_suite_missing", "message": "Full-suite execution evidence is missing."})
    elif not (
        full_execution.get("terminal_state") == "COMPLETED"
        and full_execution.get("exit_code") == "0:0"
        and identity
        and full_execution.get("exact_git_commit") == identity.get("exact_git_commit")
    ):
        blockers.append({"code": "execution_full_suite_failed", "message": "Full-suite Slurm terminal evidence did not pass."})
    blockers = sorted_blockers(blockers)
    status = round14_gate_status(blockers)
    report = {
        "schema_version": "neuronsifter-dt-round14-validation-v1",
        "validator_uid": ROUND14_VALIDATOR_UID,
        "status": "passed" if not blockers else "blocked",
        "gate_assessment": status,
        "materialization": materialization,
        "focused_tests": focused,
        "prefull_validation": prefull,
        "execution_identity": identity or {"exact_git_commit": git_commit(repository_root), "dirty_status": git_dirty_status(repository_root), "reproduction_relation": "not_exact_clean_commit"},
        "full_suite": full,
        "full_suite_execution": full_execution,
        "test_inventory": inventory,
        "frozen_gap_count": materialization.get("gap_count", 0),
        "evidence_fabric": {"materialization": "materialized_unsealed", "sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "blockers": blockers,
        "blocker_count": len(blockers),
    }
    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    input_hashes = {
        name: value
        for name, value in materialization.get("artifact_hashes", {}).items()
        if isinstance(value, str)
    }
    for name, path in (
        ("focused_junit", focused_junit_path), ("prefull_validation", prefull_validation_path),
        ("execution_identity", execution_identity_path), ("full_suite_junit", full_suite_junit_path),
        ("full_suite_execution", full_suite_execution_path), ("test_inventory", test_inventory_path),
    ):
        if path and path.is_file():
            input_hashes[name] = sha256_file(path)
    gate = build_canonical_gate_decision(
        authority,
        gate_uid="round_14_evidence_fabric_v0",
        stage_uid="round_14_evidence_fabric_v0",
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid=ROUND14_VALIDATOR_UID,
        validator_report={"validator_uid": ROUND14_VALIDATOR_UID, "status": report["status"], "assessment": status},
        input_artifact_hashes=input_hashes,
        config_hash=sha256_file(repository_root / "Libs/configs/data/digital_twin_evidence_fabric.yaml"),
        commit=git_commit(repository_root),
        decision="accepted" if not blockers else "blocked",
        blockers=blockers,
        lineage_parent="round_13_action_observation_contracts",
    ) | {
        "round_uid": "round_14_evidence_fabric_v0",
        "status": status,
        "overall_gate_status": "accepted" if not blockers else "blocked",
        "validation_status": report["status"],
        "blocker_count": len(blockers),
        "blocker_classes": sorted({status.removeprefix("blocked_")} if blockers else set()),
        "frozen_gap_count": materialization.get("gap_count", 0),
        "execution_identity": report["execution_identity"],
        "evidence_fabric": {"materialization": "evidence_fabric_v0_materialized" if not blockers else "materialized_unsealed", "sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "round_11_closure": "accepted_unchanged",
        "round_12_gate": "typed_evidence_relationships_resolved_unchanged",
        "round_13_gate": "action_observation_contracts_resolved_unchanged",
    }
    return report, gate


def build_round15_handoff(report: dict[str, object], gate: dict[str, object], source_output_root: Path) -> dict[str, object]:
    release = _load_json(source_output_root / "Output/evidence_fabric/v0/evidence_fabric_release_manifest.json")
    coverage = _load_json(source_output_root / "Output/audits/dt_round14_evidence_fabric_coverage.json")
    gaps = read_tsv(source_output_root / "Output/audits/dt_round14_frozen_gap_resolution_ledger.tsv")
    return {
        "schema_version": "neuronsifter-dt-round15-handoff-v1",
        "status": "ready_for_clean_action_complete_trajectory_construction",
        "parent_gate": gate["status"],
        "execution_commit": gate["git_commit"],
        "evidence_fabric_v0_manifest": release,
        "admitted_evidence_roles": report["materialization"]["terminal_counts"],
        "resolved_gaps": [row for row in gaps if str(row["current_status"]).startswith("resolved_")],
        "unresolved_gaps": [row for row in gaps if not str(row["current_status"]).startswith("resolved_")],
        "action_complete_endpoint_coverage": coverage.get("perturbation_endpoints_admitted", 0) if coverage else 0,
        "legacy_trajectory_restrictions": ["legacy_dynamics_pretraining", "numerical_dynamics_only", "stress_only", "quarantined"],
        "candidate_mechanistic_models": coverage.get("mechanistic_models", 0) if coverage else 0,
        "available_simulator_tool_families": {"trajectory_families": coverage.get("trajectory_families", 0) if coverage else 0, "tool_contract_items": coverage.get("tool_contract_items", 0) if coverage else 0},
        "required_clean_trajectory_domains": ["action_complete", "provenance_complete", "support_domain_explicit", "conflicts_preserved"],
        "trajectory_admission_contract": "Round-14 E4 admission contract",
        "minimum_dt_core_trajectory_evidence": "not_evaluated_in_round14",
        "evidence_fabric": {"materialization": "evidence_fabric_v0_materialized", "sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the unsealed Round-14 Evidence Fabric V0 release.")
    parser.add_argument("--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--source-output-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--focused-junit")
    parser.add_argument("--execution-identity")
    parser.add_argument("--prefull-validation")
    parser.add_argument("--full-suite-junit")
    parser.add_argument("--full-suite-execution")
    parser.add_argument("--test-inventory")
    parser.add_argument("--prefull-only", action="store_true")
    args = parser.parse_args()
    repository_root = Path(args.repository_root)
    source_output_root = Path(args.source_output_root)
    output_root = Path(args.output_root)
    config = load_config(Path(args.config))
    materialization_blockers, materialization = validate_materialized_release(
        config=config, repository_root=repository_root, source_output_root=source_output_root
    )
    identity_blockers, identity = _execution_identity_blockers(
        Path(args.execution_identity) if args.execution_identity else None, repository_root
    )
    focused = _junit_summary(Path(args.focused_junit) if args.focused_junit else None)
    if args.prefull_only:
        prefull = build_prefull_report(
            materialization_blockers=materialization_blockers,
            materialization_summary=materialization,
            focused=focused,
            identity=identity,
            identity_blockers=identity_blockers,
            execution_dirty_status=git_dirty_status(repository_root),
        )
        write_json(output_root / "Output/audits/dt_round14_prefull_validation.json", prefull)
        print(f"round14_prefull_{prefull['status']} blockers={prefull['blocker_count']}")
        return 0 if prefull["status"] == "passed" else 1
    report, gate = validate_round14_outputs(
        config=config,
        repository_root=repository_root,
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
    if gate["status"] == "evidence_fabric_v0_materialized":
        write_json(output_root / ROUND15_HANDOFF_PATH, build_round15_handoff(report, gate, source_output_root))
    print(f"{gate['status']} blockers={report['blocker_count']} gaps={report['frozen_gap_count']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
