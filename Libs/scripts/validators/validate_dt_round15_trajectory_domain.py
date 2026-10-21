"""Validate Round-15 E4 qualification and issue the canonical Gate."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_dt_round15_trajectory_domain import (  # noqa: E402
    PRODUCER_VALIDATOR_UID,
    sha256_file,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json  # noqa: E402
from Libs.utils.data.trajectory_contracts import (  # noqa: E402
    canonical_sha256,
    release_content_hash,
    round15_config,
    sorted_blockers,
    validate_action_binding,
    validate_admission_decision,
    validate_capability_matrix,
    validate_constraint_qc,
    validate_failure_ledger,
    validate_gap_trajectory_map,
    validate_independence_groups,
    validate_initial_state,
    validate_numerical_qc,
    validate_producer,
    validate_release_candidate,
    validate_round15_config,
    validate_trajectory,
)
from Libs.utils.exp.orchestration_controller import (  # noqa: E402
    CANONICAL_CONTROLLER_UID,
    build_canonical_gate_decision,
    git_commit,
    git_dirty_status,
    load_authority_contract,
    sha256_payload,
)


AUTHORITY_PATH = "Libs/configs/validation/round11f_controller_authority.yaml"
VALIDATION_PATH = "Output/audits/dt_round15_validation.json"
GATE_PATH = "Output/audits/dt_round15_gate.json"
HANDOFF_PATH = "Output/audits/dt_round16_handoff.json"
EXECUTION_IDENTITY_SCHEMA_VERSION = "neuronsifter-dt-round15-execution-identity-v1"
PREFULL_SCHEMA_VERSION = "neuronsifter-dt-round15-prefull-validation-v1"
FULL_SUITE_EXECUTION_SCHEMA_VERSION = "neuronsifter-dt-round15-full-suite-execution-v1"
TEST_INVENTORY_SCHEMA_VERSION = "neuronsifter-dt-round15-test-inventory-diff-v1"

MANDATORY_NEGATIVE_TEST_NAMES = frozenset(
    {
        "test_trajectory_without_complete_initial_state_rejected",
        "test_trajectory_without_complete_action_rejected",
        "test_drug_id_used_as_full_action_rejected_round15",
        "test_missing_dose_unit_rejected_round15",
        "test_missing_required_route_rejected_round15",
        "test_unknown_simulator_version_rejected",
        "test_unknown_parameter_provenance_rejected_round15",
        "test_silent_default_initial_state_rejected",
        "test_silent_default_action_parameters_rejected",
        "test_static_endpoint_promoted_to_trajectory_rejected_round15",
        "test_e3_promoted_without_regeneration_rejected",
        "test_hypothetical_scenario_promoted_to_e4_rejected",
        "test_failed_solver_output_admitted_as_strong_rejected",
        "test_nan_inf_trajectory_admitted_rejected",
        "test_constraint_failure_ignored_rejected",
        "test_unpaired_action_trajectory_contrast_rejected",
        "test_different_seed_roles_treated_as_identical_samples_rejected",
        "test_low_fidelity_silently_substituted_rejected",
        "test_unknown_support_domain_admitted_rejected_round15",
        "test_missing_artifact_hash_admitted_rejected_round15",
        "test_failed_attempt_overwritten_by_retry_rejected",
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
    return [
        {key: _json_cell(value) for key, value in row.items()}
        for row in pq.read_table(path).to_pylist()
    ]


def _junit_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "testcase_names": [],
        }
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
        "config_hash": canonical_sha256(load_config(repository_root / "Libs/configs/config_stack.yaml")),
        "schema_registry_hash": sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        "trajectory_contracts_hash": sha256_file(repository_root / "Libs/utils/data/trajectory_contracts.py"),
        "dynamics_fields_hash": sha256_file(repository_root / "Libs/utils/model/dynamics_fields.py"),
        "builder_hash": sha256_file(
            repository_root / "Libs/scripts/builders/build_dt_round15_trajectory_domain.py"
        ),
        "validator_hash": sha256_file(
            repository_root / "Libs/scripts/validators/validate_dt_round15_trajectory_domain.py"
        ),
        "test_manifest_hash": test_manifest_hash(repository_root),
        "reproduction_relation": "exact_clean_commit",
        "execution_ref": "refs/heads/round15-execution",
        "index_lock_absent": True,
        "active_git_writer_absent": True,
    }


def _execution_identity_blockers(
    path: Path | None,
    repository_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    value = _load_json(path)
    if value is None:
        return [{"code": "execution_identity_missing", "message": "Exact-clean Round-15 execution identity is missing."}], None
    blockers: list[dict[str, object]] = []
    for field, expected in expected_execution_identity_fields(repository_root).items():
        if value.get(field) != expected:
            blockers.append(
                {
                    "code": "execution_identity_not_exact_clean",
                    "message": "Execution identity differs from the exact clean Round-15 commit.",
                    "field": field,
                    "expected": expected,
                    "actual": value.get(field),
                }
            )
    components = value.get("environment_components")
    if not isinstance(components, dict) or value.get("environment_hash") != canonical_sha256(components or {}):
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
                "message": "Execution input bundle hash is absent or invalid.",
            }
        )
    return sorted_blockers(blockers), value


def _schema_blockers(config: Mapping[str, object], repository_root: Path) -> list[dict[str, object]]:
    expected = {
        "clean_action_complete_trajectory",
        "trajectory_producer",
        "trajectory_producer_qualification",
        "trajectory_initial_state",
        "trajectory_action_binding",
        "trajectory_numerical_qc",
        "trajectory_constraint_qc",
        "trajectory_control_pair",
        "trajectory_failure",
        "trajectory_admission_decision",
        "trajectory_family_manifest",
        "round15_e4_release_candidate",
    }
    registry = dict(config["schemas"].get("round15_trajectory_schema_registry", {}))
    blockers: list[dict[str, object]] = []
    if set(registry) != expected:
        blockers.append(
            {
                "code": "schema_registry_incomplete",
                "message": "Round-15 schemas are not registered exactly once in the canonical registry.",
                "missing": sorted(expected - set(registry)),
                "unexpected": sorted(set(registry) - expected),
            }
        )
        return blockers
    for name, entry_value in registry.items():
        entry = dict(entry_value)
        path = repository_root / str(entry["path"])
        if not path.is_file():
            blockers.append({"code": "schema_registry_missing_path", "message": "Registered schema path is absent.", "schema": name})
            continue
        schema = json.loads(path.read_text(encoding="utf-8"))
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:
            blockers.append({"code": "schema_registry_invalid", "message": str(exc), "schema": name})
        if schema.get("properties", {}).get("schema_version", {}).get("const") != entry.get("schema_version"):
            blockers.append({"code": "schema_registry_version_mismatch", "message": "Schema version differs from registry.", "schema": name})
    return sorted_blockers(blockers)


def _output_paths(config: Mapping[str, object], output_root: Path) -> dict[str, Path]:
    return {
        name: output_root / str(path)
        for name, path in dict(round15_config(config)["output_paths"]).items()
    }


def _schema_validate(
    *,
    repository_root: Path,
    config: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    registry = dict(config["schemas"]["round15_trajectory_schema_registry"])
    for name, value in objects.items():
        if name not in registry:
            blockers.append({"code": "schema_object_unregistered", "message": "Materialized object has no registered schema.", "schema": name})
            continue
        schema = json.loads((repository_root / str(registry[name]["path"])).read_text(encoding="utf-8"))
        for error in Draft202012Validator(schema).iter_errors(dict(value)):
            blockers.append(
                {
                    "code": "schema_validation_failed",
                    "message": error.message,
                    "schema": name,
                    "path": ".".join(str(part) for part in error.absolute_path),
                }
            )
    return sorted_blockers(blockers)


def validate_pilot_outputs(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    source_output_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    paths = _output_paths(config, source_output_root)
    blockers = [*validate_round15_config(config), *_schema_blockers(config, repository_root)]
    required = (
        "capability_matrix",
        "producer_qualification_tsv",
        "producer_qualification_json",
        "pilot_manifest",
        "pilot_validation",
        "production_plan",
        "failure_ledger",
    )
    missing = [name for name in required if not paths[name].is_file()]
    if missing:
        blockers.append({"code": "pilot_artifact_missing", "message": "Round-15 pilot artifacts are missing.", "artifacts": missing})
        return sorted_blockers(blockers), {"artifact_hashes": {}, "missing": missing}
    capability = read_tsv(paths["capability_matrix"])
    qualification = _load_json(paths["producer_qualification_json"]) or {}
    pilot = _load_json(paths["pilot_manifest"]) or {}
    pilot_validation = _load_json(paths["pilot_validation"]) or {}
    production_plan = _load_json(paths["production_plan"]) or {}
    failure_rows = read_tsv(paths["failure_ledger"])
    producer = dict(pilot.get("producer", {}))
    initial_state = dict(pilot.get("initial_state", {}))
    action_binding = dict(pilot.get("action_binding", {}))
    trajectory = dict(pilot.get("trajectory", {}))
    numerical_path = source_output_root / str(trajectory.get("numerical_qc_ref", ""))
    constraint_path = source_output_root / str(trajectory.get("physical_qc_ref", ""))
    state_path = source_output_root / str(trajectory.get("state_values_ref", ""))
    numerical = _load_json(numerical_path) or {}
    constraint = _load_json(constraint_path) or {}
    state_values = np.load(state_path, allow_pickle=False) if state_path.is_file() else None
    blockers.extend(validate_capability_matrix(capability))
    blockers.extend(validate_producer(producer, known_support_domains={str(producer.get("support_domain_ref", ""))}))
    blockers.extend(validate_initial_state(initial_state, known_support_domains={str(producer.get("support_domain_ref", ""))}))
    blockers.extend(validate_action_binding(action_binding, producer=producer, known_support_domains={str(producer.get("support_domain_ref", ""))}))
    blockers.extend(
        validate_trajectory(
            trajectory,
            producer=producer,
            initial_state=initial_state,
            action_binding=action_binding,
            state_values=state_values,
            known_support_domains={str(producer.get("support_domain_ref", ""))},
        )
    )
    blockers.extend(validate_numerical_qc(numerical))
    blockers.extend(validate_constraint_qc(constraint))
    blockers.extend(validate_failure_ledger(failure_rows))
    if qualification.get("status") != "qualified_with_restrictions" or qualification.get("blockers") != []:
        blockers.append({"code": "producer_qualification_failed", "message": "Round-15 producer qualification did not pass with explicit restrictions."})
    if pilot_validation.get("status") != "passed" or pilot_validation.get("blocker_count") != 0:
        blockers.append({"code": "pilot_validation_failed", "message": "Round-15 qualification pilot did not pass."})
    if pilot.get("formal_e4_admission") is not False:
        blockers.append({"code": "pilot_admitted_to_formal_e4", "message": "Pilot trajectory cannot be admitted automatically."})
    if pilot.get("evidence_fabric_sealed") is not False or pilot.get("dt_core_status") != "not_evaluated":
        blockers.append({"code": "scope_boundary_violation", "message": "Pilot crossed Evidence Fabric or DT_CORE boundaries."})
    if int(production_plan.get("expected_run_count", 0)) != 1 or production_plan.get("model_training") is not False:
        blockers.append({"code": "production_plan_invalid", "message": "Round-15 production plan is not the bounded qualification-only plan."})
    schema_objects = {
        "trajectory_producer": producer,
        "trajectory_producer_qualification": qualification,
        "trajectory_initial_state": initial_state,
        "trajectory_action_binding": action_binding,
        "clean_action_complete_trajectory": trajectory,
        "trajectory_numerical_qc": numerical,
        "trajectory_constraint_qc": constraint,
        "trajectory_failure": dict(pilot.get("intentional_failure", {})),
    }
    blockers.extend(_schema_validate(repository_root=repository_root, config=config, objects=schema_objects))
    blockers = sorted_blockers(blockers)
    summary = {
        "producer_uid": producer.get("producer_uid"),
        "producer_version": producer.get("version"),
        "producer_qualification_status": qualification.get("status"),
        "support_domain_ref": producer.get("support_domain_ref"),
        "pilot_trajectory_uid": trajectory.get("trajectory_uid"),
        "pilot_trajectory_hash": trajectory.get("state_values_hash"),
        "capability_cell_count": len(capability),
        "supported_cell_count": sum(row.get("state") in {"supported", "supported_with_restrictions"} for row in capability),
        "failure_count": len(failure_rows),
        "frozen_gap_count": 20,
        "artifact_hashes": {name: sha256_file(paths[name]) for name in required},
    }
    return blockers, summary


def validate_formal_outputs(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    source_output_root: Path,
    parent_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    blockers, pilot_summary = validate_pilot_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_output_root,
    )
    paths = _output_paths(config, source_output_root)
    formal_summary_path = source_output_root / "Output/audits/dt_round15_formal_production_summary.json"
    required = (
        "numerical_qc_catalog",
        "e4_catalog",
        "e4_pairing_catalog",
        "admission_ledger",
        "frozen_gap_map",
        "independence_groups",
        "leakage_readiness",
        "coverage",
        "effective_units",
    )
    missing = [name for name in required if not paths[name].is_file()]
    if not formal_summary_path.is_file():
        missing.append("formal_production_summary")
    if missing:
        blockers.append({"code": "formal_artifact_missing", "message": "Round-15 formal E4 artifacts are missing.", "artifacts": missing})
        return sorted_blockers(blockers), {"pilot": pilot_summary, "artifact_hashes": {}, "missing": missing}
    formal_summary = _load_json(formal_summary_path) or {}
    trajectory_rows = _parquet_rows(paths["e4_catalog"])
    numerical_rows = _parquet_rows(paths["numerical_qc_catalog"])
    pairing_rows = _parquet_rows(paths["e4_pairing_catalog"])
    admission_rows = read_tsv(paths["admission_ledger"])
    gap_rows = read_tsv(paths["frozen_gap_map"])
    independence_rows = read_tsv(paths["independence_groups"])
    leakage = _load_json(paths["leakage_readiness"]) or {}
    coverage = _load_json(paths["coverage"]) or {}
    effective = _load_json(paths["effective_units"]) or {}
    if len(trajectory_rows) != 1 or len(numerical_rows) != 1 or len(pairing_rows) != 1 or len(admission_rows) != 1:
        blockers.append({"code": "formal_catalog_cardinality", "message": "Bounded Round-15 release requires one formal trajectory and one admission."})
        return sorted_blockers(blockers), {"pilot": pilot_summary, "artifact_hashes": {name: sha256_file(paths[name]) for name in required}}
    trajectory = trajectory_rows[0]
    producer_qualification = _load_json(paths["producer_qualification_json"]) or {}
    pilot = _load_json(paths["pilot_manifest"]) or {}
    producer = dict(pilot.get("producer", {}))
    trajectory_dir = source_output_root / str(Path(str(trajectory["state_values_ref"])).parent)
    initial_state = _load_json(trajectory_dir / "initial_state.json") or {}
    action_binding = _load_json(trajectory_dir / "action_binding.json") or {}
    numerical = _load_json(trajectory_dir / "numerical_qc.json") or {}
    constraint = _load_json(trajectory_dir / "constraint_qc.json") or {}
    family_manifest = _load_json(trajectory_dir / "trajectory_family_manifest.json") or {}
    state_path = source_output_root / str(trajectory["state_values_ref"])
    state_values = np.load(state_path, allow_pickle=False) if state_path.is_file() else None
    blockers.extend(
        validate_trajectory(
            trajectory,
            producer=producer,
            initial_state=initial_state,
            action_binding=action_binding,
            state_values=state_values,
            known_support_domains={str(producer.get("support_domain_ref", ""))},
        )
    )
    blockers.extend(validate_numerical_qc(numerical))
    blockers.extend(validate_constraint_qc(constraint))
    admission = {key: _json_cell(value) for key, value in admission_rows[0].items()}
    blockers.extend(validate_admission_decision(admission, numerical_qc=numerical, constraint_qc=constraint))
    if admission.get("decision") != "admitted_e4_restricted" or admission.get("eligibility") != "qualification_numerical_dynamics_only":
        blockers.append({"code": "trajectory_role_upgrade", "message": "Round-15 formal trajectory exceeded qualification-only eligibility."})
    pairing = pairing_rows[0]
    if pairing.get("pairing_decision") != "unpaired_trajectory" or pairing.get("action_effect_claim_eligible") is not False:
        blockers.append({"code": "control_pairing_invalid", "message": "No-action singleton must remain unpaired and ineligible for action-effect claims."})
    blockers.extend(validate_gap_trajectory_map(gap_rows, admitted_trajectory_uids={str(trajectory["trajectory_uid"])}, expected_gap_count=20))
    if any(row.get("current_status") == "resolved_by_admitted_e4" for row in gap_rows):
        blockers.append({"code": "gap_fabrication", "message": "Qualification-only E4 cannot close a frozen empirical gap."})
    blockers.extend(validate_independence_groups(independence_rows))
    if leakage.get("status") != "passed" or leakage.get("training_splits_created") is not False:
        blockers.append({"code": "independence_grouping_failed", "message": "E4 leakage readiness is not passing or created unauthorized splits."})
    if coverage.get("broad_cns_coverage_claim") is not False or coverage.get("biological_validation_claim") is not False:
        blockers.append({"code": "coverage_overclaim", "message": "Qualification trajectory cannot support broad biological claims."})
    if effective.get("status") != "passed" or int(effective.get("effective_independent_unit_count", 0)) != 1:
        blockers.append({"code": "effective_units_invalid", "message": "Effective trajectory units are incomplete."})
    if formal_summary.get("status") != "passed" or formal_summary.get("evidence_fabric_sealed") is not False or formal_summary.get("dt_core_status") != "not_evaluated":
        blockers.append({"code": "formal_production_summary_invalid", "message": "Formal production summary violates Round-15 boundaries."})
    release_path = source_output_root / str(formal_summary.get("release_path", ""))
    release = _load_json(release_path) or {}
    blockers.extend(validate_release_candidate(release))
    parent_manifest = parent_root / str(round15_config(config)["parent_artifacts"]["round14_release_manifest"])
    if release.get("parent_release_manifest_hash") != sha256_file(parent_manifest):
        blockers.append({"code": "parent_release_hash_mismatch", "message": "Round-15 release does not bind the unchanged Round-14 release manifest."})
    if release.get("trajectory_catalog_hash") != sha256_file(paths["e4_catalog"]):
        blockers.append({"code": "release_catalog_hash_mismatch", "message": "Round-15 release trajectory catalog hash is stale."})
    if release.get("content_hash") != release_content_hash(release):
        blockers.append({"code": "release_content_hash_mismatch", "message": "Round-15 release content hash is stale."})
    schema_objects = {
        "clean_action_complete_trajectory": trajectory,
        "trajectory_producer": producer,
        "trajectory_producer_qualification": producer_qualification,
        "trajectory_initial_state": initial_state,
        "trajectory_action_binding": action_binding,
        "trajectory_numerical_qc": numerical,
        "trajectory_constraint_qc": constraint,
        "trajectory_control_pair": pairing,
        "trajectory_admission_decision": admission,
        "trajectory_family_manifest": family_manifest,
        "round15_e4_release_candidate": release,
    }
    blockers.extend(_schema_validate(repository_root=repository_root, config=config, objects=schema_objects))
    blockers = sorted_blockers(blockers)
    artifact_paths = {name: paths[name] for name in required}
    artifact_paths["formal_production_summary"] = formal_summary_path
    artifact_paths["release_candidate"] = release_path
    summary = {
        "pilot": pilot_summary,
        "producer_uid": producer.get("producer_uid"),
        "producer_version": producer.get("version"),
        "producer_qualification_status": producer_qualification.get("status"),
        "trajectory_uid": trajectory.get("trajectory_uid"),
        "trajectory_family_uid": trajectory.get("trajectory_family_uid"),
        "trajectory_count": len(trajectory_rows),
        "admission_counts": dict(Counter(str(row.get("decision")) for row in admission_rows)),
        "release_uid": release.get("release_uid"),
        "release_path": str(release_path),
        "release_content_hash": release.get("content_hash"),
        "code_commit": release.get("code_commit"),
        "config_hash": release.get("config_hash"),
        "schema_registry_hash": release.get("schema_registry_hash"),
        "input_bundle_hash": release.get("input_bundle_hash"),
        "environment_hash": release.get("environment_hash"),
        "gap_count": len(gap_rows),
        "resolved_gap_count": sum(row.get("current_status") == "resolved_by_admitted_e4" for row in gap_rows),
        "effective_independent_units": int(effective.get("effective_independent_unit_count", 0)),
        "failure_count": len(read_tsv(paths["failure_ledger"])),
        "e3_promoted_in_place": False,
        "artifact_hashes": {name: sha256_file(path) for name, path in artifact_paths.items()},
    }
    return blockers, summary


def build_prefull_report(
    *,
    pilot_blockers: Sequence[Mapping[str, object]],
    pilot_summary: Mapping[str, object],
    focused: Mapping[str, object],
    identity: Mapping[str, object] | None,
    identity_blockers: Sequence[Mapping[str, object]],
    execution_dirty_status: str,
) -> dict[str, object]:
    focused_names = set(focused["testcase_names"])
    focused_ok = (
        focused["available"]
        and focused["failures"] == 0
        and focused["errors"] == 0
        and focused["skipped"] == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(focused_names)
    )
    blockers = [*pilot_blockers, *identity_blockers]
    if not focused_ok:
        blockers.append(
            {
                "code": "focused_negative_tests_not_verified",
                "message": "Mandatory Round-15 negative tests did not all pass.",
                "missing": sorted(MANDATORY_NEGATIVE_TEST_NAMES - focused_names),
            }
        )
    if execution_dirty_status != "clean":
        blockers.append({"code": "execution_worktree_dirty", "message": "Exact execution worktree became dirty."})
    blockers = sorted_blockers(blockers)
    return {
        "schema_version": PREFULL_SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "pilot": dict(pilot_summary),
        "focused_tests": dict(focused),
        "mandatory_negative_cases_verified": focused_ok,
        "execution_identity": dict(identity) if identity else None,
        "execution_worktree_status": execution_dirty_status,
        "producer_qualification_status": pilot_summary.get("producer_qualification_status"),
        "frozen_gap_count": 20,
        "formal_production_authorized": not blockers,
        "evidence_fabric": {"materialization": "materialized_unsealed", "sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "blockers": blockers,
        "blocker_count": len(blockers),
    }


def _inventory_blockers(
    path: Path | None,
    full: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    value = _load_json(path)
    if value is None:
        return [{"code": "inventory_missing", "message": "Round-15 test inventory reconciliation is missing."}], None
    blockers: list[dict[str, object]] = []
    if value.get("schema_version") != TEST_INVENTORY_SCHEMA_VERSION:
        blockers.append({"code": "inventory_schema_invalid", "message": "Round-15 inventory schema is invalid."})
    if value.get("round14_parent_test_count") != 468:
        blockers.append({"code": "inventory_parent_count_invalid", "message": "Round-14 parent test count is not the accepted 468."})
    if value.get("unexplained_missing_tests") != 0 or value.get("mandatory_disappearance") != 0:
        blockers.append({"code": "inventory_mandatory_disappearance", "message": "Mandatory parent tests disappeared."})
    if full.get("available") and value.get("round15_current_test_count") != full.get("tests"):
        blockers.append({"code": "inventory_count_mismatch", "message": "Round-15 inventory count differs from the full JUnit result."})
    return sorted_blockers(blockers), value


def _gate_status(blockers: Sequence[Mapping[str, object]]) -> str:
    if not blockers:
        return "clean_action_complete_trajectory_domain_resolved"
    codes = {str(row.get("code", "")) for row in blockers}
    mappings = (
        ("blocked_execution_verification", ("execution", "full_suite", "inventory", "prefull", "focused")),
        ("blocked_gap_fabrication", ("gap_", "frozen_gap")),
        ("blocked_independence_grouping", ("independence", "effective_units", "seed_role", "fidelity")),
        ("blocked_control_pairing", ("pair", "unpaired")),
        ("blocked_trajectory_provenance", ("hash", "provenance", "schema", "commit", "release")),
        ("blocked_support_domain", ("support_domain",)),
        ("blocked_physical_constraints", ("constraint", "monotonicity")),
        ("blocked_numerical_qc", ("numerical", "nan_inf", "solver", "replay", "state_bound")),
        ("blocked_initial_state", ("initial_state", "synthetic_state")),
        ("blocked_action_completeness", ("action", "dose", "route", "occupancy")),
        ("blocked_producer_qualification", ("producer", "qualification", "capability", "license")),
    )
    for state, markers in mappings:
        if any(any(marker in code for marker in markers) for code in codes):
            return state
    return "blocked_trajectory_provenance"


def build_round15_gate_record(
    *,
    authority: Mapping[str, object],
    assessment: str,
    blockers: list[dict[str, object]],
    input_hashes: Mapping[str, str],
    config_hash: str,
    commit: str,
    report: Mapping[str, object],
    materialization: Mapping[str, object],
) -> dict[str, object]:
    """Build an accepted canonical Gate or a non-authoritative blocked assessment."""

    common = {
        "round_uid": "round_15_clean_action_complete_trajectory_domain",
        "status": assessment,
        "overall_gate_status": "blocked" if blockers else "accepted",
        "validation_status": report["status"],
        "blocker_count": len(blockers),
        "blocker_classes": sorted(
            {_gate_status([blocker]).removeprefix("blocked_") for blocker in blockers}
        ),
        "frozen_gap_count": materialization.get("gap_count", 0),
        "resolved_gap_count": materialization.get("resolved_gap_count", 0),
        "execution_identity": report["execution_identity"],
        "evidence_fabric": report["evidence_fabric"],
        "dt_core": {"status": "not_evaluated"},
        "round_11_closure": "accepted_unchanged",
        "round_12_gate": "typed_evidence_relationships_resolved_unchanged",
        "round_13_gate": "action_observation_contracts_resolved_unchanged",
        "round_14_gate": "evidence_fabric_v0_materialized_unchanged",
    }
    if blockers:
        return {
            "schema_version": "neuronsifter-dt-round15-blocked-gate-assessment-v1",
            "gate_uid": "round_15_clean_action_complete_trajectory_domain",
            "stage_uid": "round_15_clean_action_complete_trajectory_domain",
            "writer_kind": "approved_validator_blocked_assessment",
            "validator_uid": PRODUCER_VALIDATOR_UID,
            "canonical_controller_uid": CANONICAL_CONTROLLER_UID,
            "input_artifact_hashes": dict(sorted(input_hashes.items())),
            "config_hash": config_hash,
            "git_commit": commit,
            "decision": "blocked",
            "blockers": blockers,
            "lineage_parent": "round_14_evidence_fabric_v0",
        } | common
    return build_canonical_gate_decision(
        authority,
        gate_uid="round_15_clean_action_complete_trajectory_domain",
        stage_uid="round_15_clean_action_complete_trajectory_domain",
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid=PRODUCER_VALIDATOR_UID,
        validator_report={
            "validator_uid": PRODUCER_VALIDATOR_UID,
            "status": "passed",
            "assessment": assessment,
        },
        input_artifact_hashes=input_hashes,
        config_hash=config_hash,
        commit=commit,
        decision="accepted",
        blockers=[],
        lineage_parent="round_14_evidence_fabric_v0",
    ) | common


def validate_round15_outputs(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    source_output_root: Path,
    parent_root: Path,
    focused_junit_path: Path | None,
    prefull_validation_path: Path | None,
    execution_identity_path: Path | None,
    full_suite_junit_path: Path | None,
    full_suite_execution_path: Path | None,
    test_inventory_path: Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    blockers, materialization = validate_formal_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_output_root,
        parent_root=parent_root,
    )
    identity_blockers, identity = _execution_identity_blockers(execution_identity_path, repository_root)
    blockers.extend(identity_blockers)
    if identity:
        for identity_field, materialization_field in (
            ("exact_git_commit", "code_commit"),
            ("config_hash", "config_hash"),
            ("schema_registry_hash", "schema_registry_hash"),
            ("input_bundle_hash", "input_bundle_hash"),
            ("environment_hash", "environment_hash"),
        ):
            if identity.get(identity_field) != materialization.get(materialization_field):
                blockers.append(
                    {
                        "code": "execution_materialization_identity_mismatch",
                        "message": "Formal E4 release differs from the exact execution identity.",
                        "field": identity_field,
                    }
                )
    focused = _junit_summary(focused_junit_path)
    focused_names = set(focused["testcase_names"])
    if not (
        focused["available"]
        and focused["failures"] == 0
        and focused["errors"] == 0
        and focused["skipped"] == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(focused_names)
    ):
        blockers.append({"code": "focused_negative_tests_not_verified", "message": "Mandatory Round-15 focused tests are not verified."})
    prefull = _load_json(prefull_validation_path)
    if not prefull or prefull.get("schema_version") != PREFULL_SCHEMA_VERSION or prefull.get("status") != "passed":
        blockers.append({"code": "prefull_not_verified", "message": "Exact-clean Round-15 pre-full Gate did not pass."})
    full = _junit_summary(full_suite_junit_path)
    if not (
        full["available"]
        and full["tests"] >= 468
        and full["failures"] == 0
        and full["errors"] == 0
        and full["skipped"] == 0
    ):
        blockers.append({"code": "full_suite_not_verified", "message": "Full Libs/tests suite is not verified."})
    inventory_blockers, inventory = _inventory_blockers(test_inventory_path, full)
    blockers.extend(inventory_blockers)
    full_execution = _load_json(full_suite_execution_path)
    if not full_execution or full_execution.get("schema_version") != FULL_SUITE_EXECUTION_SCHEMA_VERSION:
        blockers.append({"code": "execution_full_suite_missing", "message": "Full-suite CPU05 execution evidence is missing."})
    elif not (
        full_execution.get("terminal_state") == "COMPLETED"
        and full_execution.get("exit_code") == "0:0"
        and full_execution.get("node") == "cpu05"
        and identity
        and full_execution.get("exact_git_commit") == identity.get("exact_git_commit")
    ):
        blockers.append({"code": "execution_full_suite_failed", "message": "Full-suite Slurm terminal evidence did not pass on CPU05."})
    if git_dirty_status(repository_root) != "clean":
        blockers.append({"code": "execution_worktree_dirty", "message": "Exact execution worktree is dirty after validation."})
    blockers = sorted_blockers(blockers)
    assessment = _gate_status(blockers)
    report = {
        "schema_version": "neuronsifter-dt-round15-validation-v1",
        "validator_uid": PRODUCER_VALIDATOR_UID,
        "status": "passed" if not blockers else "blocked",
        "gate_assessment": assessment,
        "materialization": materialization,
        "focused_tests": focused,
        "prefull_validation": prefull,
        "execution_identity": identity or {
            "exact_git_commit": git_commit(repository_root),
            "dirty_status": git_dirty_status(repository_root),
            "reproduction_relation": "not_exact_clean_commit",
        },
        "full_suite": full,
        "full_suite_execution": full_execution,
        "test_inventory": inventory,
        "frozen_gap_count": materialization.get("gap_count", 0),
        "resolved_gap_count": materialization.get("resolved_gap_count", 0),
        "evidence_fabric": {
            "materialization": "materialized_unsealed",
            "e4_status": "clean_action_complete_trajectory_domain_resolved" if not blockers else "blocked",
            "sealed": False,
        },
        "dt_core": {"status": "not_evaluated"},
        "blockers": blockers,
        "blocker_count": len(blockers),
    }
    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    input_hashes = {
        name: value
        for name, value in dict(materialization.get("artifact_hashes", {})).items()
        if isinstance(value, str)
    }
    for name, path in (
        ("focused_junit", focused_junit_path),
        ("prefull_validation", prefull_validation_path),
        ("execution_identity", execution_identity_path),
        ("full_suite_junit", full_suite_junit_path),
        ("full_suite_execution", full_suite_execution_path),
        ("test_inventory", test_inventory_path),
    ):
        if path and path.is_file():
            input_hashes[name] = sha256_file(path)
    gate = build_round15_gate_record(
        authority=authority,
        assessment=assessment,
        blockers=blockers,
        input_hashes=input_hashes,
        config_hash=sha256_file(repository_root / "Libs/configs/data/round15_trajectory_domain.yaml"),
        commit=git_commit(repository_root),
        report=report,
        materialization=materialization,
    )
    return report, gate


def build_round16_handoff(
    report: Mapping[str, object],
    gate: Mapping[str, object],
    source_output_root: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    materialization = dict(report["materialization"])
    paths = _output_paths(config, source_output_root)
    gaps = read_tsv(paths["frozen_gap_map"])
    qualification = _load_json(paths["producer_qualification_json"]) or {}
    production_plan = _load_json(paths["production_plan"]) or {}
    formal_summary = _load_json(source_output_root / "Output/audits/dt_round15_formal_production_summary.json") or {}
    return {
        "schema_version": "neuronsifter-dt-round16-handoff-v1",
        "status": "ready_for_governed_what_if_engine_contracts",
        "parent_gate": gate["status"],
        "execution_commit": gate["git_commit"],
        "qualified_trajectory_producers": [
            {
                "producer_uid": materialization["producer_uid"],
                "producer_version": materialization["producer_version"],
                "qualification_status": qualification["status"],
                "restrictions": qualification["restrictions"],
            }
        ],
        "supported_trajectory_domains": [round15_config(config)["supported_domain"]],
        "e4_release_candidate": {
            "release_uid": materialization["release_uid"],
            "release_path": formal_summary["release_path"],
            "release_content_hash": materialization["release_content_hash"],
        },
        "action_control_pairing_rules": {
            "qualified_action_classes": ["no_action_control"],
            "action_effect_claim_requires_pair": True,
            "current_action_effect_claim_eligible_pairs": 0,
        },
        "trajectory_admission_rules": [
            "qualified_producer",
            "complete_initial_state",
            "complete_world_action",
            "exact_provenance",
            "numerical_qc_passed",
            "required_constraints_passed",
            "content_hash_complete",
            "no_blocking_conflict",
        ],
        "numerical_qc_thresholds": round15_config(config)["numerical_qc"],
        "physical_qc_thresholds": round15_config(config)["constraint_qc"],
        "independence_groups_hash": materialization["artifact_hashes"]["independence_groups"],
        "remaining_frozen_gaps": gaps,
        "allowed_world_action_training_roles": [],
        "qualification_only_roles": ["qualification_numerical_dynamics_only"],
        "what_if_eligibility_boundaries": {
            "contract_engine_testing": True,
            "publication_level_what_if": False,
            "biological_validation": False,
            "clinical_recommendation": False,
        },
        "minimum_uncertainty_reliability_diagnostics": [
            "producer_identity",
            "initial_state_identity",
            "action_program_hash",
            "parameter_provenance",
            "numerical_qc",
            "constraint_qc",
            "support_domain",
            "pairing_decision",
            "independence_groups",
        ],
        "production_plan_scope": production_plan["justification"],
        "evidence_fabric": {
            "materialization": "materialized_unsealed",
            "e4_status": "clean_action_complete_trajectory_domain_resolved",
            "sealed": False,
        },
        "dt_core": {"status": "not_evaluated"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Round-15 clean trajectory qualification and E4 admission.")
    parser.add_argument("--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--source-output-root", required=True)
    parser.add_argument("--parent-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--focused-junit")
    parser.add_argument("--execution-identity")
    parser.add_argument("--prefull-validation")
    parser.add_argument("--full-suite-junit")
    parser.add_argument("--full-suite-execution")
    parser.add_argument("--test-inventory")
    parser.add_argument("--prefull-only", action="store_true")
    args = parser.parse_args()
    repository_root = Path(args.repository_root).resolve()
    source_output_root = Path(args.source_output_root).resolve()
    parent_root = Path(args.parent_root).resolve() if args.parent_root else source_output_root
    output_root = Path(args.output_root).resolve()
    config = load_config(Path(args.config))
    identity_blockers, identity = _execution_identity_blockers(
        Path(args.execution_identity) if args.execution_identity else None,
        repository_root,
    )
    focused = _junit_summary(Path(args.focused_junit) if args.focused_junit else None)
    if args.prefull_only:
        pilot_blockers, pilot_summary = validate_pilot_outputs(
            config=config,
            repository_root=repository_root,
            source_output_root=source_output_root,
        )
        prefull = build_prefull_report(
            pilot_blockers=pilot_blockers,
            pilot_summary=pilot_summary,
            focused=focused,
            identity=identity,
            identity_blockers=identity_blockers,
            execution_dirty_status=git_dirty_status(repository_root),
        )
        write_json(output_root / "Output/audits/dt_round15_prefull_validation.json", prefull)
        print(f"round15_prefull_{prefull['status']} blockers={prefull['blocker_count']}")
        return 0 if prefull["status"] == "passed" else 1
    report, gate = validate_round15_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_output_root,
        parent_root=parent_root,
        focused_junit_path=Path(args.focused_junit) if args.focused_junit else None,
        prefull_validation_path=Path(args.prefull_validation) if args.prefull_validation else None,
        execution_identity_path=Path(args.execution_identity) if args.execution_identity else None,
        full_suite_junit_path=Path(args.full_suite_junit) if args.full_suite_junit else None,
        full_suite_execution_path=Path(args.full_suite_execution) if args.full_suite_execution else None,
        test_inventory_path=Path(args.test_inventory) if args.test_inventory else None,
    )
    write_json(output_root / VALIDATION_PATH, report)
    write_json(output_root / GATE_PATH, gate)
    if gate["status"] == "clean_action_complete_trajectory_domain_resolved":
        write_json(output_root / HANDOFF_PATH, build_round16_handoff(report, gate, source_output_root, config))
    print(f"{gate['status']} blockers={report['blocker_count']} gaps={report['frozen_gap_count']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
