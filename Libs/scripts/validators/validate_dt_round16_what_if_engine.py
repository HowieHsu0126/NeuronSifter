"""Validate Round-16 governed what-if outputs and issue the canonical Gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_dt_round16_what_if_engine import (  # noqa: E402
    ROLLOUT_VALIDATOR_UID,
    sha256_file,
    validate_parent_state,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json  # noqa: E402
from Libs.utils.data.what_if_contracts import (  # noqa: E402
    SCHEMA_VERSIONS,
    release_content_hash,
    round16_config,
    sorted_blockers,
    validate_action_binding,
    validate_attempts,
    validate_baseline,
    validate_constraint_report,
    validate_contrast,
    validate_control_pair,
    validate_eligibility,
    validate_engine,
    validate_engine_compatibility,
    validate_gap_map,
    validate_lineage_acyclic,
    validate_release_candidate,
    validate_reliability,
    validate_result,
    validate_rollout_request,
    validate_rollout_trajectory,
    validate_round16_config,
    validate_uncertainty,
    validate_unique_uids,
    validate_what_if_request,
)
from Libs.utils.exp.orchestration_controller import (  # noqa: E402
    CANONICAL_CONTROLLER_UID,
    build_canonical_gate_decision,
    git_commit,
    git_dirty_status,
    load_authority_contract,
)


AUTHORITY_PATH = "Libs/configs/validation/round11f_controller_authority.yaml"
VALIDATION_PATH = "Output/audits/dt_round16_validation.json"
GATE_PATH = "Output/audits/dt_round16_gate.json"
HANDOFF_PATH = "Output/audits/dt_round17_handoff.json"
EXECUTION_IDENTITY_SCHEMA_VERSION = "neuronsifter-dt-round16-execution-identity-v1"
PREFULL_SCHEMA_VERSION = "neuronsifter-dt-round16-prefull-validation-v1"
FULL_SUITE_EXECUTION_SCHEMA_VERSION = "neuronsifter-dt-round16-full-suite-execution-v1"
TEST_INVENTORY_SCHEMA_VERSION = "neuronsifter-dt-round16-test-inventory-diff-v1"
ROUND16_CPU_FULL_SUITE_NODES = frozenset({"cpu03", "cpu04"})

MANDATORY_NEGATIVE_TEST_NAMES = frozenset(
    {
        "test_missing_baseline_rejected_round16",
        "test_incomplete_action_rejected_round16",
        "test_compound_id_used_as_complete_action_rejected_round16",
        "test_unregistered_rollout_engine_rejected",
        "test_untrained_model_declared_qualified_rejected",
        "test_unsupported_action_class_rejected_round16",
        "test_unsupported_state_namespace_rejected_round16",
        "test_outside_horizon_formal_use_rejected",
        "test_unpaired_rollout_effect_rejected_round16",
        "test_constraint_failed_rollout_contrast_rejected_round16",
        "test_unknown_uncertainty_zero_rejected",
        "test_model_rollout_as_empirical_observation_rejected",
        "test_model_result_closes_empirical_gap_rejected",
        "test_hypothetical_result_assimilation_observation_rejected",
        "test_support_domain_expansion_rejected_round16",
        "test_engine_disagreement_discarded_rejected",
        "test_failed_attempt_overwritten_rejected_round16",
        "test_causal_effect_language_rejected_round16",
        "test_clinical_recommendation_rejected_round16",
        "test_noncanonical_formal_authority_rejected_round16",
    }
)


def _load_json(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else None


def _junit_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {"available": False, "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "testcase_names": []}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    names = [case.get("name", "") for case in root.iter("testcase")]
    return {
        "available": True,
        "tests": sum(int(suite.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.get("skipped", 0)) for suite in suites),
        "testcase_names": sorted(names),
    }


def _schema_validate(
    *, repository_root: Path, schema_entry: Mapping[str, object], instance: Mapping[str, object], object_uid: str
) -> list[dict[str, object]]:
    path = repository_root / str(schema_entry["path"])
    schema = json.loads(path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    return [
        {
            "code": "round16_schema_validation_failed",
            "message": error.message,
            "object_uid": object_uid,
            "schema_path": str(path.relative_to(repository_root)),
            "instance_path": "/".join(str(value) for value in error.path),
        }
        for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.path))
    ]


def _paths(config: Mapping[str, object], root: Path) -> dict[str, Path]:
    return {name: root / str(path) for name, path in dict(round16_config(config)["output_paths"]).items()}


def _identity_blockers(
    identity_path: Path | None, repository_root: Path, release: Mapping[str, object] | None
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    identity = _load_json(identity_path)
    if identity is None:
        return ([{"code": "execution_identity_missing", "message": "Exact-clean Round-16 identity is missing."}], None)
    blockers: list[dict[str, object]] = []
    if identity.get("schema_version") != EXECUTION_IDENTITY_SCHEMA_VERSION:
        blockers.append({"code": "execution_identity_schema_invalid", "message": "Execution identity schema is invalid."})
    if identity.get("exact_git_commit") != git_commit(repository_root):
        blockers.append({"code": "execution_commit_mismatch", "message": "Execution commit differs from the code worktree."})
    if identity.get("dirty_status") != "clean" or identity.get("reproduction_relation") != "exact_clean_commit":
        blockers.append({"code": "execution_identity_not_exact_clean", "message": "Execution identity is not exact-clean."})
    if identity.get("code_worktree_read_only") is not True:
        blockers.append({"code": "execution_code_not_read_only", "message": "Execution code worktree was not read-only."})
    if identity.get("writable_runtime_output") is not True or identity.get("source_output_isolated") is not True:
        blockers.append({"code": "execution_output_not_isolated", "message": "Runtime output was not isolated and writable."})
    if release:
        for identity_field, release_field in (
            ("config_hash", "config_hash"),
            ("schema_registry_hash", "schema_registry_hash"),
            ("input_bundle_hash", "input_bundle_hash"),
            ("environment_hash", "environment_hash"),
        ):
            if identity.get(identity_field) != release.get(release_field):
                blockers.append({"code": "execution_materialization_identity_mismatch", "message": "Execution identity differs from materialization.", "field": identity_field})
    return sorted_blockers(blockers), identity


def validate_pilot_outputs(
    *, config: Mapping[str, object], repository_root: Path, source_output_root: Path, parent_root: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    blockers = validate_round16_config(config)
    try:
        parent = validate_parent_state(config, parent_root)
    except (FileNotFoundError, ValueError, TypeError) as error:
        return ([{"code": "round15_parent_invalid", "message": str(error)}], {})
    paths = _paths(config, source_output_root)
    pilot = _load_json(paths["pilot_validation"])
    graph = _load_json(paths["graph"])
    if pilot is None or graph is None:
        return ([{"code": "round16_pilot_missing", "message": "Round-16 pilot outputs are missing."}], {})
    release_path = source_output_root / str(pilot.get("release_path", ""))
    release = _load_json(release_path)
    if release is None:
        blockers.append({"code": "round16_release_missing", "message": "Round-16 successor release candidate is missing."})
    objects = dict(graph.get("objects", {}))
    request = dict(objects.get("what_if_request", {}))
    baseline = dict(objects.get("baseline", {}))
    action_binding = dict(objects.get("action_binding", {}))
    engine = dict(objects.get("engine", {}))
    rollout_request = dict(objects.get("rollout_request", {}))
    attempts = [dict(value) for value in objects.get("attempts", [])]
    rollouts = [dict(value) for value in objects.get("rollouts", [])]
    control_pair = dict(objects.get("control_pair", {}))
    constraint_report = dict(objects.get("constraint_report", {}))
    uncertainty = dict(objects.get("uncertainty", {}))
    reliability = dict(objects.get("reliability", {}))
    contrast = dict(objects.get("contrast", {}))
    eligibility = dict(objects.get("eligibility", {}))
    result = dict(objects.get("result", {}))
    known_domains = {str(engine.get("support_domain_ref", ""))}
    blockers += validate_what_if_request(request)
    blockers += validate_baseline(baseline, known_support_domains=known_domains)
    blockers += validate_action_binding(action_binding, known_support_domains=known_domains)
    blockers += validate_engine(engine, known_support_domains=known_domains)
    _, _, compatibility_blockers = validate_engine_compatibility(
        request=request, baseline=baseline, action_binding=action_binding, engine=engine
    )
    blockers += compatibility_blockers
    blockers += validate_rollout_request(rollout_request)
    blockers += validate_attempts(attempts)
    for rollout in rollouts:
        blockers += validate_rollout_trajectory(rollout)
    blockers += validate_control_pair(control_pair)
    blockers += validate_constraint_report(constraint_report)
    blockers += validate_uncertainty(uncertainty)
    blockers += validate_reliability(reliability)
    blockers += validate_contrast(contrast, constraint_state=str(constraint_report.get("handling_state", "invalid")))
    blockers += validate_eligibility(eligibility, canonical_controller_uid=CANONICAL_CONTROLLER_UID)
    blockers += validate_result(result)
    uid_values = [
        request.get("request_uid"), baseline.get("baseline_uid"), action_binding.get("binding_uid"), engine.get("engine_uid"),
        rollout_request.get("rollout_request_uid"), *(row.get("attempt_uid") for row in attempts),
        *(row.get("rollout_uid") for row in rollouts), control_pair.get("pair_uid"),
        constraint_report.get("constraint_report_uid"), uncertainty.get("uncertainty_uid"),
        reliability.get("reliability_uid"), contrast.get("contrast_uid"), eligibility.get("decision_uid"), result.get("result_uid"),
    ]
    blockers += validate_unique_uids([{"uid": value} for value in uid_values], ("uid",))
    gap_rows = read_tsv(paths["frozen_gap_map"]) if paths["frozen_gap_map"].is_file() else []
    blockers += validate_gap_map(gap_rows)
    edges = [
        (str(row.get("parent_uid", "")), str(row.get("child_uid", "")))
        for row in graph.get("lineage_edges", [])
    ]
    blockers += validate_lineage_acyclic(edges)
    registry = dict(config["schemas"]["round16_what_if_schema_registry"])
    schema_objects = {
        "what_if_request": [request], "what_if_baseline": [baseline], "what_if_action_binding": [action_binding],
        "rollout_engine": [engine], "rollout_request": [rollout_request], "rollout_attempt": attempts,
        "rollout_trajectory": rollouts, "rollout_control_pair": [control_pair], "counterfactual_contrast": [contrast],
        "rollout_uncertainty": [uncertainty], "rollout_constraint_report": [constraint_report],
        "rollout_reliability_report": [reliability], "what_if_eligibility_decision": [eligibility],
        "what_if_result": [result], "round16_what_if_release_candidate": [release or {}],
    }
    for name, values in schema_objects.items():
        for index, value in enumerate(values):
            blockers += _schema_validate(
                repository_root=repository_root, schema_entry=registry[name], instance=value,
                object_uid=str(value.get("result_uid") or value.get("rollout_uid") or value.get("request_uid") or f"{name}-{index}"),
            )
    if release:
        blockers += validate_release_candidate(release)
        artifact_paths = {
            name: path
            for name, path in paths.items()
            if name in {
                "request_catalog", "attempt_catalog", "rollout_catalog", "control_pair_catalog", "contrast_catalog",
                "uncertainty_catalog", "reliability_catalog", "eligibility_catalog", "graph", "frozen_gap_map",
            }
        }
        for name, expected_hash in dict(release.get("artifact_hashes", {})).items():
            path = artifact_paths.get(name)
            if path is None or not path.is_file() or sha256_file(path) != expected_hash:
                blockers.append({"code": "release_artifact_hash_mismatch", "message": "Release artifact hash changed.", "artifact": name})
        if release.get("release_content_hash") != release_content_hash(release):
            blockers.append({"code": "release_content_hash_mismatch", "message": "Release content hash changed."})
    for name in (
        "request_catalog", "attempt_catalog", "rollout_catalog", "control_pair_catalog", "contrast_catalog",
        "uncertainty_catalog", "reliability_catalog", "eligibility_catalog",
    ):
        if not paths[name].is_file() or not read_tsv(paths[name]):
            blockers.append({"code": "pilot_catalog_missing", "message": "Required pilot catalog is missing.", "catalog": name})
    required_negative_cases = {
        "unsupported_action_rejection", "outside_horizon_rejection", "out_of_domain_rejection",
        "constraint_failure_quarantine", "unknown_uncertainty_preservation", "hypothetical_to_observation_rejection",
        "unpaired_descriptive_only", "untrained_learned_engine", "clinical_recommendation",
    }
    negative_cases = {str(row.get("case")): str(row.get("result")) for row in objects.get("negative_cases", [])}
    if not required_negative_cases.issubset(negative_cases) or any(negative_cases[name] not in {"rejected", "quarantined", "unknown_preserved", "descriptive_only"} for name in required_negative_cases):
        blockers.append({"code": "pilot_negative_cases_incomplete", "message": "Mandatory pilot rejection paths are incomplete."})
    if graph.get("frozen_gap_resolved_by_model_rollout") != 0 or pilot.get("frozen_gap_resolved_by_model_rollout") != 0:
        blockers.append({"code": "model_result_closes_empirical_frozen_gap", "message": "Model outputs cannot resolve frozen gaps."})
    domain_comparison = dict(graph.get("support_domain_comparison", {}))
    required_domain_axes = {"organism", "disease_or_context", "region", "compartment", "cell_type", "state_variables", "action_class", "dose_or_regimen", "time_horizon", "fidelity"}
    if not required_domain_axes.issubset(domain_comparison) or any(dict(domain_comparison[axis]).get("difference") is not False for axis in required_domain_axes):
        blockers.append({"code": "support_domain_comparison_incomplete", "message": "Every support-domain axis must be compared explicitly."})
    if len(rollouts) != 2 or pilot.get("matched_control_pair_count") != 1:
        blockers.append({"code": "restricted_pilot_scope_invalid", "message": "Pilot must contain two matched replay roles and one pair."})
    if pilot.get("contrast_is_causal") is not False or pilot.get("empirical_observation_count") != 0:
        blockers.append({"code": "claim_boundary_invalid", "message": "Pilot contrast must be non-causal and non-empirical."})
    if pilot.get("evidence_fabric_sealed") is not False or pilot.get("dt_core_status") != "not_evaluated":
        blockers.append({"code": "round16_scope_violation", "message": "Evidence Fabric or DT_CORE boundary changed."})
    summary = {
        "pilot_status": pilot.get("status"), "request_uid": request.get("request_uid"), "engine_uid": engine.get("engine_uid"),
        "engine_status": engine.get("status"), "rollout_count": len(rollouts), "matched_control_pair_count": 1,
        "gap_count": len(gap_rows), "resolved_by_model_rollout": 0, "release_uid": release.get("release_uid") if release else None,
        "release_path": str(release_path), "release": release, "artifact_hashes": release.get("artifact_hashes", {}) if release else {},
        "parent_input_hashes": parent["hashes"], "graph_code_commit": graph.get("code_commit"),
        "evidence_fabric": {"materialization": "materialized_unsealed", "sealed": False}, "dt_core": {"status": "not_evaluated"},
    }
    return sorted_blockers(blockers), summary


def build_prefull_report(
    *, pilot_blockers: Sequence[Mapping[str, object]], pilot_summary: Mapping[str, object], focused: Mapping[str, object],
    identity: Mapping[str, object] | None, identity_blockers: Sequence[Mapping[str, object]], execution_dirty_status: str
) -> dict[str, object]:
    blockers = [dict(row) for row in pilot_blockers] + [dict(row) for row in identity_blockers]
    names = set(focused.get("testcase_names", []))
    if not (
        focused.get("available") and focused.get("failures") == 0 and focused.get("errors") == 0
        and focused.get("skipped") == 0 and MANDATORY_NEGATIVE_TEST_NAMES.issubset(names)
    ):
        blockers.append({"code": "focused_negative_tests_not_verified", "message": "Mandatory Round-16 focused tests are not verified."})
    if execution_dirty_status != "clean":
        blockers.append({"code": "execution_worktree_dirty", "message": "Pre-full code worktree is dirty."})
    blockers = sorted_blockers(blockers)
    return {
        "schema_version": PREFULL_SCHEMA_VERSION, "validator_uid": ROLLOUT_VALIDATOR_UID,
        "status": "passed" if not blockers else "blocked", "blocker_count": len(blockers), "blockers": blockers,
        "focused_tests": dict(focused), "execution_identity": dict(identity or {}), "pilot_summary": dict(pilot_summary),
        "full_suite_authorized": not blockers, "evidence_fabric": {"sealed": False}, "dt_core": {"status": "not_evaluated"},
    }


def _inventory_blockers(path: Path | None, full: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    value = _load_json(path)
    if value is None:
        return ([{"code": "inventory_missing", "message": "Round-16 test inventory reconciliation is missing."}], None)
    blockers: list[dict[str, object]] = []
    if value.get("schema_version") != TEST_INVENTORY_SCHEMA_VERSION:
        blockers.append({"code": "inventory_schema_invalid", "message": "Round-16 inventory schema is invalid."})
    if value.get("round15_parent_test_count") != 492:
        blockers.append({"code": "inventory_parent_count_invalid", "message": "Round-15 parent count must remain 492."})
    if value.get("unexplained_missing_tests") != 0 or value.get("mandatory_disappearance") != 0:
        blockers.append({"code": "inventory_mandatory_disappearance", "message": "Mandatory parent tests disappeared."})
    if full.get("available") and value.get("round16_current_test_count") != full.get("tests"):
        blockers.append({"code": "inventory_count_mismatch", "message": "Inventory and full JUnit counts differ."})
    return sorted_blockers(blockers), value


def _full_suite_execution_blockers(
    full_execution: Mapping[str, object] | None,
    identity: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    if full_execution is None or full_execution.get("schema_version") != FULL_SUITE_EXECUTION_SCHEMA_VERSION:
        return [{"code": "execution_full_suite_missing", "message": "Round-16 CPU full-suite evidence is missing."}]
    if not (
        full_execution.get("terminal_state") == "COMPLETED"
        and full_execution.get("exit_code") == "0:0"
        and full_execution.get("node") in ROUND16_CPU_FULL_SUITE_NODES
        and identity
        and full_execution.get("exact_git_commit") == identity.get("exact_git_commit")
        and int(full_execution.get("mandatory_skipped", -1)) == 0
        and int(full_execution.get("core_xfail", -1)) == 0
    ):
        return [{"code": "execution_full_suite_failed", "message": "CPU full-suite terminal evidence did not pass."}]
    return []


def _gate_status(blockers: Sequence[Mapping[str, object]]) -> str:
    if not blockers:
        return "governed_what_if_engine_resolved"
    codes = {str(row.get("code", "")) for row in blockers}
    mappings = (
        ("blocked_execution_verification", ("execution", "full_suite", "inventory", "prefull", "focused")),
        ("blocked_gap_fabrication", ("gap", "frozen")),
        ("blocked_claim_boundary", ("claim", "causal", "clinical", "empirical")),
        ("blocked_support_domain", ("support_domain", "domain")),
        ("blocked_reliability", ("reliability", "disagreement")),
        ("blocked_uncertainty_reporting", ("uncertainty",)),
        ("blocked_constraint_validation", ("constraint",)),
        ("blocked_horizon_qualification", ("horizon",)),
        ("blocked_control_pairing", ("pair", "unpaired")),
        ("blocked_engine_qualification", ("engine", "untrained")),
        ("blocked_action_completeness", ("action", "compound", "dose", "route")),
        ("blocked_baseline_identity", ("baseline",)),
    )
    for state, markers in mappings:
        if any(any(marker in code for marker in markers) for code in codes):
            return state
    return "blocked_execution_verification"


def build_round16_gate_record(
    *, authority: Mapping[str, object], assessment: str, blockers: list[dict[str, object]], input_hashes: Mapping[str, str],
    config_hash: str, commit: str, report: Mapping[str, object]
) -> dict[str, object]:
    common = {
        "round_uid": "round_16_governed_what_if_engine", "status": assessment,
        "overall_gate_status": "blocked" if blockers else "accepted", "validation_status": report["status"],
        "blocker_count": len(blockers),
        "blocker_classes": sorted({_gate_status([row]).removeprefix("blocked_") for row in blockers}),
        "frozen_gap_count": report.get("frozen_gap_count", 0), "frozen_gap_resolved_by_model_rollout": 0,
        "execution_identity": report["execution_identity"], "evidence_fabric": report["evidence_fabric"],
        "dt_core": {"status": "not_evaluated"}, "round_11_closure": "accepted_unchanged",
        "round_12_gate": "typed_evidence_relationships_resolved_unchanged",
        "round_13_gate": "action_observation_contracts_resolved_unchanged",
        "round_14_gate": "evidence_fabric_v0_materialized_unchanged",
        "round_15_gate": "clean_action_complete_trajectory_domain_resolved_unchanged",
        "claim_boundary": {"causal_effect_claim": False, "clinical_recommendation": False, "empirical_observation": False},
    }
    if blockers:
        return {
            "schema_version": "neuronsifter-dt-round16-blocked-gate-assessment-v1",
            "gate_uid": "round_16_governed_what_if_engine", "stage_uid": "round_16_governed_what_if_engine",
            "writer_kind": "approved_validator_blocked_assessment", "validator_uid": ROLLOUT_VALIDATOR_UID,
            "canonical_controller_uid": CANONICAL_CONTROLLER_UID, "input_artifact_hashes": dict(sorted(input_hashes.items())),
            "config_hash": config_hash, "git_commit": commit, "decision": "blocked", "blockers": blockers,
            "lineage_parent": "round_15_clean_action_complete_trajectory_domain",
        } | common
    return build_canonical_gate_decision(
        authority, gate_uid="round_16_governed_what_if_engine", stage_uid="round_16_governed_what_if_engine",
        controller_uid=CANONICAL_CONTROLLER_UID, validator_uid=ROLLOUT_VALIDATOR_UID,
        validator_report={"validator_uid": ROLLOUT_VALIDATOR_UID, "status": "passed", "assessment": assessment},
        input_artifact_hashes=input_hashes, config_hash=config_hash, commit=commit, decision="accepted", blockers=[],
        lineage_parent="round_15_clean_action_complete_trajectory_domain",
    ) | common


def validate_round16_outputs(
    *, config: Mapping[str, object], repository_root: Path, source_output_root: Path, parent_root: Path,
    focused_junit_path: Path | None, prefull_validation_path: Path | None, execution_identity_path: Path | None,
    full_suite_junit_path: Path | None, full_suite_execution_path: Path | None, test_inventory_path: Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    blockers, summary = validate_pilot_outputs(
        config=config, repository_root=repository_root, source_output_root=source_output_root, parent_root=parent_root
    )
    release = summary.get("release") if isinstance(summary.get("release"), Mapping) else None
    identity_blockers, identity = _identity_blockers(execution_identity_path, repository_root, release)
    blockers += identity_blockers
    focused = _junit_summary(focused_junit_path)
    if not (
        focused["available"] and focused["failures"] == 0 and focused["errors"] == 0 and focused["skipped"] == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(set(focused["testcase_names"]))
    ):
        blockers.append({"code": "focused_negative_tests_not_verified", "message": "Mandatory Round-16 focused tests are not verified."})
    prefull = _load_json(prefull_validation_path)
    if prefull is None or prefull.get("schema_version") != PREFULL_SCHEMA_VERSION or prefull.get("status") != "passed":
        blockers.append({"code": "prefull_not_verified", "message": "Exact-clean Round-16 pre-full Gate did not pass."})
    full = _junit_summary(full_suite_junit_path)
    if not (full["available"] and full["tests"] >= 492 and full["failures"] == 0 and full["errors"] == 0 and full["skipped"] == 0):
        blockers.append({"code": "full_suite_not_verified", "message": "Full Libs/tests suite is not verified."})
    inventory_blockers, inventory = _inventory_blockers(test_inventory_path, full)
    blockers += inventory_blockers
    full_execution = _load_json(full_suite_execution_path)
    blockers += _full_suite_execution_blockers(full_execution, identity)
    if git_dirty_status(repository_root) != "clean":
        blockers.append({"code": "execution_worktree_dirty", "message": "Exact execution worktree is dirty."})
    blockers = sorted_blockers(blockers)
    assessment = _gate_status(blockers)
    report = {
        "schema_version": "neuronsifter-dt-round16-validation-v1", "validator_uid": ROLLOUT_VALIDATOR_UID,
        "status": "passed" if not blockers else "blocked", "gate_assessment": assessment, "pilot_summary": summary,
        "focused_tests": focused, "prefull_validation": prefull,
        "execution_identity": identity or {"exact_git_commit": git_commit(repository_root), "dirty_status": git_dirty_status(repository_root), "reproduction_relation": "not_exact_clean_commit"},
        "full_suite": full, "full_suite_execution": full_execution, "test_inventory": inventory,
        "frozen_gap_count": summary.get("gap_count", 0), "frozen_gap_resolved_by_model_rollout": 0,
        "evidence_fabric": {"materialization": "materialized_unsealed", "sealed": False},
        "dt_core": {"status": "not_evaluated"}, "blockers": blockers, "blocker_count": len(blockers),
    }
    input_hashes = {name: str(value) for name, value in dict(summary.get("artifact_hashes", {})).items()}
    for name, path in (
        ("focused_junit", focused_junit_path), ("prefull_validation", prefull_validation_path),
        ("execution_identity", execution_identity_path), ("full_suite_junit", full_suite_junit_path),
        ("full_suite_execution", full_suite_execution_path), ("test_inventory", test_inventory_path),
    ):
        if path and path.is_file():
            input_hashes[name] = sha256_file(path)
    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    gate = build_round16_gate_record(
        authority=authority, assessment=assessment, blockers=blockers, input_hashes=input_hashes,
        config_hash=sha256_file(repository_root / "Libs/configs/data/round16_what_if_engine.yaml"),
        commit=git_commit(repository_root), report=report,
    )
    return report, gate


def build_round17_handoff(report: Mapping[str, object], gate: Mapping[str, object]) -> dict[str, object]:
    summary = dict(report["pilot_summary"])
    release = dict(summary["release"])
    return {
        "schema_version": "neuronsifter-dt-round17-handoff-v1", "status": "ready_for_scitil_twin_gated_agent_contracts",
        "parent_gate": gate["status"], "execution_commit": gate["git_commit"],
        "what_if_request_schema": SCHEMA_VERSIONS["what_if_request"],
        "qualified_rollout_engines": [{"engine_uid": summary["engine_uid"], "status": summary["engine_status"], "allowed_role": "restricted_contract_and_runtime_validation_only"}],
        "baseline_action_requirements": ["complete_governed_baseline", "complete_round13_world_action", "no_silent_imputation", "exact_parameter_provenance"],
        "control_pair_rules": ["matched_control_required_for_contrast", "unpaired_is_descriptive_only", "shared_engine_parameter_fidelity_grid_seed_domain_identity"],
        "uncertainty_components": list(round16_config(load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"))["pilot"].get("uncertainty_requirements", [])) or [
            "initial_state_uncertainty", "action_parameter_uncertainty", "model_parameter_uncertainty", "stochastic_rollout_uncertainty",
            "numerical_uncertainty", "model_family_disagreement", "fidelity_disagreement", "support_domain_extrapolation", "observation_or_calibration_uncertainty",
        ],
        "constraint_rules": ["required_constraints_fail_closed", "failed_or_unknown_required_constraint_blocks_formal_contrast"],
        "horizon_qualification": {"short_to_long_extrapolation_forbidden": True, "current_maximum_hours": 672.0, "current_role": "contract_testing_only"},
        "reliability_thresholds": ["repeatability", "seed_stability", "parameter_sensitivity", "fidelity_consistency", "horizon_degradation", "constraint_violation_rate", "model_family_disagreement", "baseline_perturbation_sensitivity", "action_perturbation_sensitivity", "support_domain_distance"],
        "support_domain_policy": "exact_or_explicitly_restricted_no_silent_expansion",
        "claim_language_restrictions": {"model_based_counterfactual": True, "causal_effect_claim": False, "clinical_recommendation": False, "empirical_observation": False},
        "what_if_eligibility_roles": ["contract_validation", "runtime_validation", "restricted_method_result", "stress_analysis", "hypothesis_generation"],
        "remaining_gaps": {"count": report["frozen_gap_count"], "resolved_by_model_rollout": 0},
        "allowed_agent_proposal_interface": ["propose_what_if_request", "propose_information_action", "request_controller_validation"],
        "forbidden_agent_state_update_paths": ["direct_twin_state_write", "direct_assimilation_approval", "model_rollout_as_observation", "llm_output_as_state", "formal_result_authorization"],
        "twin_gate_requirements": ["canonical_controller_authority", "approved_validator", "complete_provenance", "support_domain_match", "uncertainty_preserved", "conflicts_preserved", "no_direct_state_update"],
        "round16_release": {"release_uid": release["release_uid"], "release_content_hash": release["release_content_hash"], "sealed": False},
        "evidence_fabric": {"sealed": False}, "dt_core": {"status": "not_evaluated"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Round-16 governed what-if execution.")
    parser.add_argument("--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--source-output-root", required=True)
    parser.add_argument("--parent-root", required=True)
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
    parent_root = Path(args.parent_root).resolve()
    output_root = Path(args.output_root).resolve()
    config = load_config(Path(args.config))
    pilot_blockers, pilot_summary = validate_pilot_outputs(
        config=config, repository_root=repository_root, source_output_root=source_output_root, parent_root=parent_root
    )
    release = pilot_summary.get("release") if isinstance(pilot_summary.get("release"), Mapping) else None
    identity_blockers, identity = _identity_blockers(
        Path(args.execution_identity) if args.execution_identity else None, repository_root, release
    )
    focused = _junit_summary(Path(args.focused_junit) if args.focused_junit else None)
    if args.prefull_only:
        prefull = build_prefull_report(
            pilot_blockers=pilot_blockers, pilot_summary=pilot_summary, focused=focused, identity=identity,
            identity_blockers=identity_blockers, execution_dirty_status=git_dirty_status(repository_root),
        )
        write_json(output_root / "Output/audits/dt_round16_prefull_validation.json", prefull)
        print(f"round16_prefull_{prefull['status']} blockers={prefull['blocker_count']}")
        return 0 if prefull["status"] == "passed" else 1
    report, gate = validate_round16_outputs(
        config=config, repository_root=repository_root, source_output_root=source_output_root, parent_root=parent_root,
        focused_junit_path=Path(args.focused_junit) if args.focused_junit else None,
        prefull_validation_path=Path(args.prefull_validation) if args.prefull_validation else None,
        execution_identity_path=Path(args.execution_identity) if args.execution_identity else None,
        full_suite_junit_path=Path(args.full_suite_junit) if args.full_suite_junit else None,
        full_suite_execution_path=Path(args.full_suite_execution) if args.full_suite_execution else None,
        test_inventory_path=Path(args.test_inventory) if args.test_inventory else None,
    )
    write_json(output_root / VALIDATION_PATH, report)
    write_json(output_root / GATE_PATH, gate)
    if gate["status"] == "governed_what_if_engine_resolved":
        write_json(output_root / HANDOFF_PATH, build_round17_handoff(report, gate))
    print(f"{gate['status']} blockers={report['blocker_count']} gaps={report['frozen_gap_count']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
