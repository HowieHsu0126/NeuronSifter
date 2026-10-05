"""Fail-closed validator for Round-24 prospective adaptation evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
BLOCKED_STATUS = "blocked_adaptive_development"
C2B_STATUS = "blocked_insufficient_real_e2_e4"
EVIDENCE_ROLE = "development_only"
PRIMARY_BRANCHES = ("frozen_twin_frozen_agent", "adaptive_twin_frozen_agent")
CONDITIONAL_BRANCHES = ("frozen_twin_adaptive_agent", "adaptive_twin_adaptive_agent")
SCIENTIFIC_METRIC_STATUS = "not_evaluable_missing_model_bound_pre_post_evaluation"
EVIDENCE_BOUND_EXPORTS = {
    "prospective_tool_results": "Output/audits/round24_prospective_tool_results_evidence_bound.tsv",
    "adaptation_matrix": "Output/paper_reports/round24_development_adaptation_matrix_evidence_bound.tsv",
    "twin_adaptation": "Output/paper_reports/round24_dev_twin_adaptation_evidence_bound.tsv",
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


def _path(relative: str) -> Path:
    return REPO_ROOT / relative


def _load(relative: str) -> dict[str, Any]:
    path = _path(relative)
    if not path.is_file():
        raise ValueError(f"missing Round-24 audit: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Round-24 audit is not an object: {relative}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_identity(identity: Mapping[str, Any]) -> Path:
    path_value = identity.get("path")
    expected_hash = identity.get("sha256")
    if not isinstance(path_value, str) or not path_value or not isinstance(expected_hash, str):
        raise ValueError("Round-24 artifact identity is incomplete")
    path = _path(path_value)
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError(f"Round-24 artifact identity drift: {path_value}")
    if int(identity.get("bytes", path.stat().st_size)) != path.stat().st_size:
        raise ValueError(f"Round-24 artifact byte count drift: {path_value}")
    return path


def _read_tsv(relative: str) -> list[dict[str, str]]:
    path = _path(relative)
    if not path.is_file():
        raise ValueError(f"missing Round-24 table: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _parse_junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag.endswith("testsuite") else list(root)
    counts = {
        "tests_collected": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "tests_failed": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "tests_errored": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "tests_skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
    }
    counts["tests_passed"] = counts["tests_collected"] - counts["tests_failed"] - counts["tests_errored"] - counts["tests_skipped"]
    return counts


def _validate_contract() -> dict[str, Any]:
    config = load_config(_path("Libs/configs/config_stack.yaml"))
    contract = config.get("neuronsifter_vnext")
    if not isinstance(contract, dict):
        raise ValueError("config stack did not load the NeuronSifter vNext contract")
    schema = json.loads(
        _path("Libs/configs/schema/neuronsifter_vnext_research_contract.schema.json").read_text(encoding="utf-8")
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors({"neuronsifter_vnext": contract}),
        key=lambda error: list(error.path),
    )
    if errors:
        raise ValueError("Round-24 contract schema failure: " + "; ".join(error.message for error in errors))
    return contract


def _validate_primary_rounds() -> tuple[int, int]:
    run_root = _path("Output/runs/round24/twin_only")
    completed = 0
    unique_requests: set[str] = set()
    for branch in PRIMARY_BRANCHES:
        previous_uid: str | None = None
        selected_action_uids: set[str] = set()
        for round_index in range(1, 4):
            round_dir = run_root / branch / f"round_{round_index}"
            manifest = json.loads((round_dir / "round_input_manifest.json").read_text(encoding="utf-8"))
            selection = json.loads((round_dir / "selected_information_action.json").read_text(encoding="utf-8"))
            tool = json.loads((round_dir / "tool_result.json").read_text(encoding="utf-8"))
            update = json.loads((round_dir / "posterior_update.json").read_text(encoding="utf-8"))
            if (
                manifest.get("branch_id") != branch
                or int(manifest.get("round_index", -1)) != round_index
                or manifest.get("outcome_visible") is not False
                or manifest.get("evidence_role") != EVIDENCE_ROLE
                or manifest.get("publication_claim_eligible") is not False
                or int(manifest.get("protected_test_access", -1)) != 0
            ):
                raise ValueError(f"Round-24 input freeze failed: {branch}/{round_index}")
            if selection.get("selection_before_outcome") is not True or selection.get("world_action_uid") is not None:
                raise ValueError(f"Round-24 selection/world-action boundary failed: {branch}/{round_index}")
            action_uid = str(selection.get("action", {}).get("query", {}).get("action_uid", ""))
            if not action_uid or action_uid in selected_action_uids:
                raise ValueError(f"Round-24 query-without-replacement failed: {branch}/{round_index}")
            selected_action_uids.add(action_uid)
            request_uid = str(selection.get("request_uid", ""))
            if not request_uid or request_uid in unique_requests:
                raise ValueError(f"Round-24 request identity is missing or duplicated: {branch}/{round_index}")
            unique_requests.add(request_uid)
            if tool.get("posterior_uid") != selection.get("posterior_uid") or tool.get("status") not in {"completed", "partial"}:
                raise ValueError(f"Round-24 primary dispatch is incomplete: {branch}/{round_index}")
            parent_uid = str(update.get("parent_posterior_uid", ""))
            decision_uid = str(update.get("decision_posterior_uid", ""))
            if not parent_uid or not decision_uid or (previous_uid is not None and parent_uid != previous_uid):
                raise ValueError(f"Round-24 posterior lineage failed: {branch}/{round_index}")
            previous_uid = decision_uid
            if update.get("decision_consistency_preserved") is not True:
                raise ValueError(f"Round-24 posterior trace is inconsistent: {branch}/{round_index}")
            completed += 1
    return completed, len(unique_requests)


def _validate_full_suite() -> dict[str, Any]:
    full = _load("Output/audits/round24_full_suite_execution.json")
    if full.get("schema_version") != "neuronsifter-round24-full-suite-execution-v1":
        raise ValueError("Round-24 full-suite schema drift")
    junit_path = _path(str(full.get("junit_path", "")))
    if not junit_path.is_file() or _sha256(junit_path) != full.get("junit_sha256"):
        raise ValueError("Round-24 full-suite JUnit identity drift")
    counts = _parse_junit(junit_path)
    if any(int(full.get(key, -1)) != value for key, value in counts.items()):
        raise ValueError("Round-24 full-suite counts do not match the archived JUnit")
    source_rows = full.get("source_files")
    if not isinstance(source_rows, list) or int(full.get("source_file_count", -1)) != len(source_rows):
        raise ValueError("Round-24 full-suite source manifest is incomplete")
    for row in source_rows:
        _verify_identity(row)
    green = bool(
        full.get("full_suite_green") is True
        and full.get("status") == "passed"
        and int(full.get("pytest_exit_code", -1)) == 0
        and counts["tests_collected"] > 0
        and counts["tests_failed"] == 0
        and counts["tests_errored"] == 0
        and counts["tests_skipped"] == 0
        and full.get("node_local_staging_used") is True
        and int(full.get("protected_test_access_count", -1)) == 0
        and full.get("evidence_role") == EVIDENCE_ROLE
        and full.get("publication_claim_eligible") is False
    )
    if not green:
        raise ValueError("Round-24 full suite is not green and source-bound")
    return full


def _validate_evidence_bound_exports(program_gate: Mapping[str, Any]) -> None:
    artifact = program_gate.get("artifacts", {}).get("evidence_bound_export_catalog")
    if not isinstance(artifact, Mapping):
        raise ValueError("Round-24 final gate lacks the evidence-bound export catalog")
    catalog_path = _verify_identity(artifact)
    catalog = _load(str(artifact["path"]))
    if (
        catalog.get("schema_version")
        != "neuronsifter-round24-evidence-bound-export-catalog-v1"
        or catalog.get("status") != "current_successor_exports_bound"
        or catalog.get("evidence_role") != EVIDENCE_ROLE
        or catalog.get("publication_claim_eligible") is not False
        or int(catalog.get("protected_test_access", -1)) != 0
    ):
        raise ValueError("Round-24 evidence-bound export catalog is malformed")

    historical = catalog.get("historical_raw_exports")
    successors = catalog.get("successor_exports")
    if not isinstance(historical, Mapping) or not isinstance(successors, Mapping):
        raise ValueError("Round-24 evidence-bound export catalog is incomplete")
    if set(historical) != set(EVIDENCE_BOUND_EXPORTS) or set(successors) != set(EVIDENCE_BOUND_EXPORTS):
        raise ValueError("Round-24 evidence-bound export catalog has an unexpected export set")
    for name, expected_path in EVIDENCE_BOUND_EXPORTS.items():
        historical_identity = historical[name]
        successor_identity = successors[name]
        if not isinstance(historical_identity, Mapping) or not isinstance(successor_identity, Mapping):
            raise ValueError("Round-24 export identity is malformed")
        if (
            historical_identity.get("status")
            != "invalidated_raw_export_not_current_evidence_view"
            or historical_identity.get("reason")
            != "missing_row_level_evidence_boundary_or_invalidated_placeholder_metrics"
        ):
            raise ValueError("Round-24 historical raw export is not explicitly invalidated")
        _verify_identity(historical_identity)
        successor_path = _verify_identity(successor_identity)
        if successor_identity.get("path") != expected_path:
            raise ValueError(f"Round-24 successor export path drifted: {name}")
        rows = _read_tsv(successor_path.resolve().relative_to(REPO_ROOT).as_posix())
        if not rows:
            raise ValueError(f"Round-24 successor export is empty: {name}")
        required = {
            "scientific_metric_status",
            "historical_raw_metric_values_invalidated",
            "evidence_role",
            "publication_claim_eligible",
            "protected_test_access",
        }
        if any(not required.issubset(row) for row in rows):
            raise ValueError(f"Round-24 successor export lacks boundary columns: {name}")
        for row in rows:
            if (
                row["scientific_metric_status"] != SCIENTIFIC_METRIC_STATUS
                or row["historical_raw_metric_values_invalidated"] != "True"
                or row["evidence_role"] != EVIDENCE_ROLE
                or row["publication_claim_eligible"] != "False"
                or row["protected_test_access"] != "0"
            ):
                raise ValueError(f"Round-24 successor export crosses its evidence boundary: {name}")
        if name in {"prospective_tool_results", "twin_adaptation"}:
            if any(
                row.get("world_action_uid") or row.get("world_action_binding_status")
                != "absent_in_preserved_prospective_unit"
                for row in rows
            ):
                raise ValueError("Round-24 successor export backfilled an absent world-action binding")
        if name in {"adaptation_matrix", "twin_adaptation"}:
            if any(
                row.get("twin_parameter_change_scientific_evidence_admissible") != "False"
                for row in rows
            ):
                raise ValueError("Round-24 successor export overstates Twin parameter evidence")
            if any(
                row.get("branch_evidence_status")
                == "invalidated_premature_conditional_execution"
                and row.get("agent_parameters_changed") not in {"False", "True"}
                for row in rows
            ):
                raise ValueError("Round-24 conditional execution export is malformed")


def _validate_terminal_accounting(
    full: Mapping[str, Any], program_gate: Mapping[str, Any]
) -> None:
    job_id = str(full.get("slurm", {}).get("job_id", ""))
    expected_identity = full.get("slurm", {}).get("terminal_accounting")
    gate_identity = program_gate.get("artifacts", {}).get("slurm_terminal_accounting")
    if (
        full.get("slurm", {}).get("terminal_accounting_status") != "terminal_completed"
        or not job_id.isdigit()
        or not isinstance(expected_identity, Mapping)
        or not isinstance(gate_identity, Mapping)
    ):
        raise ValueError("Round-24 final validation lacks terminal accounting binding")
    accounting_path = _verify_identity(expected_identity)
    if dict(gate_identity) != _identity(accounting_path):
        raise ValueError("Round-24 final gate terminal-accounting identity drifted")
    accounting = _load(str(expected_identity["path"]))
    job = accounting.get("job", {})
    if (
        accounting.get("schema_version")
        != "neuronsifter-round24-slurm-terminal-accounting-v1"
        or accounting.get("status") != "terminal_completed"
        or str(job.get("job_id", "")) != job_id
        or job.get("state") != "COMPLETED"
        or job.get("exit_code") != "0:0"
        or job.get("job_name") != full.get("slurm", {}).get("job_name")
        or job.get("node_list") != full.get("slurm", {}).get("node")
        or accounting.get("evidence_role") != EVIDENCE_ROLE
        or accounting.get("publication_claim_eligible") is not False
        or int(accounting.get("protected_test_access", -1)) != 0
    ):
        raise ValueError("Round-24 terminal accounting artifact is not a matching green readback")


def validate(*, require_final: bool = False) -> dict[str, Any]:
    contract = _validate_contract()
    freeze = _load("Output/audits/round24_static_reference_freeze.json")
    mechanism = _load("Output/audits/round24_round23_mechanism_resolution.json")
    tools = _load("Output/audits/round24_tool_capability_freeze.json")
    twin_gate = _load("Output/audits/round24_twin_adaptation_gate.json")
    program_gate = _load("Output/audits/round24_vnext_development_gate.json")
    metrics = _load("Output/audits/round24_scientific_metric_provenance.json")
    matrix = _load("Output/audits/round24_adaptation_matrix_freeze.json")
    joint = _load("Output/audits/round24_joint_adaptation_resolution.json")
    boundary = _load("Output/audits/round24_evidence_boundary.json")
    c2b = _load("Output/audits/round24_c2b_blocked.json")
    bad = _load("Output/audits/round24_bad_update_demonstration.json")
    rollback = _load("Output/audits/round24_rollback_restoration.json")
    premature = _load("Output/audits/round24_premature_conditional_execution.json")

    if freeze.get("status") != "frozen" or freeze.get("mechanism_resolution") not in {"supported", "partial", "failed"}:
        raise ValueError("Round-24 static reference is not frozen with a known mechanism status")
    if mechanism.get("status") != "partial" or mechanism.get("evidence_role") != EVIDENCE_ROLE or mechanism.get("publication_claim_eligible") is not False:
        raise ValueError("Round-24 mechanism resolution is not the evidence-bound partial result")
    capabilities = tools.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities or any(item.get("available") is not True for item in capabilities):
        raise ValueError("Round-24 has no valid frozen registered tool capability")
    if int(tools.get("protected_test_access", -1)) != 0:
        raise ValueError("Round-24 tool capability opened protected access")

    primary_round_count, primary_request_count = _validate_primary_rounds()
    if primary_round_count != 6 or primary_request_count != 6:
        raise ValueError("Round-24 primary comparison is incomplete")

    metric_values_admissible = metrics.get("metrics", {}).get("metric_values_admissible")
    config_binding = contract["round24"]["evaluation_binding"]
    model_bound = bool(
        config_binding["c1_posterior_consumed_by_encoder"]
        and config_binding["c2s_posterior_consumed_by_world_action_predictor"]
        and config_binding["pre_post_metric_inputs_available"]
    )
    if metric_values_admissible is not False or model_bound:
        raise ValueError("Round-24 metric provenance does not match the blocked model binding")
    invalidated_fields = metrics.get("metrics", {}).get("historical_constant_fields_invalidated")
    if not isinstance(invalidated_fields, list) or not {"old_domain_retention", "calibration_drift", "aurc", "new_domain_gain"}.issubset(invalidated_fields):
        raise ValueError("Round-24 placeholder scientific metrics were not explicitly invalidated")
    for name, payload in metrics.get("metrics", {}).items():
        if name in {"metric_values_admissible", "historical_constant_fields_invalidated"}:
            continue
        if not isinstance(payload, Mapping) or payload.get("status") != "not_evaluable":
            raise ValueError(f"Round-24 unavailable metric is not fail-closed: {name}")

    if twin_gate.get("status") != BLOCKED_STATUS or program_gate.get("status") != BLOCKED_STATUS:
        raise ValueError("Round-24 cannot resolve adaptation without model-bound metrics")
    if twin_gate.get("reliability", {}).get("status") != "not_evaluable" or twin_gate.get("retention", {}).get("status") != "not_evaluable":
        raise ValueError("Round-24 Twin gate fabricated reliability or retention evidence")
    if twin_gate.get("twin", {}).get("model_bound_adaptation_evaluated") is not False:
        raise ValueError("Round-24 Twin gate overstates model-bound adaptation")
    if matrix.get("conditional_agent_adaptation_authorized") is not False or matrix.get("conditional_branch_scientific_evidence_admissible") is not False:
        raise ValueError("Round-24 conditional Agent adaptation was not closed")
    if joint.get("authorized_after_twin_gate") is not False or joint.get("joint_adaptation_supported") is not False or joint.get("empirical_coevolution_evidence") is not False:
        raise ValueError("Round-24 joint adaptation crossed the blocked Twin gate")
    if premature.get("status") != "invalidated_for_scientific_evidence" or premature.get("agent_or_joint_adaptation_evidence_admissible") is not False:
        raise ValueError("Round-24 premature conditional execution is not invalidated")

    failure_rows = _read_tsv("Output/audits/round24_tool_failure_catalog.tsv")
    preserved_failure = [row for row in failure_rows if row.get("attempt_id") == "slurm_512625" and row.get("status") == "failed"]
    if len(preserved_failure) != 1 or preserved_failure[0].get("successful_scientific_measurement") != "False":
        raise ValueError("Round-24 preserved tool failure is missing from the authoritative catalog")

    for artifact in (bad, rollback):
        expected = {
            "candidate_update_attempted": True,
            "promotion_rejected": True,
            "rollback_executed": True,
            "prior_checkpoint_restored": True,
            "runtime_capability_restored": True,
            "posterior_identity_restored": True,
            "biological_state_moved": False,
        }
        if any(artifact.get(field) is not value for field, value in expected.items()):
            raise ValueError("Round-24 rollback demonstration is incomplete")

    if (
        boundary.get("status") != "closed"
        or boundary.get("protected_final_holdout_accessed") is not False
        or int(boundary.get("protected_test_access_count", -1)) != 0
        or boundary.get("external_baseline_comparison_resolved") is not False
        or boundary.get("headline_comparison_authorized") is not False
        or boundary.get("round20qr_attempts_preserved") != [23, 24, 25, 26]
    ):
        raise ValueError("Round-24 evidence boundary is not closed")
    for attempt in (23, 24, 25, 26):
        if not _path(f"Output/audits/round20qr_v3/source_qualification_attempt_{attempt}").is_dir():
            raise ValueError(f"Round20QR v3 attempt {attempt} is not preserved")
    if c2b.get("status") != C2B_STATUS or program_gate.get("c2b", {}).get("status") != C2B_STATUS:
        raise ValueError("Round-24 C2-B status drifted")
    if any(payload.get("evidence_role", EVIDENCE_ROLE) != EVIDENCE_ROLE for payload in (twin_gate, program_gate, metrics, joint)):
        raise ValueError("Round-24 artifact lost its development-only label")
    if any(payload.get("publication_claim_eligible") is not False for payload in (twin_gate, program_gate, metrics, matrix, joint)):
        raise ValueError("Round-24 artifact became publication eligible")
    _validate_evidence_bound_exports(program_gate)

    full_suite: dict[str, Any] | None = None
    if _path("Output/audits/round24_full_suite_execution.json").is_file():
        full_suite = _validate_full_suite()
    if require_final:
        if full_suite is None:
            raise ValueError("Round-24 final validation requires a source-bound full suite")
        if program_gate.get("validation", {}).get("round24_validator_green") is not True or program_gate.get("validation", {}).get("full_suite_green") is not True:
            raise ValueError("Round-24 final gate does not bind the green validator/full suite")
        validation_identity = program_gate.get("artifacts", {}).get("full_suite_execution")
        if not isinstance(validation_identity, Mapping):
            raise ValueError("Round-24 final gate lacks full-suite identity")
        _verify_identity(validation_identity)
        _validate_terminal_accounting(full_suite, program_gate)

    return {
        "status": "passed",
        "program_status": program_gate["status"],
        "twin_gate_status": twin_gate["status"],
        "prospective_primary_rounds": 3,
        "prospective_primary_dispatches": primary_request_count,
        "scientific_adaptation_evaluated": False,
        "full_suite_green": bool(full_suite and full_suite["full_suite_green"]),
        "protected_holdout_access": 0,
        "evidence_role": EVIDENCE_ROLE,
        "publication_claim_eligible": False,
        "c2b": C2B_STATUS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-final", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(validate(require_final=arguments.require_final), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
