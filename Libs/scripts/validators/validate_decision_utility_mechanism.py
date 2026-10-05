"""Fail-closed validation for the decision-utility mechanism diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import sha256_file

BOUNDARY = {
    "evidence_role": "development_only",
    "publication_claim_eligible": False,
    "protected_holdout_access": 0,
}
UNIT_TABLES = (
    REPO_ROOT / "Output/audits/round26_diagnostic_unit_manifest.tsv",
    REPO_ROOT / "Output/paper_reports/round26_s_u_policy_equivalence.tsv",
    REPO_ROOT / "Output/paper_reports/round26_decision_sensitivity.tsv",
    REPO_ROOT / "Output/paper_reports/round26_tri_flattening.tsv",
    REPO_ROOT / "Output/paper_reports/round26_information_action_value.tsv",
)
JSON_OUTPUTS = (
    REPO_ROOT / "Output/audits/round26_round25_negative_result_freeze.json",
    REPO_ROOT / "Output/audits/round26_utility_decomposition.json",
    REPO_ROOT / "Output/audits/round26_decision_transmission_analysis.json",
    REPO_ROOT / "Output/audits/round26_decision_sensitive_tool_coverage.json",
    REPO_ROOT / "Output/audits/round26_mechanism_root_cause.json",
    REPO_ROOT / "Output/audits/round26_information_action_gate.json",
    REPO_ROOT / "Output/audits/round26_canonical_architecture.json",
    REPO_ROOT / "Output/audits/round26_next_scientific_route.json",
)
FULL_SUITE_PATH = REPO_ROOT / "Output/audits/round26_full_suite_execution.json"
FULL_SUITE_ACCOUNTING_PATH = REPO_ROOT / "Output/audits/round26_full_suite_slurm_accounting.json"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _validate_full_suite(blockers: list[str]) -> None:
    if not FULL_SUITE_PATH.is_file():
        blockers.append("full_suite:missing_execution")
        return
    if not FULL_SUITE_ACCOUNTING_PATH.is_file():
        blockers.append("full_suite:missing_accounting")
        return
    try:
        execution = json.loads(FULL_SUITE_PATH.read_text(encoding="utf-8"))
        accounting = json.loads(
            FULL_SUITE_ACCOUNTING_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"full_suite:invalid_json:{exc}")
        return
    if execution.get("status") != "passed" or execution.get("full_suite_green") is not True:
        blockers.append("full_suite:status")
    if execution.get("pytest_exit_code") != 0:
        blockers.append("full_suite:pytest_exit_code")
    if any(execution.get(key, -1) != 0 for key in ("tests_failed", "tests_errored", "tests_skipped")):
        blockers.append("full_suite:nonzero_failures")
    if execution.get("tests_collected") != execution.get("tests_passed"):
        blockers.append("full_suite:count_mismatch")
    for key, expected in BOUNDARY.items():
        if execution.get(key) != expected or accounting.get(key) != expected:
            blockers.append(f"full_suite:boundary:{key}")
    if accounting.get("state") != "COMPLETED" or accounting.get("exit_code") != "0:0":
        blockers.append("full_suite:accounting_terminal")
    for section in ("junit", "stdout", "stderr", "test_inventory"):
        binding = execution.get(section)
        if not isinstance(binding, dict):
            blockers.append(f"full_suite:missing_binding:{section}")
            continue
        path = REPO_ROOT / str(binding.get("path", ""))
        if not path.is_file() or sha256_file(path) != str(binding.get("sha256", "")):
            blockers.append(f"full_suite:binding_hash:{section}")
    for section in ("source_files", "evaluation_files"):
        bindings = execution.get(section)
        if not isinstance(bindings, list) or not bindings:
            blockers.append(f"full_suite:missing_bindings:{section}")
            continue
        for binding in bindings:
            path = REPO_ROOT / str(binding.get("path", ""))
            if not path.is_file() or sha256_file(path) != str(binding.get("sha256", "")):
                blockers.append(f"full_suite:binding_hash:{section}")
    if int(execution.get("evaluation_file_count", -1)) != 13:
        blockers.append("full_suite:evaluation_file_count")
    if int(execution.get("source_file_count", -1)) != 11:
        blockers.append("full_suite:source_file_count")
    for binding in execution.get("checkpoints", []):
        path = REPO_ROOT / str(binding.get("path", ""))
        if not path.is_file() or path.stat().st_size != int(binding.get("bytes", -1)):
            blockers.append("full_suite:checkpoint_binding")


def validate(*, require_full_suite: bool = False) -> list[str]:
    blockers: list[str] = []
    tables: dict[Path, list[dict[str, str]]] = {}
    json_payloads: dict[Path, dict[str, Any]] = {}
    for path in UNIT_TABLES:
        if not path.is_file():
            blockers.append(f"missing:table:{path}")
            continue
        rows = _read_tsv(path)
        tables[path] = rows
        if not rows:
            blockers.append(f"empty:table:{path}")
        for index, row in enumerate(rows):
            for key, expected in BOUNDARY.items():
                actual: Any = row.get(key)
                if key == "protected_holdout_access":
                    actual = int(actual) if actual not in (None, "") else actual
                elif key == "publication_claim_eligible":
                    actual = str(actual).lower() == "true"
                if actual != expected:
                    blockers.append(f"boundary:{path.name}:row_{index}:{key}")
    manifest = tables.get(UNIT_TABLES[0], [])
    if len(manifest) != 12:
        blockers.append(f"unit_count:{len(manifest)}")
    if {row.get("policy_uid") for row in manifest} != {
        "R_random_information_action",
        "F_fixed_cascade",
        "U_uncertainty_only",
        "S_shared_belief_decision_risk",
    }:
        blockers.append("policy_inventory")
    if any(not row.get("posterior_uid_pre") or not row.get("posterior_uid_post") for row in manifest):
        blockers.append("posterior_identity")
    required_manifest_fields = (
        "selected_query",
        "s_utility",
        "u_utility",
        "s_query_rank",
        "u_query_rank",
        "world_action_forecasts_pre",
        "world_action_forecasts_post",
        "intervention_ranks_pre",
        "intervention_ranks_post",
        "reliability_components",
        "support_state",
        "terminal_decision",
        "terminal_decision_risk",
        "realized_cost",
    )
    for index, row in enumerate(manifest):
        for field in required_manifest_fields:
            if row.get(field) in (None, ""):
                blockers.append(f"manifest:row_{index}:missing:{field}")
        for field in (
            "s_query_rank",
            "u_query_rank",
            "world_action_forecasts_pre",
            "world_action_forecasts_post",
            "intervention_ranks_pre",
            "intervention_ranks_post",
        ):
            try:
                value = json.loads(row.get(field, ""))
            except (TypeError, json.JSONDecodeError):
                blockers.append(f"manifest:row_{index}:invalid_json:{field}")
                continue
            if not isinstance(value, list) or not value:
                blockers.append(f"manifest:row_{index}:empty_rank:{field}")
    for path in JSON_OUTPUTS:
        if not path.is_file():
            blockers.append(f"missing:json:{path}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"invalid:json:{path}:{exc}")
            continue
        if not isinstance(payload, dict):
            blockers.append(f"invalid:object:{path}")
            continue
        json_payloads[path] = payload
        for key, expected in BOUNDARY.items():
            if payload.get(key) != expected:
                blockers.append(f"boundary:{path.name}:{key}")
    gate_path = REPO_ROOT / "Output/audits/round26_information_action_gate.json"
    if gate_path in json_payloads:
        gate = json_payloads[gate_path]
        if gate.get("status") not in {
            "rescued_decision_consistent_information_action",
            "failed_information_action_mechanism_confirmed",
            "blocked_insufficient_decision_sensitive_information",
        }:
            blockers.append("gate:terminal_status")
        if int(gate.get("rescue_attempts", -1)) > 1:
            blockers.append("gate:rescue_attempts")
        if gate.get("protected_holdout_access") != 0:
            blockers.append("gate:protected_holdout_access")
    tool_path = REPO_ROOT / "Output/audits/round26_decision_sensitive_tool_coverage.json"
    if tool_path in json_payloads:
        tool = json_payloads[tool_path]
        if tool.get("decision_changing_units", 0) != 0 and tool.get("insufficient_decision_sensitive_information"):
            blockers.append("tool:inconsistent_decision_sensitive_summary")
        expected_counts = {
            "registered_tool_outcome_count": 28,
            "registered_information_action_pair_count": 14,
            "prospective_information_action_pair_count": 12,
            "terminal_evaluation_pair_count": 2,
        }
        for key, expected in expected_counts.items():
            if tool.get(key) != expected:
                blockers.append(f"tool:{key}")
    tool_rows = tables.get(UNIT_TABLES[4], [])
    if len(tool_rows) != 28:
        blockers.append(f"tool:outcome_row_count:{len(tool_rows)}")
    if {row.get("action_role") for row in tool_rows} != {"intervention", "matched_control"}:
        blockers.append("tool:action_roles")
    decomposition_path = REPO_ROOT / "Output/audits/round26_utility_decomposition.json"
    if decomposition_path in json_payloads:
        decomposition = json_payloads[decomposition_path]
        component_rows = decomposition.get("component_rows")
        if decomposition.get("component_row_count") != 168:
            blockers.append("utility:component_row_count")
        if not isinstance(component_rows, list) or len(component_rows) != 168:
            blockers.append("utility:component_rows")
    transmission_path = REPO_ROOT / "Output/audits/round26_decision_transmission_analysis.json"
    if transmission_path in json_payloads:
        transmission = json_payloads[transmission_path]
        if transmission.get("tri_score_available") is not False:
            blockers.append("transmission:tri_score_available")
        if transmission.get("tri_flattening_assessment") != "not_evaluable_model_tri_unavailable":
            blockers.append("transmission:tri_flattening_assessment")
        if transmission.get("s_u_disagreements_before_routing", 0) <= 0:
            blockers.append("transmission:missing_pre_routing_disagreement")
        if transmission.get("reliability_proxy_flattening_supported") is not False:
            blockers.append("transmission:reliability_proxy_flattening")
    freeze_path = REPO_ROOT / "Output/audits/round26_round25_negative_result_freeze.json"
    if freeze_path in json_payloads:
        freeze = json_payloads[freeze_path]
        for name, binding in freeze.get("source_artifacts", {}).items():
            source = REPO_ROOT / str(binding.get("path", ""))
            if not source.is_file():
                blockers.append(f"freeze:missing_source:{name}")
            elif sha256_file(source) != str(binding.get("sha256", "")):
                blockers.append(f"freeze:source_hash:{name}")
        if freeze.get("round25_raw_results_mutated") is not False:
            blockers.append("freeze:round25_raw_results_mutated")
        if freeze.get("unknown_identity_rows") != 0:
            blockers.append("freeze:unknown_identity_rows")
        source_names = set(freeze.get("source_artifacts", {}))
        if sum(name.startswith("round_receipt__") for name in source_names) != 12:
            blockers.append("freeze:round_receipt_count")
        if sum(name.startswith("round_report__") for name in source_names) != 12:
            blockers.append("freeze:round_report_count")
    if require_full_suite:
        _validate_full_suite(blockers)
    return sorted(set(blockers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-full-suite", action="store_true")
    args = parser.parse_args()
    blockers = validate(require_full_suite=args.require_full_suite)
    result = {"blocker_count": len(blockers), "blockers": blockers, "status": "passed" if not blockers else "failed"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
