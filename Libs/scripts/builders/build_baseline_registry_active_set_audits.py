from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_baseline_registry_audits import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_REGISTRY,
    DEFAULT_SCHEMA,
    load_inputs,
    repo_path,
)
from Libs.utils.data.raw_data_registry import write_json, write_tsv  # noqa: E402


MAIN_BASELINES = (
    "persistence_or_no_action",
    "linear_delta_or_state_space",
    "gru",
    "chemcpa",
    "cellot",
    "fno",
    "transolver_plus",
    "wdno",
    "deep_ensemble",
    "conformal_calibration",
)
MAIN_ABLATIONS = (
    "neuronsifter_deterministic",
    "neuronsifter_drug_id_conditioning",
    "neuronsifter_scalar_exposure",
    "neuronsifter_without_constraints_and_reliability",
    "neuronsifter_without_assimilation_and_what_if_pairing",
)
REQUIRED_CLAIMS = ("C2", "C4")
DEFERRED_CLAIMS = {
    "C1": "inactive_until_c1_reopened",
    "C3": "separate_c3_protocol",
    "C5": "supporting_scale_protocol_not_in_active_set",
}
REPLACEMENT_GROUPS: dict[str, dict[str, Any]] = {
    "chemical_perturbation": {
        "primary": "chemcpa",
        "replacements": ["cpa", "scgen"],
        "status": "primary_active_replacements_conditional",
    },
    "distribution_mapping": {
        "primary": "cellot",
        "replacements": ["conditional_monge_gap"],
        "status": "primary_active_replacements_conditional",
    },
    "deterministic_operator": {
        "primary": "transolver_plus",
        "replacements": ["transolver", "deeponet"],
        "status": "primary_active_replacements_conditional",
    },
    "stochastic_operator": {
        "primary": "wdno",
        "replacements": ["diffusionpde", "pdediff"],
        "status": "primary_active_replacements_conditional",
    },
    "state_foundation": {
        "primary": None,
        "replacements": ["geneformer", "scgpt", "scfoundation", "uce"],
        "status": "inactive_until_c1_reopened",
    },
    "tool_acquisition": {
        "primary": None,
        "replacements": ["infobax", "psbax", "mfmes", "pbgi", "cost_aware_bo"],
        "status": "separate_c3_protocol",
    },
}
AUDIT_FILES = {
    "active_set": "baseline_v31_active_set.tsv",
    "replacement_groups": "baseline_v31_replacement_groups.json",
    "claim_coverage": "baseline_v31_claim_coverage.json",
    "resource_reduction": "baseline_v31_resource_reduction.json",
    "gate": "baseline_v31_gate.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ICLR active-set reduction audits for Baseline Registry V3.1.")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--lock", dest="lock_path", default=DEFAULT_LOCK)
    parser.add_argument("--audit-root", default="Output/audits")
    return parser.parse_args()


def _profile_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = registry["active_set_profiles"]
    expected = set(MAIN_BASELINES) | set(MAIN_ABLATIONS)
    actual = {row["active_set_id"] for row in profiles}
    if actual != expected:
        raise ValueError(f"Active profile IDs mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if len(profiles) != len(actual):
        raise ValueError("Active profile IDs are not unique")
    registry_by_id = {row["baseline_id"]: row for row in registry["baselines"]}
    rows = []
    for profile in profiles:
        bound_id = profile["registry_baseline_id"]
        bound = registry_by_id.get(bound_id) if bound_id else None
        if bound_id and bound is None:
            raise ValueError(f"Active profile {profile['active_set_id']} binds unknown registry record {bound_id}")
        blockers = list(bound["blockers"]) if bound else ["profile_source_or_adapter_pending"]
        effective_source_status = profile["source_status"]
        if bound and (
            bound["official_repository"]["verification_status"] not in {"verified", "not_applicable"}
            or bound["license"] == "unknown"
            or "license_unclear" in bound["blockers"]
        ):
            effective_source_status = "source_verification_pending"
        if registry["round11_boundary"]["round11_exit_gate"] == "blocked_full_suite":
            blockers.append("blocked_round11_full_suite")
        rows.append(
            {
                "active_set_id": profile["active_set_id"],
                "registry_baseline_id": bound_id or "",
                "claim_ids": ",".join(profile["claim_ids"]),
                "paper_role": profile["paper_role"],
                "execution_tier": profile["execution_tier"],
                "replacement_group": profile["replacement_group"],
                "activation_condition": profile["activation_condition"],
                "main_table": str(profile["main_table"]).lower(),
                "source_status": effective_source_status,
                "upstream_status": bound["upstream_status"] if bound else "not_run",
                "execution_status": "not_started_round11_blocked",
                "blockers": ";".join(dict.fromkeys(blockers)),
            }
        )
    return rows


def _replacement_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active_ids = {row["active_set_id"] for row in rows}
    groups = {}
    blockers = []
    for name, spec in REPLACEMENT_GROUPS.items():
        primary = spec["primary"]
        active_replacements = [candidate for candidate in spec["replacements"] if candidate in active_ids]
        primary_active = primary in active_ids if primary else False
        ambiguous = len(active_replacements) > 1 or (primary_active and active_replacements)
        if ambiguous:
            blockers.append(f"replacement_group_ambiguous:{name}")
        groups[name] = {
            **spec,
            "primary_active": primary_active,
            "active_replacements": active_replacements,
            "activation_invariant_passed": not ambiguous,
        }
    return {
        "schema_version": "baseline_registry_active_set_replacement_groups",
        "groups": groups,
        "blockers": blockers,
        "status": "passed" if not blockers else "blocked_replacement_ambiguity",
    }


def _claim_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {row["active_set_id"] for row in rows}
    coverage = {
        "C1": {"required_for_active_set": False, "status": "deferred", "activation_condition": DEFERRED_CLAIMS["C1"], "trivial_or_simple_baseline": ["from_scratch_supervised_encoder"], "historical_or_classical_baseline": [], "current_strong_baseline": [], "same_backbone_ablation": []},
        "C2": {"required_for_active_set": True, "status": "covered", "activation_condition": "round11_full_suite_reopened", "trivial_or_simple_baseline": ["persistence_or_no_action", "linear_delta_or_state_space", "gru"], "historical_or_classical_baseline": ["linear_delta_or_state_space", "fno"], "current_strong_baseline": ["chemcpa", "cellot", "transolver_plus", "wdno"], "same_backbone_ablation": ["neuronsifter_deterministic", "neuronsifter_drug_id_conditioning", "neuronsifter_scalar_exposure", "neuronsifter_without_assimilation_and_what_if_pairing"]},
        "C3": {"required_for_active_set": False, "status": "deferred", "activation_condition": DEFERRED_CLAIMS["C3"], "trivial_or_simple_baseline": ["random_acquisition"], "historical_or_classical_baseline": ["fixed_cascade"], "current_strong_baseline": [], "same_backbone_ablation": []},
        "C4": {"required_for_active_set": True, "status": "covered", "activation_condition": "round11_full_suite_reopened", "trivial_or_simple_baseline": ["deep_ensemble"], "historical_or_classical_baseline": ["conformal_calibration"], "current_strong_baseline": ["deep_ensemble", "conformal_calibration"], "same_backbone_ablation": ["neuronsifter_without_constraints_and_reliability", "neuronsifter_without_assimilation_and_what_if_pairing"]},
        "C5": {"required_for_active_set": False, "status": "deferred", "activation_condition": DEFERRED_CLAIMS["C5"], "trivial_or_simple_baseline": ["linear_delta_or_state_space"], "historical_or_classical_baseline": [], "current_strong_baseline": [], "same_backbone_ablation": []},
    }
    for claim_id, entry in coverage.items():
        for key in ("trivial_or_simple_baseline", "historical_or_classical_baseline", "current_strong_baseline", "same_backbone_ablation"):
            entry[f"{key}_present"] = bool(entry[key]) and all(item in ids for item in entry[key])
        if entry["required_for_active_set"] and not all(entry[f"{key}_present"] for key in ("trivial_or_simple_baseline", "historical_or_classical_baseline", "current_strong_baseline", "same_backbone_ablation")):
            entry["status"] = "missing_required_comparator"
    return {"schema_version": "baseline_registry_active_set_claim_coverage", "claims": coverage, "required_claims": list(REQUIRED_CLAIMS), "status": "passed" if all(coverage[c]["status"] == "covered" for c in REQUIRED_CLAIMS) else "blocked_missing_claim_coverage"}


def build_all(registry_path: Path, schema_path: Path, lock_path: Path) -> dict[str, Any]:
    registry, _ = load_inputs(registry_path, schema_path, lock_path)
    rows = _profile_rows(registry)
    replacements = _replacement_report(rows)
    claims = _claim_coverage(rows)
    main_count = sum(row["paper_role"] == "main_baseline" for row in rows)
    ablation_count = sum(row["paper_role"] == "main_ablation" for row in rows)
    historical_count = int(registry["active_set_contract"]["historical_record_count"])
    resources = {
        "schema_version": "baseline_registry_active_set_resource_reduction",
        "historical_registry_record_count": historical_count,
        "active_main_baseline_count": main_count,
        "active_main_ablation_count": ablation_count,
        "active_full_reproduction_scope_count": main_count + ablation_count,
        "excluded_from_iclr_active_execution_count": historical_count - (main_count + ablation_count),
        "active_fraction_of_historical_records": round((main_count + ablation_count) / historical_count, 6),
        "planned_native_reproduction_count": main_count + ablation_count,
        "planned_matched_training_count": main_count + ablation_count,
        "executed_native_reproduction_count": 0,
        "executed_matched_training_count": 0,
        "compute_budget_estimate": "not_estimated_before_round11_gate",
        "formal_training_allowed": False,
    }
    gate_status = "active_set_frozen"
    blockers = []
    if main_count != 10 or ablation_count != 5:
        gate_status = "blocked_registry_inconsistency"
        blockers.append("active_count_mismatch")
    if replacements["status"] != "passed":
        gate_status = "blocked_replacement_ambiguity"
        blockers.extend(replacements["blockers"])
    if claims["status"] != "passed":
        gate_status = "blocked_missing_claim_coverage"
        blockers.append("required_claim_coverage_missing")
    gate = {
        "schema_version": "baseline_registry_active_set_gate",
        "status": gate_status,
        "historical_registry_record_count": historical_count,
        "main_baseline_count": main_count,
        "main_ablation_count": ablation_count,
        "headline_ready_count": 0,
        "formal_matched_training_allowed": False,
        "headline_comparison_allowed": False,
        "evidence_fabric_sealing_allowed": False,
        "DT_CORE_allowed": False,
        "round11_exit_gate": "blocked_full_suite",
        "round11_ceiling": "upstream_smoke_complete",
        "blockers": list(dict.fromkeys(blockers)),
        "source_or_license_pending_active_profiles": [row["active_set_id"] for row in rows if row["source_status"] == "source_verification_pending"],
    }
    return {"active_set": rows, "replacement_groups": replacements, "claim_coverage": claims, "resource_reduction": resources, "gate": gate}


def write_all(payload: dict[str, Any], audit_root: Path) -> None:
    write_tsv(audit_root / AUDIT_FILES["active_set"], list(payload["active_set"][0]), payload["active_set"])
    write_json(audit_root / AUDIT_FILES["replacement_groups"], payload["replacement_groups"])
    write_json(audit_root / AUDIT_FILES["claim_coverage"], payload["claim_coverage"])
    write_json(audit_root / AUDIT_FILES["resource_reduction"], payload["resource_reduction"])
    write_json(audit_root / AUDIT_FILES["gate"], payload["gate"])


def main() -> None:
    args = parse_args()
    payload = build_all(repo_path(args.registry), repo_path(args.schema), repo_path(args.lock_path))
    write_all(payload, repo_path(args.audit_root))
    print({"status": payload["gate"]["status"], "main_baseline_count": payload["gate"]["main_baseline_count"], "main_ablation_count": payload["gate"]["main_ablation_count"]})


if __name__ == "__main__":
    main()
