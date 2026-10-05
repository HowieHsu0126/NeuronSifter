from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic


PASS = "pass"
FAIL = "fail"
PENDING = "pending"
BLOCKER = "blocker"
MAJOR = "major"
MINOR = "minor"
SELF_AUDIT_FILENAMES = {
    "claim_gated_wording_audit.tsv",
    "g0_claim_wording_remediation_report.json",
    "world_foundation_model_claims_audit.json",
    "world_foundation_model_claims_audit.md",
}


@dataclass(frozen=True)
class ClaimCheck:
    check_id: str
    check_name: str
    status: str
    severity: str
    evidence: list[str]
    required_fix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NeuronSifter world/foundation-model claim discipline.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--mode", choices=["prelaunch", "postrun"], default="prelaunch")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--md-output", default="")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def artifact_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_row(
    check_id: str,
    check_name: str,
    condition: bool,
    evidence: list[str],
    required_fix: str,
    severity: str = BLOCKER,
) -> ClaimCheck:
    return ClaimCheck(
        check_id=check_id,
        check_name=check_name,
        status=PASS if condition else FAIL,
        severity=MINOR if condition else severity,
        evidence=evidence,
        required_fix="No fix required." if condition else required_fix,
    )


def pending_row(
    check_id: str,
    check_name: str,
    evidence: list[str],
    required_fix: str,
    severity: str = MAJOR,
) -> ClaimCheck:
    return ClaimCheck(
        check_id=check_id,
        check_name=check_name,
        status=PENDING,
        severity=severity,
        evidence=evidence,
        required_fix=required_fix,
    )


def transition_backbone(config: dict[str, object]) -> dict[str, object]:
    return dict(
        config["screening_pipeline"]["screening_benchmark"]
        ["model_architecture"]["rollout_operator"]["transition_backbone"]
    )


def operator_config(config: dict[str, object]) -> dict[str, object]:
    return dict(
        config["screening_pipeline"]["screening_benchmark"]
        ["model_architecture"]["rollout_operator"]
    )


def overclaim_fragments(config: dict[str, object]) -> list[str]:
    fragments: list[str] = [
        "G5 passed",
        "1M trajectory readiness passed",
        "materialized 1M trajectory gate passed",
        "general drug screening",
        "direct clinical endpoint prediction",
        "prospective efficacy prediction",
        "private clinical data training",
        "patient-level longitudinal supervision",
        "PD readiness",
        "FTD readiness",
        "ALS readiness",
    ]
    suite = dict(config.get("paper_grade_full_suite", {}))
    for rule in suite.get("claim_prohibition_rules", []):
        fragments.extend(str(value) for value in dict(rule).get("forbidden_fragments", []))
    return list(dict.fromkeys(fragments))


def guardrail_context(text: str, index: int) -> bool:
    start = max(index - 240, 0)
    end = min(index + 240, len(text))
    window = text[start:end].lower()
    guardrail_terms = [
        "claim_guardrails",
        "forbidden",
        "forbidden_claims",
        "prohibited",
        "must not",
        "do not",
        "do not claim",
        "no ",
        "not claim",
        "not_claimed",
        "not passed",
        "not_passed",
        "without claiming",
    ]
    return any(term in window for term in guardrail_terms)


def overclaim_matches(config: dict[str, object]) -> list[str]:
    roots = [
        REPO_ROOT / "Output/paper_reports",
        REPO_ROOT / "Output/audits",
        REPO_ROOT / "Docs/manuscript",
    ]
    matches: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".tex", ".txt", ".tsv"}:
                continue
            if path.name in SELF_AUDIT_FILENAMES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            lowered = text.lower()
            for fragment in overclaim_fragments(config):
                fragment_lower = fragment.lower()
                search_index = lowered.find(fragment_lower)
                while search_index >= 0:
                    if not guardrail_context(text, search_index):
                        matches.append(f"{repo_relative(path)}::{fragment}")
                    search_index = lowered.find(fragment_lower, search_index + len(fragment_lower))
    return matches


def artifact_exists(path_value: object) -> bool:
    return artifact_path(path_value).exists()


def build_report(config_path: str | Path, mode: str = "prelaunch") -> dict[str, object]:
    config = load_config(config_path)
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    architecture = benchmark["model_architecture"]
    backbone = transition_backbone(config)
    checks: list[ClaimCheck] = []

    world_state_factors = config["foundation_twin"]["world_state_factors"]
    hidden_factor_ids = [
        str(factor["factor_id"])
        for factor in world_state_factors
        if bool(factor.get("hidden_state_support"))
    ]
    checks.append(
        check_row(
            "WORLD-001",
            "Structured world-state representation remains explicit.",
            {"z_tx", "z_pk", "z_env", "z_ephys", "z_morph"} <= set(hidden_factor_ids),
            ["hidden_factor_ids=" + ",".join(hidden_factor_ids)],
            "Restore explicit treatment, PK/exposure, environment, electrophysiology, and morphology factors.",
        )
    )

    world_state_config = dict(architecture.get("world_state_posterior", {}))
    screening_system_text = source_text(REPO_ROOT / "Libs/models/networks/screening_system.py")
    checks.append(
        check_row(
            "WORLD-002",
            "World-state posterior exposes uncertainty-aware inference.",
            bool(world_state_config)
            and "infer_world_state" in screening_system_text
            and "posterior_payload" in screening_system_text
            and "field_trajectory_std" in screening_system_text,
            [
                "world_state_posterior_keys=" + ",".join(sorted(world_state_config)),
                "runtime=infer_world_state",
            ],
            "Restore factorized posterior configuration and runtime posterior uncertainty outputs.",
        )
    )

    action_config = dict(operator_config(config)["action_representation"])
    pkpd_registry = config["screening_pipeline"]["pkpd_ready_registry"]
    action_registry = config["screening_pipeline"]["action_ready_registry"]
    pkpd_paths = set(str(key) for key in pkpd_registry["relative_paths"])
    action_paths = set(str(key) for key in action_registry["relative_paths"])
    checks.append(
        check_row(
            "WORLD-003",
            "Rollout dynamics consume structured occupancy-conditioned actions.",
            str(action_config["active_policy"]) == "occupancy_program"
            and {"route_catalog", "schedule_catalog", "dose_grid_catalog"} <= pkpd_paths
            and {"target_program_catalog", "drug_program_catalog", "regimen_catalog"} <= action_paths,
            [
                f"active_action_policy={action_config['active_policy']}",
                "structured_action_terms=target_program,drug_program,dose,schedule,route",
            ],
            "Restore the occupancy_program action representation and structured action fields.",
        )
    )

    mechanism_moe = dict(backbone.get("mechanism_moe", {}))
    required_diagnostics = {
        "expert_utilization",
        "router_entropy",
        "operator_splitting_species_attribution",
        "operator_splitting_coupling_matrix",
        "operator_splitting_score_norm_by_expert",
        "operator_splitting_uncertainty_by_expert",
        "operator_splitting_case_activation",
    }
    diagnostics = set(str(value) for value in backbone.get("mechanism_explainability", {}).get("required_diagnostics", []))
    checks.append(
        check_row(
            "WORLD-004",
            "NeuronSifter remains the mainline mechanism-informed diffusion neural operator.",
            str(backbone["kind"]) == "neuronsifter_world_action"
            and len(mechanism_moe.get("expert_names", [])) >= 10
            and required_diagnostics <= diagnostics,
            [
                f"operator_kind={backbone['kind']}",
                f"expert_count={len(mechanism_moe.get('expert_names', []))}",
                "missing_diagnostics=" + ",".join(sorted(required_diagnostics - diagnostics)),
            ],
            "Do not promote Transolver or external operators to mainline; restore required mechanism diagnostics.",
        )
    )

    serving_text = source_text(REPO_ROOT / "Libs/scripts/services/serve_screening_api.py")
    endpoint_set = set(str(value) for value in config["screening_pipeline"]["screening_serving"]["public_endpoints"])
    required_endpoints = {
        "/infer_world_state",
        "/rollout_counterfactuals",
        "/plan_interventions",
        "/explain_decision",
        "/ingest_feedback",
    }
    checks.append(
        check_row(
            "WORLD-005",
            "Serving exposes rollout, planning, decision explanation, and feedback endpoints.",
            required_endpoints <= endpoint_set
            and all(token in serving_text for token in ["rollout_counterfactuals", "plan_interventions", "explain_decision", "ingest_feedback"]),
            [
                "configured_endpoints=" + ",".join(sorted(endpoint_set)),
                "missing_endpoints=" + ",".join(sorted(required_endpoints - endpoint_set)),
            ],
            "Restore the public world-state, counterfactual, planning, explanation, and feedback serving endpoints.",
        )
    )

    decision_gate = dict(benchmark["decision_gate"])
    labels = set(str(value) for value in decision_gate["decision_labels"].values())
    required_labels = {"accept", "reject", "active_learning", "external_validation", "insufficient_evidence"}
    checks.append(
        check_row(
            "WORLD-006",
            "TRI/OOD-aware decision routing is configured.",
            required_labels <= labels
            and "tri_accept_threshold" in decision_gate
            and "ood_reject_threshold" in decision_gate
            and "pkpd_consistency_threshold" in decision_gate,
            [
                "decision_labels=" + ",".join(sorted(labels)),
                f"tri_accept_threshold={decision_gate.get('tri_accept_threshold')}",
                f"ood_reject_threshold={decision_gate.get('ood_reject_threshold')}",
                f"pkpd_consistency_threshold={decision_gate.get('pkpd_consistency_threshold')}",
            ],
            "Restore decision labels and TRI, OOD, and PK/PD gate thresholds.",
        )
    )

    training_config = benchmark["training"]
    pretraining_objectives = dict(training_config["pretraining_objectives"])
    teacher_family_training = dict(training_config["teacher_family_training"])
    checks.append(
        check_row(
            "WORLD-007",
            "Foundation-style pretraining is configured with multi-objective teacher supervision.",
            bool(pretraining_objectives)
            and bool(teacher_family_training.get("enabled"))
            and len(teacher_family_training.get("families", [])) >= 5,
            [
                "pretraining_objectives=" + ",".join(sorted(pretraining_objectives)),
                f"teacher_family_count={len(teacher_family_training.get('families', []))}",
            ],
            "Restore masked reconstruction, next-state, rollout, distillation, counterfactual, and router objectives.",
        )
    )

    suite = dict(config.get("paper_grade_full_suite", {}))
    scratch_template = str(
        suite.get("models", {})
        .get("neuronsifter_scratch", {})
        .get("result_catalog_template", "")
    )
    finetuned_artifact = str(
        suite.get("models", {})
        .get("neuronsifter_mainline", {})
        .get("result_catalog_template", "")
    )
    scratch_exists = any(artifact_exists(scratch_template.format(seed=seed)) for seed in suite.get("formal_seed_values", [])) if scratch_template else False
    finetuned_exists = any(artifact_exists(finetuned_artifact.format(seed=seed)) for seed in suite.get("formal_seed_values", [])) if finetuned_artifact else False
    if mode == "postrun":
        checks.append(
            check_row(
                "WORLD-008",
                "Pretrained-vs-scratch formal evidence is materialized.",
                scratch_exists and finetuned_exists,
                [f"scratch_exists={scratch_exists}", f"mainline_exists={finetuned_exists}"],
                "Run the suite mainline and scratch profiles for all fixed suite seeds before making empirical foundation claims.",
            )
        )
    else:
        checks.append(
            pending_row(
                "WORLD-008",
                "Pretrained-vs-scratch formal evidence is pending prelaunch.",
                [f"scratch_template={scratch_template}", f"mainline_template={finetuned_artifact}"],
                "Materialize postrun evidence before upgrading any foundation-model empirical claim.",
            )
        )

    frozen_finetuned_summary = artifact_path("Output/paper_reports/frozen_vs_finetuned_summary.json")
    if mode == "postrun":
        checks.append(
            check_row(
                "WORLD-009",
                "Frozen-vs-finetuned formal evidence is materialized.",
                frozen_finetuned_summary.exists(),
                [repo_relative(frozen_finetuned_summary)],
                "Run and report frozen-vs-finetuned evidence before making adaptation claims.",
            )
        )
    else:
        checks.append(
            pending_row(
                "WORLD-009",
                "Frozen-vs-finetuned formal evidence is pending prelaunch.",
                [repo_relative(frozen_finetuned_summary)],
                "Keep adaptation claims pending until the postrun summary exists.",
            )
        )

    matches = overclaim_matches(config)
    checks.append(
        check_row(
            "WORLD-010",
            "Public artifacts do not overclaim beyond current evidence.",
            not matches,
            matches or ["no public overclaim fragments found outside guardrail context"],
            "Remove or evidence-gate public claims about G5, 1M readiness, general screening, private training, endpoint prediction, or secondary-disease readiness.",
        )
    )

    failed_checks = [check for check in checks if check.status == FAIL]
    return {
        "audit_name": "world_foundation_model_claims",
        "mode": mode,
        "config_path": repo_relative(artifact_path(config_path)),
        "status": "pass" if not failed_checks else "fail",
        "failed_check_count": len(failed_checks),
        "pending_check_count": sum(1 for check in checks if check.status == PENDING),
        "checks": [asdict(check) for check in checks],
    }


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# World/Foundation Model Claims Audit",
        "",
        f"- mode: `{report['mode']}`",
        f"- status: `{report['status']}`",
        f"- failed_check_count: `{report['failed_check_count']}`",
        f"- pending_check_count: `{report['pending_check_count']}`",
        "",
        "| check_id | status | severity | check_name | evidence | required_fix |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["checks"]:
        evidence = "<br>".join(str(value) for value in row["evidence"])
        lines.append(
            f"| {row['check_id']} | {row['status']} | {row['severity']} | "
            f"{row['check_name']} | {evidence} | {row['required_fix']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = config.get("paper_grade_full_suite", {}).get("output_paths", {})
    json_output = args.json_output or str(outputs.get("world_foundation_json", "Output/audits/world_foundation_model_claims_audit.json"))
    md_output = args.md_output or str(outputs.get("world_foundation_md", "Output/audits/world_foundation_model_claims_audit.md"))
    report = build_report(args.config, args.mode)
    write_json(artifact_path(json_output), report)
    write_text_atomic(artifact_path(md_output), markdown_report(report))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
