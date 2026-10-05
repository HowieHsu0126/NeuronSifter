"""Validate structural-scarcity adjudication and the prospective protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_evidence_scarcity_aware_validation import (  # noqa: E402
    PROTOCOL_SCHEMA_PATH,
    SCARCITY_SCHEMA_PATH,
    SOURCE_RECOVERY_SCHEMA_PATH,
    YIELD_SCHEMA_PATH,
    build,
)
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _schema_blockers(payload: dict[str, Any], path: Path, prefix: str) -> list[str]:
    schema = _read_json(path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{prefix}:{list(error.absolute_path)}:{error.message}" for error in errors]


def validate(*, recompute: bool = False) -> dict[str, Any]:
    policy = load_config(CONFIG_PATH)["evidence_scarcity_aware_validation"]
    outputs = {
        key: (REPO_ROOT / str(value)).resolve()
        for key, value in policy["outputs"].items()
    }
    blockers: list[str] = []
    try:
        recovery = _read_json(outputs["source_data_recovery_audit"])
        yield_audit = _read_json(outputs["ranking_yield_audit"])
        scarcity = _read_json(outputs["structural_scarcity_decision"])
        protocol = _read_json(outputs["prospective_protocol_audit"])
        report = outputs["failure_analysis_report"].read_text(encoding="utf-8")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "blocked", "passed": False, "blockers": [f"artifact_read:{exc}"]}
    blockers.extend(_schema_blockers(recovery, SOURCE_RECOVERY_SCHEMA_PATH, "recovery_schema"))
    blockers.extend(_schema_blockers(yield_audit, YIELD_SCHEMA_PATH, "yield_schema"))
    blockers.extend(_schema_blockers(scarcity, SCARCITY_SCHEMA_PATH, "scarcity_schema"))
    blockers.extend(_schema_blockers(protocol, PROTOCOL_SCHEMA_PATH, "protocol_schema"))
    if recovery.get("metadata_asset_count") != 79 or recovery.get("queue_record_count") != 0:
        blockers.append("source_data_recovery_funnel")
    if yield_audit.get("aggregate_funnel") != {
        "publication_metadata_records": 163,
        "repository_metadata_assets": 79,
        "queued_sources": 1,
        "recovered_sources": 1,
        "formal_abc_admissions": 0,
        "new_decision_sensitive_groups": 0,
    }:
        blockers.append("ranking_yield_funnel")
    if any(item.get("status") != "resolved" for item in recovery.get("repository_family_status", [])):
        blockers.append("source_data_repository_family_unresolved")
    if scarcity.get("current_evidence_state", {}).get("independent_decision_sensitive_groups") != 2:
        blockers.append("scarcity_current_group_count")
    if scarcity.get("adjudication", {}).get("global_absence_claimed") is not False:
        blockers.append("scarcity_global_absence_overclaim")
    if scarcity.get("adjudication", {}).get("ambiguous_context_promotion_allowed") is not False:
        blockers.append("scarcity_ambiguous_promotion")
    if scarcity.get("formal_authority", {}).get("additional_evidence_search") is not False:
        blockers.append("scarcity_additional_search_authority")
    population = protocol.get("evaluation_population", {})
    if population.get("positive_ranking_context_count") != 2:
        blockers.append("protocol_positive_context_count")
    if population.get("ambiguous_context_count") != 4:
        blockers.append("protocol_ambiguous_context_count")
    if population.get("train_context_count") != 0 or population.get("calibration_context_count") != 0:
        blockers.append("protocol_evaluation_context_training_leakage")
    authority = protocol.get("execution_authority", {})
    for key in (
        "execution_authorized",
        "split_authorized",
        "world_action_authorized",
        "gpu_authorized",
        "tri_authorized",
        "agent_in_twin_authorized",
        "coevolution_authorized",
    ):
        if authority.get(key) is not False:
            blockers.append(f"protocol_authority:{key}")
    if authority.get("protected_holdout_access") != 0:
        blockers.append("protocol_authority:protected_holdout_access")
    if scarcity["status"] not in report or "2 / 3" not in report:
        blockers.append("failure_analysis_report_content")
    if recompute and not blockers:
        try:
            rebuilt = build(write_output=False)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"recompute:{exc}")
        else:
            for artifact_key, fields in {
                "source_data_recovery": (
                    "status",
                    "metadata_asset_count",
                    "metadata_eligible_count",
                    "queue_record_count",
                    "source_recovered_count",
                    "formal_abc_admitted_count",
                    "new_legal_candidate_count",
                    "new_decision_sensitive_group_count",
                    "final_state",
                    "next_action",
                ),
                "structural_scarcity": (
                    "status",
                    "scope_boundary",
                    "route_evidence",
                    "current_evidence_state",
                    "adjudication",
                    "prohibited_interpretations",
                    "authorized_next_route",
                    "formal_authority",
                ),
                "ranking_yield": (
                    "status",
                    "route_count",
                    "routes",
                    "aggregate_funnel",
                    "unit_caveat",
                    "current_gate",
                    "additional_homologous_search_authorized",
                ),
                "prospective_protocol": (
                    "status",
                    "protocol_uid",
                    "claim_boundaries",
                    "evaluation_population",
                    "model_route_comparison",
                    "tasks",
                    "allowed_information_actions",
                    "baselines",
                    "metrics",
                    "reporting_policy",
                    "acceptance_conditions",
                    "execution_authority",
                ),
            }.items():
                current = {
                    "source_data_recovery": recovery,
                    "ranking_yield": yield_audit,
                    "structural_scarcity": scarcity,
                    "prospective_protocol": protocol,
                }[artifact_key]
                for field in fields:
                    if rebuilt[artifact_key].get(field) != current.get(field):
                        blockers.append(f"recompute:{artifact_key}:{field}")
    unique = sorted(set(blockers))
    return {
        "schema_version": "neuronsifter-evidence-scarcity-aware-validation-v1",
        "status": "passed" if not unique else "blocked",
        "passed": not unique,
        "blocker_count": len(unique),
        "blockers": unique,
        "structural_scarcity_status": scarcity.get("status"),
        "ranking_yield_status": yield_audit.get("status"),
        "prospective_protocol_status": protocol.get("status"),
        "positive_ranking_context_count": population.get("positive_ranking_context_count"),
        "ambiguous_context_count": population.get("ambiguous_context_count"),
        "execution_authorized": authority.get("execution_authorized"),
        "protected_holdout_access": authority.get("protected_holdout_access"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    result = validate(recompute=args.recompute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
