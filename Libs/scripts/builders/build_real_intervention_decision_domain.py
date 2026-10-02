"""Qualify the real-intervention domain for decision-sensitive benchmarking."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
BUILDER_ID = "Libs/scripts/builders/build_real_intervention_decision_domain.py"
UNAVAILABLE = "explicit_unavailable"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes the repository: {value}")
    return path


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _context_group_uid(values: Sequence[str]) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:24]
    return f"real_intervention_context_group:{digest}"


def _historical_outcome_uid(source_key: str, baseline_state_uid: str, time_grid: str) -> str:
    if source_key == "clue_npc_neu":
        return "clue_npc_neu_without_materialized_controls"
    if source_key == "pantethine_hippocampus":
        genotype = baseline_state_uid.rsplit(":", 1)[-1]
        return f"E-MTAB-6772:{genotype}:pantethine_vs_vehicle"
    if source_key == "dexamethasone_neurons":
        return "GSE262992:dexamethasone_vs_control"
    if source_key == "amyloid_peptide_timecourse":
        match = re.search(r"[-+]?\d+(?:\.\d+)?", time_grid)
        if match is None:
            raise ValueError(f"Amyloid time grid has no numeric value: {time_grid}")
        time_value = float(match.group(0))
        return f"E-MTAB-1841:SH-SY5Y:{time_value:g}h"
    raise ValueError(f"Unsupported real-intervention source: {source_key}")


def classify_context_group(
    *,
    source_key: str,
    action_count: int,
    outcome_rank_classification: str,
) -> str:
    if source_key == "clue_npc_neu":
        return "ambiguous"
    if outcome_rank_classification in {"near_null", "null"}:
        return "null"
    if outcome_rank_classification in {
        "decision_sensitive",
        "weakly_decision_sensitive",
        "scientifically_informative_but_not_rank_sensitive",
        "scientifically_informative_but_not_rankable",
    }:
        return "scientifically_informative_but_not_rankable"
    if action_count < 2 and outcome_rank_classification == "ambiguous":
        return "ambiguous"
    return "ambiguous"


def _blockers(
    *,
    source_key: str,
    action_count: int,
    decision_objective_available: bool,
) -> tuple[str, list[str]]:
    secondary: list[str] = []
    if action_count < 2:
        primary = "fewer_than_two_candidate_actions"
    elif source_key == "clue_npc_neu":
        primary = "matched_control_outcome_values_unavailable"
    elif not decision_objective_available:
        primary = "prespecified_therapeutic_decision_objective_unavailable"
    else:
        primary = "no_blocker"
    if not decision_objective_available and primary != "prespecified_therapeutic_decision_objective_unavailable":
        secondary.append("prespecified_therapeutic_decision_objective_unavailable")
    if source_key == "amyloid_peptide_timecourse":
        secondary.append("pathogenic_perturbations_are_not_therapeutic_candidate_actions")
    if source_key == "clue_npc_neu" and primary != "matched_control_outcome_values_unavailable":
        secondary.append("matched_control_outcome_values_unavailable")
    return primary, secondary


def build_context_groups(
    episodes: Sequence[Mapping[str, str]],
    actions: Sequence[Mapping[str, str]],
    historical_outcomes: Sequence[Mapping[str, str]],
    group_fields: Sequence[str],
) -> list[dict[str, object]]:
    action_uids = {str(row["structured_action_uid"]) for row in actions}
    if len(action_uids) != len(actions):
        raise ValueError("Structured action identities are not unique")
    grouped: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for episode in episodes:
        action_uid = str(episode["structured_action_uid"])
        if action_uid not in action_uids:
            raise ValueError(f"Episode references an unknown action: {action_uid}")
        grouped[tuple(str(episode[field]) for field in group_fields)].append(episode)
    historical_by_uid = {str(row["candidate_set_uid"]): row for row in historical_outcomes}
    if len(historical_by_uid) != len(historical_outcomes):
        raise ValueError("Historical outcome candidate identities are not unique")

    rows: list[dict[str, object]] = []
    for group_key, members in sorted(grouped.items()):
        representative = members[0]
        source_key = str(representative["source_key"])
        structured_action_uids = sorted({str(row["structured_action_uid"]) for row in members})
        control_uids = sorted({str(row["control_uid"]) for row in members})
        historical_uid = _historical_outcome_uid(
            source_key,
            str(representative["baseline_state_uid"]),
            str(representative["time_grid"]),
        )
        historical = historical_by_uid.get(historical_uid)
        if historical is None:
            raise ValueError(f"No frozen real-outcome diagnostic for context group: {historical_uid}")
        action_count = len(structured_action_uids)
        outcome_classification = str(historical["outcome_rank_classification"])
        decision_objective_available = _as_bool(historical["decision_objective_available"])
        classification = classify_context_group(
            source_key=source_key,
            action_count=action_count,
            outcome_rank_classification=outcome_classification,
        )
        primary_blocker, secondary_blockers = _blockers(
            source_key=source_key,
            action_count=action_count,
            decision_objective_available=decision_objective_available,
        )
        outcome_values_materialized = source_key != "clue_npc_neu"
        structural_multi_action_set = action_count >= 2
        qualified = bool(
            structural_multi_action_set
            and outcome_values_materialized
            and decision_objective_available
            and classification in {"decision_sensitive", "weakly_decision_sensitive"}
        )
        action_set_role = {
            "clue_npc_neu": "compound_intervention_candidates",
            "amyloid_peptide_timecourse": "pathogenic_perturbation_comparison",
            "pantethine_hippocampus": "single_intervention_contrast",
            "dexamethasone_neurons": "single_intervention_contrast",
        }[source_key]
        rows.append(
            {
                "context_group_uid": _context_group_uid(group_key),
                "source_key": source_key,
                "baseline_state_uid": representative["baseline_state_uid"],
                "cell_or_tissue_context": representative["cell_or_tissue_context"],
                "time_grid": representative["time_grid"],
                "measurement_namespace": representative["observation_measurement_namespace"],
                "episode_count": len(members),
                "action_count": action_count,
                "structured_action_uids": json.dumps(structured_action_uids, separators=(",", ":")),
                "control_uid_count": len(control_uids),
                "control_uids": json.dumps(control_uids, separators=(",", ":")),
                "structural_multi_action_set": structural_multi_action_set,
                "action_set_role": action_set_role,
                "historical_outcome_candidate_uid": historical_uid,
                "real_outcome_diagnostic_bound": True,
                "outcome_values_materialized": outcome_values_materialized,
                "effect_magnitude": historical["effect_magnitude"],
                "effect_direction": historical["effect_direction"],
                "between_action_contrast": historical["between_action_contrast"],
                "within_control_variability": historical["within_control_variability"],
                "rank_separability": historical["rank_separability"],
                "decision_margin": historical["decision_margin"],
                "decision_margin_ci95": historical["decision_margin_ci95"],
                "outcome_rank_classification": outcome_classification,
                "therapeutic_decision_objective_available": decision_objective_available,
                "decision_domain_classification": classification,
                "qualified_decision_candidate_set": qualified,
                "split_eligible": qualified,
                "primary_blocker": primary_blocker,
                "secondary_blockers": "|".join(secondary_blockers),
                "model_predictions_inspected": False,
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
            }
        )
    if sum(int(row["episode_count"]) for row in rows) != len(episodes):
        raise ValueError("Context-group membership does not cover every episode exactly once")
    return rows


def _fieldnames() -> list[str]:
    return [
        "context_group_uid",
        "source_key",
        "baseline_state_uid",
        "cell_or_tissue_context",
        "time_grid",
        "measurement_namespace",
        "episode_count",
        "action_count",
        "structured_action_uids",
        "control_uid_count",
        "control_uids",
        "structural_multi_action_set",
        "action_set_role",
        "historical_outcome_candidate_uid",
        "real_outcome_diagnostic_bound",
        "outcome_values_materialized",
        "effect_magnitude",
        "effect_direction",
        "between_action_contrast",
        "within_control_variability",
        "rank_separability",
        "decision_margin",
        "decision_margin_ci95",
        "outcome_rank_classification",
        "therapeutic_decision_objective_available",
        "decision_domain_classification",
        "qualified_decision_candidate_set",
        "split_eligible",
        "primary_blocker",
        "secondary_blockers",
        "model_predictions_inspected",
        "evidence_role",
        "publication_claim_eligible",
        "protected_holdout_access",
    ]


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("Decision-domain qualification is identity-bound to the repository root")
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]["decision_domain_qualification"]
    if not isinstance(policy, Mapping):
        raise TypeError("Decision-domain qualification policy must be a mapping")
    inputs = {
        "episode_catalog": _repo_path(str(policy["input_episode_catalog"])),
        "action_catalog": _repo_path(str(policy["input_action_catalog"])),
        "bridge_quality": _repo_path(str(policy["input_bridge_quality"])),
        "bridge_integrity": _repo_path(str(policy["input_bridge_integrity"])),
        "outcome_sensitivity": _repo_path(str(policy["input_outcome_sensitivity"])),
        "outcome_sensitivity_summary": _repo_path(str(policy["input_outcome_sensitivity_summary"])),
    }
    outputs = {name: _repo_path(str(path)) for name, path in policy["outputs"].items()}
    planned_splits = {name: _repo_path(str(path)) for name, path in policy["planned_split_paths"].items()}
    existing_splits = [path for path in planned_splits.values() if path.exists()]
    if existing_splits:
        raise FileExistsError(f"Decision benchmark splits exist before qualification: {existing_splits}")

    bridge_quality = _json(inputs["bridge_quality"])
    bridge_integrity = _json(inputs["bridge_integrity"])
    outcome_summary = _json(inputs["outcome_sensitivity_summary"])
    if bridge_quality.get("model_predictions_inspected") is not False:
        raise ValueError("Bridge quality does not preserve pre-model qualification")
    if bridge_integrity.get("status") != "passed":
        raise ValueError("Bridge integrity is not passed")
    if outcome_summary.get("classification_is_based_on_real_outcomes") is not True:
        raise ValueError("Frozen sensitivity diagnostics are not real-outcome based")
    if outcome_summary.get("model_predictions_inspected") is not False:
        raise ValueError("Frozen sensitivity diagnostics inspected model predictions")

    episodes = read_tsv(inputs["episode_catalog"])
    actions = read_tsv(inputs["action_catalog"])
    historical_outcomes = read_tsv(inputs["outcome_sensitivity"])
    group_fields = [str(field) for field in policy["candidate_context_fields"]]
    rows = build_context_groups(episodes, actions, historical_outcomes, group_fields)
    classification_values = {str(value) for value in policy["classification_values"]}
    observed_classifications = {str(row["decision_domain_classification"]) for row in rows}
    if not observed_classifications <= classification_values:
        raise ValueError(f"Unknown decision-domain classifications: {observed_classifications - classification_values}")
    write_tsv(outputs["context_group_qualification"], _fieldnames(), rows)

    multi_action_rows = [row for row in rows if bool(row["structural_multi_action_set"])]
    qualified_rows = [row for row in rows if bool(row["qualified_decision_candidate_set"])]
    if qualified_rows:
        raise ValueError("Qualified sets require a separate pre-model split-freeze implementation")
    split_audit = {
        "schema_version": "neuronsifter-real-intervention-decision-benchmark-split-audit-v1",
        "status": "not_created_no_qualified_candidate_sets",
        "split_policy": policy["split_policy"],
        "structural_multi_action_set_count": len(multi_action_rows),
        "qualified_decision_candidate_set_count": 0,
        "planned_split_paths": {
            name: path.relative_to(REPO_ROOT).as_posix() for name, path in planned_splits.items()
        },
        "split_files_created": False,
        "development_splits_frozen": False,
        "leakage_audit_status": "not_applicable_no_benchmark",
        "model_training_authorized": False,
        "model_predictions_inspected": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    write_json(outputs["split_audit"], split_audit)

    context_classifications = Counter(str(row["decision_domain_classification"]) for row in rows)
    candidate_classifications = Counter(
        str(row["decision_domain_classification"]) for row in multi_action_rows
    )
    source_group_counts = Counter(str(row["source_key"]) for row in rows)
    source_candidate_counts = Counter(str(row["source_key"]) for row in multi_action_rows)
    summary: dict[str, object] = {
        "schema_version": policy["schema_version"],
        "created_at": _now(),
        "status": policy["blocked_status"],
        "gate_uid": "real_intervention_decision_domain_qualified",
        "gate_resolved": True,
        "decision_sensitive_domain_available": False,
        "model_training_domain_qualified": False,
        "decision_sensitivity_known": True,
        "candidate_context_fields": group_fields,
        "episode_count": len(episodes),
        "action_count": len(actions),
        "context_group_count": len(rows),
        "structural_multi_action_set_count": len(multi_action_rows),
        "single_action_context_group_count": len(rows) - len(multi_action_rows),
        "multi_action_episode_count": sum(int(row["episode_count"]) for row in multi_action_rows),
        "qualified_decision_candidate_set_count": 0,
        "context_classification_counts": dict(sorted(context_classifications.items())),
        "candidate_classification_counts": dict(sorted(candidate_classifications.items())),
        "source_context_group_counts": dict(sorted(source_group_counts.items())),
        "source_candidate_set_counts": dict(sorted(source_candidate_counts.items())),
        "real_action_response_variation_available": True,
        "prespecified_therapeutic_decision_objective_available": False,
        "dominant_blockers": [
            "matched_control_outcome_values_unavailable_for_clue_candidate_sets",
            "prespecified_therapeutic_decision_objective_unavailable",
            "pathogenic_amyloid_perturbations_are_not_therapeutic_candidate_actions",
        ],
        "classification_is_based_on_real_outcomes": True,
        "historical_outcome_diagnostics_mutated": False,
        "model_predictions_inspected": False,
        "split_files_created": False,
        "development_splits_frozen": False,
        "next_route": "target_evidence_expansion_for_decision_objective_and_matched_control_outcomes",
        "world_action_target_claim_retained": True,
        "agent_in_twin_target_claim_retained": True,
        "coevolution_target_claim_retained": True,
        "agent_in_twin_authorized": False,
        "coevolution_authorized": False,
        "publication_comparison_open": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
        "input_bindings": {name: _artifact(path) for name, path in inputs.items()},
        "implementation_bindings": {
            "schema": _artifact(_repo_path(str(policy["summary_schema_path"]))),
            "config": _artifact(REPO_ROOT / "Libs/configs/data/real_intervention_evidence.yaml"),
            "builder": _artifact(REPO_ROOT / BUILDER_ID),
        },
        "output_bindings": {
            "context_group_qualification": _artifact(outputs["context_group_qualification"]),
            "split_audit": _artifact(outputs["split_audit"]),
        },
    }
    write_json(outputs["domain_summary"], summary)
    return summary


def main() -> int:
    summary = build()
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "episode_count",
                    "action_count",
                    "context_group_count",
                    "structural_multi_action_set_count",
                    "qualified_decision_candidate_set_count",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
