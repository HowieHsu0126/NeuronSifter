"""Validate the real-intervention evidence and frozen World-Action closure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
ALLOWED_TERMINAL_STATUSES = {
    "c2b_real_intervention_development_resolved",
    "failed_real_intervention_world_action",
    "blocked_insufficient_real_intervention_evidence",
}


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes the repository: {value}")
    return path


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _artifact_matches(binding: Mapping[str, object]) -> bool:
    path = _repo_path(str(binding.get("path", "")))
    return (
        path.is_file()
        and path.stat().st_size == int(binding.get("bytes", -1))
        and sha256_file(path) == binding.get("sha256")
    )


def _historical_artifact_matches_successor(binding: Mapping[str, object]) -> bool:
    """Accept an immutable historical byte binding with an audited successor.

    Round-27 reproducibility manifests remain historical records.  A later
    targeted-policy edit may legitimately change a source file, but only an
    append-only successor chain may authorize that drift; the historical
    manifest itself is never rewritten.
    """
    successor_path = REPO_ROOT / "Output/audits/round23_current_source_successor_binding_v4.json"
    predecessor_path = REPO_ROOT / "Output/audits/real_intervention_round23_source_successor_binding_v3.json"
    if not successor_path.is_file() or not predecessor_path.is_file():
        return False
    try:
        successor = _json(successor_path)
        predecessor = _json(predecessor_path)
        declared_predecessor = successor.get("predecessor_binding", {})
        if (
            successor.get("schema_version")
            != "neuronsifter-round23-current-source-successor-binding-v4"
            or successor.get("status") != "bound_successor_development"
            or successor.get("publication_claim_eligible") is not False
            or successor.get("protected_test_access") != 0
            or not isinstance(declared_predecessor, Mapping)
            or declared_predecessor.get("path")
            != predecessor_path.relative_to(REPO_ROOT).as_posix()
            or declared_predecessor.get("sha256") != sha256_file(predecessor_path)
        ):
            return False
        path = str(binding.get("path", ""))
        rows = {
            str(row.get("path", "")): row
            for row in successor.get("source_transitions", [])
            if isinstance(row, Mapping)
        }
        row = rows.get(path)
        if not isinstance(row, Mapping):
            return False
        current_path = _repo_path(path)
        return (
            binding.get("sha256") == row.get("predecessor_sha256")
            and current_path.is_file()
            and sha256_file(current_path) == row.get("current_sha256")
        )
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _historical_reproducibility_artifact_matches_successor(
    binding: Mapping[str, object], manifest_path: Path
) -> bool:
    successor_path = REPO_ROOT / "Output/audits/round27_reproducibility_source_successor_binding_v1.json"
    if not successor_path.is_file() or not manifest_path.is_file():
        return False
    try:
        successor = _json(successor_path)
        predecessor = successor.get("predecessor_manifest", {})
        if (
            successor.get("schema_version")
            != "neuronsifter-round27-reproducibility-source-successor-binding-v1"
            or successor.get("status") != "bound_successor_development"
            or successor.get("historical_manifest_rewritten") is not False
            or successor.get("publication_claim_eligible") is not False
            or successor.get("protected_holdout_access") != 0
            or not isinstance(predecessor, Mapping)
            or predecessor.get("path") != manifest_path.relative_to(REPO_ROOT).as_posix()
            or predecessor.get("sha256") != sha256_file(manifest_path)
        ):
            return False
        rows = {
            str(row.get("path", "")): row
            for row in successor.get("source_transitions", [])
            if isinstance(row, Mapping)
        }
        row = rows.get(str(binding.get("path", "")))
        path = _repo_path(str(binding.get("path", "")))
        return bool(
            isinstance(row, Mapping)
            and binding.get("sha256") == row.get("predecessor_sha256")
            and binding.get("bytes") == row.get("predecessor_bytes")
            and path.is_file()
            and path.stat().st_size == row.get("current_bytes")
            and sha256_file(path) == row.get("current_sha256")
        )
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def validate(require_full_suite: bool = False) -> dict[str, object]:
    blockers: list[str] = []
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]
    outputs = {name: _repo_path(str(value)) for name, value in policy["outputs"].items()}
    for name, path in outputs.items():
        if not path.is_file():
            blockers.append(f"missing_artifact:{name}")
    if blockers:
        return {"status": "blocked", "blocker_count": len(blockers), "blockers": blockers}

    boundary = policy["evidence_boundary"]
    route_freeze = _json(outputs["scientific_route_freeze"])
    claims = _json(outputs["claim_reselection"])
    coverage = _json(outputs["real_e2_e4_coverage"])
    integrity = _json(outputs["episode_integrity"])
    sensitivity = _json(outputs["decision_sensitive_summary"])
    leakage = _json(outputs["leakage_audit"])
    u0_gate = _json(outputs["u0_gate"])
    training = _json(outputs["training_gate"])
    scope = _json(outputs["scope_audit"])
    gate = _json(outputs["c2b_gate"])
    architecture = _json(outputs["canonical_architecture"])
    next_route = _json(outputs["next_scientific_route"])
    reproducibility = _json(outputs["reproducibility_manifest"])
    sources = _tsv(outputs["source_manifest"])
    observations = _tsv(outputs["observation_manifest"])
    episodes = _tsv(outputs["episode_catalog"])
    actions = _tsv(outputs["action_completeness"])
    sensitivity_rows = _tsv(outputs["decision_sensitivity"])
    u0_rows = _tsv(outputs["u0_results"])
    quality_rows = _tsv(outputs["evidence_quality"])
    split_rows = {
        "train": _tsv(outputs["train_split"]),
        "validation": _tsv(outputs["validation_split"]),
        "development_eval": _tsv(outputs["development_eval_split"]),
    }

    expected_source_keys = set(policy["sources"])
    if {row["source_key"] for row in sources} != expected_source_keys:
        blockers.append("source_inventory")
    if len(quality_rows) != len(sources):
        blockers.append("source_quality_inventory")
    if any(row["evidence_class"] not in set(policy["evidence_classes"]["quantitative"]) for row in sources):
        blockers.append("non_quantitative_source_promoted")
    if any(_truthy(row["simulator_promoted_to_real_evidence"]) or _truthy(row["author_claim_promoted_to_truth"]) for row in quality_rows):
        blockers.append("invalid_real_evidence_promotion")
    if int(coverage["simulator_rows_counted_as_real"]) != 0 or int(coverage["author_claim_rows_counted_as_quantitative_truth"]) != 0:
        blockers.append("coverage_evidence_role")
    if int(coverage["quantitative_E2_episode_count"]) != len(episodes):
        blockers.append("episode_coverage_count")
    if int(coverage["same_unit_longitudinal_E4_trajectory_count"]) != 0:
        blockers.append("false_longitudinal_e4_claim")
    if not str(coverage["real_E4_status"]).endswith("not_same_unit_longitudinal"):
        blockers.append("e4_semantics")

    episode_uids = [row["episode_uid"] for row in episodes]
    observation_uids = {row["observation_uid"] for row in observations}
    action_uids = {row["world_action_uid"] for row in actions}
    if len(episode_uids) != len(set(episode_uids)) or len(episode_uids) != int(integrity["episode_count"]):
        blockers.append("episode_identity")
    if any(row["post_action_observation_uid"] not in observation_uids for row in episodes):
        blockers.append("episode_observation_join")
    if any(row["world_action_uid"] not in action_uids for row in episodes):
        blockers.append("episode_action_join")
    if any(row[field] in {"", "explicit_unavailable"} for row in episodes for field in ("publication_uid", "accession_uid", "biological_unit_uid", "baseline_state_uid", "world_action_uid", "post_action_observation_uid", "control_uid")):
        blockers.append("episode_required_identity")
    if any(_truthy(row["same_unit_longitudinal_trajectory"]) for row in episodes):
        blockers.append("episode_longitudinal_semantics")
    if any(_truthy(row["u0_supported"]) or _truthy(row["model_training_eligible"]) for row in episodes):
        blockers.append("unsupported_episode_promoted")
    if not integrity.get("cross_sectional_timecourse_semantics_explicit") or integrity.get("same_unit_longitudinal_trajectory_claimed"):
        blockers.append("episode_integrity_semantics")

    if any(not row["missing_fields"] for row in actions):
        blockers.append("action_missing_fields_not_explicit")
    if any(_truthy(row["u0_measurement_compatible"]) for row in actions):
        blockers.append("action_measurement_compatibility")
    if any(row["action_completeness"] not in {"full_structured_action", "partial_structured_action", "directional_intervention_only"} for row in actions):
        blockers.append("action_completeness_class")

    allowed_sensitivity = {
        "decision_sensitive",
        "scientifically_informative_but_not_rank_sensitive",
        "near_null",
        "ambiguous",
        "insufficient_context",
    }
    if any(row["outcome_rank_classification"] not in allowed_sensitivity for row in sensitivity_rows):
        blockers.append("decision_sensitivity_class")
    if not sensitivity.get("classification_is_based_on_real_outcomes") or sensitivity.get("model_predictions_inspected"):
        blockers.append("decision_sensitivity_provenance")
    if sensitivity.get("decision_sensitive_world_action_domain_available") or sensitivity.get("model_training_domain_qualified"):
        blockers.append("decision_domain_promotion")
    if int(sensitivity["world_action_decision_sensitive_set_count"]) != 0:
        blockers.append("world_action_decision_sensitive_count")

    split_episode_uids = {split: {row["episode_uid"] for row in rows} for split, rows in split_rows.items()}
    if any(not rows for rows in split_rows.values()):
        blockers.append("empty_development_split")
    if set().union(*split_episode_uids.values()) != set(episode_uids):
        blockers.append("split_episode_inventory")
    if any(split_episode_uids[left] & split_episode_uids[right] for left in split_episode_uids for right in split_episode_uids if left < right):
        blockers.append("split_episode_overlap")
    if leakage.get("status") != "passed" or not leakage.get("leakage_free") or not leakage.get("development_splits_frozen_before_model_results"):
        blockers.append("leakage_gate")
    if any(leakage.get("group_overlap", {}).get(field) for field in policy["split_policy"]["prohibit_overlap"]):
        blockers.append("split_group_overlap")

    if u0_gate.get("canonical_twin") != "U0_frozen" or u0_gate.get("u0_parameter_update"):
        blockers.append("u0_identity")
    if int(u0_gate["supported_episode_count"]) != 0 or int(u0_gate["numeric_prediction_count"]) != 0:
        blockers.append("u0_unsupported_prediction")
    if float(u0_gate["coverage"]) != 0.0 or float(u0_gate["abstention_rate"]) != 1.0:
        blockers.append("u0_reliability")
    if any(int(row["numeric_prediction_count"]) != 0 or float(row["coverage"]) != 0.0 or row["decision_label"] != "insufficient_evidence" for row in u0_rows):
        blockers.append("u0_result_rows")
    if any(row["endpoint_error"] != "not_computed_unsupported_domain" for row in u0_rows):
        blockers.append("u0_numeric_metric_fabrication")

    if int(training["training_attempts"]) != 0 or training["successor_lineage_created"] or training["gpu_allocated"]:
        blockers.append("unauthorized_successor_training")
    if training["status"] != "not_authorized_incompatible_world_action_domain":
        blockers.append("training_gate_status")
    if gate.get("status") not in ALLOWED_TERMINAL_STATUSES:
        blockers.append("c2b_terminal_status")
    if gate.get("status") != "blocked_insufficient_real_intervention_evidence" or not gate.get("gate_resolved"):
        blockers.append("c2b_resolution")
    if gate.get("policy_failure_claimed") or gate.get("model_failure_claimed"):
        blockers.append("unsupported_failure_claim")
    if architecture.get("canonical_twin") != "U0_frozen" or architecture.get("real_intervention_static_successor") != "not_authorized":
        blockers.append("canonical_architecture")
    if architecture.get("decision_aware_information_acquisition_headline") or architecture.get("online_twin_adaptation") or architecture.get("agent_adaptation") or architecture.get("coevolution"):
        blockers.append("closed_architecture_surface")
    if next_route.get("external_publication_baselines_resume_authorized"):
        blockers.append("publication_baseline_boundary")
    if scope.get("u0_supported_region") != "none_in_current_real_intervention_catalog" or float(scope["u0_coverage"]) != 0.0:
        blockers.append("scope_support")

    boundary_payloads = [route_freeze, claims, coverage, sensitivity, u0_gate, training, scope, gate, architecture, next_route, reproducibility]
    for payload in boundary_payloads:
        if payload.get("evidence_role") != boundary["evidence_role"] or payload.get("publication_claim_eligible") is not False or int(payload.get("protected_holdout_access", -1)) != 0:
            blockers.append(f"evidence_boundary:{payload.get('schema_version', 'unknown')}")
    if route_freeze.get("round25_execution_binding_caveat", {}).get("preserved_not_rewritten") is not True:
        blockers.append("round25_caveat")

    if int(reproducibility["episode_count"]) != len(episodes):
        blockers.append("reproducibility_episode_count")
    reproducibility_path = outputs["reproducibility_manifest"]
    for binding in reproducibility.get("source_bindings", {}).values():
        if not isinstance(binding, Mapping) or (
            not _artifact_matches(binding)
            and not _historical_artifact_matches_successor(binding)
            and not _historical_reproducibility_artifact_matches_successor(
                binding, reproducibility_path
            )
        ):
            blockers.append("reproducibility_source_binding")
            break
    for name, binding in reproducibility.get("output_bindings", {}).items():
        if name == "reproducibility_manifest":
            blockers.append("self_referential_reproducibility_binding")
            continue
        if not isinstance(binding, Mapping) or not _artifact_matches(binding):
            blockers.append(f"reproducibility_output_binding:{name}")

    if require_full_suite:
        execution_path = REPO_ROOT / "Output/audits/round27_full_suite_execution.json"
        if not execution_path.is_file():
            blockers.append("full_suite_missing")
        else:
            execution = _json(execution_path)
            if execution.get("status") != "passed" or int(execution.get("tests_failed", -1)) != 0 or int(execution.get("tests_errors", -1)) != 0:
                blockers.append("full_suite_status")
            if int(execution.get("protected_holdout_access", -1)) != 0:
                blockers.append("full_suite_holdout_boundary")

    blockers = sorted(set(blockers))
    return {
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "source_count": len(sources),
        "observation_count": len(observations),
        "episode_count": len(episodes),
        "action_count": len(actions),
        "terminal_status": gate.get("status"),
        "protected_holdout_access": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-full-suite", action="store_true")
    args = parser.parse_args()
    result = validate(require_full_suite=args.require_full_suite)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["blocker_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
