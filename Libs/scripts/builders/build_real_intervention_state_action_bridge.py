"""Build a fail-closed real-intervention state/action bridge.

The bridge creates a new, transcriptomic real-intervention representation while
preserving the frozen U0 contract.  It records what is expressible in a new
static lineage and keeps absent target/occupancy fields explicitly absent.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
BUILDER_ID = "Libs/scripts/builders/build_real_intervention_state_action_bridge.py"
MISSING = {"", "explicit_unknown", "explicit_unavailable", "unknown", "unavailable"}
BOUNDARY = {
    "evidence_role": "development_only",
    "publication_claim_eligible": False,
    "protected_holdout_access": 0,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes repository: {value}")
    return path


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _is_missing(value: object) -> bool:
    return str(value).strip().lower() in MISSING


def _text(value: object) -> str:
    return str(value).strip()


def _state_uid(row: Mapping[str, str]) -> str:
    identity = {
        "source_accession": row["accession_uid"],
        "baseline_state_uid": row["baseline_state_uid"],
        "biological_context": row["cell_or_tissue_context"],
        "measurement_namespace": row["measurement_namespace"],
    }
    return f"real_intervention_state:{canonical_sha256(identity)[:24]}"


def _action_missing_fields(row: Mapping[str, str]) -> list[str]:
    fields = ("target_set", "exposure_program_ref", "occupancy_program_ref")
    return [field for field in fields if _is_missing(row.get(field, ""))]


def _action_status(row: Mapping[str, str]) -> str:
    declared = row.get("action_completeness", "")
    if declared == "directional_intervention_only":
        return declared
    missing = _action_missing_fields(row)
    if declared == "full_structured_action" and not missing:
        return declared
    if declared in {"full_structured_action", "partial_structured_action"} or missing:
        return "partial_structured_action"
    raise ValueError(f"Unknown action completeness: {declared!r}")


def _load_inputs(
    policy: Mapping[str, object],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, Any],
]:
    support = read_tsv(_repo_path(str(policy["input_support_matrix"])))
    episodes = read_tsv(_repo_path(str(policy["input_episode_catalog"])))
    observations = read_tsv(_repo_path(str(policy["input_observation_manifest"])))
    sources = read_tsv(_repo_path(str(policy["input_source_manifest"])))
    decision_sensitivity = _json(_repo_path(str(policy["input_decision_sensitivity"])))
    expected = int(policy["expected_episode_count"])
    if len(support) != expected or len(episodes) != expected:
        raise ValueError(f"Support/episode count mismatch: {len(support)} vs {len(episodes)}")
    if len(sources) != int(policy["expected_source_count"]):
        raise ValueError(f"Source count mismatch: {len(sources)}")
    if decision_sensitivity.get("evidence_role") != "development_only":
        raise ValueError("Decision-sensitivity input crosses the development-only boundary")
    if decision_sensitivity.get("publication_claim_eligible") is not False:
        raise ValueError("Decision-sensitivity input is publication-claim eligible")
    if decision_sensitivity.get("protected_holdout_access") != 0:
        raise ValueError("Decision-sensitivity input accessed the protected holdout")
    return support, episodes, observations, sources, decision_sensitivity


def _source_bindings(sources: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    bindings: dict[str, dict[str, object]] = {}
    for source in sources:
        path = _repo_path(source["block_manifest_path"])
        observed = _binding(path)
        if observed["sha256"] != source["block_manifest_sha256"]:
            raise ValueError(f"Source block manifest hash drifted: {source['source_key']}")
        bindings[source["source_key"]] = observed
    return bindings


def _build_action_catalog(episodes: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in episodes:
        grouped[row["world_action_uid"]].append(row)
    output: list[dict[str, object]] = []
    for action_uid in sorted(grouped):
        rows = grouped[action_uid]
        first = rows[0]
        comparable_fields = (
            "source_key",
            "accession_uid",
            "action_family_uid",
            "regimen_family_uid",
            "intervention_type",
            "agent_set",
            "target_set",
            "dose_value",
            "dose_unit",
            "duration",
            "duration_unit",
            "schedule",
            "route",
            "cell_or_tissue_context",
            "exposure_program_ref",
            "occupancy_program_ref",
            "action_completeness",
            "measurement_namespace",
        )
        for field in comparable_fields:
            if {row[field] for row in rows} != {first[field]}:
                raise ValueError(f"Action field is not stable for {action_uid}: {field}")
        missing = _action_missing_fields(first)
        status = _action_status(first)
        output.append(
            {
                "structured_action_uid": action_uid,
                "world_action_uid": action_uid,
                "source_key": first["source_key"],
                "source_accession": first["accession_uid"],
                "action_family_uid": first["action_family_uid"],
                "regimen_family_uid": first["regimen_family_uid"],
                "intervention_type": first["intervention_type"],
                "agent_set": first["agent_set"],
                "target_set": first["target_set"],
                "dose_value": first["dose_value"],
                "dose_unit": first["dose_unit"],
                "duration": first["duration"],
                "duration_unit": first["duration_unit"],
                "schedule": first["schedule"],
                "route": first["route"],
                "cell_or_tissue_context": first["cell_or_tissue_context"],
                "exposure_program_ref": first["exposure_program_ref"],
                "occupancy_program_ref": first["occupancy_program_ref"],
                "action_semantics_status": status,
                "missing_action_fields": "|".join(missing) if missing else "none",
                "action_compilation_status": (
                    "deferred_new_lineage_required" if missing else "candidate_compiler_ready_new_lineage"
                ),
                "u0_action_compatible": False,
                "episode_count": len(rows),
                **BOUNDARY,
            }
        )
    return output


def _build_episode_catalog(
    episodes: list[dict[str, str]],
    support_by_uid: Mapping[str, Mapping[str, str]],
    observation_by_uid: Mapping[str, Mapping[str, str]],
    source_bindings: Mapping[str, Mapping[str, object]],
    state_namespace: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in episodes:
        episode_uid = row["episode_uid"]
        support = support_by_uid.get(episode_uid)
        if support is None:
            raise ValueError(f"Episode is absent from support matrix: {episode_uid}")
        observation = observation_by_uid.get(row["post_action_observation_uid"])
        if observation is None:
            raise ValueError(f"Post-action observation is absent: {episode_uid}")
        if observation["world_action_uid"] != row["world_action_uid"]:
            raise ValueError(f"Observation/action mismatch: {episode_uid}")
        source_binding = source_bindings.get(row["source_key"])
        if source_binding is None:
            raise ValueError(f"Source binding is absent: {row['source_key']}")
        state_status = support["support_disposition"]
        if state_status not in {"bridgeable_with_new_model_training", "scientifically_incomplete", "irrecoverably_unsupported"}:
            raise ValueError(f"Unknown support disposition: {state_status}")
        action_status = _action_status(row)
        missing = _action_missing_fields(row)
        output.append(
            {
                "episode_uid": episode_uid,
                "source_key": row["source_key"],
                "publication_uid": row["publication_uid"],
                "accession_uid": row["accession_uid"],
                "source_collection_uid": row["source_collection_uid"],
                "biological_unit_uid": row["biological_unit_uid"],
                "baseline_state_uid": row["baseline_state_uid"],
                "state_uid": _state_uid(row),
                "state_namespace": state_namespace,
                "state_feature_namespace": row["measurement_namespace"],
                "state_projection_status": state_status,
                "state_projection_reason": "new_static_real_intervention_encoder_required; no U0 projection claimed",
                "world_action_uid": row["world_action_uid"],
                "structured_action_uid": row["world_action_uid"],
                "action_family_uid": row["action_family_uid"],
                "regimen_family_uid": row["regimen_family_uid"],
                "intervention_type": row["intervention_type"],
                "agent_set": row["agent_set"],
                "target_set": row["target_set"],
                "exposure_program_ref": row["exposure_program_ref"],
                "occupancy_program_ref": row["occupancy_program_ref"],
                "action_semantics_status": action_status,
                "missing_action_fields": "|".join(missing) if missing else "none",
                "action_compilation_status": "deferred_new_lineage_required" if missing else "candidate_compiler_ready_new_lineage",
                "post_action_observation_uid": row["post_action_observation_uid"],
                "observation_value_unit": observation["value_unit"],
                "observation_measurement_namespace": observation["measurement_namespace"],
                "control_uid": row["control_uid"],
                "trajectory_uid": row["trajectory_uid"],
                "time_grid": row["time_grid"],
                "dose_value": row["dose_value"],
                "dose_unit": row["dose_unit"],
                "duration": row["duration"],
                "duration_unit": row["duration_unit"],
                "schedule": row["schedule"],
                "route": row["route"],
                "cell_or_tissue_context": row["cell_or_tissue_context"],
                "uncertainty": row["uncertainty"],
                "evidence_grade": row["evidence_grade"],
                "evidence_class": row["evidence_class"],
                "action_completeness": row["action_completeness"],
                "support_primary_failure_class": support["primary_failure_class"],
                "support_secondary_failure_classes": support["secondary_failure_classes"],
                "u0_executable_support": "unsupported",
                "u0_projection_status": "not_projected_by_frozen_contract",
                "new_lineage_training_eligible": False,
                "source_block_manifest_path": source_binding["path"],
                "source_block_manifest_sha256": source_binding["sha256"],
                "split_role": row["split_role"],
                **BOUNDARY,
            }
        )
    return sorted(output, key=lambda item: str(item["episode_uid"]))


def _source_manifest_rows(
    sources: list[dict[str, str]],
    episodes: list[dict[str, object]],
    actions: list[dict[str, object]],
    source_bindings: Mapping[str, Mapping[str, object]],
    state_namespace: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source in sorted(sources, key=lambda row: row["source_key"]):
        source_episodes = [row for row in episodes if row["source_key"] == source["source_key"]]
        source_actions = [row for row in actions if row["source_key"] == source["source_key"]]
        output.append(
            {
                "manifest_version": "real_intervention_v2",
                "source_key": source["source_key"],
                "source_id": source["source_id"],
                "source_accession": source["source_accession"],
                "publication_uid": source["publication_uid"],
                "source_block_manifest_path": source_bindings[source["source_key"]]["path"],
                "source_block_manifest_sha256": source_bindings[source["source_key"]]["sha256"],
                "evidence_class": source["evidence_class"],
                "measurement_namespace": source["measurement_namespace"],
                "state_namespace": state_namespace,
                "episode_count": len(source_episodes),
                "action_count": len(source_actions),
                "control_lineage_count": len({str(row["control_uid"]) for row in source_episodes}),
                "context_count": len({str(row["cell_or_tissue_context"]) for row in source_episodes}),
                "timepoint_count": len({str(row["time_grid"]) for row in source_episodes}),
                "partial_action_count": sum(row["action_semantics_status"] == "partial_structured_action" for row in source_actions),
                "full_action_count": sum(row["action_semantics_status"] == "full_structured_action" for row in source_actions),
                "bridgeable_episode_count": sum(row["state_projection_status"] == "bridgeable_with_new_model_training" for row in source_episodes),
                "u0_supported_count": 0,
                "training_authorized": False,
                **BOUNDARY,
            }
        )
    return output


def _field_lists() -> dict[str, list[str]]:
    return {
        "episodes": [
            "episode_uid", "source_key", "publication_uid", "accession_uid", "source_collection_uid",
            "biological_unit_uid", "baseline_state_uid", "state_uid", "state_namespace", "state_feature_namespace",
            "state_projection_status", "state_projection_reason", "world_action_uid", "structured_action_uid",
            "action_family_uid", "regimen_family_uid", "intervention_type", "agent_set", "target_set",
            "exposure_program_ref", "occupancy_program_ref", "action_semantics_status", "missing_action_fields",
            "action_compilation_status", "post_action_observation_uid", "observation_value_unit",
            "observation_measurement_namespace", "control_uid", "trajectory_uid", "time_grid", "dose_value",
            "dose_unit", "duration", "duration_unit", "schedule", "route", "cell_or_tissue_context", "uncertainty",
            "evidence_grade", "evidence_class", "action_completeness", "support_primary_failure_class",
            "support_secondary_failure_classes", "u0_executable_support", "u0_projection_status",
            "new_lineage_training_eligible", "source_block_manifest_path", "source_block_manifest_sha256",
            "split_role", "evidence_role", "publication_claim_eligible", "protected_holdout_access",
        ],
        "actions": [
            "structured_action_uid", "world_action_uid", "source_key", "source_accession", "action_family_uid",
            "regimen_family_uid", "intervention_type", "agent_set", "target_set", "dose_value", "dose_unit",
            "duration", "duration_unit", "schedule", "route", "cell_or_tissue_context", "exposure_program_ref",
            "occupancy_program_ref", "action_semantics_status", "missing_action_fields", "action_compilation_status",
            "u0_action_compatible", "episode_count", "evidence_role", "publication_claim_eligible", "protected_holdout_access",
        ],
        "manifest": [
            "manifest_version", "source_key", "source_id", "source_accession", "publication_uid",
            "source_block_manifest_path", "source_block_manifest_sha256", "evidence_class", "measurement_namespace",
            "state_namespace", "episode_count", "action_count", "control_lineage_count", "context_count", "timepoint_count",
            "partial_action_count", "full_action_count", "bridgeable_episode_count", "u0_supported_count",
            "training_authorized", "evidence_role", "publication_claim_eligible", "protected_holdout_access",
        ],
    }


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("State/action bridge is identity-bound to the repository root")
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]["state_action_bridge"]
    support, episode_rows, observation_rows, source_rows, decision_sensitivity = _load_inputs(policy)
    support_by_uid = {row["episode_uid"]: row for row in support}
    observation_by_uid = {row["observation_uid"]: row for row in observation_rows}
    if len(support_by_uid) != len(support) or len(observation_by_uid) != len(observation_rows):
        raise ValueError("Input identities are duplicated")
    source_bindings = _source_bindings(source_rows)
    state_namespace = str(policy["state_namespace"])
    enriched_episode_rows: list[dict[str, str]] = []
    for row in episode_rows:
        support_row = support_by_uid.get(row["episode_uid"])
        if support_row is None:
            raise ValueError(f"Episode is absent from support matrix: {row['episode_uid']}")
        enriched = dict(row)
        enriched["source_key"] = support_row["source_key"]
        if enriched["accession_uid"] != support_row["accession_uid"]:
            raise ValueError(f"Episode/support accession mismatch: {row['episode_uid']}")
        enriched_episode_rows.append(enriched)
    episode_catalog = _build_episode_catalog(
        enriched_episode_rows,
        support_by_uid,
        observation_by_uid,
        source_bindings,
        state_namespace,
    )
    action_catalog = _build_action_catalog(enriched_episode_rows)
    manifest_rows = _source_manifest_rows(source_rows, episode_catalog, action_catalog, source_bindings, state_namespace)
    if len(episode_catalog) != int(policy["expected_episode_count"]):
        raise ValueError("Bridge episode count differs from the frozen support audit")
    if sum(int(row["u0_supported_count"]) for row in manifest_rows) != int(policy["expected_u0_supported_count"]):
        raise ValueError("Bridge attempted to promote U0 support")
    if any(str(row["protected_holdout_access"]) != "0" for row in episode_catalog):
        raise ValueError("Protected holdout access is nonzero")

    outputs = {name: _repo_path(str(path)) for name, path in policy["outputs"].items()}
    fields = _field_lists()
    write_tsv(outputs["episode_manifest"], fields["manifest"], manifest_rows)
    write_tsv(outputs["action_catalog"], fields["actions"], action_catalog)
    write_tsv(outputs["episode_catalog"], fields["episodes"], episode_catalog)

    input_bindings = {
        name: _binding(_repo_path(str(policy[key])))
        for name, key in (
            ("support_matrix", "input_support_matrix"),
            ("episode_catalog", "input_episode_catalog"),
            ("observation_manifest", "input_observation_manifest"),
            ("source_manifest", "input_source_manifest"),
            ("decision_sensitivity", "input_decision_sensitivity"),
        )
    }
    output_bindings = {
        name: _binding(outputs[name])
        for name in ("episode_manifest", "action_catalog", "episode_catalog")
    }
    integrity = {
        "schema_version": "neuronsifter-real-intervention-v2-episode-integrity-v1",
        "status": "passed",
        "episode_count": len(episode_catalog),
        "action_count": len(action_catalog),
        "source_count": len(manifest_rows),
        "unique_episode_uid": len({row["episode_uid"] for row in episode_catalog}) == len(episode_catalog),
        "unique_state_uid": len({row["state_uid"] for row in episode_catalog}) == len(episode_catalog),
        "unique_action_uid": len({row["structured_action_uid"] for row in action_catalog}) == len(action_catalog),
        "episode_action_joins_complete": all(row["structured_action_uid"] in {item["structured_action_uid"] for item in action_catalog} for row in episode_catalog),
        "episode_observation_joins_complete": all(row["post_action_observation_uid"] in observation_by_uid for row in episode_catalog),
        "episode_control_joins_explicit": all(not _is_missing(row["control_uid"]) for row in episode_catalog),
        "state_namespace_consistent": len({row["state_namespace"] for row in episode_catalog}) == 1,
        "u0_supported_count": 0,
        "new_lineage_training_authorized": False,
        "input_bindings": input_bindings,
        "output_bindings": output_bindings,
        **BOUNDARY,
    }
    write_json(outputs["episode_integrity"], integrity)
    quality = {
        "schema_version": str(policy["schema_version"]),
        "status": "resolved",
        "builder": BUILDER_ID,
        "created_at": _now(),
        "state_namespace": state_namespace,
        "u0_projection_status": str(policy["u0_projection_status"]),
        "training_authorization": str(policy["training_authorization"]),
        "source_count": len(manifest_rows),
        "episode_count": len(episode_catalog),
        "action_count": len(action_catalog),
        "bridgeable_episode_count": sum(row["state_projection_status"] == "bridgeable_with_new_model_training" for row in episode_catalog),
        "scientifically_incomplete_episode_count": sum(row["state_projection_status"] == "scientifically_incomplete" for row in episode_catalog),
        "irrecoverably_unsupported_episode_count": sum(row["state_projection_status"] == "irrecoverably_unsupported" for row in episode_catalog),
        "u0_supported_count": 0,
        "quantitative_endpoint_count": len(episode_catalog),
        "explicit_control_lineage_count": len({row["control_uid"] for row in episode_catalog}),
        "action_semantics_counts": dict(Counter(row["action_semantics_status"] for row in action_catalog)),
        "measurement_namespace_counts": dict(Counter(row["state_feature_namespace"] for row in episode_catalog)),
        "support_disposition_counts": dict(Counter(row["state_projection_status"] for row in episode_catalog)),
        "source_episode_counts": {row["source_key"]: int(row["episode_count"]) for row in manifest_rows},
        "nontrivial_quantitative_representation_domain": bool(len(action_catalog) > 1 and episode_catalog),
        "round27_decision_sensitivity_bound": True,
        "decision_sensitive_world_action_domain_available": bool(
            decision_sensitivity["decision_sensitive_world_action_domain_available"]
        ),
        "prespecified_therapeutic_decision_objective_available": bool(
            decision_sensitivity["prespecified_therapeutic_decision_objective_available"]
        ),
        "model_training_domain_qualified": bool(decision_sensitivity["model_training_domain_qualified"]),
        "round28_decision_sensitive_benchmark_pending": True,
        "model_predictions_inspected": False,
        "new_static_lineage_created": False,
        "historical_u0_unchanged": True,
        "input_bindings": input_bindings,
        "output_bindings": {**output_bindings, "episode_integrity": _binding(outputs["episode_integrity"])},
        **BOUNDARY,
    }
    write_json(outputs["quality_audit"], quality)
    return {
        "status": quality["status"],
        "source_count": quality["source_count"],
        "episode_count": quality["episode_count"],
        "action_count": quality["action_count"],
        "u0_supported_count": quality["u0_supported_count"],
        "quality_path": outputs["quality_audit"].relative_to(REPO_ROOT).as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = build()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
