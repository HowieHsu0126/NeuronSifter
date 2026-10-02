"""Audit every real-intervention episode against the frozen U0 support contract."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


UNKNOWN_VALUES = {
    "",
    "explicit_unknown",
    "explicit_unavailable",
    "unknown",
    "unavailable",
}
SUPPORTED_NAMESPACE = "neuronsifter_microenvironment_state_grid"
DISPOSITIONS = {
    "bridgeable_without_scientific_redefinition",
    "bridgeable_with_new_model_training",
    "scientifically_incomplete",
    "irrecoverably_unsupported",
}
FAILURE_CLASSES = {
    "no_failure",
    "state_representation_gap",
    "ontology_mapping_gap",
    "unsupported_modality",
    "world_action_mapping_gap",
    "action_program_compilation_gap",
    "temporal_grid_gap",
    "missing_quantitative_endpoint",
    "missing_control",
    "missing_action_context",
    "true_model_support_gap",
    "evidence_quality_failure",
}
AXIS_FIELDS = (
    "source_eligibility",
    "biological_context",
    "species",
    "brain_region",
    "compartment",
    "state_representation",
    "modality",
    "feature_namespace",
    "world_action_semantics",
    "agent_target_mapping",
    "dose_magnitude",
    "route",
    "schedule_time",
    "trajectory_time_grid",
    "observation_endpoint",
    "control",
    "model_capability",
    "tool_capability",
)


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes repository: {value}")
    return path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_binding(path: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _is_missing(value: object) -> bool:
    return str(value).strip().lower() in UNKNOWN_VALUES


def _nonempty(value: object) -> bool:
    return not _is_missing(value)


def _deep_axis_value(payload: object, axis: str) -> object | None:
    if isinstance(payload, Mapping):
        if axis in payload and _nonempty(payload[axis]):
            return payload[axis]
        for value in payload.values():
            found = _deep_axis_value(value, axis)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _deep_axis_value(value, axis)
            if found is not None:
                return found
    return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _source_status(source: Mapping[str, str], manifest: Mapping[str, Any]) -> str:
    manifest_path = _repo_path(source["block_manifest_path"])
    if not manifest_path.is_file():
        return "missing_explicit"
    if source.get("block_manifest_sha256") != _sha256_file(manifest_path):
        return "incompatible"
    if manifest.get("admission_allowed") is not True:
        return "incompatible"
    return "supported"


def _axis_mapping_status(manifest: Mapping[str, Any], axis: str) -> str:
    value = _deep_axis_value(manifest, axis)
    return "supported" if value is not None else "missing_explicit"


def _action_status(action: Mapping[str, str], field: str) -> str:
    missing = {item for item in action.get("missing_fields", "").split("|") if item}
    if field in missing or _is_missing(action.get(field, "")):
        return "missing_explicit"
    return "supported"


def _classify_episode(
    episode: Mapping[str, str],
    action: Mapping[str, str],
    source: Mapping[str, str],
    source_manifest: Mapping[str, Any],
    observation_uids: set[str],
) -> dict[str, object]:
    source_status = _source_status(source, source_manifest)
    measurement_namespace = episode.get("measurement_namespace", "")
    state_status = "supported"
    if measurement_namespace != SUPPORTED_NAMESPACE:
        state_status = "incompatible"
    modality_status = "supported" if measurement_namespace == SUPPORTED_NAMESPACE else "incompatible"
    trajectory_uid = episode.get("trajectory_uid", "")
    if _is_missing(trajectory_uid):
        trajectory_status = "missing_explicit"
    elif trajectory_uid.startswith("explicit_unavailable"):
        trajectory_status = "endpoint_only"
    elif source.get("evidence_class") == "E4_REAL_TRAJECTORY" or source_manifest.get("trajectory_unit_semantics"):
        trajectory_status = "cross_sectional_only"
    else:
        trajectory_status = "supported"
    endpoint_status = (
        "supported"
        if _nonempty(episode.get("post_action_observation_uid"))
        and episode.get("post_action_observation_uid") in observation_uids
        and episode.get("evidence_class") in {"E2_REAL_ENDPOINT", "E4_REAL_TRAJECTORY"}
        else "missing_explicit"
    )
    control_status = "supported" if _nonempty(episode.get("control_uid")) else "missing_explicit"
    context_status = "supported" if _nonempty(episode.get("cell_or_tissue_context")) else "missing_explicit"
    species_status = _axis_mapping_status(source_manifest, "species")
    if species_status != "supported":
        species_status = _axis_mapping_status(source_manifest, "organism")
    region_status = _axis_mapping_status(source_manifest, "brain_region")
    compartment_status = _axis_mapping_status(source_manifest, "compartment")
    feature_status = "supported" if _nonempty(episode.get("measurement_namespace")) else "missing_explicit"
    target_status = _action_status(action, "target_set")
    exposure_status = _action_status(action, "exposure_program_ref")
    occupancy_status = _action_status(action, "occupancy_program_ref")
    action_semantics_status = (
        "supported"
        if target_status == exposure_status == occupancy_status == "supported"
        else "missing_explicit"
    )
    dose_status = "supported" if _nonempty(episode.get("dose_value")) and _nonempty(episode.get("dose_unit")) else "missing_explicit"
    route_value = str(episode.get("route", ""))
    route_status = "not_applicable" if "not_applicable" in route_value else ("supported" if _nonempty(route_value) else "missing_explicit")
    schedule_status = "supported" if _nonempty(episode.get("time_grid")) else "missing_explicit"
    if episode.get("schedule", "") in {"explicit_unavailable", "not_applicable_in_vitro"}:
        schedule_status = "partial_explicit"
    agent_target_status = "supported" if target_status == "supported" and _nonempty(action.get("agent_set")) else "missing_explicit"
    state_support = state_status == "supported" and species_status == region_status == compartment_status == "supported"
    model_status = "supported" if state_support and modality_status == "supported" and action_semantics_status == "supported" else "available_but_model_unsupported"
    tool_status = "supported" if model_status == "supported" and endpoint_status == control_status == "supported" else "available_but_model_unsupported"
    axes = {
        "source_eligibility": source_status,
        "biological_context": context_status,
        "species": species_status,
        "brain_region": region_status,
        "compartment": compartment_status,
        "state_representation": state_status if state_support else "incompatible",
        "modality": modality_status,
        "feature_namespace": feature_status,
        "world_action_semantics": action_semantics_status,
        "agent_target_mapping": agent_target_status,
        "dose_magnitude": dose_status,
        "route": route_status,
        "schedule_time": schedule_status,
        "trajectory_time_grid": trajectory_status,
        "observation_endpoint": endpoint_status,
        "control": control_status,
        "model_capability": model_status,
        "tool_capability": tool_status,
    }
    failures: list[tuple[str, str]] = []
    if source_status != "supported":
        failures.append(("source_eligibility", "evidence_quality_failure"))
    if context_status != "supported":
        failures.append(("biological_context", "missing_action_context"))
    if endpoint_status != "supported":
        failures.append(("observation_endpoint", "missing_quantitative_endpoint"))
    if control_status != "supported":
        failures.append(("control", "missing_control"))
    if state_status != "supported":
        failures.append(("state_representation", "state_representation_gap"))
    if species_status != "supported" or region_status != "supported" or compartment_status != "supported":
        failures.append(("ontology", "ontology_mapping_gap"))
    if modality_status != "supported":
        failures.append(("modality", "unsupported_modality"))
    if target_status != "supported":
        failures.append(("target", "world_action_mapping_gap"))
    if exposure_status != "supported" or occupancy_status != "supported":
        failures.append(("action_program", "action_program_compilation_gap"))
    if dose_status != "supported" or schedule_status not in {"supported", "not_applicable"}:
        failures.append(("dose_schedule", "action_program_compilation_gap"))
    if trajectory_status in {"missing_explicit", "cross_sectional_only"}:
        failures.append(("trajectory", "temporal_grid_gap"))
    if model_status != "supported":
        failures.append(("model", "true_model_support_gap"))
    if tool_status != "supported":
        failures.append(("tool", "action_program_compilation_gap"))
    if not failures:
        primary_failure = "no_failure"
    else:
        primary_failure = failures[0][1]
    secondary = []
    for _, failure_class in failures[1:]:
        if failure_class != primary_failure and failure_class not in secondary:
            secondary.append(failure_class)
    irrecoverable = source_status != "supported" or episode.get("evidence_class") not in {"E2_REAL_ENDPOINT", "E4_REAL_TRAJECTORY"}
    has_scientific_context = context_status == "supported" and endpoint_status == control_status == "supported" and dose_status == "supported" and schedule_status in {"supported", "partial_explicit", "not_applicable"}
    if irrecoverable:
        disposition = "irrecoverably_unsupported"
    elif not has_scientific_context:
        disposition = "scientifically_incomplete"
    elif state_support and modality_status == "supported":
        disposition = "bridgeable_without_scientific_redefinition"
    else:
        disposition = "bridgeable_with_new_model_training"
    return {
        "episode_uid": episode["episode_uid"],
        "source_key": episode.get("source_collection_uid", "").split(":", 1)[0],
        "accession_uid": episode.get("accession_uid", ""),
        "world_action_uid": episode.get("world_action_uid", ""),
        "biological_unit_uid": episode.get("biological_unit_uid", ""),
        "primary_failure_class": primary_failure,
        "secondary_failure_classes": "|".join(secondary) if secondary else "none",
        "support_disposition": disposition,
        "u0_executable_support": "supported" if tool_status == "supported" else "unsupported",
        **axes,
    }


def _fields() -> list[str]:
    return [
        "episode_uid",
        "source_key",
        "accession_uid",
        "world_action_uid",
        "biological_unit_uid",
        "primary_failure_class",
        "secondary_failure_classes",
        "support_disposition",
        "u0_executable_support",
        *AXIS_FIELDS,
    ]


def audit() -> tuple[list[dict[str, object]], dict[str, object]]:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    policy = config["real_intervention_evidence"]
    contract = policy["support_audit"]
    input_paths = {name: _repo_path(str(value)) for name, value in contract["inputs"].items()}
    source_inventory_rows = _read_tsv(input_paths["source_manifest"])
    observation_rows = _read_tsv(input_paths["observation_manifest"])
    episode_rows = _read_tsv(input_paths["episode_catalog"])
    action_rows = _read_tsv(input_paths["action_completeness"])
    source_by_key = {row["source_key"]: row for row in source_inventory_rows}
    action_by_uid = {row["world_action_uid"]: row for row in action_rows}
    observation_uids = {row["observation_uid"] for row in observation_rows}
    if len(source_by_key) != len(source_inventory_rows) or len(action_by_uid) != len(action_rows):
        raise ValueError("Support audit inputs contain duplicate source or action identities")
    rows: list[dict[str, object]] = []
    for episode in episode_rows:
        action = action_by_uid.get(episode["world_action_uid"])
        source_key = next((key for key, value in source_by_key.items() if value["source_accession"] == episode["accession_uid"]), None)
        if action is None or source_key is None:
            raise ValueError(f"Episode is not joinable to source/action: {episode['episode_uid']}")
        source = source_by_key[source_key]
        manifest_path = _repo_path(source["block_manifest_path"])
        source_manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
        row = _classify_episode(episode, action, source, source_manifest, observation_uids)
        row["source_key"] = source_key
        rows.append(row)
    if len(rows) != len(episode_rows):
        raise ValueError("Support audit row count changed during classification")
    primary_counts = Counter(str(row["primary_failure_class"]) for row in rows)
    disposition_counts = Counter(str(row["support_disposition"]) for row in rows)
    source_summary: dict[str, object] = {}
    for source_key in sorted({str(row["source_key"]) for row in rows}):
        source_episode_rows = [row for row in rows if row["source_key"] == source_key]
        source_summary[source_key] = {
            "episode_count": len(source_episode_rows),
            "u0_executable_support_count": sum(row["u0_executable_support"] == "supported" for row in source_episode_rows),
            "primary_failure_class_counts": dict(sorted(Counter(str(row["primary_failure_class"]) for row in source_episode_rows).items())),
            "support_disposition_counts": dict(sorted(Counter(str(row["support_disposition"]) for row in source_episode_rows).items())),
        }
    axis_counts = {
        axis: dict(Counter(str(row[axis]) for row in rows))
        for axis in AXIS_FIELDS
    }
    invalid_primary = sorted(set(primary_counts) - FAILURE_CLASSES)
    invalid_disposition = sorted(set(disposition_counts) - DISPOSITIONS)
    if invalid_primary or invalid_disposition:
        raise ValueError(f"Support audit classification contract failed: {invalid_primary}, {invalid_disposition}")
    summary: dict[str, object] = {
        "schema_version": "neuronsifter-real-intervention-support-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "resolved",
        "episode_count": len(rows),
        "source_count": len(source_inventory_rows),
        "u0_executable_support_count": sum(row["u0_executable_support"] == "supported" for row in rows),
        "unknown_primary_failure_count": 0,
        "unknown_disposition_count": 0,
        "all_episode_support_dispositions_known": True,
        "primary_failure_class_counts": dict(sorted(primary_counts.items())),
        "support_disposition_counts": dict(sorted(disposition_counts.items())),
        "source_summary": source_summary,
        "axis_status_counts": axis_counts,
        "bridgeable_without_scientific_redefinition": disposition_counts.get("bridgeable_without_scientific_redefinition", 0),
        "bridgeable_with_new_model_training": disposition_counts.get("bridgeable_with_new_model_training", 0),
        "scientifically_incomplete": disposition_counts.get("scientifically_incomplete", 0),
        "irrecoverably_unsupported": disposition_counts.get("irrecoverably_unsupported", 0),
        "input_bindings": {name: _input_binding(path) for name, path in input_paths.items()},
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
        "model_or_policy_failure_claimed": False,
        "interpretation": "Support failure is decomposed into explicit representation, action, temporal, evidence, model, and tool axes; no episode is promoted to U0 support.",
    }
    return rows, summary


def write_outputs(rows: list[dict[str, object]], summary: dict[str, object], output_paths: Mapping[str, str]) -> None:
    matrix_path = _repo_path(str(output_paths["support_matrix"]))
    root_cause_path = _repo_path(str(output_paths["root_cause"]))
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    root_cause_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fields(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    root_cause_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-matrix", type=Path)
    parser.add_argument("--root-cause", type=Path)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    output_paths = dict(config["real_intervention_evidence"]["support_audit"]["outputs"])
    if args.support_matrix is not None:
        output_paths["support_matrix"] = args.support_matrix.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    if args.root_cause is not None:
        output_paths["root_cause"] = args.root_cause.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    rows, summary = audit()
    write_outputs(rows, summary, output_paths)
    print(json.dumps({key: summary[key] for key in ("status", "episode_count", "u0_executable_support_count", "support_disposition_counts")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
