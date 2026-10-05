"""Resolve the real-intervention evidence and frozen World-Action development gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402
from Libs.utils.evaluation.real_intervention_evidence import (  # noqa: E402
    candidate_set_diagnostics,
    overlap_values,
    rms,
    single_action_effect,
    transformed_expression,
)
from Libs.utils.model.dynamics_store import load_zarr_full_array  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
BUILDER_ID = "Libs/scripts/experiments/resolve_real_intervention_world_action.py"
UNKNOWN = "explicit_unavailable"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _clean(value: object, default: str = UNKNOWN) -> object:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    return value if text else default


def _text(value: object, default: str = UNKNOWN) -> str:
    cleaned = _clean(value, default)
    return str(cleaned)


def _numeric(value: object) -> float | str:
    cleaned = _clean(value)
    if cleaned == UNKNOWN:
        return UNKNOWN
    result = float(cleaned)
    return result if math.isfinite(result) else UNKNOWN


def _write_table(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    write_tsv(path, fields, rows)


@dataclass(frozen=True)
class EvidenceBlock:
    source_key: str
    contract: Mapping[str, object]
    manifest_path: Path
    manifest: Mapping[str, object]
    root: Path
    observations: pd.DataFrame
    actions: pd.DataFrame
    controls: pd.DataFrame
    control_block_manifest_path: Path | None = None
    control_block_root: Path | None = None
    control_observations: pd.DataFrame | None = None


def _bound_manifest_from_audit(audit_path: Path) -> Path:
    audit = _json(audit_path)
    binding = audit.get("canonical_block_manifest")
    if not isinstance(binding, Mapping):
        raise ValueError(f"Block audit lacks canonical binding: {audit_path}")
    path = _repo_path(str(binding["path"]))
    observed = _artifact(path)
    if observed["bytes"] != int(binding["bytes"]) or observed["sha256"] != binding["sha256"]:
        raise ValueError(f"Canonical block binding drifted: {path}")
    return path


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _load_blocks(policy: Mapping[str, object]) -> dict[str, EvidenceBlock]:
    result: dict[str, EvidenceBlock] = {}
    sources = policy["sources"]
    assert isinstance(sources, Mapping)
    for source_key, raw_contract in sources.items():
        if not isinstance(raw_contract, Mapping):
            raise TypeError(f"Source contract must be a mapping: {source_key}")
        if "block_manifest_path" in raw_contract:
            manifest_path = _repo_path(str(raw_contract["block_manifest_path"]))
        else:
            manifest_path = _bound_manifest_from_audit(_repo_path(str(raw_contract["block_manifest_audit_path"])))
        manifest = _json(manifest_path)
        if manifest.get("source_id") != raw_contract["source_id"] or manifest.get("source_accession") != raw_contract["source_accession"]:
            raise ValueError(f"Source identity differs from the block manifest: {source_key}")
        root = manifest_path.parent
        observations = pd.read_parquet(root / str(manifest["sample_metadata_path"]))
        actions = _read_optional_parquet(root / str(manifest.get("action_catalog_path", "")))
        controls = _read_optional_parquet(root / str(manifest.get("control_metadata_path", "")))
        control_manifest_path: Path | None = None
        control_root: Path | None = None
        control_observations: pd.DataFrame | None = None
        if "control_block_manifest_path" in raw_contract:
            control_manifest_path = _repo_path(str(raw_contract["control_block_manifest_path"]))
            control_manifest = _json(control_manifest_path)
            if control_manifest.get("source_accession") != raw_contract["source_accession"]:
                raise ValueError(f"Control block accession differs from treatment block: {source_key}")
            control_root = control_manifest_path.parent
            control_observations = pd.read_parquet(control_root / str(control_manifest["sample_metadata_path"]))
        result[str(source_key)] = EvidenceBlock(
            source_key=str(source_key),
            contract=raw_contract,
            manifest_path=manifest_path,
            manifest=manifest,
            root=root,
            observations=observations,
            actions=actions,
            controls=controls,
            control_block_manifest_path=control_manifest_path,
            control_block_root=control_root,
            control_observations=control_observations,
        )
    return result


def _measurement_namespace(block: EvidenceBlock) -> str:
    direct = block.manifest.get("measurement_namespace")
    if direct:
        return str(direct)
    join = block.manifest.get("observation_value_join_contract", {})
    if isinstance(join, Mapping) and join.get("feature_namespace"):
        return str(join["feature_namespace"])
    if "feature_namespace" in block.observations.columns:
        values = sorted(set(block.observations["feature_namespace"].dropna().astype(str)))
        return "|".join(values) if values else UNKNOWN
    return UNKNOWN


def _value_unit(block: EvidenceBlock) -> str:
    direct = block.manifest.get("value_unit")
    if direct:
        return str(direct)
    join = block.manifest.get("observation_value_join_contract", {})
    if isinstance(join, Mapping) and join.get("value_unit"):
        return str(join["value_unit"])
    if "value_unit" in block.observations.columns:
        values = sorted(set(block.observations["value_unit"].dropna().astype(str)))
        return "|".join(values) if values else UNKNOWN
    return UNKNOWN


def _standard_observation(
    block: EvidenceBlock,
    row: Mapping[str, object],
    *,
    role: str,
    manifest_path: Path,
) -> dict[str, object]:
    source_key = block.source_key
    evidence_class = str(block.contract["evidence_class"])
    observation_uid = _text(row.get("observation_uid") or row.get("control_uid"))
    action_uid = _text(row.get("world_action_uid") or row.get("action_uid"))
    condition_uid = _text(row.get("condition_uid") or row.get("compound_uid"), "vehicle_control" if role == "matched_control" else UNKNOWN)
    if role == "matched_control":
        action_uid = f"no_action:{source_key}:{_text(row.get('control_group_uid'))}"
        action_family_uid = f"no_action:{source_key}"
    else:
        action_family_uid = _text(row.get("action_family_uid") or row.get("compound_uid") or action_uid)
    trajectory_uid = _text(row.get("trajectory_uid"), "explicit_unavailable_static_endpoint")
    if source_key != "amyloid_peptide_timecourse":
        trajectory_uid = "explicit_unavailable_static_endpoint"
    uncertainty = _text(row.get("uncertainty_semantics"))
    if uncertainty == UNKNOWN:
        if source_key == "clue_npc_neu":
            uncertainty = f"Level5_consensus_from_{_text(row.get('source_instance_count'), 'unknown')}_source_instances"
        elif source_key == "dexamethasone_neurons":
            uncertainty = "condition_aggregate_without_replicate_level_values"
        else:
            uncertainty = "biological_replicate_group_available"
    return {
        "observation_uid": observation_uid,
        "source_key": source_key,
        "source_id": block.contract["source_id"],
        "source_accession": block.contract["source_accession"],
        "source_collection_uid": _text(row.get("source_collection_uid"), f"{block.contract['source_id']}:{block.contract['source_accession']}"),
        "publication_uid": _text(row.get("publication_uid"), _text(block.manifest.get("publication_uid") or block.manifest.get("publication_accession_group"))),
        "accession_uid": _text(row.get("accession_uid"), str(block.contract["source_accession"])),
        "biological_unit_uid": _text(row.get("biological_unit_uid")),
        "sample_uid": _text(row.get("sample_uid"), observation_uid),
        "cell_or_tissue_context": _text(row.get("cell_or_tissue_context")),
        "condition_uid": condition_uid,
        "observation_role": role,
        "world_action_uid": action_uid,
        "action_family_uid": action_family_uid,
        "regimen_family_uid": _text(row.get("regimen_family_uid"), action_uid),
        "trajectory_uid": trajectory_uid,
        "control_group_uid": _text(row.get("control_group_uid")),
        "time_value": _numeric(row.get("time_value")),
        "time_unit": _text(row.get("time_unit")),
        "dose_value": _numeric(row.get("dose_value")),
        "dose_unit": _text(row.get("dose_unit")),
        "measurement_namespace": _measurement_namespace(block),
        "value_unit": _value_unit(block),
        "uncertainty": uncertainty,
        "evidence_class": str(block.contract.get("endpoint_evidence_class", evidence_class)) if role != "matched_control" else "E2_REAL_ENDPOINT",
        "trajectory_evidence_class": evidence_class if evidence_class == "E4_REAL_TRAJECTORY" else "not_applicable",
        "evidence_grade": _text(block.manifest.get("qc_grade"), _text(block.contract.get("evidence_grade"))),
        "split_role": block.contract["split_role"],
        "source_value_row_index": int(row.get("value_row_index", -1)),
        "source_block_manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }


def _standard_observations(blocks: Mapping[str, EvidenceBlock]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source_key, block in blocks.items():
        for row in block.observations.to_dict(orient="records"):
            if source_key == "amyloid_peptide_timecourse":
                role = str(row["observation_role"])
            elif source_key == "dexamethasone_neurons":
                role = "intervention_endpoint" if _text(row.get("action_uid")) != UNKNOWN else "matched_control"
            else:
                role = "intervention_endpoint"
            output.append(_standard_observation(block, row, role=role, manifest_path=block.manifest_path))
        if block.control_observations is not None and block.control_block_manifest_path is not None:
            for row in block.control_observations.to_dict(orient="records"):
                output.append(
                    _standard_observation(
                        block,
                        row,
                        role="matched_control",
                        manifest_path=block.control_block_manifest_path,
                    )
                )
    identities = [str(row["observation_uid"]) for row in output]
    if len(identities) != len(set(identities)) or UNKNOWN in identities:
        raise ValueError("Real-intervention observation identities are missing or duplicated")
    return sorted(output, key=lambda row: (str(row["source_key"]), str(row["observation_uid"])))


def _missing_action_fields(row: Mapping[str, object]) -> list[str]:
    aliases = {
        "target_set": ("target_set", "target_set_or_explicit_unknown"),
        "exposure_program_ref": ("exposure_program_ref", "exposure_program_ref_or_explicit_unavailable"),
        "occupancy_program_ref": ("occupancy_program_ref", "occupancy_program_ref_or_explicit_unavailable"),
    }
    missing: list[str] = []
    for field, candidates in aliases.items():
        value = next((_text(row.get(candidate)) for candidate in candidates if _text(row.get(candidate)) != UNKNOWN), UNKNOWN)
        if value == UNKNOWN or "unknown" in value or "unavailable" in value:
            missing.append(field)
    return missing


def _action_audit(blocks: Mapping[str, EvidenceBlock]) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lookup: dict[str, dict[str, object]] = {}
    for source_key, block in blocks.items():
        for action in block.actions.to_dict(orient="records"):
            action_uid = _text(action.get("world_action_uid") or action.get("action_uid"))
            if action_uid == UNKNOWN:
                raise ValueError(f"Action identity is missing for source {source_key}")
            missing = _missing_action_fields(action)
            row = {
                "world_action_uid": action_uid,
                "source_key": source_key,
                "source_accession": block.contract["source_accession"],
                "publication_uid": _text(action.get("publication_uid") or action.get("source_publication")),
                "action_family_uid": _text(action.get("action_family_uid") or action.get("agent_uid") or action.get("agent_set"), action_uid),
                "regimen_family_uid": _text(action.get("regimen_family_uid"), action_uid),
                "intervention_type": _text(action.get("intervention_type")),
                "agent_set": _text(action.get("agent_set") or action.get("agent_uid")),
                "target_set": _text(action.get("target_set") or action.get("target_set_or_explicit_unknown")),
                "dose_value": _numeric(action.get("dose_value") or action.get("physical_dose")),
                "dose_unit": _text(action.get("dose_unit")),
                "duration": _clean(action.get("duration")),
                "duration_unit": _text(action.get("duration_unit")),
                "schedule": _text(action.get("schedule")),
                "route": _text(action.get("route") or action.get("route_or_not_applicable")),
                "cell_or_tissue_context": _text(action.get("cell_or_tissue_context")),
                "exposure_program_ref": _text(action.get("exposure_program_ref") or action.get("exposure_program_ref_or_explicit_unavailable")),
                "occupancy_program_ref": _text(action.get("occupancy_program_ref") or action.get("occupancy_program_ref_or_explicit_unavailable")),
                "action_completeness": block.contract["action_completeness"],
                "missing_fields": "|".join(missing) if missing else "none",
                "u0_action_compatible": not missing,
                "measurement_namespace": _measurement_namespace(block),
                "u0_measurement_compatible": False,
                "evidence_grade": _text(block.manifest.get("qc_grade"), _text(block.contract.get("evidence_grade"))),
            }
            rows.append(row)
            lookup[action_uid] = row
    if len(rows) != len(lookup):
        raise ValueError("World-action identities overlap across real-intervention sources")
    return sorted(rows, key=lambda row: str(row["world_action_uid"])), lookup


def _episodes(
    observations: list[dict[str, object]],
    actions: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for observation in observations:
        if observation["observation_role"] != "intervention_endpoint":
            continue
        action_uid = str(observation["world_action_uid"])
        action = actions.get(action_uid)
        if action is None:
            raise ValueError(f"Observation references an unknown action: {action_uid}")
        identity = {
            "observation_uid": observation["observation_uid"],
            "world_action_uid": action_uid,
            "control_group_uid": observation["control_group_uid"],
        }
        rows.append(
            {
                "episode_uid": f"real_intervention_episode:{canonical_sha256(identity)[:24]}",
                "publication_uid": observation["publication_uid"],
                "accession_uid": observation["accession_uid"],
                "source_collection_uid": observation["source_collection_uid"],
                "biological_unit_uid": observation["biological_unit_uid"],
                "baseline_state_uid": observation["control_group_uid"],
                "world_action_uid": action_uid,
                "action_family_uid": action["action_family_uid"],
                "regimen_family_uid": action["regimen_family_uid"],
                "intervention_type": action["intervention_type"],
                "agent_set": action["agent_set"],
                "post_action_observation_uid": observation["observation_uid"],
                "trajectory_uid": observation["trajectory_uid"],
                "control_uid": observation["control_group_uid"],
                "time_grid": f"[{observation['time_value']} {observation['time_unit']}]",
                "dose_value": action["dose_value"],
                "dose_unit": action["dose_unit"],
                "duration": action["duration"],
                "duration_unit": action["duration_unit"],
                "schedule": action["schedule"],
                "route": action["route"],
                "cell_or_tissue_context": action["cell_or_tissue_context"],
                "measurement_namespace": observation["measurement_namespace"],
                "uncertainty": observation["uncertainty"],
                "evidence_grade": observation["evidence_grade"],
                "evidence_class": observation["evidence_class"],
                "action_completeness": action["action_completeness"],
                "target_set": action["target_set"],
                "exposure_program_ref": action["exposure_program_ref"],
                "occupancy_program_ref": action["occupancy_program_ref"],
                "same_unit_longitudinal_trajectory": False,
                "decision_objective_available": False,
                "u0_supported": False,
                "model_training_eligible": False,
                "split_role": observation["split_role"],
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
            }
        )
    identities = [str(row["episode_uid"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Real-intervention episode identities are duplicated")
    return sorted(rows, key=lambda row: str(row["episode_uid"]))


def _source_rows(
    blocks: Mapping[str, EvidenceBlock],
    observations: list[dict[str, object]],
    episodes: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    quality: list[dict[str, object]] = []
    for source_key, block in blocks.items():
        source_observations = [row for row in observations if row["source_key"] == source_key]
        source_episodes = [row for row in episodes if row["accession_uid"] == block.contract["source_accession"]]
        publication_uid = str(source_observations[0]["publication_uid"])
        if publication_uid.startswith(("doi:", "PMID:")):
            publication_status = "peer_reviewed_publication_identifier_bound"
        elif source_key == "amyloid_peptide_timecourse":
            publication_status = "official_accession_manuscript_in_preparation"
        else:
            publication_status = "official_accession_without_publication_identifier_in_block"
        control_rows = sum(row["observation_role"] == "matched_control" for row in source_observations)
        action_families = {str(row["action_family_uid"]) for row in source_episodes}
        contexts = {str(row["cell_or_tissue_context"]) for row in source_observations}
        row = {
            "source_key": source_key,
            "source_id": block.contract["source_id"],
            "source_accession": block.contract["source_accession"],
            "publication_uid": publication_uid,
            "publication_status": publication_status,
            "block_manifest_path": block.manifest_path.relative_to(REPO_ROOT).as_posix(),
            "block_manifest_sha256": sha256_file(block.manifest_path),
            "evidence_class": block.contract["evidence_class"],
            "endpoint_evidence_class": block.contract.get("endpoint_evidence_class", block.contract["evidence_class"]),
            "qc_grade": _text(block.manifest.get("qc_grade"), _text(block.contract.get("evidence_grade"))),
            "observation_count": len(source_observations),
            "intervention_episode_count": len(source_episodes),
            "materialized_control_count": control_rows,
            # Episode rows use the canonical `control_uid` field.  The source
            # observation schema retains `control_group_uid`; do not mix the
            # two schemas while aggregating the materialized episode catalog.
            "control_lineage_count": sum(_text(row["control_uid"]) != UNKNOWN for row in source_episodes),
            "effective_sample_size": len({str(row["biological_unit_uid"]) for row in source_observations}),
            "action_family_count": len(action_families),
            "context_family_count": len(contexts),
            "timepoint_count": len({str(row["time_value"]) for row in source_observations}),
            "dose_coverage_count": len(
                {
                    f"{row['dose_value']}:{row['dose_unit']}"
                    for row in source_episodes
                    if _text(row["dose_value"]) != UNKNOWN and _text(row["dose_unit"]) != UNKNOWN
                }
            ),
            "trajectory_count": len({str(row["trajectory_uid"]) for row in source_episodes if not str(row["trajectory_uid"]).startswith("explicit_unavailable")}),
            "same_unit_longitudinal_trajectory_count": 0,
            "action_completeness": block.contract["action_completeness"],
            "measurement_namespace": _measurement_namespace(block),
            "value_unit": _value_unit(block),
            "split_role": block.contract["split_role"],
            "quantitative_c2b_eligible": True,
            "u0_contract_compatible": False,
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        }
        rows.append(row)
        quality.append(
            {
                **row,
                "biological_context_explicit": all(value != UNKNOWN for value in contexts),
                "intervention_identity_explicit": all(str(item["world_action_uid"]) != UNKNOWN for item in source_episodes),
                "control_identity_explicit": all(str(item["control_uid"]) != UNKNOWN for item in source_episodes),
                "time_explicit": all(item["time_grid"] != f"[{UNKNOWN} {UNKNOWN}]" for item in source_episodes),
                "outcome_quantitative": True,
                "author_claim_promoted_to_truth": False,
                "simulator_promoted_to_real_evidence": False,
                "quality_status": "admitted_quantitative_real_intervention_evidence",
            }
        )
    return rows, quality


def _block_matrix(block: EvidenceBlock) -> np.ndarray:
    return load_zarr_full_array(block.root / str(block.manifest["value_block_path"]), "values")


def _decision_sensitivity(
    policy: Mapping[str, object],
    blocks: Mapping[str, EvidenceBlock],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, float]]:
    qualification = policy["qualification"]
    assert isinstance(qualification, Mapping)
    rows: list[dict[str, object]] = [
        {
            "candidate_set_uid": "clue_npc_neu_without_materialized_controls",
            "source_key": "clue_npc_neu",
            "starting_context": "NPC_or_NEU_compound_dose_time_context",
            "action_count": int(blocks["clue_npc_neu"].manifest.get("action_count", 0)),
            "effect_magnitude": UNKNOWN,
            "effect_direction": UNKNOWN,
            "between_action_contrast": UNKNOWN,
            "within_control_variability": UNKNOWN,
            "time_response_variation": "available_across_signatures_not_matched_as_one_candidate_set",
            "dose_response_variation": "available_across_signatures_not_matched_as_one_candidate_set",
            "rank_separability": False,
            "decision_margin": UNKNOWN,
            "decision_margin_ci95": UNKNOWN,
            "outcome_rank_classification": "insufficient_context",
            "decision_objective_available": False,
            "world_action_decision_classification": "insufficient_context",
            "reason": "Level5 signatures bind control lineage but the canonical block lacks realized matched-control values and a prespecified therapeutic utility.",
        }
    ]
    sanity_metrics: dict[str, float] = {}

    pantethine = blocks["pantethine_hippocampus"]
    if pantethine.control_block_root is None or pantethine.control_observations is None:
        raise ValueError("Pantethine control block is required for sensitivity analysis")
    treated_features = pd.read_parquet(pantethine.root / str(pantethine.manifest["feature_metadata_path"]))
    control_manifest = _json(pantethine.control_block_manifest_path)
    control_features = pd.read_parquet(pantethine.control_block_root / str(control_manifest["feature_metadata_path"]))
    if treated_features["feature_id"].astype(str).tolist() != control_features["feature_id"].astype(str).tolist():
        raise ValueError("Pantethine treated/control feature axes differ")
    treated_matrix = transformed_expression(_block_matrix(pantethine))
    control_matrix = transformed_expression(
        load_zarr_full_array(pantethine.control_block_root / str(control_manifest["value_block_path"]), "values")
    )
    for genotype in sorted(set(pantethine.observations["genotype_class"].astype(str))):
        treated_indices = pantethine.observations.index[pantethine.observations["genotype_class"].astype(str) == genotype].to_numpy()
        control_indices = pantethine.control_observations.index[
            pantethine.control_observations["genotype_class"].astype(str) == genotype
        ].to_numpy()
        diagnostic = single_action_effect(treated_matrix[treated_indices], control_matrix[control_indices])
        rows.append(
            {
                "candidate_set_uid": f"E-MTAB-6772:{genotype}:pantethine_vs_vehicle",
                "source_key": "pantethine_hippocampus",
                "starting_context": f"mouse_hippocampus:{genotype}",
                "action_count": 1,
                "effect_magnitude": diagnostic["effect_magnitude"],
                "effect_direction": diagnostic["effect_direction"],
                "between_action_contrast": UNKNOWN,
                "within_control_variability": diagnostic["within_control_variability"],
                "time_response_variation": "single_terminal_endpoint",
                "dose_response_variation": "single_dose",
                "rank_separability": False,
                "decision_margin": UNKNOWN,
                "decision_margin_ci95": UNKNOWN,
                "outcome_rank_classification": "scientifically_informative_but_not_rank_sensitive",
                "decision_objective_available": False,
                "world_action_decision_classification": "scientifically_informative_but_not_rank_sensitive",
                "reason": "Matched quantitative endpoint has only one active action in this starting context.",
            }
        )
    sanity_metrics["pantethine_no_action_endpoint_error"] = float(
        np.mean([float(row["effect_magnitude"]) for row in rows if str(row["candidate_set_uid"]).startswith("E-MTAB-6772")])
    )

    dex = blocks["dexamethasone_neurons"]
    dex_matrix = transformed_expression(_block_matrix(dex))
    dex_control_index = int(dex.observations.index[dex.observations["action_uid"].fillna("").astype(str) == ""][0])
    dex_action_index = int(dex.observations.index[dex.observations["action_uid"].fillna("").astype(str) != ""][0])
    dex_effect = rms(dex_matrix[dex_action_index] - dex_matrix[dex_control_index])
    rows.append(
        {
            "candidate_set_uid": "GSE262992:dexamethasone_vs_control",
            "source_key": "dexamethasone_neurons",
            "starting_context": "mouse_hippocampal_neural_stem_cell_model",
            "action_count": 1,
            "effect_magnitude": dex_effect,
            "effect_direction": "mixed_transcriptomic_direction",
            "between_action_contrast": UNKNOWN,
            "within_control_variability": UNKNOWN,
            "time_response_variation": "single_terminal_endpoint",
            "dose_response_variation": "single_dose",
            "rank_separability": False,
            "decision_margin": UNKNOWN,
            "decision_margin_ci95": UNKNOWN,
            "outcome_rank_classification": "ambiguous",
            "decision_objective_available": False,
            "world_action_decision_classification": "insufficient_context",
            "reason": "Condition aggregates provide no replicate-level control variability and only one active action.",
        }
    )
    sanity_metrics["dexamethasone_no_action_endpoint_error"] = dex_effect

    amyloid = blocks["amyloid_peptide_timecourse"]
    amyloid_matrix = transformed_expression(_block_matrix(amyloid))
    observations = amyloid.observations.reset_index(drop=True)
    endpoint_errors: list[float] = []
    for time_value in sorted(set(observations["time_value"].astype(float))):
        time_mask = observations["time_value"].astype(float) == time_value
        control_indices = observations.index[time_mask & (observations["observation_role"] == "matched_control")].to_numpy()
        control = amyloid_matrix[control_indices]
        actions: dict[str, np.ndarray] = {}
        for action_uid in sorted(
            set(observations.loc[time_mask & (observations["observation_role"] == "intervention_endpoint"), "world_action_uid"].astype(str))
        ):
            action_indices = observations.index[time_mask & (observations["world_action_uid"].astype(str) == action_uid)].to_numpy()
            actions[action_uid] = amyloid_matrix[action_indices]
            endpoint_errors.extend(np.sqrt(np.mean(np.square(amyloid_matrix[action_indices] - control.mean(axis=0)), axis=1)).tolist())
        diagnostic = candidate_set_diagnostics(
            actions,
            control,
            resamples=int(qualification["bootstrap_resamples"]),
            seed=int(qualification["bootstrap_seed"]) + int(time_value),
            effect_noise_floor=float(qualification["effect_to_control_noise_floor"]),
            contrast_noise_floor=float(qualification["action_contrast_to_control_noise_floor"]),
        )
        rows.append(
            {
                "candidate_set_uid": f"E-MTAB-1841:SH-SY5Y:{time_value:g}h",
                "source_key": "amyloid_peptide_timecourse",
                "starting_context": f"human_SH-SY5Y_vehicle_matched:{time_value:g}h",
                "action_count": len(actions),
                "effect_magnitude": json.dumps(diagnostic["effect_magnitudes"], sort_keys=True, separators=(",", ":")),
                "effect_direction": json.dumps(diagnostic["effect_directions"], sort_keys=True, separators=(",", ":")),
                "between_action_contrast": json.dumps(diagnostic["pairwise_action_contrasts"], sort_keys=True, separators=(",", ":")),
                "within_control_variability": diagnostic["within_control_variability"],
                "time_response_variation": "four_prespecified_timepoints",
                "dose_response_variation": "single_dose",
                "rank_separability": diagnostic["rank_separable"],
                "decision_margin": diagnostic["decision_margin"],
                "decision_margin_ci95": f"[{diagnostic['decision_margin_ci95_lower']},{diagnostic['decision_margin_ci95_upper']}]",
                "outcome_rank_classification": diagnostic["outcome_rank_classification"],
                "decision_objective_available": False,
                "world_action_decision_classification": "scientifically_informative_but_not_rank_sensitive",
                "reason": "Real peptide-response contrasts are measurable, but perturbation magnitude is not a prespecified therapeutic utility and the action lacks target occupancy.",
            }
        )
    sanity_metrics["amyloid_no_action_endpoint_error"] = float(np.mean(endpoint_errors))
    outcome_sensitive = sum(row["outcome_rank_classification"] == "decision_sensitive" for row in rows)
    summary = {
        "schema_version": "neuronsifter-real-intervention-decision-sensitivity-v1",
        "status": "decision_sensitivity_resolved",
        "candidate_set_count": len(rows),
        "outcome_rank_sensitive_set_count": outcome_sensitive,
        "world_action_decision_sensitive_set_count": 0,
        "real_action_response_variation_available": True,
        "prespecified_therapeutic_decision_objective_available": False,
        "decision_sensitive_world_action_domain_available": False,
        "model_training_domain_qualified": False,
        "dominant_limitation": "real_transcriptomic_action_responses_lack_world_state_grid_occupancy_and_prespecified_therapeutic_utility",
        "classification_is_based_on_real_outcomes": True,
        "model_predictions_inspected": False,
        "analysis_transform": "log2(value+1)_within_source_only",
        "bootstrap_resamples": qualification["bootstrap_resamples"],
        "bootstrap_seed_base": qualification["bootstrap_seed"],
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    return rows, summary, sanity_metrics


def _split_audit(
    policy: Mapping[str, object],
    episodes: list[dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    rows_by_split = {
        split: [row for row in episodes if row["split_role"] == split]
        for split in ("train", "validation", "development_eval")
    }
    if any(not rows for rows in rows_by_split.values()):
        raise ValueError("Every development split must contain at least one episode")
    split_policy = policy["split_policy"]
    assert isinstance(split_policy, Mapping)
    overlap = {field: overlap_values(rows_by_split, str(field)) for field in split_policy["prohibit_overlap"]}
    episode_overlap = overlap_values(rows_by_split, "episode_uid")
    leakage_free = not episode_overlap and all(not value for value in overlap.values())
    audit = {
        "schema_version": "neuronsifter-real-intervention-leakage-audit-v1",
        "status": "passed" if leakage_free else "failed",
        "split_unit": split_policy["unit"],
        "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "episode_uid_overlap": episode_overlap,
        "group_overlap": overlap,
        "replicate_measurement_cross_split_overlap": False,
        "leakage_free": leakage_free,
        "development_splits_frozen_before_model_results": True,
        "protected_publication_holdout_access": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    if not leakage_free:
        raise ValueError(f"Real-intervention split leakage detected: {overlap}")
    return rows_by_split, audit


def _route_freeze(boundary: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    round26_paths = {
        "information_action_gate": REPO_ROOT / "Output/audits/round26_information_action_gate.json",
        "canonical_architecture": REPO_ROOT / "Output/audits/round26_canonical_architecture.json",
        "next_scientific_route": REPO_ROOT / "Output/audits/round26_next_scientific_route.json",
    }
    round25_execution = REPO_ROOT / "Output/audits/round25_full_suite_execution.json"
    freeze = {
        "schema_version": "neuronsifter-post-information-action-scientific-route-freeze-v1",
        "status": "post_round26_static_architecture_frozen",
        "canonical_twin": "U0_frozen",
        "active_components": [
            "Scientific Belief",
            "Structured World-Action",
            "Observation assimilation",
            "Reliability/support routing",
            "Insufficient-evidence routing",
        ],
        "inactive_headlines": {
            "decision_aware_information_acquisition": "blocked_insufficient_decision_sensitive_information",
            "online_twin_adaptation": "failed_prospective_twin_adaptation",
            "agent_adaptation": "unauthorized",
            "coevolution": "unsupported",
        },
        "source_bindings": {name: _artifact(path) for name, path in round26_paths.items()},
        "round25_execution_binding_caveat": {
            "preserved_not_rewritten": True,
            "execution_artifact": _artifact(round25_execution),
            "current_status": _json(round25_execution).get("status"),
            "known_validator_blockers": [
                "full_suite_status",
                "full_suite_tests_failed",
                "full_suite_test_count",
                "full_suite_terminal_accounting",
                "architecture_full_suite_binding",
                "handoff_full_suite_binding",
            ],
            "scientific_route_dependency": False,
        },
        **boundary,
    }
    claims = {
        "schema_version": "neuronsifter-real-intervention-claim-reselection-v1",
        "status": "claim_map_frozen_for_real_intervention_development",
        "claims": {
            "scientific_belief": "supported_development_component",
            "structured_world_action": "supported_simulator_grounded_component_requires_real_intervention_test",
            "observation_assimilation": "supported_development_component",
            "reliability_and_abstention": "supported_development_component",
            "decision_aware_information_acquisition": "blocked_insufficient_decision_sensitive_information_not_active_headline",
            "online_twin_adaptation": "failed_prospective_twin_adaptation_not_active_headline",
            "agent_adaptation": "unauthorized",
            "coevolution": "unsupported_not_active_headline",
            "real_intervention_world_action": "under_development_evaluation",
        },
        **boundary,
    }
    return freeze, claims


def _field_lists() -> dict[str, list[str]]:
    return {
        "sources": [
            "source_key", "source_id", "source_accession", "publication_uid", "publication_status",
            "block_manifest_path", "block_manifest_sha256", "evidence_class", "endpoint_evidence_class",
            "qc_grade", "observation_count", "intervention_episode_count", "materialized_control_count",
            "control_lineage_count", "effective_sample_size", "action_family_count", "context_family_count",
            "timepoint_count", "dose_coverage_count", "trajectory_count", "same_unit_longitudinal_trajectory_count",
            "action_completeness", "measurement_namespace", "value_unit", "split_role",
            "quantitative_c2b_eligible", "u0_contract_compatible", "evidence_role",
            "publication_claim_eligible", "protected_holdout_access",
        ],
        "observations": [
            "observation_uid", "source_key", "source_id", "source_accession", "source_collection_uid",
            "publication_uid", "accession_uid", "biological_unit_uid", "sample_uid", "cell_or_tissue_context",
            "condition_uid", "observation_role", "world_action_uid", "action_family_uid", "regimen_family_uid",
            "trajectory_uid", "control_group_uid", "time_value", "time_unit", "dose_value", "dose_unit",
            "measurement_namespace", "value_unit", "uncertainty", "evidence_class", "trajectory_evidence_class",
            "evidence_grade", "split_role", "source_value_row_index", "source_block_manifest_path",
            "evidence_role", "publication_claim_eligible", "protected_holdout_access",
        ],
        "episodes": [
            "episode_uid", "publication_uid", "accession_uid", "source_collection_uid", "biological_unit_uid",
            "baseline_state_uid", "world_action_uid", "action_family_uid", "regimen_family_uid",
            "intervention_type", "agent_set", "post_action_observation_uid", "trajectory_uid", "control_uid", "time_grid",
            "dose_value", "dose_unit", "duration", "duration_unit", "schedule", "route", "cell_or_tissue_context",
            "measurement_namespace",
            "uncertainty", "evidence_grade", "evidence_class", "action_completeness", "target_set",
            "exposure_program_ref", "occupancy_program_ref", "same_unit_longitudinal_trajectory",
            "decision_objective_available", "u0_supported", "model_training_eligible", "split_role",
            "evidence_role", "publication_claim_eligible", "protected_holdout_access",
        ],
        "actions": [
            "world_action_uid", "source_key", "source_accession", "publication_uid", "action_family_uid",
            "regimen_family_uid", "intervention_type", "agent_set", "target_set", "dose_value", "dose_unit",
            "duration", "duration_unit", "schedule", "route", "cell_or_tissue_context",
            "exposure_program_ref", "occupancy_program_ref", "action_completeness", "missing_fields",
            "u0_action_compatible", "measurement_namespace", "u0_measurement_compatible", "evidence_grade",
        ],
        "sensitivity": [
            "candidate_set_uid", "source_key", "starting_context", "action_count", "effect_magnitude",
            "effect_direction", "between_action_contrast", "within_control_variability", "time_response_variation",
            "dose_response_variation", "rank_separability", "decision_margin", "decision_margin_ci95",
            "outcome_rank_classification", "decision_objective_available",
            "world_action_decision_classification", "reason",
        ],
    }


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("Real-intervention closure is identity-bound to the repository root")
    config = load_config(CONFIG_PATH)
    policy = config["real_intervention_evidence"]
    if not isinstance(policy, Mapping):
        raise TypeError("Real-intervention evidence policy must be a mapping")
    boundary = dict(policy["evidence_boundary"])
    outputs = {name: _repo_path(str(path)) for name, path in policy["outputs"].items()}
    fields = _field_lists()
    freeze, claims = _route_freeze(boundary)
    write_json(outputs["scientific_route_freeze"], freeze)
    write_json(outputs["claim_reselection"], claims)

    blocks = _load_blocks(policy)
    observations = _standard_observations(blocks)
    actions, action_lookup = _action_audit(blocks)
    episodes = _episodes(observations, action_lookup)
    source_rows, quality_rows = _source_rows(blocks, observations, episodes)
    _write_table(outputs["source_manifest"], source_rows, fields["sources"])
    _write_table(outputs["observation_manifest"], observations, fields["observations"])
    _write_table(outputs["episode_catalog"], episodes, fields["episodes"])
    _write_table(outputs["evidence_quality"], quality_rows, [*fields["sources"], "biological_context_explicit", "intervention_identity_explicit", "control_identity_explicit", "time_explicit", "outcome_quantitative", "author_claim_promoted_to_truth", "simulator_promoted_to_real_evidence", "quality_status"])
    _write_table(outputs["action_completeness"], actions, fields["actions"])

    e2_sources = [row for row in source_rows if row["endpoint_evidence_class"] == "E2_REAL_ENDPOINT"]
    e4_sources = [row for row in source_rows if row["evidence_class"] == "E4_REAL_TRAJECTORY"]
    coverage = {
        "schema_version": "neuronsifter-real-e2-e4-coverage-v1",
        "status": "real_intervention_evidence_materialized",
        "independent_study_count": len(source_rows),
        "peer_reviewed_publication_identifier_count": sum(row["publication_status"] == "peer_reviewed_publication_identifier_bound" for row in source_rows),
        "accession_only_study_count": sum(row["publication_status"] != "peer_reviewed_publication_identifier_bound" for row in source_rows),
        "independent_biological_unit_count": len({str(row["biological_unit_uid"]) for row in observations}),
        "action_family_count": len({str(row["action_family_uid"]) for row in episodes}),
        "context_family_count": len({str(row["cell_or_tissue_context"]) for row in observations}),
        "quantitative_E2_source_count": len(e2_sources),
        "quantitative_E2_episode_count": len(episodes),
        "quantitative_E4_source_count": len(e4_sources),
        "quantitative_E4_cross_sectional_trajectory_count": sum(int(row["trajectory_count"]) for row in e4_sources),
        "same_unit_longitudinal_E4_trajectory_count": 0,
        "materialized_control_observation_count": sum(row["observation_role"] == "matched_control" for row in observations),
        "control_linked_episode_count": sum(str(row["control_uid"]) != UNKNOWN for row in episodes),
        "effective_sample_size": len({str(row["biological_unit_uid"]) for row in observations}),
        "simulator_rows_counted_as_real": 0,
        "author_claim_rows_counted_as_quantitative_truth": 0,
        "real_E2_status": "available_quantitative_transcriptomic_endpoints",
        "real_E4_status": "available_quantitative_cross_sectional_timecourse_not_same_unit_longitudinal",
        **boundary,
    }
    write_json(outputs["real_e2_e4_coverage"], coverage)
    episode_integrity = {
        "schema_version": "neuronsifter-real-intervention-episode-integrity-v1",
        "status": "passed",
        "episode_count": len(episodes),
        "unique_episode_count": len({str(row["episode_uid"]) for row in episodes}),
        "unknown_identity_rows": sum(any(_text(row[field]) == UNKNOWN for field in ("episode_uid", "publication_uid", "accession_uid", "biological_unit_uid", "baseline_state_uid", "world_action_uid", "post_action_observation_uid", "control_uid")) for row in episodes),
        "action_reference_complete": all(str(row["world_action_uid"]) in action_lookup for row in episodes),
        "observation_reference_complete": all(str(row["post_action_observation_uid"]) in {str(item["observation_uid"]) for item in observations} for row in episodes),
        "absent_action_values_silently_filled": False,
        "same_unit_longitudinal_trajectory_claimed": False,
        "cross_sectional_timecourse_semantics_explicit": True,
        "simulator_rows_counted_as_real": 0,
        "protected_holdout_access": 0,
    }
    if episode_integrity["unknown_identity_rows"] or not episode_integrity["action_reference_complete"] or not episode_integrity["observation_reference_complete"]:
        raise ValueError(f"Real-intervention episode integrity failed: {episode_integrity}")
    write_json(outputs["episode_integrity"], episode_integrity)

    sensitivity_rows, sensitivity_summary, sanity_metrics = _decision_sensitivity(policy, blocks)
    _write_table(outputs["decision_sensitivity"], sensitivity_rows, fields["sensitivity"])
    write_json(outputs["decision_sensitive_summary"], sensitivity_summary)

    rows_by_split, leakage = _split_audit(policy, episodes)
    _write_table(outputs["train_split"], rows_by_split["train"], fields["episodes"])
    _write_table(outputs["validation_split"], rows_by_split["validation"], fields["episodes"])
    _write_table(outputs["development_eval_split"], rows_by_split["development_eval"], fields["episodes"])
    write_json(outputs["leakage_audit"], leakage)

    u0_rows: list[dict[str, object]] = []
    for source in source_rows:
        u0_rows.append(
            {
                "source_key": source["source_key"],
                "source_accession": source["source_accession"],
                "evaluation_unit_count": source["intervention_episode_count"],
                "supported_unit_count": 0,
                "coverage": 0.0,
                "abstention_rate": 1.0,
                "decision_label": "insufficient_evidence",
                "support_reason": "transcriptomic_measurement_namespace_and_partial_action_semantics_do_not_match_u0_state_grid_occupancy_contract",
                "effect_direction_accuracy": "not_computed_unsupported_domain",
                "endpoint_error": "not_computed_unsupported_domain",
                "pairwise_action_ranking": "not_computed_unsupported_domain",
                "trajectory_error": "not_computed_unsupported_domain",
                "calibration": "not_computed_no_numeric_prediction",
                "numeric_prediction_count": 0,
                "u0_parameter_update": False,
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
            }
        )
    u0_fields = list(u0_rows[0])
    _write_table(outputs["u0_results"], u0_rows, u0_fields)
    u0_gate = {
        "schema_version": "neuronsifter-u0-real-intervention-gate-v1",
        "status": "blocked_zero_supported_real_intervention_coverage",
        "canonical_twin": "U0_frozen",
        "u0_parameter_update": False,
        "evaluation_episode_count": len(episodes),
        "supported_episode_count": 0,
        "numeric_prediction_count": 0,
        "coverage": 0.0,
        "abstention_rate": 1.0,
        "u0_real_intervention_support_known": True,
        "u0_real_intervention_performance_status": "not_estimable_zero_supported_coverage",
        "unsupported_values_used_as_successful_results": False,
        "state_grid_compatible_episode_count": 0,
        "occupancy_complete_episode_count": 0,
        "measurement_namespace_compatible_episode_count": 0,
        **boundary,
    }
    write_json(outputs["u0_gate"], u0_gate)

    world_action_rows: list[dict[str, object]] = []
    for source in source_rows:
        source_key = str(source["source_key"])
        world_action_rows.append(
            {
                "source_key": source_key,
                "method": "U0_frozen",
                "evaluation_status": "unsupported_explicit_abstention",
                "endpoint_error": "not_computed",
                "trajectory_error": "not_computed",
                "coverage": 0.0,
                "notes": "No adapter was invented across transcriptomics and the U0 microenvironment state grid.",
            }
        )
        metric_key = {
            "pantethine_hippocampus": "pantethine_no_action_endpoint_error",
            "dexamethasone_neurons": "dexamethasone_no_action_endpoint_error",
            "amyloid_peptide_timecourse": "amyloid_no_action_endpoint_error",
        }.get(source_key)
        world_action_rows.append(
            {
                "source_key": source_key,
                "method": "no_action_persistence",
                "evaluation_status": "evaluated_within_source" if metric_key else "not_valid_without_materialized_control_values",
                "endpoint_error": sanity_metrics[metric_key] if metric_key else "not_computed",
                "trajectory_error": "not_computed_no_same_unit_longitudinal_trajectory",
                "coverage": 1.0 if metric_key else 0.0,
                "notes": "Log2 expression endpoint sanity control; not a NeuronSifter prediction.",
            }
        )
        world_action_rows.append(
            {
                "source_key": source_key,
                "method": "observed_delta_linear_control",
                "evaluation_status": "not_valid_without_trainable_same_namespace_baseline_to_endpoint_pairs",
                "endpoint_error": "not_computed",
                "trajectory_error": "not_computed",
                "coverage": 0.0,
                "notes": "The evidence does not support fitting a leakage-safe common linear control across sources.",
            }
        )
    _write_table(outputs["world_action_results"], world_action_rows, list(world_action_rows[0]))
    training_gate = {
        "schema_version": "neuronsifter-real-intervention-static-training-gate-v1",
        "status": "not_authorized_incompatible_world_action_domain",
        "training_attempts": 0,
        "successor_lineage_created": False,
        "u0_immutable": True,
        "offline_static_training_only_if_authorized": True,
        "model_training_domain_qualified": False,
        "reasons": [
            "zero_episode_matches_the_u0_15_species_state_grid_measurement_contract",
            "zero_episode_binds_complete_target_exposure_occupancy_semantics",
            "no_prespecified_therapeutic_decision_objective_is_observed",
            "cross_source_transcriptomic_measurements_do_not_form_one_trainable_world_state_contract",
        ],
        "large_scale_training_started": False,
        "gpu_allocated": False,
        **boundary,
    }
    write_json(outputs["training_gate"], training_gate)

    reliability_rows = [
        {"stratum": "all_real_intervention_episodes", "episode_count": len(episodes), "u0_supported_count": 0, "coverage": 0.0, "abstention_rate": 1.0, "false_confident_prediction_count": 0, "status": "insufficient_evidence"},
        {"stratum": "in_support", "episode_count": 0, "u0_supported_count": 0, "coverage": 0.0, "abstention_rate": "not_applicable", "false_confident_prediction_count": 0, "status": "empty_supported_region"},
        {"stratum": "incomplete_action_metadata", "episode_count": len(episodes), "u0_supported_count": 0, "coverage": 0.0, "abstention_rate": 1.0, "false_confident_prediction_count": 0, "status": "target_or_occupancy_missing"},
        {"stratum": "transcriptomic_measurement_namespace", "episode_count": len(episodes), "u0_supported_count": 0, "coverage": 0.0, "abstention_rate": 1.0, "false_confident_prediction_count": 0, "status": "state_grid_incompatible"},
        {"stratum": "cross_sectional_real_E4", "episode_count": sum(row["evidence_class"] == "E2_REAL_ENDPOINT" and not str(row["trajectory_uid"]).startswith("explicit_unavailable") for row in episodes), "u0_supported_count": 0, "coverage": 0.0, "abstention_rate": 1.0, "false_confident_prediction_count": 0, "status": "not_same_unit_longitudinal"},
    ]
    _write_table(outputs["reliability_report"], reliability_rows, list(reliability_rows[0]))
    failure_rows = [
        {"failure_uid": "measurement_namespace_incompatibility", "affected_episode_count": len(episodes), "failure_class": "evidence_model_contract", "scientific_interpretation": "Real transcriptomic outcomes cannot be numerically evaluated by the frozen 15-species microenvironment Twin.", "response": "explicit_abstention"},
        {"failure_uid": "occupancy_semantics_unavailable", "affected_episode_count": len(episodes), "failure_class": "structured_action_incompleteness", "scientific_interpretation": "No admitted episode binds the target occupancy program required by U0.", "response": "do_not_invent_action_fields"},
        {"failure_uid": "therapeutic_decision_objective_unavailable", "affected_episode_count": len(episodes), "failure_class": "decision_benchmark_incompleteness", "scientific_interpretation": "Observed transcriptomic perturbation magnitude is not a therapeutic utility.", "response": "do_not_promote_action_contrast_to_decision_gain"},
        {"failure_uid": "same_unit_longitudinal_E4_unavailable", "affected_episode_count": sum(not str(row["trajectory_uid"]).startswith("explicit_unavailable") for row in episodes), "failure_class": "trajectory_semantics", "scientific_interpretation": "The time course uses independent destructive samples at each timepoint.", "response": "retain_cross_sectional_trajectory_label"},
    ]
    _write_table(outputs["failure_report"], failure_rows, list(failure_rows[0]))
    scope = {
        "schema_version": "neuronsifter-real-intervention-scope-v1",
        "status": "supported_region_empty_for_u0",
        "real_quantitative_evidence_available": True,
        "real_action_response_contrasts_available": True,
        "u0_supported_region": "none_in_current_real_intervention_catalog",
        "u0_coverage": 0.0,
        "abstention_rate": 1.0,
        "false_confident_predictions": 0,
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
        "sample_state_level_only": True,
        "reliability_thresholds_changed": False,
        **boundary,
    }
    write_json(outputs["scope_audit"], scope)

    c2b_gate = {
        "schema_version": "neuronsifter-c2b-real-intervention-development-gate-v1",
        "gate_uid": "c2b_real_intervention_development_resolved",
        "status": "blocked_insufficient_real_intervention_evidence",
        "gate_resolved": True,
        "c2b_real_intervention_development_resolved": False,
        "real_intervention_evidence_inventory_complete": True,
        "action_semantics_explicit": True,
        "real_E2_status": coverage["real_E2_status"],
        "real_E4_status": coverage["real_E4_status"],
        "decision_sensitivity_known": True,
        "decision_sensitive_world_action_domain_available": False,
        "development_splits_frozen": True,
        "leakage_audit_complete": True,
        "u0_real_intervention_performance_status": u0_gate["u0_real_intervention_performance_status"],
        "new_static_model_status": training_gate["status"],
        "reliability_supported_region_known": True,
        "dominant_blocker": "quantitative_real_e2_e4_is_transcriptomic_and_lacks_u0_state_grid_occupancy_and_therapeutic_decision_semantics",
        "policy_failure_claimed": False,
        "model_failure_claimed": False,
        "next_route": "acquire_contract_aligned_real_intervention_state_trajectories_or_redesign_cross_modal_world_state_mapping_before_model_training",
        **boundary,
    }
    write_json(outputs["c2b_gate"], c2b_gate)
    architecture = {
        "schema_version": "neuronsifter-canonical-architecture-after-real-intervention-audit-v1",
        "status": "frozen_static_architecture_real_intervention_support_blocked",
        "canonical_twin": "U0_frozen",
        "components": ["Scientific Belief", "Structured World-Action", "Observation assimilation", "Reliability/support routing", "Insufficient-evidence routing"],
        "real_intervention_static_successor": "not_authorized",
        "decision_aware_information_acquisition_headline": False,
        "online_twin_adaptation": False,
        "agent_adaptation": False,
        "coevolution": False,
        "publication_comparison_open": False,
        **boundary,
    }
    write_json(outputs["canonical_architecture"], architecture)
    next_route = {
        "schema_version": "neuronsifter-next-scientific-route-after-real-intervention-audit-v1",
        "status": "evidence_domain_expansion_or_cross_modal_world_state_redesign_required",
        "priority": [
            "real_intervention_outcomes_aligned_to_microenvironment_state_variables",
            "explicit_target_exposure_occupancy_programs",
            "prespecified_therapeutic_decision_objective",
            "only_then_offline_static_world_action_training",
        ],
        "prohibited_default_routes": ["information_policy_rescue", "online_twin_adaptation", "agent_adaptation", "coevolution", "model_scaling"],
        "external_publication_baselines_resume_authorized": False,
        **boundary,
    }
    write_json(outputs["next_scientific_route"], next_route)

    output_bindings = {
        name: _artifact(path)
        for name, path in outputs.items()
        if name != "reproducibility_manifest"
    }
    source_files = [
        CONFIG_PATH,
        REPO_ROOT / "Libs/configs/data/real_intervention_evidence.yaml",
        REPO_ROOT / "Libs/scripts/builders/retrieve_real_intervention_source.py",
        REPO_ROOT / "Libs/scripts/builders/materialize_real_intervention_timecourse.py",
        REPO_ROOT / "Libs/scripts/experiments/resolve_real_intervention_world_action.py",
        REPO_ROOT / "Libs/utils/evaluation/real_intervention_evidence.py",
    ]
    reproducibility = {
        "schema_version": "neuronsifter-real-intervention-reproducibility-manifest-v1",
        "producer": BUILDER_ID,
        "created_at": _now(),
        "source_bindings": {path.relative_to(REPO_ROOT).as_posix(): _artifact(path) for path in source_files},
        "input_block_bindings": {key: _artifact(block.manifest_path) for key, block in blocks.items()},
        "output_bindings": output_bindings,
        "output_artifact_count": len(output_bindings),
        "source_study_count": len(blocks),
        "episode_count": len(episodes),
        "bootstrap_seed": policy["qualification"]["bootstrap_seed"],
        "bootstrap_resamples": policy["qualification"]["bootstrap_resamples"],
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    write_json(outputs["reproducibility_manifest"], reproducibility)
    return c2b_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = build(REPO_ROOT)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
