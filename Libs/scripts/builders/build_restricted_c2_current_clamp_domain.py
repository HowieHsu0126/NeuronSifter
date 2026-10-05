"""Execute and audit a restricted, action-complete current-clamp C2 domain.

The domain is deliberately scoped to source-backed simulator dynamics.  It
does not represent a pharmacological intervention or a real observation-linked
trajectory.  Every intervention protocol is paired with a producer-executed
zero-current control under the same source, initial state, parameters, and
time grid.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.teacher_generation.run_neuron_rxd_microenvironment_teacher import (
    protocol_to_trajectory,
    run_protocol_batch,
    sanity_check,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path, write_json, write_tsv
from Libs.utils.model.dynamics_store import (
    load_zarr_full_array,
    load_zarr_rows,
    read_zarr_array_metadata,
)
from Libs.utils.model.microenvironment_teacher import default_coordinate_payload, json_payload, teacher_contract_version, write_microenvironment_zarr
from Libs.utils.model.screening_core import stable_uid


ACTION_FIELDS = [
    "action_uid",
    "intervention_type",
    "target_set",
    "agent_set",
    "dose_value",
    "dose_unit",
    "route",
    "schedule",
    "duration_ms",
    "context",
    "exposure_semantics",
    "occupancy_semantics",
    "source_model_id",
    "source_archive_sha256",
    "initial_state_uid",
    "parameter_family_uid",
    "fidelity",
    "time_grid_uid",
    "seed_policy",
    "action_role",
    "pair_uid",
    "control_action_uid",
    "split_role",
]

TRAJECTORY_FIELDS = [
    "schema_version",
    "trajectory_uid",
    "domain_uid",
    "family_uid",
    "initial_observation_uid",
    "action_uid",
    "pair_uid",
    "axes",
    "artifact_uri",
    "artifact_hash",
    "teacher_family",
    "teacher_contract",
    "source_model_id",
    "source_archive_sha256",
    "producer_execution_uid",
    "uncertainty_policy",
    "qc_grade",
    "train_eval_role",
    "source_holdout_group",
    "action_effect_max_abs",
    "finite_values",
    "nonempty_values",
    "claim_scope",
]

CONTROL_FIELDS = [
    "pair_uid",
    "intervention_action_uid",
    "control_action_uid",
    "intervention_trajectory_uid",
    "control_trajectory_uid",
    "same_source_model",
    "same_initial_state",
    "same_parameter_family",
    "same_fidelity",
    "same_time_grid",
    "compatible_seed_policy",
    "control_producer_executed",
    "effect_max_abs",
    "split_role",
]

SPLIT_FIELDS = [
    "split_uid",
    "split_role",
    "pair_uid",
    "action_uid",
    "trajectory_uid",
    "group_key",
    "source_holdout_group",
    "unseen_action_holdout",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _utc_uid(domain_uid: str, source_hash: str) -> tuple[str, str]:
    retrieval = datetime.now(timezone.utc).isoformat()
    payload = f"{domain_uid}|{source_hash}|{retrieval}".encode("utf-8")
    return f"execution_{hashlib.sha256(payload).hexdigest()[:24]}", retrieval


def _domain_config(config: dict[str, Any]) -> dict[str, Any]:
    teacher = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
    domain = teacher.get("c2_domain")
    if not isinstance(domain, dict):
        raise KeyError("dynamics_ready_registry.neuron_rxd_teacher.c2_domain is required")
    protocols = domain.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("The restricted C2 domain must declare non-empty protocols")
    return domain


def _source_archive(config: dict[str, Any]) -> tuple[Path, str]:
    modeldb = config["dynamics_ready_registry"]["neuron_rxd_teacher"]["model_sources"]["modeldb"]
    archive = resolve_repo_path(str(modeldb["archive_path"]))
    if not archive.is_file() or archive.stat().st_size <= 0:
        raise FileNotFoundError(f"Public ModelDB source archive is not materialized: {archive}")
    return archive, _sha256_file(archive)


def _required_action_fields(protocol: dict[str, Any]) -> list[str]:
    required = {
        "action_uid": protocol.get("action_uid"),
        "action_agent": protocol.get("action_agent"),
        "action_target": protocol.get("action_target"),
        "amplitude_nA": protocol.get("amplitude_nA"),
        "action_route": protocol.get("action_route"),
        "schedule": protocol.get("schedule"),
        "duration_ms": protocol.get("duration_ms"),
        "context": protocol.get("context"),
        "exposure_semantics": protocol.get("exposure_semantics"),
    }
    return [name for name, value in required.items() if value is None or value == ""]


def _action_row(
    domain: dict[str, Any],
    protocol: dict[str, Any],
    source_hash: str,
    control_action_uid: str,
) -> dict[str, Any]:
    missing = _required_action_fields(protocol)
    if missing:
        raise ValueError(f"Action {protocol.get('action_uid')} is incomplete: {missing}")
    return {
        "action_uid": str(protocol["action_uid"]),
        "intervention_type": "electrical_current_clamp",
        "target_set": json_payload([str(protocol["action_target"])]),
        "agent_set": json_payload([str(protocol["action_agent"])]),
        "dose_value": float(protocol["amplitude_nA"]),
        "dose_unit": "nA",
        "route": str(protocol["action_route"]),
        "schedule": str(protocol["schedule"]),
        "duration_ms": float(protocol["duration_ms"]),
        "context": str(protocol["context"]),
        "exposure_semantics": str(protocol["exposure_semantics"]),
        "occupancy_semantics": str(protocol.get("occupancy_semantics", "not_applicable")),
        "source_model_id": str(domain["source_model_id"]),
        "source_archive_sha256": source_hash,
        "initial_state_uid": str(domain["initial_state_uid"]),
        "parameter_family_uid": str(domain["parameter_family_uid"]),
        "fidelity": str(domain["fidelity"]),
        "time_grid_uid": str(domain["time_grid_uid"]),
        "seed_policy": str(domain["seed_policy"]),
        "action_role": str(protocol["role"]),
        "pair_uid": str(protocol["pair_uid"]),
        "control_action_uid": control_action_uid,
        "split_role": str(protocol["split_role"]),
    }


def _validate_protocol_pairs(protocols: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    interventions = {str(row["pair_uid"]): row for row in protocols if row.get("role") == "intervention"}
    controls = {str(row["pair_uid"]): row for row in protocols if row.get("role") == "matched_control"}
    if not interventions:
        raise ValueError("The restricted C2 domain has no intervention protocols")
    if set(interventions) != set(controls):
        raise ValueError(f"Intervention/control pair mismatch: {sorted(set(interventions) ^ set(controls))}")
    for pair_uid, intervention in interventions.items():
        if float(intervention["amplitude_nA"]) <= 0.0:
            raise ValueError(f"Intervention {intervention['action_uid']} must have non-zero current")
        if float(controls[pair_uid]["amplitude_nA"]) != 0.0:
            raise ValueError(f"Control {controls[pair_uid]['action_uid']} must have zero current")
        if str(intervention["split_role"]) != str(controls[pair_uid]["split_role"]):
            raise ValueError(f"Pair {pair_uid} assigns intervention and control to different splits")
    return interventions, controls


def _validate_production_store(store_path: Path, expected: np.ndarray) -> dict[str, Any]:
    """Exercise the project row reader against the emitted C2 trajectory store."""
    metadata = read_zarr_array_metadata(store_path, "trajectories")
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    if shape != tuple(expected.shape):
        raise ValueError(f"C2 trajectory store shape {shape} does not match emitted trajectories {expected.shape}")
    if any(chunk_size > axis_size for chunk_size, axis_size in zip(chunks, shape)):
        raise ValueError(f"C2 trajectory store has oversized chunks {chunks} for shape {shape}")
    row_indices = np.arange(expected.shape[0], dtype=np.int64)
    loaded = load_zarr_rows(store_path, "trajectories", row_indices)
    if loaded.shape != expected.shape or not np.array_equal(loaded, expected):
        raise ValueError("C2 trajectory store did not round-trip through the production row reader")
    return {"status": "passed", "reader": "dynamics_store.load_zarr_rows", "shape": list(shape), "chunks": list(chunks)}


def _validate_zarr_three_store(store_path: Path, expected: np.ndarray) -> dict[str, Any]:
    """Require an independent Zarr reader to decode the emitted store."""
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("The C2 integrity contract requires a Zarr 3 reader.") from exc
    array = zarr.open_array((store_path / "trajectories").as_posix(), mode="r")
    loaded = np.asarray(array[:])
    if loaded.shape != expected.shape or not np.array_equal(loaded, expected):
        raise ValueError("C2 trajectory store did not round-trip through the Zarr 3 reader")
    return {"status": "passed", "reader": "zarr.open_array", "shape": list(loaded.shape)}


def _bool(value: object) -> bool:
    return value is True or str(value).strip().casefold() == "true"


def _validate_manifest_integrity(
    action_rows: list[dict[str, str]],
    trajectory_rows: list[dict[str, str]],
    control_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    *,
    store_hash: str,
    trajectory_count: int,
) -> dict[str, Any]:
    """Validate the emitted C2 joins before reporting a landed restricted domain."""
    action_ids = [str(row["action_uid"]) for row in action_rows]
    trajectory_ids = [str(row["trajectory_uid"]) for row in trajectory_rows]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("C2 action manifest contains duplicate action identifiers")
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("C2 trajectory manifest contains duplicate trajectory identifiers")
    if len(trajectory_rows) != trajectory_count:
        raise ValueError("C2 trajectory manifest does not account for every store row")
    trajectory_by_action = {str(row["action_uid"]): row for row in trajectory_rows}
    if set(action_ids) != set(trajectory_by_action):
        raise ValueError("C2 action and trajectory manifests are not one-to-one")
    for row in trajectory_rows:
        artifact_uri = str(row.get("artifact_uri", ""))
        if "#simulation=" not in artifact_uri:
            raise ValueError("C2 trajectory manifest lacks a store-row binding")
        try:
            index = int(artifact_uri.rsplit("#simulation=", 1)[1])
        except ValueError as exc:
            raise ValueError("C2 trajectory manifest has an invalid store-row binding") from exc
        if index < 0 or index >= trajectory_count:
            raise ValueError("C2 trajectory manifest references a store row outside the emitted array")
        if str(row.get("artifact_hash", "")) != store_hash:
            raise ValueError("C2 trajectory manifest has a store-hash mismatch")
    if len({str(row.get("artifact_uri", "")) for row in trajectory_rows}) != trajectory_count:
        raise ValueError("C2 trajectory manifest maps multiple records to the same store row")
    pair_ids = {str(row["pair_uid"]) for row in control_rows}
    if len(pair_ids) != len(control_rows):
        raise ValueError("C2 matched-control manifest contains duplicate pairs")
    for row in control_rows:
        intervention = str(row["intervention_action_uid"])
        control = str(row["control_action_uid"])
        if intervention not in trajectory_by_action or control not in trajectory_by_action:
            raise ValueError("C2 matched-control manifest references an unknown action")
        if str(trajectory_by_action[intervention]["pair_uid"]) != str(row["pair_uid"]):
            raise ValueError("C2 intervention trajectory has a mismatched pair")
        if str(trajectory_by_action[control]["pair_uid"]) != str(row["pair_uid"]):
            raise ValueError("C2 control trajectory has a mismatched pair")
        if not _bool(row.get("control_producer_executed")):
            raise ValueError("C2 matched control was not producer-executed")
    split_actions = [str(row["action_uid"]) for row in split_rows]
    split_trajectories = [str(row["trajectory_uid"]) for row in split_rows]
    if len(split_actions) != len(set(split_actions)) or set(split_actions) != set(action_ids):
        raise ValueError("C2 strict split manifest does not partition actions exactly once")
    if len(split_trajectories) != len(set(split_trajectories)) or set(split_trajectories) != set(trajectory_ids):
        raise ValueError("C2 strict split manifest does not partition trajectories exactly once")
    split_role_by_pair: dict[str, str] = {}
    for row in split_rows:
        pair_uid = str(row["pair_uid"])
        role = str(row["split_role"])
        previous_role = split_role_by_pair.setdefault(pair_uid, role)
        if previous_role != role:
            raise ValueError("C2 intervention/control pair crosses strict split roles")
    roles = sorted({str(row["split_role"]) for row in split_rows})
    overlap_by_key: dict[str, int] = {}
    for key in ("pair_uid", "action_uid", "trajectory_uid", "group_key"):
        values_by_role = {
            role: {str(row[key]) for row in split_rows if str(row["split_role"]) == role}
            for role in roles
        }
        overlaps: set[str] = set()
        for index, left in enumerate(roles):
            for right in roles[index + 1 :]:
                overlaps |= values_by_role[left] & values_by_role[right]
        overlap_by_key[key] = len(overlaps)
    if any(overlap_by_key.values()):
        raise ValueError(f"C2 strict split overlap detected: {overlap_by_key}")
    test_actions = {str(row["action_uid"]) for row in split_rows if str(row["split_role"]) == "test"}
    development_actions = {
        str(row["action_uid"])
        for row in split_rows
        if str(row["split_role"]) in {"train", "validation"}
    }
    if not test_actions or test_actions & development_actions:
        raise ValueError("C2 final holdout is missing or overlaps the development actions")
    return {
        "status": "passed",
        "counts": {
            "actions": len(action_rows),
            "trajectories": len(trajectory_rows),
            "matched_controls": len(control_rows),
            "strict_split_rows": len(split_rows),
        },
        "overlap_by_key": overlap_by_key,
        "final_holdout_unseen": True,
        "source_holdout_group_count": len({str(row["source_holdout_group"]) for row in split_rows}),
    }


def verify_existing_domain() -> dict[str, Any]:
    """Rebuild the integrity audit from the current immutable C2 artifacts."""
    readiness_path = resolve_repo_path("Output/audits/c2_domain_readiness.json")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    producer = readiness.get("producer")
    if not isinstance(producer, dict):
        raise ValueError("C2 readiness artifact lacks a producer binding")
    relative_store = Path(str(producer.get("trajectory_store", "")))
    if relative_store.is_absolute() or ".." in relative_store.parts:
        raise ValueError("C2 readiness artifact has an unsafe trajectory store path")
    store_path = REPO_ROOT / relative_store
    if not store_path.is_dir():
        raise FileNotFoundError(f"C2 trajectory store is unavailable: {store_path}")
    trajectories = load_zarr_full_array(store_path, "trajectories")
    if trajectories.size == 0 or not np.isfinite(trajectories).all():
        raise ValueError("C2 trajectory store is empty or contains non-finite values")
    store_hash = _sha256_tree(store_path)
    expected_hash = str(producer.get("trajectory_store_sha256", ""))
    if store_hash != expected_hash:
        raise ValueError("C2 readiness artifact does not match the current trajectory store hash")
    production_validation = _validate_production_store(store_path, trajectories)
    zarr_validation = _validate_zarr_three_store(store_path, trajectories)
    manifest_root = resolve_repo_path("Input/manifests/c2")
    manifest_validation = _validate_manifest_integrity(
        read_tsv(manifest_root / "action_catalog.tsv"),
        read_tsv(manifest_root / "trajectory_manifest.tsv"),
        read_tsv(manifest_root / "matched_control_manifest.tsv"),
        read_tsv(manifest_root / "strict_split_manifest.tsv"),
        store_hash=store_hash,
        trajectory_count=int(trajectories.shape[0]),
    )
    warnings = []
    source_holdout_status = str(readiness.get("source_holdout_status", ""))
    if source_holdout_status:
        warnings.append(source_holdout_status)
    audit = {
        "schema_version": "neuronsifter-c2-integrity-audit",
        "status": "passed",
        "claim_scope": readiness.get("claim_scope"),
        **manifest_validation,
        "strict_split_overlap": manifest_validation["overlap_by_key"]["group_key"],
        "store_shape": list(trajectories.shape),
        "store_chunks": production_validation["chunks"],
        "store_hash": store_hash,
        "store_hash_matches_readiness": True,
        "project_reader_status": production_validation["status"],
        "zarr_three_reader_status": zarr_validation["status"],
        "source_holdout_status": source_holdout_status,
        "warnings": warnings,
    }
    audit_path = resolve_repo_path("Output/audits/c2/current_clamp/integrity_audit.json")
    write_json(audit_path, audit)
    return audit


def execute_domain(config: dict[str, Any]) -> dict[str, Any]:
    domain = _domain_config(config)
    archive, source_hash = _source_archive(config)
    protocols = [dict(row) for row in domain["protocols"]]
    interventions, controls = _validate_protocol_pairs(protocols)
    execution_uid, retrieval_time = _utc_uid(str(domain["domain_uid"]), source_hash)
    store_path = resolve_repo_path("Input/processed/c2") / str(domain["domain_uid"]) / execution_uid / "trajectories.zarr"
    trajectories: list[np.ndarray] = []
    payloads: list[dict[str, Any]] = []
    protocol_indexes: dict[str, int] = {}
    runtimes = [
        {
            "stimulus_amplitude_nA": float(protocol["amplitude_nA"]),
            "stimulus_delay_ms": float(protocol["delay_ms"]),
            "stimulus_duration_ms": float(protocol["duration_ms"]),
        }
        for protocol in protocols
    ]
    for index, (protocol, payload) in enumerate(zip(protocols, run_protocol_batch(config, str(domain["source_model_id"]), runtimes), strict=True)):
        sanity_check(config, payload)
        trajectory = protocol_to_trajectory(config, payload)
        if trajectory.size == 0 or not np.isfinite(trajectory).all():
            raise ValueError(f"Protocol {protocol['protocol_uid']} produced empty or non-finite values")
        protocol_indexes[str(protocol["action_uid"])] = index
        trajectories.append(trajectory)
        payloads.append(payload)
    trajectory_array = np.stack(trajectories, axis=0).astype(np.float32)
    uncertainty_floor = float(config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]["uncertainty_floor"])
    uncertainty = np.full((trajectory_array.shape[0], trajectory_array.shape[-1]), uncertainty_floor, dtype=np.float32)
    write_microenvironment_zarr(
        store_path,
        trajectory_array,
        uncertainty,
        config,
        {
            "domain_uid": domain["domain_uid"],
            "execution_uid": execution_uid,
            "claim_scope": domain["claim_scope"],
            "source_model_id": domain["source_model_id"],
            "source_archive_sha256": source_hash,
            "retrieved_at": retrieval_time,
        },
    )
    production_store_validation = _validate_production_store(store_path, trajectory_array)
    store_hash = _sha256_tree(store_path)
    control_by_pair = {pair_uid: protocol_indexes[str(control["action_uid"])] for pair_uid, control in controls.items()}
    intervention_by_pair = {pair_uid: protocol_indexes[str(intervention["action_uid"])] for pair_uid, intervention in interventions.items()}
    action_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    effect_by_pair: dict[str, float] = {}
    for pair_uid, intervention in interventions.items():
        control = controls[pair_uid]
        intervention_index = intervention_by_pair[pair_uid]
        control_index = control_by_pair[pair_uid]
        effect = float(np.max(np.abs(trajectory_array[intervention_index] - trajectory_array[control_index])))
        if effect <= 1.0e-7:
            raise ValueError(f"Action {intervention['action_uid']} is indistinguishable from its control")
        effect_by_pair[pair_uid] = effect
        action_rows.append(_action_row(domain, intervention, source_hash, str(control["action_uid"])))
        action_rows.append(_action_row(domain, control, source_hash, str(control["action_uid"])))
        split_role = str(intervention["split_role"])
        for protocol, index in ((intervention, intervention_index), (control, control_index)):
            action_uid = str(protocol["action_uid"])
            trajectory_uid = stable_uid("c2_trajectory", domain["domain_uid"], execution_uid, action_uid)
            initial_observation_uid = stable_uid("c2_initial_state", domain["domain_uid"], domain["initial_state_uid"])
            trajectory_rows.append({
                "schema_version": "neuronsifter-c2-trajectory",
                "trajectory_uid": trajectory_uid,
                "domain_uid": str(domain["domain_uid"]),
                "family_uid": "simulator_grounded_c2_restricted",
                "initial_observation_uid": initial_observation_uid,
                "action_uid": action_uid,
                "pair_uid": pair_uid,
                "axes": json_payload(default_coordinate_payload(config)),
                "artifact_uri": f"{store_path.resolve().relative_to(REPO_ROOT)}#simulation={index}",
                "artifact_hash": store_hash,
                "teacher_family": "neuron_rxd_microenvironment",
                "teacher_contract": teacher_contract_version(config),
                "source_model_id": str(domain["source_model_id"]),
                "source_archive_sha256": source_hash,
                "producer_execution_uid": execution_uid,
                "uncertainty_policy": json_payload({"policy": "deterministic_floor", "value": uncertainty_floor, "epistemic_estimate": False}),
                "qc_grade": "S",
                "train_eval_role": f"restricted_c2_{split_role}",
                "source_holdout_group": source_hash,
                "action_effect_max_abs": effect if protocol.get("role") == "intervention" else 0.0,
                "finite_values": True,
                "nonempty_values": True,
                "claim_scope": str(domain["claim_scope"]),
            })
            split_rows.append({
                "split_uid": stable_uid("c2_split", domain["domain_uid"], split_role),
                "split_role": split_role,
                "pair_uid": pair_uid,
                "action_uid": action_uid,
                "trajectory_uid": trajectory_uid,
                "group_key": pair_uid,
                "source_holdout_group": source_hash,
                "unseen_action_holdout": split_role == "test",
            })
        control_rows.append({
            "pair_uid": pair_uid,
            "intervention_action_uid": str(intervention["action_uid"]),
            "control_action_uid": str(control["action_uid"]),
            "intervention_trajectory_uid": trajectory_rows[-2]["trajectory_uid"],
            "control_trajectory_uid": trajectory_rows[-1]["trajectory_uid"],
            "same_source_model": True,
            "same_initial_state": True,
            "same_parameter_family": True,
            "same_fidelity": True,
            "same_time_grid": True,
            "compatible_seed_policy": True,
            "control_producer_executed": True,
            "effect_max_abs": effect,
            "split_role": split_role,
        })
    manifest_root = resolve_repo_path("Input/manifests/c2")
    schema_map = config.get("schemas", {})
    action_fields = list(schema_map.get("c2_action_fields", ACTION_FIELDS))
    trajectory_fields = list(schema_map.get("c2_trajectory_fields", TRAJECTORY_FIELDS))
    control_fields = list(schema_map.get("c2_matched_control_fields", CONTROL_FIELDS))
    split_fields = list(schema_map.get("c2_split_fields", SPLIT_FIELDS))
    write_tsv(manifest_root / "action_catalog.tsv", action_fields, action_rows)
    write_tsv(manifest_root / "trajectory_manifest.tsv", trajectory_fields, trajectory_rows)
    write_tsv(manifest_root / "matched_control_manifest.tsv", control_fields, control_rows)
    write_tsv(manifest_root / "strict_split_manifest.tsv", split_fields, split_rows)
    source_holdout_groups = {str(row["source_holdout_group"]) for row in split_rows}
    action_groups = {str(row["group_key"]) for row in split_rows}
    split_group_sets = {
        role: {str(row["group_key"]) for row in split_rows if row["split_role"] == role}
        for role in {str(row["split_role"]) for row in split_rows}
    }
    overlap = set()
    roles = sorted(split_group_sets)
    for left_index, left_role in enumerate(roles):
        for right_role in roles[left_index + 1 :]:
            overlap |= split_group_sets[left_role] & split_group_sets[right_role]
    scorecard_path = resolve_repo_path("Output/audits/c2_domain_scorecard.tsv")
    existing_rows: list[dict[str, str]] = []
    if scorecard_path.is_file():
        from Libs.utils.data.raw_data_registry import read_tsv

        existing_rows = read_tsv(scorecard_path)
    existing_rows = [row for row in existing_rows if row.get("candidate_uid") != str(domain["domain_uid"])]
    existing_rows.append({
        "candidate_uid": str(domain["domain_uid"]),
        "source": str(domain["source_accession"]),
        "public": "modeldb",
        "action_complete": "True",
        "matched_controls_complete": "True",
        "independent": "True",
        "strict_split_available": "True",
        "status": "simulator_grounded_c2_restricted",
        "reason": "Source-backed current-clamp programs have producer-executed zero-current controls; claims remain simulator-restricted.",
    })
    scorecard_fields = list(existing_rows[0].keys()) if existing_rows else ["candidate_uid", "source", "public", "action_complete", "matched_controls_complete", "independent", "strict_split_available", "status", "reason"]
    write_tsv(scorecard_path, scorecard_fields, existing_rows)
    qualification_rows = [{
        "domain_uid": str(domain["domain_uid"]),
        "producer_uid": str(domain["source_model_id"]),
        "source_repository": str(domain["source_repository_url"]),
        "source_accession": str(domain["source_accession"]),
        "source_archive_sha256": source_hash,
        "execution_uid": execution_uid,
        "protocol_count": len(protocols),
        "intervention_count": len(interventions),
        "matched_control_count": len(control_rows),
        "trajectory_store_hash": store_hash,
        "qualified_producer": True,
        "parameter_provenance_complete": True,
        "source_holdout_status": "not_applicable_single_public_model_restricted_scope",
        "claim_scope": str(domain["claim_scope"]),
    }]
    write_tsv(resolve_repo_path("Output/audits/c2_producer_qualification.tsv"), list(qualification_rows[0].keys()), qualification_rows)
    audit_root = resolve_repo_path("Output/audits/c2/current_clamp")
    write_json(audit_root / "parser_validation.json", {"status": "passed", "protocol_count": len(protocols), "finite_trajectory_values": True, "nonempty_trajectory_values": True})
    write_json(audit_root / "canonicalization_summary.json", {"status": "passed", "action_count": len(action_rows), "trajectory_count": len(trajectory_rows), "biological_unit": "simulator_protocol_execution"})
    write_json(audit_root / "biological_unit_audit.json", {"status": "passed", "unit_type": "simulator_protocol_execution", "unit_count": len(trajectory_rows), "biological_replication": False, "restriction": "not_real_observation_linked"})
    write_json(audit_root / "qc_summary.json", {"status": "passed", "qc_grade": "S", "effect_max_abs_by_pair": effect_by_pair, "finite_values": True, "split_overlap": len(overlap)})
    write_json(audit_root / "consumer_execution.json", {"status": "completed", "consumer": "restricted_c2_current_clamp_manifest_consumer", "action_count": len(action_rows), "trajectory_count": len(trajectory_rows), "control_count": len(control_rows), "store_hash": store_hash, "production_store_validation": production_store_validation})
    readiness = {
        "schema_version": "neuronsifter-c2-domain-readiness",
        "status": str(domain["claim_scope"]),
        "action_complete": True,
        "qualified_producer": True,
        "matched_controls_complete": all(bool(row["control_producer_executed"]) for row in control_rows),
        "independent_trajectory_units_nonzero": len(interventions) > 0,
        "independent_trajectory_unit_count": len(interventions),
        "parameter_provenance_complete": True,
        "strict_split_overlap": len(overlap),
        "final_holdout_unseen": bool(split_group_sets.get("test")) and not bool(split_group_sets.get("test", set()) & (split_group_sets.get("train", set()) | split_group_sets.get("validation", set()))),
        "source_holdout_overlap": len(source_holdout_groups) if source_holdout_groups else 0,
        "source_holdout_status": "not_applicable_single_public_model_restricted_scope",
        "real_observation_linked": False,
        "pharmacological_intervention_supported": False,
        "claim_scope": str(domain["claim_scope"]),
        "selected_domain": str(domain["domain_uid"]),
        "baseline_challenge_allowed": False,
        "manifests": {
            "action_catalog": "Input/manifests/c2/action_catalog.tsv",
            "trajectory_manifest": "Input/manifests/c2/trajectory_manifest.tsv",
            "matched_control_manifest": "Input/manifests/c2/matched_control_manifest.tsv",
            "strict_split_manifest": "Input/manifests/c2/strict_split_manifest.tsv",
        },
        "producer": {
            "source_model_id": str(domain["source_model_id"]),
            "source_accession": str(domain["source_accession"]),
            "source_archive_sha256": source_hash,
            "execution_uid": execution_uid,
            "trajectory_store": str(store_path.resolve().relative_to(REPO_ROOT)),
            "trajectory_store_sha256": store_hash,
            "production_store_validation": production_store_validation,
        },
        "next_action": "qualify_restricted_c2_baselines_after_manifest_review",
    }
    write_json(resolve_repo_path("Output/audits/c2_domain_readiness.json"), readiness)
    write_json(resolve_repo_path("Output/audits/c2_selected_domain.json"), readiness)
    integrity = verify_existing_domain()
    readiness["integrity_audit"] = "Output/audits/c2/current_clamp/integrity_audit.json"
    readiness["integrity_status"] = integrity["status"]
    write_json(resolve_repo_path("Output/audits/c2_domain_readiness.json"), readiness)
    write_json(resolve_repo_path("Output/audits/c2_selected_domain.json"), readiness)
    return readiness


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the source-backed restricted C2 current-clamp domain.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Rebuild the C2 integrity audit from the current store and manifests without executing the producer.",
    )
    args = parser.parse_args()
    if args.verify_existing:
        audit = verify_existing_domain()
        print(json.dumps({"status": audit["status"], "store_hash": audit["store_hash"]}, sort_keys=True))
        return
    config = load_config(args.config)
    result = execute_domain(config)
    print(json.dumps({"status": result["status"], "selected_domain": result["selected_domain"], "strict_split_overlap": result["strict_split_overlap"]}, sort_keys=True))


if __name__ == "__main__":
    main()
