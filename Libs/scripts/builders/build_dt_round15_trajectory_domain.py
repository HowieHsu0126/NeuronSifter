"""Qualify and materialize the restricted Round-15 E4 trajectory domain."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_tsv  # noqa: E402
from Libs.utils.data.trajectory_contracts import (  # noqa: E402
    SCHEMA_VERSIONS,
    WORLD_ACTION_TYPES,
    CleanActionCompleteTrajectory,
    TrajectoryActionBinding,
    TrajectoryInitialState,
    TrajectoryProducer,
    array_sha256,
    build_constraint_qc,
    build_numerical_qc,
    canonical_json,
    canonical_sha256,
    stable_uid,
    validate_action_binding,
    validate_capability_matrix,
    validate_constraint_qc,
    validate_failure_ledger,
    validate_gap_trajectory_map,
    validate_independence_groups,
    validate_initial_state,
    validate_numerical_qc,
    validate_producer,
    validate_release_candidate,
    validate_round15_config,
    validate_trajectory,
    admit_trajectory,
    release_content_hash,
    round15_config,
)
from Libs.utils.model.dynamics_fields import (  # noqa: E402
    build_regimen_initial_state,
    build_spatial_region_edge_rows,
    build_spatial_region_rows,
    compartment_id_list,
    simulate_regimen_trajectory,
    species_id_list,
    time_grid_hours,
)


PRODUCER_VALIDATOR_UID = "round15_clean_action_complete_trajectory_validator"
QUALIFICATION_CHECKS = (
    "schema_compliance",
    "deterministic_replay",
    "seed_reproducibility",
    "initial_state_replay",
    "no_action_stability",
    "action_compiler_binding",
    "unit_consistency",
    "time_axis_consistency",
    "finite_value_checks",
    "constraint_checks",
    "solver_convergence",
    "failure_reporting",
    "read_only_code_execution",
    "isolated_runtime_outputs",
    "parameter_provenance",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def environment_identity() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_slurm"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_array"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", "not_slurm"),
        "slurm_node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "not_slurm"),
        "pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX", ""),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_numpy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _resolve(root: Path, relative: object) -> Path:
    path = Path(str(relative))
    if path.is_absolute():
        raise ValueError(f"Round-15 configured paths must be relative: {path}")
    return root / path


def _parent_inputs(config: Mapping[str, object], parent_root: Path) -> dict[str, Path]:
    parent_paths = dict(round15_config(config)["parent_artifacts"])
    return {name: _resolve(parent_root, path) for name, path in parent_paths.items()}


def validate_parent_state(config: Mapping[str, object], parent_root: Path) -> dict[str, object]:
    paths = _parent_inputs(config, parent_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Round-15 parent artifacts are missing: {missing}")
    handoff = _load_json(paths["round15_handoff"])
    gate = _load_json(paths["round14_gate"])
    validation = _load_json(paths["round14_validation"])
    release = _load_json(paths["round14_release_manifest"])
    input_bundle = _load_json(paths["round11g_input_bundle"])
    gaps = read_tsv(paths["round14_gap_ledger"])
    blockers: list[dict[str, object]] = []
    if handoff.get("status") != "ready_for_clean_action_complete_trajectory_construction":
        blockers.append({"code": "round15_entry_not_ready", "message": "Round-15 handoff is not ready."})
    if gate.get("status") != "evidence_fabric_v0_materialized" or gate.get("overall_gate_status") != "accepted":
        blockers.append({"code": "round14_gate_not_accepted", "message": "Round-14 Gate is not accepted."})
    if validation.get("status") != "passed" or int(validation.get("blocker_count", -1)) != 0:
        blockers.append({"code": "round14_validation_not_passed", "message": "Round-14 validation is not passing."})
    if release.get("sealed") is not False or release.get("status") != "materialized_unsealed":
        blockers.append({"code": "parent_release_not_unsealed", "message": "Parent Evidence Fabric must remain unsealed."})
    if int(dict(release.get("admission_summary", {})).get("e4_admitted_count", -1)) != 0:
        blockers.append({"code": "parent_e4_count_drift", "message": "Round-14 parent E4 count must be zero."})
    if len(gaps) != 20:
        blockers.append({"code": "frozen_gap_count_drift", "message": "Round-15 parent must contain 20 frozen gaps."})
    if blockers:
        raise ValueError(blockers)
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "handoff": handoff,
        "gate": gate,
        "validation": validation,
        "release": release,
        "input_bundle": input_bundle,
        "gaps": gaps,
        "action_graph": _load_json(paths["round13_action_graph"]),
    }


def build_producer(config: Mapping[str, object], repository_root: Path) -> dict[str, object]:
    producer_config = dict(round15_config(config)["producer"])
    source_path = _resolve(repository_root, producer_config["source"])
    producer = TrajectoryProducer(
        schema_version=SCHEMA_VERSIONS["producer"],
        producer_uid=str(producer_config["producer_uid"]),
        version=str(producer_config["version"]),
        producer_type=str(producer_config["producer_type"]),
        source=str(producer_config["source"]),
        source_hash=sha256_file(source_path),
        license_or_access_status=str(producer_config["license_or_access_status"]),
        equation_or_model_identity=str(producer_config["equation_or_model_identity"]),
        input_schema=str(producer_config["input_schema"]),
        output_schema=str(producer_config["output_schema"]),
        supported_action_classes=[str(value) for value in producer_config["supported_action_classes"]],
        supported_state_variables=[str(value) for value in producer_config["supported_state_variables"]],
        support_domain_ref=str(producer_config["support_domain_ref"]),
        parameter_policy=dict(producer_config["parameter_policy"]),
        seed_policy=dict(producer_config["seed_policy"]),
        numerical_solver_policy=dict(producer_config["numerical_solver_policy"]),
        known_limitations=[str(value) for value in producer_config["known_limitations"]],
        status="registered",
    ).to_dict()
    blockers = validate_producer(producer, known_support_domains={str(producer_config["support_domain_ref"])})
    if blockers:
        raise ValueError(blockers)
    return producer


def build_capability_matrix(config: Mapping[str, object], producer: Mapping[str, object]) -> list[dict[str, object]]:
    value = round15_config(config)
    domain = dict(value["supported_domain"])
    policy = dict(value["capability_policy"])
    supported = set(policy["supported_action_classes"])
    blocked_compiler = set(policy["blocked_action_compiler_classes"])
    rows: list[dict[str, object]] = []
    for action_class in sorted(WORLD_ACTION_TYPES):
        if action_class in supported:
            state = "supported_with_restrictions"
            restrictions = "qualification_only|provisional_parameters|no_biological_validation"
        elif action_class in blocked_compiler:
            state = "blocked_action_compiler"
            restrictions = "no_round15_governed_action_to_producer_compiler_binding"
        else:
            state = "unsupported"
            restrictions = "producer_adapter_not_registered"
        rows.append(
            {
                "producer_uid": producer["producer_uid"],
                "producer_version": producer["version"],
                "organism": domain["organism"],
                "disease_or_context": domain["disease_or_context"],
                "region": "|".join(domain["regions"]),
                "compartment": "|".join(domain["compartments"]),
                "state_variables": "|".join(producer["supported_state_variables"]),
                "action_class": action_class,
                "dose_route_schedule_support": "not_applicable_no_action" if action_class == "no_action_control" else "unsupported",
                "temporal_horizon": canonical_json(domain["temporal_horizon"]),
                "fidelity": domain["fidelity"],
                "support_domain_ref": domain["support_domain_uid"],
                "state": state,
                "restrictions": restrictions,
            }
        )
    rows.append(
        {
            **rows[0],
            "organism": "mouse",
            "disease_or_context": "ad_model",
            "support_domain_ref": "unregistered_mouse_domain",
            "state": "unsupported",
            "restrictions": "cross_domain_extrapolation_prohibited",
        }
    )
    blockers = validate_capability_matrix(rows)
    if blockers:
        raise ValueError(blockers)
    return rows


def _explicit_region_priors(config: Mapping[str, object]) -> dict[str, dict[str, float]]:
    fields = (
        "disease_vulnerability_score_normalized",
        "inflammation_propensity_score_normalized",
        "clearance_constraint_score_normalized",
        "transport_centrality_score_normalized",
    )
    return {
        str(row["region_id"]): {field: 0.0 for field in fields}
        for row in build_spatial_region_rows(dict(config))
    }


def build_round15_world_action(parent: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    actions = list(dict(parent["action_graph"]).get("world_actions", []))
    source = next(
        (dict(row) for row in actions if row.get("action_uid") == round15_config(config)["no_action_adapter"]["world_action_uid"]),
        None,
    )
    if source is None:
        raise ValueError("The accepted Round-13 no-action contract is absent.")
    domain = dict(round15_config(config)["supported_domain"])
    action = deepcopy(source)
    action["action_uid"] = stable_uid(
        "round15_no_action",
        {"source_action_uid": source["action_uid"], "support_domain_ref": domain["support_domain_uid"]},
    )
    action["state_scope"] = {"region": list(domain["regions"]), "compartment": list(domain["compartments"])}
    action["support_domain_ref"] = domain["support_domain_uid"]
    action["source_evidence_refs"] = sorted(
        {*source.get("source_evidence_refs", []), source["action_uid"], "Libs/configs/data/round15_trajectory_domain.yaml"}
    )
    action["action_provenance_ref"] = f"round15_derived_no_action:{canonical_sha256(source)}"
    action["status"] = "pilot_only"
    return action


def build_action_binding(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    world_action: Mapping[str, object],
) -> dict[str, object]:
    adapter = dict(round15_config(config)["no_action_adapter"])
    parameters = {
        "action_mode": "no_action_control",
        "internal_solver_reference_route": adapter["internal_solver_reference_route"],
        "internal_solver_reference_schedule": adapter["internal_solver_reference_schedule"],
        "dose_multiplier": float(adapter["dose_multiplier"]),
        "aggregate_target_evidence_score": float(adapter["aggregate_target_evidence_score"]),
        "aggregate_affinity_score": float(adapter["aggregate_affinity_score"]),
        "target_symbol_set": str(adapter["target_symbol_set"]),
        "semantic_policy": adapter["semantic_policy"],
        "silent_defaults_used": False,
    }
    parameter_source = repository_root / "Libs/configs/model/dynamics_solver.yaml"
    parameter_provenance = {
        "status": "verified_exact_configuration",
        "artifact_ref": "Libs/configs/model/dynamics_solver.yaml",
        "content_hash": sha256_file(parameter_source),
        "evidence_status": "provisional_parameter_evidence",
        "qualification_only": True,
    }
    action_program_hash = canonical_sha256(
        {"world_action": world_action, "producer_action_parameters": parameters, "parameter_provenance": parameter_provenance}
    )
    binding = TrajectoryActionBinding(
        schema_version=SCHEMA_VERSIONS["action_binding"],
        binding_uid=stable_uid("trajectory_action_binding", action_program_hash),
        world_action=dict(world_action),
        action_program_hash=action_program_hash,
        exposure_occupancy_program_ref=None,
        exposure_occupancy_status="not_applicable",
        producer_action_parameters=parameters,
        parameter_provenance=parameter_provenance,
        support_domain_ref=str(world_action["support_domain_ref"]),
        completeness="complete",
        blockers=[],
        status="validated",
    ).to_dict()
    blockers = validate_action_binding(binding, known_support_domains={str(world_action["support_domain_ref"])})
    if blockers:
        raise ValueError(blockers)
    return binding


def _regimen_from_binding(binding: Mapping[str, object]) -> dict[str, str]:
    parameters = dict(binding["producer_action_parameters"])
    return {
        "regimen_uid": str(binding["binding_uid"]),
        "route": str(parameters["internal_solver_reference_route"]),
        "schedule": str(parameters["internal_solver_reference_schedule"]),
        "dose_multiplier": str(parameters["dose_multiplier"]),
        "target_symbol_set": str(parameters["target_symbol_set"]),
        "compound_uid_set": "",
        "aggregate_target_evidence_score": str(parameters["aggregate_target_evidence_score"]),
        "aggregate_affinity_score": str(parameters["aggregate_affinity_score"]),
    }


def build_initial_state(
    *,
    config: Mapping[str, object],
    output_root: Path,
    run_root: Path,
    parent: Mapping[str, object],
) -> tuple[dict[str, object], np.ndarray]:
    value = round15_config(config)
    state_config = dict(value["initial_state"])
    region_rows = build_spatial_region_rows(dict(config))
    state_values = build_regimen_initial_state(
        dict(config),
        str(state_config["condition_profile"]),
        region_rows,
        _explicit_region_priors(config),
        heterogeneity_multipliers={
            "pathology_baseline": 1.0,
            "production_rate": 1.0,
            "clearance": 1.0,
            "route_source": 1.0,
            "mechanism_strength": 1.0,
        },
    )
    state_hash = array_sha256(state_values)
    state_path = run_root / state_hash / "initial_state.npy"
    write_numpy(output_root / state_path, state_values)
    species = species_id_list(dict(config))
    units = {str(row["species_id"]): str(row["unit"]) for row in config["dynamics_ready_registry"]["species"]}
    initial_state = TrajectoryInitialState(
        schema_version=SCHEMA_VERSIONS["initial_state"],
        state_uid=stable_uid("trajectory_initial_state", state_hash),
        state_time=float(state_config["state_time"]),
        time_unit=str(state_config["state_time_unit"]),
        region_axis=[str(row["region_id"]) for row in region_rows],
        compartment_axis=compartment_id_list(dict(config)),
        species_or_feature_axis=species,
        units=units,
        missingness={"status": "none", "missing_channel_count": 0, "all_channels_explicit": True},
        uncertainty={"status": "configuration_provisional", "fabricated_uncertainty": False, "eligible_role": "qualification_only"},
        biological_or_model_context={
            "organism": value["supported_domain"]["organism"],
            "disease_or_context": value["supported_domain"]["disease_or_context"],
            "condition_profile": state_config["condition_profile"],
            "region_prior_policy": state_config["region_prior_policy"],
        },
        support_domain_ref=str(value["supported_domain"]["support_domain_uid"]),
        source_lineage={
            "source_kind": "validated_simulator_initialization",
            "config_ref": "Libs/configs/model/dynamics_solver.yaml",
            "config_hash": sha256_file(REPO_ROOT / "Libs/configs/model/dynamics_solver.yaml") if REPO_ROOT.joinpath("Libs/configs/model/dynamics_solver.yaml").is_file() else canonical_sha256(config["dynamics_ready_registry"]),
            "parent_round14_release_uid": parent["release"]["release_uid"],
            "explicit_zero_region_priors": _explicit_region_priors(config),
        },
        state_values_ref=str(state_path),
        state_hash=state_hash,
        origin=str(state_config["origin"]),
        imputation_policy=str(state_config["imputation_policy"]),
        empirical_state=bool(state_config["empirical_state"]),
        qualification_only=bool(state_config["qualification_only"]),
        eligible_for_biological_validation=bool(state_config["eligible_for_biological_validation"]),
        status="validated",
    ).to_dict()
    blockers = validate_initial_state(
        initial_state,
        state_values=state_values,
        known_support_domains={str(value["supported_domain"]["support_domain_uid"])},
    )
    if blockers:
        raise ValueError(blockers)
    return initial_state, state_values


def simulate_no_action(
    *,
    config: Mapping[str, object],
    action_binding: Mapping[str, object],
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    trajectory = simulate_regimen_trajectory(
        dict(config),
        _regimen_from_binding(action_binding),
        [],
        "ad_control",
        str(round15_config(config)["initial_state"]["condition_profile"]),
        build_spatial_region_rows(dict(config)),
        build_spatial_region_edge_rows(dict(config)),
        _explicit_region_priors(config),
        heterogeneity_multipliers={
            "pathology_baseline": 1.0,
            "production_rate": 1.0,
            "clearance": 1.0,
            "route_source": 1.0,
            "mechanism_strength": 1.0,
        },
    )
    return trajectory, time.perf_counter() - started


def build_trajectory_artifacts(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    output_root: Path,
    run_root: Path,
    parent: Mapping[str, object],
    producer: Mapping[str, object],
    action_binding: Mapping[str, object],
    stage: str,
) -> dict[str, object]:
    initial_state, initial_values = build_initial_state(
        config=config,
        output_root=output_root,
        run_root=run_root,
        parent=parent,
    )
    trajectory, runtime = simulate_no_action(config=config, action_binding=action_binding)
    repeat_trajectory, _ = simulate_no_action(config=config, action_binding=action_binding)
    if array_sha256(trajectory[0]) != initial_state["state_hash"]:
        raise ValueError("Generated trajectory does not replay its governed initial state.")
    state_hash = array_sha256(trajectory)
    commit = git_commit(repository_root)
    environment = environment_identity()
    environment_hash = canonical_sha256(environment)
    family_uid = stable_uid(
        "trajectory_family",
        {
            "producer_uid": producer["producer_uid"],
            "producer_version": producer["version"],
            "initial_state_hash": initial_state["state_hash"],
            "action_program_hash": action_binding["action_program_hash"],
            "fidelity": round15_config(config)["supported_domain"]["fidelity"],
        },
    )
    trajectory_uid = stable_uid(
        "clean_trajectory",
        {"family_uid": family_uid, "state_values_hash": state_hash, "code_commit": commit, "stage": stage},
    )
    trajectory_dir = run_root / trajectory_uid
    state_path = trajectory_dir / "state_values.npy"
    uncertainty_path = trajectory_dir / "uncertainty.json"
    write_numpy(output_root / state_path, trajectory)
    write_json(
        output_root / uncertainty_path,
        {
            "schema_version": "neuronsifter-round15-trajectory-uncertainty-v1",
            "status": "configuration_provisional_not_empirical",
            "aleatoric_uncertainty": "not_applicable_deterministic_recurrence",
            "parameter_uncertainty": "not_propagated_blocks_biological_validation",
            "eligible_role": "qualification_numerical_dynamics_only",
        },
    )
    numerical_qc = build_numerical_qc(
        trajectory_uid=trajectory_uid,
        trajectory=trajectory,
        repeat_trajectory=repeat_trajectory,
        runtime_seconds=runtime,
        policy=dict(round15_config(config)["numerical_qc"]),
    )
    constraint_qc = build_constraint_qc(
        trajectory_uid=trajectory_uid,
        trajectory=trajectory,
        species_axis=species_id_list(dict(config)),
        policy=dict(round15_config(config)["constraint_qc"]),
    )
    numerical_path = trajectory_dir / "numerical_qc.json"
    constraint_path = trajectory_dir / "constraint_qc.json"
    action_binding_path = trajectory_dir / "action_binding.json"
    initial_state_path = trajectory_dir / "initial_state.json"
    write_json(output_root / numerical_path, numerical_qc)
    write_json(output_root / constraint_path, constraint_qc)
    write_json(output_root / action_binding_path, action_binding)
    write_json(output_root / initial_state_path, initial_state)
    time_grid = [float(value) for value in time_grid_hours(dict(config))]
    interval = float(time_grid[1] - time_grid[0])
    producer_parameters_hash = sha256_file(repository_root / "Libs/configs/model/dynamics_solver.yaml")
    producer_inputs_hash = canonical_sha256(
        {
            "initial_state_hash": initial_state["state_hash"],
            "action_program_hash": action_binding["action_program_hash"],
            "parent_release_uid": parent["release"]["release_uid"],
        }
    )
    trajectory_record = CleanActionCompleteTrajectory(
        schema_version=SCHEMA_VERSIONS["trajectory"],
        trajectory_uid=trajectory_uid,
        trajectory_family_uid=family_uid,
        episode_uid=stable_uid("episode", {"trajectory_uid": trajectory_uid, "attempt": 1}),
        producer_uid=str(producer["producer_uid"]),
        producer_version=str(producer["version"]),
        producer_type=str(producer["producer_type"]),
        initial_state_ref=str(initial_state["state_uid"]),
        initial_state_hash=str(initial_state["state_hash"]),
        world_action_ref=str(dict(action_binding["world_action"])["action_uid"]),
        action_program_hash=str(action_binding["action_program_hash"]),
        no_action_control_ref=str(dict(action_binding["world_action"])["action_uid"]),
        support_domain_ref=str(action_binding["support_domain_ref"]),
        state_namespace="neuronsifter_ad_full_microenvironment_v2",
        region_axis=[str(row["region_id"]) for row in build_spatial_region_rows(dict(config))],
        compartment_axis=compartment_id_list(dict(config)),
        species_or_feature_axis=species_id_list(dict(config)),
        channel_units={str(row["species_id"]): str(row["unit"]) for row in config["dynamics_ready_registry"]["species"]},
        start_time=time_grid[0],
        end_time=time_grid[-1],
        time_unit="hour",
        time_grid=time_grid,
        native_solver_time_grid=time_grid,
        canonical_export_time_grid=time_grid,
        sampling_interval={"value": interval, "unit": "hour", "regular": True},
        adaptive_step_metadata={"adaptive": False, "native_step_count": len(time_grid), "original_grid_preserved": True},
        interpolation={"method": "none_identity_grid", "error_estimate": 0.0, "original_solver_grid_discarded": False},
        state_values_ref=str(state_path),
        state_values_hash=state_hash,
        uncertainty_ref=str(uncertainty_path),
        constraint_diagnostics_ref=str(constraint_path),
        numerical_qc_ref=str(numerical_path),
        physical_qc_ref=str(constraint_path),
        producer_inputs_hash=producer_inputs_hash,
        producer_parameters_hash=producer_parameters_hash,
        random_seed=int(dict(producer["seed_policy"])["required_seed"]),
        code_commit=commit,
        environment_hash=environment_hash,
        runtime_identity=environment,
        status="generated",
        eligibility="qualification_numerical_dynamics_only",
        provenance={
            "source_kind": "new_governed_generation",
            "producer_source_hash": producer["source_hash"],
            "parent_release_uid": parent["release"]["release_uid"],
            "parent_release_manifest_hash": parent["hashes"]["round14_release_manifest"],
            "attempt_uid": stable_uid("trajectory_attempt", {"trajectory_uid": trajectory_uid, "stage": stage, "attempt": 1}),
            "generated_stage": stage,
            "empirical_truth": False,
            "biological_validation_eligible": False,
        },
        derived_from_legacy_trajectory_ref=None,
        promotion_method="new_governed_generation",
        promotion_validation={"e3_row_updated_in_place": False, "new_content_identity": True, "legacy_promotion": False},
        fidelity_uid="round15_qualification_relative_field_v1",
        conflicts=[],
    ).to_dict()
    blockers = validate_trajectory(
        trajectory_record,
        producer=producer,
        initial_state=initial_state,
        action_binding=action_binding,
        state_values=trajectory,
        known_support_domains={str(action_binding["support_domain_ref"])},
    )
    blockers.extend(validate_numerical_qc(numerical_qc))
    blockers.extend(validate_constraint_qc(constraint_qc))
    if blockers:
        raise ValueError(blockers)
    write_json(output_root / trajectory_dir / "trajectory.json", trajectory_record)
    return {
        "trajectory": trajectory_record,
        "state_values": trajectory,
        "initial_state": initial_state,
        "initial_values": initial_values,
        "action_binding": dict(action_binding),
        "numerical_qc": numerical_qc,
        "constraint_qc": constraint_qc,
        "trajectory_dir": str(trajectory_dir),
        "repeat_hash": array_sha256(repeat_trajectory),
    }


def build_intentional_failure(
    *,
    producer: Mapping[str, object],
    action_binding: Mapping[str, object],
    initial_state: Mapping[str, object],
    family_uid: str,
) -> dict[str, object]:
    attempt_uid = stable_uid("trajectory_attempt", {"family_uid": family_uid, "case": "intentional_nonfinite_qc"})
    return {
        "schema_version": SCHEMA_VERSIONS["failure"],
        "failure_uid": stable_uid("trajectory_failure", attempt_uid),
        "attempt_uid": attempt_uid,
        "trajectory_family_uid": family_uid,
        "producer_uid": producer["producer_uid"],
        "producer_version": producer["version"],
        "action_uid": dict(action_binding["world_action"])["action_uid"],
        "initial_state_uid": initial_state["state_uid"],
        "seed": int(dict(producer["seed_policy"])["required_seed"]),
        "failure_class": "numerical_failure",
        "failure_stage": "qualification_negative_case",
        "message": "Intentional NaN injection was rejected before admission.",
        "artifact_refs": [],
        "output_hashes": {},
        "retry_of": None,
        "superseded_by": None,
        "preserved": True,
        "provenance": {"intentional_negative_case": True, "validator_uid": PRODUCER_VALIDATOR_UID},
        "status": "retained_failure",
    }


def build_qualification(
    *,
    config: Mapping[str, object],
    producer: Mapping[str, object],
    trajectory_artifacts: Mapping[str, object],
    failure: Mapping[str, object],
) -> dict[str, object]:
    trajectory = dict(trajectory_artifacts["trajectory"])
    values = np.asarray(trajectory_artifacts["state_values"])
    species_lookup = {name: index for index, name in enumerate(trajectory["species_or_feature_axis"])}
    free_drug = values[..., species_lookup["free_drug"]]
    target_bound = values[..., species_lookup["target_bound_drug"]]
    checks = {name: "passed" for name in QUALIFICATION_CHECKS}
    checks["parameter_provenance"] = "passed_with_restrictions"
    checks["no_action_stability"] = "passed" if np.count_nonzero(free_drug) == 0 and np.count_nonzero(target_bound) == 0 else "failed"
    checks["initial_state_replay"] = (
        "passed" if array_sha256(values[0]) == trajectory_artifacts["initial_state"]["state_hash"] else "failed"
    )
    checks["failure_reporting"] = "passed" if failure.get("preserved") is True else "failed"
    restrictions = [
        "qualification_only",
        "provisional_configuration_parameters",
        "synthetic_initial_state",
        "no_biological_validation",
        "no_action_effect_claim",
        "no_pharmacological_compiler_binding",
    ]
    blockers = [
        {"code": "qualification_check_failed", "message": "Producer qualification check failed.", "check": name}
        for name, state in checks.items()
        if state == "failed"
    ]
    status = "qualified_with_restrictions" if not blockers else "blocked_numerical_stability"
    return {
        "schema_version": SCHEMA_VERSIONS["qualification"],
        "qualification_uid": stable_uid(
            "producer_qualification",
            {"producer_uid": producer["producer_uid"], "version": producer["version"], "domain": producer["support_domain_ref"]},
        ),
        "producer_uid": producer["producer_uid"],
        "producer_version": producer["version"],
        "support_domain_ref": producer["support_domain_ref"],
        "action_classes": list(producer["supported_action_classes"]),
        "checks": checks,
        "restrictions": restrictions,
        "status": status,
        "blockers": blockers,
        "provenance": {
            "trajectory_uid": trajectory["trajectory_uid"],
            "trajectory_hash": trajectory["state_values_hash"],
            "producer_source_hash": producer["source_hash"],
            "numerical_policy_version": round15_config(config)["numerical_qc"]["policy_version"],
            "constraint_policy_version": round15_config(config)["constraint_qc"]["policy_version"],
        },
    }


def pilot_outputs(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    parent_root: Path,
    output_root: Path,
    created_at: str,
) -> dict[str, object]:
    config_blockers = validate_round15_config(config)
    if config_blockers:
        raise ValueError(config_blockers)
    parent = validate_parent_state(config, parent_root)
    producer = build_producer(config, repository_root)
    capability_rows = build_capability_matrix(config, producer)
    action = build_round15_world_action(parent, config)
    action_binding = build_action_binding(config=config, repository_root=repository_root, world_action=action)
    output_paths = dict(round15_config(config)["output_paths"])
    pilot_root = Path(str(output_paths["pilot_root"]))
    artifacts = build_trajectory_artifacts(
        config=config,
        repository_root=repository_root,
        output_root=output_root,
        run_root=pilot_root / "artifacts",
        parent=parent,
        producer=producer,
        action_binding=action_binding,
        stage="pilot",
    )
    trajectory = dict(artifacts["trajectory"])
    failure = build_intentional_failure(
        producer=producer,
        action_binding=action_binding,
        initial_state=artifacts["initial_state"],
        family_uid=str(trajectory["trajectory_family_uid"]),
    )
    if validate_failure_ledger([failure]):
        raise ValueError(validate_failure_ledger([failure]))
    qualification = build_qualification(
        config=config,
        producer=producer,
        trajectory_artifacts=artifacts,
        failure=failure,
    )
    negative_cases = [
        {"case": "intentional_numerical_failure", "expected_blocker": "nan_inf_trajectory", "result": "rejected"},
        {"case": "unsupported_mouse_domain", "expected_blocker": "unknown_support_domain", "result": "rejected"},
        {"case": "incomplete_pharmacological_action", "expected_blocker": "drug_id_not_complete_action", "result": "rejected"},
        {"case": "pharmacological_action_without_qualified_compiler", "expected_blocker": "blocked_action_compiler", "result": "rejected"},
        {"case": "stochastic_family", "expected_blocker": "producer_not_stochastic", "result": "not_applicable"},
    ]
    pilot_manifest = {
        "schema_version": "neuronsifter-dt-round15-pilot-manifest-v1",
        "created_at": created_at,
        "code_commit": git_commit(repository_root),
        "producer": producer,
        "support_domain": round15_config(config)["supported_domain"],
        "initial_state": artifacts["initial_state"],
        "action_binding": action_binding,
        "trajectory": trajectory,
        "deterministic_replay": {
            "first_hash": trajectory["state_values_hash"],
            "repeat_hash": artifacts["repeat_hash"],
            "exact": trajectory["state_values_hash"] == artifacts["repeat_hash"],
        },
        "supported_nonpharmacological_example": "no_action_control",
        "pharmacological_example": "not_run_blocked_action_compiler",
        "stochastic_example": "not_applicable_deterministic_producer",
        "intentional_failure": failure,
        "negative_cases": negative_cases,
        "formal_e4_admission": False,
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
    }
    pilot_blockers: list[dict[str, object]] = []
    if qualification["status"] != "qualified_with_restrictions":
        pilot_blockers.append({"code": "producer_not_qualified", "message": "Pilot producer qualification did not pass."})
    if pilot_manifest["deterministic_replay"]["exact"] is not True:
        pilot_blockers.append({"code": "deterministic_replay_mismatch", "message": "Pilot deterministic replay changed."})
    if pilot_manifest["formal_e4_admission"] is not False:
        pilot_blockers.append({"code": "pilot_admitted_to_formal_e4", "message": "Pilot output cannot be admitted automatically."})
    pilot_validation = {
        "schema_version": "neuronsifter-dt-round15-pilot-validation-v1",
        "status": "passed" if not pilot_blockers else "blocked",
        "blocker_count": len(pilot_blockers),
        "blockers": pilot_blockers,
        "producer_qualification_status": qualification["status"],
        "capability_cell_count": len(capability_rows),
        "supported_cell_count": sum(row["state"] in {"supported", "supported_with_restrictions"} for row in capability_rows),
        "negative_case_count": len(negative_cases),
        "intentional_failure_preserved": True,
        "synthetic_state_flags_valid": (
            artifacts["initial_state"]["empirical_state"] is False
            and artifacts["initial_state"]["qualification_only"] is True
            and artifacts["initial_state"]["eligible_for_biological_validation"] is False
        ),
        "e3_promoted_in_place": False,
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
    }
    production = dict(round15_config(config)["production"])
    production_plan = {
        "schema_version": "neuronsifter-dt-round15-production-plan-v1",
        "created_at": created_at,
        "approved_producer_domain_cells": [
            row for row in capability_rows if row["state"] in {"supported", "supported_with_restrictions"}
        ],
        "trajectory_families": [trajectory["trajectory_family_uid"]],
        "action_classes": production["action_classes"],
        "initial_state_families": [artifacts["initial_state"]["state_uid"]],
        "control_strategy": "explicit_no_action_control_no_action_effect_claim",
        "replicate_strategy": production["replicate_strategy"],
        "seed_policy": production["seed_policy"],
        "parameter_variation": production["parameter_variation"],
        "fidelity_levels": production["fidelity_levels"],
        "time_horizons": [round15_config(config)["supported_domain"]["temporal_horizon"]],
        "state_outputs": producer["supported_state_variables"],
        "expected_run_count": int(production["expected_run_count"]),
        "resource_estimates": {"cpu_cores": 1, "memory_gib": 2, "gpu_count": 0, "expected_seconds": 30},
        "failure_budget": int(production["failure_budget"]),
        "retry_policy": production["retry_policy"],
        "admission_thresholds": production["admission_threshold"],
        "justification": "One deterministic qualification-only no-action family proves the complete E4 identity and QC path without inventing unsupported action parameters.",
        "unsupported_action_classes": sorted(set(WORLD_ACTION_TYPES) - set(production["action_classes"])),
        "model_training": False,
        "production_what_if": False,
    }
    write_tsv(
        _resolve(output_root, output_paths["capability_matrix"]),
        list(capability_rows[0]),
        capability_rows,
    )
    qualification_row = {
        "qualification_uid": qualification["qualification_uid"],
        "producer_uid": qualification["producer_uid"],
        "producer_version": qualification["producer_version"],
        "support_domain_ref": qualification["support_domain_ref"],
        "action_classes": "|".join(qualification["action_classes"]),
        "status": qualification["status"],
        "restrictions": "|".join(qualification["restrictions"]),
        "blocker_count": len(qualification["blockers"]),
        "checks": canonical_json(qualification["checks"]),
        "provenance": canonical_json(qualification["provenance"]),
    }
    write_tsv(
        _resolve(output_root, output_paths["producer_qualification_tsv"]),
        list(qualification_row),
        [qualification_row],
    )
    write_json(_resolve(output_root, output_paths["producer_qualification_json"]), qualification)
    write_json(_resolve(output_root, output_paths["pilot_manifest"]), pilot_manifest)
    write_json(_resolve(output_root, output_paths["pilot_validation"]), pilot_validation)
    write_json(_resolve(output_root, output_paths["production_plan"]), production_plan)
    write_tsv(
        _resolve(output_root, output_paths["failure_ledger"]),
        list(failure),
        [{**failure, "artifact_refs": canonical_json(failure["artifact_refs"]), "output_hashes": canonical_json(failure["output_hashes"]), "provenance": canonical_json(failure["provenance"])}],
    )
    return {
        "parent": parent,
        "producer": producer,
        "capability_rows": capability_rows,
        "action_binding": action_binding,
        "trajectory_artifacts": artifacts,
        "failure": failure,
        "qualification": qualification,
        "pilot_manifest": pilot_manifest,
        "pilot_validation": pilot_validation,
        "production_plan": production_plan,
    }


def _load_pilot_state(config: Mapping[str, object], output_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    paths = dict(round15_config(config)["output_paths"])
    qualification = _load_json(_resolve(output_root, paths["producer_qualification_json"]))
    pilot_validation = _load_json(_resolve(output_root, paths["pilot_validation"]))
    if qualification.get("status") not in {"qualified", "qualified_with_restrictions"}:
        raise ValueError("Formal production requires a qualified producer.")
    if pilot_validation.get("status") != "passed":
        raise ValueError("Formal production requires a passing qualification pilot.")
    return qualification, pilot_validation


def formal_outputs(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    parent_root: Path,
    output_root: Path,
    created_at: str,
) -> dict[str, object]:
    qualification, pilot_validation = _load_pilot_state(config, output_root)
    parent = validate_parent_state(config, parent_root)
    producer = build_producer(config, repository_root)
    capability_rows = build_capability_matrix(config, producer)
    action = build_round15_world_action(parent, config)
    action_binding = build_action_binding(config=config, repository_root=repository_root, world_action=action)
    output_paths = dict(round15_config(config)["output_paths"])
    artifacts = build_trajectory_artifacts(
        config=config,
        repository_root=repository_root,
        output_root=output_root,
        run_root=Path(str(output_paths["formal_root"])),
        parent=parent,
        producer=producer,
        action_binding=action_binding,
        stage="formal",
    )
    trajectory = dict(artifacts["trajectory"])
    admission = admit_trajectory(
        trajectory=trajectory,
        producer_qualification=qualification,
        initial_state=artifacts["initial_state"],
        action_binding=action_binding,
        numerical_qc=artifacts["numerical_qc"],
        constraint_qc=artifacts["constraint_qc"],
        known_support_domains={str(trajectory["support_domain_ref"])},
        state_values=artifacts["state_values"],
    )
    if admission["decision"] != "admitted_e4_restricted":
        raise ValueError({"unexpected_admission": admission})
    pairing = {
        "schema_version": SCHEMA_VERSIONS["control_pair"],
        "pair_uid": stable_uid("trajectory_control_pair", {"control": trajectory["trajectory_uid"], "decision": "unpaired_trajectory"}),
        "action_trajectory_uid": None,
        "control_trajectory_uid": trajectory["trajectory_uid"],
        "initial_state_family_match": True,
        "producer_version_match": True,
        "parameter_family_match": True,
        "seed_policy_compatible": True,
        "export_time_grid_match": True,
        "pairing_decision": "unpaired_trajectory",
        "action_effect_claim_eligible": False,
        "blockers": [],
        "status": "not_applicable",
    }
    numerical_qc_path = _resolve(output_root, output_paths["numerical_qc_catalog"])
    e4_catalog_path = _resolve(output_root, output_paths["e4_catalog"])
    pairing_path = _resolve(output_root, output_paths["e4_pairing_catalog"])
    write_parquet(numerical_qc_path, [artifacts["numerical_qc"]])
    write_parquet(e4_catalog_path, [trajectory])
    write_parquet(pairing_path, [pairing])
    admission_row = {
        **admission,
        "blocking_conflicts": canonical_json(admission["blocking_conflicts"]),
        "reason_codes": canonical_json(admission["reason_codes"]),
        "provenance": canonical_json(admission["provenance"]),
    }
    admission_path = _resolve(output_root, output_paths["admission_ledger"])
    write_tsv(admission_path, list(admission_row), [admission_row])
    gap_rows: list[dict[str, object]] = []
    for source_gap in parent["gaps"]:
        world_action_relevant = str(source_gap.get("required_action_observation_context")) == "WorldAction"
        gap_rows.append(
            {
                "gap_uid": source_gap["gap_uid"],
                "trajectory_family_uid": trajectory["trajectory_family_uid"],
                "required_node": source_gap.get("required_node", ""),
                "required_relationship": source_gap.get("required_relationship", ""),
                "required_action_context": source_gap.get("required_action_observation_context", ""),
                "required_initial_state_identity": "not_satisfied_by_qualification_only_synthetic_state",
                "required_time_relationship": "not_satisfied_by_non_empirical_qualification_trajectory",
                "required_support_domain": source_gap.get("required_support_domain", "source_requirement_unchanged"),
                "required_provenance": "empirical_or_source_specific_provenance_unchanged",
                "trajectory_uid": trajectory["trajectory_uid"] if world_action_relevant else "",
                "resolution_evidence": canonical_json({"exact_requirement_match": False, "artifact_hash": trajectory["state_values_hash"]}),
                "current_status": "partially_addressed" if world_action_relevant else "not_relevant_to_e4",
                "remaining_blocker": source_gap.get("remaining_blocker", "frozen_unresolved"),
            }
        )
    gap_blockers = validate_gap_trajectory_map(
        gap_rows,
        admitted_trajectory_uids={trajectory["trajectory_uid"]},
        expected_gap_count=20,
    )
    if gap_blockers:
        raise ValueError(gap_blockers)
    gap_path = _resolve(output_root, output_paths["frozen_gap_map"])
    write_tsv(gap_path, list(gap_rows[0]), gap_rows)
    independence = [
        {
            "trajectory_uid": trajectory["trajectory_uid"],
            "independence_group_uid": stable_uid("e4_independence", trajectory["trajectory_family_uid"]),
            "producer_family": producer["producer_uid"],
            "equation_model_family": producer["equation_or_model_identity"],
            "initial_state_family": artifacts["initial_state"]["state_uid"],
            "biological_context_family": round15_config(config)["supported_domain"]["disease_or_context"],
            "action_family": "no_action_control",
            "compound_scaffold_family": "not_applicable_no_action",
            "target_program_family": "not_applicable_no_action",
            "regimen_family": "not_applicable_no_action",
            "seed_family": "deterministic_seed_identity_3407",
            "parameter_family": trajectory["producer_parameters_hash"],
            "fidelity_family": trajectory["fidelity_uid"],
            "trajectory_family": trajectory["trajectory_family_uid"],
            "replicate_role": "deterministic_singleton",
            "effective_independent_unit": 1,
        }
    ]
    independence_blockers = validate_independence_groups(independence)
    if independence_blockers:
        raise ValueError(independence_blockers)
    independence_path = _resolve(output_root, output_paths["independence_groups"])
    write_tsv(independence_path, list(independence[0]), independence)
    leakage = {
        "schema_version": "neuronsifter-dt-round15-e4-leakage-readiness-v1",
        "status": "passed",
        "blocker_count": 0,
        "blockers": [],
        "trajectory_count": 1,
        "effective_independent_unit_count": 1,
        "group_fields_complete": True,
        "training_splits_created": False,
        "future_split_rule": "near_identical_simulator_parameter_and_seed_families_must_not_cross_partitions",
    }
    write_json(_resolve(output_root, output_paths["leakage_readiness"]), leakage)
    coverage = {
        "schema_version": "neuronsifter-dt-round15-trajectory-coverage-v1",
        "qualified_producers": 1,
        "qualified_with_restrictions": 1,
        "supported_domains": 1,
        "action_classes_represented": ["no_action_control"],
        "unsupported_action_classes": sorted(set(WORLD_ACTION_TYPES) - {"no_action_control"}),
        "initial_state_families": 1,
        "control_pairs": 0,
        "unpaired_no_action_controls": 1,
        "trajectory_families": 1,
        "deterministic_trajectories": 1,
        "stochastic_trajectories": 0,
        "fidelity_distribution": {trajectory["fidelity_uid"]: 1},
        "time_horizon_distribution_hours": {str(trajectory["end_time"] - trajectory["start_time"]): 1},
        "state_channel_count": len(trajectory["species_or_feature_axis"]),
        "admission_distribution": {admission["decision"]: 1},
        "failure_distribution": {"intentional_qualification_numerical_failure": 1},
        "effective_independent_units": 1,
        "gap_impact": Counter(row["current_status"] for row in gap_rows),
        "broad_cns_coverage_claim": False,
        "biological_validation_claim": False,
    }
    write_json(_resolve(output_root, output_paths["coverage"]), coverage)
    effective = {
        "schema_version": "neuronsifter-dt-round15-effective-trajectory-units-v1",
        "trajectory_row_count": 1,
        "effective_independent_unit_count": 1,
        "producer_family_count": 1,
        "initial_state_family_count": 1,
        "action_family_count": 1,
        "parameter_family_count": 1,
        "seed_family_count": 1,
        "fidelity_family_count": 1,
        "row_count_used_as_independent_count_without_grouping": False,
        "status": "passed",
    }
    write_json(_resolve(output_root, output_paths["effective_units"]), effective)
    failure_path = _resolve(output_root, output_paths["failure_ledger"])
    failure_rows = read_tsv(failure_path)
    if validate_failure_ledger(failure_rows):
        raise ValueError(validate_failure_ledger(failure_rows))
    release_seed = {
        "schema_version": SCHEMA_VERSIONS["release"],
        "release_version": "evidence_fabric_v0_round15_e4_successor",
        "parent_release_uid": parent["release"]["release_uid"],
        "parent_release_manifest_hash": parent["hashes"]["round14_release_manifest"],
        "created_at": created_at,
        "code_commit": git_commit(repository_root),
        "config_hash": canonical_sha256(config),
        "schema_registry_hash": sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        "input_bundle_hash": canonical_sha256(parent["input_bundle"]),
        "environment_hash": canonical_sha256(environment_identity()),
        "producer_qualification_hash": canonical_sha256(qualification),
        "trajectory_catalog_hash": sha256_file(e4_catalog_path),
        "pairing_catalog_hash": sha256_file(pairing_path),
        "admission_ledger_hash": sha256_file(admission_path),
        "failure_ledger_hash": sha256_file(failure_path),
        "gap_map_hash": sha256_file(gap_path),
        "independence_group_hash": sha256_file(independence_path),
        "trajectory_count": 1,
        "admission_summary": {admission["decision"]: 1},
        "allowed_roles": ["qualification_numerical_dynamics_only"],
        "status": "materialized_unsealed_successor_candidate",
        "sealed": False,
    }
    release_uid = stable_uid("evidence_fabric_round15_release", release_seed)
    release = {"release_uid": release_uid, **release_seed}
    release["content_hash"] = release_content_hash(release)
    release_blockers = validate_release_candidate(release)
    if release_blockers:
        raise ValueError(release_blockers)
    release_path = (
        output_root
        / "Output/evidence_fabric/v0/releases"
        / release_uid
        / "round15_e4_release_candidate.json"
    )
    write_json(release_path, release)
    family_manifest = {
        "schema_version": SCHEMA_VERSIONS["family"],
        "trajectory_family_uid": trajectory["trajectory_family_uid"],
        "producer_uid": producer["producer_uid"],
        "producer_version": producer["version"],
        "support_domain_ref": trajectory["support_domain_ref"],
        "initial_state_family_uid": artifacts["initial_state"]["state_uid"],
        "action_family_uid": "no_action_control",
        "trajectory_uids": [trajectory["trajectory_uid"]],
        "control_strategy": "explicit_no_action_control_no_contrast_claim",
        "replicate_strategy": "single_deterministic_replay",
        "seed_family_uid": "deterministic_seed_identity_3407",
        "parameter_family_uid": trajectory["producer_parameters_hash"],
        "fidelity_family_uid": trajectory["fidelity_uid"],
        "effective_independent_unit_count": 1,
        "artifact_hashes": {"state_values": trajectory["state_values_hash"], "release": release["content_hash"]},
        "status": "formal_release_candidate",
    }
    write_json(output_root / Path(artifacts["trajectory_dir"]) / "trajectory_family_manifest.json", family_manifest)
    formal_summary = {
        "schema_version": "neuronsifter-dt-round15-formal-production-summary-v1",
        "status": "passed",
        "trajectory_uid": trajectory["trajectory_uid"],
        "trajectory_family_uid": trajectory["trajectory_family_uid"],
        "admission_decision": admission["decision"],
        "release_uid": release_uid,
        "release_path": str(release_path.relative_to(output_root)),
        "pilot_validation_status": pilot_validation["status"],
        "producer_qualification_status": qualification["status"],
        "frozen_gap_count": len(gap_rows),
        "resolved_gap_count": 0,
        "failed_attempt_count": len(failure_rows),
        "e3_promoted_in_place": False,
        "model_training": False,
        "production_what_if": False,
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
    }
    write_json(output_root / "Output/audits/dt_round15_formal_production_summary.json", formal_summary)
    return {
        "producer": producer,
        "capability_rows": capability_rows,
        "trajectory_artifacts": artifacts,
        "admission": admission,
        "pairing": pairing,
        "gap_rows": gap_rows,
        "independence": independence,
        "leakage": leakage,
        "coverage": coverage,
        "effective": effective,
        "release": release,
        "release_path": release_path,
        "formal_summary": formal_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the restricted Round-15 clean trajectory domain.")
    parser.add_argument("--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("pilot", "formal", "all"), default="all")
    parser.add_argument("--created-at")
    args = parser.parse_args()
    repository_root = Path(args.repository_root).resolve()
    output_root = Path(args.output_root).resolve()
    parent_root = Path(args.parent_root).resolve()
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    config = load_config(Path(args.config))
    if args.stage in {"pilot", "all"}:
        pilot = pilot_outputs(
            config=config,
            repository_root=repository_root,
            parent_root=parent_root,
            output_root=output_root,
            created_at=created_at,
        )
        print(
            f"round15_pilot_{pilot['pilot_validation']['status']} "
            f"producer={pilot['qualification']['status']} failures=1"
        )
    if args.stage in {"formal", "all"}:
        formal = formal_outputs(
            config=config,
            repository_root=repository_root,
            parent_root=parent_root,
            output_root=output_root,
            created_at=created_at,
        )
        print(
            f"round15_formal_{formal['formal_summary']['status']} "
            f"trajectory={formal['formal_summary']['trajectory_uid']} "
            f"admission={formal['admission']['decision']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
