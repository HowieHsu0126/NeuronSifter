from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import hashlib
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.scripts.builders.build_sample_observation_registry import produce_controlled_observations
from Libs.tests.test_joint_world_posterior import joint_case, raw_joint_source
from Libs.utils.data.action_observation_contracts import compile_pharmacological_action
from Libs.utils.data.native_diffusion_data import bind_joint_programs
from Libs.utils.data import native_diffusion_data as native
from Libs.utils.data.native_field_normalization import bind_artifact, fit_transform
from Libs.utils.data.physical_occupancy import compile_physical_record, digest, equilibrium_fraction
from Libs.utils.model import controlled_cns_source as controlled
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
from Libs.scripts.builders.build_simulation_readiness_audit import confidence_label, grade_row
from Libs.exp.training import train_screening_system
from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentTeacherFamilyDataset
from Libs.utils.model.dynamics_fields import (
    build_parameter_rows,
    build_species_rows,
    node_adjacency_matrix,
    quantitative_case_validation_rows,
    region_id_list,
    route_ordering_exposure,
    simulation_rows_and_store,
    species_id_list,
    time_grid_hours,
)
from Libs.utils.model.dynamics_store import initialize_chunk_store, load_chunked_rows, load_zarr_rows, write_chunked_array, write_zarr_array
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv
from Libs.utils.model.screening_core import TRI_COMPONENT_NAMES, tri_component_loss, tri_context_tensors, tri_target_tensors
from Libs.utils.model.screening_model_config import (
    baseline_field_policy_from_config,
    rollout_update_policy_from_config,
    solver_config_from_config,
    solver_state_layout_from_config,
    world_state_posterior_config_from_config,
)
from Libs.utils.evaluation.screening_datasets import patient_state_feature_matrix, regimen_features, target_program_features
from Libs.utils.data.structural_prior_registry import (
    structural_prior_local_response_head_rows,
    structural_prior_region_rows,
    write_structural_prior_catalogs,
)


def build_world_model(config: dict) -> ADScreeningSystem:
    _, target_feature_matrix = target_program_features(smoke_target_program_rows())
    _, regimen_feature_matrix = regimen_features(smoke_regimen_rows())
    species_lookup = {species_id: index for index, species_id in enumerate(species_id_list(config))}
    rollout_config = config["screening_pipeline"]["screening_benchmark"]["long_horizon_rollout"]
    return ADScreeningSystem(
        patient_state_dim=4,
        target_feature_dim=target_feature_matrix.shape[1],
        regimen_feature_dim=regimen_feature_matrix.shape[1],
        hidden_dim=8,
        latent_dim=4,
        dropout_rate=0.0,
        compartment_count=len(config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]),
        tri_weight_by_component={
            "calibration": 0.25,
            "coverage": 0.2,
            "pkpd_physics_consistency": 0.2,
            "ranking_stability": 0.2,
            "ood_score": 0.15,
        },
        region_count=len(region_id_list(config)),
        species_count=len(species_id_list(config)),
        rollout_step_count=int(config["dynamics_ready_registry"]["time_grid"]["step_count"]),
        adjacency_matrix=node_adjacency_matrix(config).tolist(),
        region_prior_feature_dim=6,
        default_region_prior=np.zeros((len(region_id_list(config)), 6), dtype=np.float32).tolist(),
        fast_steps_per_slow_step=int(rollout_config["fast_steps_per_slow_step"]),
        fast_state_indices=[species_lookup[species_id] for species_id in rollout_config["fast_state_species_ids"]],
        slow_state_indices=[species_lookup[species_id] for species_id in rollout_config["slow_state_species_ids"]],
        slow_update_scale=float(rollout_config["slow_update_scale"]),
        baseline_field_policy=baseline_field_policy_from_config(config),
        rollout_update_policy=rollout_update_policy_from_config(config),
        solver_state_layout=solver_state_layout_from_config(config),
        solver_config=solver_config_from_config(config),
        world_state_posterior_config=world_state_posterior_config_from_config(config),
    )


def tri_context(batch_size: int, regimen_count: int) -> dict[str, torch.Tensor]:
    return {
        "coverage": torch.full((batch_size, regimen_count), 0.75, dtype=torch.float32),
        "ood_score": torch.full((batch_size, regimen_count), 0.50, dtype=torch.float32),
    }


def smoke_state_features(latent: np.ndarray, probabilities: np.ndarray, matrix: np.ndarray, differential: np.ndarray) -> np.ndarray:
    return patient_state_feature_matrix(latent, probabilities, matrix, differential)


def smoke_compound_rows() -> list[dict[str, str]]:
    return [
        {"compound_uid": "compound_lecanemab", "preferred_name": "LECANEMAB"},
        {"compound_uid": "compound_donanemab", "preferred_name": "DONANEMAB"},
        {"compound_uid": "compound_verubecestat", "preferred_name": "VERUBECESTAT"},
    ]


def smoke_regimen_rows() -> list[dict[str, str]]:
    return [
        {
            "regimen_uid": "regimen_lecanemab",
            "compound_uid_set": "compound_lecanemab",
            "target_uid_set": "target_app",
            "target_symbol_set": "APP",
            "regimen_scaffold_set": "scaffold_alpha",
            "route": "intrathecal",
            "schedule": "weekly",
            "dose_level_id": "dose_100",
            "dose_multiplier": "1.0",
            "aggregate_target_evidence_score": "0.8",
            "aggregate_affinity_score": "1.0",
            "clue_reversal_proxy_score": "0.5",
            "pk_feasibility_score": "0.7",
            "regimen_size": "1",
            "selection_reason": "smoke",
            "note": "smoke",
        },
        {
            "regimen_uid": "regimen_donanemab",
            "compound_uid_set": "compound_donanemab",
            "target_uid_set": "target_app",
            "target_symbol_set": "APP",
            "regimen_scaffold_set": "scaffold_beta",
            "route": "intravenous",
            "schedule": "weekly",
            "dose_level_id": "dose_100",
            "dose_multiplier": "2.0",
            "aggregate_target_evidence_score": "1.4",
            "aggregate_affinity_score": "2.0",
            "clue_reversal_proxy_score": "0.6",
            "pk_feasibility_score": "0.85",
            "regimen_size": "1",
            "selection_reason": "smoke",
            "note": "smoke",
        },
        {
            "regimen_uid": "regimen_verubecestat",
            "compound_uid_set": "compound_verubecestat",
            "target_uid_set": "target_bace1",
            "target_symbol_set": "BACE1",
            "regimen_scaffold_set": "scaffold_gamma",
            "route": "oral",
            "schedule": "daily",
            "dose_level_id": "dose_100",
            "dose_multiplier": "1.0",
            "aggregate_target_evidence_score": "0.4",
            "aggregate_affinity_score": "0.6",
            "clue_reversal_proxy_score": "0.1",
            "pk_feasibility_score": "0.4",
            "regimen_size": "1",
            "selection_reason": "smoke",
            "note": "smoke",
        },
    ]


def smoke_target_rows() -> list[dict[str, str]]:
    return [
        {"target_uid": "target_app", "canonical_entity_id": "GENE1"},
        {"target_uid": "target_bace1", "canonical_entity_id": "GENE2"},
    ]


def smoke_target_program_rows() -> list[dict[str, str]]:
    return [
        {
            "target_program_uid": "target_program_app",
            "atlas_entity_uid": "atlas_entity_app",
            "atlas_name": "ALLEN_ABC",
            "program_id": "program_app",
            "canonical_entity_id": "GENE1",
            "canonical_entity_label": "APP",
            "mapped_feature_count": "1",
            "coverage_fraction": "1.0",
            "target_evidence_score": "0.9",
            "target_class": "enzyme",
            "source_repository": "atlas+drug_target",
        },
        {
            "target_program_uid": "target_program_bace1",
            "atlas_entity_uid": "atlas_entity_bace1",
            "atlas_name": "ALLEN_ABC",
            "program_id": "program_bace1",
            "canonical_entity_id": "GENE2",
            "canonical_entity_label": "BACE1",
            "mapped_feature_count": "1",
            "coverage_fraction": "1.0",
            "target_evidence_score": "0.8",
            "target_class": "enzyme",
            "source_repository": "atlas+drug_target",
        },
    ]


def write_smoke_dynamics_registry(config: dict, sample_rows: list[dict[str, object]]) -> None:
    root = REPO_ROOT / config["dynamics_ready_registry"]["local_root"]
    relative_paths = config["dynamics_ready_registry"]["relative_paths"]
    species_rows = build_species_rows(config)
    parameter_rows = build_parameter_rows(config)
    simulation_rows, _ = simulation_rows_and_store(config, smoke_regimen_rows(), smoke_compound_rows())
    sample_alignment_rows = [
        {
            "field_alignment_uid": f"field_alignment_{row['sample_uid']}",
            "study_uid": str(row["study_uid"]),
            "source_key": "geo",
            "source_accession": "GSE13214",
            "sample_uid": str(row["sample_uid"]),
            "modality": "transcriptomics",
            "observed_region_set": "middle_temporal_gyrus",
            "baseline_severity_score": 0.8 if row["sample_uid"].endswith("train") else 0.2,
            "simulation_condition": "ad_positive" if row["sample_uid"].endswith("train") else "control",
            "field_alignment_bundle_path": "Output/pytest_field_alignment.tsv",
            "note": "isf",
        }
        for row in sample_rows
    ]
    case_validation_rows = []
    for row in config["dynamics_ready_registry"]["case_validation_entries"]:
        case_validation_rows.append(
            {
                "validation_uid": f"{row['case_study_name']}_{row['intervention_label']}".lower(),
                **row,
            }
        )
    write_tsv(root / relative_paths["species_catalog"], config["schemas"]["dynamics_species_fields"], species_rows)
    write_tsv(root / relative_paths["parameter_catalog"], config["schemas"]["dynamics_parameter_fields"], parameter_rows)
    write_tsv(root / relative_paths["simulation_catalog"], config["schemas"]["dynamics_simulation_fields"], simulation_rows)
    write_tsv(root / relative_paths["sample_alignment_catalog"], config["schemas"]["dynamics_sample_alignment_fields"], sample_alignment_rows)
    write_tsv(root / relative_paths["case_validation_catalog"], config["schemas"]["dynamics_case_validation_fields"], case_validation_rows)


def write_smoke_structural_prior_registry(config: dict) -> None:
    region_rows = structural_prior_region_rows(config)
    head_rows = structural_prior_local_response_head_rows(config)
    write_structural_prior_catalogs(config, region_rows, head_rows)


def write_smoke_supporting_registries(
    config: dict,
    sample_rows: list[dict[str, object]] | None = None,
) -> None:
    effective_sample_rows = [] if sample_rows is None else sample_rows
    write_smoke_structural_prior_registry(config)
    write_smoke_dynamics_registry(config, effective_sample_rows)


def smoke_config(tmp_path: Path) -> dict:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    config["screening_pipeline"]["screening_benchmark"]["artifact_root"] = str((REPO_ROOT / "Output" / "pytest_neurodegeneration_artifacts" / tmp_path.name).relative_to(REPO_ROOT))
    config["screening_pipeline"]["screening_benchmark"]["benchmark_name"] = "ad_screening_system_smoke"
    config["dynamics_ready_registry"]["local_root"] = str((REPO_ROOT / "Output" / "pytest_dynamics_registry" / tmp_path.name).relative_to(REPO_ROOT))
    config["spatial_atlas_registry"]["local_root"] = str((REPO_ROOT / "Output" / "pytest_spatial_atlas" / tmp_path.name).relative_to(REPO_ROOT))
    config["structural_prior_registry"]["local_root"] = str((REPO_ROOT / "Output" / "pytest_structural_prior_registry" / tmp_path.name).relative_to(REPO_ROOT))
    config["screening_pipeline"]["screening_benchmark"]["training"]["max_epochs"] = 1
    config["screening_pipeline"]["screening_benchmark"]["training"]["patience_epochs"] = 1
    config["screening_pipeline"]["screening_benchmark"]["training"]["batch_size"] = 1
    config["screening_pipeline"]["screening_benchmark"]["training"]["memory_policy"]["effective_batch_size"] = 1
    config["screening_pipeline"]["screening_benchmark"]["training"]["memory_policy"]["micro_batch_size"] = 1
    config["screening_pipeline"]["screening_benchmark"]["training"]["memory_policy"]["gradient_accumulation_steps"] = 1
    config["screening_pipeline"]["screening_benchmark"]["training"]["training_target_count"] = 2
    config["screening_pipeline"]["screening_benchmark"]["training"]["training_regimen_count"] = 3
    config["screening_pipeline"]["screening_benchmark"]["training"]["cache_policy"]["enabled"] = False
    return config


def test_chunked_store_roundtrip(tmp_path: Path) -> None:
    store_path = tmp_path / "trajectory_store.zarr"
    array = np.arange(24, dtype=np.float32).reshape(6, 4)
    initialize_chunk_store(
        store_path,
        axis_names=["simulation", "feature"],
        coordinate_payload={"feature": np.asarray(["a", "b", "c", "d"], dtype=str)},
        metadata={"kind": "smoke"},
    )
    write_chunked_array(store_path, "trajectories", array, chunk_size=2)
    loaded = load_chunked_rows(store_path, "trajectories", np.asarray([0, 3, 5], dtype=np.int64))
    assert loaded.shape == (3, 4)
    assert np.allclose(loaded[0], array[0])
    assert np.allclose(loaded[1], array[3])
    assert np.allclose(loaded[2], array[5])


def test_zarr_row_loader_groups_repeated_row_chunks(tmp_path: Path) -> None:
    store_path = tmp_path / "zarr_store"
    array = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    write_zarr_array(store_path, "trajectories", array, chunks=(2, 3, 2))
    loaded = load_zarr_rows(store_path, "trajectories", np.asarray([0, 1, 3], dtype=np.int64))
    assert loaded.shape == (3, 3, 2)
    assert np.allclose(loaded[0], array[0])
    assert np.allclose(loaded[1], array[1])
    assert np.allclose(loaded[2], array[3])


def test_teacher_family_batch_loader_matches_single_row_chunk_reads(tmp_path: Path) -> None:
    store_path = tmp_path / "teacher_family_chunk_store"
    array = np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)
    initialize_chunk_store(
        store_path,
        axis_names=["simulation", "time", "species"],
        coordinate_payload={"species": np.asarray(["s0", "s1", "s2"], dtype=str)},
        metadata={"kind": "teacher_family_smoke"},
    )
    write_chunked_array(store_path, "trajectories", array, chunk_size=2)

    dataset = MicroenvironmentTeacherFamilyDataset.__new__(MicroenvironmentTeacherFamilyDataset)
    dataset.native_registry = None
    dataset.config = {"screening_pipeline": {"screening_benchmark": {"model_architecture": {
        "rollout_operator": {"transition_backbone": {"kind": "legacy_chunk_fixture"}}}}}}
    dataset.split_name = "train"
    dataset.parameter_prior_family_counts = {}
    dataset.rows = [
        {
            "teacher_family": "full_microenvironment_teacher",
            "trajectory_store_path": store_path,
            "trajectory_index": trajectory_index,
            "teacher_loss_weight": 0.5 + float(position),
            "store_format": "neuronsifter_chunk_store",
            "trajectory_array_name": "trajectories",
            "uncertainty_array_name": "teacher_uncertainty",
            "default_uncertainty_value": 0.25,
        }
        for position, trajectory_index in enumerate([3, 0, 5])
    ]

    batched_rows = dataset.__getitems__([0, 1, 2])
    single_rows = [dataset[index] for index in [0, 1, 2]]
    assert len(batched_rows) == len(single_rows)
    for batched, single in zip(batched_rows, single_rows):
        assert torch.allclose(batched["trajectory"], single["trajectory"])
        assert torch.allclose(batched["uncertainty"], single["uncertainty"])
        assert torch.allclose(batched["teacher_loss_weight"], single["teacher_loss_weight"])


def test_route_ordering_exposure_aligns_replicate_axis_from_catalog(tmp_path: Path) -> None:
    config = smoke_config(tmp_path)
    root = REPO_ROOT / config["dynamics_ready_registry"]["local_root"]
    relative_paths = config["dynamics_ready_registry"]["relative_paths"]
    regimen_rows = [smoke_regimen_rows()[0], smoke_regimen_rows()[2]]
    route_config = config["dynamics_ready_registry"]["field_dynamics_priors"]["route_ordering_evaluation"]
    region_index = region_id_list(config).index(str(route_config["region_id"]))
    compartment_ids = [row["compartment_id"] for row in config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]]
    compartment_indices = [compartment_ids.index(compartment_id) for compartment_id in route_config["compartment_ids"]]
    species_index = species_id_list(config).index(str(route_config["species_id"]))
    simulation_rows = []
    for trajectory_index, regimen_row in enumerate([regimen_rows[0], regimen_rows[0], regimen_rows[1], regimen_rows[1]]):
        simulation_rows.append(
            {
                "simulation_uid": f"simulation_route_alignment_{trajectory_index}",
                "disease_program": route_config["disease_program"],
                "simulation_condition": route_config["simulation_condition"],
                "heterogeneity_replicate_id": f"replicate_{trajectory_index}",
                "parameter_draw_uid": f"draw_{trajectory_index}",
                "parameter_evidence_status": "pytest",
                "regimen_uid": regimen_row["regimen_uid"],
                "route": regimen_row["route"],
                "schedule": regimen_row["schedule"],
                "dose_level_id": regimen_row["dose_level_id"],
                "dose_multiplier": regimen_row["dose_multiplier"],
                "observed_region_set": "|".join(region_id_list(config)),
                "structural_prior_uid": "pytest",
                "local_response_head_uid": "pytest",
                "trajectory_index": trajectory_index,
                "trajectory_store_path": "pytest",
                "time_step_count": config["dynamics_ready_registry"]["time_grid"]["step_count"],
                "region_count": len(region_id_list(config)),
                "compartment_count": len(compartment_ids),
                "species_count": len(species_id_list(config)),
                "note": "replicate-aligned route-ordering regression",
            }
        )
    write_tsv(root / relative_paths["simulation_catalog"], config["schemas"]["dynamics_simulation_fields"], simulation_rows)
    field_trajectory = np.zeros(
        (
            1,
            len(simulation_rows),
            int(config["dynamics_ready_registry"]["time_grid"]["step_count"]),
            len(region_id_list(config)),
            len(compartment_ids),
            len(species_id_list(config)),
        ),
        dtype=np.float32,
    )
    for axis_index, signal_value in enumerate([0.8, 0.7, 0.1, 0.1]):
        for compartment_index in compartment_indices:
            field_trajectory[0, axis_index, int(route_config["time_index"]), region_index, compartment_index, species_index] = signal_value
    route_exposure = route_ordering_exposure(field_trajectory, pd.DataFrame(regimen_rows), config)
    assert route_exposure["intrathecal"] > route_exposure["oral"]


def joint_field_output_source(model: ADScreeningSystem, config: dict, regimen_rows: list[dict]) -> dict:
    # Extend the existing controlled unit prior to this test's complete field grid.
    _, small, _, _, channels = joint_case()
    def expand_batch(value: torch.Tensor) -> torch.Tensor:
        return value.expand(2, *value.shape[1:]).clone()
    support = replace(
        small,
        sample_uids=("sample-a", "sample-b"),
        particle_uids=(("a-low", "a-high"), ("b-low", "b-high")),
        regime_ids=(("low", "high"), ("low", "high")),
        factors={name: small.factors[name][..., :1].expand(2, 2, width).clone()
                 for name, width in model.world_state_posterior.factor_dimensions.items()},
        initial_field=small.initial_field[:, :, :1, :1, :1].expand(
            2, 2, model.region_count, model.compartment_count, model.species_count).clone(),
        physical_parameters=small.physical_parameters[..., :1].expand(2, 2, model.species_count).clone(),
        prior_weights=expand_batch(small.prior_weights),
        species_units=("relative",) * model.species_count,
        readout_parameters={name: {key: expand_batch(value) if torch.is_tensor(value) else value
                                  for key, value in fields.items()}
                            for name, fields in small.readout_parameters.items()},
        source_nuisance={name: expand_batch(value) for name, value in small.source_nuisance.items()},
    ).validate()
    channels = copy.deepcopy(channels)
    for channel in channels:
        for name in ("observed", "library_offset", "detection_limit"):
            if name in channel:
                channel[name] = channel[name] * 2
    production = produce_controlled_observations(support, channels, latent_seed=7, observation_seed=13)
    source = raw_joint_source(support, production)
    programs = source["particle_programs"]
    times = time_grid_hours(config).tolist()
    schedules = {row["schedule"]: row for row in config["screening_pipeline"]["pkpd_ready_registry"]["schedules"]}
    records = []
    for sample_records in programs["records"]:
        particles = []
        for particle_records in sample_records:
            actions = []
            for regimen in regimen_rows:
                record = copy.deepcopy(particle_records[0])
                uid = regimen["regimen_uid"]
                targets = regimen["target_uid_set"].split("|")
                action = record["action"]
                # The inherited unit action declares a synthetic 1 mg/kg reference dose.
                action.update(action_uid="action-" + uid, target_set=targets,
                    agent_set=regimen["compound_uid_set"].split("|"),
                    dose=action["dose"] * float(regimen["dose_multiplier"]), route=regimen["route"],
                    schedule={"kind": "interval", "interval": 1.0 / float(schedules[regimen["schedule"]]["frequency_per_day"]), "unit": "day"},
                    duration={"value": float(times[-1] - times[0]), "unit": "hour"},
                    state_scope={"regions": region_id_list(config)})
                record["pkpd_context"].update(time_grid=times, target_program={"targets": targets})
                compiled = compile_pharmacological_action(action, pkpd_context=record["pkpd_context"],
                    baseline_state_posterior=record["baseline_state_posterior"], graph_context=record["graph_context"])
                assert compiled["status"] == "compiled_with_explicit_uncertainty"
                record.update(program_uid=compiled["program_uid"], action_uid=action["action_uid"],
                    regimen_uid=uid, target_program_uid=regimen["target_uid_set"], compiled_program=compiled)
                actions.append(record)
            particles.append(actions)
        records.append(particles)
    shape = (2, 2, len(regimen_rows), len(times), model.region_count, model.compartment_count, 1)
    programs.update(records=records, regimen_uids=[row["regimen_uid"] for row in regimen_rows],
        occupancy=torch.rand(shape, generator=torch.Generator().manual_seed(31)).tolist(),
        support=torch.ones(shape).tolist(),
        physical_time=torch.tensor(times).reshape(1, 1, 1, -1, 1, 1, 1).expand(shape).tolist())
    return source


def test_score_regimens_returns_field_outputs() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    model = build_world_model(config).eval()
    regimen_frame, regimen_feature_matrix = regimen_features(smoke_regimen_rows())
    source = joint_field_output_source(model, config, regimen_frame.to_dict("records"))
    with torch.no_grad():
        posterior = model.infer_world_state(joint_support=source["support"], observations=source["observations"])["joint_posterior"]
        programs = bind_joint_programs(posterior, source, action_policy=model.field_rollout_operator.action_policy_name)
        outputs = model.score_regimens(
            posterior.observation_features,
            torch.tensor(regimen_feature_matrix, dtype=torch.float32),
            tri_context(2, 3),
            joint_posterior=posterior,
            particle_programs=programs,
            regimen_uids=regimen_frame["regimen_uid"].tolist(),
        )
    assert outputs["baseline_field"].shape == (2, len(region_id_list(config)), 6, len(species_id_list(config)))
    assert outputs["field_trajectory"].shape == (2, 3, int(config["dynamics_ready_registry"]["time_grid"]["step_count"]), len(region_id_list(config)), 6, len(species_id_list(config)))
    assert outputs["final_field"].shape == (2, 3, len(region_id_list(config)), 6, len(species_id_list(config)))
    assert outputs["regimen_scores"].shape == (2, 3)
    assert outputs["pkpd_consistency"].shape == (2, 3)
    assert outputs["tri_score"].shape == (2, 3)
    assert outputs["exposure"].shape == (3, 6)
    assert outputs["occupancy"].shape == (3, 6)
    assert set(outputs["tri_components"]) == set(TRI_COMPONENT_NAMES)
    assert outputs["field_trajectory_std"].shape == outputs["field_trajectory"].shape
    assert torch.isfinite(outputs["field_trajectory_std"]).all()
    torch.testing.assert_close(outputs["field_trajectory_std"].square(),
        outputs["within_diffusion_variance"] + outputs["between_particle_variance"])
    assert outputs["particle_uids"] == posterior.support.particle_uids


def test_simulator_route_ordering_and_quantitative_case_validation(tmp_path: Path) -> None:
    config = smoke_config(tmp_path)
    write_smoke_supporting_registries(config)
    simulation_catalog_path = (
        REPO_ROOT
        / config["dynamics_ready_registry"]["local_root"]
        / config["dynamics_ready_registry"]["relative_paths"]["simulation_catalog"]
    )
    simulation_rows = read_tsv(simulation_catalog_path)
    field_trajectory = np.stack(
        [simulate_row for simulate_row in load_chunked_rows(REPO_ROOT / config["dynamics_ready_registry"]["local_root"] / config["dynamics_ready_registry"]["relative_paths"]["trajectory_store"], "trajectories", np.arange(len(simulation_rows), dtype=np.int64))],
        axis=0,
    )
    regimen_frame = pd.DataFrame(smoke_regimen_rows())
    ad_rows = [row for row in simulation_rows if row["disease_program"] == "ad_control" and row["simulation_condition"] == "ad_positive"]
    ad_indices = np.asarray([int(row["trajectory_index"]) for row in ad_rows], dtype=np.int64)
    predicted_field = load_chunked_rows(
        REPO_ROOT / config["dynamics_ready_registry"]["local_root"] / config["dynamics_ready_registry"]["relative_paths"]["trajectory_store"],
        "trajectories",
        ad_indices,
    )[None, ...]
    assert predicted_field.shape[1] > len(regimen_frame)
    route_exposure = route_ordering_exposure(predicted_field, regimen_frame, config)
    assert route_exposure["intrathecal"] > route_exposure["intravenous"] > route_exposure["oral"]
    case_validation_catalog = (
        REPO_ROOT
        / config["dynamics_ready_registry"]["local_root"]
        / config["dynamics_ready_registry"]["relative_paths"]["case_validation_catalog"]
    )
    assert case_validation_catalog.exists()
    case_rows = quantitative_case_validation_rows(config, regimen_frame, smoke_compound_rows(), predicted_field)
    effect_by_intervention = {row["intervention_label"]: row["predicted_standardized_effect"] for row in case_rows}
    assert effect_by_intervention["DONANEMAB"] > effect_by_intervention["LECANEMAB"] > effect_by_intervention["VERUBECESTAT"]


def test_tri_losses_are_finite() -> None:
    patient_probabilities = np.asarray([0.9, 0.2], dtype=np.float32)
    patient_labels = np.asarray([1, 0], dtype=np.int64)
    patient_matrix = np.asarray([[1.0, 0.4], [0.2, 0.9]], dtype=np.float32)
    latent_matrix = np.asarray([[0.2, 0.1, 0.4, 0.3], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    regimen_labels = np.asarray([[1.2, 0.6], [0.4, 0.8]], dtype=np.float32)
    predicted_components = {
        component_name: torch.full((2, 2), 0.5, dtype=torch.float32)
        for component_name in TRI_COMPONENT_NAMES
    }
    tri_targets = tri_target_tensors(
        patient_probabilities,
        patient_labels,
        patient_matrix,
        latent_matrix,
        regimen_labels,
        torch.full((2, 2), 0.7, dtype=torch.float32),
        torch.device("cpu"),
    )
    tri_loss = tri_component_loss(predicted_components, tri_targets)
    assert torch.isfinite(tri_loss)



def diagnostic_route_parameters(parameters: dict, action: dict, source: dict) -> tuple[np.ndarray, float]:
    """Hypothetical site allocation; these values are not measured drug PK."""
    route = source["diagnostic_route_pk"][action["route"]]
    weights = np.asarray([route["site_weights"][site["compartment_id"]] for site in source["sites"]])
    weighted_delivery = np.asarray(parameters["delivery_fraction"]) * weights
    return route["bioavailability"] * weighted_delivery / weighted_delivery.sum(), route["absorption_per_hour"]


def diagnostic_route_pk(times, parameters, action, source, *, high_precision=False,
                        initial_pre_dose=True, active_through=None):
    """Independent-site bolus or depot absorption followed by first-order elimination."""
    times = np.asarray(times, dtype=float)
    fractions, ka = diagnostic_route_parameters(parameters, action, source)
    volume = np.asarray(parameters["volume_l"])
    rates = np.asarray(parameters["elimination_per_hour"])
    kd = np.asarray(parameters["kd_mol_per_l"])
    bmax = np.asarray(parameters["bmax_mol_per_l"])
    dose = source["pk"]["total_dose_mg_per_kg"] * source["pk"]["body_mass_kg"] * 1e-3 / source["pk"]["molecular_weight_g_per_mol"]
    events = [(t, f) for t, f in zip(action["dose_times_hours"], action["fractions"])
              if active_through is None or t <= active_through]
    concentration = np.zeros((len(times), len(volume)))
    occupancy = np.zeros_like(concentration)
    bound = np.zeros_like(concentration)
    if high_precision:
        with localcontext() as context:
            context.prec = source["numerics"]["decimal_precision"]
            D = lambda value: Decimal(str(float(value)))
            q = D(source["pk"]["total_dose_mg_per_kg"]) * D(source["pk"]["body_mass_kg"]) * Decimal("0.001") / D(source["pk"]["molecular_weight_g_per_mol"])
            for ti, t in enumerate(times):
                for si in range(len(volume)):
                    amount = Decimal(0)
                    if not (initial_pre_dose and t == 0):
                        for event, fraction in events:
                            if t < event:
                                continue
                            lag, rate = D(t) - D(event), D(rates[si])
                            decay = (-rate * lag).exp()
                            if action["route"] == "oral":
                                absorption = D(ka)
                                decay = (absorption * lag * decay if absorption == rate else
                                         absorption / (absorption - rate) * (decay - (-absorption * lag).exp()))
                            amount += D(fractions[si]) * q * D(fraction) * decay
                    value = amount / D(volume[si])
                    occupied = value / (D(kd[si]) + value)
                    concentration[ti, si], occupancy[ti, si], bound[ti, si] = float(value), float(occupied), float(D(bmax[si]) * occupied)
        return concentration, occupancy, bound
    for event, fraction in events:
        mask = times >= event
        if initial_pre_dose:
            mask &= times != 0
        lag = times[mask, None] - event
        response = np.exp(-lag * rates)
        if action["route"] == "oral":
            difference = ka - rates
            # The declared ka differs from every site's k; reject a changed definition.
            assert np.all(difference != 0)
            response *= ka * (-np.expm1(-lag * difference)) / difference
        concentration[mask] += fractions * dose * fraction / volume * response
    occupancy = equilibrium_fraction(concentration, kd)
    return concentration, occupancy, bmax * occupancy


def diagnostic_pk_balance(parameters: dict, action: dict, source: dict) -> dict:
    """Check the closed form against depot/central/eliminated amount ODEs."""
    fractions, ka = diagnostic_route_parameters(parameters, action, source)
    k = np.asarray(parameters["elimination_per_hour"])
    volume = np.asarray(parameters["volume_l"])
    q = source["pk"]["total_dose_mg_per_kg"] * source["pk"]["body_mass_kg"] * 1e-3 / source["pk"]["molecular_weight_g_per_mol"]
    times = source["grid"]["time_hours"]
    events = dict(zip(action["dose_times_hours"], action["fractions"]))
    boundaries = sorted(set(times) | set(events))
    state = np.zeros((3, len(k)))  # depot, central amount, eliminated amount [mol]
    delivered = np.zeros(len(k))
    saved = {0.: np.zeros(len(k))}
    balance_error = 0.
    oral = action["route"] == "oral"
    def rhs(t, flat):
        depot, amount, eliminated = flat.reshape(3, -1)
        uptake = ka * depot if oral else np.zeros_like(depot)
        return np.stack((-uptake, uptake - k * amount, k * amount)).ravel()
    for left, right in zip(boundaries, boundaries[1:]):
        if left in events:
            increment = fractions * q * events[left]
            state[0 if oral else 1] += increment
            delivered += increment
        solved = solve_ivp(rhs, (left, right), state.ravel(), method="DOP853", t_eval=[right],
            rtol=source["numerics"]["reference_rtol"],
            atol=float(np.min(fractions * q)) * source["numerics"]["reference_atol_scale"],
            max_step=source["numerics"]["reference_max_step_hours"])
        assert solved.success, solved.message
        state = solved.y[:, -1].reshape(3, -1)
        balance_error = max(balance_error, float(np.max(np.abs(state.sum(0) - delivered) / (fractions * q))))
        if right in times:
            amount = state[1].copy()
            if right in events and not oral:
                amount += fractions * q * events[right]
            saved[right] = amount / volume
    numerical = np.stack([saved[t] for t in times])
    closed = diagnostic_route_pk(times, parameters, action, source)[0]
    curve_error = float(np.max(np.abs(numerical - closed) / (fractions * q / volume)))
    return {"route": action["route"], "curve_relative_max": curve_error,
            "depot_central_eliminated_balance_max": balance_error,
            "included_oral_depot": oral, "modeled_bioavailability": float(fractions.sum()),
            "excluded_dose_fraction": float(1 - fractions.sum()),
            "passed": max(curve_error, balance_error) <= source["numerics"]["balance_tolerance"]}


def write_native_training_smoke_source(config: dict, root: Path) -> None:
    """Manufactured numerical data for training plumbing, not AD PK or efficacy evidence."""
    root.mkdir(parents=True)
    source = copy.deepcopy(load_config(str(REPO_ROOT / "Libs/configs/exp/controlled_native_source.yaml"))["controlled_cns_source"])
    source.update(source_uid="training_artifact_diagnostic", snapshot_ref=str(root / "definition.json"),
        seed=41, interpretation="TEST ONLY: hypothetical route PK with reused controlled reaction/transport equations",
        budget={"cases": 6}, replicates=[0], source_license=str(root / "use_terms.txt"))
    source["diagnostic_route_pk"] = {
        "intravenous": {"bioavailability": 1., "absorption_per_hour": 0.,
            "site_weights": dict(zip(["plasma", "bbb_interface", "csf", "pvs", "isf", "meningeal_drainage"], [9., 1., .2, .2, .3, .1]))},
        "intrathecal": {"bioavailability": 1., "absorption_per_hour": 0.,
            "site_weights": dict(zip(["plasma", "bbb_interface", "csf", "pvs", "isf", "meningeal_drainage"], [.1, .2, 9., 2., 1., 2.]))},
        "oral": {"bioavailability": .65, "absorption_per_hour": .08,
            "site_weights": dict(zip(["plasma", "bbb_interface", "csf", "pvs", "isf", "meningeal_drainage"], [8., 1., .1, .2, .2, .1]))},
    }
    source["diagnostic_definition"] = {
        "release": "diagnostic-v1", "parameter_origin": "explicit hypothetical unit parameters; not measured real-drug PK",
        "dose_reference_mg_per_kg_per_administration": .01,
        "route_allocation": "F_s=bioavailability*base_delivery_s*route_site_weight_s/sum(base_delivery*route_site_weight)",
        "iv_it_law": "C_s=sum F_s*q_event/V_s*exp(-k_s*lag)",
        "oral_law": "dD/dt=-ka*D; dA/dt=ka*D-k*A; dE/dt=k*A; C=A/V",
        "oral_dose": "F_s*q_event enters depot; the remaining non-bioavailable dose is outside the modeled pool",
        "units": {"time": "hour", "amount": "mol", "volume": "L", "concentration": "mol/L", "rates": "1/hour"},
        "source_only_forcing_composition": "controlled.pk_values temporarily replaced by diagnostic_route_pk and restored before admission/model training",
    }
    times = source["grid"]["time_hours"]
    assert times == time_grid_hours(config).tolist()
    schedules = {row["schedule"]: row for row in config["screening_pipeline"]["pkpd_ready_registry"]["schedules"]}
    source["actions"] = []
    for row in smoke_regimen_rows():
        events = np.arange(0., times[-1], 24. / float(schedules[row["schedule"]]["frequency_per_day"])).tolist()
        source["actions"].append({**row, "dose_times_hours": events, "fractions": [1. / len(events)] * len(events)})
    source["solver_sha256"] = hashlib.sha256(Path(controlled.__file__).read_bytes()).hexdigest()
    source["fixture_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(root / "definition.json", source)
    (root / "use_terms.txt").write_text("Locally manufactured diagnostic data for internal software tests. No measured drug PK, biological efficacy, clinical validation or public release is claimed.\n")
    splits = [{"sample_uid": "sample_train", "role": "train", "condition": "ad_positive"},
              {"sample_uid": "sample_validation", "role": "validation", "condition": "control"}]
    write_json(root / "splits.json", splits)
    store = root / "native_raw.zarr"
    registry = {"schema_version": native.SCHEMA_VERSION, "grid": controlled.source_grid(source),
                "posterior_family": "finite_joint_particles", "sources": [], "joint_sources": [],
                "teachers": [], "programs": [], "observations": []}
    config["native_diffusion_data"].update(species_units=copy.deepcopy(source["species_units"]),
        registry_path=str(root / "registry.json"))
    scales = np.asarray([source["reference_scales"][name] for name in source["grid"]["species_id"]])
    audit = config["ai_ready_preprocessing"]["simulation_audit"]
    trajectories, uncertainty, confidence, checks, qc_rows = [], [], [], [], []
    original_pk = controlled.pk_values
    with patch.object(controlled, "pk_values", diagnostic_route_pk):
        for split in splits:
            current = copy.deepcopy(source)
            current["source_uid"] += ":" + split["role"]
            sample = split["sample_uid"]
            group = {**split, "support_uid": sample + ":support", "posterior_uid": sample + ":prior",
                     "particle_uids": [sample + ":particle:" + str(i) for i in range(4)]}
            # Each role uses precisely seeds41/42/43. Different initial conditions retain distinct sources.
            support, production, truth = controlled.draw_world(current, group, 0)
            programs = [[] for _ in range(4)]
            registry["sources"].append({"source_uid": current["source_uid"], "public": True, "protected": False,
                "license": str(root / "use_terms.txt"), "release": "diagnostic-v1", "snapshot_ref": source["snapshot_ref"],
                "split_evidence_ref": str(root / "splits.json"), "unit_evidence_ref": str(root / "definition.json"),
                "source_kind": "controlled_simulated", "interpretation": source["interpretation"]})
            for action in current["actions"]:
                case_source = copy.deepcopy(current)
                case_source["pk"].update(total_dose_mg_per_kg=.01 * float(action["dose_multiplier"]) * len(action["fractions"]),
                    ligand_uid=action["compound_uid_set"], target_uid=action["target_uid_set"])
                pk_error, pk_serial = np.zeros(15), np.zeros(15)
                pk_checks = []
                for atom in range(4):
                    law = controlled.ControlledLaw(case_source, {key: value[0, atom].numpy() for key, value in support.factors.items()}, support.initial_field[0, atom].numpy())
                    pk = diagnostic_route_pk(times, law.pk, action, case_source)
                    precise = diagnostic_route_pk(times, law.pk, action, case_source, high_precision=True)
                    pk_checks.append(diagnostic_pk_balance(law.pk, action, case_source))
                    for name, index in [("free_drug", 0), ("target_bound_drug", 2)]:
                        si = source["grid"]["species_id"].index(name)
                        pk_error[si] = max(pk_error[si], float(np.max(np.abs(pk[index] - precise[index]))))
                        pk_serial[si] = max(pk_serial[si], float(np.max(np.abs(pk[index] - pk[index].astype(np.float32).astype(float)))))
                    # Reuse the physical export layout, then compile the actual diagnostic route.
                    export = controlled.compiled_particle_program(case_source, group, support, atom, action, pk, law.pk, source["solver_sha256"])
                    inputs = copy.deepcopy(export["physical_program"]["inputs"])
                    inputs["action"]["route"] = action["route"]
                    inputs["run_snapshot"]["settings"]["route"] = action["route"]
                    inputs["run_snapshot"]["diagnostic_route_definition"] = source["diagnostic_route_pk"][action["route"]]
                    inputs["provenance"]["run_snapshot_sha256"] = digest(inputs["run_snapshot"])
                    record = compile_physical_record(inputs)
                    assert record["action"]["route"] == action["route"]
                    np.testing.assert_allclose(np.asarray(record["physical_program"]["free_concentration_mol_per_l"])[..., 0].reshape(len(times), -1), pk[0], rtol=0, atol=0)
                    np.testing.assert_allclose(np.asarray(record["physical_program"]["occupancy"])[..., 0].reshape(len(times), -1), pk[1], rtol=0, atol=0)
                    programs[atom].append(record)
                law = controlled.ControlledLaw(case_source, {key: value[0, truth].numpy() for key, value in support.factors.items()}, support.initial_field[0, truth].numpy())
                results = {label: controlled.integrate(law, action, method, step) for label, method, step in
                    [("coarse", "RK4", source["numerics"]["rk4_steps_hours"][0]),
                     ("fine", "RK4", source["numerics"]["rk4_steps_hours"][1]),
                     ("reference", "DOP853", source["numerics"]["reference_max_step_hours"])]}
                quality = controlled.numerical_summary(results["coarse"], results["fine"], results["reference"], scales, pk_error, pk_serial, source["numerics"])
                index = len(trajectories)
                field = results["reference"]["trajectory"].astype(np.float32)
                content = hashlib.sha256(field.tobytes()).hexdigest()
                numeric_ok = quality["admissible"] and all(item["passed"] for item in pk_checks)
                schema_ok = field.shape == (28, 6, 6, 15) and bool(np.isfinite(field).all())
                uncertainty_ok = bool(np.isfinite(quality["uncertainty"]).all() and (np.asarray(quality["uncertainty"]) > 0).all())
                leakage_ok = all(row["content_hash"] != content for row in registry["teachers"])
                measured_label = confidence_label(float(np.mean(np.asarray(quality["uncertainty"]) / scales)), audit)
                grade, use, weight = grade_row(schema_ok, numeric_ok, all(record["action"]["route"] == action["route"] for record in [row[-1] for row in programs]),
                    uncertainty_ok, leakage_ok, measured_label, ["hypothetical_test_only_source"], audit)
                check = {"sample_uid": sample, "regimen_uid": action["regimen_uid"], "teacher_atom_index": truth,
                         "pk_checks": pk_checks, "numerical": quality, "schema_ok": schema_ok, "uncertainty_ok": uncertainty_ok,
                         "leakage_ok": leakage_ok, "confidence_label": measured_label, "qc_grade": grade}
                checks.append(check)
                np.savez_compressed(root / ("numerical_case_" + str(index) + ".npz"), **{key: value["trajectory"] for key, value in results.items()})
                check.update(qc_use=use, sample_weight=weight)
                write_json(root / "numerical_checks.json", checks)
                qc_rows.append({"trajectory_uid": sample + ":" + action["regimen_uid"], "sample_uid": sample,
                    "regimen_uid": action["regimen_uid"], "split_name": split["role"], "qc_grade": grade,
                    "qc_use": use, "sample_weight": weight, "teacher_confidence": quality["teacher_confidence"]})
                write_tsv(root / "simulation_qc_catalog.tsv", list(qc_rows[0]), qc_rows)
                assert grade in {"A", "B"}, check
                program = programs[0][-1]
                teacher = {**{key: program[key] for key in ["program_uid", "posterior_uid", "source_uid", "action_uid", "regimen_uid", "target_program_uid"]},
                    "trajectory_uid": sample + ":" + action["regimen_uid"], "sample_uid": sample, "family_uid": sample + ":diagnostic_family",
                    "content_hash": content, "trajectory_store_path": str(store), "store_format": "zarr_v2", "trajectory_index": index,
                    "teacher_family": "manufactured_training_diagnostic", "qc_catalog_ref": str(root / "simulation_qc_catalog.tsv"),
                    "source_snapshot_ref": source["snapshot_ref"], "uncertainty_origin": "independent_supplied", "teacher_confidence": quality["teacher_confidence"],
                    "qc_grade": grade, "qc_use": use, "sample_weight": weight, "split_name": split["role"], "source_holdout_role": split["role"]}
                registry["teachers"].append(teacher)
                trajectories.append(field); uncertainty.append(quality["uncertainty"]); confidence.append(quality["teacher_confidence"])
            particle_programs = {"source_posterior_uid": group["posterior_uid"], "source_snapshot_ref": source["snapshot_ref"],
                "support_uid": support.support_uid, "source_uid": support.source_uid, "sample_uids": [sample],
                "particle_uids": [group["particle_uids"]], "regimen_uids": [row["regimen_uid"] for row in current["actions"]], "records": [programs]}
            for key in ["occupancy", "support", "physical_time"]:
                particle_programs[key] = np.asarray([[record["physical_program"][key] for record in row] for row in programs])[None].tolist()
            raw = {"schema_version": native.JOINT_SOURCE_SCHEMA, "source_posterior_uid": group["posterior_uid"],
                   "support": support.to_payload(), "observations": production.inference_payload()["observations"],
                   "particle_programs": particle_programs, "producer_seeds": {"latent_seed": production.latent_seed, "observation_seed": production.observation_seed}}
            native.parse_joint_source(raw)
            registry["joint_sources"].append(raw)
    assert controlled.pk_values is original_pk
    arrays = {"trajectory": np.asarray(trajectories, dtype=np.float32),
              "uncertainty": np.asarray(uncertainty, dtype=np.float32),
              "teacher_confidence": np.asarray(confidence, dtype=np.float32)}
    write_native_microenvironment_store(store, config, registry, arrays)
    train_records = native.select_records(registry, config, "train", ["sample_train"],
                                          [row["regimen_uid"] for row in source["actions"]])
    train_arrays, _ = native.load_records(registry, train_records, config)
    normalization = fit_transform(registry["grid"], train_records, train_arrays, config)
    assert normalization["trajectory_count"] == 3 and {row["sample_uid"] for row in normalization["training_records"]} == {"sample_train"}
    normalization_path = root / "train_normalization.json"
    write_json(normalization_path, normalization)
    final_store = root / "native_source.zarr"
    final_registry = native.reindex_registry(registry, native.join_registry(registry, config), final_store, "zarr_v2")
    final_registry["grid"] = bind_artifact(registry["grid"], normalization_path, config)
    write_native_microenvironment_store(final_store, config, final_registry, arrays)
    write_json(root / "registry.json", final_registry)


def test_train_one_scope_smoke_writes_unified_artifacts(tmp_path: Path) -> None:
    config = smoke_config(tmp_path)
    config["screening_pipeline"]["screening_benchmark"]["training"].update(
        seed_policy="manuscript_native_three", random_seed=41,
    )
    config["selected_formal_seed"] = 41
    config["paper_grade_full_suite"].update(formal_seed_values=[41, 42, 43], seeds=[41, 42, 43])
    config["paper_grade_baseline_registry"]["seed_values"] = [41, 42, 43]
    # Reuse joint_case's diagnostic loss balance to exercise both posterior objectives.
    objectives = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["world_state_posterior"]["joint_objectives"]
    objectives.update(observation_likelihood_weight=0.7, categorical_kl_weight=0.2)
    config_path = REPO_ROOT / "Output" / "pytest_neurodegeneration_artifacts" / tmp_path.name / "smoke_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    entity_index = pd.DataFrame({"canonical_entity_id": ["GENE1", "GENE2"]})
    train_payload = {
        "matrix": np.asarray([[1.0, 0.2]], dtype=np.float32),
        "labels": np.asarray([1], dtype=np.int64),
        "metadata": pd.DataFrame({"sample_uid": ["sample_train"], "study_uid": ["study_alpha"]}),
        "entity_index": entity_index,
        "latent": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        "state_features": smoke_state_features(
            np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            np.asarray([0.9], dtype=np.float32),
            np.asarray([[1.0, 0.2]], dtype=np.float32),
            np.asarray([1.0, 0.8], dtype=np.float32),
        ),
        "probabilities": np.asarray([0.9], dtype=np.float32),
        "differential": np.asarray([1.0, 0.8], dtype=np.float32),
    }
    validation_payload = {
        "matrix": np.asarray([[0.8, 0.1]], dtype=np.float32),
        "labels": np.asarray([0], dtype=np.int64),
        "metadata": pd.DataFrame({"sample_uid": ["sample_validation"], "study_uid": ["study_alpha"]}),
        "entity_index": entity_index,
        "latent": np.asarray([[0.3, 0.2, 0.1, 0.4]], dtype=np.float32),
        "state_features": smoke_state_features(
            np.asarray([[0.3, 0.2, 0.1, 0.4]], dtype=np.float32),
            np.asarray([0.3], dtype=np.float32),
            np.asarray([[0.8, 0.1]], dtype=np.float32),
            np.asarray([1.0, 0.8], dtype=np.float32),
        ),
        "probabilities": np.asarray([0.3], dtype=np.float32),
        "differential": np.asarray([1.0, 0.8], dtype=np.float32),
    }
    write_smoke_supporting_registries(
        config,
        [
            {"sample_uid": "sample_train", "study_uid": "study_alpha"},
            {"sample_uid": "sample_validation", "study_uid": "study_alpha"},
        ],
    )
    write_native_training_smoke_source(config, tmp_path / "native_training_source")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    modality_payload = {
        "transcriptomics": {
            "train": train_payload,
            "validation": validation_payload,
            "test": copy.deepcopy(validation_payload),
        }
    }
    checkpoint_path = train_screening_system.train_one_scope(
        config,
        config_path,
        ["pytest"],
        "fold",
        "fold_alpha",
        modality_payload,
        smoke_target_program_rows(),
        smoke_regimen_rows(),
        smoke_target_rows(),
        smoke_compound_rows(),
        [],
        torch.device("cpu"),
    )
    assert "native_diffusion" not in modality_payload["transcriptomics"]["test"]
    assert checkpoint_path.exists()
    manifest_path = checkpoint_path.parent / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == manifest["ensemble_member_seed"] == 41
    assert manifest["training_seed_binding"] == {
        "schema_version": "native_training_seed_binding.v1",
        "seed_policy": "manuscript_native_three",
        "allowed_training_seeds": [41, 42, 43],
        "selected_formal_seed": 41,
        "base_training_seed": 41,
        "model_seed": 41,
        "scope": ["fold", "fold_alpha", "ad_screening_system"],
    }
    assert manifest["model_name"] == "ad_screening_system"
    assert manifest["field_axis_fields"]["region_count"] == len(region_id_list(config))


def test_build_training_targets_keeps_case_study_seed_regimen() -> None:
    entity_index = pd.DataFrame({"canonical_entity_id": ["GENE1"]})
    modality_payload = {
        "transcriptomics": {
            "train": {
                "matrix": np.asarray([[1.0]], dtype=np.float32),
                "labels": np.asarray([1], dtype=np.int64),
                "metadata": pd.DataFrame({"sample_uid": ["sample_a"], "study_uid": ["study_alpha"]}),
                "entity_index": entity_index,
                "latent": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                "state_features": smoke_state_features(
                    np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                    np.asarray([0.9], dtype=np.float32),
                    np.asarray([[1.0]], dtype=np.float32),
                    np.asarray([1.0], dtype=np.float32),
                ),
                "probabilities": np.asarray([0.9], dtype=np.float32),
                "differential": np.asarray([1.0], dtype=np.float32),
            },
            "validation": {
                "matrix": np.asarray([[1.0]], dtype=np.float32),
                "labels": np.asarray([1], dtype=np.int64),
                "metadata": pd.DataFrame({"sample_uid": ["sample_b"], "study_uid": ["study_alpha"]}),
                "entity_index": entity_index,
                "latent": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                "state_features": smoke_state_features(
                    np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                    np.asarray([0.9], dtype=np.float32),
                    np.asarray([[1.0]], dtype=np.float32),
                    np.asarray([1.0], dtype=np.float32),
                ),
                "probabilities": np.asarray([0.9], dtype=np.float32),
                "differential": np.asarray([1.0], dtype=np.float32),
            },
        }
    }
    target_program_rows = [
        {
            "target_program_uid": "target_program_glp1r",
            "canonical_entity_id": "GENE1",
            "canonical_entity_label": "GLP1R",
            "mapped_feature_count": "1",
            "coverage_fraction": "1.0",
            "target_evidence_score": "0.9",
            "target_class": "receptor",
        }
    ]
    regimen_rows = [
        {
            "regimen_uid": "ranked_regimen",
            "compound_uid_set": "compound_ranked",
            "target_uid_set": "target_ranked",
            "target_symbol_set": "GLP1R",
            "route": "oral",
            "schedule": "daily",
            "dose_level_id": "dose_100",
            "aggregate_target_evidence_score": "0.7",
            "clue_reversal_proxy_score": "0.7",
            "pk_feasibility_score": "0.7",
            "regimen_size": "1",
            "selection_reason": "ranked_selection",
        },
        {
            "regimen_uid": "case_regimen",
            "compound_uid_set": "compound_case",
            "target_uid_set": "target_case",
            "target_symbol_set": "GLP1R",
            "route": "intrathecal",
            "schedule": "weekly",
            "dose_level_id": "dose_100",
            "aggregate_target_evidence_score": "0.1",
            "clue_reversal_proxy_score": "0.1",
            "pk_feasibility_score": "0.1",
            "regimen_size": "1",
            "selection_reason": "case_study_seed",
        },
    ]
    target_rows = [{"target_uid": "target_case", "canonical_entity_id": "GENE1"}]
    compound_rows = [
        {"compound_uid": "compound_case", "preferred_name": "EXENATIDE"},
        {"compound_uid": "compound_ranked", "preferred_name": "RANKED"},
    ]
    strict_case_rows = [{"anchor_targets": ["GLP1R"], "anchor_compounds": ["EXENATIDE"]}]
    _, selected_regimen_rows, _ = train_screening_system.build_training_targets(
        modality_payload,
        target_program_rows,
        regimen_rows,
        target_rows,
        compound_rows,
        training_target_count=1,
        training_regimen_count=1,
        strict_case_rows=strict_case_rows,
    )
    assert selected_regimen_rows[0]["selection_reason"] == "case_study_seed"


def test_build_training_targets_preserves_anchor_route_schedule_grid() -> None:
    entity_index = pd.DataFrame({"canonical_entity_id": ["GENE1"]})
    modality_payload = {
        "transcriptomics": {
            "train": {
                "matrix": np.asarray([[1.0]], dtype=np.float32),
                "labels": np.asarray([1], dtype=np.int64),
                "metadata": pd.DataFrame({"sample_uid": ["sample_a"], "study_uid": ["study_alpha"]}),
                "entity_index": entity_index,
                "latent": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                "state_features": smoke_state_features(
                    np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                    np.asarray([0.9], dtype=np.float32),
                    np.asarray([[1.0]], dtype=np.float32),
                    np.asarray([1.0], dtype=np.float32),
                ),
                "probabilities": np.asarray([0.9], dtype=np.float32),
                "differential": np.asarray([1.0], dtype=np.float32),
            },
            "validation": {
                "matrix": np.asarray([[1.0]], dtype=np.float32),
                "labels": np.asarray([1], dtype=np.int64),
                "metadata": pd.DataFrame({"sample_uid": ["sample_b"], "study_uid": ["study_alpha"]}),
                "entity_index": entity_index,
                "latent": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                "state_features": smoke_state_features(
                    np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                    np.asarray([0.9], dtype=np.float32),
                    np.asarray([[1.0]], dtype=np.float32),
                    np.asarray([1.0], dtype=np.float32),
                ),
                "probabilities": np.asarray([0.9], dtype=np.float32),
                "differential": np.asarray([1.0], dtype=np.float32),
            },
        }
    }
    target_program_rows = [
        {
            "target_program_uid": "target_program_app",
            "canonical_entity_id": "GENE1",
            "canonical_entity_label": "APP",
            "mapped_feature_count": "1",
            "coverage_fraction": "1.0",
            "target_evidence_score": "0.9",
            "target_class": "receptor",
        }
    ]
    route_names = ["intrathecal", "intravenous", "oral"]
    schedule_names = ["daily", "twice_daily", "weekly"]
    regimen_rows: list[dict[str, str]] = []
    for route_name in route_names:
        for schedule_name in schedule_names:
            regimen_rows.append(
                {
                    "regimen_uid": f"case_{route_name}_{schedule_name}",
                    "compound_uid_set": "compound_case",
                    "target_uid_set": "target_case",
                    "target_symbol_set": "APP",
                    "route": route_name,
                    "schedule": schedule_name,
                    "dose_level_id": "dose_100",
                    "aggregate_target_evidence_score": "0.2",
                    "clue_reversal_proxy_score": "0.2",
                    "pk_feasibility_score": "0.2",
                    "regimen_size": "1",
                    "selection_reason": "case_study_seed",
                }
            )
    for regimen_index in range(24):
        regimen_rows.append(
            {
                "regimen_uid": f"ranked_{regimen_index}",
                "compound_uid_set": f"compound_ranked_{regimen_index}",
                "target_uid_set": "target_case",
                "target_symbol_set": "APP",
                "route": "intravenous",
                "schedule": "daily",
                "dose_level_id": "dose_100",
                "aggregate_target_evidence_score": "1.0",
                "clue_reversal_proxy_score": "1.0",
                "pk_feasibility_score": "1.0",
                "regimen_size": "1",
                "selection_reason": "ranked_selection",
            }
        )
    target_rows = [{"target_uid": "target_case", "canonical_entity_id": "GENE1"}]
    compound_rows = [{"compound_uid": "compound_case", "preferred_name": "LECANEMAB"}]
    strict_case_rows = [{"anchor_targets": ["APP"], "anchor_compounds": ["LECANEMAB"]}]
    _, selected_regimen_rows, _ = train_screening_system.build_training_targets(
        modality_payload,
        target_program_rows,
        regimen_rows,
        target_rows,
        compound_rows,
        training_target_count=1,
        training_regimen_count=9,
        strict_case_rows=strict_case_rows,
    )
    selected_frame = pd.DataFrame(selected_regimen_rows)
    assert set(selected_frame["route"].tolist()) == set(route_names)
    assert set(selected_frame["schedule"].tolist()) == set(schedule_names)
    assert set(selected_frame["compound_uid_set"].tolist()) == {"compound_case"}
