from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from Libs.utils.model.dynamics_store import initialize_chunk_store, load_chunked_rows, load_full_array, write_chunk_sequence
from Libs.utils.data.raw_data_registry import read_tsv, repo_root, resolve_repo_path, write_tsv
from Libs.utils.model.screening_core import normalize_text, stable_uid
from Libs.utils.data.structural_prior_registry import (
    structural_prior_lookup,
    structural_prior_tensor,
    structural_prior_uid_lookup,
)
from Libs.utils.evaluation.strict_case_discovery import regimen_anchor_mask


def spatial_atlas_root(config: dict) -> Path:
    return resolve_repo_path(config["spatial_atlas_registry"]["local_root"])


def dynamics_root(config: dict) -> Path:
    return resolve_repo_path(config["dynamics_ready_registry"]["local_root"])


def compartment_rows(config: dict) -> list[dict[str, object]]:
    return list(config["screening_pipeline"]["pkpd_ready_registry"]["compartments"])


def build_spatial_region_rows(config: dict) -> list[dict[str, object]]:
    return [dict(row) for row in config["spatial_atlas_registry"]["regions"]]


def build_spatial_region_edge_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for edge_row in config["spatial_atlas_registry"]["region_edges"]:
        rows.append(
            {
                "edge_uid": stable_uid(
                    "spatial_edge",
                    str(edge_row["source_region_id"]),
                    str(edge_row["target_region_id"]),
                    str(edge_row["edge_type"]),
                ),
                **edge_row,
            }
        )
    return rows


def build_spatial_region_compartment_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region_row in build_spatial_region_rows(config):
        for current_compartment in compartment_rows(config):
            rows.append(
                {
                    "region_compartment_uid": stable_uid(
                        "region_compartment",
                        str(region_row["region_id"]),
                        str(current_compartment["compartment_id"]),
                    ),
                    "region_id": str(region_row["region_id"]),
                    "compartment_id": str(current_compartment["compartment_id"]),
                    "compartment_role": str(current_compartment["compartment_role"]),
                    "note": f"{region_row['region_id']}::{current_compartment['compartment_id']}",
                }
            )
    return rows


def build_spatial_observation_mapping_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mapping_row in config["spatial_atlas_registry"]["observation_mappings"]:
        rows.append(
            {
                "mapping_uid": stable_uid(
                    "observation_mapping",
                    str(mapping_row["modality"]),
                    str(mapping_row["tissue_keyword"]),
                    str(mapping_row["observed_region_id"]),
                    str(mapping_row["observed_compartment_id"]),
                ),
                **mapping_row,
            }
        )
    return rows


def load_spatial_atlas_catalogs(config: dict) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    root = spatial_atlas_root(config)
    relative_paths = config["spatial_atlas_registry"]["relative_paths"]
    return (
        read_tsv(root / relative_paths["region_catalog"]),
        read_tsv(root / relative_paths["region_edge_catalog"]),
        read_tsv(root / relative_paths["region_compartment_catalog"]),
        read_tsv(root / relative_paths["observation_mapping_catalog"]),
    )


def load_dynamics_catalogs(config: dict) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    root = dynamics_root(config)
    relative_paths = config["dynamics_ready_registry"]["relative_paths"]
    return (
        read_tsv(root / relative_paths["species_catalog"]),
        read_tsv(root / relative_paths["parameter_catalog"]),
        read_tsv(root / relative_paths["simulation_catalog"]),
        read_tsv(root / relative_paths["sample_alignment_catalog"]),
    )


def region_id_list(config: dict) -> list[str]:
    return [str(row["region_id"]) for row in config["spatial_atlas_registry"]["regions"]]


def compartment_id_list(config: dict) -> list[str]:
    return [str(row["compartment_id"]) for row in compartment_rows(config)]


def species_id_list(config: dict) -> list[str]:
    return [str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"]]


def time_grid_hours(config: dict) -> np.ndarray:
    time_config = config["dynamics_ready_registry"]["time_grid"]
    return np.linspace(
        float(time_config["start_hours"]),
        float(time_config["stop_hours"]),
        int(time_config["step_count"]),
        dtype=np.float32,
    )


def species_index_lookup(config: dict) -> dict[str, int]:
    return {species_id: index for index, species_id in enumerate(species_id_list(config))}


def profile_baseline_value(profile: dict[str, float], species_id: str) -> float:
    return float(profile[f"baseline_{species_id}"])


def optional_species_index(species_lookup: dict[str, int], species_id: str) -> int | None:
    return species_lookup[species_id] if species_id in species_lookup else None


def bounded_species_update(value: np.ndarray, species_id: str, priors: dict[str, object]) -> np.ndarray:
    upper_bounds = dict(priors["state_upper_bound_by_species"])
    upper_bound = float(upper_bounds[species_id]) if species_id in upper_bounds else None
    return np.clip(value, a_min=0.0, a_max=upper_bound)


def observation_region_assignment(
    modality: str,
    tissue_or_biofluid: str,
    mapping_rows: list[dict[str, str]],
) -> tuple[str, str]:
    normalized_tissue = normalize_text(tissue_or_biofluid)
    candidate_rows = [
        row
        for row in mapping_rows
        if str(row["modality"]) == modality and normalize_text(str(row["tissue_keyword"])) in normalized_tissue
    ]
    if candidate_rows:
        chosen_row = candidate_rows[0]
        return str(chosen_row["observed_region_id"]), str(chosen_row["observed_compartment_id"])
    if modality == "transcriptomics":
        return "middle_temporal_gyrus", "isf"
    return "systemic_circulation", "plasma"


def baseline_severity_score(sample_diagnosis_label: str, case_control_label: str) -> float:
    normalized_diagnosis = normalize_text(sample_diagnosis_label)
    normalized_case_control = normalize_text(case_control_label)
    if "alzheimer" in normalized_diagnosis:
        return 0.85
    if normalized_case_control == "case":
        return 0.75
    return 0.20


def simulation_condition_for_sample(sample_diagnosis_label: str, case_control_label: str) -> str:
    normalized_case_control = normalize_text(case_control_label)
    if normalized_case_control != "case":
        return "control"
    return "ad_positive"


def positive_condition_name(disease_program: str) -> str:
    return "ad_positive"


def build_species_rows(config: dict) -> list[dict[str, object]]:
    return [dict(row) for row in config["dynamics_ready_registry"]["species"]]


def build_parameter_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    priors = config["dynamics_ready_registry"]["field_dynamics_priors"]
    provenance_config = config["dynamics_ready_registry"].get("compartment_parameter_provenance", {})
    pkpd_root = resolve_repo_path(config["screening_pipeline"]["pkpd_ready_registry"]["local_root"])
    pkpd_parameter_catalog_path = pkpd_root / config["screening_pipeline"]["pkpd_ready_registry"]["relative_paths"]["parameter_prior_catalog"]
    pkpd_parameter_lookup: dict[tuple[str, str], dict[str, str]] = {}
    if pkpd_parameter_catalog_path.exists():
        for row in read_tsv(pkpd_parameter_catalog_path):
            pkpd_parameter_lookup[(str(row["parameter_name"]), str(row["compartment_id"]))] = row
    for region_row in build_spatial_region_rows(config):
        for current_compartment in compartment_rows(config):
            for species_row in config["dynamics_ready_registry"]["species"]:
                base_value = float(priors["diffusion_scale_by_compartment"][current_compartment["compartment_id"]])
                if species_row["species_id"] in {"misfolded_protein_load", "aggregate_burden", "neuroinflammation"}:
                    base_value = float(priors["clearance_scale_by_compartment"][current_compartment["compartment_id"]])
                provenance_row = provenance_config.get(current_compartment["compartment_id"], {})
                source_parameter_name = str(provenance_row.get("source_parameter_name", "")).strip()
                source_compartment_id = str(
                    provenance_row.get("source_compartment_id", current_compartment["compartment_id"])
                ).strip()
                source_row = pkpd_parameter_lookup.get((source_parameter_name, source_compartment_id), {})
                source_repository = str(source_row.get("source_repository", "screening_pipeline")).strip() or "screening_pipeline"
                note_parts = ["AD-first mesoscale transport prior"]
                if source_parameter_name:
                    note_parts.append(f"source_parameter={source_parameter_name}")
                if source_row.get("central_value", ""):
                    note_parts.append(f"source_central_value={source_row['central_value']}")
                rows.append(
                    {
                        "parameter_uid": stable_uid(
                            "dynamics_parameter",
                            str(region_row["region_id"]),
                            str(current_compartment["compartment_id"]),
                            str(species_row["species_id"]),
                        ),
                        "parameter_name": "mesoscale_transport_scale",
                        "region_id": str(region_row["region_id"]),
                        "compartment_id": str(current_compartment["compartment_id"]),
                        "species_id": str(species_row["species_id"]),
                        "central_value": base_value,
                        "lower_bound": max(0.0, base_value * 0.5),
                        "upper_bound": base_value * 1.5 + 1e-6,
                        "source_repository": source_repository,
                        "note": "; ".join(note_parts),
                    }
                )
    return rows


def region_adjacency_matrix(region_rows: list[dict[str, str]], edge_rows: list[dict[str, str]]) -> np.ndarray:
    region_ids = [str(row["region_id"]) for row in region_rows]
    region_lookup = {region_id: index for index, region_id in enumerate(region_ids)}
    adjacency = np.eye(len(region_ids), dtype=np.float32)
    for edge_row in edge_rows:
        source_index = region_lookup[str(edge_row["source_region_id"])]
        target_index = region_lookup[str(edge_row["target_region_id"])]
        scale = float(edge_row["transport_scale"])
        adjacency[source_index, target_index] = scale
        adjacency[target_index, source_index] = max(adjacency[target_index, source_index], scale)
    row_sums = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(row_sums, 1e-6)


def compartment_adjacency_matrix(config: dict) -> np.ndarray:
    rows = compartment_rows(config)
    compartment_ids = [str(row["compartment_id"]) for row in rows]
    compartment_lookup = {compartment_id: index for index, compartment_id in enumerate(compartment_ids)}
    adjacency = np.eye(len(compartment_ids), dtype=np.float32)
    for current_row in rows:
        upstream_id = str(current_row["upstream_compartment_id"])
        if not upstream_id:
            continue
        current_index = compartment_lookup[str(current_row["compartment_id"])]
        upstream_index = compartment_lookup[upstream_id]
        adjacency[current_index, upstream_index] = 1.0
        adjacency[upstream_index, current_index] = 1.0
    row_sums = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(row_sums, 1e-6)


def node_adjacency_matrix(config: dict) -> np.ndarray:
    region_rows = build_spatial_region_rows(config)
    edge_rows = build_spatial_region_edge_rows(config)
    region_ids = [str(row["region_id"]) for row in region_rows]
    compartment_ids = compartment_id_list(config)
    region_adj = region_adjacency_matrix(region_rows, edge_rows)
    compartment_adj = compartment_adjacency_matrix(config)
    node_count = len(region_ids) * len(compartment_ids)
    adjacency = np.zeros((node_count, node_count), dtype=np.float32)
    for region_index in range(len(region_ids)):
        for compartment_index in range(len(compartment_ids)):
            source_node = region_index * len(compartment_ids) + compartment_index
            for other_region_index in range(len(region_ids)):
                target_node = other_region_index * len(compartment_ids) + compartment_index
                adjacency[source_node, target_node] = max(adjacency[source_node, target_node], region_adj[region_index, other_region_index])
            for other_compartment_index in range(len(compartment_ids)):
                target_node = region_index * len(compartment_ids) + other_compartment_index
                adjacency[source_node, target_node] = max(adjacency[source_node, target_node], compartment_adj[compartment_index, other_compartment_index])
    row_sums = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(row_sums, 1e-6)


def compound_name_lookup(compound_rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        str(row["compound_uid"]): str(row["preferred_name"]).strip().upper()
        for row in compound_rows
        if str(row.get("preferred_name", "")).strip()
    }


def regimen_mechanism_strength(
    regimen_row: dict[str, str],
    compound_rows: list[dict[str, str]],
    config: dict,
) -> float:
    mechanism_defaults = config["dynamics_ready_registry"]["regimen_mechanism_defaults"]
    target_coefficients = mechanism_defaults["target_coefficients"]
    compound_overrides = mechanism_defaults["compound_overrides"]
    compound_lookup = compound_name_lookup(compound_rows)
    compound_names = [
        compound_lookup[compound_uid]
        for compound_uid in str(regimen_row["compound_uid_set"]).split("|")
        if compound_uid in compound_lookup
    ]
    override_scores = [float(compound_overrides[compound_name]) for compound_name in compound_names if compound_name in compound_overrides]
    target_scores = [
        float(target_coefficients[target_symbol.upper()])
        for target_symbol in str(regimen_row["target_symbol_set"]).split("|")
        if target_symbol.upper() in target_coefficients
    ]
    affinity_value = float(regimen_row["aggregate_affinity_score"]) if str(regimen_row["aggregate_affinity_score"]).strip() not in {"", "nan"} else 0.0
    evidence_score = float(regimen_row["aggregate_target_evidence_score"])
    mechanism_score = max(override_scores) if override_scores else (np.mean(target_scores) if target_scores else 0.05)
    return float(max(0.02, mechanism_score + 0.03 * evidence_score + 0.02 * min(affinity_value, 2.0)))


def condition_profile(config: dict, condition_name: str) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in config["dynamics_ready_registry"]["condition_profiles"][condition_name].items()
    }


def region_prior_lookup(config: dict, disease_program: str) -> dict[str, dict[str, float]]:
    if not bool(config["screening_pipeline"]["screening_benchmark"]["component_switches"].get("structural_prior", True)):
        return {}
    return structural_prior_lookup(config, disease_program)


def dominant_structural_prior_uid(config: dict, disease_program: str, regimen_row: dict[str, str]) -> str:
    profile_lookup = structural_prior_uid_lookup(config, disease_program)
    target_symbols = {value.strip().upper() for value in str(regimen_row["target_symbol_set"]).split("|") if value.strip()}
    if disease_program == "ad_control" and target_symbols & {"APP", "BACE1", "MAPT", "APOE"}:
        return profile_lookup.get("hippocampus_ca1", next(iter(profile_lookup.values()), ""))
    return profile_lookup.get("middle_temporal_gyrus", next(iter(profile_lookup.values()), ""))


def dominant_local_response_head_uid(disease_program: str, regimen_row: dict[str, str]) -> str:
    target_symbols = {value.strip().upper() for value in str(regimen_row["target_symbol_set"]).split("|") if value.strip()}
    if target_symbols & {"APP", "BACE1", "APOE"}:
        return "entorhinal_hippocampal_proteostasis"
    return "ca3_ca1_circuit_instability"


def build_regimen_initial_state(
    config: dict,
    condition_name: str,
    region_rows: list[dict[str, str]],
    region_prior_by_region: dict[str, dict[str, float]],
    heterogeneity_multipliers: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the exact state consumed at time zero by the mesoscale recurrence."""

    region_ids = [str(row["region_id"]) for row in region_rows]
    compartment_ids = compartment_id_list(config)
    species_ids = species_id_list(config)
    species_lookup = {species_id: index for index, species_id in enumerate(species_ids)}
    state = np.zeros((len(region_ids), len(compartment_ids), len(species_ids)), dtype=np.float32)
    profile = condition_profile(config, condition_name)
    for species_id in species_ids:
        state[:, :, species_lookup[species_id]] = profile_baseline_value(profile, species_id)

    pathology_species = {
        "misfolded_protein_load",
        "aggregate_burden",
        "neuroinflammation",
        "amyloid_beta_soluble",
        "amyloid_plaque",
        "phosphorylated_tau",
        "inflammatory_mediator",
        "calcium_concentration",
        "membrane_voltage",
        "spike_rate",
    }
    multipliers = {} if heterogeneity_multipliers is None else dict(heterogeneity_multipliers)
    pathology_multiplier = float(multipliers.get("pathology_baseline", 1.0))
    for species_id in pathology_species:
        if species_id in species_lookup:
            state[:, :, species_lookup[species_id]] *= pathology_multiplier

    solver_coefficients = dict(config["dynamics_ready_registry"]["field_dynamics_priors"]["mesoscopic_solver_coefficients"])
    for region_index, region_id in enumerate(region_ids):
        prior = region_prior_by_region.get(region_id, {})
        for species_id, gain_config in dict(solver_coefficients["initial_region_prior_gain"]).items():
            if species_id not in species_lookup:
                continue
            prior_key = str(gain_config["prior_key"])
            prior_value = (
                float(1.0 - prior.get("clearance_constraint_score_normalized", 0.0))
                if prior_key == "inverse_clearance_constraint_score_normalized"
                else float(prior.get(prior_key, 0.0))
            )
            state[region_index, :, species_lookup[species_id]] += float(gain_config["gain"]) * prior_value
    return state


def simulate_regimen_trajectory(
    config: dict,
    regimen_row: dict[str, str],
    compound_rows: list[dict[str, str]],
    disease_program: str,
    condition_name: str,
    region_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
    region_prior_by_region: dict[str, dict[str, float]],
    heterogeneity_multipliers: dict[str, float] | None = None,
) -> np.ndarray:
    region_ids = [str(row["region_id"]) for row in region_rows]
    compartment_ids = compartment_id_list(config)
    species_ids = [str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"]]
    species_lookup = {species_id: index for index, species_id in enumerate(species_ids)}
    time_values = time_grid_hours(config)
    region_adj = region_adjacency_matrix(region_rows, edge_rows)
    compartment_adj = compartment_adjacency_matrix(config)
    trajectory = np.zeros(
        (time_values.shape[0], len(region_ids), len(compartment_ids), len(species_ids)),
        dtype=np.float32,
    )
    profile = condition_profile(config, condition_name)
    priors = config["dynamics_ready_registry"]["field_dynamics_priors"]
    solver_coefficients = dict(priors["mesoscopic_solver_coefficients"])
    full_microenvironment_coupling = dict(solver_coefficients["full_microenvironment_coupling"])
    route_name = str(regimen_row["route"])
    schedule_name = str(regimen_row["schedule"])
    dose_multiplier = float(regimen_row["dose_multiplier"])
    target_symbols = {value.strip().upper() for value in str(regimen_row["target_symbol_set"]).split("|") if value.strip()}
    multipliers = {} if heterogeneity_multipliers is None else dict(heterogeneity_multipliers)
    schedule_multiplier = float(priors["schedule_frequency_multiplier"][schedule_name])
    mechanism_strength = regimen_mechanism_strength(regimen_row, compound_rows, config) * float(multipliers.get("mechanism_strength", 1.0))
    target_evidence_score = float(regimen_row["aggregate_target_evidence_score"])
    binding_gain = float(priors["target_binding_gain"]) * (float(solver_coefficients["binding_evidence_offset"]) + target_evidence_score)
    amyloid_reduction_gain = float(priors["amyloid_reduction_gain"]) * mechanism_strength
    plaque_clearance_gain = float(priors["plaque_clearance_gain"]) * mechanism_strength
    excitability_amyloid_gain = float(priors["excitability_amyloid_gain"])
    excitability_plaque_gain = float(priors["excitability_plaque_gain"])
    excitability_relief_gain = float(priors["excitability_relief_gain"])
    target_adverse_configs = [dict(priors["target_adverse_dynamics"][target_symbol]) for target_symbol in target_symbols if target_symbol in priors["target_adverse_dynamics"]]
    adverse_neuroinflammation_gain = sum(float(row["exposure_neuroinflammation_gain"]) for row in target_adverse_configs)
    adverse_circuit_instability_gain = sum(float(row["exposure_circuit_instability_gain"]) for row in target_adverse_configs)
    target_bound_relief_attenuation = sum(float(row["target_bound_relief_attenuation"]) for row in target_adverse_configs)
    if target_bound_relief_attenuation < 0.0 or target_bound_relief_attenuation >= 1.0:
        raise ValueError(f"target_bound_relief_attenuation must stay in [0, 1): {target_bound_relief_attenuation}")
    target_bound_relief_multiplier = 1.0 - target_bound_relief_attenuation
    local_response_enabled = bool(config["screening_pipeline"]["screening_benchmark"]["component_switches"].get("local_response_heads", True))
    if not local_response_enabled:
        excitability_amyloid_gain = 0.0
        excitability_plaque_gain = 0.0
        excitability_relief_gain = 0.0
    minimum_bound_fraction = float(priors["minimum_bound_fraction"])
    maximum_bound_fraction = float(priors["maximum_bound_fraction"])
    route_source = {
        compartment_id: float(priors["route_source_by_compartment"][route_name][compartment_id])
        * dose_multiplier
        * schedule_multiplier
        * float(multipliers.get("route_source", 1.0))
        for compartment_id in compartment_ids
    }
    diffusion_scale = {compartment_id: float(priors["diffusion_scale_by_compartment"][compartment_id]) for compartment_id in compartment_ids}
    clearance_scale = {
        compartment_id: float(priors["clearance_scale_by_compartment"][compartment_id]) * float(multipliers.get("clearance", 1.0))
        for compartment_id in compartment_ids
    }

    trajectory[0] = build_regimen_initial_state(
        config,
        condition_name,
        region_rows,
        region_prior_by_region,
        heterogeneity_multipliers=multipliers,
    )
    production_multiplier = float(multipliers.get("production_rate", 1.0))
    profile["misfolded_protein_production_rate"] *= production_multiplier
    profile["aggregate_accumulation_rate"] *= production_multiplier

    for time_index in range(1, time_values.shape[0]):
        delta_time = float(time_values[time_index] - time_values[time_index - 1]) / max(float(time_values[-1]), 1.0)
        previous_state = trajectory[time_index - 1].copy()
        current_state = previous_state.copy()
        free_drug = previous_state[:, :, species_lookup["free_drug"]]
        target_bound = previous_state[:, :, species_lookup["target_bound_drug"]]
        misfolded_protein = previous_state[:, :, species_lookup["misfolded_protein_load"]]
        aggregate_burden = previous_state[:, :, species_lookup["aggregate_burden"]]
        neuroinflammation = previous_state[:, :, species_lookup["neuroinflammation"]]

        for compartment_index, compartment_id in enumerate(compartment_ids):
            regional_component = free_drug[:, compartment_index]
            regional_diffusion = region_adj @ regional_component - regional_component
            compartment_component = free_drug[:, compartment_index]
            compartment_mix = np.zeros_like(compartment_component)
            for other_compartment_index in range(len(compartment_ids)):
                weight = compartment_adj[compartment_index, other_compartment_index]
                compartment_mix += weight * free_drug[:, other_compartment_index]
            compartment_transfer = compartment_mix - compartment_component
            source_vector = np.zeros((len(region_ids),), dtype=np.float32)
            source_vector[:] = float(solver_coefficients["source_background_fraction"]) * route_source[compartment_id]
            if compartment_id == "plasma":
                source_vector[region_ids.index("systemic_circulation")] = route_source[compartment_id]
            if compartment_id == "csf" and route_name == "intrathecal":
                source_vector[region_ids.index("hippocampus_ca1")] = route_source[compartment_id]
                source_vector[region_ids.index("entorhinal_cortex")] = float(solver_coefficients["intrathecal_entorhinal_fraction"]) * route_source[compartment_id]
            region_scale = np.asarray(
                [
                    float(solver_coefficients["transport_centrality_baseline"])
                    + float(solver_coefficients["transport_centrality_gain"])
                    * float(region_prior_by_region.get(region_id, {}).get("transport_centrality_score_normalized", 0.0))
                    for region_id in region_ids
                ],
                dtype=np.float32,
            )
            clearance_bias = np.asarray(
                [
                    float(solver_coefficients["clearance_bias_baseline"])
                    + float(solver_coefficients["clearance_bias_gain"])
                    * float(1.0 - region_prior_by_region.get(region_id, {}).get("clearance_constraint_score_normalized", 0.0))
                    for region_id in region_ids
                ],
                dtype=np.float32,
            )
            decay = clearance_scale[compartment_id] * clearance_bias * compartment_component
            updated_free_drug = np.clip(
                compartment_component
                + delta_time
                * (
                    diffusion_scale[compartment_id] * region_scale * regional_diffusion
                    + float(solver_coefficients["compartment_transfer_fraction"]) * diffusion_scale[compartment_id] * compartment_transfer
                    + region_scale * source_vector
                    - decay
                ),
                a_min=0.0,
                a_max=None,
            )
            current_state[:, compartment_index, species_lookup["free_drug"]] = updated_free_drug

            updated_target_bound = np.clip(
                target_bound[:, compartment_index]
                + delta_time
                * (
                    binding_gain * mechanism_strength * updated_free_drug
                    - float(solver_coefficients["bound_decay_rate"]) * target_bound[:, compartment_index]
                ),
                a_min=minimum_bound_fraction * updated_free_drug,
                a_max=maximum_bound_fraction,
            )
            current_state[:, compartment_index, species_lookup["target_bound_drug"]] = updated_target_bound

            updated_misfolded = np.clip(
                misfolded_protein[:, compartment_index]
                + delta_time
                * (
                    profile["misfolded_protein_production_rate"]
                    - amyloid_reduction_gain * target_bound_relief_multiplier * updated_target_bound
                    - float(solver_coefficients["soluble_pathology_clearance_scale"]) * clearance_scale[compartment_id] * clearance_bias * misfolded_protein[:, compartment_index]
                ),
                a_min=0.0,
                a_max=None,
            )
            current_state[:, compartment_index, species_lookup["misfolded_protein_load"]] = updated_misfolded

            updated_aggregate = np.clip(
                aggregate_burden[:, compartment_index]
                + delta_time
                * (
                    profile["aggregate_accumulation_rate"] * misfolded_protein[:, compartment_index]
                    - plaque_clearance_gain * target_bound_relief_multiplier * updated_target_bound
                    - float(solver_coefficients["aggregate_pathology_clearance_scale"]) * clearance_scale[compartment_id] * clearance_bias * aggregate_burden[:, compartment_index]
                ),
                a_min=0.0,
                a_max=None,
            )
            current_state[:, compartment_index, species_lookup["aggregate_burden"]] = updated_aggregate

            updated_inflammation = np.clip(
                neuroinflammation[:, compartment_index]
                + delta_time
                * (
                    float(priors["inflammation_coupling"]["misfolded_protein_load"]) * updated_misfolded
                    + float(priors["inflammation_coupling"]["aggregate_burden"]) * updated_aggregate
                    + adverse_neuroinflammation_gain * updated_free_drug
                    - float(solver_coefficients["target_bound_inflammation_relief"]) * target_bound_relief_multiplier * updated_target_bound
                    - float(priors["inflammation_decay"]) * neuroinflammation[:, compartment_index]
                ),
                a_min=0.0,
                a_max=float(priors["state_upper_bound_by_species"]["neuroinflammation"]),
            )
            current_state[:, compartment_index, species_lookup["neuroinflammation"]] = updated_inflammation

            clearance_proxy = np.clip(
                float(priors["clearance_flux_scale"]["free_drug"]) * clearance_scale[compartment_id] * updated_free_drug
                + float(priors["clearance_flux_scale"]["misfolded_relief"]) * np.maximum(misfolded_protein[:, compartment_index] - updated_misfolded, 0.0),
                a_min=0.0,
                a_max=None,
            )
            current_state[:, compartment_index, species_lookup["clearance_flux"]] = clearance_proxy

            current_state[:, compartment_index, species_lookup["circuit_instability"]] = np.clip(
                current_state[:, compartment_index, species_lookup["circuit_instability"]]
                + excitability_amyloid_gain * updated_misfolded
                + excitability_plaque_gain * updated_aggregate
                + float(priors["circuit_instability_scale"]["neuroinflammation"]) * updated_inflammation
                + adverse_circuit_instability_gain * updated_free_drug
                - excitability_relief_gain * target_bound_relief_multiplier * updated_target_bound,
                a_min=0.0,
                a_max=float(priors["state_upper_bound_by_species"]["circuit_instability"]),
            )
            soluble_index = optional_species_index(species_lookup, "amyloid_beta_soluble")
            if soluble_index is not None:
                coupling = dict(full_microenvironment_coupling["amyloid_beta_soluble"])
                soluble_state = previous_state[:, compartment_index, soluble_index]
                updated_soluble = bounded_species_update(
                    soluble_state
                    + delta_time
                    * (
                        float(coupling["production_gain"]) * updated_misfolded
                        - float(coupling["clearance_gain"]) * clearance_proxy * soluble_state
                        - float(coupling["target_bound_relief_gain"]) * target_bound_relief_multiplier * updated_target_bound
                        - float(coupling["plaque_conversion_gain"]) * soluble_state
                    ),
                    "amyloid_beta_soluble",
                    priors,
                )
                current_state[:, compartment_index, soluble_index] = updated_soluble
            else:
                updated_soluble = updated_misfolded
            plaque_index = optional_species_index(species_lookup, "amyloid_plaque")
            if plaque_index is not None:
                coupling = dict(full_microenvironment_coupling["amyloid_plaque"])
                plaque_state = previous_state[:, compartment_index, plaque_index]
                updated_plaque = bounded_species_update(
                    plaque_state
                    + delta_time
                    * (
                        float(coupling["soluble_conversion_gain"]) * updated_soluble
                        + float(coupling["source_species_gain"]) * updated_aggregate
                        - float(coupling["clearance_gain"]) * clearance_proxy * plaque_state
                        - float(coupling["target_bound_relief_gain"]) * target_bound_relief_multiplier * updated_target_bound
                    ),
                    "amyloid_plaque",
                    priors,
                )
                current_state[:, compartment_index, plaque_index] = updated_plaque
            else:
                updated_plaque = updated_aggregate
            tau_index = optional_species_index(species_lookup, "phosphorylated_tau")
            if tau_index is not None:
                coupling = dict(full_microenvironment_coupling["phosphorylated_tau"])
                tau_state = previous_state[:, compartment_index, tau_index]
                updated_tau = bounded_species_update(
                    tau_state
                    + delta_time
                    * (
                        float(coupling["amyloid_gain"]) * updated_soluble
                        + float(coupling["plaque_gain"]) * updated_plaque
                        + float(coupling["inflammation_gain"]) * updated_inflammation
                        - float(coupling["clearance_gain"]) * clearance_proxy * tau_state
                        - float(coupling["target_bound_relief_gain"]) * target_bound_relief_multiplier * updated_target_bound
                    ),
                    "phosphorylated_tau",
                    priors,
                )
                current_state[:, compartment_index, tau_index] = updated_tau
            else:
                updated_tau = updated_aggregate
            mediator_index = optional_species_index(species_lookup, "inflammatory_mediator")
            if mediator_index is not None:
                coupling = dict(full_microenvironment_coupling["inflammatory_mediator"])
                mediator_state = previous_state[:, compartment_index, mediator_index]
                updated_mediator = bounded_species_update(
                    mediator_state
                    + delta_time
                    * (
                        float(coupling["amyloid_gain"]) * updated_soluble
                        + float(coupling["plaque_gain"]) * updated_plaque
                        + float(coupling["tau_gain"]) * updated_tau
                        - float(coupling["decay_gain"]) * mediator_state
                        - float(coupling["target_bound_relief_gain"]) * target_bound_relief_multiplier * updated_target_bound
                    ),
                    "inflammatory_mediator",
                    priors,
                )
                current_state[:, compartment_index, mediator_index] = updated_mediator
            else:
                updated_mediator = updated_inflammation
            calcium_index = optional_species_index(species_lookup, "calcium_concentration")
            if calcium_index is not None:
                coupling = dict(full_microenvironment_coupling["calcium_concentration"])
                calcium_state = previous_state[:, compartment_index, calcium_index]
                updated_calcium = bounded_species_update(
                    calcium_state
                    + delta_time
                    * (
                        float(coupling["circuit_gain"]) * current_state[:, compartment_index, species_lookup["circuit_instability"]]
                        + float(coupling["inflammation_gain"]) * updated_mediator
                        - float(coupling["decay_gain"]) * calcium_state
                    ),
                    "calcium_concentration",
                    priors,
                )
                current_state[:, compartment_index, calcium_index] = updated_calcium
            else:
                updated_calcium = current_state[:, compartment_index, species_lookup["circuit_instability"]]
            voltage_index = optional_species_index(species_lookup, "membrane_voltage")
            if voltage_index is not None:
                coupling = dict(full_microenvironment_coupling["membrane_voltage"])
                voltage_state = previous_state[:, compartment_index, voltage_index]
                updated_voltage = bounded_species_update(
                    voltage_state
                    + delta_time
                    * (
                        float(coupling["circuit_gain"]) * current_state[:, compartment_index, species_lookup["circuit_instability"]]
                        + float(coupling["calcium_gain"]) * updated_calcium
                        - float(coupling["decay_gain"]) * voltage_state
                    ),
                    "membrane_voltage",
                    priors,
                )
                current_state[:, compartment_index, voltage_index] = updated_voltage
            else:
                updated_voltage = current_state[:, compartment_index, species_lookup["circuit_instability"]]
            spike_index = optional_species_index(species_lookup, "spike_rate")
            if spike_index is not None:
                coupling = dict(full_microenvironment_coupling["spike_rate"])
                spike_state = previous_state[:, compartment_index, spike_index]
                current_state[:, compartment_index, spike_index] = bounded_species_update(
                    spike_state
                    + delta_time
                    * (
                        float(coupling["voltage_gain"]) * updated_voltage
                        + float(coupling["calcium_gain"]) * updated_calcium
                        - float(coupling["decay_gain"]) * spike_state
                    ),
                    "spike_rate",
                    priors,
                )
            vascular_index = optional_species_index(species_lookup, "vascular_perivascular_transport")
            if vascular_index is not None:
                coupling = dict(full_microenvironment_coupling["vascular_perivascular_transport"])
                vascular_state = previous_state[:, compartment_index, vascular_index]
                current_state[:, compartment_index, vascular_index] = bounded_species_update(
                    vascular_state
                    + delta_time
                    * (
                        float(coupling["clearance_flux_gain"]) * clearance_proxy
                        + float(coupling["free_drug_gain"]) * updated_free_drug
                        - float(coupling["inflammation_penalty_gain"]) * updated_mediator
                        - float(coupling["decay_gain"]) * vascular_state
                    ),
                    "vascular_perivascular_transport",
                    priors,
                )
        trajectory[time_index] = current_state
    return trajectory.astype(np.float32)


def _chunk_regimen_rows(regimen_rows: list[dict[str, str]], chunk_count: int) -> list[list[dict[str, str]]]:
    if not regimen_rows:
        return []
    effective_chunk_count = max(1, min(int(chunk_count), len(regimen_rows)))
    chunk_size = int(math.ceil(len(regimen_rows) / effective_chunk_count))
    return [
        regimen_rows[index : index + chunk_size]
        for index in range(0, len(regimen_rows), chunk_size)
    ]


def _regimen_generation_batches(config: dict, regimen_rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    condition_count = len(active_generation_condition_names(config))
    replicate_count = active_generation_replicate_count(config)
    trajectory_chunk_size = int(config["dynamics_ready_registry"]["chunk_size"])
    trajectories_per_regimen = max(1, condition_count * replicate_count)
    regimen_batch_size = max(1, trajectory_chunk_size // trajectories_per_regimen)
    return [regimen_rows[index : index + regimen_batch_size] for index in range(0, len(regimen_rows), regimen_batch_size)]


def ad_first_readiness_config(config: dict) -> dict[str, object]:
    return dict(config["dynamics_ready_registry"]["ad_first_data_readiness"])


def active_generation_stage_config(config: dict) -> dict[str, object]:
    readiness = ad_first_readiness_config(config)
    stage_name = str(readiness["active_generation_stage"])
    return dict(readiness["generation_stages"][stage_name])


def active_generation_replicate_count(config: dict) -> int:
    return int(active_generation_stage_config(config)["replicate_count"])


def active_generation_condition_names(config: dict) -> list[str]:
    return [str(value) for value in ad_first_readiness_config(config)["condition_names"]]


def deterministic_unit_interval(hash_algorithm: str, *parts: str) -> float:
    hash_state = hashlib.new(hash_algorithm)
    hash_state.update("::".join(str(part) for part in parts).encode("utf-8"))
    digest = hash_state.hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def bounded_multiplier(centered_unit_value: float, span: float, minimum_value: float, maximum_value: float) -> float:
    return float(np.clip(1.0 + centered_unit_value * float(span), float(minimum_value), float(maximum_value)))


def heterogeneity_multipliers(config: dict, regimen_uid: str, condition_name: str, replicate_index: int) -> dict[str, float]:
    sampler = dict(ad_first_readiness_config(config)["heterogeneity_sampler"])
    seed = str(sampler["seed"])
    hash_algorithm = str(sampler["hash_algorithm"])
    minimum_value = float(sampler["minimum_multiplier"])
    maximum_value = float(sampler["maximum_multiplier"])
    span_by_name = {
        "route_source": float(sampler["route_source_multiplier_span"]),
        "pathology_baseline": float(sampler["pathology_baseline_multiplier_span"]),
        "production_rate": float(sampler["production_rate_multiplier_span"]),
        "clearance": float(sampler["clearance_multiplier_span"]),
        "mechanism_strength": float(sampler["mechanism_strength_multiplier_span"]),
    }
    shared_patient_names = {str(value) for value in sampler["shared_patient_multiplier_names"]}
    regimen_conditioned_names = {str(value) for value in sampler["regimen_conditioned_multiplier_names"]}
    declared_names = shared_patient_names | regimen_conditioned_names
    if declared_names != set(span_by_name):
        raise ValueError(f"heterogeneity_sampler multiplier names do not match configured spans: {sorted(declared_names ^ set(span_by_name))}")
    multiplier_payload: dict[str, float] = {}
    for name, span in span_by_name.items():
        hash_parts = [seed, condition_name, str(replicate_index), name]
        if name in regimen_conditioned_names:
            hash_parts.insert(1, regimen_uid)
        multiplier_payload[name] = bounded_multiplier(
            2.0 * deterministic_unit_interval(hash_algorithm, *hash_parts) - 1.0,
            span,
            minimum_value,
            maximum_value,
        )
    return multiplier_payload


def parameter_draw_uid(regimen_uid: str, condition_name: str, replicate_index: int, multipliers: dict[str, float]) -> str:
    multiplier_payload = "|".join(f"{name}={multipliers[name]:.6f}" for name in sorted(multipliers))
    return stable_uid("parameter_draw", regimen_uid, condition_name, str(replicate_index), multiplier_payload)


def _simulate_regimen_batch(
    task_payload: tuple[
        dict,
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
    ],
    ) -> list[tuple[dict[str, str], str, str, int, str, dict[str, float], np.ndarray]]:
    config, regimen_batch, compound_rows, region_rows, edge_rows = task_payload
    batch_payload: list[tuple[dict[str, str], str, str, int, str, dict[str, float], np.ndarray]] = []
    disease_program = str(ad_first_readiness_config(config)["primary_disease_program"])
    region_prior_by_region = region_prior_lookup(config, disease_program)
    replicate_count = active_generation_replicate_count(config)
    for regimen_row in regimen_batch:
        regimen_uid = str(regimen_row["regimen_uid"])
        for replicate_index in range(replicate_count):
            for condition_name in active_generation_condition_names(config):
                multipliers = heterogeneity_multipliers(config, regimen_uid, condition_name, replicate_index)
                draw_uid = parameter_draw_uid(regimen_uid, condition_name, replicate_index, multipliers)
                batch_payload.append(
                    (
                        regimen_row,
                        disease_program,
                        condition_name,
                        replicate_index,
                        draw_uid,
                        multipliers,
                        simulate_regimen_trajectory(
                            config,
                            regimen_row,
                            compound_rows,
                            disease_program,
                            condition_name,
                            region_rows,
                            edge_rows,
                            region_prior_by_region,
                            multipliers,
                        ),
                    )
                )
    return batch_payload



def dynamics_process_context(config: dict) -> mp.context.BaseContext:
    process_start_method = str(config["dynamics_ready_registry"]["process_start_method"])
    return mp.get_context(process_start_method)

def simulation_rows_and_store(
    config: dict,
    regimen_rows: list[dict[str, str]],
    compound_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], Path]:
    root = dynamics_root(config)
    relative_paths = config["dynamics_ready_registry"]["relative_paths"]
    store_path = root / relative_paths["trajectory_store"]
    region_rows = build_spatial_region_rows(config)
    edge_rows = build_spatial_region_edge_rows(config)
    region_ids = np.asarray([row["region_id"] for row in region_rows], dtype=str)
    compartment_ids = np.asarray(compartment_id_list(config), dtype=str)
    species_ids = np.asarray([row["species_id"] for row in config["dynamics_ready_registry"]["species"]], dtype=str)
    time_values = time_grid_hours(config)
    rows: list[dict[str, object]] = []
    chunk_size = int(config["dynamics_ready_registry"]["chunk_size"])
    trajectory_shape = (
        len(regimen_rows) * active_generation_replicate_count(config) * len(active_generation_condition_names(config)),
        int(time_values.shape[0]),
        int(region_ids.shape[0]),
        int(compartment_ids.shape[0]),
        int(species_ids.shape[0]),
    )
    parameter_evidence_status = str(ad_first_readiness_config(config)["parameter_evidence_policy"]["evidence_status"])
    parallel_workers = int(config["dynamics_ready_registry"].get("parallel_workers", 1))

    initialize_chunk_store(
        store_path,
        axis_names=["simulation", "time", "region", "compartment", "species"],
        coordinate_payload={
            "time": time_values,
            "region": region_ids,
            "compartment": compartment_ids,
            "species": species_ids,
        },
        metadata={
            "disease_programs": str(ad_first_readiness_config(config)["primary_disease_program"]),
            "contract_version": str(config["dynamics_ready_registry"]["contract_version"]),
            "active_generation_stage": str(ad_first_readiness_config(config)["active_generation_stage"]),
            "replicate_count": active_generation_replicate_count(config),
            "parameter_evidence_status": parameter_evidence_status,
        },
    )

    def simulation_chunk_iterable():
        trajectory_chunk: list[np.ndarray] = []
        trajectory_index = 0

        def consume_payload(simulation_payload):
            nonlocal trajectory_chunk, trajectory_index
            for regimen_row, disease_program, condition_name, replicate_index, draw_uid, multipliers, trajectory in simulation_payload:
                rows.append(
                    {
                        "simulation_uid": stable_uid("simulation", disease_program, str(regimen_row["regimen_uid"]), condition_name, str(replicate_index), draw_uid),
                        "disease_program": disease_program,
                        "simulation_condition": condition_name,
                        "heterogeneity_replicate_id": f"replicate_{replicate_index}",
                        "parameter_draw_uid": draw_uid,
                        "parameter_evidence_status": parameter_evidence_status,
                        "regimen_uid": str(regimen_row["regimen_uid"]),
                        "route": str(regimen_row["route"]),
                        "schedule": str(regimen_row["schedule"]),
                        "dose_level_id": str(regimen_row["dose_level_id"]),
                        "dose_multiplier": str(regimen_row["dose_multiplier"]),
                        "observed_region_set": "|".join(region_ids.tolist()),
                        "structural_prior_uid": dominant_structural_prior_uid(config, disease_program, regimen_row),
                        "local_response_head_uid": dominant_local_response_head_uid(disease_program, regimen_row),
                        "trajectory_index": trajectory_index,
                        "trajectory_store_path": root / relative_paths["trajectory_store"],
                        "time_step_count": int(time_values.shape[0]),
                        "region_count": int(region_ids.shape[0]),
                        "compartment_count": int(compartment_ids.shape[0]),
                        "species_count": int(species_ids.shape[0]),
                        "note": f"{disease_program}::{condition_name}::{regimen_row['regimen_uid']}::replicate_{replicate_index}::provisional_parameter_evidence::{multipliers}",
                    }
                )
                trajectory_chunk.append(trajectory.astype(np.float32))
                trajectory_index += 1
                if len(trajectory_chunk) == chunk_size:
                    chunk = np.stack(trajectory_chunk, axis=0).astype(np.float32)
                    trajectory_chunk = []
                    yield chunk

        if parallel_workers > 1 and len(regimen_rows) > 1:
            regimen_batches = _regimen_generation_batches(config, regimen_rows)
            process_context = dynamics_process_context(config)
            with ProcessPoolExecutor(max_workers=parallel_workers, mp_context=process_context) as executor:
                for batch_payload in executor.map(
                    _simulate_regimen_batch,
                    [(config, regimen_batch, compound_rows, region_rows, edge_rows) for regimen_batch in regimen_batches],
                ):
                    yield from consume_payload(batch_payload)
        else:
            yield from consume_payload(_simulate_regimen_batch((config, regimen_rows, compound_rows, region_rows, edge_rows)))
        if trajectory_chunk:
            yield np.stack(trajectory_chunk, axis=0).astype(np.float32)

    write_chunk_sequence(store_path, "trajectories", trajectory_shape, "float32", chunk_size, simulation_chunk_iterable())
    return rows, store_path


def build_sample_alignment_rows(config: dict) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    mapping_rows = build_spatial_observation_mapping_rows(config)
    model_input_root = resolve_repo_path(config["model_input_registry"]["local_root"])
    world_model_rows = read_tsv(model_input_root / config["model_input_registry"]["relative_paths"]["study_bundle_catalog"])
    rows: list[dict[str, object]] = []
    rows_by_study: dict[str, list[dict[str, object]]] = defaultdict(list)
    for study_row in world_model_rows:
        modality = "transcriptomics" if str(study_row["source_key"]) in {"geo", "arrayexpress"} else "metabolomics"
        sample_index = pd.read_parquet(resolve_repo_path(study_row["bundle_root"]) / "sample_index.parquet")
        for sample_row in sample_index.itertuples(index=False):
            observed_region_id, observed_compartment_id = observation_region_assignment(
                modality,
                str(sample_row.tissue_or_biofluid),
                mapping_rows,
            )
            severity_score = baseline_severity_score(
                str(sample_row.sample_diagnosis_label),
                str(sample_row.case_control_label),
            )
            alignment_row = {
                "field_alignment_uid": stable_uid("field_alignment", str(study_row["study_uid"]), str(sample_row.sample_uid)),
                "study_uid": str(study_row["study_uid"]),
                "source_key": str(study_row["source_key"]),
                "source_accession": str(study_row["source_accession"]),
                "sample_uid": str(sample_row.sample_uid),
                "modality": modality,
                "observed_region_set": observed_region_id,
                "baseline_severity_score": severity_score,
                "simulation_condition": simulation_condition_for_sample(
                    str(sample_row.sample_diagnosis_label),
                    str(sample_row.case_control_label),
                ),
                "field_alignment_bundle_path": "",
                "note": observed_compartment_id,
            }
            rows.append(alignment_row)
            rows_by_study[str(study_row["study_uid"])].append(alignment_row)
    return rows, rows_by_study


def write_study_alignment_catalogs(config: dict, rows_by_study: dict[str, list[dict[str, object]]]) -> dict[str, Path]:
    root = dynamics_root(config)
    relative_paths = config["dynamics_ready_registry"]["relative_paths"]
    study_alignment_root = root / relative_paths["study_alignment_root"]
    path_by_study: dict[str, Path] = {}
    for study_uid, study_rows in rows_by_study.items():
        path = study_alignment_root / f"{study_uid}_field_alignment.tsv"
        write_tsv(path, config["schemas"]["dynamics_sample_alignment_fields"], study_rows)
        path_by_study[study_uid] = path
    return path_by_study


def load_sample_alignment_lookup(config: dict) -> dict[tuple[str, str], dict[str, str]]:
    _, _, _, sample_alignment_rows = load_dynamics_catalogs(config)
    return {
        (str(row["study_uid"]), str(row["sample_uid"])): row
        for row in sample_alignment_rows
    }


def observed_region_mask(metadata_frame: pd.DataFrame, config: dict) -> np.ndarray:
    region_ids = region_id_list(config)
    sample_lookup = load_sample_alignment_lookup(config)
    mask = np.zeros((metadata_frame.shape[0], len(region_ids)), dtype=np.float32)
    region_lookup = {region_id: index for index, region_id in enumerate(region_ids)}
    for row_index, metadata_row in enumerate(metadata_frame.itertuples(index=False)):
        key = (str(metadata_row.study_uid), str(metadata_row.sample_uid))
        if key not in sample_lookup:
            continue
        observed_regions = str(sample_lookup[key]["observed_region_set"]).split("|")
        for region_id in observed_regions:
            if region_id in region_lookup:
                mask[row_index, region_lookup[region_id]] = 1.0
    return mask


def selected_regimen_simulation_tensors(
    config: dict,
    regimen_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    _, _, simulation_rows, _ = load_dynamics_catalogs(config)
    store_path = dynamics_root(config) / config["dynamics_ready_registry"]["relative_paths"]["trajectory_store"]
    active_disease_program = str(config["screening_pipeline"]["screening_benchmark"]["active_disease_program"])
    lookup: dict[tuple[str, str], int] = {}
    for row in simulation_rows:
        if str(row["disease_program"]) != active_disease_program:
            continue
        replicate_id = str(row.get("heterogeneity_replicate_id", "replicate_0"))
        if replicate_id and replicate_id != "replicate_0":
            continue
        lookup[(str(row["regimen_uid"]), str(row["simulation_condition"]))] = int(row["trajectory_index"])
    payload: dict[str, np.ndarray] = {}
    regimen_uids = regimen_frame["regimen_uid"].astype(str).tolist()
    for condition_name, payload_key in [("control", "control"), (positive_condition_name(active_disease_program), "positive")]:
        row_indices = np.asarray([lookup[(regimen_uid, condition_name)] for regimen_uid in regimen_uids], dtype=np.int64)
        payload[payload_key] = load_chunked_rows(store_path, "trajectories", row_indices).astype(np.float32)
    return payload


def batch_simulation_targets(
    simulation_targets_by_condition: dict[str, np.ndarray],
    patient_labels: np.ndarray,
) -> np.ndarray:
    targets = [
        simulation_targets_by_condition["positive" if int(label_value) == 1 else "control"]
        for label_value in patient_labels.astype(np.int64).tolist()
    ]
    return np.stack(targets, axis=0).astype(np.float32)


def field_summary_by_species(field_trajectory: np.ndarray, species_lookup: dict[str, int]) -> dict[str, np.ndarray]:
    final_state = field_trajectory[:, :, -1]
    return {
        "free_drug": final_state[..., species_lookup["free_drug"]].mean(axis=(2, 3)),
        "target_bound_drug": final_state[..., species_lookup["target_bound_drug"]].mean(axis=(2, 3)),
        "misfolded_protein_load": final_state[..., species_lookup["misfolded_protein_load"]].mean(axis=(2, 3)),
        "aggregate_burden": final_state[..., species_lookup["aggregate_burden"]].mean(axis=(2, 3)),
        "neuroinflammation": final_state[..., species_lookup["neuroinflammation"]].mean(axis=(2, 3)),
        "clearance_flux": final_state[..., species_lookup["clearance_flux"]].mean(axis=(2, 3)),
        "circuit_instability": final_state[..., species_lookup["circuit_instability"]].mean(axis=(2, 3)),
    }


def standardized_effect_from_summary(summary: dict[str, np.ndarray], metric_weights: dict[str, float]) -> np.ndarray:
    aggregate_reduction = np.clip(1.0 - summary["aggregate_burden"], a_min=0.0, a_max=1.0)
    protein_reduction = np.clip(1.0 - summary["misfolded_protein_load"], a_min=0.0, a_max=1.0)
    inflammation_reduction = np.clip(1.0 - summary["neuroinflammation"], a_min=0.0, a_max=1.0)
    circuit_reduction = np.clip(1.0 - summary["circuit_instability"], a_min=0.0, a_max=1.0)
    return np.clip(
        float(metric_weights["aggregate_reduction"]) * aggregate_reduction
        + float(metric_weights["protein_reduction"]) * protein_reduction
        + float(metric_weights["inflammation_reduction"]) * inflammation_reduction
        + float(metric_weights["circuit_instability_reduction"]) * circuit_reduction,
        a_min=0.0,
        a_max=1.0,
    )


def direction_hit_from_effect(predicted_effect: float, expected_direction: str, metric_thresholds: dict[str, float]) -> float:
    direction = expected_direction.strip().lower()
    if direction == "favorable":
        return float(predicted_effect >= float(metric_thresholds["favorable_min_effect"]))
    if direction == "weak_or_negative":
        return float(predicted_effect <= float(metric_thresholds["weak_or_negative_max_effect"]))
    if direction == "modest_or_mixed":
        return float(float(metric_thresholds["mixed_min_effect"]) <= predicted_effect <= float(metric_thresholds["mixed_max_effect"]))
    return 0.0


def expected_rank_group(expected_direction: str) -> float:
    direction = expected_direction.strip().lower()
    if direction == "favorable":
        return 2.0
    if direction == "modest_or_mixed":
        return 1.0
    if direction == "weak_or_negative":
        return 0.0
    return 0.0


def predicted_rank_group(predicted_effect: float, metric_thresholds: dict[str, float]) -> float:
    if predicted_effect >= float(metric_thresholds["rank_high_min_effect"]):
        return 2.0
    if predicted_effect >= float(metric_thresholds["rank_mid_min_effect"]):
        return 1.0
    return 0.0


def intervention_mask(
    regimen_frame: pd.DataFrame,
    compound_rows: list[dict[str, str]],
    intervention_label: str,
) -> np.ndarray:
    compound_lookup = compound_name_lookup(compound_rows)
    intervention_text = intervention_label.strip().upper()
    masks = []
    for row in regimen_frame.itertuples(index=False):
        compound_names = {
            compound_lookup[compound_uid]
            for compound_uid in str(row.compound_uid_set).split("|")
            if compound_uid in compound_lookup
        }
        masks.append(intervention_text in compound_names)
    return np.asarray(masks, dtype=bool)


def quantitative_case_validation_rows(
    config: dict,
    regimen_frame: pd.DataFrame,
    compound_rows: list[dict[str, str]],
    field_trajectory: np.ndarray,
    tri_scores: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    root = dynamics_root(config)
    relative_paths = config["dynamics_ready_registry"]["relative_paths"]
    validation_rows = read_tsv(root / relative_paths["case_validation_catalog"])
    species_lookup = species_index_lookup(config)
    summary = field_summary_by_species(field_trajectory, species_lookup)
    metric_weights = {key: float(value) for key, value in config["dynamics_ready_registry"]["case_validation_metric_weights"].items()}
    metric_thresholds = {key: float(value) for key, value in config["dynamics_ready_registry"]["case_validation_metric_thresholds"].items()}
    standardized_effect = standardized_effect_from_summary(summary, metric_weights)
    axis_frame = pd.DataFrame(route_ordering_axis_rows(int(standardized_effect.shape[1]), regimen_frame, config))
    aligned_tri_scores: np.ndarray | None = None
    if tri_scores is not None:
        aligned_tri_scores = np.asarray(tri_scores, dtype=np.float64)
        if aligned_tri_scores.ndim != 2:
            raise ValueError(f"tri_scores must be a 2D array, got shape {aligned_tri_scores.shape}.")
        if aligned_tri_scores.shape[1] == regimen_frame.shape[0] and axis_frame.shape[0] != regimen_frame.shape[0]:
            regimen_uid_by_index = [str(row.regimen_uid) for row in regimen_frame.itertuples(index=False)]
            axis_regimen_indices = [regimen_uid_by_index.index(str(row.regimen_uid)) for row in axis_frame.itertuples(index=False)]
            aligned_tri_scores = aligned_tri_scores[:, axis_regimen_indices]
        if aligned_tri_scores.shape[1] != axis_frame.shape[0]:
            raise ValueError(f"tri_scores regimen axis {aligned_tri_scores.shape[1]} does not match trajectory axis {axis_frame.shape[0]}.")
    tri_threshold = float(config["screening_pipeline"]["screening_benchmark"]["tri_threshold"])
    rows: list[dict[str, float | str]] = []
    for validation_row in validation_rows:
        mask = intervention_mask(axis_frame, compound_rows, str(validation_row["intervention_label"]))
        if not mask.any():
            continue
        predicted_effect = float(standardized_effect[:, mask].mean())
        expected_effect = float(validation_row["expected_standardized_effect"])
        expected_direction = str(validation_row["expected_effect_direction"])
        direction_hit = direction_hit_from_effect(predicted_effect, expected_direction, metric_thresholds)
        rank_recovery_hit = float(predicted_rank_group(predicted_effect, metric_thresholds) == expected_rank_group(expected_direction))
        if aligned_tri_scores is None:
            tri_gate_consistency = 0.0
        else:
            intervention_tri = aligned_tri_scores[:, mask]
            accepted_fraction = float((intervention_tri >= tri_threshold).mean())
            tri_gate_consistency = accepted_fraction if expected_direction == "favorable" else 1.0 - accepted_fraction
        clinical_summary = str(validation_row.get("clinical_endpoint_summary", ""))
        rows.append(
            {
                "case_study_name": str(validation_row["case_study_name"]),
                "intervention_label": str(validation_row["intervention_label"]),
                "expected_effect_direction": expected_direction,
                "predicted_standardized_effect": predicted_effect,
                "standardized_effect_error": abs(predicted_effect - expected_effect),
                "biomarker_effect_error": abs(predicted_effect - expected_effect),
                "direction_hit": direction_hit,
                "rank_recovery_hit": rank_recovery_hit,
                "tri_gate_consistency": tri_gate_consistency,
                "clinical_endpoint_qualitative_presence": float(bool(clinical_summary)),
                "clinical_endpoint_alignment_summary": clinical_summary,
            }
        )
    return rows


def route_ordering_evaluation_config(config: dict) -> dict[str, object]:
    route_config = dict(config["dynamics_ready_registry"]["field_dynamics_priors"]["route_ordering_evaluation"])
    required_fields = [
        "disease_program",
        "simulation_condition",
        "region_id",
        "compartment_ids",
        "species_id",
        "time_index",
        "normalization_policy",
        "minimum_normalization_denominator",
    ]
    missing_fields = [field_name for field_name in required_fields if field_name not in route_config]
    if missing_fields:
        raise ValueError(f"Missing route_ordering_evaluation fields: {missing_fields}")
    if not route_config["compartment_ids"]:
        raise ValueError("route_ordering_evaluation.compartment_ids must not be empty.")
    return route_config


def route_ordering_axis_rows(field_axis_count: int, regimen_frame: pd.DataFrame, config: dict) -> list[dict[str, object]]:
    regimen_rows = regimen_frame.to_dict(orient="records")
    if field_axis_count == len(regimen_rows):
        return regimen_rows
    route_config = route_ordering_evaluation_config(config)
    catalog_path = dynamics_root(config) / config["dynamics_ready_registry"]["relative_paths"]["simulation_catalog"]
    simulation_rows = read_tsv(catalog_path)
    selected_rows = [
        dict(row)
        for row in simulation_rows
        if str(row["disease_program"]) == str(route_config["disease_program"])
        and str(row["simulation_condition"]) == str(route_config["simulation_condition"])
    ]
    selected_rows = sorted(selected_rows, key=lambda row: int(row["trajectory_index"]))
    if len(selected_rows) != field_axis_count:
        raise ValueError(
            "Route-ordering trajectory axis does not match the configured simulation catalog slice: "
            f"axis={field_axis_count}, catalog_rows={len(selected_rows)}."
        )
    regimen_by_uid = {str(row["regimen_uid"]): row for row in regimen_rows}
    aligned_rows: list[dict[str, object]] = []
    for row in selected_rows:
        regimen_uid = str(row["regimen_uid"])
        if regimen_uid not in regimen_by_uid:
            raise ValueError(f"Route-ordering catalog row references regimen outside the scoring frame: {regimen_uid}")
        regimen_row = regimen_by_uid[regimen_uid]
        for field_name in ["route", "schedule", "dose_multiplier"]:
            if str(row[field_name]) != str(regimen_row[field_name]):
                raise ValueError(f"Route-ordering catalog/regimen mismatch for {regimen_uid}.{field_name}.")
        aligned_rows.append({**regimen_row, **row})
    return aligned_rows


def route_ordering_normalization(row: dict[str, object], schedule_multiplier_lookup: dict[str, object], policy: str, minimum_denominator: float) -> float:
    if policy == "dose_schedule_frequency":
        schedule_name = str(row["schedule"])
        if schedule_name not in schedule_multiplier_lookup:
            raise ValueError(f"Route-ordering schedule not declared in schedule_frequency_multiplier: {schedule_name}")
        denominator = float(row["dose_multiplier"]) * float(schedule_multiplier_lookup[schedule_name])
    elif policy == "dose_only":
        denominator = float(row["dose_multiplier"])
    elif policy == "none":
        denominator = 1.0
    else:
        raise ValueError(f"Unsupported route_ordering_evaluation.normalization_policy: {policy}")
    if denominator <= float(minimum_denominator):
        raise ValueError(f"Route-ordering normalization denominator is not positive enough: {denominator}")
    return denominator


def route_ordering_exposure(field_trajectory: np.ndarray, regimen_frame: pd.DataFrame, config: dict) -> dict[str, float]:
    if field_trajectory.ndim != 6:
        raise ValueError(f"route_ordering_exposure expects a 6D trajectory tensor, got shape {field_trajectory.shape}.")
    route_config = route_ordering_evaluation_config(config)
    species_lookup = species_index_lookup(config)
    region_lookup = {region_id: index for index, region_id in enumerate(region_id_list(config))}
    compartment_lookup = {compartment_id: index for index, compartment_id in enumerate(compartment_id_list(config))}
    species_id = str(route_config["species_id"])
    region_id = str(route_config["region_id"])
    compartment_ids = [str(compartment_id) for compartment_id in route_config["compartment_ids"]]
    for required_species_id in [species_id]:
        if required_species_id not in species_lookup:
            raise ValueError(f"Route-ordering species not declared in dynamics species: {required_species_id}")
    if region_id not in region_lookup:
        raise ValueError(f"Route-ordering region not declared in spatial atlas: {region_id}")
    missing_compartment_ids = [compartment_id for compartment_id in compartment_ids if compartment_id not in compartment_lookup]
    if missing_compartment_ids:
        raise ValueError(f"Route-ordering compartments not declared in PK/PD registry: {missing_compartment_ids}")
    time_index = int(route_config["time_index"])
    if time_index < 0 or time_index >= field_trajectory.shape[2]:
        raise ValueError(f"Route-ordering time_index {time_index} outside trajectory length {field_trajectory.shape[2]}.")
    region_index = region_lookup[region_id]
    compartment_indices = [compartment_lookup[compartment_id] for compartment_id in compartment_ids]
    axis_rows = route_ordering_axis_rows(int(field_trajectory.shape[1]), regimen_frame, config)
    exposures: dict[str, list[float]] = defaultdict(list)
    schedule_multiplier_lookup = config["dynamics_ready_registry"]["field_dynamics_priors"]["schedule_frequency_multiplier"]
    normalization_policy = str(route_config["normalization_policy"])
    minimum_denominator = float(route_config["minimum_normalization_denominator"])
    for axis_index, axis_row in enumerate(axis_rows):
        compartment_values = [
            field_trajectory[:, axis_index, time_index, region_index, compartment_index, species_lookup[species_id]]
            for compartment_index in compartment_indices
        ]
        route_signal = np.stack(compartment_values, axis=0).mean(axis=0)
        normalization = route_ordering_normalization(axis_row, schedule_multiplier_lookup, normalization_policy, minimum_denominator)
        exposures[str(axis_row["route"])].append(float(np.mean(route_signal) / normalization))
    return {
        route_name: float(np.mean(route_values))
        for route_name, route_values in exposures.items()
    }


def load_all_trajectories(config: dict) -> np.ndarray:
    store_path = dynamics_root(config) / config["dynamics_ready_registry"]["relative_paths"]["trajectory_store"]
    return load_full_array(store_path, "trajectories").astype(np.float32)


def default_region_prior(config: dict) -> np.ndarray:
    if not bool(config["screening_pipeline"]["screening_benchmark"]["component_switches"].get("structural_prior", True)):
        return np.zeros((len(region_id_list(config)), 6), dtype=np.float32)
    active_disease_program = str(config["screening_pipeline"]["screening_benchmark"]["active_disease_program"])
    return structural_prior_tensor(config, region_id_list(config), active_disease_program).astype(np.float32)


def update_world_model_rows_with_alignment(
    world_model_rows: list[dict[str, str]],
    sample_alignment_rows: list[dict[str, object]],
    path_by_study: dict[str, Path],
) -> list[dict[str, object]]:
    observed_regions_by_study: dict[str, set[str]] = defaultdict(set)
    for row in sample_alignment_rows:
        observed_regions_by_study[str(row["study_uid"])].update(str(row["observed_region_set"]).split("|"))
    updated_rows: list[dict[str, object]] = []
    for row in world_model_rows:
        study_uid = str(row["study_uid"])
        updated_rows.append(
            {
                **row,
                "field_alignment_uid": stable_uid("study_field_alignment", study_uid),
                "field_alignment_bundle_path": str(path_by_study[study_uid].resolve().relative_to(repo_root())),
                "observed_region_set": "|".join(sorted(observed_regions_by_study[study_uid])),
            }
        )
    return updated_rows


def strict_case_dynamics_metrics(
    config: dict,
    case_row: dict[str, object],
    regimen_frame: pd.DataFrame,
    compound_rows: list[dict[str, str]],
    field_trajectory: np.ndarray,
    tri_accepted_mask: np.ndarray,
) -> dict[str, float]:
    species_lookup = species_index_lookup(config)
    summary = field_summary_by_species(field_trajectory, species_lookup)
    axis_frame = pd.DataFrame(route_ordering_axis_rows(int(summary["free_drug"].shape[1]), regimen_frame, config))
    anchor_mask = regimen_anchor_mask(axis_frame, case_row, compound_rows)
    if not anchor_mask.any():
        return {
            "field_plausible_anchor_count": 0.0,
            "quantitative_anchor_hit": 0.0,
            "structural_plausibility_hit": 0.0,
        }
    metric_weights = {key: float(value) for key, value in config["dynamics_ready_registry"]["case_validation_metric_weights"].items()}
    metric_thresholds = {key: float(value) for key, value in config["dynamics_ready_registry"]["case_validation_metric_thresholds"].items()}
    standardized_effect = standardized_effect_from_summary(summary, metric_weights).mean(axis=0)
    tri_mask = np.asarray(tri_accepted_mask, dtype=bool)
    if tri_mask.shape[0] == regimen_frame.shape[0] and axis_frame.shape[0] != regimen_frame.shape[0]:
        regimen_tri_lookup = {
            str(row.regimen_uid): bool(tri_mask[regimen_index])
            for regimen_index, row in enumerate(regimen_frame.itertuples(index=False))
        }
        tri_mask = np.asarray([regimen_tri_lookup[str(row.regimen_uid)] for row in axis_frame.itertuples(index=False)], dtype=bool)
    if tri_mask.shape[0] != axis_frame.shape[0]:
        raise ValueError(f"strict_case_dynamics_metrics tri mask length {tri_mask.shape[0]} does not match trajectory axis {axis_frame.shape[0]}.")
    anchor_effect = standardized_effect[anchor_mask]
    anchor_tri_mask = tri_mask[anchor_mask]
    structural_prior = region_prior_lookup(config, str(case_row["disease_program"]))
    region_support = float(structural_prior.get("hippocampus_ca1", {}).get("disease_vulnerability_score_normalized", 0.0))
    entorhinal_support = float(structural_prior.get("entorhinal_cortex", {}).get("disease_vulnerability_score_normalized", 0.0))
    return {
        "field_plausible_anchor_count": float(np.sum(anchor_effect >= float(metric_thresholds["favorable_min_effect"]))),
        "quantitative_anchor_hit": float(np.any((anchor_effect >= float(metric_thresholds["favorable_min_effect"])) & anchor_tri_mask)),
        "structural_plausibility_hit": float(max(region_support, entorhinal_support) >= float(metric_thresholds["structural_support_minimum"])),
    }
