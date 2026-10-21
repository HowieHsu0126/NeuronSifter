from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Libs.utils.model.dynamics_fields import compartment_id_list, region_id_list, species_id_list, time_grid_hours
from Libs.utils.model.dynamics_store import initialize_zarr_store, write_zarr_array, write_zarr_array_from_loader, initialize_chunk_store, write_chunked_array
from Libs.utils.data.raw_data_registry import ensure_directory, resolve_repo_path
from Libs.utils.data.native_diffusion_data import (ARRAY_NAMES, consumed_array_names, join_registry, validate_arrays,
    native_action_policy, scalar_exposure_view, verify_scalar_arrays)


TEACHER_ROLE_VALUES = {
    "gold_observed",
    "mechanism_teacher",
    "parameter_predicted_teacher",
    "legacy_baseline",
}


def teacher_registry_root(config: dict) -> Path:
    return resolve_repo_path(config["data"]["full_microenvironment_teacher_ready_registry"]["local_root"])


def teacher_relative_paths(config: dict) -> dict[str, str]:
    return dict(config["data"]["full_microenvironment_teacher_ready_registry"]["relative_paths"])


def teacher_contract_version(config: dict) -> str:
    return str(config["data"]["full_microenvironment_teacher_ready_registry"]["contract_version"])


def validate_teacher_role(role_name: str) -> None:
    if role_name not in TEACHER_ROLE_VALUES:
        raise ValueError(f"Unsupported teacher_role {role_name}; expected one of {sorted(TEACHER_ROLE_VALUES)}.")


def teacher_precision_from_uncertainty(config: dict, uncertainty: float) -> float:
    weighting = config["screening_pipeline"]["screening_benchmark"]["training"]["uncertainty_weighting"]
    precision = 1.0 / max(float(uncertainty), float(weighting["minimum_teacher_precision"]))
    return float(np.clip(precision, float(weighting["minimum_teacher_precision"]), float(weighting["maximum_teacher_precision"])))


def default_coordinate_payload(config: dict) -> dict[str, object]:
    return {
        "time_hours": time_grid_hours(config).astype(float).tolist(),
        "region_id": region_id_list(config),
        "compartment_id": compartment_id_list(config),
        "species_id": species_id_list(config),
    }


def microenvironment_chunk_shape(config: dict) -> tuple[int, int, int, int, int]:
    osp_chunking = config["dynamics_ready_registry"]["osp_pksim_runtime"]["zarr_chunking"]
    return (
        int(osp_chunking["simulation_chunk_size"]),
        int(osp_chunking["time_chunk_size"]),
        len(region_id_list(config)),
        int(osp_chunking["compartment_chunk_size"]),
        int(osp_chunking["species_chunk_size"]),
    )


def initialize_microenvironment_zarr(store_path: Path, config: dict, metadata: dict[str, object]) -> None:
    ensure_directory(store_path)
    initialize_zarr_store(
        store_path,
        {
            "format_name": "zarr_v2",
            "contract_version": teacher_contract_version(config),
            "axis_names": ["simulation", "time", "region", "compartment", "species"],
            "coordinates": default_coordinate_payload(config),
            **metadata,
        },
    )


def write_microenvironment_zarr(
    store_path: Path,
    trajectories: np.ndarray,
    uncertainty: np.ndarray,
    config: dict,
    metadata: dict[str, object],
    native_payload: dict | None = None,
) -> None:
    if native_payload is not None:
        write_native_microenvironment_store(store_path, config, native_payload["registry"],
            {**native_payload["arrays"], "trajectory": trajectories, "uncertainty": uncertainty}, "zarr_v2", metadata)
        return
    if trajectories.ndim != 5:
        raise ValueError(f"Trajectory tensor must be [simulation,time,region,compartment,species], got {trajectories.shape}.")
    if uncertainty.shape != (trajectories.shape[0], trajectories.shape[-1]):
        raise ValueError(f"Uncertainty tensor must be [simulation,species], got {uncertainty.shape} for trajectories {trajectories.shape}.")
    initialize_microenvironment_zarr(store_path, config, metadata)
    chunk_shape = microenvironment_chunk_shape(config)
    write_zarr_array(store_path, "trajectories", trajectories.astype(np.float32), chunk_shape, attrs={"axis_names": ["simulation", "time", "region", "compartment", "species"]})
    write_zarr_array(store_path, "teacher_uncertainty", uncertainty.astype(np.float32), (chunk_shape[0], trajectories.shape[-1]), attrs={"axis_names": ["simulation", "species"]})


def write_microenvironment_zarr_from_loaders(
    store_path: Path,
    trajectory_shape: tuple[int, int, int, int, int],
    uncertainty_shape: tuple[int, int],
    config: dict,
    metadata: dict[str, object],
    trajectory_chunk_loader,
    uncertainty_chunk_loader,
    native_payload: dict | None = None,
) -> None:
    if native_payload is not None:
        # The same producer validates all independent channels before serialization.
        trajectories = trajectory_chunk_loader(tuple(slice(0, size) for size in trajectory_shape))
        uncertainty = uncertainty_chunk_loader(tuple(slice(0, size) for size in uncertainty_shape))
        write_microenvironment_zarr(store_path, trajectories, uncertainty, config, metadata, native_payload)
        return
    if len(trajectory_shape) != 5:
        raise ValueError(f"Trajectory tensor must be [simulation,time,region,compartment,species], got {trajectory_shape}.")
    if uncertainty_shape != (trajectory_shape[0], trajectory_shape[-1]):
        raise ValueError(f"Uncertainty tensor must be [simulation,species], got {uncertainty_shape} for trajectories {trajectory_shape}.")
    initialize_microenvironment_zarr(store_path, config, metadata)
    chunk_shape = microenvironment_chunk_shape(config)
    write_zarr_array_from_loader(
        store_path,
        "trajectories",
        trajectory_shape,
        chunk_shape,
        np.dtype(np.float32),
        trajectory_chunk_loader,
        attrs={"axis_names": ["simulation", "time", "region", "compartment", "species"]},
    )
    write_zarr_array_from_loader(
        store_path,
        "teacher_uncertainty",
        uncertainty_shape,
        (chunk_shape[0], trajectory_shape[-1]),
        np.dtype(np.float32),
        uncertainty_chunk_loader,
        attrs={"axis_names": ["simulation", "species"]},
    )


def json_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def write_native_microenvironment_store(store_path: Path, config: dict, registry: dict,
                                       arrays: dict, store_format: str = "zarr_v2",
                                       metadata: dict | None = None, *, normalization_context=None) -> None:
    """Write genuine producer channels with their registry joins; never infer inputs."""
    records = join_registry(registry, config, normalization_context=normalization_context)
    if native_action_policy(config) == "scalar_exposure":
        if registry.get("posterior_family") == "finite_joint_particles":
            if any(key in arrays for key in ("scalar_exposure_auc", "scalar_exposure_interval")):
                if len(registry["joint_sources"]) != 1:
                    raise ValueError("Scalar arrays require one explicit whole-particle source")
                verify_scalar_arrays(arrays, scalar_exposure_view(registry["joint_sources"][0]["particle_programs"]["records"]))
        else:
            verify_scalar_arrays(arrays, scalar_exposure_view(registry["programs"]))
    selected = {}
    for key in consumed_array_names(registry):
        owner, index_key = ("teacher", "trajectory_index") if key in {"trajectory", "uncertainty", "teacher_confidence", "quantity_support"} else (
            ("observation", "observation_index") if key in {"initial_state", "posterior_context"} else ("program", "program_index"))
        selected[key] = np.stack([np.asarray(arrays[key])[record[owner][index_key]] for record in records])
        path_key = "trajectory_store_path" if owner == "teacher" else "store_path"
        for record in records:
            if resolve_repo_path(record[owner][path_key]) != store_path.resolve() or record[owner]["store_format"] != store_format:
                raise ValueError("Native producer registry must bind the exact output store and format.")
    validate_arrays(selected, registry, records, config)
    attributes = {**(metadata or {}), "native_diffusion": registry, "coordinates": registry["grid"]}
    if store_format == "zarr_v2":
        initialize_microenvironment_zarr(store_path, config, attributes)
    elif store_format == "neuronsifter_chunk_store":
        initialize_chunk_store(store_path, ["simulation", "time", "region", "compartment", "species"], {}, attributes)
    else:
        raise ValueError(f"Unsupported native store format: {store_format}")
    row_chunk = microenvironment_chunk_shape(config)[0]
    for key, name in consumed_array_names(registry).items():
        array = np.asarray(arrays[key], dtype=np.bool_ if key == "quantity_support" else np.float32)
        if store_format == "zarr_v2":
            write_zarr_array(store_path, name, array, (row_chunk, *array.shape[1:]))
        else:
            write_chunked_array(store_path, name, array, row_chunk)
