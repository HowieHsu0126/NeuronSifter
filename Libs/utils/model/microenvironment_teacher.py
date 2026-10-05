from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Libs.utils.model.dynamics_fields import compartment_id_list, region_id_list, species_id_list, time_grid_hours
from Libs.utils.model.dynamics_store import initialize_zarr_store, write_zarr_array, write_zarr_array_from_loader
from Libs.utils.data.occupancy_conditioning import write_conditioning_arrays
from Libs.utils.data.raw_data_registry import ensure_directory, resolve_repo_path


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
    conditioning_arrays: dict[str, np.ndarray] | None = None,
    conditioning_manifest: dict | None = None,
) -> None:
    if trajectories.ndim != 5:
        raise ValueError(f"Trajectory tensor must be [simulation,time,region,compartment,species], got {trajectories.shape}.")
    if uncertainty.shape != (trajectories.shape[0], trajectories.shape[-1]):
        raise ValueError(f"Uncertainty tensor must be [simulation,species], got {uncertainty.shape} for trajectories {trajectories.shape}.")
    initialize_microenvironment_zarr(store_path, config, metadata)
    chunk_shape = microenvironment_chunk_shape(config)
    write_zarr_array(store_path, "trajectories", trajectories.astype(np.float32), chunk_shape, attrs={"axis_names": ["simulation", "time", "region", "compartment", "species"]})
    write_zarr_array(store_path, "teacher_uncertainty", uncertainty.astype(np.float32), (chunk_shape[0], trajectories.shape[-1]), attrs={"axis_names": ["simulation", "species"]})
    if conditioning_arrays is not None or conditioning_manifest is not None:
        if conditioning_arrays is None or conditioning_manifest is None:
            raise ValueError("Independent conditioning arrays and provenance must be supplied together.")
        write_conditioning_arrays(store_path, conditioning_arrays, trajectories.shape, config, conditioning_manifest)


def write_microenvironment_zarr_from_loaders(
    store_path: Path,
    trajectory_shape: tuple[int, int, int, int, int],
    uncertainty_shape: tuple[int, int],
    config: dict,
    metadata: dict[str, object],
    trajectory_chunk_loader,
    uncertainty_chunk_loader,
) -> None:
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
