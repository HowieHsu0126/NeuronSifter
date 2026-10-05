"""Registered exogenous conditioning for full-horizon world-model training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.model.dynamics_store import load_zarr_rows, write_zarr_array


def conditioning_config(config: dict) -> dict:
    return config["screening_pipeline"]["screening_benchmark"]["training"][
        "zarr_loader"
    ]["conditioning"]


def validate_conditioning_manifest(manifest: dict, config: dict) -> dict[int, dict]:
    spec = conditioning_config(config)
    if manifest.get("schema_version") != spec["schema_version"]:
        raise ValueError(
            "Missing or incompatible independent occupancy conditioning schema."
        )
    if manifest.get("origin") != "exogenous_structured_action_pkpd":
        raise ValueError(
            "Occupancy must originate from independent structured-action PK/PD inputs."
        )
    if (
        manifest.get("occupancy_units") != "fraction"
        or manifest.get("time_units") != "hours"
    ):
        raise ValueError(
            "Conditioning requires occupancy fractions and physical hours."
        )
    if not manifest.get("producer") or not manifest.get("source_manifest"):
        raise ValueError("Conditioning producer and source manifest must be declared.")
    channel_ids = config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"][
        "conditioning"
    ]["occupancy_channel_ids"]
    if manifest.get("occupancy_channel_ids") != channel_ids:
        raise ValueError(
            "Occupancy target-channel ordering differs from the model contract."
        )
    rows = {}
    keys = set()
    for record in manifest.get("records", []):
        required = config["schemas"]["occupancy_conditioned_trajectory"][
            "required_record_fields"
        ]
        if any(not str(record.get(key, "")).strip() for key in required):
            raise ValueError(
                "Conditioning record lacks trajectory/action/parameter/source identity."
            )
        if (
            record.get("qc_grade") not in {"A", "B"}
            or record.get("qc_use") != "main_pretraining"
        ):
            raise ValueError(
                "Conditional main training admits QC Grade A/B main_pretraining only."
            )
        index = int(record["trajectory_index"])
        key = int(record["conditioning_sample_key"])
        if index < 0 or index in rows or key < 0 or key in keys:
            raise ValueError(
                "Conditioning indices and nonnegative sample keys must be unique."
            )
        if (
            not np.isfinite(float(record["sample_weight"]))
            or float(record["sample_weight"]) <= 0
        ):
            raise ValueError(
                "Conditioning requires a finite positive QC sample weight."
            )
        rows[index] = record
        keys.add(key)
    if not rows:
        raise ValueError("Conditioning manifest is empty.")
    return rows


def read_conditioning_manifest(store: Path, config: dict) -> dict[int, dict]:
    path = store / conditioning_config(config)["manifest_filename"]
    if not path.is_file():
        raise ValueError(f"Independent occupancy conditioning missing: {path}")
    return validate_conditioning_manifest(json.loads(path.read_text()), config)


def validate_conditioning_arrays(
    arrays: dict[str, np.ndarray], trajectory_shape: tuple, config: dict
) -> None:
    n, t, r, c, s = trajectory_shape
    channels = int(
        config["neuronsifter_scale_registry"]["quantized_mddm_main_rollout"][
            "conditioning"
        ]["occupancy_channel_count"]
    )
    expected = {
        "occupancy_program": (n, t, r, c, channels),
        "occupancy_support": (n, t, r, c, channels),
        "posterior_context": (n, s),
        "time_hours": (n, t),
        "sample_keys": (n,),
    }
    for name, shape in expected.items():
        if name not in arrays or tuple(arrays[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values.")
    occupancy = arrays["occupancy_program"]
    if (occupancy < 0).any() or (occupancy > 1).any():
        raise ValueError("Occupancy fractions must lie in [0,1].")
    if (
        arrays["occupancy_support"].dtype != np.bool_
        or not arrays["occupancy_support"].all()
    ):
        raise ValueError(
            "Unavailable occupancy is not a supported zero; full conditioning is required."
        )
    times = arrays["time_hours"]
    if t < 2 or (times[:, 0] != 0).any() or (np.diff(times, axis=1) <= 0).any():
        raise ValueError(
            "Teacher grids must start at zero and increase strictly in hours."
        )
    if (
        not np.issubdtype(arrays["sample_keys"].dtype, np.integer)
        or (arrays["sample_keys"] < 0).any()
    ):
        raise ValueError("Conditioning sample keys must be nonnegative integers.")


def load_conditioning_rows(
    store: Path,
    indices: np.ndarray,
    trajectory_shape: tuple,
    config: dict,
    records: dict[int, dict],
    split_name: str,
    expected_uids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    for position, index in enumerate(indices):
        record = records.get(int(index))
        if record is None or record["split_name"] != split_name:
            raise ValueError(
                "Conditioning selection is absent or outside the requested split."
            )
        if (
            expected_uids is not None
            and record["trajectory_uid"] != expected_uids[position]
        ):
            raise ValueError(
                "QC trajectory identity does not match conditioning record."
            )
    spec = conditioning_config(config)
    names = {
        "occupancy_program": "occupancy_array",
        "occupancy_support": "support_array",
        "posterior_context": "posterior_array",
        "time_hours": "time_array",
        "sample_keys": "sample_key_array",
    }
    arrays = {
        key: load_zarr_rows(store, spec[field], indices) for key, field in names.items()
    }
    validate_conditioning_arrays(arrays, trajectory_shape, config)
    for position, index in enumerate(indices):
        if int(arrays["sample_keys"][position]) != int(
            records[int(index)]["conditioning_sample_key"]
        ):
            raise ValueError(
                "Conditioning sample key differs from its provenance record."
            )
    return arrays


def write_conditioning_arrays(
    store: Path,
    arrays: dict[str, np.ndarray],
    trajectory_shape: tuple,
    config: dict,
    manifest: dict,
) -> None:
    """Attach independent arrays to a newly built store; never overwrite evidence."""
    spec = conditioning_config(config)
    records = validate_conditioning_manifest(manifest, config)
    if set(records) != set(range(trajectory_shape[0])):
        raise ValueError(
            "Conditioning records must align one-to-one with stored trajectories."
        )
    validate_conditioning_arrays(arrays, trajectory_shape, config)
    names = {
        "occupancy_program": "occupancy_array",
        "occupancy_support": "support_array",
        "posterior_context": "posterior_array",
        "time_hours": "time_array",
        "sample_keys": "sample_key_array",
    }
    if (store / spec["manifest_filename"]).exists() or any(
        (store / spec[field]).exists() for field in names.values()
    ):
        raise FileExistsError("Refusing to overwrite existing conditioning artifacts.")
    for index, record in records.items():
        if int(arrays["sample_keys"][index]) != int(record["conditioning_sample_key"]):
            raise ValueError("Conditioning sample key differs from the manifest.")
    for name, field in names.items():
        array = arrays[name]
        write_zarr_array(store, spec[field], array, (1, *array.shape[1:]))
    write_json(store / spec["manifest_filename"], manifest)
