from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, write_json, write_tsv
from Libs.utils.model.dynamics_store import chunk_filename, read_store_manifest, read_zarr_array_metadata, zarr_chunk_filename
from Libs.utils.model.screening_core import stable_uid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simulation QC, leakage, pairing, and readiness audit catalogs.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(REPO_ROOT))


def hash_split(value: str, split_policy: dict[str, object]) -> str:
    digest = hashlib.new(str(split_policy["hash_algorithm"]), value.encode("utf-8")).hexdigest()
    remainder = int(digest, 16) % int(split_policy["modulus"])
    if remainder in {int(item) for item in split_policy["validation_remainders"]}:
        return "validation"
    if remainder in {int(item) for item in split_policy["test_remainders"]}:
        return "test"
    return "train"


def load_manifest_splits(config: dict[str, object]) -> dict[tuple[str, str], str]:
    preprocessing = config["ai_ready_preprocessing"]
    manifest_path = resolve_repo_path(str(preprocessing["local_root"])) / str(preprocessing["relative_paths"]["simulation_manifest"])
    if not manifest_path.exists():
        return {}
    return {
        (str(row["source_collection"]), str(row["source_uid"])): str(row["split_name"])
        for row in read_tsv(manifest_path)
    }


def zarr_shape(store_path: Path, array_name: str) -> list[int]:
    return [int(value) for value in read_zarr_array_metadata(store_path, array_name)["shape"]]


def chunk_store_shape(store_path: Path, array_name: str) -> list[int]:
    manifest = read_store_manifest(store_path)
    return [int(value) for value in manifest["arrays"][array_name]["shape"]]


def store_shape(store_path: Path, array_name: str, store_format: str) -> list[int]:
    if store_format == "zarr_v2":
        return zarr_shape(store_path, array_name)
    if store_format == "neuronsifter_chunk_store":
        return chunk_store_shape(store_path, array_name)
    raise ValueError(f"Unsupported simulation audit store format: {store_format}")


def species_ids(config: dict[str, object]) -> list[str]:
    return [str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"]]


def index_set(values: list[str], names: list[str]) -> list[int]:
    value_lookup = {name: index for index, name in enumerate(values)}
    return [value_lookup[name] for name in names if name in value_lookup]


def numeric_stats(trajectory: np.ndarray, species_names: list[str], audit_config: dict[str, object]) -> dict[str, object]:
    numeric_qc = audit_config["numeric_qc"]
    mechanism_qc = audit_config["mechanism_qc"]
    trajectory_array = np.asarray(trajectory, dtype=np.float32)
    finite_flag = bool(np.isfinite(trajectory_array).all())
    finite_values = trajectory_array[np.isfinite(trajectory_array)]
    if finite_values.size == 0:
        min_value = math.nan
        max_value = math.nan
        std_value = math.nan
        terminal_abs_max = math.nan
    else:
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        std_value = float(finite_values.std())
        terminal_abs_max = float(np.abs(trajectory_array[-1]).max())
    if trajectory_array.shape[0] > 1:
        max_abs_jump = float(np.abs(np.diff(trajectory_array, axis=0)).max())
    else:
        max_abs_jump = 0.0
    concentration_indices = index_set(species_names, [str(item) for item in numeric_qc["concentration_like_species"]])
    pathology_indices = index_set(species_names, [str(item) for item in numeric_qc["pathology_like_species"]])
    spike_indices = index_set(species_names, [str(item) for item in numeric_qc["spike_rate_species"]])
    occupancy_indices = index_set(species_names, [str(item) for item in numeric_qc["occupancy_species"]])
    failure_flags: list[str] = []
    warning_flags: list[str] = []
    if bool(numeric_qc["reject_on_nan"]) and np.isnan(trajectory_array).any():
        failure_flags.append("nan_value")
    if bool(numeric_qc["reject_on_inf"]) and np.isinf(trajectory_array).any():
        failure_flags.append("inf_value")
    if concentration_indices and bool(numeric_qc["reject_negative_concentration"]) and float(trajectory_array[..., concentration_indices].min()) < 0.0:
        failure_flags.append("negative_concentration")
    if pathology_indices and bool(numeric_qc["reject_negative_pathology"]) and float(trajectory_array[..., pathology_indices].min()) < 0.0:
        failure_flags.append("negative_pathology")
    if spike_indices and bool(numeric_qc["reject_negative_spike_rate"]) and float(trajectory_array[..., spike_indices].min()) < 0.0:
        failure_flags.append("negative_spike_rate")
    if occupancy_indices:
        occupancy_values = trajectory_array[..., occupancy_indices]
        if float(occupancy_values.min()) < float(numeric_qc["occupancy_min"]) or float(occupancy_values.max()) > float(numeric_qc["occupancy_max"]):
            failure_flags.append("occupancy_out_of_bounds")
    for species_name, bounds in dict(mechanism_qc["ephys_ranges"]).items():
        species_index = species_names.index(species_name) if species_name in species_names else -1
        if species_index < 0:
            continue
        species_values = trajectory_array[..., species_index]
        if float(species_values.min()) < float(bounds["min"]) or float(species_values.max()) > float(bounds["max"]):
            warning_flags.append(f"{species_name}_outside_physiology_range")
    if finite_values.size > 0 and abs(max_value) > 0.0:
        baseline_abs_mean = abs(float(np.mean(np.abs(trajectory_array[0]))))
        terminal_ratio = terminal_abs_max / max(baseline_abs_mean, float(np.finfo(np.float32).eps))
        if terminal_ratio > float(numeric_qc["max_terminal_explosion_ratio"]):
            failure_flags.append("terminal_explosion")
    if finite_values.size > 0 and std_value < float(numeric_qc["min_dynamic_standard_deviation"]):
        warning_flags.append("low_dynamic_variance")
    return {
        "finite_flag": finite_flag,
        "min_value": min_value,
        "max_value": max_value,
        "standard_deviation": std_value,
        "max_abs_jump": max_abs_jump,
        "terminal_abs_max": terminal_abs_max,
        "content_hash": hashlib.sha256(np.ascontiguousarray(trajectory_array).view(np.uint8)).hexdigest(),
        "numeric_failure_flags": failure_flags,
        "numeric_warning_flags": warning_flags,
    }


def scan_zarr_trajectories(store_path: Path, indices: set[int], array_name: str, species_names: list[str], audit_config: dict[str, object]) -> dict[int, dict[str, object]]:
    metadata = read_zarr_array_metadata(store_path, array_name)
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    dtype = np.dtype(str(metadata["dtype"]))
    row_chunk_size = chunks[0]
    indices_by_chunk: dict[int, list[int]] = defaultdict(list)
    for row_index in sorted(indices):
        indices_by_chunk[int(row_index // row_chunk_size)].append(int(row_index))
    result: dict[int, dict[str, object]] = {}
    array_root = store_path / array_name
    tail_ranges = [range(int(math.ceil(axis_size / chunk_size))) for axis_size, chunk_size in zip(shape[1:], chunks[1:])]
    for row_chunk_index, row_indices in indices_by_chunk.items():
        row_start = row_chunk_index * row_chunk_size
        row_stop = min(row_start + row_chunk_size, shape[0])
        block = np.zeros((row_stop - row_start, *shape[1:]), dtype=dtype)
        for tail_chunk_index in product(*tail_ranges):
            chunk_index = (row_chunk_index, *tail_chunk_index)
            chunk_slices_tuple = tuple(
                slice(index * chunk_size, min((index + 1) * chunk_size, axis_size))
                for index, axis_size, chunk_size in zip(chunk_index, shape, chunks)
            )
            chunk_shape = tuple(slice_item.stop - slice_item.start for slice_item in chunk_slices_tuple)
            chunk = np.fromfile(array_root / zarr_chunk_filename(chunk_index), dtype=dtype).reshape(chunk_shape)
            block_slices = tuple(slice_item for slice_item in chunk_slices_tuple[1:])
            block[(slice(None), *block_slices)] = chunk
        for row_index in row_indices:
            result[row_index] = numeric_stats(block[row_index - row_start], species_names, audit_config)
    return result


def scan_chunk_store_trajectories(store_path: Path, indices: set[int], array_name: str, species_names: list[str], audit_config: dict[str, object]) -> dict[int, dict[str, object]]:
    manifest = read_store_manifest(store_path)
    array_manifest = manifest["arrays"][array_name]
    chunk_size = int(array_manifest["chunk_size"])
    chunk_root = store_path / str(array_manifest["relative_root"])
    indices_by_chunk: dict[int, list[int]] = defaultdict(list)
    for row_index in sorted(indices):
        indices_by_chunk[int(row_index // chunk_size)].append(int(row_index))
    result: dict[int, dict[str, object]] = {}
    for chunk_index, row_indices in indices_by_chunk.items():
        chunk = np.load(chunk_root / chunk_filename(chunk_index), allow_pickle=False)
        for row_index in row_indices:
            result[row_index] = numeric_stats(chunk[row_index % chunk_size], species_names, audit_config)
    return result


def scan_uncertainty_zarr(store_path: Path, indices: set[int], array_name: str) -> dict[int, dict[str, float | bool]]:
    metadata = read_zarr_array_metadata(store_path, array_name)
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    dtype = np.dtype(str(metadata["dtype"]))
    row_chunk_size = chunks[0]
    indices_by_chunk: dict[int, list[int]] = defaultdict(list)
    for row_index in sorted(indices):
        indices_by_chunk[int(row_index // row_chunk_size)].append(int(row_index))
    array_root = store_path / array_name
    result: dict[int, dict[str, float | bool]] = {}
    for row_chunk_index, row_indices in indices_by_chunk.items():
        chunk_path = array_root / zarr_chunk_filename((row_chunk_index, 0))
        row_start = row_chunk_index * row_chunk_size
        row_stop = min(row_start + row_chunk_size, shape[0])
        chunk = np.fromfile(chunk_path, dtype=dtype).reshape((row_stop - row_start, shape[1]))
        for row_index in row_indices:
            row = np.asarray(chunk[row_index - row_start], dtype=np.float32)
            result[row_index] = {
                "finite": bool(np.isfinite(row).all()),
                "positive": bool((row > 0.0).all()),
                "mean": float(row.mean()),
                "max": float(row.max()),
            }
    return result


def scan_numeric(
    qc_rows: list[dict[str, object]],
    config: dict[str, object],
    audit_config: dict[str, object],
) -> tuple[dict[tuple[str, int], dict[str, object]], dict[tuple[str, int], dict[str, float | bool]]]:
    species_names = species_ids(config)
    numeric_by_key: dict[tuple[str, int], dict[str, object]] = {}
    uncertainty_by_key: dict[tuple[str, int], dict[str, float | bool]] = {}
    rows_by_store: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in qc_rows:
        rows_by_store[(str(row["source_collection"]), str(row["trajectory_store_path"]), str(row["store_format"]))].add(int(row["trajectory_index"]))
    for (source_collection, store_path_value, store_format), indices in rows_by_store.items():
        store_path = resolve_repo_path(store_path_value)
        if store_format == "zarr_v2":
            stats_by_index = scan_zarr_trajectories(store_path, indices, "trajectories", species_names, audit_config)
            if (store_path / "teacher_uncertainty").exists():
                uncertainty_stats = scan_uncertainty_zarr(store_path, indices, "teacher_uncertainty")
            else:
                default_uncertainty = float(audit_config["teacher_levels"][source_collection]["default_uncertainty_value"])
                uncertainty_stats = {
                    index: {"finite": True, "positive": default_uncertainty > 0.0, "mean": default_uncertainty, "max": default_uncertainty}
                    for index in indices
                }
        elif store_format == "neuronsifter_chunk_store":
            stats_by_index = scan_chunk_store_trajectories(store_path, indices, "trajectories", species_names, audit_config)
            default_uncertainty = float(audit_config["teacher_levels"][source_collection]["default_uncertainty_value"])
            uncertainty_stats = {
                index: {"finite": True, "positive": default_uncertainty > 0.0, "mean": default_uncertainty, "max": default_uncertainty}
                for index in indices
            }
        else:
            raise ValueError(f"Unsupported store format in simulation audit: {store_format}")
        for trajectory_index, stats in stats_by_index.items():
            numeric_by_key[(source_collection, int(trajectory_index))] = stats
        for trajectory_index, stats in uncertainty_stats.items():
            uncertainty_by_key[(source_collection, int(trajectory_index))] = stats
    return numeric_by_key, uncertainty_by_key


def family_uid(row: dict[str, object]) -> str:
    return stable_uid(
        "trajectory_family",
        str(row["disease_program"]),
        str(row.get("source_simulation_uid", "")),
        str(row.get("source_parameter_uid", "")),
        str(row["regimen_uid"]),
    )


def trajectory_uid(row: dict[str, object]) -> str:
    return stable_uid(
        "trajectory",
        str(row["disease_program"]),
        str(row.get("contract_version", "")),
        str(row["source_collection"]),
        str(row["source_uid"]),
        str(row.get("source_simulation_uid", "")),
        str(row.get("source_parameter_uid", "")),
        str(row["regimen_uid"]),
        str(row["trajectory_index"]),
    )


def source_holdout_split(row: dict[str, object]) -> str:
    teacher_family = str(row.get("teacher_family", ""))
    if teacher_family in {"osp_pksim_pbpk", "neuron_rxd_microenvironment"}:
        return "teacher_generator"
    return "trajectory_pretraining"


def base_qc_rows(config: dict[str, object], audit_config: dict[str, object]) -> list[dict[str, object]]:
    preprocessing = config["ai_ready_preprocessing"]
    source_collections = preprocessing["source_collections"]
    rows: list[dict[str, object]] = []
    mesoscopic_config = source_collections["mesoscopic_solver_teacher"]
    mesoscopic_store = resolve_repo_path(str(mesoscopic_config["required_trajectory_store"]))
    mesoscopic_shape = store_shape(mesoscopic_store, "trajectories", "neuronsifter_chunk_store")
    for trajectory_index, row in enumerate(read_tsv(resolve_repo_path(str(mesoscopic_config["catalog_path"])))):
        source_row = {
            "source_collection": "mesoscopic_solver_teacher",
            "source_uid": row[str(mesoscopic_config["uid_field"])],
            "teacher_level": "mesoscopic",
            "teacher_family": row.get("teacher_family", ""),
            "teacher_role": row.get("teacher_role", ""),
            "disease_program": row.get("disease_program", ""),
            "regimen_uid": row.get("regimen_uid", ""),
            "route": row.get("route", ""),
            "schedule": row.get("schedule", ""),
            "dose_level_id": row.get("dose_level_id", ""),
            "source_simulation_uid": row.get("source_simulation_uid", ""),
            "source_parameter_uid": row.get("source_simulation_uid", ""),
            "contract_version": str(audit_config["teacher_levels"]["full_microenvironment_teacher"]["contract_version"]),
            "trajectory_store_path": repo_relative(mesoscopic_store),
            "trajectory_index": trajectory_index,
            "trajectory_count_in_store": mesoscopic_shape[0],
            "species_count": mesoscopic_shape[4],
            "store_format": "neuronsifter_chunk_store",
        }
        source_row["trajectory_family_uid"] = family_uid(source_row)
        source_row["trajectory_uid"] = trajectory_uid(source_row)
        source_row["paired_uid"] = str(source_row["source_simulation_uid"])
        rows.append(source_row)
    full_config = source_collections["full_microenvironment_teacher"]
    full_shape_cache: dict[str, list[int]] = {}
    full_catalog_path = resolve_repo_path(str(full_config["catalog_path"]))
    with full_catalog_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            trajectory_index_value = row.get(str(full_config["trajectory_index_field"]), "").strip()
            trajectory_store_value = row.get(str(full_config["trajectory_store_field"]), "").strip()
            if not trajectory_index_value or not trajectory_store_value:
                continue
            resolved_store_path = resolve_repo_path(trajectory_store_value)
            resolved_store_key = repo_relative(resolved_store_path)
            if resolved_store_key not in full_shape_cache:
                full_shape_cache[resolved_store_key] = store_shape(resolved_store_path, "trajectories", "zarr_v2")
            source_row = {
                "source_collection": "full_microenvironment_teacher",
                "source_uid": row[str(full_config["uid_field"])],
                "teacher_level": "full_microenvironment",
                "teacher_family": row.get("teacher_family", ""),
                "teacher_role": row.get("teacher_role", ""),
                "disease_program": row.get("disease_program", ""),
                "regimen_uid": row.get("regimen_uid", ""),
                "route": row.get("route", ""),
                "schedule": row.get("schedule", ""),
                "dose_level_id": row.get("dose_level_id", ""),
                "source_simulation_uid": row.get("source_simulation_uid", ""),
                "source_parameter_uid": row.get("source_parameter_uid", ""),
                "contract_version": row.get("contract_version", ""),
                "trajectory_store_path": resolved_store_key,
                "trajectory_index": int(trajectory_index_value),
                "trajectory_count_in_store": full_shape_cache[resolved_store_key][0],
                "species_count": len(str(row.get("field_species_set", "")).split("|")) if row.get("field_species_set", "") else 0,
                "field_species_set": row.get("field_species_set", ""),
                "uncertainty_mean_catalog": row.get("uncertainty_mean", ""),
                "store_format": "zarr_v2",
            }
            source_row["trajectory_family_uid"] = family_uid(source_row)
            source_row["trajectory_uid"] = trajectory_uid(source_row)
            source_row["paired_uid"] = str(source_row["source_simulation_uid"]) or str(source_row["source_uid"])
            rows.append(source_row)
    return rows


def schema_pass(row: dict[str, object], config: dict[str, object], audit_config: dict[str, object]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    expected = audit_config["teacher_levels"][str(row["source_collection"])]
    if int(row["species_count"]) != int(expected["expected_species_count"]):
        failures.append("species_count_mismatch")
    if str(row["source_collection"]) == "full_microenvironment_teacher" and str(row["contract_version"]) != str(expected["contract_version"]):
        failures.append("contract_version_mismatch")
    if bool(audit_config["mechanism_qc"]["require_complete_species_set"]) and str(row["source_collection"]) == "full_microenvironment_teacher":
        observed_species = set(str(row.get("field_species_set", "")).split("|"))
        required_species = set(species_ids(config))
        if not required_species.issubset(observed_species):
            failures.append("species_set_incomplete")
    for field_name in ["source_uid", "disease_program", "teacher_family", "regimen_uid", "trajectory_store_path"]:
        if not str(row.get(field_name, "")).strip():
            failures.append(f"missing_{field_name}")
    return not failures, failures


def mechanism_pass(row: dict[str, object], numeric: dict[str, object], audit_config: dict[str, object]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if bool(audit_config["mechanism_qc"]["require_route_schedule_dose"]):
        for field_name in ["route", "schedule", "dose_level_id"]:
            if not str(row.get(field_name, "")).strip():
                failures.append(f"missing_{field_name}")
    failures.extend(str(flag) for flag in numeric["numeric_failure_flags"])
    return not failures, failures


def confidence_label(uncertainty_mean: float, audit_config: dict[str, object]) -> str:
    uncertainty_qc = audit_config["uncertainty_qc"]
    if uncertainty_mean <= float(uncertainty_qc["high_confidence_uncertainty_max"]):
        return "high"
    if uncertainty_mean <= float(uncertainty_qc["medium_confidence_uncertainty_max"]):
        return "medium"
    if uncertainty_mean <= float(uncertainty_qc["unreliable_uncertainty_max"]):
        return "low"
    return "unreliable"


def grade_row(
    schema_ok: bool,
    numeric_ok: bool,
    mechanism_ok: bool,
    uncertainty_ok: bool,
    leakage_ok: bool,
    confidence: str,
    warning_flags: list[str],
    audit_config: dict[str, object],
) -> tuple[str, str, float]:
    if not (schema_ok and numeric_ok and mechanism_ok and uncertainty_ok and leakage_ok):
        grade = "D"
    elif confidence == "high" and not warning_flags:
        grade = "A"
    elif confidence in {"high", "medium"}:
        grade = "B"
    elif confidence == "low":
        grade = "C"
    else:
        grade = "D"
    return (
        grade,
        str(audit_config["grading"]["qc_use_by_grade"][grade]),
        float(audit_config["grading"]["sample_weight_by_grade"][grade]),
    )


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        parent_index = self.parent[index]
        if parent_index != index:
            self.parent[index] = self.find(parent_index)
        return self.parent[index]

    def union(self, left_index: int, right_index: int) -> None:
        left_root = self.find(left_index)
        right_root = self.find(right_index)
        if left_root != right_root:
            self.parent[right_root] = left_root


def leakage_safe_splits(
    rows: list[dict[str, object]],
    numeric_by_key: dict[tuple[str, int], dict[str, object]],
    split_policy: dict[str, object],
) -> dict[str, str]:
    disjoint_set = DisjointSet(len(rows))
    grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        numeric = numeric_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]
        grouping_values = {
            "family": str(row["trajectory_family_uid"]),
            "content": str(numeric["content_hash"]),
            "regimen": str(row["regimen_uid"]),
            "target_program_proxy": str(row["regimen_uid"]),
        }
        for group_name, group_value in grouping_values.items():
            grouped_indices[(group_name, group_value)].append(row_index)
    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        first_index = indices[0]
        for current_index in indices[1:]:
            disjoint_set.union(first_index, current_index)
    component_members: dict[int, list[int]] = defaultdict(list)
    for row_index in range(len(rows)):
        component_members[disjoint_set.find(row_index)].append(row_index)
    split_by_uid: dict[str, str] = {}
    for member_indices in component_members.values():
        component_key = stable_uid("split_component", *[str(rows[index]["trajectory_family_uid"]) for index in member_indices])
        component_split = hash_split(component_key, split_policy)
        for row_index in member_indices:
            split_by_uid[str(rows[row_index]["trajectory_uid"])] = component_split
    return split_by_uid


def leakage_report(rows: list[dict[str, object]], split_policy: dict[str, object], audit_config: dict[str, object]) -> dict[str, object]:
    del split_policy

    def overlap_count(key_field: str) -> int:
        splits_by_key: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = str(row[key_field])
            splits_by_key[value].add(str(row["split_name"]))
        return sum(1 for split_set in splits_by_key.values() if len(split_set) > 1)

    content_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        content_hash = str(row.get("content_hash", ""))
        if content_hash:
            content_splits[content_hash].add(str(row["split_name"]))
    source_roles = audit_config["split_audit"]["source_holdout_roles"]
    role_sources: dict[str, set[str]] = {
        str(role_name): {str(source) for source in sources}
        for role_name, sources in dict(source_roles).items()
    }
    source_overlap_count = 0
    role_names = sorted(role_sources)
    for left_index, left_role in enumerate(role_names):
        for right_role in role_names[left_index + 1 :]:
            if role_sources[left_role] & role_sources[right_role]:
                source_overlap_count += 1
    return {
        "trajectory_random_uid_overlap": overlap_count("trajectory_uid"),
        "family_holdout_family_overlap": overlap_count("trajectory_family_uid"),
        "regimen_holdout_regimen_overlap": overlap_count("regimen_uid"),
        "target_program_holdout_overlap": overlap_count("regimen_uid"),
        "content_hash_overlap": sum(1 for split_set in content_splits.values() if len(split_set) > 1),
        "source_holdout_role_overlap": source_overlap_count,
        "required_splits": [str(item) for item in audit_config["split_audit"]["required_splits"]],
        "leakage_status": "pass" if source_overlap_count == 0 else "fail",
    }


def pair_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    mesoscopic_by_pair = {
        str(row["paired_uid"]): row
        for row in rows
        if str(row["source_collection"]) == "mesoscopic_solver_teacher" and str(row["paired_uid"])
    }
    full_by_pair = {
        str(row["paired_uid"]): row
        for row in rows
        if str(row["source_collection"]) == "full_microenvironment_teacher" and str(row["paired_uid"])
    }
    paired_uids = sorted(set(mesoscopic_by_pair) | set(full_by_pair))
    alignment_rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    for paired_uid in paired_uids:
        mesoscopic_row = mesoscopic_by_pair.get(paired_uid)
        full_row = full_by_pair.get(paired_uid)
        if mesoscopic_row and full_row:
            paired_status = "paired"
        elif mesoscopic_row:
            paired_status = "mesoscopic_only"
        else:
            paired_status = "full_only"
        status_counts[paired_status] += 1
        pair_action_consistency = bool(
            mesoscopic_row
            and full_row
            and mesoscopic_row["regimen_uid"] == full_row["regimen_uid"]
            and mesoscopic_row["route"] == full_row["route"]
            and mesoscopic_row["schedule"] == full_row["schedule"]
            and mesoscopic_row["dose_level_id"] == full_row["dose_level_id"]
        )
        pair_parameter_consistency = bool(
            mesoscopic_row
            and full_row
            and str(mesoscopic_row.get("source_parameter_uid", "")) == str(full_row.get("source_parameter_uid", ""))
        )
        alignment_rows.append(
            {
                "paired_uid": paired_uid,
                "paired_status": paired_status,
                "mesoscopic_source_uid": "" if mesoscopic_row is None else mesoscopic_row["source_uid"],
                "full_source_uid": "" if full_row is None else full_row["source_uid"],
                "mesoscopic_trajectory_index": "" if mesoscopic_row is None else mesoscopic_row["trajectory_index"],
                "full_trajectory_index": "" if full_row is None else full_row["trajectory_index"],
                "pair_action_consistency": str(pair_action_consistency).lower(),
                "pair_parameter_consistency": str(pair_parameter_consistency).lower(),
                "pair_projection_error": "",
                "pair_terminal_error": "",
                "pair_trend_agreement": "",
            }
        )
    return alignment_rows, {key: int(value) for key, value in status_counts.items()}


def readiness_status(summary: dict[str, object], audit_config: dict[str, object]) -> str:
    gate = audit_config["readiness_gates"]["g2_100k"]
    full_count = int(summary["full_microenvironment_trajectory_count"])
    if full_count < int(gate["full_min_count"]):
        return "G2_fail"
    failures = []
    for metric_name, threshold_key, comparison in [
        ("schema_pass_rate", "schema_pass_rate", "min"),
        ("train_eligible_rate", "min_train_eligible_rate", "min"),
        ("high_or_medium_confidence_rate", "min_high_or_medium_confidence_rate", "min"),
        ("species_coverage_complete_rate", "species_coverage_complete_rate", "min"),
    ]:
        metric_value = float(summary[metric_name])
        threshold = float(gate[threshold_key])
        if comparison == "min" and metric_value < threshold:
            failures.append(metric_name)
    if float(summary["hard_reject_rate"]) > float(gate["max_hard_reject_rate"]):
        failures.append("hard_reject_rate")
    if bool(gate["require_zero_split_leakage"]) and int(summary["total_split_leakage_count"]) != 0:
        failures.append("split_leakage")
    if failures:
        return "G2_fail"
    if int(summary["paired_full_only_count"]) or int(summary["paired_mesoscopic_only_count"]):
        return "G2_pass_with_warnings"
    return "G2_pass"


def build_audit(config: dict[str, object]) -> dict[str, object]:
    audit_config = config["ai_ready_preprocessing"]["simulation_audit"]
    output_root = resolve_repo_path(str(audit_config["local_root"]))
    ensure_directory(output_root)
    split_policy = config["ai_ready_preprocessing"]["split_policy"]
    qc_rows = base_qc_rows(config, audit_config)
    numeric_by_key, uncertainty_by_key = scan_numeric(qc_rows, config, audit_config) if bool(audit_config["numeric_qc"]["scan_enabled"]) else ({}, {})
    split_by_trajectory_uid = leakage_safe_splits(qc_rows, numeric_by_key, split_policy)
    alignment_rows, pair_status_counts = pair_rows(qc_rows)
    full_paired_status = {
        str(row["full_source_uid"]): str(row["paired_status"])
        for row in alignment_rows
        if str(row["full_source_uid"])
    }
    mesoscopic_paired_status = {
        str(row["mesoscopic_source_uid"]): str(row["paired_status"])
        for row in alignment_rows
        if str(row["mesoscopic_source_uid"])
    }
    output_rows: list[dict[str, object]] = []
    for row in qc_rows:
        key = (str(row["source_collection"]), int(row["trajectory_index"]))
        numeric = numeric_by_key[key]
        uncertainty = uncertainty_by_key[key]
        schema_ok, schema_failures = schema_pass(row, config, audit_config)
        numeric_ok = bool(numeric["finite_flag"]) and not numeric["numeric_failure_flags"]
        mechanism_ok, mechanism_failures = mechanism_pass(row, numeric, audit_config)
        uncertainty_ok = bool(uncertainty["finite"]) and bool(uncertainty["positive"]) and float(uncertainty["mean"]) <= float(audit_config["uncertainty_qc"]["unreliable_uncertainty_max"])
        confidence = confidence_label(float(uncertainty["mean"]), audit_config)
        leakage_ok = True
        paired_status = (
            mesoscopic_paired_status[str(row["source_uid"])]
            if str(row["source_collection"]) == "mesoscopic_solver_teacher"
            else full_paired_status[str(row["source_uid"])]
        )
        warning_flags = [str(flag) for flag in numeric["numeric_warning_flags"]]
        failure_flags = schema_failures + mechanism_failures
        grade, qc_use, sample_weight = grade_row(
            schema_ok,
            numeric_ok,
            mechanism_ok,
            uncertainty_ok,
            leakage_ok,
            confidence,
            warning_flags,
            audit_config,
        )
        output_rows.append(
            {
                "trajectory_uid": row["trajectory_uid"],
                "trajectory_family_uid": row["trajectory_family_uid"],
                "paired_uid": row["paired_uid"],
                "source_collection": row["source_collection"],
                "source_uid": row["source_uid"],
                "teacher_level": row["teacher_level"],
                "teacher_family": row["teacher_family"],
                "teacher_role": row["teacher_role"],
                "disease_program": row["disease_program"],
                "regimen_uid": row["regimen_uid"],
                "route": row["route"],
                "schedule": row["schedule"],
                "dose_level_id": row["dose_level_id"],
                "parameter_profile_uid": row["source_parameter_uid"],
                "trajectory_store_path": row["trajectory_store_path"],
                "trajectory_index": row["trajectory_index"],
                "split_name": split_by_trajectory_uid[str(row["trajectory_uid"])],
                "source_holdout_split": source_holdout_split(row),
                "teacher_confidence": confidence,
                "schema_pass_flag": str(schema_ok).lower(),
                "numeric_pass_flag": str(numeric_ok).lower(),
                "mechanism_pass_flag": str(mechanism_ok).lower(),
                "uncertainty_pass_flag": str(uncertainty_ok).lower(),
                "leakage_pass_flag": str(leakage_ok).lower(),
                "paired_status": paired_status,
                "content_hash": numeric["content_hash"],
                "qc_grade": grade,
                "qc_use": qc_use,
                "sample_weight": sample_weight,
                "failure_flags": "|".join(sorted(set(failure_flags))),
                "warning_flags": "|".join(sorted(set(warning_flags))),
            }
        )
    leakage = leakage_report(output_rows, split_policy, audit_config)
    total_split_leakage_count = sum(
        int(leakage[key])
        for key in [
            "trajectory_random_uid_overlap",
            "family_holdout_family_overlap",
            "regimen_holdout_regimen_overlap",
            "target_program_holdout_overlap",
            "content_hash_overlap",
            "source_holdout_role_overlap",
        ]
    )
    if total_split_leakage_count:
        for row in output_rows:
            row["leakage_pass_flag"] = "false"
            if row["qc_grade"] in {"A", "B", "C"}:
                row["qc_grade"] = "D"
                row["qc_use"] = audit_config["grading"]["qc_use_by_grade"]["D"]
                row["sample_weight"] = audit_config["grading"]["sample_weight_by_grade"]["D"]
                row["failure_flags"] = "|".join(sorted(set(str(row["failure_flags"]).split("|") + ["split_leakage"])))
    fieldnames = config["schemas"]["simulation_qc_catalog_fields"]
    relative_paths = audit_config["relative_paths"]
    write_tsv(output_root / str(relative_paths["qc_catalog"]), fieldnames, output_rows)
    write_tsv(output_root / str(relative_paths["pair_alignment_audit"]), config["schemas"]["simulation_pair_alignment_fields"], alignment_rows)
    numeric_rows = [
        {
            "source_collection": row["source_collection"],
            "trajectory_index": row["trajectory_index"],
            "finite_flag": numeric_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["finite_flag"],
            "min_value": numeric_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["min_value"],
            "max_value": numeric_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["max_value"],
            "standard_deviation": numeric_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["standard_deviation"],
            "max_abs_jump": numeric_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["max_abs_jump"],
            "terminal_abs_max": numeric_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["terminal_abs_max"],
            "content_hash": numeric_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["content_hash"],
        }
        for row in output_rows
    ]
    write_tsv(output_root / str(relative_paths["numeric_audit"]), config["schemas"]["simulation_numeric_audit_fields"], numeric_rows)
    uncertainty_group_rows: list[dict[str, object]] = []
    for (source_collection, teacher_family), group_rows in group_rows_by(output_rows, ["source_collection", "teacher_family"]).items():
        uncertainty_values = [
            float(uncertainty_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["mean"])
            for row in group_rows
        ]
        confidence_counts = Counter(str(row["teacher_confidence"]) for row in group_rows)
        finite_count = sum(
            1
            for row in group_rows
            if bool(uncertainty_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["finite"])
        )
        positive_count = sum(
            1
            for row in group_rows
            if bool(uncertainty_by_key[(str(row["source_collection"]), int(row["trajectory_index"]))]["positive"])
        )
        uncertainty_group_rows.append(
            {
                "source_collection": source_collection,
                "teacher_family": teacher_family,
                "finite_rate": finite_count / max(len(group_rows), 1),
                "positive_rate": positive_count / max(len(group_rows), 1),
                "mean_uncertainty": float(np.mean(uncertainty_values)),
                "max_uncertainty": float(np.max(uncertainty_values)),
                "high_confidence_count": confidence_counts["high"],
                "medium_confidence_count": confidence_counts["medium"],
                "low_confidence_count": confidence_counts["low"],
            }
        )
    write_tsv(output_root / str(relative_paths["teacher_uncertainty_audit"]), config["schemas"]["simulation_teacher_uncertainty_audit_fields"], uncertainty_group_rows)
    grade_counts = Counter(str(row["qc_grade"]) for row in output_rows)
    use_counts = Counter(str(row["qc_use"]) for row in output_rows)
    source_counts = Counter(str(row["source_collection"]) for row in output_rows)
    full_rows = [row for row in output_rows if str(row["source_collection"]) == "full_microenvironment_teacher"]
    summary = {
        "readiness_gate": "g2_100k",
        "full_microenvironment_trajectory_count": len(full_rows),
        "mesoscopic_trajectory_count": source_counts["mesoscopic_solver_teacher"],
        "unique_trajectory_uid_count": len({str(row["trajectory_uid"]) for row in output_rows}),
        "unique_trajectory_family_count": len({str(row["trajectory_family_uid"]) for row in output_rows}),
        "unique_regimen_count": len({str(row["regimen_uid"]) for row in output_rows}),
        "unique_target_program_count": len({str(row["regimen_uid"]) for row in output_rows}),
        "high_confidence_trajectory_count": sum(1 for row in output_rows if row["teacher_confidence"] == "high"),
        "medium_confidence_trajectory_count": sum(1 for row in output_rows if row["teacher_confidence"] == "medium"),
        "train_eligible_count": use_counts["main_pretraining"],
        "weak_teacher_count": use_counts["weak_teacher_calibration"],
        "stress_only_count": use_counts["stress_test_only"],
        "reject_count": use_counts["reject"],
        "schema_pass_rate": sum(1 for row in output_rows if row["schema_pass_flag"] == "true") / max(len(output_rows), 1),
        "train_eligible_rate": use_counts["main_pretraining"] / max(len(output_rows), 1),
        "hard_reject_rate": use_counts["reject"] / max(len(output_rows), 1),
        "high_or_medium_confidence_rate": sum(1 for row in output_rows if row["teacher_confidence"] in {"high", "medium"}) / max(len(output_rows), 1),
        "species_coverage_complete_rate": sum(1 for row in output_rows if row["schema_pass_flag"] == "true") / max(len(output_rows), 1),
        "paired_count": pair_status_counts.get("paired", 0),
        "paired_full_only_count": pair_status_counts.get("full_only", 0),
        "paired_mesoscopic_only_count": pair_status_counts.get("mesoscopic_only", 0),
        "total_split_leakage_count": total_split_leakage_count,
        "qc_grade_counts": dict(grade_counts),
        "qc_use_counts": dict(use_counts),
        "source_collection_counts": dict(source_counts),
    }
    summary["readiness_status"] = readiness_status(summary, audit_config)
    write_json(output_root / str(relative_paths["inventory_audit"]), {
        "catalog_row_count": len(qc_rows),
        "qc_catalog_row_count": len(output_rows),
        "source_collection_counts": dict(source_counts),
        "pair_status_counts": pair_status_counts,
    })
    write_json(output_root / str(relative_paths["coverage_audit"]), {
        "coverage_by_route": dict(Counter(str(row["route"]) for row in output_rows)),
        "coverage_by_schedule": dict(Counter(str(row["schedule"]) for row in output_rows)),
        "coverage_by_dose": dict(Counter(str(row["dose_level_id"]) for row in output_rows)),
        "coverage_by_regimen": dict(Counter(str(row["regimen_uid"]) for row in output_rows)),
        "coverage_by_teacher_confidence": dict(Counter(str(row["teacher_confidence"]) for row in output_rows)),
        "coverage_by_teacher_family": dict(Counter(str(row["teacher_family"]) for row in output_rows)),
    })
    write_json(output_root / str(relative_paths["split_leakage_audit"]), leakage)
    write_json(output_root / str(relative_paths["qc_summary"]), summary)
    return {
        "qc_catalog": repo_relative(output_root / str(relative_paths["qc_catalog"])),
        "qc_summary": repo_relative(output_root / str(relative_paths["qc_summary"])),
        **summary,
    }


def group_rows_by(rows: list[dict[str, object]], fields: list[str]) -> dict[tuple[str, ...], list[dict[str, object]]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in fields)].append(row)
    return grouped


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    payload = build_audit(config)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
