from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import re
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.physical_occupancy import compile_physical_record, digest, require, whole_particle_basis
from Libs.utils.data.native_diffusion_data import JOINT_SOURCE_SCHEMA, parse_joint_source, native_action_policy
from Libs.utils.model.dynamics_fields import compartment_id_list, region_id_list, species_id_list, time_grid_hours
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.microenvironment_teacher import json_payload, teacher_contract_version, write_microenvironment_zarr
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_tsv
from Libs.utils.model.screening_core import stable_uid


REQUIRED_DESIGN_FIELDS = [
    "design_id",
    "source_model_id",
    "model_path",
    "compound_uid",
    "canonical_compound_id",
    "subject_covariate_profile",
    "calibration_role",
    "regimen_variants",
]
REQUIRED_VARIANT_FIELDS = ["regimen_variant_id", "route", "schedule", "dose_level_id", "dose_multiplier"]


def slugify_source_model_id(model_path: Path, library_root: Path) -> str:
    label = model_path.resolve().relative_to(library_root.resolve()).with_suffix("").as_posix()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    if not slug:
        raise ValueError(f"Unable to derive OSP source model id from {model_path}")
    return slug


def discovered_osp_designs(runtime: dict[str, object]) -> list[dict[str, object]]:
    discovery = dict(runtime["model_library"].get("library_discovery", {}))
    if not bool(discovery.get("enabled", False)):
        return []
    required_discovery_fields = [
        "design_id_prefix",
        "subject_covariate_profile",
        "calibration_role",
        "regimen_variant_id",
        "route",
        "schedule",
        "dose_level_id",
        "dose_multiplier",
    ]
    missing_fields = [field for field in required_discovery_fields if field not in discovery]
    if missing_fields:
        raise ValueError(f"OSP model library discovery is missing required fields: {missing_fields}")
    library_root = resolve_repo_path(str(runtime["model_library"]["local_repo_path"]))
    model_files = sorted(library_root.glob(str(runtime["model_library"]["model_file_glob"])))
    if not model_files:
        raise FileNotFoundError(f"OSP model library discovery found no models under {library_root}")
    designs: list[dict[str, object]] = []
    for model_path in model_files:
        source_id = slugify_source_model_id(model_path, library_root)
        relative_model_path = model_path.resolve().relative_to(REPO_ROOT)
        designs.append(
            {
                "design_id": f"{discovery['design_id_prefix']}_{source_id}",
                "source_model_id": f"osp_{source_id}",
                "model_path": str(relative_model_path),
                "compound_uid": f"osp_{source_id}",
                "canonical_compound_id": source_id,
                "subject_covariate_profile": str(discovery["subject_covariate_profile"]),
                "calibration_role": str(discovery["calibration_role"]),
                "regimen_variants": [
                    {
                        "regimen_variant_id": str(discovery["regimen_variant_id"]),
                        "route": str(discovery["route"]),
                        "schedule": str(discovery["schedule"]),
                        "dose_level_id": str(discovery["dose_level_id"]),
                        "dose_multiplier": float(discovery["dose_multiplier"]),
                    }
                ],
            }
        )
    return designs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository-owned OSP/PK-Sim PBPK rollouts and map them to AD full microenvironment trajectories.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--design-id", default="")
    parser.add_argument("--native-physical-input", help="Explicit source export/settings design JSON; no discovery or model execution")
    parser.add_argument("--native-output", help="New native joint observation source JSON")
    parser.add_argument("--max-designs", type=int, default=0)
    parser.add_argument("--max-regimens", type=int, default=0)
    return parser.parse_args()


def require_fields(payload: dict[str, object], required_fields: list[str], label: str) -> None:
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise ValueError(f"{label} is missing required fields: {missing_fields}")


def validate_model_path(runtime: dict[str, object], model_path: Path) -> None:
    model_library = runtime["model_library"]
    if not model_path.exists():
        raise FileNotFoundError(f"Configured OSP model does not exist: {model_path}")
    supported_extensions = {str(extension).lower() for extension in model_library["supported_model_extensions"]}
    if model_path.suffix.lower() not in supported_extensions:
        raise ValueError(f"Unsupported OSP model extension for {model_path}; expected one of {sorted(supported_extensions)}")


def rollout_designs(runtime: dict[str, object], args: argparse.Namespace) -> list[dict[str, object]]:
    configured_designs = list(runtime["rollout_designs"])
    configured_model_paths = {str(resolve_repo_path(str(design["model_path"])).resolve()) for design in configured_designs}
    discovered_designs = [
        design
        for design in discovered_osp_designs(runtime)
        if str(resolve_repo_path(str(design["model_path"])).resolve()) not in configured_model_paths
    ]
    designs = configured_designs + discovered_designs
    if args.design_id:
        designs = [design for design in designs if str(design["design_id"]) == args.design_id]
    if args.max_designs > 0:
        designs = designs[: args.max_designs]
    if not designs:
        raise ValueError("No OSP rollout designs selected from configured or discovered OSP PBPK library models")
    for design in designs:
        require_fields(design, REQUIRED_DESIGN_FIELDS, f"OSP rollout design {design}")
        validate_model_path(runtime, resolve_repo_path(str(design["model_path"])))
        variants = design["regimen_variants"]
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"OSP rollout design {design['design_id']} requires non-empty regimen_variants")
        for variant in variants:
            require_fields(variant, REQUIRED_VARIANT_FIELDS, f"OSP regimen variant {variant}")
    return designs


def rollout_rows(runtime: dict[str, object], args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for design in rollout_designs(runtime, args):
        for variant in design["regimen_variants"]:
            regimen_uid = stable_uid("osp_pbpk_regimen", design["design_id"], variant["regimen_variant_id"])
            rows.append({"design": design, "variant": variant, "regimen_uid": regimen_uid})
    if args.max_regimens > 0:
        rows = rows[: args.max_regimens]
    if not rows:
        raise ValueError("No OSP rollout regimen rows selected")
    return rows


def validate_cached_osp_rollout_output(runtime: dict[str, object], output_path: Path) -> None:
    cache_config = runtime["raw_output_cache"]
    minimum_size_bytes = int(cache_config["minimum_size_bytes"])
    if output_path.stat().st_size < minimum_size_bytes:
        raise ValueError(f"Cached OSP rollout output is smaller than configured minimum_size_bytes: {output_path}")
    header_frame = pd.read_csv(output_path, sep="\t", nrows=1)
    missing_columns = [
        str(column_name)
        for column_name in cache_config["required_columns"]
        if str(column_name) not in header_frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Cached OSP rollout output is missing required columns {missing_columns}: {output_path}")
    policy_uid = str(header_frame["standardization_policy_uid"].iloc[0])
    if policy_uid != str(cache_config["standardization_policy_uid"]):
        raise ValueError(
            "Cached OSP rollout output was generated with a different standardization policy: "
            f"{policy_uid} != {cache_config['standardization_policy_uid']} ({output_path})"
        )


def run_osp_rollout(config: dict, design: dict[str, object], variant: dict[str, object], output_path: Path) -> None:
    runtime = config["dynamics_ready_registry"]["osp_pksim_runtime"]
    cache_config = runtime["raw_output_cache"]
    if bool(cache_config["reuse_existing_outputs"]) and output_path.exists():
        try:
            validate_cached_osp_rollout_output(runtime, output_path)
            return
        except ValueError as exc:
            if not bool(cache_config["regenerate_invalid_cache"]):
                raise
            sys.stderr.write(f"Regenerating invalid OSP cache {output_path}: {exc}\n")
            output_path.unlink()
    environment = runtime["conda_environment"]
    mapping = runtime["output_path_mapping"]
    command = [
        "conda",
        "run",
        "-n",
        str(environment["environment_name"]),
        str(environment["rscript_executable"]),
        str(REPO_ROOT / "Libs/scripts/teacher_generation/run_osp_pksim_rollout.R"),
        "--config",
        str(REPO_ROOT / runtime["environment_manifest_path"]),
        "--model",
        str(resolve_repo_path(str(design["model_path"]))),
        "--output",
        str(output_path),
        "--plasma-regex",
        str(mapping["plasma_regex"]),
        "--csf-regex",
        str(mapping["csf_regex"]),
        "--brain-isf-regex",
        str(mapping["brain_isf_regex"]),
        "--result-file-regex",
        str(mapping["result_file_regex"]),
        "--csv-metadata-columns",
        "|".join(str(column_name) for column_name in mapping["csv_metadata_columns"]),
        "--measure-column-keep-regex",
        str(mapping["measure_column_keep_regex"]),
        "--standardization-policy-uid",
        str(cache_config["standardization_policy_uid"]),
        "--max-measure-columns-per-chunk",
        str(mapping["max_measure_columns_per_chunk"]),
    ]
    if "dose_parameter_path" in variant:
        if "dose_value" not in variant:
            raise ValueError(f"OSP regimen variant {variant['regimen_variant_id']} declares dose_parameter_path without dose_value")
        command.extend(["--dose-parameter", str(variant["dose_parameter_path"]), "--dose", str(variant["dose_value"])])
        if "dose_unit" in variant:
            command.extend(["--dose-unit", str(variant["dose_unit"])])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def finite_curve(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        raise ValueError(f"OSP rollout output is missing column {column_name}")
    time_values = pd.to_numeric(frame["time_hours"], errors="coerce")
    curve_values = pd.to_numeric(frame[column_name], errors="coerce")
    finite_mask = np.isfinite(time_values.to_numpy(dtype=np.float64)) & np.isfinite(curve_values.to_numpy(dtype=np.float64))
    return frame.loc[finite_mask, ["time_hours", column_name]].copy()


def interpolate_to_grid(frame: pd.DataFrame, column_name: str, grid: np.ndarray, required: bool) -> np.ndarray:
    curve_frame = finite_curve(frame, column_name)
    if curve_frame.empty:
        if required:
            raise ValueError(f"Required OSP concentration curve is absent: {column_name}")
        return np.zeros((grid.shape[0],), dtype=np.float32)
    grouped = curve_frame.groupby("time_hours", as_index=False)[column_name].mean().sort_values("time_hours")
    source_time = grouped["time_hours"].to_numpy(dtype=np.float32)
    source_value = grouped[column_name].to_numpy(dtype=np.float32)
    return np.interp(grid.astype(np.float32), source_time, source_value).astype(np.float32)


def normalized(values: np.ndarray) -> np.ndarray:
    denominator = float(np.max(np.abs(values)))
    if denominator <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return (values / denominator).astype(np.float32)


def auc(time_hours: np.ndarray, values: np.ndarray) -> float:
    return float(np.trapezoid(values.astype(np.float64), time_hours.astype(np.float64)))


def cmax(values: np.ndarray) -> float:
    return float(np.max(values.astype(np.float64)))


def tmax(time_hours: np.ndarray, values: np.ndarray) -> float:
    return float(time_hours[int(np.argmax(values))])


def curve_presence_payload(frame: pd.DataFrame) -> dict[str, bool]:
    return {
        "plasma_concentration": not finite_curve(frame, "plasma_concentration").empty,
        "csf_concentration": not finite_curve(frame, "csf_concentration").empty,
        "brain_isf_concentration": not finite_curve(frame, "brain_isf_concentration").empty,
    }


def frame_to_trajectory(config: dict, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float], dict[str, bool]]:
    runtime = config["dynamics_ready_registry"]["osp_pksim_runtime"]
    mapping = runtime["output_path_mapping"]
    required_columns = {str(column_name) for column_name in mapping["required_curve_columns"]}
    time_values = time_grid_hours(config)
    region_ids = region_id_list(config)
    compartment_ids = compartment_id_list(config)
    species_ids = species_id_list(config)
    species_lookup = {species_id: index for index, species_id in enumerate(species_ids)}
    curve_presence = curve_presence_payload(frame)
    for column_name in required_columns:
        if not curve_presence[column_name]:
            raise ValueError(f"OSP rollout lacks required curve {column_name}; update output_path_mapping or choose an OSP model with that output")
    plasma_curve = normalized(interpolate_to_grid(frame, "plasma_concentration", time_values, "plasma_concentration" in required_columns))
    csf_curve = normalized(interpolate_to_grid(frame, "csf_concentration", time_values, "csf_concentration" in required_columns))
    isf_curve = normalized(interpolate_to_grid(frame, "brain_isf_concentration", time_values, "brain_isf_concentration" in required_columns))
    trajectory = np.zeros((time_values.shape[0], len(region_ids), len(compartment_ids), len(species_ids)), dtype=np.float32)
    for compartment_index, compartment_id in enumerate(compartment_ids):
        if compartment_id == "plasma":
            trajectory[:, :, compartment_index, species_lookup["free_drug"]] = plasma_curve[:, None]
        elif compartment_id == "csf":
            trajectory[:, :, compartment_index, species_lookup["free_drug"]] = csf_curve[:, None]
        else:
            trajectory[:, :, compartment_index, species_lookup["free_drug"]] = isf_curve[:, None]
    trajectory[..., species_lookup["target_bound_drug"]] = np.clip(trajectory[..., species_lookup["free_drug"]] * float(config["dynamics_ready_registry"]["field_dynamics_priors"]["target_binding_gain"]), 0.0, 1.0)
    if "vascular_perivascular_transport" in species_lookup:
        trajectory[..., species_lookup["vascular_perivascular_transport"]] = np.clip(trajectory[..., species_lookup["free_drug"]] * float(config["dynamics_ready_registry"]["field_dynamics_priors"]["clearance_flux_scale"]["vascular_perivascular_transport"]), 0.0, 1.5)
    plasma_auc = auc(time_values, plasma_curve)
    metrics = {
        "plasma_auc": plasma_auc,
        "plasma_cmax": cmax(plasma_curve),
        "plasma_tmax": tmax(time_values, plasma_curve),
        "csf_auc": auc(time_values, csf_curve),
        "isf_auc": auc(time_values, isf_curve),
        "brain_plasma_auc_ratio": auc(time_values, isf_curve) / max(plasma_auc, 1.0e-12),
    }
    return trajectory, metrics, curve_presence



def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_native_physical_source(config: dict, packet: dict, base_directory: Path) -> dict:
    """Standardize explicit completed OSP exports and assemble genuine joint programs."""
    require(bool(packet.get("design_id")), "an explicit physical design/input is required")
    runtime = config["dynamics_ready_registry"]["osp_pksim_runtime"]
    require(runtime["native_physical_export"]["mode"] == "osp_source_bound_equilibrium.v1", "native export mode mismatch")
    support = packet["support"]
    regions, compartments = packet["region_ids"], packet["compartment_ids"]
    require(regions == region_id_list(config) and compartments == compartment_id_list(config),
            "declared region/compartment order does not match configured grid")
    actions = packet["actions"]
    regimens = [entry["regimen_uid"] for entry in actions]
    require(bool(regimens) and len(set(regimens)) == len(regimens), "unique selected actions required")
    exports = {}
    for export in packet["exports"]:
        key = tuple(export[name] for name in ("sample_uid", "particle_uid", "regimen_uid"))
        require(key not in exports, "duplicate particle/action export")
        exports[key] = export
    environment = runtime["conda_environment"]
    records, occupancy, masks, physical_time = [], [], [], []
    used = set()
    with tempfile.TemporaryDirectory(prefix="nrnsift_physical_") as scratch:
        for bi, sample in enumerate(support["sample_uids"]):
            sample_records, sample_occupancy, sample_masks, sample_times = [], [], [], []
            for ji, particle in enumerate(support["particle_uids"][bi]):
                particle_records, particle_occupancy, particle_masks, particle_times = [], [], [], []
                for action_entry in actions:
                    regimen = action_entry["regimen_uid"]
                    key = (sample, particle, regimen)
                    require(key in exports, "missing whole-particle/action export")
                    used.add(key)
                    export = exports[key]
                    csv_path = (base_directory / export["csv_path"]).resolve()
                    snapshot_path = (base_directory / export["run_snapshot_path"]).resolve()
                    snapshot = json.loads(snapshot_path.read_text())
                    model_path = (snapshot_path.parent / snapshot["model_path"]).resolve()
                    require(_sha256(csv_path) == snapshot["export_sha256"], "source export bytes changed")
                    require(_sha256(model_path) == snapshot["model_sha256"], "source model bytes changed")
                    binding = copy.deepcopy(export["binding"])
                    for name, value in {"source_uid": support["source_uid"], "source_snapshot_ref": support["source_snapshot_ref"],
                        "source_posterior_uid": packet["source_posterior_uid"], "support_uid": support["support_uid"],
                        "sample_uid": sample, "particle_uid": particle, "regimen_uid": regimen}.items():
                        require(binding.get(name) == value, "selected " + name + " binding mismatch")
                    require(binding.get("particle_basis") == whole_particle_basis(support, bi, ji),
                            "source whole-particle tuple mismatch")
                    output = Path(scratch) / "physical.tsv"
                    command = ["conda", "run", "--no-capture-output", "-n", str(environment["environment_name"]),
                        str(environment["rscript_executable"]), str(REPO_ROOT / "Libs/scripts/teacher_generation/run_osp_pksim_rollout.R"),
                        "--standardize-csv-input", str(csv_path), "--physical-binding", str(snapshot_path), "--output", str(output)]
                    subprocess.run(command, cwd=REPO_ROOT, check=True)
                    rows = pd.read_csv(output, sep="\t", keep_default_na=False).to_dict("records")
                    inputs = {"binding": binding, "run_snapshot": snapshot, "action": action_entry["action"],
                        "source_rows": rows, "region_ids": regions, "compartment_ids": compartments,
                        "time_grid_hours": packet["time_grid_hours"], "organism_context": packet["organism_context"],
                        "graph_ref": packet["graph_ref"], "provenance": {"model_path": str(model_path),
                            "export_path": str(csv_path), "run_snapshot_path": str(snapshot_path),
                            "model_sha256": snapshot["model_sha256"], "export_sha256": snapshot["export_sha256"],
                            "run_snapshot_sha256": digest(snapshot)}}
                    record = compile_physical_record(inputs)
                    particle_records.append(record)
                    physical = record["physical_program"]
                    particle_occupancy.append(copy.deepcopy(physical["occupancy"]))
                    particle_masks.append(copy.deepcopy(physical["support"]))
                    particle_times.append(copy.deepcopy(physical["physical_time"]))
                sample_records.append(particle_records)
                sample_occupancy.append(particle_occupancy)
                sample_masks.append(particle_masks)
                sample_times.append(particle_times)
            records.append(sample_records)
            occupancy.append(sample_occupancy)
            masks.append(sample_masks)
            physical_time.append(sample_times)
    require(used == set(exports), "unselected export records are not admitted")
    programs = {"source_posterior_uid": packet["source_posterior_uid"], "support_uid": support["support_uid"],
        "source_uid": support["source_uid"], "source_snapshot_ref": support["source_snapshot_ref"],
        "sample_uids": support["sample_uids"], "particle_uids": support["particle_uids"], "regimen_uids": regimens,
        "records": records, "occupancy": occupancy, "support": masks, "physical_time": physical_time}
    source = {"schema_version": JOINT_SOURCE_SCHEMA, "source_posterior_uid": packet["source_posterior_uid"],
        "support": copy.deepcopy(support), "observations": copy.deepcopy(packet["observations"]), "particle_programs": programs}
    parse_joint_source(source, action_policy=native_action_policy(config))
    return source


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.native_physical_input:
        require(bool(args.native_output), "--native-output is required")
        output = Path(args.native_output)
        require(not output.exists(), "native output already exists; choose a new output path")
        selected = Path(args.native_physical_input).resolve()
        source = build_native_physical_source(config, json.loads(selected.read_text()), selected.parent)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
        return
    require(not args.native_output, "--native-output requires --native-physical-input")
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    root = resolve_repo_path(registry["local_root"])
    relative_paths = registry["relative_paths"]
    runtime = config["dynamics_ready_registry"]["osp_pksim_runtime"]
    output_root = resolve_repo_path(runtime["rollout_output_root"])
    ensure_directory(output_root)
    selected_rows = rollout_rows(runtime, args)
    trajectories = []
    uncertainty_rows = []
    catalog_rows = []
    for trajectory_index, selected_row in enumerate(selected_rows):
        design = selected_row["design"]
        variant = selected_row["variant"]
        regimen_uid = str(selected_row["regimen_uid"])
        raw_output_path = output_root / f"{regimen_uid}.tsv"
        run_osp_rollout(config, design, variant, raw_output_path)
        raw_frame = pd.read_csv(raw_output_path, sep="\t")
        trajectory, metrics, curve_presence = frame_to_trajectory(config, raw_frame)
        trajectories.append(trajectory)
        uncertainty = np.full((len(species_id_list(config)),), float(config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]["uncertainty_floor"]), dtype=np.float32)
        uncertainty_rows.append(uncertainty)
        uncertainty_payload = {species_id: float(value) for species_id, value in zip(species_id_list(config), uncertainty.tolist())}
        uncertainty_payload["curve_presence"] = curve_presence
        catalog_rows.append(
            {
                "pbpk_rollout_uid": stable_uid("osp_pbpk_rollout", regimen_uid),
                "contract_version": teacher_contract_version(config),
                "compound_uid": design["compound_uid"],
                "canonical_compound_id": design["canonical_compound_id"],
                "regimen_uid": regimen_uid,
                "route": variant["route"],
                "schedule": variant["schedule"],
                "dose_multiplier": variant["dose_multiplier"],
                "source_model_id": design["source_model_id"],
                "source_repository": runtime["source_repository"],
                "provenance_level": runtime["provenance_level"],
                "observed_or_synthetic": runtime["observed_or_synthetic"],
                "teacher_role": runtime["teacher_role"],
                "calibration_role": design["calibration_role"],
                "trajectory_store_path": str((root / relative_paths["pbpk_trajectory_store"]).resolve().relative_to(REPO_ROOT)),
                "trajectory_index": trajectory_index,
                "plasma_auc": metrics["plasma_auc"],
                "plasma_cmax": metrics["plasma_cmax"],
                "plasma_tmax": metrics["plasma_tmax"],
                "csf_auc": metrics["csf_auc"],
                "isf_auc": metrics["isf_auc"],
                "brain_plasma_auc_ratio": metrics["brain_plasma_auc_ratio"],
                "uncertainty_payload": json_payload(uncertainty_payload),
                "note": f"Repository-generated OSPSuite PBPK rollout from {design['model_path']} mapped to AD full microenvironment tensor; curve_presence={json_payload(curve_presence)}",
            }
        )
    trajectory_tensor = np.stack(trajectories, axis=0).astype(np.float32)
    uncertainty_tensor = np.stack(uncertainty_rows, axis=0).astype(np.float32)
    store_path = root / relative_paths["pbpk_trajectory_store"]
    write_microenvironment_zarr(
        store_path,
        trajectory_tensor,
        uncertainty_tensor,
        config,
        {"teacher_family": "osp_pksim_pbpk", "source_design_count": len(selected_rows)},
    )
    output_path = root / relative_paths["pbpk_rollout_catalog"]
    write_tsv(output_path, config["schemas"]["pbpk_rollout_fields"], catalog_rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        pbpk_rollout_row_count=len(catalog_rows),
        trajectory_store_path=store_path,
    )


if __name__ == "__main__":
    main()
