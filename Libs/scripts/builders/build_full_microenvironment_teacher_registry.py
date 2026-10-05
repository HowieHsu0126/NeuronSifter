from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.model.dynamics_store import load_chunked_rows, load_zarr_rows, read_store_manifest
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.model.microenvironment_teacher import json_payload, teacher_contract_version, teacher_precision_from_uncertainty, validate_teacher_role, write_microenvironment_zarr_from_loaders
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, temporary_output_path, write_tsv
from Libs.utils.model.screening_core import stable_uid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the AD full microenvironment teacher registry from OSP, ChEMBL, and NEURON/RxD artifacts.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def required_rows(path: Path, catalog_name: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required {catalog_name} is missing: {path}")
    rows = read_tsv(path)
    if not rows:
        raise ValueError(f"Required {catalog_name} is empty: {path}")
    return rows


def require_streaming_catalog(path: Path, catalog_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {catalog_name} is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if not header:
            raise ValueError(f"Required {catalog_name} is empty: {path}")
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"Required {catalog_name} has no data rows: {path}")


def iter_tsv(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def pbpk_teacher_row(config: dict, row: dict[str, str]) -> dict[str, object]:
    role = str(row["teacher_role"])
    validate_teacher_role(role)
    uncertainty_payload = {"pbpk_rollout_uncertainty": row["uncertainty_payload"]}
    uncertainty_mean = float(config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]["uncertainty_floor"])
    return {
        "teacher_uid": stable_uid("full_microenvironment_teacher", row["pbpk_rollout_uid"]),
        "contract_version": teacher_contract_version(config),
        "disease_program": "ad_control",
        "disease_family": "AD",
        "teacher_family": "osp_pksim_pbpk",
        "teacher_role": role,
        "calibration_role": row["calibration_role"],
        "observed_or_synthetic": row["observed_or_synthetic"],
        "source_model_id": row["source_model_id"],
        "source_repository": row["source_repository"],
        "provenance_level": row["provenance_level"],
        "teacher_evidence_level": "repository_generated",
        "calibration_status": "calibration_pending",
        "main_pretraining_eligibility": "calibration_or_compatibility_only",
        "source_simulation_uid": row["pbpk_rollout_uid"],
        "source_protocol_uid": "",
        "source_parameter_uid": "",
        "regimen_uid": row["regimen_uid"],
        "compound_uid": row["compound_uid"],
        "route": row["route"],
        "schedule": row["schedule"],
        "dose_level_id": "",
        "field_species_set": "free_drug|target_bound_drug|vascular_perivascular_transport",
        "trajectory_index": row["trajectory_index"],
        "trajectory_store_path": row["trajectory_store_path"],
        "uncertainty_mean": uncertainty_mean,
        "uncertainty_payload": json_payload(uncertainty_payload),
        "teacher_loss_weight": teacher_precision_from_uncertainty(config, uncertainty_mean),
        "note": "Primary PBPK mechanism teacher row from repository-run OSP/PK-Sim",
    }


def pbpk_teacher_rows(config: dict, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [pbpk_teacher_row(config, row) for row in rows]


def chembl_teacher_row(config: dict, row: dict[str, str]) -> dict[str, object]:
    role = str(row["teacher_role"])
    validate_teacher_role(role)
    uncertainty_mean = float(row["uncertainty"])
    return {
        "teacher_uid": stable_uid("full_microenvironment_teacher", row["parameter_prediction_uid"]),
        "contract_version": teacher_contract_version(config),
        "disease_program": "ad_control",
        "disease_family": "AD",
        "teacher_family": "chembl_pbpk_parameterization",
        "teacher_role": role,
        "calibration_role": row["calibration_role"],
        "observed_or_synthetic": row["observed_or_synthetic"],
        "source_model_id": row["parameter_name"],
        "source_repository": row["source_repository"],
        "provenance_level": row["provenance_level"],
        "teacher_evidence_level": "configuration_generated_provisional",
        "calibration_status": "calibration_pending",
        "main_pretraining_eligibility": "calibration_or_compatibility_only",
        "source_simulation_uid": "",
        "source_protocol_uid": "",
        "source_parameter_uid": row["parameter_prediction_uid"],
        "regimen_uid": "",
        "compound_uid": row["compound_uid"],
        "route": "",
        "schedule": "",
        "dose_level_id": "",
        "field_species_set": "free_drug|vascular_perivascular_transport",
        "trajectory_index": "",
        "trajectory_store_path": "",
        "uncertainty_mean": uncertainty_mean,
        "uncertainty_payload": json_payload({"parameter_uncertainty": uncertainty_mean}),
        "teacher_loss_weight": teacher_precision_from_uncertainty(config, uncertainty_mean),
        "note": "PBPK parameter predicted teacher row with explicit uncertainty",
    }


def neuron_teacher_row(config: dict, row: dict[str, str]) -> dict[str, object]:
    role = str(row["teacher_role"])
    validate_teacher_role(role)
    uncertainty_mean = float(config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]["uncertainty_floor"])
    return {
        "teacher_uid": stable_uid("full_microenvironment_teacher", row["neuron_rxd_protocol_uid"]),
        "contract_version": teacher_contract_version(config),
        "disease_program": row["disease_program"],
        "disease_family": "AD",
        "teacher_family": "neuron_rxd_microenvironment",
        "teacher_role": role,
        "calibration_role": "mechanism_teacher_loss",
        "observed_or_synthetic": row["observed_or_synthetic"],
        "source_model_id": row["source_model_id"],
        "source_repository": row["source_repository"],
        "provenance_level": row["provenance_level"],
        "teacher_evidence_level": "repository_generated",
        "calibration_status": "calibration_pending",
        "main_pretraining_eligibility": "calibration_or_compatibility_only",
        "source_simulation_uid": "",
        "source_protocol_uid": row["neuron_rxd_protocol_uid"],
        "source_parameter_uid": "",
        "regimen_uid": "",
        "compound_uid": "",
        "route": "",
        "schedule": "",
        "dose_level_id": "",
        "field_species_set": "membrane_voltage|spike_rate|calcium_concentration|amyloid_beta_soluble|phosphorylated_tau|inflammatory_mediator|circuit_instability",
        "trajectory_index": row["trajectory_index"],
        "trajectory_store_path": row["trajectory_store_path"],
        "uncertainty_mean": uncertainty_mean,
        "uncertainty_payload": row["uncertainty_payload"],
        "teacher_loss_weight": teacher_precision_from_uncertainty(config, uncertainty_mean),
        "note": "Main NEURON/RxD full microenvironment mechanism teacher row",
    }


def neuron_teacher_rows(config: dict, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [neuron_teacher_row(config, row) for row in rows]


def mainline_dynamics_teacher_rows(config: dict) -> list[dict[str, object]]:
    root = resolve_repo_path(config["dynamics_ready_registry"]["local_root"])
    relative_paths = config["dynamics_ready_registry"]["relative_paths"]
    simulation_rows = read_tsv(root / relative_paths["simulation_catalog"])
    active_program = str(config["dynamics_ready_registry"]["ad_first_data_readiness"]["primary_disease_program"])
    uncertainty_mean = float(config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]["uncertainty_floor"])
    species_set = "|".join(str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"])
    rows: list[dict[str, object]] = []
    for simulation_row in simulation_rows:
        if str(simulation_row["disease_program"]) != active_program:
            continue
        rows.append(
            {
                "teacher_uid": stable_uid("full_microenvironment_teacher", simulation_row["simulation_uid"]),
                "contract_version": teacher_contract_version(config),
                "disease_program": active_program,
                "disease_family": "AD",
                "teacher_family": "ad_full_microenvironment_main_dynamics",
                "teacher_role": "mechanism_teacher",
                "calibration_role": "mechanism_teacher_loss",
                "observed_or_synthetic": "synthetic_mechanistic_main_dynamics",
                "source_model_id": "ad_full_microenvironment_main_dynamics",
                "source_repository": "neuronsifter_main_dynamics_yaml",
                "provenance_level": "provisional_parameter_evidence",
                "teacher_evidence_level": "configuration_generated_provisional",
                "calibration_status": "calibration_pending",
                "main_pretraining_eligibility": "calibration_or_compatibility_only",
                "source_simulation_uid": simulation_row["simulation_uid"],
                "source_protocol_uid": "",
                "source_parameter_uid": simulation_row.get("parameter_draw_uid", ""),
                "regimen_uid": simulation_row["regimen_uid"],
                "compound_uid": "",
                "route": simulation_row["route"],
                "schedule": simulation_row["schedule"],
                "dose_level_id": simulation_row["dose_level_id"],
                "field_species_set": species_set,
                "trajectory_index": simulation_row["trajectory_index"],
                "trajectory_store_path": str((root / relative_paths["trajectory_store"]).resolve().relative_to(REPO_ROOT)),
                "uncertainty_mean": uncertainty_mean,
                "uncertainty_payload": json_payload({"parameter_evidence_status": simulation_row.get("parameter_evidence_status", "provisional_parameter_evidence")}),
                "teacher_loss_weight": teacher_precision_from_uncertainty(config, uncertainty_mean),
                "note": f"AD-first full microenvironment main dynamics trajectory; {simulation_row.get('note', '')}",
            }
        )
    if not rows:
        raise ValueError("Main dynamics simulation catalog has no AD trajectory-backed rows for full microenvironment assembly.")
    return rows


def modeldb_materialization_status(modeldb_config: dict[str, object]) -> str:
    mechanism_root = resolve_repo_path(str(modeldb_config["mechanism_root"]))
    compiled_library = mechanism_root / str(modeldb_config["compiled_library_relpath"])
    if compiled_library.exists():
        return "compiled_public_source"
    if resolve_repo_path(str(modeldb_config["extracted_root"])).exists():
        return "downloaded_public_source_pending_compile"
    return "configured_public_source_pending_download"


def allen_materialization_status(allen_config: dict[str, object]) -> str:
    local_root = resolve_repo_path(str(allen_config["local_root"]))
    compiled_count = 0
    materialized_count = 0
    for neuronal_model_id in allen_config["neuronal_model_ids"]:
        model_id_text = str(neuronal_model_id)
        model_root = local_root / str(allen_config["extracted_root_template"]).format(neuronal_model_id=model_id_text)
        if model_root.exists():
            materialized_count += 1
        compiled_library = model_root / str(allen_config["mechanism_root"]) / str(allen_config["compiled_library_relpath"])
        if compiled_library.exists():
            compiled_count += 1
    if compiled_count == len(allen_config["neuronal_model_ids"]):
        return "compiled_public_source"
    if materialized_count > 0:
        return "downloaded_public_source_pending_compile"
    return "configured_public_source_pending_download"


def source_model_rows(config: dict) -> list[dict[str, object]]:
    osp_library = config["dynamics_ready_registry"]["osp_pksim_runtime"]["model_library"]
    neuron_sources = config["dynamics_ready_registry"]["neuron_rxd_teacher"]["model_sources"]
    return [
        {
            "source_model_id": "ad_full_microenvironment_main_dynamics",
            "source_repository": "neuronsifter_main_dynamics_yaml",
            "source_url": "Libs/configs/model/dynamics_solver.yaml",
            "local_path": "Libs/utils/model/dynamics_fields.py",
            "model_family": "full_microenvironment_main_dynamics",
            "disease_scope": "ad_control",
            "license_or_terms": "project_internal_research_code",
            "provenance_level": "provisional_parameter_evidence",
            "materialization_status": "configured_streaming_generator",
            "note": "AD-first main dynamics generator with YAML-controlled provisional parameter evidence",
        },
        {
            "source_model_id": "osp_pbpk_model_library",
            "source_repository": "osp_pksim_model_library",
            "source_url": osp_library["repository_url"],
            "local_path": osp_library["local_repo_path"],
            "model_family": "pbpk",
            "disease_scope": "ad_control",
            "license_or_terms": "open_systems_pharmacology_public_repository_terms",
            "provenance_level": "repository_generated_mechanistic_rollout",
            "materialization_status": "configured_public_repository",
            "note": "OSP/PK-Sim PBPK model library used as mechanism teacher source",
        },
        {
            "source_model_id": neuron_sources["modeldb"]["source_model_id"],
            "source_repository": "modeldb",
            "source_url": neuron_sources["modeldb"]["repository_url"],
            "local_path": neuron_sources["modeldb"]["extracted_root"],
            "model_family": "neuron_rxd",
            "disease_scope": "ad_control",
            "license_or_terms": "public_modeldb_terms",
            "provenance_level": "repository_generated_biophysical_rollout",
            "materialization_status": modeldb_materialization_status(neuron_sources["modeldb"]),
            "note": "ModelDB source priority for AD microenvironment electrophysiology",
        },
        {
            "source_model_id": neuron_sources["allen_cell_types"]["source_model_id"],
            "source_repository": "allen_cell_types",
            "source_url": neuron_sources["allen_cell_types"]["repository_url"],
            "local_path": neuron_sources["allen_cell_types"]["local_root"],
            "model_family": "neuron_rxd",
            "disease_scope": "ad_control",
            "license_or_terms": "allen_institute_public_terms",
            "provenance_level": "repository_generated_biophysical_rollout",
            "materialization_status": allen_materialization_status(neuron_sources["allen_cell_types"]),
            "note": "Allen Cell Types source priority for biophysical cell models",
        },
    ]


def validate_teacher_row(row: dict[str, object]) -> None:
    if row["teacher_role"] == "legacy_baseline" and row["calibration_role"] != "legacy_baseline":
        raise ValueError("Legacy baseline rows cannot be used outside legacy_baseline calibration_role.")
    if row["teacher_role"] != "legacy_baseline" and row["calibration_role"] == "compatibility_baseline":
        raise ValueError("Primary teacher rows cannot use compatibility_baseline calibration_role.")


def source_trajectory_shape(row: dict[str, object]) -> tuple[int, int, int, int, int]:
    source_store_path = REPO_ROOT / str(row["trajectory_store_path"])
    if (source_store_path / "store_manifest.json").exists():
        manifest = read_store_manifest(source_store_path)
        return tuple(int(value) for value in manifest["arrays"]["trajectories"]["shape"])
    metadata_path = source_store_path / "trajectories" / ".zarray"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Trajectory metadata missing for source store: {source_store_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return tuple(int(value) for value in metadata["shape"])


def load_source_rows(source_store_path: Path, row_indices: np.ndarray, uncertainty_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (source_store_path / "store_manifest.json").exists():
        trajectories = load_chunked_rows(source_store_path, "trajectories", row_indices)
        uncertainty = np.repeat(np.asarray(uncertainty_values, dtype=np.float32)[:, None], trajectories.shape[-1], axis=1)
        return trajectories, uncertainty
    return (
        load_zarr_rows(source_store_path, "trajectories", row_indices),
        load_zarr_rows(source_store_path, "teacher_uncertainty", row_indices),
    )


def rebuild_primary_trajectory_store(config: dict, root: Path, relative_paths: dict[str, str], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    trajectory_rows = [row for row in rows if str(row["trajectory_store_path"]).strip()]
    primary_store_path = root / relative_paths["trajectory_store"]
    source_store_paths: set[str] = set()
    for unified_index, row in enumerate(trajectory_rows):
        source_store_path = REPO_ROOT / str(row["trajectory_store_path"])
        source_store_paths.add(str(source_store_path.resolve().relative_to(REPO_ROOT)))
        row["source_trajectory_store_path"] = row["trajectory_store_path"]
        row["source_trajectory_index"] = row["trajectory_index"]
        row["trajectory_index"] = unified_index
        row["trajectory_store_path"] = str(primary_store_path.resolve().relative_to(REPO_ROOT))
    if not trajectory_rows:
        raise ValueError("Full microenvironment registry requires trajectory-backed OSP and NEURON/RxD teacher rows.")
    reference_shape = source_trajectory_shape(trajectory_rows[0])
    trajectory_shape = (len(trajectory_rows), reference_shape[1], reference_shape[2], reference_shape[3], reference_shape[4])
    uncertainty_shape = (len(trajectory_rows), reference_shape[4])

    def selected_source_groups(row_slice: slice) -> dict[str, list[tuple[int, dict[str, object]]]]:
        selected_rows = trajectory_rows[row_slice.start : row_slice.stop]
        source_groups: dict[str, list[tuple[int, dict[str, object]]]] = {}
        for output_position, row in enumerate(selected_rows):
            source_groups.setdefault(str(row["source_trajectory_store_path"]), []).append((output_position, row))
        return source_groups

    def trajectory_rows_for_slice(slices: tuple[slice, ...]) -> np.ndarray:
        trajectories = np.zeros(tuple(axis_slice.stop - axis_slice.start for axis_slice in slices), dtype=np.float32)
        source_groups = selected_source_groups(slices[0])
        for source_store_path_text, group_rows in source_groups.items():
            row_indices = np.asarray([int(row["source_trajectory_index"]) for _, row in group_rows], dtype=np.int64)
            uncertainty_values = np.asarray([float(row["uncertainty_mean"]) for _, row in group_rows], dtype=np.float32)
            source_trajectories, _ = load_source_rows(REPO_ROOT / source_store_path_text, row_indices, uncertainty_values)
            for group_position, (output_position, _) in enumerate(group_rows):
                trajectories[output_position] = source_trajectories[group_position, slices[1], slices[2], slices[3], slices[4]]

        return trajectories

    def uncertainty_rows_for_slice(slices: tuple[slice, ...]) -> np.ndarray:
        selected_count = slices[0].stop - slices[0].start
        uncertainty = np.zeros((selected_count, slices[1].stop - slices[1].start), dtype=np.float32)
        source_groups = selected_source_groups(slices[0])
        for source_store_path_text, group_rows in source_groups.items():
            row_indices = np.asarray([int(row["source_trajectory_index"]) for _, row in group_rows], dtype=np.int64)
            uncertainty_values = np.asarray([float(row["uncertainty_mean"]) for _, row in group_rows], dtype=np.float32)
            _, source_uncertainty = load_source_rows(REPO_ROOT / source_store_path_text, row_indices, uncertainty_values)
            for group_position, (output_position, _) in enumerate(group_rows):
                uncertainty[output_position] = source_uncertainty[group_position, slices[1]]

        return uncertainty

    def trajectory_chunk_loader(slices: tuple[slice, ...]) -> np.ndarray:
        return trajectory_rows_for_slice(slices).astype(np.float32)

    def uncertainty_chunk_loader(slices: tuple[slice, ...]) -> np.ndarray:
        return uncertainty_rows_for_slice(slices).astype(np.float32)

    write_microenvironment_zarr_from_loaders(
        primary_store_path,
        trajectory_shape,
        uncertainty_shape,
        config,
        {"teacher_family": "ad_full_microenvironment_unified", "source_store_count": len(source_store_paths)},
        trajectory_chunk_loader,
        uncertainty_chunk_loader,
    )
    for row in trajectory_rows:
        del row["source_trajectory_store_path"]
        del row["source_trajectory_index"]
    return rows


def write_teacher_catalog_streaming(config: dict, output_path: Path, trajectory_rows: list[dict[str, object]], chembl_path: Path) -> tuple[int, int]:
    fieldnames = config["schemas"]["full_microenvironment_teacher_fields"]
    ensure_directory(output_path.parent)
    temporary_path = temporary_output_path(output_path)
    total_count = 0
    trajectory_count = 0
    chembl_count = 0
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in trajectory_rows:
            validate_teacher_row(row)
            writer.writerow(row)
            total_count += 1
            trajectory_count += 1 if str(row["trajectory_store_path"]).strip() else 0
        for source_row in iter_tsv(chembl_path):
            row = chembl_teacher_row(config, source_row)
            validate_teacher_row(row)
            writer.writerow(row)
            total_count += 1
            chembl_count += 1
    if chembl_count == 0:
        raise ValueError(f"Required chembl_pbpk_parameter_catalog has no data rows: {chembl_path}")
    temporary_path.replace(output_path)
    return total_count, trajectory_count


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    root = resolve_repo_path(registry["local_root"])
    relative_paths = registry["relative_paths"]
    ensure_directory(root)
    pbpk_rows = required_rows(root / relative_paths["pbpk_rollout_catalog"], "pbpk_rollout_catalog")
    neuron_rows = required_rows(root / relative_paths["neuron_rxd_protocol_catalog"], "neuron_rxd_protocol_catalog")
    chembl_path = root / relative_paths["chembl_pbpk_parameter_catalog"]
    require_streaming_catalog(chembl_path, "chembl_pbpk_parameter_catalog")
    trajectory_rows = mainline_dynamics_teacher_rows(config) + pbpk_teacher_rows(config, pbpk_rows) + neuron_teacher_rows(config, neuron_rows)
    trajectory_rows = rebuild_primary_trajectory_store(config, root, relative_paths, trajectory_rows)
    teacher_output_path = root / relative_paths["full_microenvironment_teacher_catalog"]
    source_output_path = root / relative_paths["source_model_catalog"]
    full_count, trajectory_count = write_teacher_catalog_streaming(config, teacher_output_path, trajectory_rows, chembl_path)
    source_rows = source_model_rows(config)
    write_tsv(source_output_path, config["schemas"]["teacher_source_model_fields"], source_rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=teacher_output_path,
        full_microenvironment_teacher_row_count=full_count,
        trajectory_backed_row_count=trajectory_count,
        source_model_row_count=len(source_rows),
    )


if __name__ == "__main__":
    main()
