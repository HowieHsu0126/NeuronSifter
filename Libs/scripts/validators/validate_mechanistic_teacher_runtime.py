from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path
import shutil
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AD-first PBPK and NEURON/RxD teacher runtime contracts.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def require_mapping_fields(payload: dict[str, object], required_fields: list[str], label: str) -> None:
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise ValueError(f"{label} is missing required fields: {missing_fields}")


def load_yaml_manifest(manifest_path: str) -> dict[str, object]:
    resolved_path = resolve_repo_path(manifest_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML manifest root must be a mapping: {resolved_path}")
    return payload


def validate_osp_runtime(config: dict, pipeline_logger) -> None:
    dynamics_config = config["dynamics_ready_registry"]
    osp_runtime = dynamics_config["osp_pksim_runtime"]
    require_mapping_fields(
        osp_runtime,
        [
            "runtime_strategy",
            "required_r_packages",
            "environment_manifest_path",
            "rollout_output_root",
            "raw_output_cache",
            "horizon_hours",
            "step_hours",
            "model_library",
            "output_path_mapping",
            "rollout_designs",
            "zarr_chunking",
            "conda_environment",
            "teacher_role",
            "source_repository",
            "provenance_level",
        ],
        "osp_pksim_runtime",
    )
    if osp_runtime["runtime_strategy"] != "reproducible_environment_script":
        raise ValueError("OSP runtime strategy must be reproducible_environment_script.")
    require_mapping_fields(
        osp_runtime["raw_output_cache"],
        ["reuse_existing_outputs", "regenerate_invalid_cache", "standardization_policy_uid", "minimum_size_bytes", "required_columns"],
        "osp_pksim_runtime.raw_output_cache",
    )
    if not str(osp_runtime["raw_output_cache"]["standardization_policy_uid"]).strip():
        raise ValueError("osp_pksim_runtime.raw_output_cache.standardization_policy_uid must be non-empty.")
    if int(osp_runtime["raw_output_cache"]["minimum_size_bytes"]) <= 0:
        raise ValueError("osp_pksim_runtime.raw_output_cache.minimum_size_bytes must be positive.")
    missing_cached_columns = sorted(
        {
            "standardization_policy_uid",
            "source_output_path",
            "time_hours",
            "concentration",
            "plasma_concentration",
            "csf_concentration",
            "brain_isf_concentration",
        }
        - {str(value) for value in osp_runtime["raw_output_cache"]["required_columns"]}
    )
    if missing_cached_columns:
        raise ValueError(f"osp_pksim_runtime.raw_output_cache.required_columns is missing: {missing_cached_columns}")
    require_mapping_fields(
        osp_runtime["model_library"],
        ["repository_url", "local_repo_path", "model_file_glob", "supported_model_extensions", "smoke_test_model_path", "library_discovery"],
        "osp_pksim_runtime.model_library",
    )
    require_mapping_fields(
        osp_runtime["model_library"]["library_discovery"],
        ["enabled", "design_id_prefix", "subject_covariate_profile", "calibration_role", "regimen_variant_id", "route", "schedule", "dose_level_id", "dose_multiplier"],
        "osp_pksim_runtime.model_library.library_discovery",
    )
    model_library_root = resolve_repo_path(str(osp_runtime["model_library"]["local_repo_path"]))
    if not model_library_root.exists():
        raise FileNotFoundError(f"Configured OSP model library is missing: {model_library_root}")
    model_files = list(model_library_root.glob(str(osp_runtime["model_library"]["model_file_glob"])))
    if not model_files:
        raise FileNotFoundError(f"Configured OSP model glob found no files: {model_library_root / str(osp_runtime['model_library']['model_file_glob'])}")
    smoke_model_path = resolve_repo_path(str(osp_runtime["model_library"]["smoke_test_model_path"]))
    if not smoke_model_path.exists():
        raise FileNotFoundError(f"Configured OSP smoke-test model is missing: {smoke_model_path}")
    if smoke_model_path.suffix.lower() not in {str(value).lower() for value in osp_runtime["model_library"]["supported_model_extensions"]}:
        raise ValueError("OSP smoke-test model extension is not declared in supported_model_extensions.")
    require_mapping_fields(
        osp_runtime["output_path_mapping"],
        [
            "plasma_regex",
            "csf_regex",
            "brain_isf_regex",
            "result_file_regex",
            "csv_metadata_columns",
            "measure_column_keep_regex",
            "max_measure_columns_per_chunk",
            "required_curve_columns",
            "optional_curve_columns",
        ],
        "osp_pksim_runtime.output_path_mapping",
    )
    if int(osp_runtime["output_path_mapping"]["max_measure_columns_per_chunk"]) <= 0:
        raise ValueError("osp_pksim_runtime.output_path_mapping.max_measure_columns_per_chunk must be positive.")
    if "IndividualId" not in {str(value) for value in osp_runtime["output_path_mapping"]["csv_metadata_columns"]}:
        raise ValueError("osp_pksim_runtime.output_path_mapping.csv_metadata_columns must include IndividualId.")
    keep_regex = str(osp_runtime["output_path_mapping"]["measure_column_keep_regex"])
    for required_token in ["plasma", "csf", "brain"]:
        if required_token not in keep_regex.lower():
            raise ValueError(f"osp_pksim_runtime.output_path_mapping.measure_column_keep_regex must include {required_token}.")
    require_mapping_fields(
        osp_runtime["conda_environment"],
        ["environment_name", "environment_file", "execution_strategy", "r_executable", "rscript_executable", "setup_script"],
        "osp_pksim_runtime.conda_environment",
    )
    if osp_runtime["teacher_role"] != "mechanism_teacher":
        raise ValueError("OSP rollout rows must be mechanism_teacher, not compatibility baseline.")
    environment_manifest = load_yaml_manifest(str(osp_runtime["conda_environment"]["environment_file"]))
    manifest_root = environment_manifest["osp_pksim_environment"]
    require_mapping_fields(
        manifest_root,
        ["conda_environment_name", "conda_no_plugins", "conda_solver", "conda_quiet", "conda_channels", "conda_packages", "rscript_executable", "cran_r_packages", "github_r_packages", "required_r_packages", "proxy_policy", "model_library", "output_path_mapping", "rollout_designs"],
        "osp_pksim_environment",
    )
    if manifest_root["conda_environment_name"] != osp_runtime["conda_environment"]["environment_name"]:
        raise ValueError("OSP environment manifest and runtime config disagree on conda environment name.")
    if manifest_root["model_library"]["model_file_glob"] != osp_runtime["model_library"]["model_file_glob"]:
        raise ValueError("OSP environment manifest and runtime config disagree on model_file_glob.")
    if manifest_root["model_library"]["library_discovery"] != osp_runtime["model_library"]["library_discovery"]:
        raise ValueError("OSP environment manifest and runtime config disagree on library_discovery.")
    if manifest_root["output_path_mapping"] != osp_runtime["output_path_mapping"]:
        raise ValueError("OSP environment manifest and runtime config disagree on output_path_mapping.")
    if shutil.which("conda") is None:
        raise FileNotFoundError("conda executable is required to provision the OSP/PK-Sim runtime.")
    pipeline_logger.progress(
        "osp_runtime_configured",
        conda_environment=osp_runtime["conda_environment"]["environment_name"],
        model_file_count=len(model_files),
        smoke_model=str(smoke_model_path.relative_to(REPO_ROOT)),
    )


def validate_chembl_parameterization(config: dict) -> None:
    chembl_parameterization = config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]
    require_mapping_fields(
        chembl_parameterization,
        ["predictor_output_root", "feature_source_catalog", "teacher_role", "source_repository", "provenance_level", "uncertainty_floor", "uncertainty_scale", "parameter_bounds", "descriptor_coefficients"],
        "chembl_pbpk_parameterization",
    )
    if chembl_parameterization["teacher_role"] != "parameter_predicted_teacher":
        raise ValueError("ChEMBL PBPK parameterization must be parameter_predicted_teacher.")


def validate_neuron_teacher(config: dict) -> None:
    neuron_teacher = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
    require_mapping_fields(
        neuron_teacher,
        ["disease_scope", "source_priority", "teacher_role", "source_download_proxy_policy", "protocol_output_root", "state_outputs", "rxd_species", "sanity_bounds", "model_sources", "zarr_chunking", "protocol_runtime", "source_repository", "provenance_level"],
        "neuron_rxd_teacher",
    )
    if neuron_teacher["disease_scope"] != "ad_control":
        raise ValueError("NEURON/RxD teacher must be AD-first for this implementation round.")
    if neuron_teacher["teacher_role"] != "mechanism_teacher":
        raise ValueError("NEURON/RxD teacher must be mechanism_teacher in the main dynamics teacher.")
    require_mapping_fields(neuron_teacher["protocol_runtime"], ["h_dt_ms", "simulation_duration_ms", "stimulus_delay_ms", "stimulus_duration_ms", "stimulus_amplitude_nA"], "neuron_rxd_teacher.protocol_runtime")
    require_mapping_fields(
        neuron_teacher["source_download_proxy_policy"],
        ["unset_loopback_proxy_on_compute", "loopback_host_patterns", "environment_variables"],
        "neuron_rxd_teacher.source_download_proxy_policy",
    )
    require_mapping_fields(
        neuron_teacher["model_sources"]["modeldb"],
        ["source_model_id", "repository_url", "download_url", "local_root", "archive_path", "extracted_root", "mechanism_root", "simulation_hoc_path", "morphology_root", "compiled_library_relpath"],
        "neuron_rxd_teacher.model_sources.modeldb",
    )
    require_mapping_fields(
        neuron_teacher["model_sources"]["allen_cell_types"],
        ["source_model_id", "repository_url", "download_url_template", "retrieval_strategy", "local_root", "neuronal_model_ids", "archive_template", "extracted_root_template", "mechanism_root", "compiled_library_relpath"],
        "neuron_rxd_teacher.model_sources.allen_cell_types",
    )
    for source_name in neuron_teacher["source_priority"]:
        if source_name not in {neuron_teacher["model_sources"]["modeldb"]["source_model_id"], neuron_teacher["model_sources"]["allen_cell_types"]["source_model_id"]}:
            raise ValueError(f"NEURON/RxD source_priority contains undeclared source: {source_name}")
    modeldb_config = neuron_teacher["model_sources"]["modeldb"]
    for path_key in ["archive_path", "extracted_root", "mechanism_root", "simulation_hoc_path", "morphology_root"]:
        materialized_path = resolve_repo_path(str(modeldb_config[path_key]))
        if not materialized_path.exists():
            raise FileNotFoundError(f"ModelDB source path is not materialized for {path_key}: {materialized_path}")
    compiled_library = resolve_repo_path(str(modeldb_config["mechanism_root"])) / str(modeldb_config["compiled_library_relpath"])
    if not compiled_library.exists():
        raise FileNotFoundError(f"ModelDB source mechanisms are not compiled: {compiled_library}")
    if find_spec("neuron") is None:
        raise ModuleNotFoundError("Python package 'neuron' is required for the NEURON/RxD teacher runtime.")
    from neuron import h, rxd

    if h is None or rxd is None:
        raise RuntimeError("NEURON h and rxd modules must be importable.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    dynamics_config = config["dynamics_ready_registry"]
    require_mapping_fields(
        dynamics_config,
        ["teacher_source_policy", "osp_pksim_runtime", "chembl_pbpk_parameterization", "neuron_rxd_teacher"],
        "dynamics_ready_registry",
    )
    teacher_policy = dynamics_config["teacher_source_policy"]
    if teacher_policy["primary_pbpk_teacher"] != "osp_pksim_rollout":
        raise ValueError("PBPK primary teacher must be osp_pksim_rollout.")
    if teacher_policy["legacy_solver_role"] != "compatibility_baseline":
        raise ValueError("Legacy dynamics solver must remain a compatibility_baseline.")
    validate_osp_runtime(config, pipeline_logger)
    validate_chembl_parameterization(config)
    validate_neuron_teacher(config)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        pbpk_teacher=teacher_policy["primary_pbpk_teacher"],
        neuron_teacher_scope=dynamics_config["neuron_rxd_teacher"]["disease_scope"],
        neuron_runtime="available",
    )


if __name__ == "__main__":
    main()
