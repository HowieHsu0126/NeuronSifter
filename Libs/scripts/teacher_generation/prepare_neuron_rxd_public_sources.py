from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_tsv


def source_download_environment(neuron_config: dict[str, object]) -> dict[str, str]:
    environment = dict(os.environ)
    proxy_policy = dict(neuron_config["source_download_proxy_policy"])
    if bool(proxy_policy["unset_loopback_proxy_on_compute"]):
        loopback_patterns = [str(value) for value in proxy_policy["loopback_host_patterns"]]
        for variable_name in proxy_policy["environment_variables"]:
            value = environment.get(str(variable_name), "")
            if any(pattern in value for pattern in loopback_patterns):
                environment.pop(str(variable_name), None)
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize public ModelDB and Allen NEURON/RxD source models for AD microenvironment teachers.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--skip-allen-download", action="store_true")
    parser.add_argument("--compile-modeldb", action="store_true")
    parser.add_argument("--compile-allen", action="store_true")
    return parser.parse_args()


def require_executable(executable_name: str) -> None:
    if shutil.which(executable_name) is None:
        raise FileNotFoundError(f"Required executable is not available on PATH: {executable_name}")


def materialize_modeldb_source(modeldb_config: dict[str, object], compile_modeldb: bool, download_environment: dict[str, str]) -> dict[str, object]:
    archive_path = resolve_repo_path(str(modeldb_config["archive_path"]))
    extracted_root = resolve_repo_path(str(modeldb_config["extracted_root"]))
    mechanism_root = resolve_repo_path(str(modeldb_config["mechanism_root"]))
    simulation_hoc_path = resolve_repo_path(str(modeldb_config["simulation_hoc_path"]))
    ensure_directory(archive_path.parent)
    if not archive_path.exists():
        require_executable("curl")
        subprocess.run(["curl", "-L", str(modeldb_config["download_url"]), "-o", str(archive_path)], check=True, cwd=REPO_ROOT, env=download_environment)
    if not extracted_root.exists():
        with ZipFile(archive_path) as archive:
            archive.extractall(resolve_repo_path(str(modeldb_config["local_root"])) / "source")
    required_paths = [extracted_root, mechanism_root, simulation_hoc_path]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"ModelDB source materialization is incomplete: {missing_paths}")
    compiled_library = mechanism_root / str(modeldb_config["compiled_library_relpath"])
    if compile_modeldb:
        require_executable("nrnivmodl")
        subprocess.run(["nrnivmodl"], check=True, cwd=mechanism_root)
    return {
        "source_model_id": modeldb_config["source_model_id"],
        "source_repository": "modeldb",
        "source_url": modeldb_config["repository_url"],
        "local_path": str(extracted_root.resolve().relative_to(REPO_ROOT)),
        "model_family": "neuron_rxd",
        "disease_scope": "ad_control",
        "license_or_terms": "public_modeldb_terms",
        "provenance_level": "repository_generated_biophysical_rollout",
        "materialization_status": "compiled_public_source" if compiled_library.exists() else "downloaded_public_source_pending_compile",
        "note": f"ModelDB source archive {archive_path.resolve().relative_to(REPO_ROOT)} with mechanism root {mechanism_root.resolve().relative_to(REPO_ROOT)}",
    }


def materialize_allen_source(allen_config: dict[str, object], skip_download: bool, compile_allen: bool, download_environment: dict[str, str]) -> dict[str, object]:
    local_root = resolve_repo_path(str(allen_config["local_root"]))
    ensure_directory(local_root)
    if not skip_download:
        require_executable("curl")
        for neuronal_model_id in allen_config["neuronal_model_ids"]:
            model_id_text = str(neuronal_model_id)
            archive_path = local_root / str(allen_config["archive_template"]).format(neuronal_model_id=model_id_text)
            extracted_root = local_root / str(allen_config["extracted_root_template"]).format(neuronal_model_id=model_id_text)
            if not archive_path.exists():
                download_url = str(allen_config["download_url_template"]).format(neuronal_model_id=model_id_text)
                subprocess.run(["curl", "-L", download_url, "-o", str(archive_path)], check=True, cwd=REPO_ROOT, env=download_environment)
            if not extracted_root.exists():
                ensure_directory(extracted_root)
                with ZipFile(archive_path) as archive:
                    archive.extractall(extracted_root)
            mechanism_root = extracted_root / str(allen_config["mechanism_root"])
            if compile_allen:
                require_executable("nrnivmodl")
                subprocess.run(["nrnivmodl"], check=True, cwd=mechanism_root)
    materialized_models = [path for path in local_root.iterdir() if path.is_dir()]
    compiled_models = [
        model_root
        for model_root in materialized_models
        if (model_root / str(allen_config["mechanism_root"]) / str(allen_config["compiled_library_relpath"])).exists()
    ]
    return {
        "source_model_id": allen_config["source_model_id"],
        "source_repository": "allen_cell_types",
        "source_url": allen_config["repository_url"],
        "local_path": str(local_root.resolve().relative_to(REPO_ROOT)),
        "model_family": "neuron_rxd",
        "disease_scope": "ad_control",
        "license_or_terms": "allen_institute_public_terms",
        "provenance_level": "repository_generated_biophysical_rollout",
        "materialization_status": "compiled_public_source" if compiled_models else ("downloaded_public_source" if materialized_models else "configured_public_source_pending_download"),
        "note": f"Allen Cell Types retrieval strategy {allen_config['retrieval_strategy']} for neuronal_model_ids={allen_config['neuronal_model_ids']}",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    output_path = resolve_repo_path(registry["local_root"]) / registry["relative_paths"]["source_model_catalog"]
    neuron_config = config["dynamics_ready_registry"]["neuron_rxd_teacher"]
    model_sources = neuron_config["model_sources"]
    download_environment = source_download_environment(neuron_config)
    rows = [
        materialize_modeldb_source(model_sources["modeldb"], bool(args.compile_modeldb), download_environment),
        materialize_allen_source(model_sources["allen_cell_types"], bool(args.skip_allen_download), bool(args.compile_allen), download_environment),
    ]
    write_tsv(output_path, config["schemas"]["teacher_source_model_fields"], rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        source_model_row_count=len(rows),
        modeldb_status=rows[0]["materialization_status"],
        allen_status=rows[1]["materialization_status"],
    )


if __name__ == "__main__":
    main()
