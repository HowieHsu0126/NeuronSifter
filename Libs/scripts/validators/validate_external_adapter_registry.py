from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate external adapter contract catalog.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--strict-processed", action="store_true")
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def artifact_has_rows(path_value: str) -> bool:
    path = resolve_repo_path(path_value)
    if path.is_dir():
        return any(path.iterdir())
    if not path.exists():
        return False
    if path.suffix.lower() == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in handle) > 1
    return path.stat().st_size > 0


def require_rows(path: Path) -> list[dict[str, str]]:
    rows = read_tsv(path)
    if not rows:
        raise ValueError(f"Required processed adapter catalog is empty: {path}")
    return rows


def iter_tsv_rows(path: Path) -> object:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def first_tsv_row(path: Path) -> dict[str, str]:
    for row in iter_tsv_rows(path):
        return row
    raise ValueError(f"Required processed adapter catalog is empty: {path}")


def require_finite(row: dict[str, str], field_name: str, path: Path) -> None:
    value = float(row[field_name])
    if not math.isfinite(value):
        raise ValueError(f"{path} has non-finite {field_name}: {row}")


def numeric_summary_output_field(field_name: str) -> str:
    mapping = {
        "S_sum": "s_sum",
        "S_mean": "s_mean",
        "S_max": "s_max",
    }
    return mapping.get(field_name, field_name)


def validate_dgidb_processed_catalog(path: Path, config: dict) -> None:
    validate_header(path, config["schemas"]["dgidb_drug_gene_interaction_fields"])
    required_nonempty = [
        "interaction_uid",
        "source_release",
        "retrieval_date",
        "query_gene_symbol",
        "gene_name",
        "gene_concept_id",
        "drug_name",
        "drug_concept_id",
        "matched_target_uid",
        "matched_compound_uid",
        "interaction_id",
        "source_db_set",
        "provenance_level",
        "source_holdout_role",
        "train_flag",
        "eval_flag",
    ]
    row_count = 0
    for row in iter_tsv_rows(path):
        row_count += 1
        missing = [field_name for field_name in required_nonempty if not str(row[field_name]).strip()]
        if missing:
            raise ValueError(f"DGIdb processed catalog row lacks required fields {missing}: {row}")
        require_finite(row, "interaction_score", path)
        if row["drug_in_neuronsifter_universe"] != "yes":
            raise ValueError(f"DGIdb strict catalog contains a row outside the NeuronSifter drug universe: {row}")
        if row["provenance_level"] != "dgidb_api_target_universe_query":
            raise ValueError(f"DGIdb processed catalog has invalid provenance: {row}")
    if row_count == 0:
        raise ValueError(f"Required processed adapter catalog is empty: {path}")


def validate_drugcomb_processed_catalog(path: Path, config: dict) -> None:
    validate_header(path, config["schemas"]["drugcomb_combination_prior_fields"])
    materialization = config["validation"]["external_adapter_registry"]["materialization"]["drugcomb"]
    numeric_fields = [str(value) for value in materialization["required_numeric_summary_fields"]]
    matrix_dir = (
        resolve_repo_path(str(config["validation"]["external_adapter_registry"]["local_root"]))
        / str(config["validation"]["external_adapter_registry"]["relative_paths"]["drugcomb_dose_response_matrix_dir"])
    )
    manifest_path = (
        resolve_repo_path(str(config["validation"]["external_adapter_registry"]["local_root"]))
        / str(config["validation"]["external_adapter_registry"]["relative_paths"]["drugcomb_dose_response_manifest"])
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"DrugComb dose-response manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("full_matrix_flag") != "yes" or str(manifest.get("max_rows", "")).strip():
        raise ValueError(f"DrugComb strict validation requires the full matrix manifest, not smoke output: {manifest_path}")
    if not str(manifest.get("dose_response_source_sha256", "")).strip():
        raise ValueError(f"DrugComb dose-response manifest lacks source hash: {manifest_path}")
    if not str(manifest.get("synergy_summary_source_sha256", "")).strip():
        raise ValueError(f"DrugComb synergy manifest lacks source hash: {manifest_path}")
    if not matrix_dir.exists() or not matrix_dir.is_dir():
        raise FileNotFoundError(f"DrugComb full dose-response matrix directory is missing: {matrix_dir}")
    part_files = sorted(matrix_dir.glob("part-*.tsv"))
    if not part_files:
        raise FileNotFoundError(f"DrugComb full dose-response matrix directory contains no part files: {matrix_dir}")
    for part_file in part_files:
        validate_header(part_file, config["schemas"]["drugcomb_dose_response_matrix_fields"])
    first_part_row = first_tsv_row(part_files[0])
    for field_name in materialization["required_numeric_dose_fields"]:
        require_finite(first_part_row, str(field_name), part_files[0])
    row_count = 0
    for row in iter_tsv_rows(path):
        row_count += 1
        required_nonempty = [
            "combination_uid",
            "block_id",
            "drug_row",
            "drug_col",
            "dose_response_part_path",
            "dose_response_row_count",
            "provenance_level",
            "source_holdout_role",
            "train_flag",
            "eval_flag",
        ]
        missing = [field_name for field_name in required_nonempty if not str(row[field_name]).strip()]
        if missing:
            raise ValueError(f"DrugComb processed catalog row lacks required fields {missing}: {row}")
        for field_name in numeric_fields:
            require_finite(row, numeric_summary_output_field(field_name), path)
        require_finite(row, "primary_synergy_value", path)
        if int(float(row["dose_response_row_count"])) <= 0:
            raise ValueError(f"DrugComb processed catalog has non-positive linked matrix row count: {row}")
        if row["provenance_level"] != "drugcomb_public_synergy_summary_linked_to_full_matrix":
            raise ValueError(f"DrugComb processed catalog has invalid provenance: {row}")
    if row_count == 0:
        raise ValueError(f"Required processed adapter catalog is empty: {path}")


def validate_moleculenet_deepchem_catalog(path: Path, config: dict) -> None:
    validate_header(path, config["schemas"]["molecular_property_baseline_fields"])
    required_nonempty = [
        "baseline_uid",
        "source_key",
        "source_release",
        "materialization_date",
        "compound_uid",
        "deepchem_dataset_class",
        "deepchem_version",
        "provenance_level",
        "source_holdout_role",
        "train_flag",
        "eval_flag",
    ]
    numeric_fields = [
        "feature_count",
        "molecular_weight",
        "alogp",
        "tpsa",
        "hba",
        "hbd",
        "rotatable_bonds",
        "ring_count",
        "lipinski_violation_count",
        "cns_rule_pass_count",
        "cns_rule_total_count",
        "cns_property_score",
        "lipophilicity_window_score",
        "polar_surface_penalty",
    ]
    row_count = 0
    for row in iter_tsv_rows(path):
        row_count += 1
        missing = [field_name for field_name in required_nonempty if not str(row[field_name]).strip()]
        if missing:
            raise ValueError(f"MoleculeNet/DeepChem baseline row lacks required fields {missing}: {row}")
        if row["source_key"] != "moleculenet_deepchem":
            raise ValueError(f"MoleculeNet/DeepChem baseline has invalid source_key: {row}")
        if row["provenance_level"] != "deepchem_numpy_dataset_descriptor_property_baseline":
            raise ValueError(f"MoleculeNet/DeepChem baseline has invalid provenance: {row}")
        for field_name in numeric_fields:
            require_finite(row, field_name, path)
        score = float(row["cns_property_score"])
        if score < 0.0 or score > 1.0:
            raise ValueError(f"MoleculeNet/DeepChem baseline has out-of-range cns_property_score: {row}")
    if row_count == 0:
        raise ValueError(f"Required processed adapter catalog is empty: {path}")


def validate_source_specific_processed_catalog(row: dict[str, str], config: dict) -> None:
    path = resolve_repo_path(str(row["processed_artifact_path"]))
    if row["source_key"] == "dgidb":
        validate_dgidb_processed_catalog(path, config)
    if row["source_key"] == "drugcomb":
        validate_drugcomb_processed_catalog(path, config)
    if row["source_key"] == "moleculenet_deepchem":
        validate_moleculenet_deepchem_catalog(path, config)


def validate_external_adapter_rows(rows: list[dict[str, str]], config: dict[str, object], strict_processed: bool) -> None:
    registry_config = config["validation"]["external_adapter_registry"]
    if not rows:
        raise ValueError("external_adapter_catalog.tsv must not be empty.")
    required_sources = {str(row["source_key"]) for row in registry_config["adapters"]}
    observed_sources = {str(row["source_key"]) for row in rows}
    missing_sources = sorted(required_sources - observed_sources)
    if missing_sources:
        raise ValueError(f"external_adapter_catalog.tsv missing sources: {missing_sources}")
    for row in rows:
        if row["method_contribution_flag"] != "no":
            raise ValueError(f"External adapter is incorrectly marked as a method contribution: {row}")
        if not row["adapter_role"] or not row["provenance_level"] or not row["validation_policy"]:
            raise ValueError(f"External adapter row lacks role/provenance/policy: {row}")
        raw_root = str(row["raw_root"])
        if raw_root.startswith("python_package:"):
            package_name = raw_root.split(":", 1)[1]
            importlib.import_module(package_name)
        elif not raw_root.startswith("external_binary:") and not resolve_repo_path(raw_root).exists():
            raise FileNotFoundError(f"External adapter raw root is unavailable: {raw_root}")
        if strict_processed and row["required_processed_artifact_flag"] == "yes" and not artifact_has_rows(str(row["processed_artifact_path"])):
            raise FileNotFoundError(
                "Required processed external adapter artifact is missing or empty: "
                f"{row['source_key']} -> {row['processed_artifact_path']}"
            )
        if strict_processed and row["source_key"] in {"dgidb", "drugcomb", "moleculenet_deepchem"}:
            validate_source_specific_processed_catalog(row, config)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    registry_config = config["validation"]["external_adapter_registry"]
    path = resolve_repo_path(str(registry_config["local_root"])) / str(registry_config["relative_paths"]["external_adapter_catalog"])
    if not path.exists():
        raise FileNotFoundError(f"Missing external adapter catalog: {path}")
    validate_header(path, config["schemas"]["external_adapter_fields"])
    rows = read_tsv(path)
    validate_external_adapter_rows(rows, config, bool(args.strict_processed))
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        external_adapter_row_count=len(rows),
        strict_processed=bool(args.strict_processed),
    )


if __name__ == "__main__":
    main()
