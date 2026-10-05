from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path
from Libs.utils.data.manual_access_workflows import manual_expected_root_text, manual_workflow, requirement_for_asset, required_pattern_for_requirement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the processed registry catalogs.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def validate_required_outputs(config: dict, processed_root: Path) -> dict[str, Path]:
    relative_paths = config["processed_registry"]["relative_paths"]
    output_paths = {
        "asset_catalog": processed_root / relative_paths["asset_catalog"],
        "study_catalog": processed_root / relative_paths["study_catalog"],
        "manual_access_catalog": processed_root / relative_paths["manual_access_catalog"],
        "registry_summary": processed_root / relative_paths["registry_summary"],
    }
    for output_path in output_paths.values():
        if not output_path.exists():
            raise FileNotFoundError(f"Processed registry output is missing: {output_path}")
    return output_paths


def validate_model_ready_rows(study_rows: list[dict[str, str]]) -> None:
    for row in study_rows:
        if row["model_ready_flag"] != "yes":
            continue
        if row["include_flag"] != "yes":
            raise ValueError(f"Model-ready study must be included: {row['study_uid']}")
        ready_count = int(row["downloaded_asset_count"]) + int(row["reused_cached_asset_count"])
        if ready_count <= 0:
            raise ValueError(f"Model-ready study lacks ready assets: {row['study_uid']}")


def validate_manual_rows(
    config: dict,
    asset_rows: list[dict[str, str]],
    manual_rows: list[dict[str, str]],
) -> None:
    allowed_statuses = (
        set(config["processed_registry"]["blocked_statuses"])
        | set(config["processed_registry"]["prepared_statuses"])
        | set(config["processed_registry"]["resolved_statuses"])
    )
    asset_keys = {
        (
            row["source_key"],
            row["source_accession"],
            row["asset_name"],
            row["status"],
            row["asset_url"],
        )
        for row in asset_rows
    }
    for row in manual_rows:
        if row["status"] not in allowed_statuses:
            raise ValueError(f"Manual access row has invalid status: {row['manual_uid']}")
        if row["resolution_status"] not in {"pending", "prepared", "resolved"}:
            raise ValueError(f"Manual access row has invalid resolution_status: {row['manual_uid']}")
        if row["manual_action_required_flag"] not in {"yes", "no"}:
            raise ValueError(f"Manual access row has invalid manual_action_required_flag: {row['manual_uid']}")
        asset_key = (
            row["source_key"],
            row["source_accession"],
            row["asset_name"],
            row["status"],
            row["asset_url"],
        )
        if asset_key not in asset_keys:
            raise ValueError(f"Manual access row does not match any asset row: {row['manual_uid']}")
        workflow = manual_workflow(config, row["source_key"])
        requirement = requirement_for_asset(workflow, row["source_accession"], row["asset_name"])
        expected_root = manual_expected_root_text(row["source_key"], workflow) if workflow else ""
        expected_pattern = required_pattern_for_requirement(requirement) if requirement else ""
        if row["manual_drop_expected_root"] != expected_root:
            raise ValueError(f"Manual drop root mismatch for {row['manual_uid']}")
        if row["required_artifact_pattern"] != expected_pattern:
            raise ValueError(f"Required artifact pattern mismatch for {row['manual_uid']}")
        if row["status"] in config["processed_registry"]["blocked_statuses"] and row["resolution_status"] != "pending":
            raise ValueError(f"Blocked manual row must remain pending: {row['manual_uid']}")
        if row["status"] in config["processed_registry"]["prepared_statuses"] and row["resolution_status"] != "prepared":
            raise ValueError(f"Prepared manual row must remain prepared: {row['manual_uid']}")
        if row["status"] in config["processed_registry"]["resolved_statuses"] and row["resolution_status"] != "resolved":
            raise ValueError(f"Resolved manual row must remain resolved: {row['manual_uid']}")


def validate_summary_rows(
    config: dict,
    asset_rows: list[dict[str, str]],
    study_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> None:
    if len(summary_rows) != len(config["sources"]):
        raise ValueError("Processed registry summary does not cover every source.")

    blocked_statuses = set(config["processed_registry"]["blocked_statuses"])
    prepared_statuses = set(config["processed_registry"]["prepared_statuses"])
    not_available_statuses = set(config["processed_registry"]["not_available_statuses"])
    asset_group_lookup: dict[str, str] = {}
    for asset_group, source_keys in config["processed_registry"]["asset_group_map"].items():
        for source_key in source_keys:
            if source_key in asset_group_lookup:
                continue
            asset_group_lookup[source_key] = asset_group

    asset_rows_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    study_rows_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in asset_rows:
        asset_rows_by_source[row["source_key"]].append(row)
    for row in study_rows:
        study_rows_by_source[row["source_key"]].append(row)

    nemo_source_root = resolve_repo_path(config["sources"]["nemo_archive"]["local_root"])
    nemo_reprobe_config = config["reprobe_public_sources"]["sources"]["nemo_archive"]
    nemo_candidate_audit_path = nemo_source_root / nemo_reprobe_config["candidate_sync_audit_relative_path"]
    nemo_primary_queue_path = nemo_source_root / nemo_reprobe_config["primary_candidate_queue_relative_path"]
    nemo_candidate_audit_rows = read_tsv(nemo_candidate_audit_path) if nemo_candidate_audit_path.exists() else []
    nemo_primary_queue_rows = read_tsv(nemo_primary_queue_path) if nemo_primary_queue_path.exists() else []
    discovered_path = nemo_source_root / nemo_reprobe_config["discovered_collections_relative_path"]
    expected_nemo_metrics = {
        "nemo_public_discovery_complete_flag": "yes" if discovered_path.exists() else "no",
        "nemo_primary_open_candidate_count": len(nemo_primary_queue_rows),
        "nemo_primary_materialized_asset_count": sum(
            int(row["materialized_asset_count"])
            for row in nemo_candidate_audit_rows
            if row["match_scope"] == str(nemo_reprobe_config["primary_match_scope"])
        ),
        "nemo_reference_materialized_asset_count": sum(
            int(row["materialized_asset_count"])
            for row in nemo_candidate_audit_rows
            if row["match_scope"] == str(nemo_reprobe_config["reference_match_scope"])
        ),
    }

    for summary_row in summary_rows:
        source_key = summary_row["source_key"]
        if source_key not in config["sources"]:
            raise ValueError(f"Unknown source in processed registry summary: {source_key}")
        if summary_row["asset_group"] != asset_group_lookup[source_key]:
            raise ValueError(f"Asset group mismatch in processed registry summary: {source_key}")

        source_assets = asset_rows_by_source.get(source_key, [])
        source_studies = study_rows_by_source.get(source_key, [])
        expected = {
            "receipt_row_count": len(source_assets),
            "downloaded_asset_count": sum(1 for row in source_assets if row["status"] == "downloaded"),
            "reused_cached_asset_count": sum(1 for row in source_assets if row["status"] == "reused_cached"),
            "blocked_asset_count": sum(1 for row in source_assets if row["status"] in blocked_statuses),
            "prepared_asset_count": sum(1 for row in source_assets if row["status"] in prepared_statuses),
            "not_available_asset_count": sum(1 for row in source_assets if row["status"] in not_available_statuses),
            "study_row_count": len(source_studies),
            "included_study_count": sum(1 for row in source_studies if row["include_flag"] == "yes"),
            "model_ready_study_count": sum(1 for row in source_studies if row["model_ready_flag"] == "yes"),
            "patient_cohort_study_count": sum(1 for row in source_studies if row["sample_context"] == "patient_cohort"),
            "mechanistic_study_count": sum(1 for row in source_studies if row["sample_context"] == "mechanistic_model"),
        }
        if source_key == "nemo_archive":
            expected["nemo_primary_open_candidate_count"] = expected_nemo_metrics["nemo_primary_open_candidate_count"]
            expected["nemo_primary_materialized_asset_count"] = expected_nemo_metrics["nemo_primary_materialized_asset_count"]
            expected["nemo_reference_materialized_asset_count"] = expected_nemo_metrics["nemo_reference_materialized_asset_count"]
        else:
            expected["nemo_primary_open_candidate_count"] = 0
            expected["nemo_primary_materialized_asset_count"] = 0
            expected["nemo_reference_materialized_asset_count"] = 0
        for field_name, expected_value in expected.items():
            if int(summary_row[field_name]) != expected_value:
                raise ValueError(f"Summary mismatch for {source_key} field {field_name}: {summary_row[field_name]} != {expected_value}")
        expected_discovery_flag = expected_nemo_metrics["nemo_public_discovery_complete_flag"] if source_key == "nemo_archive" else "no"
        if summary_row["nemo_public_discovery_complete_flag"] != expected_discovery_flag:
            raise ValueError(
                f"Summary mismatch for {source_key} field nemo_public_discovery_complete_flag: "
                f"{summary_row['nemo_public_discovery_complete_flag']} != {expected_discovery_flag}"
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    processed_root = resolve_repo_path(config["processed_registry"]["local_root"])
    output_paths = validate_required_outputs(config, processed_root)

    validate_header(output_paths["asset_catalog"], config["schemas"]["processed_asset_fields"])
    validate_header(output_paths["study_catalog"], config["schemas"]["processed_study_fields"])
    validate_header(output_paths["manual_access_catalog"], config["schemas"]["processed_manual_access_fields"])
    validate_header(output_paths["registry_summary"], config["schemas"]["processed_registry_summary_fields"])

    asset_rows = read_tsv(output_paths["asset_catalog"])
    study_rows = read_tsv(output_paths["study_catalog"])
    manual_rows = read_tsv(output_paths["manual_access_catalog"])
    summary_rows = read_tsv(output_paths["registry_summary"])

    validate_model_ready_rows(study_rows)
    validate_manual_rows(config, asset_rows, manual_rows)
    validate_summary_rows(config, asset_rows, study_rows, summary_rows)
    pipeline_logger.progress(
        "processed_registry_validated",
        artifact=output_paths["registry_summary"],
        asset_row_count=len(asset_rows),
        study_row_count=len(study_rows),
        manual_row_count=len(manual_rows),
        summary_row_count=len(summary_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_paths["registry_summary"],
        asset_row_count=len(asset_rows),
        study_row_count=len(study_rows),
        manual_row_count=len(manual_rows),
        summary_row_count=len(summary_rows),
    )

if __name__ == "__main__":
    main()
