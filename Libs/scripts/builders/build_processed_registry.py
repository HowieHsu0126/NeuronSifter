from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, slugify, write_tsv
from Libs.utils.data.manual_access_workflows import (
    manual_expected_root_text,
    manual_workflow,
    requirement_for_asset,
    required_pattern_for_requirement,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed registry catalogs from the raw data registry.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def stable_uid(*parts: str) -> str:
    normalized_parts = [part.strip() for part in parts if part and part.strip()]
    joined = "::".join(normalized_parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    slug_seed = "_".join(normalized_parts[:4]) if normalized_parts else "record"
    return f"{slugify(slug_seed)[:80]}_{digest}"


def load_manifest(source_root: Path) -> dict[str, object]:
    manifest_path = source_root / "manifests" / "source_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def asset_group_lookup(asset_group_map: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for asset_group, source_keys in asset_group_map.items():
        for source_key in source_keys:
            if source_key in lookup:
                continue
            lookup[source_key] = asset_group
    return lookup


def load_receipt_rows(source_key: str, source_root: Path, public_downloads: dict[str, dict[str, object]]) -> list[dict[str, str]]:
    download_config = public_downloads.get(source_key)
    if download_config is None:
        return []
    receipt_path = source_root / str(download_config["receipt_relative_path"])
    if not receipt_path.exists():
        return []
    return read_tsv(receipt_path)


def choose_primary_local_path(receipt_rows: list[dict[str, str]], ready_statuses: set[str], prioritized_asset_names: set[str] | None = None) -> str:
    ready_paths = sorted(
        row["local_path"]
        for row in receipt_rows
        if row["status"] in ready_statuses
        and row["local_path"]
        and (not prioritized_asset_names or row["asset_name"] in prioritized_asset_names)
    )
    if ready_paths:
        return ready_paths[0]
    if prioritized_asset_names:
        return ""
    all_paths = sorted(row["local_path"] for row in receipt_rows if row["local_path"])
    if all_paths:
        return all_paths[0]
    return ""


def count_receipt_rows(receipt_rows: list[dict[str, str]], allowed_statuses: set[str]) -> int:
    return sum(1 for row in receipt_rows if row["status"] in allowed_statuses)


def nemo_candidate_metrics(config: dict) -> dict[str, object]:
    reprobe_config = config["reprobe_public_sources"]["sources"]["nemo_archive"]
    source_root = resolve_repo_path(config["sources"]["nemo_archive"]["local_root"])
    discovered_path = source_root / reprobe_config["discovered_collections_relative_path"]
    candidate_audit_path = source_root / reprobe_config["candidate_sync_audit_relative_path"]
    primary_queue_path = source_root / reprobe_config["primary_candidate_queue_relative_path"]
    candidate_audit_rows = read_tsv(candidate_audit_path) if candidate_audit_path.exists() else []
    primary_queue_rows = read_tsv(primary_queue_path) if primary_queue_path.exists() else []
    primary_scope = str(reprobe_config["primary_match_scope"])
    reference_scope = str(reprobe_config["reference_match_scope"])
    return {
        "nemo_public_discovery_complete_flag": "yes" if discovered_path.exists() else "no",
        "nemo_primary_open_candidate_count": len(primary_queue_rows),
        "nemo_primary_materialized_asset_count": sum(
            int(row["materialized_asset_count"])
            for row in candidate_audit_rows
            if row["match_scope"] == primary_scope
        ),
        "nemo_reference_materialized_asset_count": sum(
            int(row["materialized_asset_count"])
            for row in candidate_audit_rows
            if row["match_scope"] == reference_scope
        ),
    }


def build_asset_catalog(config: dict) -> tuple[list[dict[str, object]], dict[str, list[dict[str, str]]], dict[str, str]]:
    asset_group_by_source = asset_group_lookup(config["processed_registry"]["asset_group_map"])
    public_downloads = config.get("public_downloads", {})
    ready_statuses = set(config["processed_registry"]["ready_statuses"])
    asset_rows: list[dict[str, object]] = []
    receipt_rows_by_source: dict[str, list[dict[str, str]]] = {}

    for source_key, source_config in config["sources"].items():
        if source_key not in asset_group_by_source:
            raise ValueError(f"Source key {source_key} is missing from processed_registry.asset_group_map.")
        source_root = resolve_repo_path(source_config["local_root"])
        manifest = load_manifest(source_root)
        receipt_rows = load_receipt_rows(source_key, source_root, public_downloads)
        receipt_rows_by_source[source_key] = receipt_rows
        for receipt_row in receipt_rows:
            unique_tail = receipt_row["local_path"] or receipt_row["asset_url"]
            asset_rows.append(
                {
                    "asset_uid": stable_uid(source_key, receipt_row["source_accession"], receipt_row["asset_name"], unique_tail),
                    "source_key": source_key,
                    "source_name": manifest["source_name"],
                    "asset_group": asset_group_by_source[source_key],
                    "source_accession": receipt_row["source_accession"],
                    "asset_name": receipt_row["asset_name"],
                    "asset_url": receipt_row["asset_url"],
                    "local_path": receipt_row["local_path"],
                    "status": receipt_row["status"],
                    "size_bytes": receipt_row["size_bytes"],
                    "download_date": receipt_row["download_date"],
                    "access_mode": manifest["access_mode"],
                    "license_or_terms": manifest["license_or_terms"],
                    "redistributable_flag": "yes"
                    if manifest["access_mode"] == "public" and receipt_row["status"] in ready_statuses
                    else "no",
                    "note": receipt_row["note"],
                }
            )

    asset_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["asset_name"], row["local_path"]))
    return asset_rows, receipt_rows_by_source, asset_group_by_source


def build_study_catalog(config: dict, receipt_rows_by_source: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    public_downloads = config["public_downloads"]
    ready_statuses = set(config["processed_registry"]["ready_statuses"])
    blocked_statuses = set(config["processed_registry"]["blocked_statuses"])
    prepared_statuses = set(config["processed_registry"]["prepared_statuses"])
    not_available_statuses = set(config["processed_registry"]["not_available_statuses"])
    prioritized_assets_by_source = {
        source_key: set(asset_names)
        for source_key, asset_names in config["cohort_ready_registry"]["primary_asset_priority"].items()
    }
    study_rows: list[dict[str, object]] = []

    for source_key in config["processed_registry"]["study_sources"]:
        source_root = resolve_repo_path(config["sources"][source_key]["local_root"])
        accession_table = source_root / "accession_freeze" / "accessions.tsv"
        accession_rows = read_tsv(accession_table)
        receipt_rows = receipt_rows_by_source.get(source_key, [])
        receipt_rows_by_accession: dict[str, list[dict[str, str]]] = defaultdict(list)
        for receipt_row in receipt_rows:
            receipt_rows_by_accession[receipt_row["source_accession"]].append(receipt_row)
        receipt_relative_path = public_downloads[source_key]["receipt_relative_path"]

        for accession_row in accession_rows:
            study_receipts = receipt_rows_by_accession.get(accession_row["source_accession"], [])
            downloaded_asset_count = count_receipt_rows(study_receipts, {"downloaded"})
            reused_cached_asset_count = count_receipt_rows(study_receipts, {"reused_cached"})
            prioritized_asset_names = prioritized_assets_by_source.get(source_key, set())
            prioritized_ready_count = sum(
                1
                for receipt_row in study_receipts
                if receipt_row["status"] in ready_statuses and receipt_row["asset_name"] in prioritized_asset_names
            )
            study_rows.append(
                {
                    "study_uid": stable_uid(source_key, accession_row["source_accession"]),
                    "source_key": source_key,
                    "source_accession": accession_row["source_accession"],
                    "title": accession_row["title"],
                    "species": accession_row["species"],
                    "tissue_or_biofluid": accession_row["tissue_or_biofluid"],
                    "modality": accession_row["modality"],
                    "sample_context": accession_row["sample_context"],
                    "diagnosis_label": accession_row["diagnosis_label"],
                    "case_count": accession_row["case_count"],
                    "control_count": accession_row["control_count"],
                    "include_flag": accession_row["include_flag"],
                    "exclusion_reason": accession_row["exclusion_reason"],
                    "source_repository": accession_row["source_repository"],
                    "release_type": accession_row["release_type"],
                    "data_granularity": accession_row["data_granularity"],
                    "subject_linkable_flag": accession_row["subject_linkable_flag"],
                    "provenance_level": accession_row["provenance_level"],
                    "world_model_role": accession_row["world_model_role"],
                    "cohort_or_atlas_family": accession_row["cohort_or_atlas_family"],
                    "public_url": accession_row["public_url"],
                    "raw_accession_table": f"{config['sources'][source_key]['local_root']}/accession_freeze/accessions.tsv",
                    "raw_receipt_table": f"{config['sources'][source_key]['local_root']}/{receipt_relative_path}",
                    "receipt_row_count": len(study_receipts),
                    "downloaded_asset_count": downloaded_asset_count,
                    "reused_cached_asset_count": reused_cached_asset_count,
                    "blocked_asset_count": count_receipt_rows(study_receipts, blocked_statuses),
                    "prepared_asset_count": count_receipt_rows(study_receipts, prepared_statuses),
                    "not_available_asset_count": count_receipt_rows(study_receipts, not_available_statuses),
                    "initial_batch_selected_flag": "yes" if study_receipts else "no",
                    "model_ready_flag": "yes"
                    if accession_row["include_flag"] == "yes" and prioritized_ready_count > 0
                    else "no",
                    "primary_local_path": choose_primary_local_path(study_receipts, ready_statuses, prioritized_asset_names),
                }
            )

    study_rows.sort(key=lambda row: (row["source_key"], row["source_accession"]))
    return study_rows


def build_manual_access_catalog(
    config: dict,
    asset_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    blocked_statuses = set(config["processed_registry"]["blocked_statuses"])
    prepared_statuses = set(config["processed_registry"]["prepared_statuses"])
    resolved_statuses = set(config["processed_registry"]["resolved_statuses"])
    reprobe_prefix = str(config["collection_completion_audit"]["reprobe_asset_prefix"])
    ignored_source_keys = set(config["collection_completion_audit"].get("ignored_source_keys", []))
    reprobe_date_by_asset_key: dict[tuple[str, str], str] = {}
    reprobe_date_by_source_key: dict[str, str] = {}
    for asset_row in asset_rows:
        asset_name = str(asset_row["asset_name"])
        if asset_name.startswith(reprobe_prefix) or asset_name == "manual_collection_gate":
            asset_key = (str(asset_row["source_key"]), str(asset_row["source_accession"]))
            reprobe_date_by_asset_key[asset_key] = max(
                str(asset_row["download_date"]),
                reprobe_date_by_asset_key.get(asset_key, ""),
            )
            current_source_key = str(asset_row["source_key"])
            reprobe_date_by_source_key[current_source_key] = max(
                str(asset_row["download_date"]),
                reprobe_date_by_source_key.get(current_source_key, ""),
            )
    nemo_metrics = nemo_candidate_metrics(config)
    manual_rows: list[dict[str, object]] = []

    for asset_row in asset_rows:
        status = str(asset_row["status"])
        if status not in blocked_statuses and status not in prepared_statuses and status not in resolved_statuses:
            continue
        workflow = manual_workflow(config, str(asset_row["source_key"]))
        requirement = requirement_for_asset(workflow, str(asset_row["source_accession"]), str(asset_row["asset_name"]))
        if status in resolved_statuses:
            resolution_status = "resolved"
        elif status in prepared_statuses:
            resolution_status = "prepared"
        else:
            resolution_status = "pending"
        note = asset_row["note"]
        if (
            str(asset_row["source_key"]) == "nemo_archive"
            and str(asset_row["asset_name"]) == "manual_collection_gate"
            and nemo_metrics["nemo_public_discovery_complete_flag"] == "yes"
            and status in blocked_statuses
        ):
            note = "NeMO public discovery complete, primary asset not yet materialized." if status in blocked_statuses else asset_row["note"]
        manual_action_required_flag = "yes" if status in blocked_statuses and str(asset_row["source_key"]) not in ignored_source_keys else "no"
        manual_rows.append(
            {
                "manual_uid": stable_uid(
                    "manual",
                    str(asset_row["source_key"]),
                    str(asset_row["source_accession"]),
                    str(asset_row["asset_name"]),
                    status,
                ),
                "source_key": asset_row["source_key"],
                "source_name": asset_row["source_name"],
                "source_accession": asset_row["source_accession"],
                "asset_name": asset_row["asset_name"],
                "status": status,
                "resolution_status": resolution_status,
                "access_mode": asset_row["access_mode"],
                "asset_url": asset_row["asset_url"],
                "local_path": asset_row["local_path"],
                "manual_action_required_flag": manual_action_required_flag,
                "manual_drop_expected_root": manual_expected_root_text(str(asset_row["source_key"]), workflow) if workflow else "",
                "required_artifact_pattern": required_pattern_for_requirement(requirement) if requirement else "",
                "last_reprobe_date": reprobe_date_by_asset_key.get(
                    (str(asset_row["source_key"]), str(asset_row["source_accession"])),
                    reprobe_date_by_source_key.get(str(asset_row["source_key"]), ""),
                ),
                "note": note,
            }
        )

    manual_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["asset_name"], row["status"]))
    return manual_rows


def build_registry_summary(
    config: dict,
    asset_rows: list[dict[str, object]],
    study_rows: list[dict[str, object]],
    asset_group_by_source: dict[str, str],
) -> list[dict[str, object]]:
    blocked_statuses = set(config["processed_registry"]["blocked_statuses"])
    prepared_statuses = set(config["processed_registry"]["prepared_statuses"])
    not_available_statuses = set(config["processed_registry"]["not_available_statuses"])
    asset_rows_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    study_rows_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)

    for asset_row in asset_rows:
        asset_rows_by_source[str(asset_row["source_key"])].append(asset_row)
    for study_row in study_rows:
        study_rows_by_source[str(study_row["source_key"])].append(study_row)

    nemo_metrics = nemo_candidate_metrics(config)
    summary_rows: list[dict[str, object]] = []
    for source_key in config["sources"]:
        source_assets = asset_rows_by_source.get(source_key, [])
        source_studies = study_rows_by_source.get(source_key, [])
        source_nemo_metrics = (
            nemo_metrics
            if source_key == "nemo_archive"
            else {
                "nemo_public_discovery_complete_flag": "no",
                "nemo_primary_open_candidate_count": 0,
                "nemo_primary_materialized_asset_count": 0,
                "nemo_reference_materialized_asset_count": 0,
            }
        )
        summary_rows.append(
            {
                "source_key": source_key,
                "asset_group": asset_group_by_source[source_key],
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
                **source_nemo_metrics,
            }
        )

    return summary_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    processed_root = resolve_repo_path(config["processed_registry"]["local_root"])
    ensure_directory(processed_root)

    asset_rows, receipt_rows_by_source, asset_group_by_source = build_asset_catalog(config)
    study_rows = build_study_catalog(config, receipt_rows_by_source)
    manual_rows = build_manual_access_catalog(config, asset_rows)
    summary_rows = build_registry_summary(config, asset_rows, study_rows, asset_group_by_source)
    pipeline_logger.progress(
        "loaded_processed_registry_inputs",
        source_count=len(config["sources"]),
        source_receipt_group_count=len(receipt_rows_by_source),
    )

    relative_paths = config["processed_registry"]["relative_paths"]
    write_tsv(
        processed_root / relative_paths["asset_catalog"],
        config["schemas"]["processed_asset_fields"],
        asset_rows,
    )
    write_tsv(
        processed_root / relative_paths["study_catalog"],
        config["schemas"]["processed_study_fields"],
        study_rows,
    )
    write_tsv(
        processed_root / relative_paths["manual_access_catalog"],
        config["schemas"]["processed_manual_access_fields"],
        manual_rows,
    )
    write_tsv(
        processed_root / relative_paths["registry_summary"],
        config["schemas"]["processed_registry_summary_fields"],
        summary_rows,
    )
    pipeline_logger.artifact(
        "processed_registry_written",
        artifact=processed_root,
        asset_row_count=len(asset_rows),
        study_row_count=len(study_rows),
        manual_row_count=len(manual_rows),
        summary_row_count=len(summary_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=processed_root,
        asset_row_count=len(asset_rows),
        study_row_count=len(study_rows),
        manual_row_count=len(manual_rows),
        summary_row_count=len(summary_rows),
    )

if __name__ == "__main__":
    main()
