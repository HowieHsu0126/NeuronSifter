from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import (
    load_config,
    normalize_text,
    read_tsv,
    resolve_repo_path,
    sha256_of_file,
)
from Libs.utils.data.manual_access_workflows import manual_checklist_path, manual_manifest_path, manual_workflow


def validate_manifest(source_root: Path, expected_fields: list[str]) -> None:
    manifest_path = source_root / "manifests" / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_fields = [field for field in expected_fields if field not in manifest]
    if missing_fields:
        raise ValueError(f"{manifest_path} is missing fields: {missing_fields}")


def validate_checksums(source_root: Path, checksum_relative_path: str) -> None:
    checksum_path = source_root / checksum_relative_path
    with checksum_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{checksum_path} has no checksum rows.")
    for row in rows:
        file_path = source_root / row["relative_path"]
        if not file_path.exists():
            raise FileNotFoundError(f"Checksum target is missing: {file_path}")
        digest = sha256_of_file(file_path)
        if digest != row["sha256"]:
            raise ValueError(f"Checksum mismatch for {file_path}")


def validate_required_subdirs(source_root: Path, required_subdirs: list[str]) -> None:
    for subdir_name in required_subdirs:
        subdir_path = source_root / subdir_name
        if not subdir_path.exists():
            raise FileNotFoundError(f"Required directory is missing: {subdir_path}")


def validate_accession_table(source_root: Path, expected_fields: list[str]) -> None:
    accession_path = source_root / "accession_freeze" / "accessions.tsv"
    with accession_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{accession_path} header mismatch: {header}")


def validate_literature_parameter_bank(source_root: Path, expected_fields: list[str]) -> None:
    parameter_table = source_root / "downloads" / "parameter_records.tsv"
    with parameter_table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{parameter_table} header mismatch: {header}")


def validate_receipt_table(source_root: Path, receipt_relative_path: str, expected_fields: list[str]) -> None:
    receipt_path = source_root / receipt_relative_path
    with receipt_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{receipt_path} header mismatch: {header}")


def validate_nemo_outputs(config: dict, source_root: Path) -> None:
    reprobe_config = config["reprobe_public_sources"]["sources"]["nemo_archive"]
    expected_tables = [
        (source_root / reprobe_config["discovered_collections_relative_path"], config["schemas"]["nemo_discovered_collection_fields"]),
        (source_root / reprobe_config["primary_collections_relative_path"], config["schemas"]["nemo_ad_aging_collection_fields"]),
        (source_root / reprobe_config["reference_collections_relative_path"], config["schemas"]["nemo_ad_aging_collection_fields"]),
        (source_root / reprobe_config["primary_candidate_queue_relative_path"], config["schemas"]["nemo_candidate_queue_fields"]),
        (source_root / reprobe_config["reference_candidate_queue_relative_path"], config["schemas"]["nemo_candidate_queue_fields"]),
        (source_root / reprobe_config["candidate_sync_audit_relative_path"], config["schemas"]["nemo_candidate_sync_audit_fields"]),
    ]
    for table_path, expected_fields in expected_tables:
        if not table_path.exists():
            continue
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
        if header != expected_fields:
            raise ValueError(f"{table_path} header mismatch: {header}")


def validate_manual_note(source_root: Path, manual_note_relative_path: str) -> None:
    manual_note_path = source_root / manual_note_relative_path
    if not manual_note_path.exists():
        raise FileNotFoundError(f"Manual note is missing: {manual_note_path}")


def validate_manual_workflow_artifacts(config: dict, source_key: str, source_root: Path) -> None:
    workflow = manual_workflow(config, source_key)
    if workflow is None:
        return
    manifest_path = manual_manifest_path(source_root, workflow)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manual drop manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != config["schemas"]["manual_drop_manifest_fields"]:
        raise ValueError(f"{manifest_path} header mismatch: {header}")
    checklist_path = manual_checklist_path(source_root, workflow)
    if not checklist_path.exists():
        raise FileNotFoundError(f"Manual completion checklist is missing: {checklist_path}")


def validate_no_partial_files(source_root: Path) -> None:
    partial_files = sorted(path for path in source_root.rglob("*.part") if path.is_file())
    if partial_files:
        raise ValueError(f"Incomplete partial files remain under {source_root}: {partial_files}")


def validate_scope_rules(config: dict) -> None:
    geo_rows = read_tsv(resolve_repo_path(config["sources"]["geo"]["local_root"]) / "accession_freeze" / "accessions.tsv")
    arrayexpress_rows = read_tsv(resolve_repo_path(config["sources"]["arrayexpress"]["local_root"]) / "accession_freeze" / "accessions.tsv")
    excluded_keywords = [normalize_text(term) for term in config["classification"]["excluded_disease_keywords"]]
    for row in geo_rows + arrayexpress_rows:
        diagnosis_label = normalize_text(row["diagnosis_label"])
        if row["include_flag"] == "yes" and any(term in diagnosis_label for term in excluded_keywords):
            raise ValueError(f"Out-of-scope diagnosis label retained: {row}")
    geo_accessions = {row["source_accession"] for row in geo_rows if row["include_flag"] == "yes"}
    for row in arrayexpress_rows:
        if not row["source_accession"].startswith("E-GEOD-"):
            continue
        geo_candidate = f"GSE{row['source_accession'].removeprefix('E-GEOD-')}"
        if geo_candidate in geo_accessions and row["include_flag"] == "yes":
            raise ValueError(f"GEO mirror still included in ArrayExpress freeze: {row['source_accession']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the raw data registry layout and checksum files.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the public raw data registry.",
    )
    parser.add_argument(
        "--require-empty-processed",
        action="store_true",
        help="Require Input/processed to remain empty. Use this only before processed registry generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    pipeline_logger.progress(
        "loaded_raw_inputs",
        source_count=len(config["sources"]),
        public_download_source_count=len(config.get("public_downloads", {})),
        require_empty_processed=bool(args.require_empty_processed),
    )
    if args.require_empty_processed:
        processed_root = resolve_repo_path(config["project"]["processed_root"])
        processed_files = [path for path in processed_root.rglob("*") if path.is_file()]
        if processed_files:
            raise ValueError(f"Input/processed must remain empty during startup collection: {processed_files}")

    for source_key, source_config in config["sources"].items():
        source_root = resolve_repo_path(source_config["local_root"])
        validate_required_subdirs(source_root, source_config["required_subdirs"])
        validate_no_partial_files(source_root)
        validate_manifest(source_root, config["schemas"]["source_manifest_fields"])
        validate_checksums(source_root, source_config["checksum_file"])
        validate_manual_workflow_artifacts(config, source_key, source_root)
        public_download_config = config.get("public_downloads", {}).get(source_key)
        if public_download_config is not None:
            validate_receipt_table(
                source_root,
                public_download_config["receipt_relative_path"],
                config["schemas"]["download_receipt_fields"],
            )
            if "manual_note_relative_path" in public_download_config:
                validate_manual_note(source_root, public_download_config["manual_note_relative_path"])
        if "accession_freeze" in source_config["required_subdirs"]:
            validate_accession_table(source_root, config["schemas"]["accession_fields"])
        if source_key == "literature_parameter_bank":
            validate_literature_parameter_bank(source_root, config["schemas"]["literature_parameter_fields"])
        if source_key == "nemo_archive":
            validate_nemo_outputs(config, source_root)

    validate_scope_rules(config)

    pipeline_logger.progress(
        "raw_registry_validated",
        source_count=len(config["sources"]),
        public_download_source_count=len(config.get("public_downloads", {})),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        source_count=len(config["sources"]),
        public_download_source_count=len(config.get("public_downloads", {})),
    )

if __name__ == "__main__":
    main()
