from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import (
    ensure_directory,
    load_config,
    refresh_checksums,
    resolve_repo_path,
    today_iso,
    write_json,
    write_tsv,
)
from Libs.utils.data.manual_access_workflows import ensure_manual_workflow_artifacts


def build_source_manifest(source_key: str, source_config: dict) -> dict:
    return {
        "source_name": source_config["source_name"],
        "source_url": source_config["source_url"],
        "release_label": source_config["release_label"],
        "release_date": source_config["release_date"],
        "download_date": today_iso(),
        "access_mode": source_config["access_mode"],
        "license_or_terms": source_config["license_or_terms"],
        "disease_scope": source_config["disease_scope"],
        "modality_scope": source_config["modality_scope"],
        "query_scope": source_config["query_scope"],
        "local_root": source_config["local_root"],
        "checksum_file": source_config["checksum_file"],
        "notes": source_config["notes"],
        "source_key": source_key,
    }


def bootstrap_source(source_key: str, source_config: dict, config: dict) -> None:
    schemas = config["schemas"]
    source_root = resolve_repo_path(source_config["local_root"])
    ensure_directory(source_root)
    for subdir_name in source_config["required_subdirs"]:
        ensure_directory(source_root / subdir_name)

    manifest_path = source_root / "manifests" / "source_manifest.json"
    write_json(manifest_path, build_source_manifest(source_key, source_config))

    collection_policy_path = source_root / "manifests" / "collection_policy.json"
    write_json(
        collection_policy_path,
        {
            "include_targets": source_config.get("include_targets", []),
            "exclude_targets": source_config.get("exclude_targets", []),
        },
    )

    if "accession_freeze" in source_config["required_subdirs"]:
        freeze_path = source_root / "accession_freeze" / "accessions.tsv"
        if not freeze_path.exists():
            write_tsv(
                freeze_path,
                schemas["accession_fields"],
                [{key: str(value) for key, value in row.items()} for row in source_config.get("catalog_rows", [])],
            )

    if source_key == "oasis":
        tabular_scope_path = source_root / "tabular" / "download_inventory.tsv"
        imaging_scope_path = source_root / "imaging_optional" / "session_inventory.tsv"
        tabular_rows = [
            {
                "asset_family": "clinical_tables",
                "source_release": "OASIS-3",
                "modality": "clinical",
                "priority": "primary",
                "public_url": source_config["source_url"],
                "local_target": "Input/raw/oasis/tabular",
                "status": "planned",
                "notes": "Tabular-first startup queue.",
            },
            {
                "asset_family": "cognitive_tables",
                "source_release": "OASIS-3",
                "modality": "cognitive",
                "priority": "primary",
                "public_url": source_config["source_url"],
                "local_target": "Input/raw/oasis/tabular",
                "status": "planned",
                "notes": "Tabular-first startup queue.",
            },
            {
                "asset_family": "biomarker_tables",
                "source_release": "OASIS-3",
                "modality": "biomarker",
                "priority": "primary",
                "public_url": source_config["source_url"],
                "local_target": "Input/raw/oasis/tabular",
                "status": "planned",
                "notes": "Tabular-first startup queue.",
            },
            {
                "asset_family": "tabular_metadata",
                "source_release": "OASIS-4",
                "modality": "metadata",
                "priority": "secondary",
                "public_url": source_config["source_url"],
                "local_target": "Input/raw/oasis/tabular",
                "status": "planned",
                "notes": "Secondary until non-imaging completion.",
            },
        ]
        if not tabular_scope_path.exists():
            write_tsv(tabular_scope_path, schemas["oasis_tabular_inventory_fields"], tabular_rows)
        if not imaging_scope_path.exists():
            write_tsv(imaging_scope_path, schemas["oasis_imaging_inventory_fields"], [])

    if source_key == "literature_parameter_bank":
        parameter_table_path = source_root / "downloads" / "parameter_records.tsv"
        if not parameter_table_path.exists():
            write_tsv(parameter_table_path, schemas["literature_parameter_fields"], [])
        required_categories_path = source_root / "manifests" / "required_parameter_categories.json"
        if not required_categories_path.exists():
            write_json(
                required_categories_path,
                {
                    "required_categories": source_config["required_categories"],
                    "capture_policy": source_config["notes"],
                },
            )
        term_policy_path = source_root / "terms" / "license_handling_policy.md"
        if not term_policy_path.exists():
            term_policy_path.write_text(
                "\n".join(
                    [
                        "# Literature Parameter Capture Policy",
                        "",
                        "- Store open-access PDF or HTML snapshots in `downloads/`.",
                        "- Store citation metadata for restricted texts without copying the full text.",
                        "- Reject parameter entries that lack units, compartment context, or source traceability.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    ensure_manual_workflow_artifacts(source_key, source_config, config)
    refresh_checksums(source_root, source_config["checksum_file"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the raw data directory layout and manifest files.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the public raw data registry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    raw_root = resolve_repo_path(config["project"]["raw_root"])
    ensure_directory(raw_root)
    ensure_directory(resolve_repo_path(config["project"]["processed_root"]))
    for source_key, source_config in config["sources"].items():
        bootstrap_source(source_key, source_config, config)

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
