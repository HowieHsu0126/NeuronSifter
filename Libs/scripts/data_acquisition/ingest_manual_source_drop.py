from __future__ import annotations

import argparse
import csv
from datetime import date
import fnmatch
import subprocess
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.manual_access_workflows import (
    manual_drop_root,
    manual_manifest_path,
    manual_workflow,
    requirement_for_asset,
)
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, refresh_checksums, resolve_repo_path, write_tsv
from Libs.utils.data.raw_receipts import build_receipt_row, load_receipt_rows, upsert_receipt_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest user-provided manual-drop files into Input/raw and refresh manual-access state.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--source-key",
        required=True,
        help="Manual source key to ingest from downloads/manual_drop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manual manifest and dropped files without mutating receipts or processed catalogs.",
    )
    return parser.parse_args()


def matching_requirement_pattern(required_pattern: str, relative_path: str) -> bool:
    patterns = [item.strip() for item in required_pattern.split("|") if item.strip()]
    if not patterns:
        return True
    file_name = Path(relative_path).name
    return any(fnmatch.fnmatch(file_name, pattern) for pattern in patterns)


def validate_manifest_rows(
    config: dict,
    source_key: str,
    workflow: dict,
    source_root: Path,
) -> list[dict[str, str]]:
    manifest_path = manual_manifest_path(source_root, workflow)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    if header != config["schemas"]["manual_drop_manifest_fields"]:
        raise ValueError(f"Manual drop manifest header mismatch: {manifest_path}")
    rows = read_tsv(manifest_path)
    if not rows:
        raise ValueError(f"Manual drop manifest has no rows: {manifest_path}")

    drop_root = manual_drop_root(source_root, workflow)
    unresolved_requirements: list[str] = []
    for row in rows:
        if not row["source_accession"] or not row["asset_name"]:
            raise ValueError(f"Manual drop row is missing source_accession or asset_name: {row}")
        requirement = requirement_for_asset(workflow, row["source_accession"], row["asset_name"])
        if requirement is None:
            raise ValueError(f"Manual drop row does not match any configured requirement: {row}")
        if not row["relative_path"]:
            unresolved_requirements.append(f"{row['source_accession']}::{row['asset_name']}")
            continue
        if row["license_or_terms_acknowledged_flag"] != "yes":
            raise ValueError(f"Manual drop row lacks license acknowledgement: {row}")
        required_pattern = str(requirement.get("required_artifact_pattern", ""))
        if not matching_requirement_pattern(required_pattern, row["relative_path"]):
            raise ValueError(f"Manual drop file path does not match required pattern {required_pattern}: {row}")
        local_path = drop_root / row["relative_path"]
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(f"Manual drop file is missing: {local_path}")
    if unresolved_requirements:
        raise ValueError(
            "Manual drop manifest still has unresolved template rows: "
            + ", ".join(unresolved_requirements)
        )
    return rows


def resolve_gate_row(
    receipt_rows: list[dict[str, str]],
    source_accession: str,
    asset_name: str,
    source_url: str,
) -> dict[str, object]:
    for row in receipt_rows:
        if row["source_accession"] == source_accession and row["asset_name"] == asset_name and row["status"].startswith(("blocked_", "deferred_", "resolved_")):
            return {
                "source_accession": row["source_accession"],
                "asset_name": row["asset_name"],
                "asset_url": row["asset_url"],
                "local_path": row["local_path"],
                "download_date": row["download_date"],
                "status": row["status"],
                "size_bytes": row["size_bytes"],
                "note": row["note"],
            }
    return {
        "source_accession": source_accession,
        "asset_name": asset_name,
        "asset_url": source_url,
        "local_path": "",
        "download_date": "",
        "status": "",
        "size_bytes": 0,
        "note": "",
    }


def refresh_processed_catalogs(config_path: str) -> None:
    subprocess.run([sys.executable, "Libs/scripts/builders/build_processed_registry.py", "--config", config_path], cwd=REPO_ROOT, check=True)
    subprocess.run([sys.executable, "Libs/scripts/reports/audit_collection_completion.py", "--config", config_path], cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    source_key = args.source_key
    if source_key not in config["sources"]:
        raise ValueError(f"Unknown source_key: {source_key}")
    workflow = manual_workflow(config, source_key)
    if workflow is None:
        raise ValueError(f"Source {source_key} is not configured for manual-drop ingestion.")

    source_config = config["sources"][source_key]
    source_root = resolve_repo_path(source_config["local_root"])
    ensure_directory(source_root)
    receipt_path = source_root / config["public_downloads"][source_key]["receipt_relative_path"]
    existing_receipt_rows = load_receipt_rows(receipt_path)
    manifest_rows = validate_manifest_rows(config, source_key, workflow, source_root)

    new_receipt_rows: list[dict[str, object]] = []
    resolved_gate_keys: set[tuple[str, str]] = set()
    drop_root = manual_drop_root(source_root, workflow)
    for row in manifest_rows:
        local_path = drop_root / row["relative_path"]
        asset_url = row["asset_url"] or source_config["source_url"]
        new_receipt_rows.append(
            build_receipt_row(
                row["source_accession"],
                row["asset_name"],
                asset_url,
                local_path,
                "downloaded",
                row["note"] or "Ingested from manual_drop manifest.",
            )
        )
        resolved_gate_keys.add((row["source_accession"], row["asset_name"]))

    for source_accession, asset_name in sorted(resolved_gate_keys):
        gate_row = resolve_gate_row(existing_receipt_rows, source_accession, asset_name, source_config["source_url"])
        gate_row["status"] = str(config["manual_source_workflows"]["shared"]["resolved_status"])
        gate_row["download_date"] = date.today().isoformat()
        gate_row["note"] = "Resolved by ingest_manual_source_drop.py."
        gate_row["size_bytes"] = 0
        gate_row["local_path"] = ""
        new_receipt_rows.append(gate_row)

    if args.dry_run:
        print({"source_key": source_key, "manifest_row_count": len(manifest_rows), "resolved_gate_count": len(resolved_gate_keys), "dry_run": True})
        complete_script_logging(
            pipeline_logger,
            pipeline_started_at,
            source=source_key,
            dry_run=True,
            manifest_row_count=len(manifest_rows),
            resolved_gate_count=len(resolved_gate_keys),
        )
        return

    merged_receipt_rows = upsert_receipt_rows(existing_receipt_rows, new_receipt_rows, config["schemas"]["download_receipt_fields"])
    write_tsv(receipt_path, config["schemas"]["download_receipt_fields"], merged_receipt_rows)
    refresh_checksums(source_root, source_config["checksum_file"])
    refresh_processed_catalogs(args.config)
    print({"source_key": source_key, "manifest_row_count": len(manifest_rows), "resolved_gate_count": len(resolved_gate_keys), "dry_run": False})
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        source=source_key,
        manifest_row_count=len(manifest_rows),
        resolved_gate_count=len(resolved_gate_keys),
    )


if __name__ == "__main__":
    main()
