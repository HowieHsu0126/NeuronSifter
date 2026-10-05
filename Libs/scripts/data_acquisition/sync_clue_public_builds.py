from __future__ import annotations

import argparse
from datetime import date
import json
import os
import subprocess
from pathlib import Path
import sys

import requests
from requests import RequestException

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import download_to_file, ensure_directory, load_config, refresh_checksums, resolve_repo_path, write_json, write_tsv
from Libs.utils.data.raw_receipts import build_receipt_row, load_receipt_rows, upsert_receipt_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CLUE public build metadata and selected core matrices into Input/raw/clue_lincs.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--build-names",
        default="",
        help="Optional comma-separated build names subset for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected builds and files without downloading.",
    )
    return parser.parse_args()


def selected_build_names(build_name_text: str) -> set[str]:
    return {value.strip() for value in build_name_text.split(",") if value.strip()}


def clue_api_key(config: dict) -> str:
    env_name = str(config["clue_public_build_sync"]["api_key_env"])
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise ValueError(f"Missing required CLUE API key in environment variable {env_name}.")
    return api_key


def clue_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"user_key": api_key, "Accept": "application/json"})
    return session


def role_names(build_payload: dict) -> list[str]:
    return [str(role.get("name")) for role in build_payload.get("role", []) if role.get("name")]


def is_selected_build(build_payload: dict, config: dict, selected_subset: set[str]) -> bool:
    if selected_subset and str(build_payload.get("name")) not in selected_subset:
        return False
    if str(build_payload.get("status")) not in set(config["clue_public_build_sync"]["selected_statuses"]):
        return False
    if not any(role in set(config["clue_public_build_sync"]["selected_roles"]) for role in role_names(build_payload)):
        return False
    return True


def matches_metadata_pattern(file_name: str, patterns: list[str]) -> bool:
    lowered_name = file_name.lower()
    return any(pattern.lower() in lowered_name for pattern in patterns)


def build_file_rows(build_payload: dict, config: dict) -> list[dict[str, str]]:
    patterns = list(config["clue_public_build_sync"]["metadata_name_patterns"])
    build_name = str(build_payload["name"])
    selected_names = set(config["clue_public_build_sync"]["core_build_files"].get(build_name, []))
    rows: list[dict[str, str]] = []
    for file_payload in build_payload.get("s3Folder", {}).get("downloads", []):
        file_name = str(file_payload.get("name", "")).strip()
        file_url = str(file_payload.get("s3_location", "")).strip()
        if not file_name or not file_url:
            continue
        if matches_metadata_pattern(file_name, patterns) or file_name in selected_names:
            rows.append(
                {
                    "file_name": file_name,
                    "file_url": file_url,
                    "file_size_text": str(file_payload.get("size", "")),
                    "description": str(file_payload.get("description", "")),
                }
            )
    rows.sort(key=lambda row: row["file_name"])
    return rows


def build_root(source_root: Path, build_name: str, sync_config: dict) -> Path:
    relative_path = str(sync_config["build_root_template"]).format(build_name=build_name)
    return source_root / relative_path


def download_file(url: str, destination: Path) -> tuple[str, str]:
    ensure_directory(destination.parent)
    if destination.exists():
        return "reused_cached", ""
    try:
        download_to_file(url, destination)
    except RequestException as exc:
        return "not_available", f"{type(exc).__name__}: {exc}"
    return "downloaded", ""


def refresh_processed_catalogs(config_path: str) -> None:
    subprocess.run([sys.executable, "Libs/scripts/builders/build_processed_registry.py", "--config", config_path], cwd=REPO_ROOT, check=True)
    subprocess.run([sys.executable, "Libs/scripts/reports/audit_collection_completion.py", "--config", config_path], cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    sync_config = config["clue_public_build_sync"]
    source_root = resolve_repo_path(sync_config["local_root"])
    receipt_path = source_root / sync_config["receipt_relative_path"]
    selected_subset = selected_build_names(args.build_names)
    api_key = clue_api_key(config)
    session = clue_session(api_key)

    response = session.get(sync_config["api_base_url"], timeout=60)
    response.raise_for_status()
    catalog_payload = response.json()
    selected_builds = [payload for payload in catalog_payload if is_selected_build(payload, config, selected_subset)]
    selected_builds.sort(key=lambda payload: str(payload["name"]))

    summary_rows: list[dict[str, object]] = []
    if args.dry_run:
        for build_payload in selected_builds:
            summary_rows.append(
                {
                    "build_name": str(build_payload["name"]),
                    "roles": role_names(build_payload),
                    "selected_files": [row["file_name"] for row in build_file_rows(build_payload, config)],
                    "download_count": len(build_file_rows(build_payload, config)),
                }
            )
        complete_script_logging(
            pipeline_logger,
            pipeline_started_at,
            dry_run=True,
            selected_build_count=len(selected_builds),
        )
        print(json.dumps(summary_rows, indent=2, ensure_ascii=True))
        return

    ensure_directory(source_root / "downloads" / "api")
    catalog_path = source_root / sync_config["catalog_relative_path"]
    catalog_preexisting = catalog_path.exists()
    write_json(catalog_path, catalog_payload)

    new_receipt_rows: list[dict[str, object]] = [
        build_receipt_row(
            "clue_lincs",
            "clue_build_catalog_json",
            sync_config["api_base_url"],
            catalog_path,
            "reused_cached" if catalog_preexisting else "downloaded",
            "Full CLUE build catalog snapshot.",
        ),
    ]

    existing_receipt_rows = load_receipt_rows(receipt_path)
    inaccessible_file_count = 0
    selected_file_count = 0
    for build_payload in selected_builds:
        build_name = str(build_payload["name"])
        build_root_path = build_root(source_root, build_name, sync_config)
        ensure_directory(build_root_path / "files")
        detail_path = build_root_path / "build_detail.json"
        detail_preexisting = detail_path.exists()
        write_json(detail_path, build_payload)
        new_receipt_rows.append(
            build_receipt_row(
                "clue_lincs",
                f"clue_build_detail_json::{build_name}",
                sync_config["api_base_url"],
                detail_path,
                "reused_cached" if detail_preexisting else "downloaded",
                f"CLUE build detail for {build_name}.",
            )
        )
        for file_row in build_file_rows(build_payload, config):
            selected_file_count += 1
            destination = build_root_path / "files" / file_row["file_name"]
            status, error_text = download_file(file_row["file_url"], destination)
            if status == "not_available":
                inaccessible_file_count += 1
            new_receipt_rows.append(
                build_receipt_row(
                    "clue_lincs",
                    file_row["file_name"],
                    file_row["file_url"],
                    destination if status != "not_available" else None,
                    status,
                    (
                        f"{build_name}: {file_row['description'] or file_row['file_size_text']}"
                        if status != "not_available"
                        else f"{build_name}: unavailable via direct public URL. {error_text}"
                    ),
                )
            )
    gate_row = None
    for row in existing_receipt_rows:
        if row["source_accession"] == "clue_lincs" and row["asset_name"] == "manual_collection_gate":
            gate_row = dict(row)
            break
    if gate_row is None:
        gate_row = build_receipt_row(
            "clue_lincs",
            "manual_collection_gate",
            config["sources"]["clue_lincs"]["source_url"],
            None,
            "resolved_public_accessible",
            "CLUE public build metadata and selected core matrices were synchronized with the provided API key.",
        )
    gate_row["status"] = "resolved_public_accessible"
    if inaccessible_file_count == 0:
        gate_note = "CLUE public build metadata and selected core matrices were synchronized with the provided API key."
    else:
        gate_note = (
            f"CLUE maximum reachable public subset was synchronized; {inaccessible_file_count} of "
            f"{selected_file_count} selected files remain unavailable via current account/API access and were marked not_available."
        )
    gate_row["download_date"] = date.today().isoformat()
    gate_row["note"] = gate_note
    gate_row["local_path"] = ""
    gate_row["size_bytes"] = 0
    new_receipt_rows.append(gate_row)

    merged_receipt_rows = upsert_receipt_rows(existing_receipt_rows, new_receipt_rows, config["schemas"]["download_receipt_fields"])
    write_tsv(receipt_path, config["schemas"]["download_receipt_fields"], merged_receipt_rows)
    refresh_checksums(source_root, config["sources"]["clue_lincs"]["checksum_file"])
    refresh_processed_catalogs(args.config)

    print(
        json.dumps(
            {
                "selected_build_count": len(selected_builds),
                "selected_build_names": [str(build_payload["name"]) for build_payload in selected_builds],
                "selected_file_count": selected_file_count,
                "inaccessible_file_count": inaccessible_file_count,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=receipt_path,
        selected_build_count=len(selected_builds),
        selected_file_count=selected_file_count,
        inaccessible_file_count=inaccessible_file_count,
    )


if __name__ == "__main__":
    main()
