from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import date
from pathlib import Path
import sys
from urllib.parse import urljoin, urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.scripts.data_acquisition.reprobe_blocked_public_sources import fetch_page_resilient, response_note
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, refresh_checksums, resolve_repo_path, write_text_atomic, write_tsv
from Libs.utils.data.raw_receipts import build_receipt_row, load_receipt_rows, upsert_receipt_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize NeMO public processed candidates and discovery evidence under Input/raw/nemo_archive.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--queue-scope",
        choices=["all", "primary", "reference", "all_public_processed"],
        default="all",
        help="Candidate queue subset to synchronize.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report candidate actions without downloading files.",
    )
    return parser.parse_args()


def refresh_processed_catalogs(config_path: str) -> None:
    subprocess.run([sys.executable, "Libs/scripts/builders/build_processed_registry.py", "--config", config_path], cwd=REPO_ROOT, check=True)
    subprocess.run([sys.executable, "Libs/scripts/reports/audit_collection_completion.py", "--config", config_path], cwd=REPO_ROOT, check=True)


def candidate_queue_paths(source_root: Path, reprobe_config: dict, queue_scope: str) -> list[Path]:
    queue_paths: list[Path] = []
    if queue_scope == "all_public_processed":
        return queue_paths
    if queue_scope in {"all", "primary"}:
        queue_paths.append(source_root / reprobe_config["primary_candidate_queue_relative_path"])
    if queue_scope in {"all", "reference"}:
        queue_paths.append(source_root / reprobe_config["reference_candidate_queue_relative_path"])
    return queue_paths


def load_queue_rows(queue_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_collection_ids: set[str] = set()
    for queue_path in queue_paths:
        if not queue_path.exists():
            continue
        queue_rows = read_tsv(queue_path)
        for row in queue_rows:
            collection_id = str(row["collection_id"])
            if collection_id in seen_collection_ids:
                continue
            seen_collection_ids.add(collection_id)
            rows.append(row)
    rows.sort(key=lambda row: (str(row["match_scope"]), str(row["collection_id"])))
    return rows


def load_public_processed_rows(source_root: Path, reprobe_config: dict, config: dict) -> list[dict[str, str]]:
    discovered_path = source_root / reprobe_config["discovered_collections_relative_path"]
    existing_queue_paths = [
        source_root / reprobe_config["primary_candidate_queue_relative_path"],
        source_root / reprobe_config["reference_candidate_queue_relative_path"],
    ]
    queued_collection_ids: set[str] = set()
    for queue_path in existing_queue_paths:
        if not queue_path.exists():
            continue
        queued_collection_ids.update(row["collection_id"] for row in read_tsv(queue_path))
    rows: list[dict[str, str]] = []
    for row in read_tsv(discovered_path):
        collection_id = row["collection_id"]
        if collection_id in queued_collection_ids:
            continue
        if str(row["access"]).lower() != "open":
            continue
        if int(row["processed_manifest_count"] or 0) <= 0:
            continue
        rows.append(
            {
                "collection_id": collection_id,
                "short_name": row["short_name"],
                "name": row["name"],
                "description": row["description"],
                "access": row["access"],
                "taxa": row["taxa"],
                "keywords": row["keywords"],
                "modalities": row["modalities"],
                "anatomical_regions": row["anatomical_regions"],
                "collection_type": row["collection_type"],
                "publication_title": row["publication_title"],
                "publication_doi": row["publication_doi"],
                "manifest_count": row["manifest_count"],
                "manifest_urls": row["manifest_urls"],
                "manifest_types": row["manifest_types"],
                "manifest_protocol_types": row["manifest_protocols"],
                "processed_manifest_count": row["processed_manifest_count"],
                "raw_manifest_count": row["raw_manifest_count"],
                "match_scope": "nemo_public_processed",
                "match_fields": "manifest_types",
                "keyword_match_terms": "processed",
                "eligibility_bucket": "open_processed_candidate",
                "local_candidate_root": f"Input/raw/nemo_archive/downloads/reprobe/public_processed_candidates/{collection_id.replace(':', '_')}",
                "note": "open_public_processed_manifest",
            }
        )
    rows.sort(key=lambda row: str(row["collection_id"]))
    queue_path = source_root / reprobe_config["public_processed_candidate_queue_relative_path"]
    write_tsv(queue_path, config["schemas"]["nemo_candidate_queue_fields"], rows)
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def merged_audit_rows(existing_rows: list[dict[str, str]], new_rows: list[dict[str, object]], fieldnames: list[str]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for row in existing_rows:
        merged[(row["collection_id"], row["match_scope"])] = dict(row)
    for row in new_rows:
        merged[(str(row["collection_id"]), str(row["match_scope"]))] = dict(row)
    ordered_rows = list(merged.values())
    ordered_rows.sort(key=lambda row: (str(row["match_scope"]), str(row["collection_id"])))
    return [{field: row.get(field, "") for field in fieldnames} for row in ordered_rows]


def collection_slug(collection_id: str) -> str:
    return collection_id.replace(":", "_")


def collection_metadata_path(source_root: Path, collection_id: str) -> Path:
    return source_root / "downloads" / "reprobe" / "collections" / collection_slug(collection_id) / "metadata.json"


def collection_landing_path(source_root: Path, collection_id: str) -> Path:
    return source_root / "downloads" / "reprobe" / "collections" / collection_slug(collection_id) / "landing.html"


def write_downloaded_text(url: str, destination: Path, response_text: str) -> str:
    ensure_directory(destination.parent)
    if destination.exists():
        return "reused_cached"
    write_text_atomic(destination, response_text)
    return "downloaded"


def write_downloaded_binary(url: str, destination: Path) -> str:
    ensure_directory(destination.parent)
    if destination.exists():
        return "reused_cached"
    temporary_path = destination.parent / f"{destination.name}.part"
    subprocess.run(["curl", "-L", "--fail", "--silent", "--show-error", "--output", str(temporary_path), url], cwd=REPO_ROOT, check=True)
    temporary_path.replace(destination)
    return "downloaded"


def rewritten_url(url: str, reprobe_config: dict) -> str:
    current_url = url
    for source_prefix, target_prefix in dict(reprobe_config.get("remote_url_rewrites", {})).items():
        if current_url.startswith(source_prefix):
            return target_prefix + current_url.removeprefix(source_prefix)
    return current_url


def url_has_blocked_term(url: str, blocked_terms: list[str]) -> bool:
    lowered_url = url.lower()
    return any(term.lower() in lowered_url for term in blocked_terms)


def is_listing_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path.endswith("/") or "." not in Path(parsed.path).name


def listing_metadata_links(base_url: str, response_text: str, allowed_extensions: list[str], name_patterns: list[str]) -> list[str]:
    discovered_links: list[str] = []
    for match in re.findall(r'href=["\\\']([^"\\\']+)["\\\']', response_text):
        absolute_url = urljoin(base_url, match)
        parsed = urlparse(absolute_url)
        basename = Path(parsed.path).name.lower()
        if not basename or basename == "parent directory":
            continue
        if any(basename.endswith(extension.lower()) for extension in allowed_extensions) or any(pattern.lower() in basename for pattern in name_patterns):
            if absolute_url not in discovered_links:
                discovered_links.append(absolute_url)
    return discovered_links


def candidate_root(row: dict[str, str]) -> Path:
    return resolve_repo_path(row["local_candidate_root"])


def manifest_rows_for_collection(source_root: Path, collection_id: str) -> list[dict[str, object]]:
    metadata_path = collection_metadata_path(source_root, collection_id)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return [item for item in (payload.get("manifest_file_urls") or []) if item.get("url")]


def sync_candidate(
    source_root: Path,
    row: dict[str, str],
    reprobe_config: dict,
    dry_run: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    collection_id = str(row["collection_id"])
    manifest_rows = manifest_rows_for_collection(source_root, collection_id)
    candidate_root_path = candidate_root(row)
    metadata_path = collection_metadata_path(source_root, collection_id)
    landing_path = collection_landing_path(source_root, collection_id)
    receipt_rows: list[dict[str, object]] = []
    downloaded_asset_count = 0
    reused_cached_asset_count = 0
    materialized_asset_count = 0
    discovery_evidence_count = 0
    allowed_manifest_url_count = 0
    rejected_raw_manifest_url_count = 0
    blocked_terms = list(reprobe_config["blocked_download_url_terms"])
    metadata_extensions = [str(value).lower() for value in reprobe_config["metadata_like_extensions"]]
    listing_patterns = [str(value).lower() for value in reprobe_config["listing_metadata_name_patterns"]]

    if metadata_path.exists():
        discovery_evidence_count += 1
    if landing_path.exists():
        discovery_evidence_count += 1

    for manifest_index, manifest_row in enumerate(manifest_rows, start=1):
        manifest_url = rewritten_url(str(manifest_row.get("url", "")).strip(), reprobe_config)
        if not manifest_url:
            continue
        manifest_type = str(manifest_row.get("type", "")).strip().lower()
        listing_root = candidate_root_path / "listings" / f"manifest_{manifest_index:02d}"
        if is_listing_url(manifest_url):
            if dry_run:
                allowed_manifest_url_count += 1
                continue
            listing_status_code, listing_text, listing_error = fetch_page_resilient(manifest_url)
            listing_index_path = listing_root / "index.html"
            if listing_status_code == 200 and listing_text:
                listing_status = write_downloaded_text(manifest_url, listing_index_path, listing_text)
                receipt_rows.append(
                    build_receipt_row(
                        collection_id,
                        f"nemo_candidate_listing_index::{collection_id}::{manifest_index}",
                        manifest_url,
                        listing_index_path,
                        listing_status,
                        f"{row['match_scope']} listing evidence for {collection_id}.",
                    )
                )
                if listing_status == "downloaded":
                    downloaded_asset_count += 1
                if listing_status == "reused_cached":
                    reused_cached_asset_count += 1
                discovery_evidence_count += 1
                allowed_manifest_url_count += 1
                if manifest_type != "raw":
                    for metadata_index, metadata_url in enumerate(listing_metadata_links(manifest_url, listing_text, metadata_extensions, listing_patterns), start=1):
                        metadata_url = rewritten_url(metadata_url, reprobe_config)
                        metadata_name = Path(urlparse(metadata_url).path).name or f"metadata_{metadata_index:02d}"
                        metadata_path = listing_root / metadata_name
                        metadata_status_code, metadata_text, metadata_error = fetch_page_resilient(metadata_url)
                        if metadata_status_code != 200 or not metadata_text:
                            receipt_rows.append(
                                build_receipt_row(
                                    collection_id,
                                    f"nemo_candidate_listing_metadata::{collection_id}::{manifest_index}::{metadata_index}",
                                    metadata_url,
                                    None,
                                    "not_available",
                                    response_note(metadata_url, metadata_status_code, metadata_error),
                                )
                            )
                            continue
                        metadata_status = write_downloaded_text(metadata_url, metadata_path, metadata_text)
                        receipt_rows.append(
                            build_receipt_row(
                                collection_id,
                                f"nemo_candidate_listing_metadata::{collection_id}::{manifest_index}::{metadata_index}",
                                metadata_url,
                                metadata_path,
                                metadata_status,
                                f"{row['match_scope']} metadata-like listing asset for {collection_id}.",
                            )
                        )
                        if metadata_status == "downloaded":
                            downloaded_asset_count += 1
                        if metadata_status == "reused_cached":
                            reused_cached_asset_count += 1
                        materialized_asset_count += 1
            else:
                receipt_rows.append(
                    build_receipt_row(
                        collection_id,
                        f"nemo_candidate_listing_index::{collection_id}::{manifest_index}",
                        manifest_url,
                        None,
                        "not_available",
                        response_note(manifest_url, listing_status_code, listing_error),
                    )
                )
            continue

        if manifest_type == "raw" or url_has_blocked_term(manifest_url, blocked_terms):
            rejected_raw_manifest_url_count += 1
            continue

        allowed_manifest_url_count += 1
        if dry_run:
            continue
        file_name = Path(urlparse(manifest_url).path).name or f"asset_{manifest_index:02d}"
        destination = candidate_root_path / "files" / file_name
        manifest_status = write_downloaded_binary(manifest_url, destination)
        receipt_rows.append(
            build_receipt_row(
                collection_id,
                f"nemo_candidate_direct_manifest::{collection_id}::{manifest_index}",
                manifest_url,
                destination,
                manifest_status,
                f"{row['match_scope']} processed or metadata candidate asset for {collection_id}.",
            )
        )
        if manifest_status == "downloaded":
            downloaded_asset_count += 1
        if manifest_status == "reused_cached":
            reused_cached_asset_count += 1
        materialized_asset_count += 1

    audit_row = {
        "collection_id": collection_id,
        "match_scope": row["match_scope"],
        "access": row["access"],
        "eligibility_bucket": row["eligibility_bucket"],
        "processed_manifest_count": row["processed_manifest_count"],
        "raw_manifest_count": row["raw_manifest_count"],
        "allowed_manifest_url_count": allowed_manifest_url_count,
        "rejected_raw_manifest_url_count": rejected_raw_manifest_url_count,
        "downloaded_asset_count": downloaded_asset_count,
        "reused_cached_asset_count": reused_cached_asset_count,
        "materialized_asset_count": materialized_asset_count,
        "discovery_evidence_count": discovery_evidence_count,
        "local_candidate_root": row["local_candidate_root"],
        "note": (
            f"{row['match_scope']} candidate synchronized with eligibility bucket {row['eligibility_bucket']}."
            if materialized_asset_count > 0
            else f"{row['match_scope']} candidate retained as discovery-only evidence."
        ),
    }
    return audit_row, receipt_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    source_root = resolve_repo_path(config["sources"]["nemo_archive"]["local_root"])
    reprobe_config = config["reprobe_public_sources"]["sources"]["nemo_archive"]

    subprocess.run([sys.executable, "Libs/scripts/data_acquisition/build_nemo_collection_filters.py", "--config", args.config], cwd=REPO_ROOT, check=True)

    queue_paths = candidate_queue_paths(source_root, reprobe_config, args.queue_scope)
    public_processed_queue_path = source_root / reprobe_config["public_processed_candidate_queue_relative_path"]
    public_processed_queue_preexisting = public_processed_queue_path.exists()
    queue_rows = load_public_processed_rows(source_root, reprobe_config, config) if args.queue_scope == "all_public_processed" else load_queue_rows(queue_paths)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "queue_scope": args.queue_scope,
                    "candidate_row_count": len(queue_rows),
                    "collection_ids": [row["collection_id"] for row in queue_rows],
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        complete_script_logging(
            pipeline_logger,
            pipeline_started_at,
            dry_run=True,
            scope=args.queue_scope,
            candidate_row_count=len(queue_rows),
        )
        return

    receipt_path = source_root / config["public_downloads"]["nemo_archive"]["receipt_relative_path"]
    existing_receipt_rows = load_receipt_rows(receipt_path)
    audit_rows: list[dict[str, object]] = []
    new_receipt_rows: list[dict[str, object]] = []
    if args.queue_scope == "all_public_processed":
        new_receipt_rows.append(
            build_receipt_row(
                "nemo_archive",
                "nemo_public_processed_candidate_queue_tsv",
                "https://assets.nemoarchive.org/",
                public_processed_queue_path,
                "reused_cached" if public_processed_queue_preexisting else "downloaded",
                f"Prepared {len(queue_rows)} open NeMO public processed candidates outside the primary and reference queues.",
            )
        )
    for row in queue_rows:
        audit_row, candidate_receipt_rows = sync_candidate(source_root, row, reprobe_config, dry_run=False)
        audit_rows.append(audit_row)
        new_receipt_rows.extend(candidate_receipt_rows)

    audit_path = source_root / reprobe_config["candidate_sync_audit_relative_path"]
    audit_preexisting = audit_path.exists()
    existing_audit_rows = read_tsv(audit_path) if audit_preexisting else []
    combined_audit_rows = merged_audit_rows(existing_audit_rows, audit_rows, config["schemas"]["nemo_candidate_sync_audit_fields"])
    write_tsv(audit_path, config["schemas"]["nemo_candidate_sync_audit_fields"], combined_audit_rows)
    new_receipt_rows.append(
        build_receipt_row(
            "nemo_archive",
            "nemo_candidate_sync_audit_tsv",
            "https://assets.nemoarchive.org/",
            audit_path,
            "reused_cached" if audit_preexisting else "downloaded",
            f"Synchronized {len(audit_rows)} NeMO public candidate rows.",
        )
    )

    primary_scope = str(reprobe_config["primary_match_scope"])
    reference_scope = str(reprobe_config["reference_match_scope"])
    primary_materialized_asset_count = sum(int(row["materialized_asset_count"]) for row in combined_audit_rows if row["match_scope"] == primary_scope)
    reference_materialized_asset_count = sum(int(row["materialized_asset_count"]) for row in combined_audit_rows if row["match_scope"] == reference_scope)

    gate_row = None
    for row in existing_receipt_rows:
        if row["source_accession"] == reprobe_config["gate_source_accession"] and row["asset_name"] == reprobe_config["gate_asset_name"]:
            gate_row = dict(row)
            break
    if gate_row is None:
        gate_row = build_receipt_row(
            reprobe_config["gate_source_accession"],
            reprobe_config["gate_asset_name"],
            config["sources"]["nemo_archive"]["source_url"],
            None,
            reprobe_config["primary_materialization_blocked_status"],
            "NeMO public discovery complete, primary asset not yet materialized.",
        )
    gate_row["download_date"] = date.today().isoformat()
    gate_row["local_path"] = ""
    gate_row["size_bytes"] = 0
    if primary_materialized_asset_count > 0:
        gate_row["status"] = reprobe_config["accessible_status"]
        gate_row["note"] = "NeMO public discovery is complete and primary processed or metadata assets have been materialized."
    else:
        gate_row["status"] = reprobe_config["primary_materialization_blocked_status"]
        gate_row["note"] = "NeMO public discovery complete, primary asset not yet materialized."
    new_receipt_rows.append(gate_row)

    merged_receipt_rows = upsert_receipt_rows(existing_receipt_rows, new_receipt_rows, config["schemas"]["download_receipt_fields"])
    write_tsv(receipt_path, config["schemas"]["download_receipt_fields"], merged_receipt_rows)
    refresh_checksums(source_root, config["sources"]["nemo_archive"]["checksum_file"])
    refresh_processed_catalogs(args.config)

    print(
        json.dumps(
            {
                "queue_scope": args.queue_scope,
                "candidate_row_count": len(combined_audit_rows),
                "primary_materialized_asset_count": primary_materialized_asset_count,
                "reference_materialized_asset_count": reference_materialized_asset_count,
                "audit_path": str(audit_path.relative_to(REPO_ROOT)),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        scope=args.queue_scope,
        artifact=audit_path,
        candidate_row_count=len(combined_audit_rows),
        primary_materialized_asset_count=primary_materialized_asset_count,
        reference_materialized_asset_count=reference_materialized_asset_count,
    )


if __name__ == "__main__":
    main()
