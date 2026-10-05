from __future__ import annotations

import argparse
import collections
from datetime import date
import re
import subprocess
from pathlib import Path
import sys
from urllib.parse import urljoin

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.manual_access_workflows import ensure_manual_workflow_artifacts
from Libs.utils.data.raw_data_registry import (
    download_to_file,
    ensure_directory,
    load_config,
    read_tsv,
    refresh_checksums,
    resolve_repo_path,
    write_json,
    write_text_atomic,
    write_tsv,
)
from Libs.utils.data.raw_receipts import build_receipt_row, load_receipt_rows, upsert_receipt_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reprobe currently blocked public sources and refresh raw/processed registry state.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional source-key subset to reprobe. Defaults to config order.",
    )
    return parser.parse_args()


def response_note(url: str, status_code: int | None, error_text: str) -> str:
    if status_code is not None:
        return f"{url} responded with HTTP {status_code}."
    return f"{url} probe failed: {error_text}"


def fetch_page(url: str, trust_env: bool = False) -> tuple[int | None, str, str]:
    session = requests.Session()
    session.trust_env = trust_env
    try:
        response = session.get(url, timeout=20)
        return response.status_code, response.text, ""
    except Exception as exc:  # noqa: BLE001
        return None, "", f"{type(exc).__name__}: {exc}"


def fetch_page_with_curl(url: str) -> tuple[int | None, str, str]:
    command = [
        "curl",
        "-L",
        "--max-time",
        "20",
        "-sS",
        "-w",
        "\nCURL_HTTP_STATUS:%{http_code}",
        url,
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return None, "", f"curl_exit_{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
    marker = "\nCURL_HTTP_STATUS:"
    if marker not in result.stdout:
        return None, "", "curl did not emit an HTTP status marker."
    response_text, status_text = result.stdout.rsplit(marker, 1)
    status_text = status_text.strip()
    status_code = int(status_text) if status_text.isdigit() else None
    return status_code, response_text, ""


def fetch_page_resilient(url: str) -> tuple[int | None, str, str]:
    attempts: list[str] = []
    last_status_code: int | None = None
    last_response_text = ""
    for trust_env in [False, True]:
        status_code, response_text, error_text = fetch_page(url, trust_env=trust_env)
        if status_code is not None and status_code < 400:
            return status_code, response_text, ""
        if status_code is not None:
            last_status_code = status_code
            last_response_text = response_text
            attempts.append(f"requests(trust_env={trust_env}) HTTP {status_code}")
        else:
            attempts.append(f"requests(trust_env={trust_env}): {error_text}")
    status_code, response_text, error_text = fetch_page_with_curl(url)
    if status_code is not None and status_code < 400:
        return status_code, response_text, ""
    if status_code is not None:
        last_status_code = status_code
        last_response_text = response_text
        attempts.append(f"curl HTTP {status_code}")
    else:
        attempts.append(f"curl: {error_text}")
    if last_status_code is not None:
        return last_status_code, last_response_text, " | ".join(attempts)
    return None, "", " | ".join(attempts)


def extract_public_file_links(base_url: str, response_text: str, allowed_suffixes: list[str], max_links: int) -> list[str]:
    discovered_links: list[str] = []
    for match in re.findall(r'href=["\\\']([^"\\\']+)["\\\']', response_text):
        absolute_url = urljoin(base_url, match)
        if not any(absolute_url.lower().endswith(suffix.lower()) for suffix in allowed_suffixes):
            continue
        if absolute_url not in discovered_links:
            discovered_links.append(absolute_url)
        if len(discovered_links) >= max_links:
            break
    return discovered_links


def nemo_collection_ids_from_text(response_text: str) -> list[str]:
    collection_ids = sorted(set(re.findall(r'nemo:(?:col|dat)-[a-z0-9]+', response_text)))
    return collection_ids


def assets_api_urls_from_json(response_text: str) -> list[str]:
    return sorted(
        set(
            re.findall(
                r'https://assets\.nemoarchive\.org/api/(?:collection|project|grant|subject|sample|file)/[^"\'\s,\]]+',
                response_text,
            )
        )
    )


def joined_unique_text(values: list[str]) -> str:
    ordered_values: list[str] = []
    seen_values: set[str] = set()
    for value in values:
        normalized_value = str(value).strip()
        if not normalized_value or normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        ordered_values.append(normalized_value)
    return "|".join(ordered_values)


def anatomical_region_text(payload: dict) -> str:
    region_values: list[str] = []
    for region in payload.get("anatomical_regions") or []:
        region_name = str(region.get("region_name", "")).strip()
        short_name = str(region.get("short_name", "")).strip()
        if region_name:
            region_values.append(region_name)
        if short_name:
            region_values.append(short_name)
    return joined_unique_text(region_values)


def manifest_summary(payload: dict) -> dict[str, object]:
    manifest_rows = [item for item in (payload.get("manifest_file_urls") or []) if item.get("url")]
    manifest_urls = [str(item.get("url", "")).strip() for item in manifest_rows if str(item.get("url", "")).strip()]
    manifest_types = [str(item.get("type", "")).strip().lower() for item in manifest_rows if str(item.get("type", "")).strip()]
    manifest_protocols = [str(item.get("protocol", "")).strip().lower() for item in manifest_rows if str(item.get("protocol", "")).strip()]
    manifest_file_count_total = 0
    manifest_size_bytes_total = 0
    processed_manifest_count = 0
    raw_manifest_count = 0
    for manifest_row in manifest_rows:
        manifest_type = str(manifest_row.get("type", "")).strip().lower()
        if manifest_type == "processed":
            processed_manifest_count += 1
        if manifest_type == "raw":
            raw_manifest_count += 1
        file_count = manifest_row.get("file_count")
        if isinstance(file_count, int):
            manifest_file_count_total += file_count
        size_bytes = manifest_row.get("size")
        if isinstance(size_bytes, int):
            manifest_size_bytes_total += size_bytes
    return {
        "manifest_count": len(manifest_urls),
        "manifest_urls": "|".join(manifest_urls),
        "manifest_types": joined_unique_text(manifest_types),
        "manifest_protocols": joined_unique_text(manifest_protocols),
        "manifest_file_count_total": manifest_file_count_total,
        "manifest_size_bytes_total": manifest_size_bytes_total,
        "processed_manifest_count": processed_manifest_count,
        "raw_manifest_count": raw_manifest_count,
    }


def collection_summary_row(payload: dict) -> dict[str, object]:
    publication = payload.get("publication") or {}
    manifest_details = manifest_summary(payload)
    return {
        "collection_id": payload.get("id", ""),
        "short_name": payload.get("short_name", ""),
        "name": payload.get("name", ""),
        "description": payload.get("description", ""),
        "access": payload.get("access", ""),
        "taxa": joined_unique_text([str(item.get("name", "")) for item in (payload.get("taxa") or [])]),
        "keywords": joined_unique_text([str(value) for value in (payload.get("keywords") or [])]),
        "modalities": joined_unique_text([str(value) for value in (payload.get("modalities") or [])]),
        "anatomical_regions": anatomical_region_text(payload),
        "collection_type": payload.get("collection_type", ""),
        "publication_title": publication.get("title", ""),
        "publication_doi": publication.get("doi", ""),
        **manifest_details,
    }


def human_collection_flag(row: dict[str, object], human_terms: list[str]) -> bool:
    taxa_text = str(row.get("taxa", ""))
    short_name_text = str(row.get("short_name", ""))
    name_text = str(row.get("name", ""))

    lowered_taxa = taxa_text.lower()
    lowered_short_name = short_name_text.lower()
    lowered_name = name_text.lower()
    human_flag = any(term.lower() in lowered_taxa for term in human_terms)
    if not human_flag and not taxa_text:
        human_flag = any(term.lower() in " ".join([lowered_short_name, lowered_name]) for term in human_terms)
    return human_flag


def access_allowed(row: dict[str, object], target_access_values: list[str]) -> bool:
    access_value = str(row.get("access", "")).lower()
    return access_value in {value.lower() for value in target_access_values}


def match_details(row: dict[str, object], target_keywords: list[str], match_fields: list[str]) -> tuple[list[str], list[str]]:
    matched_terms: set[str] = set()
    matched_fields: set[str] = set()
    for field_name in match_fields:
        field_text = str(row.get(field_name, ""))
        lowered_field_text = field_text.lower()
        for keyword in target_keywords:
            normalized_keyword = keyword.lower()
            if normalized_keyword and re.search(rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])", lowered_field_text):
                matched_terms.add(keyword)
                matched_fields.add(field_name)
    return sorted(matched_terms), sorted(matched_fields)


def candidate_eligibility_bucket(row: dict[str, object]) -> str:
    access_value = str(row.get("access", "")).lower()
    if access_value != "open":
        return "restricted_discovery"
    processed_manifest_count = int(row.get("processed_manifest_count", 0) or 0)
    raw_manifest_count = int(row.get("raw_manifest_count", 0) or 0)
    manifest_count = int(row.get("manifest_count", 0) or 0)
    if processed_manifest_count > 0:
        return "open_processed_candidate"
    if raw_manifest_count > 0:
        return "open_raw_only_discovery"
    if manifest_count > 0:
        return "open_manifest_discovery"
    return "open_no_manifest_discovery"


def filtered_collection_row(
    row: dict[str, object],
    human_terms: list[str],
    target_access_values: list[str],
    target_keywords: list[str],
    match_fields: list[str],
    match_scope: str,
) -> dict[str, object] | None:
    if not human_collection_flag(row, human_terms):
        return None
    if not access_allowed(row, target_access_values):
        return None
    match_terms, matched_fields = match_details(row, target_keywords, match_fields)
    if not match_terms:
        return None

    return {
        "collection_id": row["collection_id"],
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
        "manifest_protocols": row["manifest_protocols"],
        "processed_manifest_count": row["processed_manifest_count"],
        "raw_manifest_count": row["raw_manifest_count"],
        "human_flag": "yes",
        "access_bucket": str(row["access"]).lower(),
        "match_scope": match_scope,
        "match_fields": "|".join(matched_fields),
        "keyword_match_terms": "|".join(match_terms),
        "eligibility_bucket": candidate_eligibility_bucket(row),
        "note": f"{match_scope}_keyword_match",
    }


def load_nemo_candidate_sync_rows(config: dict) -> list[dict[str, str]]:
    reprobe_config = config["reprobe_public_sources"]["sources"]["nemo_archive"]
    audit_path = resolve_repo_path(config["sources"]["nemo_archive"]["local_root"]) / reprobe_config["candidate_sync_audit_relative_path"]
    if not audit_path.exists():
        return []
    return read_tsv(audit_path)


def nemo_candidate_materialization_counts(config: dict) -> dict[str, int]:
    reprobe_config = config["reprobe_public_sources"]["sources"]["nemo_archive"]
    primary_scope = str(reprobe_config["primary_match_scope"])
    reference_scope = str(reprobe_config["reference_match_scope"])
    primary_open_candidate_count = 0
    primary_materialized_asset_count = 0
    reference_materialized_asset_count = 0
    for row in load_nemo_candidate_sync_rows(config):
        if row["match_scope"] == primary_scope and row["access"] == "open":
            primary_open_candidate_count += 1
            primary_materialized_asset_count += int(row["materialized_asset_count"] or 0)
        if row["match_scope"] == reference_scope:
            reference_materialized_asset_count += int(row["materialized_asset_count"] or 0)
    return {
        "primary_open_candidate_count": primary_open_candidate_count,
        "primary_materialized_asset_count": primary_materialized_asset_count,
        "reference_materialized_asset_count": reference_materialized_asset_count,
    }


def save_probe_asset(source_root: Path, asset_config: dict, status_code: int | None, response_text: str, error_text: str) -> dict[str, object]:
    local_path = source_root / asset_config["relative_path"]
    if response_text:
        ensure_directory(local_path.parent)
        write_text_atomic(local_path, response_text)
    return build_receipt_row(
        asset_config.get("source_accession", asset_config.get("gate_source_accession", "")),
        asset_config["asset_name"],
        asset_config["url"],
        local_path if response_text else None,
        "reprobed",
        response_note(asset_config["url"], status_code, error_text),
    )


def download_binary_asset(url: str, destination: Path) -> str:
    ensure_directory(destination.parent)
    if destination.exists():
        return "reused_cached"
    download_to_file(url, destination)
    return "downloaded"


def download_text_asset(url: str, destination: Path, response_text: str) -> str:
    ensure_directory(destination.parent)
    if destination.exists():
        return "reused_cached"
    write_text_atomic(destination, response_text)
    return "downloaded"


def download_drugcomb_mirror_assets(source_root: Path, reprobe_config: dict) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    downloaded_asset_count = 0

    for asset in reprobe_config.get("mirror_landing_assets", []):
        status_code, response_text, error_text = fetch_page_resilient(asset["url"])
        destination = source_root / asset["relative_path"]
        if status_code == 200 and response_text:
            status = download_text_asset(asset["url"], destination, response_text)
            if status in {"downloaded", "reused_cached"}:
                downloaded_asset_count += 1
            rows.append(build_receipt_row("drugcomb", asset["asset_name"], asset["url"], destination, status, "Official DrugComb mirror landing page."))
        else:
            rows.append(build_receipt_row("drugcomb", asset["asset_name"], asset["url"], None, "not_available", response_note(asset["url"], status_code, error_text)))

    for record in reprobe_config.get("mirror_api_records", []):
        status_code, response_text, error_text = fetch_page_resilient(record["api_url"])
        record_json_path = source_root / record["relative_json_path"]
        if status_code != 200 or not response_text:
            rows.append(build_receipt_row("drugcomb", f"zenodo_record_{record['record_id']}", record["api_url"], None, "not_available", response_note(record["api_url"], status_code, error_text)))
            continue
        payload = requests.models.complexjson.loads(response_text)
        if record_json_path.exists():
            record_status = "reused_cached"
        else:
            write_json(record_json_path, payload)
            record_status = "downloaded"
        rows.append(build_receipt_row("drugcomb", f"zenodo_record_{record['record_id']}", record["api_url"], record_json_path, record_status, "Official DrugComb Zenodo metadata record."))
        file_lookup = {item["key"]: item for item in payload.get("files", [])}
        for file_config in record.get("files", []):
            key = file_config["key"]
            if key not in file_lookup:
                rows.append(build_receipt_row("drugcomb", key, record["api_url"], None, "not_available", f"File {key} not present in Zenodo record {record['record_id']}."))
                continue
            file_url = file_lookup[key]["links"]["self"]
            destination = source_root / file_config["relative_path"]
            status = download_binary_asset(file_url, destination)
            if status in {"downloaded", "reused_cached"}:
                downloaded_asset_count += 1
            rows.append(build_receipt_row("drugcomb", key, file_url, destination, status, f"Official DrugComb Zenodo file from record {record['record_id']}."))
    return rows, downloaded_asset_count


def update_gate_row(existing_receipt_rows: list[dict[str, str]], source_accession: str, asset_name: str, status: str, note: str) -> dict[str, object]:
    for row in existing_receipt_rows:
        if row["source_accession"] == source_accession and row["asset_name"] == asset_name and row["status"] not in {"downloaded", "reused_cached"}:
            updated = dict(row)
            updated["status"] = status
            updated["download_date"] = date.today().isoformat()
            updated["note"] = note
            updated["size_bytes"] = 0
            return updated
    return {
        "source_accession": source_accession,
        "asset_name": asset_name,
        "asset_url": "",
        "local_path": "",
        "download_date": date.today().isoformat(),
        "status": status,
        "size_bytes": 0,
        "note": note,
    }


def reprobe_drugcomb(config: dict, source_key: str, existing_receipt_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    source_root = REPO_ROOT / config["sources"][source_key]["local_root"]
    reprobe_config = config["reprobe_public_sources"]["sources"][source_key]
    rows: list[dict[str, object]] = []
    accessible_count = 0
    download_page_html = ""
    probe_failures: list[str] = []
    for asset in reprobe_config["probe_assets"]:
        status_code, response_text, error_text = fetch_page_resilient(asset["url"])
        if status_code == 200:
            accessible_count += 1
        elif error_text:
            probe_failures.append(f"{asset['asset_name']}: {error_text}")
        if asset["asset_name"] == "reprobe_download_html" and status_code == 200:
            download_page_html = response_text
        rows.append(save_probe_asset(source_root, asset, status_code, response_text, error_text))
    if download_page_html:
        for index, file_url in enumerate(
            extract_public_file_links(
                "https://www.drugcomb.org/download/",
                download_page_html,
                [".csv", ".tsv", ".json"],
                10,
            ),
            start=1,
        ):
            status_code, response_text, error_text = fetch_page_resilient(file_url)
            relative_name = file_url.rsplit("/", 1)[-1] or f"public_asset_{index}"
            rows.append(
                save_probe_asset(
                    source_root,
                    {
                        "asset_name": f"reprobe_public_file_{index}",
                        "url": file_url,
                        "relative_path": f"downloads/reprobe/public_assets/{relative_name}",
                    },
                    status_code,
                    response_text,
                    error_text,
                )
            )
    if accessible_count > 0:
        gate_status = reprobe_config["accessible_status"]
        gate_note = "DrugComb public pages are reachable from the current environment."
    else:
        gate_status = reprobe_config["inaccessible_status"]
        gate_note = "DrugComb public pages remain inaccessible from the current environment; proxy-path TLS EOF and direct-connection timeout were reproduced."
        if probe_failures:
            gate_note = gate_note + " " + " | ".join(probe_failures[:2])
    mirror_rows, mirror_download_count = download_drugcomb_mirror_assets(source_root, reprobe_config)
    rows.extend(mirror_rows)
    if mirror_download_count > 0:
        gate_status = reprobe_config["accessible_status"]
        gate_note = f"DrugComb official Helsinki/Zenodo mirror is reachable and {mirror_download_count} public assets were materialized."
    rows.append(
        update_gate_row(
            existing_receipt_rows,
            reprobe_config["gate_source_accession"],
            reprobe_config["gate_asset_name"],
            gate_status,
            gate_note,
        )
    )
    return rows


def reprobe_nemo_archive(config: dict, source_key: str, existing_receipt_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    source_root = REPO_ROOT / config["sources"][source_key]["local_root"]
    reprobe_config = config["reprobe_public_sources"]["sources"][source_key]
    rows: list[dict[str, object]] = []
    accessible_assets: dict[str, int | None] = {}
    for asset in reprobe_config["probe_assets"]:
        status_code, response_text, error_text = fetch_page_resilient(asset["url"])
        accessible_assets[asset["asset_name"]] = status_code
        rows.append(save_probe_asset(source_root, asset, status_code, response_text, error_text))
    parsed_collection_ids: set[str] = set()
    for asset in reprobe_config.get("public_metadata_assets", []):
        status_code, response_text, error_text = fetch_page_resilient(asset["url"])
        accessible_assets[asset["asset_name"]] = status_code
        rows.append(save_probe_asset(source_root, asset, status_code, response_text, error_text))
        if status_code == 200 and response_text:
            parsed_collection_ids.update(nemo_collection_ids_from_text(response_text))
    configured_collection_ids = {item["collection_id"] for item in reprobe_config.get("example_collections", [])}
    discovered_collection_urls: set[str] = {
        f"https://assets.nemoarchive.org/api/collection/{collection_id}"
        for collection_id in sorted(configured_collection_ids | parsed_collection_ids)
    }
    queue = collections.deque(sorted(discovered_collection_urls))
    seen_urls: set[str] = set()
    discovered_collection_rows: list[dict[str, object]] = []
    while queue and len(seen_urls) < int(reprobe_config["crawl_node_limit"]):
        api_url = queue.popleft()
        if api_url in seen_urls:
            continue
        seen_urls.add(api_url)
        status_code, response_text, error_text = fetch_page_resilient(api_url)
        asset_name = f"reprobe_collection_api::{api_url.rsplit('/', 1)[-1]}"
        collection_slug = api_url.rsplit("/", 1)[-1].replace(":", "_")
        relative_path = f"downloads/reprobe/collections/{collection_slug}/metadata.json"
        accessible_assets[asset_name] = status_code
        rows.append(
            save_probe_asset(
                source_root,
                {"asset_name": asset_name, "url": api_url, "relative_path": relative_path},
                status_code,
                response_text,
                error_text,
            )
        )
        if status_code != 200 or not response_text:
            continue
        if "/api/collection/" in api_url:
            payload = requests.models.complexjson.loads(response_text)
            discovered_collection_rows.append(collection_summary_row(payload))
            collection_id = str(payload.get("id", ""))
            if collection_id:
                landing_url = f"https://assets.nemoarchive.org/collection/{collection_id}"
                landing_name = f"reprobe_collection_landing::{collection_id}"
                landing_relative_path = f"downloads/reprobe/collections/{collection_slug}/landing.html"
                landing_status, landing_text, landing_error = fetch_page_resilient(landing_url)
                accessible_assets[landing_name] = landing_status
                rows.append(
                    save_probe_asset(
                        source_root,
                        {"asset_name": landing_name, "url": landing_url, "relative_path": landing_relative_path},
                        landing_status,
                        landing_text,
                        landing_error,
                    )
                )
        for linked_url in assets_api_urls_from_json(response_text):
            if linked_url not in seen_urls:
                queue.append(linked_url)
    discovered_collection_rows_by_id = {str(row["collection_id"]): row for row in discovered_collection_rows if str(row["collection_id"])}
    discovered_collection_rows = [discovered_collection_rows_by_id[collection_id] for collection_id in sorted(discovered_collection_rows_by_id)]
    if discovered_collection_rows:
        discovered_tsv_path = source_root / reprobe_config["discovered_collections_relative_path"]
        write_tsv(discovered_tsv_path, config["schemas"]["nemo_discovered_collection_fields"], discovered_collection_rows)
        rows.append(
            build_receipt_row(
                "nemo_archive",
                "reprobe_discovered_collections_tsv",
                "https://assets.nemoarchive.org/",
                discovered_tsv_path,
                "downloaded",
                f"Auto-discovered {len(discovered_collection_rows)} public collection metadata rows from the assets graph.",
            )
        )
    candidate_counts = nemo_candidate_materialization_counts(config)
    discovery_complete_flag = bool(discovered_collection_rows)
    publication_status = accessible_assets.get("reprobe_publication_release_html")
    if discovery_complete_flag and candidate_counts["primary_materialized_asset_count"] > 0:
        gate_status = reprobe_config["accessible_status"]
        gate_note = "NeMO public discovery is complete and primary processed or metadata assets have been materialized."
    elif discovery_complete_flag:
        gate_status = reprobe_config["primary_materialization_blocked_status"]
        gate_note = "NeMO public discovery complete, primary asset not yet materialized."
    elif publication_status == 200:
        gate_status = reprobe_config["discovery_blocked_status"]
        gate_note = "NeMO publication-release discovery is reachable, but stable collection classification is not yet complete."
    elif accessible_assets.get("reprobe_portal_home_html") == 200:
        gate_status = reprobe_config["discovery_blocked_status"]
        gate_note = "NeMO public pages are reachable, but publication-release discovery still does not expose a stable public listing without further discovery support."
    else:
        gate_status = reprobe_config["inaccessible_status"]
        gate_note = "NeMO public pages are currently unreachable from the present environment."
    rows.append(
        update_gate_row(
            existing_receipt_rows,
            reprobe_config["gate_source_accession"],
            reprobe_config["gate_asset_name"],
            gate_status,
            gate_note,
        )
    )
    return rows


def reprobe_clue_lincs(config: dict, source_key: str, existing_receipt_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    source_root = REPO_ROOT / config["sources"][source_key]["local_root"]
    reprobe_config = config["reprobe_public_sources"]["sources"][source_key]
    rows: list[dict[str, object]] = []
    accessible_count = 0
    for asset in reprobe_config["probe_assets"]:
        status_code, response_text, error_text = fetch_page_resilient(asset["url"])
        if status_code == 200:
            accessible_count += 1
        rows.append(save_probe_asset(source_root, asset, status_code, response_text, error_text))
    clue_sync_complete = any(
        row["source_accession"] == "clue_lincs" and row["asset_name"] == "clue_build_catalog_json"
        for row in existing_receipt_rows
    )
    if clue_sync_complete:
        gate_status = "resolved_public_accessible"
        gate_note = "CLUE public build metadata and selected core matrices are already synchronized."
    else:
        gate_status = reprobe_config["accessible_status"] if accessible_count > 0 else reprobe_config["inaccessible_status"]
        gate_note = (
            "CLUE public pages remain reachable, but signature-level data still require a CLUE account."
            if accessible_count > 0
            else "CLUE public pages are currently unreachable from the present environment."
        )
    rows.append(
        update_gate_row(
            existing_receipt_rows,
            reprobe_config["gate_source_accession"],
            reprobe_config["gate_asset_name"],
            gate_status,
            gate_note,
        )
    )
    return rows


def reprobe_oasis(config: dict, source_key: str, existing_receipt_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    source_root = REPO_ROOT / config["sources"][source_key]["local_root"]
    reprobe_config = config["reprobe_public_sources"]["sources"][source_key]
    rows: list[dict[str, object]] = []
    accessible = True
    for asset in reprobe_config["probe_pages"]:
        status_code, response_text, error_text = fetch_page_resilient(asset["url"])
        if status_code != 200:
            accessible = False
        rows.append(save_probe_asset(source_root, asset, status_code, response_text, error_text))
    for gate_asset in reprobe_config["probe_assets"]:
        gate_status = gate_asset["accessible_status"] if accessible else gate_asset["inaccessible_status"]
        gate_note = (
            "OASIS access pages remain reachable, but click-through approval is still required."
            if accessible
            else "OASIS access pages are currently unreachable from the present environment."
        )
        rows.append(
            update_gate_row(
                existing_receipt_rows,
                gate_asset["source_accession"],
                gate_asset["asset_name"],
                gate_status,
                gate_note,
            )
        )
    return rows


def refresh_processed_catalogs(config_path: str) -> None:
    subprocess.run([sys.executable, "Libs/scripts/builders/build_processed_registry.py", "--config", config_path], cwd=REPO_ROOT, check=True)
    subprocess.run([sys.executable, "Libs/scripts/reports/audit_collection_completion.py", "--config", config_path], cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    handlers = {
        "drugcomb": reprobe_drugcomb,
        "nemo_archive": reprobe_nemo_archive,
        "clue_lincs": reprobe_clue_lincs,
        "oasis": reprobe_oasis,
    }
    selected_sources = config["reprobe_public_sources"]["source_order"] if args.sources is None else args.sources

    for source_key in selected_sources:
        source_config = config["sources"][source_key]
        ensure_manual_workflow_artifacts(source_key, source_config, config)
        source_root = REPO_ROOT / source_config["local_root"]
        receipt_path = source_root / config["public_downloads"][source_key]["receipt_relative_path"]
        existing_receipt_rows = load_receipt_rows(receipt_path)
        new_rows = handlers[source_key](config, source_key, existing_receipt_rows)
        merged_rows = upsert_receipt_rows(existing_receipt_rows, new_rows, config["schemas"]["download_receipt_fields"])
        write_tsv(receipt_path, config["schemas"]["download_receipt_fields"], merged_rows)
        refresh_checksums(source_root, source_config["checksum_file"])

    if "nemo_archive" in selected_sources:
        subprocess.run([sys.executable, "Libs/scripts/data_acquisition/build_nemo_collection_filters.py", "--config", args.config], cwd=REPO_ROOT, check=True)
    refresh_processed_catalogs(args.config)

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
