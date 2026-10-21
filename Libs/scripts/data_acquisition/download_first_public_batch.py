from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    ensure_directory,
    http_get,
    load_config,
    read_tsv,
    refresh_checksums,
    resolve_repo_path,
    today_iso,
    write_text_atomic,
    write_tsv,
)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", "", value.strip().lower())).strip()


def rank_by_priority(value: str, priority_list: list[str]) -> int:
    normalized_value = normalize(value)
    for index, candidate in enumerate(priority_list):
        if normalized_value == normalize(candidate):
            return index
    return len(priority_list)


def choose_accessions(
    rows: list[dict[str, str]],
    batch_config: dict,
    tissue_priority: list[str],
    selected_accessions: set[str] | None = None,
) -> list[dict[str, str]]:
    allowed_tissues = {normalize(value) for value in batch_config["allowed_tissues"]}
    selected = [
        row
        for row in rows
        if row["include_flag"] == "yes"
        and row["sample_context"] == batch_config["sample_context"]
        and normalize(row["tissue_or_biofluid"]) in allowed_tissues
        and (not selected_accessions or row["source_accession"] in selected_accessions)
    ]
    selected.sort(
        key=lambda row: (
            rank_by_priority(row["diagnosis_label"], batch_config["diagnosis_priority"]),
            rank_by_priority(row["tissue_or_biofluid"], tissue_priority),
            row["source_accession"],
        )
    )
    return selected[: batch_config["max_accessions"]]


def write_receipts(receipt_path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    existing_rows = read_tsv(receipt_path) if receipt_path.exists() else []
    merged_by_key: dict[tuple[str, str, str], dict[str, object]] = {
        (row["source_accession"], row["asset_name"], row["asset_url"]): row
        for row in existing_rows
    }
    for row in rows:
        merged_by_key[(str(row["source_accession"]), str(row["asset_name"]), str(row["asset_url"]))] = row
    merged_rows = sorted(
        merged_by_key.values(),
        key=lambda row: (str(row["source_accession"]), str(row["asset_name"]), str(row["asset_url"]), str(row.get("local_path", ""))),
    )
    write_tsv(receipt_path, fieldnames, merged_rows)


def build_receipt_row(
    accession: str,
    asset_name: str,
    asset_url: str,
    local_path: Path | None,
    status: str,
    note: str,
) -> dict[str, object]:
    return {
        "source_accession": accession,
        "asset_name": asset_name,
        "asset_url": asset_url,
        "local_path": local_path.relative_to(REPO_ROOT).as_posix() if local_path is not None else "",
        "download_date": today_iso(),
        "status": status,
        "size_bytes": local_path.stat().st_size if local_path is not None and local_path.exists() else 0,
        "note": note,
    }


def infer_geo_series_prefix(accession: str) -> str:
    numeric_part = accession.removeprefix("GSE")
    return f"GSE{numeric_part[:-3]}nnn"


def list_ftp_links(url: str, destination: Path) -> tuple[list[str], str]:
    try:
        response = http_get(url, raise_for_status=False)
    except requests.RequestException:
        ensure_directory(destination.parent)
        write_text_atomic(destination, "")
        return [], "blocked_network_environment"
    if response.status_code == 404:
        ensure_directory(destination.parent)
        write_text_atomic(destination, "")
        return [], "not_available"
    response.raise_for_status()
    text = response.text
    write_text_atomic(destination, text)
    return sorted(set(re.findall(r'href="([^"]+)"', text))), "downloaded"


def infer_response_kind(path_value: str) -> str:
    normalized_path = path_value.lower()
    binary_suffixes = (
        ".gz",
        ".zip",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".pdf",
        ".h5",
        ".h5ad",
        ".parquet",
        ".sdf",
        ".fasta",
    )
    if normalized_path.endswith(binary_suffixes):
        return "binary"
    return "text"


def save_http_asset(asset_url: str, destination: Path, response_kind: str) -> str:
    ensure_directory(destination.parent)
    if destination.exists():
        return "reused_cached"
    if response_kind == "binary":
        try:
            response = http_get(asset_url, stream=True)
        except requests.RequestException:
            return "blocked_network_environment"
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            response.close()
            raise ValueError(f"Unexpected HTML response for binary asset: {asset_url}")
        temporary_path = destination.parent / f"{destination.name}.part"
        try:
            with temporary_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
        finally:
            response.close()
        temporary_path.replace(destination)
        return "downloaded"
    try:
        response = http_get(asset_url, stream=True)
    except requests.RequestException:
        return "blocked_network_environment"
    content_type = response.headers.get("Content-Type", "").lower()
    if destination.suffix.lower() not in {".html", ".xml"} and "text/html" in content_type:
        response.close()
        raise ValueError(f"Unexpected HTML response for text asset: {asset_url}")
    temporary_path = destination.parent / f"{destination.name}.part"
    first_chunk = b""
    try:
        with temporary_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                if not first_chunk:
                    first_chunk = chunk
                handle.write(chunk)
    finally:
        response.close()
    normalized_prefix = first_chunk.lstrip().lower()
    if destination.suffix.lower() not in {".html", ".xml"} and (
        normalized_prefix.startswith(b"<!doctype html") or normalized_prefix.startswith(b"<html")
    ):
        temporary_path.unlink()
        raise ValueError(f"Unexpected HTML response for text asset: {asset_url}")
    temporary_path.replace(destination)
    return "downloaded"


def extract_href_targets(listing_text: str) -> list[str]:
    raw_targets = re.findall(r'href="([^"]+)"', listing_text)
    normalized_targets: list[str] = []
    for target in raw_targets:
        if target in {"../", "/"}:
            continue
        normalized_targets.append(target.rsplit("/", 1)[-1])
    return sorted(set(item for item in normalized_targets if item))


def apply_url_rewrites(asset_url: str, rewrite_map: dict[str, str]) -> str:
    rewritten_url = asset_url
    for source_prefix, target_prefix in rewrite_map.items():
        if rewritten_url.startswith(source_prefix):
            rewritten_url = f"{target_prefix}{rewritten_url[len(source_prefix):]}"
            break
    return rewritten_url


def header_lookup(headers: list[str], preferred_names: list[str]) -> str:
    normalized_map = {
        re.sub(r"[^a-z0-9]+", " ", header.lower()).strip(): header
        for header in headers
    }
    for preferred_name in preferred_names:
        normalized_preferred = re.sub(r"[^a-z0-9]+", " ", preferred_name.lower()).strip()
        if normalized_preferred in normalized_map:
            return normalized_map[normalized_preferred]
    for preferred_name in preferred_names:
        normalized_preferred = re.sub(r"[^a-z0-9]+", " ", preferred_name.lower()).strip()
        for header in headers:
            normalized_header = re.sub(r"[^a-z0-9]+", " ", header.lower()).strip()
            if normalized_preferred in normalized_header:
                return header
    return ""


def arrayexpress_remote_asset_targets(
    sdrf_path: Path,
    remote_column_map: dict[str, list[str]],
) -> list[tuple[str, str]]:
    rows = read_tsv(sdrf_path)
    if not rows:
        return []
    headers = list(rows[0].keys())
    targets: dict[tuple[str, str], None] = {}
    for asset_name, preferred_headers in remote_column_map.items():
        header = header_lookup(headers, preferred_headers)
        if not header:
            continue
        for row in rows:
            asset_url = str(row.get(header, "")).strip()
            if asset_url:
                targets[(asset_name, asset_url)] = None
    return sorted(targets.keys(), key=lambda item: (item[0], item[1]))


def parse_s3_listing_keys(listing_text: str) -> list[str]:
    root = ElementTree.fromstring(listing_text)
    return sorted(
        element.text or ""
        for element in root.findall(".//{*}Contents/{*}Key")
        if element.text
    )


def attribute_lookup(attributes: list[dict[str, str]]) -> dict[str, str]:
    return {item["name"]: item["value"] for item in attributes}


def iter_arrayexpress_file_entries(node: dict | list) -> list[dict[str, object]]:
    file_entries: list[dict[str, object]] = []
    if isinstance(node, dict):
        if node.get("type") == "file":
            file_entries.append(node)
        for value in node.values():
            if isinstance(value, (dict, list)):
                file_entries.extend(iter_arrayexpress_file_entries(value))
        return file_entries
    for item in node:
        if isinstance(item, (dict, list)):
            file_entries.extend(iter_arrayexpress_file_entries(item))
    return file_entries


def infer_arrayexpress_detail_asset_name(entry: dict[str, object], public_config: dict) -> str:
    entry_attributes = attribute_lookup(entry.get("attributes", []))
    type_name = str(entry_attributes.get("Type", "")).strip()
    if type_name:
        return type_name
    relative_file_path = str(entry.get("path", "")).strip()
    if not relative_file_path:
        return ""
    normalized_path = relative_file_path.lower()
    normalized_stem = Path(relative_file_path).stem.lower()
    if normalized_path.endswith(tuple(str(value).lower() for value in public_config["detail_raw_extensions"])):
        return "raw_archive"
    if normalized_path.endswith(tuple(str(value).lower() for value in public_config["detail_matrix_extensions"])) and any(
        str(keyword).lower() in normalized_stem
        for keyword in public_config["detail_matrix_name_keywords"]
    ):
        return "derived_matrix_archive"
    return ""


def geo_download_batch(config: dict, selected_accessions: set[str] | None = None) -> None:
    source_key = "geo"
    source_config = config["sources"][source_key]
    batch_config = config["download_batches"][source_key]
    source_root = resolve_repo_path(source_config["local_root"])
    rows = read_tsv(source_root / "accession_freeze" / "accessions.tsv")
    selected_rows = choose_accessions(rows, batch_config, config["project"]["tissue_priority"], selected_accessions)
    receipts: list[dict[str, object]] = []

    for row in selected_rows:
        accession = row["source_accession"]
        series_prefix = infer_geo_series_prefix(accession)
        accession_root = source_root / "downloads" / "series" / accession
        accession_root.mkdir(parents=True, exist_ok=True)

        series_url = f"{batch_config['ftp_base']}/{series_prefix}/{accession}/"
        soft_url = f"{series_url}soft/"
        matrix_url = f"{series_url}matrix/"

        series_index_path = accession_root / "ftp_series_index.html"
        soft_index_path = accession_root / "ftp_soft_index.html"
        matrix_index_path = accession_root / "ftp_matrix_index.html"

        _, series_index_status = list_ftp_links(series_url, series_index_path)
        soft_links, soft_index_status = list_ftp_links(soft_url, soft_index_path)
        matrix_links, matrix_index_status = list_ftp_links(matrix_url, matrix_index_path)

        for asset_name, asset_url, local_path, status in [
            ("ftp_series_index", series_url, series_index_path, series_index_status),
            ("ftp_soft_index", soft_url, soft_index_path, soft_index_status),
            ("ftp_matrix_index", matrix_url, matrix_index_path, matrix_index_status),
        ]:
            receipts.append(build_receipt_row(accession, asset_name, asset_url, local_path, status, row["title"]))

        soft_targets = [link for link in soft_links if link.endswith("_family.soft.gz")]
        if soft_targets:
            for target in soft_targets:
                destination = accession_root / target
                status = save_http_asset(f"{soft_url}{target}", destination, "binary")
                receipts.append(
                    build_receipt_row(accession, "family_soft", f"{soft_url}{target}", destination, status, row["title"])
                )
        else:
            receipts.append(
                build_receipt_row(
                    accession,
                    "family_soft",
                    soft_url,
                    None,
                    "not_available",
                    "No family soft asset was listed in the GEO soft directory.",
                )
            )

        matrix_targets = [link for link in matrix_links if link.endswith(".gz") and "series_matrix" in link]
        if matrix_targets:
            for target in matrix_targets:
                destination = accession_root / target
                status = save_http_asset(f"{matrix_url}{target}", destination, "binary")
                receipts.append(
                    build_receipt_row(
                        accession,
                        "series_matrix",
                        f"{matrix_url}{target}",
                        destination,
                        status,
                        row["title"],
                    )
                )
        else:
            receipts.append(
                build_receipt_row(
                    accession,
                    "series_matrix",
                    matrix_url,
                    None,
                    "not_available",
                    "No series matrix asset was listed in the GEO matrix directory.",
                )
            )

    write_receipts(
        source_root / "downloads" / "download_receipts.tsv",
        config["schemas"]["download_receipt_fields"],
        receipts,
    )
    refresh_checksums(source_root, source_config["checksum_file"])


def arrayexpress_download_batch(config: dict, selected_accessions: set[str] | None = None) -> None:
    source_key = "arrayexpress"
    source_config = config["sources"][source_key]
    batch_config = config["download_batches"][source_key]
    public_config = config["public_downloads"][source_key]
    query_config = config["query_repositories"][source_key]
    cohort_config = config["cohort_ready_registry"]
    source_root = resolve_repo_path(source_config["local_root"])
    rows = read_tsv(source_root / "accession_freeze" / "accessions.tsv")
    selected_rows = choose_accessions(rows, batch_config, config["project"]["tissue_priority"], selected_accessions)
    detail_root = source_root / public_config["detail_relative_dir"]
    study_root = source_root / public_config["download_relative_dir"]
    allowed_types = set(public_config["allowed_attribute_types"])
    receipts: list[dict[str, object]] = []

    for row in selected_rows:
        accession = row["source_accession"]
        detail_url = query_config["detail_url_template"].format(accession=accession)
        detail_path = detail_root / f"{accession}.json"
        detail_status = save_http_asset(detail_url, detail_path, "text")
        receipts.append(build_receipt_row(accession, "study_detail_json", detail_url, detail_path, detail_status, row["title"]))
        if detail_status == "blocked_network_environment" and not detail_path.exists():
            receipts.append(
                build_receipt_row(
                    accession,
                    "mage_tab_bundle",
                    row["public_url"],
                    None,
                    "blocked_network_environment",
                    "Study detail request failed before asset enumeration.",
                )
            )
            continue

        detail_payload = json.loads(detail_path.read_text(encoding="utf-8"))
        file_entries = iter_arrayexpress_file_entries(detail_payload.get("section", {}))
        selected_entries = [
            entry
            for entry in file_entries
            if infer_arrayexpress_detail_asset_name(entry, public_config) in allowed_types
        ]

        if not selected_entries:
            receipts.append(
                build_receipt_row(
                    accession,
                    "mage_tab_bundle",
                    row["public_url"],
                    None,
                    "not_available",
                    "No IDF, SDRF, or ADF files were listed in the BioStudies detail payload.",
                )
            )
            continue

        for entry in selected_entries:
            asset_name = infer_arrayexpress_detail_asset_name(entry, public_config)
            relative_file_path = entry["path"]
            file_url = public_config["file_url_template"].format(
                accession=accession,
                file_path=relative_file_path,
            )
            destination = study_root / accession / relative_file_path
            status = save_http_asset(file_url, destination, infer_response_kind(relative_file_path))
            receipts.append(
                build_receipt_row(
                    accession,
                    asset_name,
                    file_url,
                    destination,
                    status,
                    row["title"],
                )
            )

        sdrf_path = study_root / accession / f"{accession}.sdrf.txt"
        if sdrf_path.exists():
            remote_targets = arrayexpress_remote_asset_targets(
                sdrf_path,
                cohort_config["arrayexpress_remote_columns"],
            )
            rewrite_map = cohort_config.get("remote_url_rewrites", {})
            for asset_name, asset_url in remote_targets:
                rewritten_url = apply_url_rewrites(asset_url, rewrite_map)
                destination = study_root / accession / Path(urlparse(asset_url).path).name
                status = save_http_asset(rewritten_url, destination, infer_response_kind(destination.name))
                receipts.append(
                    build_receipt_row(
                        accession,
                        asset_name,
                        asset_url,
                        destination,
                        status,
                        row["title"],
                    )
                )

        quantitative_receipts = [
            receipt_row
            for receipt_row in receipts
            if receipt_row["source_accession"] == accession
            and receipt_row["asset_name"] in {"raw_archive", "derived_matrix_archive"}
            and receipt_row["status"] in {"downloaded", "reused_cached"}
        ]
        if not quantitative_receipts:
            for entry in file_entries:
                asset_name = infer_arrayexpress_detail_asset_name(entry, public_config)
                if asset_name not in {"raw_archive", "derived_matrix_archive"}:
                    continue
                relative_file_path = str(entry["path"]).strip()
                file_url = public_config["file_url_template"].format(
                    accession=accession,
                    file_path=relative_file_path,
                )
                destination = study_root / accession / relative_file_path
                status = save_http_asset(file_url, destination, infer_response_kind(relative_file_path))
                receipts.append(
                    build_receipt_row(
                        accession,
                        asset_name,
                        file_url,
                        destination,
                        status,
                        row["title"],
                    )
                )

    write_receipts(
        source_root / public_config["receipt_relative_path"],
        config["schemas"]["download_receipt_fields"],
        receipts,
    )
    refresh_checksums(source_root, source_config["checksum_file"])


def metabolomics_download_batch(config: dict, selected_accessions: set[str] | None = None) -> None:
    source_key = "metabolomics_workbench"
    source_config = config["sources"][source_key]
    batch_config = config["download_batches"][source_key]
    source_root = resolve_repo_path(source_config["local_root"])
    rows = read_tsv(source_root / "accession_freeze" / "accessions.tsv")
    selected_rows = choose_accessions(rows, batch_config, config["project"]["tissue_priority"], selected_accessions)
    receipts: list[dict[str, object]] = []

    for row in selected_rows:
        study_id = row["source_accession"]
        accession_root = source_root / "downloads" / "studies" / study_id
        accession_root.mkdir(parents=True, exist_ok=True)

        available_url = batch_config["available_url_template"].format(study_id=study_id)
        available_path = accession_root / f"{study_id}_available.json"
        available_status = save_http_asset(available_url, available_path, "text")
        receipts.append(
            build_receipt_row(study_id, "available_json", available_url, available_path, available_status, row["title"])
        )
        if available_status == "blocked_network_environment" and not available_path.exists():
            continue

        summary_url = batch_config["summary_url_template"].format(study_id=study_id)
        summary_path = source_root / "downloads" / "study_summaries" / f"{study_id}.json"
        summary_status = save_http_asset(summary_url, summary_path, "text")
        receipts.append(
            build_receipt_row(study_id, "summary_json", summary_url, summary_path, summary_status, row["title"])
        )

        endpoint_specs = [
            ("mwtab_txt", batch_config["mwtab_url_template"].format(study_id=study_id), accession_root / f"{study_id}_mwtab.txt"),
            ("factors_json", batch_config["factors_url_template"].format(study_id=study_id), accession_root / f"{study_id}_factors.json"),
            ("data_json", batch_config["data_url_template"].format(study_id=study_id), accession_root / f"{study_id}_data.json"),
        ]
        for asset_name, asset_url, destination in endpoint_specs:
            if destination.exists():
                receipts.append(
                    build_receipt_row(study_id, asset_name, asset_url, destination, "reused_cached", row["title"])
                )
                continue
            try:
                response = http_get(asset_url)
            except requests.RequestException:
                receipts.append(
                    build_receipt_row(
                        study_id,
                        asset_name,
                        asset_url,
                        None,
                        "blocked_network_environment",
                        row["title"],
                    )
                )
                continue
            if response.text.startswith("This output item does not exist"):
                receipts.append(
                    build_receipt_row(
                        study_id,
                        asset_name,
                        asset_url,
                        None,
                        "not_available",
                        "The study endpoint reported that the requested output item does not exist.",
                    )
                )
                continue
            write_text_atomic(destination, response.text)
            receipts.append(build_receipt_row(study_id, asset_name, asset_url, destination, "downloaded", row["title"]))

    write_receipts(
        source_root / "downloads" / "download_receipts.tsv",
        config["schemas"]["download_receipt_fields"],
        receipts,
    )
    refresh_checksums(source_root, source_config["checksum_file"])


def oasis_prepare_queue(config: dict) -> None:
    source_key = "oasis"
    source_config = config["sources"][source_key]
    batch_config = config["download_batches"][source_key]
    public_config = config["public_downloads"][source_key]
    source_root = resolve_repo_path(source_config["local_root"])
    inventory_path = source_root / "tabular" / "download_inventory.tsv"
    rows = read_tsv(inventory_path)
    updated_rows: list[dict[str, str]] = []
    for row in rows:
        if row["priority"] == "primary":
            row["status"] = batch_config["blocking_status_primary"]
            row["notes"] = batch_config["note_primary"]
        else:
            row["status"] = batch_config["blocking_status_secondary"]
            row["notes"] = batch_config["note_secondary"]
        updated_rows.append(row)
    write_tsv(inventory_path, config["schemas"]["oasis_tabular_inventory_fields"], updated_rows)

    queue_path = source_root / "tabular" / "manual_access_queue.tsv"
    queue_rows = [
        {
            "asset_family": row["asset_family"],
            "source_release": row["source_release"],
            "modality": row["modality"],
            "priority": row["priority"],
            "public_url": batch_config["access_request_url"],
            "local_target": row["local_target"],
            "status": row["status"],
            "notes": row["notes"],
        }
        for row in updated_rows
    ]
    write_tsv(queue_path, config["schemas"]["oasis_tabular_inventory_fields"], queue_rows)

    manual_note = source_root / "downloads" / "manual_access_requirements.md"
    write_text_atomic(
        manual_note,
        "\n".join(
            [
                "# OASIS Tabular Access Requirements",
                "",
                f"- Registration URL: {batch_config['registration_url']}",
                f"- Access request URL: {batch_config['access_request_url']}",
                f"- Primary queue status: {batch_config['blocking_status_primary']}",
                f"- Secondary queue status: {batch_config['blocking_status_secondary']}",
                "",
                batch_config["note_primary"],
                batch_config["note_secondary"],
                "",
            ]
        ),
    )

    receipts = [
        build_receipt_row(
            row["source_release"],
            row["asset_family"],
            batch_config["access_request_url"],
            None,
            row["status"],
            row["notes"],
        )
        for row in queue_rows
    ]
    receipts.append(
        build_receipt_row(
            "OASIS",
            "manual_access_note",
            batch_config["access_request_url"],
            manual_note,
            "prepared",
            "Manual access requirements were materialized for the OASIS tabular queue.",
        )
    )
    write_receipts(
        source_root / public_config["receipt_relative_path"],
        config["schemas"]["download_receipt_fields"],
        receipts,
    )
    refresh_checksums(source_root, source_config["checksum_file"])


def literature_parameter_bank_prepare(config: dict) -> None:
    source_key = "literature_parameter_bank"
    source_config = config["sources"][source_key]
    source_root = resolve_repo_path(source_config["local_root"])
    refresh_checksums(source_root, source_config["checksum_file"])


def download_release_api_assets(source_key: str, source_root: Path, public_config: dict) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    release_dir = source_root / public_config["release_relative_dir"]
    metadata_path = release_dir / "latest_release.json"
    metadata_status = save_http_asset(public_config["release_api_url"], metadata_path, "text")
    release_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    release_tag = release_payload["tag_name"]
    receipts.append(
        build_receipt_row(source_key, "release_metadata_json", public_config["release_api_url"], metadata_path, metadata_status, release_tag)
    )

    archive_specs = [
        ("release_tarball", release_payload["tarball_url"], release_dir / f"{release_tag}.tar.gz"),
        ("release_zipball", release_payload["zipball_url"], release_dir / f"{release_tag}.zip"),
    ]
    for asset_name, asset_url, destination in archive_specs:
        status = save_http_asset(asset_url, destination, "binary")
        receipts.append(build_receipt_row(source_key, asset_name, asset_url, destination, status, release_tag))
    return receipts


def download_directory_assets(source_key: str, source_root: Path, public_config: dict) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for asset_config in public_config.get("directory_assets", []):
        listing_path = source_root / asset_config["listing_relative_path"]
        listing_status = save_http_asset(asset_config["listing_url"], listing_path, "text")
        receipts.append(
            build_receipt_row(
                source_key,
                f"{asset_config['asset_name']}_listing",
                asset_config["listing_url"],
                listing_path,
                listing_status,
                asset_config["asset_name"],
            )
        )

        listing_text = listing_path.read_text(encoding="utf-8")
        candidates = extract_href_targets(listing_text)
        matched_files = [
            candidate
            for candidate in candidates
            if any(pattern in candidate for pattern in asset_config["include_patterns"])
        ]
        if not matched_files:
            receipts.append(
                build_receipt_row(
                    source_key,
                    asset_config["asset_name"],
                    asset_config["listing_url"],
                    None,
                    "not_available",
                    "No listed files matched the configured include patterns.",
                )
            )
            continue

        for filename in matched_files:
            asset_url = f"{asset_config['listing_url']}{filename}"
            destination = source_root / asset_config["relative_dir"] / filename
            status = save_http_asset(asset_url, destination, infer_response_kind(filename))
            receipts.append(build_receipt_row(source_key, asset_config["asset_name"], asset_url, destination, status, filename))
    return receipts


def download_s3_prefix_assets(source_key: str, source_root: Path, public_config: dict) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for asset_config in public_config.get("s3_prefix_assets", []):
        listing_path = source_root / asset_config["listing_relative_path"]
        listing_status = save_http_asset(asset_config["listing_url"], listing_path, "text")
        receipts.append(
            build_receipt_row(
                source_key,
                f"{asset_config['asset_name']}_listing",
                asset_config["listing_url"],
                listing_path,
                listing_status,
                asset_config["asset_name"],
            )
        )

        available_keys = set(parse_s3_listing_keys(listing_path.read_text(encoding="utf-8")))
        for include_file in asset_config["include_files"]:
            if include_file not in available_keys:
                receipts.append(
                    build_receipt_row(
                        source_key,
                        asset_config["asset_name"],
                        asset_config["listing_url"],
                        None,
                        "not_available",
                        f"Missing S3 key: {include_file}",
                    )
                )
                continue
            filename = Path(include_file).name
            destination = source_root / asset_config["relative_dir"] / filename
            asset_url = f"{asset_config['base_download_url']}{include_file}"
            status = save_http_asset(asset_url, destination, infer_response_kind(include_file))
            receipts.append(build_receipt_row(source_key, asset_config["asset_name"], asset_url, destination, status, filename))
    return receipts


def prepare_manual_source(source_key: str, source_config: dict, source_root: Path, public_config: dict) -> list[dict[str, object]]:
    manual_note_path = source_root / public_config["manual_note_relative_path"]
    write_text_atomic(
        manual_note_path,
        "\n".join(
            [
                f"# {source_config['source_name']} Manual Collection Requirements",
                "",
                f"- Status: {public_config['manual_status']}",
                f"- Source URL: {source_config['source_url']}",
                "",
                public_config["manual_note"],
                "",
            ]
        ),
    )
    return [
        build_receipt_row(
            source_key,
            "manual_collection_gate",
            source_config["source_url"],
            None,
            public_config["manual_status"],
            public_config["manual_note"],
        ),
        build_receipt_row(
            source_key,
            "manual_access_note",
            source_config["source_url"],
            manual_note_path,
            "prepared",
            public_config["manual_note"],
        ),
    ]


def download_public_source_assets(source_key: str, config: dict) -> None:
    source_config = config["sources"][source_key]
    public_config = config["public_downloads"][source_key]
    source_root = resolve_repo_path(source_config["local_root"])
    receipts: list[dict[str, object]] = []

    for asset in public_config.get("public_assets", []):
        destination = source_root / asset["relative_path"]
        status = save_http_asset(asset["asset_url"], destination, asset["response_kind"])
        receipts.append(
            build_receipt_row(
                source_key,
                asset["asset_name"],
                asset["asset_url"],
                destination,
                status,
                source_config["source_name"],
            )
        )

    if "release_api_url" in public_config:
        receipts.extend(download_release_api_assets(source_key, source_root, public_config))
    if "directory_assets" in public_config:
        receipts.extend(download_directory_assets(source_key, source_root, public_config))
    if "s3_prefix_assets" in public_config:
        receipts.extend(download_s3_prefix_assets(source_key, source_root, public_config))
    if "manual_note_relative_path" in public_config:
        receipts.extend(prepare_manual_source(source_key, source_config, source_root, public_config))

    write_receipts(
        source_root / public_config["receipt_relative_path"],
        config["schemas"]["download_receipt_fields"],
        receipts,
    )
    refresh_checksums(source_root, source_config["checksum_file"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the first public raw-data batch for NeuronSifter.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the public raw data registry.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional source-key subset. Defaults to all configured startup download handlers.",
    )
    parser.add_argument(
        "--source-accessions",
        default="",
        help="Optional comma-separated source_accession subset to limit cohort downloads.",
    )
    return parser.parse_args()


def selected_accession_set(accession_text: str) -> set[str]:
    return {value.strip() for value in accession_text.split(",") if value.strip()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    source_accessions = selected_accession_set(args.source_accessions)
    handlers = {
        "geo": geo_download_batch,
        "arrayexpress": arrayexpress_download_batch,
        "metabolomics_workbench": metabolomics_download_batch,
        "oasis": oasis_prepare_queue,
        "allen_brain_cell_atlas": download_public_source_assets,
        "biccn_public": download_public_source_assets,
        "nemo_archive": download_public_source_assets,
        "clue_lincs": download_public_source_assets,
        "open_targets": download_public_source_assets,
        "chembl": download_public_source_assets,
        "drugcentral": download_public_source_assets,
        "bindingdb": download_public_source_assets,
        "dgidb": download_public_source_assets,
        "drugcomb": download_public_source_assets,
        "literature_parameter_bank": literature_parameter_bank_prepare,
    }
    execution_order = list(config["source_policy"]["mainline_source_keys"])
    selected_sources = execution_order if args.sources is None else args.sources
    for source_key in selected_sources:
        handler = handlers[source_key]
        if handler is download_public_source_assets:
            handler(source_key, config)
            continue
        if source_key in {"geo", "arrayexpress", "metabolomics_workbench"}:
            handler(config, source_accessions)
            continue
        handler(config)

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
