from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import importlib
import importlib.util
from http.cookiejar import MozillaCookieJar
from http.cookiejar import CookieJar
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.controlled_research_config import (
    assert_controlled_governance_approval,
    assert_controlled_network_credentials,
    load_controlled_research_config,
)
from Libs.utils.data.raw_data_registry import (
    ensure_directory,
    refresh_checksums,
    repo_root,
    sha256_of_file,
    temporary_output_path,
    today_iso,
    write_json,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize approved ADNI controlled-access assets into the private namespace.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate cookie and manifest schema without downloading ADNI assets.")
    parser.add_argument("--collection-first-profile", default="", help="Override the YAML-declared collection-first execution profile for this run.")
    return parser.parse_args()


def adni_section(config: dict) -> dict:
    return config["adni_controlled_access"]


def acquisition_config(config: dict) -> dict:
    return adni_section(config).get("acquisition", {"mode": "approved_asset_manifest"})


def apply_collection_first_profile_override(config: dict, profile_name: str) -> None:
    if not profile_name:
        return
    collection_config = acquisition_config(config)["collection_first"]
    profile_registry = collection_config.get("profiles", {})
    if profile_name not in profile_registry:
        raise ValueError(f"Unknown ADNI collection-first execution profile: {profile_name}")
    collection_config["execution_profile"] = profile_name


def resolve_configured_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def safe_relative_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"ADNI controlled-access target path must be repository-relative without parent traversal: {path_value}")
    return path


def controlled_display_path(path: Path) -> str:
    resolved_path = path.resolve()
    repository_root = repo_root()
    if resolved_path.is_relative_to(repository_root):
        return resolved_path.relative_to(repository_root).as_posix()
    return resolved_path.as_posix()


def load_adni_credentials(config: dict) -> dict[str, str]:
    credential_config = adni_section(config)["credentials"]
    required_env_names = [
        str(credential_config["email_env"]),
        str(credential_config["password_env"]),
    ]
    missing_names = [name for name in required_env_names if not os.environ.get(name)]
    if missing_names:
        raise RuntimeError("Missing required ADNI controlled-access form-login credentials in the configured environment.")
    return {
        "email": str(os.environ[required_env_names[0]]),
        "password": str(os.environ[required_env_names[1]]),
    }


def credential_presence(config: dict) -> dict[str, str]:
    credential_config = adni_section(config)["credentials"]
    return {
        "login_identifier_configured": "set" if os.environ.get(str(credential_config["email_env"])) else "missing",
        "login_secret_configured": "set" if os.environ.get(str(credential_config["password_env"])) else "missing",
    }


def read_approved_asset_rows(config: dict) -> list[dict[str, str]]:
    controlled_config = adni_section(config)
    private_root = resolve_configured_path(controlled_config["paths"]["private_root"])
    manifest_path = private_root / controlled_config["paths"]["approved_asset_manifest"]
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames or []
        expected_fields = controlled_config["sync"]["approved_asset_manifest_fields"]
        if header != expected_fields:
            raise ValueError(f"{manifest_path} header mismatch: {header}")
        return list(reader)


def manifest_path(config: dict) -> Path:
    return resolve_configured_path(str(acquisition_config(config)["manifest_path"]))


def normalized_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower()).strip("_")


def read_json_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records: object = payload
    if isinstance(payload, dict):
        for candidate_key in ["records", "items", "data", "files", "images", "manifest"]:
            if isinstance(payload.get(candidate_key), list):
                records = payload[candidate_key]
                break
    if not isinstance(records, list):
        raise ValueError("ADNI manifest JSON must be a list of records or contain a records/items/data/files/images list.")
    rows: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("ADNI manifest JSON records must be objects.")
        rows.append({str(key): "" if value is None else str(value) for key, value in record.items()})
    return rows


def sniff_manifest_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    if path.suffix.lower() == ".csv":
        return ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
    return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter


def read_delimited_manifest(path: Path) -> list[dict[str, str]]:
    delimiter = sniff_manifest_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("ADNI manifest must include a header row.")
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]


def read_ida_export_manifest(config: dict) -> list[dict[str, str]]:
    path = manifest_path(config)
    if not path.exists():
        raise FileNotFoundError(f"ADNI acquisition manifest is missing at configured private path: {controlled_display_path(path)}")
    manifest_format = str(acquisition_config(config).get("manifest_format", "auto"))
    suffix = path.suffix.lower()
    if manifest_format == "json" or (manifest_format == "auto" and suffix == ".json"):
        return read_json_manifest(path)
    if manifest_format in {"auto", "csv", "tsv"}:
        return read_delimited_manifest(path)
    raise ValueError(f"Unsupported ADNI acquisition manifest format: {manifest_format}")


MANIFEST_COLUMN_PATTERNS: dict[str, list[str]] = {
    "file_id": [
        r"(^|_)image(uid|_uid|id|_id)$",
        r"(^|_)file(_?id)?$",
        r"(^|_)series(_?id)?$",
        r"(^|_)archive(_?id)?$",
        r"(^|_)object(_?id)?$",
    ],
    "package_id": [
        r"(^|_)package(_?id)?$",
        r"(^|_)cart(_?id)?$",
        r"(^|_)batch(_?id)?$",
    ],
    "download_url": [
        r"download.*url",
        r"(^|_)url$",
        r"(^|_)uri$",
        r"(^|_)href$",
        r"(^|_)link$",
        r"access.*url",
    ],
    "modality": [
        r"modality",
        r"image.*type",
        r"scan.*type",
        r"data.*type",
        r"series.*type",
    ],
    "visit": [
        r"(^|_)visit",
        r"viscode",
        r"visit.*code",
        r"timepoint",
    ],
    "subject": [
        r"subject",
        r"participant",
        r"patient",
        r"(^|_)ptid$",
        r"(^|_)rid$",
    ],
    "phase": [
        r"phase",
        r"project",
        r"cohort",
        r"colprot",
        r"study",
    ],
    "filename": [
        r"file.*name",
        r"filename",
        r"archive.*name",
        r"download.*name",
        r"(^|_)path$",
    ],
    "size": [
        r"(^|_)size",
        r"file.*size",
        r"bytes",
    ],
}


def infer_manifest_columns(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    columns = list(rows[0].keys()) if rows else []
    inferred = {semantic: [] for semantic in MANIFEST_COLUMN_PATTERNS}
    for column in columns:
        normalized = normalized_column_name(column)
        for semantic, patterns in MANIFEST_COLUMN_PATTERNS.items():
            if any(re.search(pattern, normalized) for pattern in patterns):
                inferred[semantic].append(column)
    for column in columns:
        if column in inferred["download_url"]:
            continue
        if any(str(row.get(column, "")).strip().startswith(("http://", "https://")) for row in rows[:100]):
            inferred["download_url"].append(column)
    return inferred


def identifier_semantic_for_column(column_name: str) -> str:
    normalized = normalized_column_name(column_name)
    if "package" in normalized or "cart" in normalized or "batch" in normalized:
        return "package_id"
    if "file" in normalized or "series" in normalized or "archive" in normalized or "object" in normalized:
        return "file_id"
    return "image_id"


def first_nonempty_value(row: dict[str, str], columns: list[str]) -> str:
    for column in columns:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def direct_download_url(row: dict[str, str], inferred_columns: dict[str, list[str]]) -> str:
    for column in inferred_columns["download_url"]:
        value = str(row.get(column, "")).strip()
        if value.startswith(("http://", "https://")):
            return value
    return ""


def identifier_request_candidate(row: dict[str, str], inferred_columns: dict[str, list[str]]) -> tuple[str, str]:
    for semantic in ["file_id", "package_id"]:
        for column in inferred_columns[semantic]:
            value = str(row.get(column, "")).strip()
            if value:
                return identifier_semantic_for_column(column), value
    return "", ""


def manifest_record_counts(rows: list[dict[str, str]], inferred_columns: dict[str, list[str]]) -> dict[str, int]:
    direct_download_count = 0
    identifier_request_count = 0
    for row in rows:
        if direct_download_url(row, inferred_columns):
            direct_download_count += 1
            continue
        identifier_kind, identifier_value = identifier_request_candidate(row, inferred_columns)
        if identifier_kind and identifier_value:
            identifier_request_count += 1
    return {
        "record_count": len(rows),
        "direct_download_record_count": direct_download_count,
        "identifier_request_record_count": identifier_request_count,
        "downloadable_record_count": direct_download_count + identifier_request_count,
    }


def manifest_schema_audit(config: dict, rows: list[dict[str, str]], cookie_session_status: str) -> dict[str, object]:
    inferred_columns = infer_manifest_columns(rows)
    counts = manifest_record_counts(rows, inferred_columns)
    minimum_downloadable_records = int(acquisition_config(config)["minimum_downloadable_records"])
    matched_roles = sorted(
        {
            role
            for row in rows
            for role in [role_for_manifest_record(row, inferred_columns, config)]
            if role
        }
    )
    missing_requirements: list[str] = []
    if not rows:
        missing_requirements.append("manifest_records")
    if counts["downloadable_record_count"] < minimum_downloadable_records:
        missing_requirements.append("downloadable_records")
    if not inferred_columns["download_url"] and not inferred_columns["file_id"] and not inferred_columns["package_id"]:
        missing_requirements.append("download_url_or_identifier_columns")
    audit = {
        "source_key": adni_section(config)["source_key"],
        "acquisition_mode": str(acquisition_config(config)["mode"]),
        "manifest_path_private": controlled_display_path(manifest_path(config)),
        "manifest_columns": list(rows[0].keys()) if rows else [],
        "inferred_columns": inferred_columns,
        "cookie_session_status": cookie_session_status,
        "minimum_downloadable_records": minimum_downloadable_records,
        "matched_required_roles": matched_roles,
        "missing_required_roles_in_manifest": sorted(set(required_table_roles(config)) - set(matched_roles)),
        "missing_requirements": missing_requirements,
        "participant_level_values_redacted": True,
        "download_url_values_redacted": True,
    }
    audit.update(counts)
    return audit


def write_manifest_schema_audit(config: dict, audit: dict[str, object]) -> None:
    controlled_config = adni_section(config)
    private_root = resolve_configured_path(controlled_config["paths"]["private_root"])
    write_json(private_root / controlled_config["paths"]["manifest_schema_audit"], audit)


def validate_manifest_schema_audit(audit: dict[str, object]) -> None:
    missing_requirements = [str(value) for value in audit.get("missing_requirements", [])]
    if missing_requirements:
        raise RuntimeError(f"ADNI acquisition manifest failed schema requirements: {', '.join(missing_requirements)}")


def redacted_manifest_dry_run_summary(audit: dict[str, object]) -> dict[str, object]:
    inferred_columns = audit["inferred_columns"]
    return {
        "cookie_session_status": audit["cookie_session_status"],
        "acquisition_mode": audit["acquisition_mode"],
        "manifest_column_count": len(audit["manifest_columns"]),
        "record_count": audit["record_count"],
        "direct_download_record_count": audit["direct_download_record_count"],
        "identifier_request_record_count": audit["identifier_request_record_count"],
        "downloadable_record_count": audit["downloadable_record_count"],
        "matched_required_role_count": len(audit.get("matched_required_roles", [])),
        "missing_required_role_count": len(audit.get("missing_required_roles_in_manifest", [])),
        "inferred_column_counts": {
            semantic: len(columns)
            for semantic, columns in inferred_columns.items()
        },
        "missing_requirements": audit["missing_requirements"],
    }


def run_manifest_first_no_download_dry_run(config: dict) -> dict[str, object]:
    build_authenticated_session(config)
    rows = read_ida_export_manifest(config)
    audit = manifest_schema_audit(config, rows, "authenticated_cookie_session_valid")
    write_manifest_schema_audit(config, audit)
    validate_manifest_schema_audit(audit)
    return audit


def safe_manifest_filename(name_value: str, default_extension: str) -> str:
    name = Path(name_value).name.strip()
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if sanitized in {"", ".", ".."}:
        return ""
    if "." not in sanitized:
        sanitized = f"{sanitized}.{default_extension}"
    return sanitized


def build_identifier_download_url(config: dict, identifier_kind: str, identifier_value: str) -> str:
    templates = acquisition_config(config)["identifier_download_request_templates"]
    template = str(templates[identifier_kind])
    return template.format(identifier=identifier_value)


def role_for_candidate_text(candidate_text: str, config: dict) -> str:
    candidate_values = [candidate_text, *candidate_text.split()]
    for table_role, role_config in role_registry(config).items():
        for pattern_text in [str(value) for value in role_config["filename_patterns"]]:
            if any(re.search(pattern_text, value) for value in candidate_values):
                return table_role
    return ""


def role_for_manifest_record(row: dict[str, str], inferred_columns: dict[str, list[str]], config: dict) -> str:
    filename = first_nonempty_value(row, inferred_columns["filename"])
    source_url = direct_download_url(row, inferred_columns)
    candidate_text = " ".join([filename, source_url, row_text(row)])
    return role_for_candidate_text(candidate_text, config)


def manifest_first_asset_rows(config: dict, rows: list[dict[str, str]], audit: dict[str, object]) -> list[dict[str, str]]:
    validate_manifest_schema_audit(audit)
    inferred_columns = audit["inferred_columns"]
    acquisition = acquisition_config(config)
    default_extension = str(acquisition["filename_default_extension"])
    download_root = safe_relative_path(str(acquisition["participant_data_download_root"]))
    allow_identifier_download = bool(acquisition["allow_identifier_download_after_dry_run"])
    asset_rows: list[dict[str, str]] = []
    for record_index, row in enumerate(rows, start=1):
        source_url = direct_download_url(row, inferred_columns)
        request_note = "manifest_first:direct_download_url"
        if not source_url:
            identifier_kind, identifier_value = identifier_request_candidate(row, inferred_columns)
            if not identifier_kind or not identifier_value:
                continue
            if not allow_identifier_download:
                raise RuntimeError("ADNI manifest contains identifier-only records; run and review the no-download dry-run audit before enabling identifier downloads.")
            source_url = build_identifier_download_url(config, identifier_kind, identifier_value)
            request_note = f"manifest_first:{identifier_kind}_request"
        filename = safe_manifest_filename(first_nonempty_value(row, inferred_columns["filename"]), default_extension)
        if not filename:
            filename = f"adni_manifest_record_{record_index:06d}.{default_extension}"
        table_role = role_for_manifest_record(row, inferred_columns, config)
        if table_role:
            role_config = role_registry(config)[table_role]
            asset_uid = f"adni_{table_role}"
            asset_kind = str(role_config["asset_kind"])
            data_domain = str(role_config["data_domain"])
            modality = str(role_config["modality"])
            target_relative_path = str(role_config["target_relative_path"])
        else:
            asset_uid = f"adni_manifest_record_{record_index:06d}"
            asset_kind = "imaging_archive"
            data_domain = "mri_structural"
            modality = first_nonempty_value(row, inferred_columns["modality"]) or "manifest_declared"
            table_role = "ida_manifest_record"
            target_relative_path = (download_root / filename).as_posix()
        asset_rows.append(
            {
                "asset_uid": asset_uid,
                "include_flag": "yes",
                "transport_kind": "authenticated_http",
                "asset_kind": asset_kind,
                "data_domain": data_domain,
                "table_role": table_role,
                "modality": modality,
                "adni_cohort": first_nonempty_value(row, inferred_columns["phase"]) or "ADNI",
                "source_url": source_url,
                "source_path": "",
                "target_relative_path": target_relative_path,
                "participant_level_flag": "yes",
                "allowed_use": "validation_internal_only",
                "notes": request_note,
            }
        )
    return asset_rows


def collection_first_config(config: dict) -> dict:
    collection_config = acquisition_config(config)["collection_first"]
    profile_name = str(collection_config.get("execution_profile", ""))
    profile_registry = collection_config.get("profiles", {})
    if not profile_name:
        return collection_config
    if profile_name not in profile_registry:
        raise ValueError(f"Unknown ADNI collection-first execution profile: {profile_name}")
    merged_config = dict(collection_config)
    merged_config.update(profile_registry[profile_name])
    merged_config["execution_profile"] = profile_name
    merged_config["profiles"] = profile_registry
    return merged_config


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def redacted_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return value.split("?", 1)[0]
    query_marker = "?<redacted>" if parsed.query else ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{query_marker}"


def page_title(page_text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", page_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))).strip()


def strip_html_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def links_from_html(page_url: str, page_text: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    anchor_pattern = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<text>.*?)</a>", flags=re.IGNORECASE | re.DOTALL)
    href_pattern = re.compile(r"\bhref=[\"']([^\"']+)[\"']", flags=re.IGNORECASE)
    for match in anchor_pattern.finditer(page_text):
        href_match = href_pattern.search(match.group("attrs"))
        if href_match is None:
            continue
        href = html.unescape(href_match.group(1))
        parsed = urlparse(href)
        if parsed.scheme in {"mailto", "javascript"}:
            continue
        links.append(
            {
                "text": strip_html_text(match.group("text")),
                "href": urljoin(page_url, href),
            }
        )
    return links


def forms_from_html(page_url: str, page_text: str) -> list[dict[str, object]]:
    forms: list[dict[str, object]] = []
    form_pattern = re.compile(r"<form\b(?P<attrs>[^>]*)>(?P<body>.*?)</form>", flags=re.IGNORECASE | re.DOTALL)
    standalone_form_pattern = re.compile(r"<form\b(?P<attrs>[^>]*)>", flags=re.IGNORECASE | re.DOTALL)
    action_pattern = re.compile(r"\baction=[\"']([^\"']*)[\"']", flags=re.IGNORECASE)
    method_pattern = re.compile(r"\bmethod=[\"']([^\"']*)[\"']", flags=re.IGNORECASE)
    input_name_pattern = re.compile(r"<input\b[^>]*\bname=[\"']([^\"']+)[\"'][^>]*>", flags=re.IGNORECASE)
    for match in form_pattern.finditer(page_text):
        attrs = match.group("attrs")
        action_match = action_pattern.search(attrs)
        method_match = method_pattern.search(attrs)
        action = urljoin(page_url, html.unescape(action_match.group(1))) if action_match else page_url
        input_names = [html.unescape(value) for value in input_name_pattern.findall(match.group("body"))]
        forms.append(
            {
                "method": html.unescape(method_match.group(1)).upper() if method_match else "GET",
                "action_hash": short_hash(action),
                "action_redacted": redacted_url(action),
                "input_name_hashes": [short_hash(value) for value in input_names],
                "input_name_samples": input_names[:20],
            }
        )
    if forms:
        return forms
    for match in standalone_form_pattern.finditer(page_text):
        attrs = match.group("attrs")
        action_match = action_pattern.search(attrs)
        method_match = method_pattern.search(attrs)
        action = urljoin(page_url, html.unescape(action_match.group(1))) if action_match else page_url
        forms.append(
            {
                "method": html.unescape(method_match.group(1)).upper() if method_match else "GET",
                "action_hash": short_hash(action),
                "action_redacted": redacted_url(action),
                "input_name_hashes": [],
                "input_name_samples": [],
            }
        )
    return forms


def button_texts_from_html(page_text: str) -> list[str]:
    button_texts: list[str] = []
    button_pattern = re.compile(r"<button\b[^>]*>(.*?)</button>", flags=re.IGNORECASE | re.DOTALL)
    input_button_pattern = re.compile(r"<input\b[^>]*\b(?:type=[\"'](?:button|submit)[\"'])[^>]*>", flags=re.IGNORECASE | re.DOTALL)
    value_pattern = re.compile(r"\bvalue=[\"']([^\"']+)[\"']", flags=re.IGNORECASE)
    for match in button_pattern.finditer(page_text):
        text = strip_html_text(match.group(1))
        if text:
            button_texts.append(text)
    for match in input_button_pattern.finditer(page_text):
        value_match = value_pattern.search(match.group(0))
        if value_match:
            button_texts.append(html.unescape(value_match.group(1)).strip())
    return button_texts


def link_matches_patterns(link: dict[str, str], patterns: list[str]) -> bool:
    candidate_text = f"{link['text']} {link['href']}"
    return any(re.search(str(pattern), candidate_text) for pattern in patterns)


def collection_candidate_links(config: dict, links: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    pattern_config = collection_first_config(config)["candidate_text_patterns"]
    candidates: dict[str, list[dict[str, str]]] = {
        "csv": [],
        "advanced_download": [],
        "one_click_download": [],
        "add_to_collection": [],
        "data_collections": [],
        "collection": [],
        "download": [],
    }
    for link in links:
        redacted_link = {
            "text_hash": short_hash(link["text"]),
            "text_sample": link["text"][:80],
            "href_hash": short_hash(link["href"]),
            "href_redacted": redacted_url(link["href"]),
        }
        for candidate_kind, patterns in pattern_config.items():
            if link_matches_patterns(link, [str(value) for value in patterns]):
                candidates[candidate_kind].append(redacted_link)
    return candidates


def page_inventory_record(config: dict, target_config: dict, status_code: int, page_text: str, page_url: str) -> dict[str, object]:
    links = links_from_html(page_url, page_text)
    button_texts = button_texts_from_html(page_text)
    return {
        "target_key": str(target_config["target_key"]),
        "target_kind": str(target_config["target_kind"]),
        "url": redacted_url(page_url),
        "url_hash": short_hash(page_url),
        "status_code": int(status_code),
        "title": page_title(page_text),
        "forms": forms_from_html(page_url, page_text),
        "link_text_hashes": [short_hash(link["text"]) for link in links if link["text"]],
        "visible_link_text_summary": [link["text"][:80] for link in links if link["text"]][:50],
        "button_text_hashes": [short_hash(text) for text in button_texts],
        "visible_button_text_summary": [text[:80] for text in button_texts][:50],
        "candidate_links": collection_candidate_links(config, links),
    }


def collect_ida_navigation_inventory_requests(config: dict, session: requests.Session) -> tuple[list[dict[str, object]], dict[str, list[dict[str, str]]]]:
    inventory: list[dict[str, object]] = []
    raw_candidates = {"csv": [], "advanced_download": [], "one_click_download": [], "collection": [], "download": []}
    for target_config in collection_first_config(config)["navigation_targets"]:
        page_url = str(target_config["url"])
        response = session.get(page_url)
        page_text = response.text
        record = page_inventory_record(config, target_config, int(response.status_code), page_text, page_url)
        inventory.append(record)
        links = links_from_html(page_url, page_text)
        pattern_config = collection_first_config(config)["candidate_text_patterns"]
        for candidate_kind in raw_candidates:
            raw_candidates[candidate_kind].extend(
                [
                    {**link, "source_target_key": str(target_config["target_key"])}
                    for link in links
                    if link_matches_patterns(link, [str(value) for value in pattern_config[candidate_kind]])
                ]
            )
        response.close()
    return inventory, raw_candidates


def playwright_available() -> bool:
    return importlib.util.find_spec("playwright") is not None and importlib.util.find_spec("playwright.sync_api") is not None


def cookie_policy_config(config: dict) -> dict:
    auth_config = adni_section(config)["sync"]["authenticated_http"]
    return auth_config.get("cookie_policy", {})


def configured_cookie_jar_paths(config: dict) -> list[Path]:
    auth_config = adni_section(config)["sync"]["authenticated_http"]
    path_values = [str(auth_config["cookie_jar_path"])]
    path_values.extend(str(value) for value in auth_config.get("candidate_cookie_jar_paths", []))
    resolved_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for path_value in path_values:
        path = resolve_configured_path(path_value)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        resolved_paths.append(path)
    return resolved_paths


def allowed_cookie_domains(config: dict) -> set[str]:
    auth_config = adni_section(config)["sync"]["authenticated_http"]
    return {str(value).lower() for value in auth_config.get("cookie_domain_allowlist", [])}


def cookie_domain_allowed(cookie_domain: str, allowed_domains: set[str]) -> bool:
    normalized_domain = cookie_domain.lower().lstrip(".")
    return any(
        normalized_domain == allowed_domain.lstrip(".")
        or normalized_domain.endswith(f".{allowed_domain.lstrip('.')}")
        for allowed_domain in allowed_domains
    )


def zero_expiry_session_cookie_names(config: dict) -> set[str]:
    cookie_policy = cookie_policy_config(config)
    return {str(value) for value in cookie_policy.get("zero_expiry_session_cookie_names", [])}


def normalized_ida_cookie(config: dict, cookie: object) -> object:
    normalized_cookie = copy.copy(cookie)
    if (
        normalized_cookie.name in zero_expiry_session_cookie_names(config)
        and normalized_cookie.expires is not None
        and int(normalized_cookie.expires) <= 0
    ):
        normalized_cookie.expires = None
        normalized_cookie.discard = True
    return normalized_cookie


def load_browser_cookie_jar_from_path(config: dict, cookie_jar_path: Path) -> MozillaCookieJar:
    auth_config = adni_section(config)["sync"]["authenticated_http"]
    cookie_jar_format = str(auth_config["cookie_jar_format"])
    if cookie_jar_format != "netscape_mozilla":
        raise ValueError(f"Unsupported ADNI browser cookie jar format: {cookie_jar_format}")
    if not cookie_jar_path.exists():
        raise FileNotFoundError(f"ADNI browser cookie jar is missing at configured private path: {controlled_display_path(cookie_jar_path)}")
    cookie_jar = MozillaCookieJar(str(cookie_jar_path))
    cookie_policy = auth_config.get("cookie_policy", {})
    cookie_jar.load(
        ignore_discard=True,
        ignore_expires=bool(cookie_policy.get("accept_zero_expiry_session_cookies", False)),
    )
    if not cookie_jar:
        raise RuntimeError("ADNI browser cookie jar is empty or all cookies are expired; re-export cookies from a logged-in IDA browser session.")
    return cookie_jar


def ida_relevant_cookies_for_path(config: dict, cookie_jar_path: Path) -> list[object]:
    allowed_domains = allowed_cookie_domains(config)
    cookies = []
    for cookie in load_browser_cookie_jar_from_path(config, cookie_jar_path):
        if allowed_domains and not cookie_domain_allowed(cookie.domain, allowed_domains):
            continue
        normalized_cookie = normalized_ida_cookie(config, cookie)
        if normalized_cookie.is_expired():
            continue
        cookies.append(normalized_cookie)
    return cookies


def ida_relevant_cookies(config: dict) -> list[object]:
    return ida_relevant_cookies_for_path(config, configured_cookie_jar_paths(config)[0])


def playwright_cookie_payload(config: dict, cookie_jar_path: Path | None = None) -> list[dict[str, object]]:
    cookies: list[dict[str, object]] = []
    relevant_cookies = ida_relevant_cookies_for_path(config, cookie_jar_path) if cookie_jar_path else ida_relevant_cookies(config)
    for cookie in relevant_cookies:
        cookie_payload: dict[str, object] = {
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain,
            "path": cookie.path,
            "secure": bool(cookie.secure),
            "httpOnly": False,
        }
        if cookie.expires is not None:
            cookie_payload["expires"] = int(cookie.expires)
        cookies.append(cookie_payload)
    return cookies


def collect_ida_navigation_inventory_playwright(config: dict, cookie_jar_path: Path | None = None) -> tuple[list[dict[str, object]], dict[str, list[dict[str, str]]]]:
    playwright_sync = importlib.import_module("playwright.sync_api")
    inventory: list[dict[str, object]] = []
    raw_candidates = {"csv": [], "advanced_download": [], "one_click_download": [], "collection": [], "download": []}
    private_root = resolve_configured_path(adni_section(config)["paths"]["private_root"])
    screenshot_path = private_root / str(collection_first_config(config)["playwright_debug_screenshot"])
    ensure_directory(screenshot_path.parent)
    playwright_config = collection_first_config(config)
    network_policy = adni_section(config)["sync"]["authenticated_http"].get("network_policy", {})
    launch_args = [str(value) for value in playwright_config.get("playwright_launch_args", [])]
    browser_env = None
    if not bool(network_policy.get("use_environment_proxy", True)):
        proxy_environment_names = {str(value) for value in playwright_config.get("proxy_environment_variables", [])}
        browser_env = {
            key: value
            for key, value in os.environ.items()
            if key not in proxy_environment_names
        }
    with playwright_sync.sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=bool(playwright_config["headless"]),
            args=launch_args,
            env=browser_env,
        )
        context = browser.new_context()
        context.add_cookies(playwright_cookie_payload(config, cookie_jar_path))
        page = context.new_page()
        for target_config in collection_first_config(config)["navigation_targets"]:
            page_url = str(target_config["url"])
            response = page.goto(page_url, wait_until="networkidle")
            page_text = page.content()
            status_code = int(response.status) if response is not None else 0
            record = page_inventory_record(config, target_config, status_code, page_text, page_url)
            record["browser_engine"] = "playwright"
            record["title"] = page.title() or str(record["title"])
            inventory.append(record)
            links = links_from_html(page_url, page_text)
            pattern_config = collection_first_config(config)["candidate_text_patterns"]
            for candidate_kind in raw_candidates:
                raw_candidates[candidate_kind].extend(
                    [
                        {**link, "source_target_key": str(target_config["target_key"])}
                        for link in links
                        if link_matches_patterns(link, [str(value) for value in pattern_config[candidate_kind]])
                    ]
                )
        page.screenshot(path=str(screenshot_path), full_page=True)
        context.close()
        browser.close()
    return inventory, raw_candidates


def browser_cookie_jar_metadata(config: dict) -> dict[str, object]:
    auth_config = adni_section(config)["sync"]["authenticated_http"]
    allowed_session_cookie_names = {
        str(value)
        for value in auth_config.get("cookie_policy", {}).get("allowed_session_cookie_names", [])
    }
    candidate_metadata = []
    for cookie_jar_path in configured_cookie_jar_paths(config):
        cookie_rows = []
        loaded_cookie_count = 0
        if cookie_jar_path.exists():
            loaded_cookie_count = len(list(load_browser_cookie_jar_from_path(config, cookie_jar_path)))
            for cookie in ida_relevant_cookies_for_path(config, cookie_jar_path):
                cookie_rows.append(
                    {
                        "name": cookie.name,
                        "domain": cookie.domain,
                        "path": cookie.path,
                        "secure": bool(cookie.secure),
                        "expires_present": cookie.expires is not None,
                        "expired": bool(cookie.is_expired()),
                        "session_cookie_candidate": cookie.name in allowed_session_cookie_names,
                    }
                )
        candidate_metadata.append(
            {
                "path_private": controlled_display_path(cookie_jar_path),
                "exists": cookie_jar_path.exists(),
                "loaded_cookie_count": loaded_cookie_count,
                "ida_relevant_cookie_count": len(cookie_rows),
                "present_session_cookie_names": sorted({row["name"] for row in cookie_rows if row["session_cookie_candidate"]}),
                "cookies": cookie_rows,
            }
        )
    primary_metadata = candidate_metadata[0] if candidate_metadata else {
        "loaded_cookie_count": 0,
        "ida_relevant_cookie_count": 0,
        "present_session_cookie_names": [],
        "cookies": [],
    }
    return {
        "loaded_cookie_count": primary_metadata["loaded_cookie_count"],
        "ida_relevant_cookie_count": primary_metadata["ida_relevant_cookie_count"],
        "cookie_domain_allowlist": sorted(allowed_cookie_domains(config)),
        "zero_expiry_session_cookie_names": sorted(zero_expiry_session_cookie_names(config)),
        "required_session_cookie_names": sorted(allowed_session_cookie_names),
        "present_session_cookie_names": primary_metadata["present_session_cookie_names"],
        "cookies": primary_metadata["cookies"],
        "candidate_cookie_jars": candidate_metadata,
        "cookie_values_redacted": True,
    }


def inventory_login_gate_status(config: dict, inventory: list[dict[str, object]]) -> dict[str, object]:
    auth_config = adni_section(config)["sync"]["authenticated_http"]
    login_patterns = [re.compile(str(value)) for value in auth_config.get("login_page_text_patterns", [])]
    login_title_patterns = [re.compile(r"(?i)\bmust\s+be\s+logged\s+in\b"), re.compile(r"(?i)\blogin\b")]
    login_like_pages: list[dict[str, object]] = []
    for record in inventory:
        title = str(record.get("title", ""))
        visible_text = " ".join(str(value) for value in record.get("visible_link_text_summary", []))
        visible_text += " " + " ".join(str(value) for value in record.get("visible_button_text_summary", []))
        form_input_names = " ".join(
            " ".join(str(value) for value in form.get("input_name_samples", []))
            for form in record.get("forms", [])
            if isinstance(form, dict)
        )
        candidate_text = " ".join([title, visible_text, form_input_names])
        if any(pattern.search(candidate_text) for pattern in login_title_patterns + login_patterns):
            login_like_pages.append(
                {
                    "target_key": record.get("target_key"),
                    "title": title,
                    "status_code": record.get("status_code"),
                }
            )
    return {
        "login_gate_detected": bool(login_like_pages),
        "login_like_pages": login_like_pages,
    }


def playwright_next_action_report(config: dict, inventory: list[dict[str, object]], missing_entry: str) -> dict[str, object]:
    login_gate_status = inventory_login_gate_status(config, inventory)
    required_next_action = "Run a Playwright trace/navigation probe to resolve the authenticated IDA Data Collections or Advanced Download selector path."
    reported_missing_entry = missing_entry
    if bool(login_gate_status["login_gate_detected"]):
        reported_missing_entry = "authenticated_playwright_session"
        required_next_action = "Re-export the IDA browser cookie jar from a logged-in browser session with session cookies included, then rerun the Playwright collection-first dry run."
    return {
        "source_key": adni_section(config)["source_key"],
        "acquisition_mode": str(acquisition_config(config)["mode"]),
        "missing_entry": reported_missing_entry,
        "raw_missing_entry": missing_entry,
        "login_gate_status": login_gate_status,
        "cookie_jar_metadata": browser_cookie_jar_metadata(config),
        "current_page_titles": [str(record.get("title", "")) for record in inventory],
        "visible_menu_link_summary": [
            {
                "target_key": record.get("target_key"),
                "visible_link_text_summary": record.get("visible_link_text_summary", []),
                "visible_button_text_summary": record.get("visible_button_text_summary", []),
            }
            for record in inventory
        ],
        "required_next_action": required_next_action,
        "password_login_fallback": "disabled",
    }


def write_collection_page_inventory(config: dict, inventory: list[dict[str, object]]) -> None:
    controlled_config = adni_section(config)
    private_root = resolve_configured_path(controlled_config["paths"]["private_root"])
    write_json(private_root / controlled_config["paths"]["collection_page_inventory"], {"pages": inventory})


def write_collection_next_action_report(config: dict, report: dict[str, object]) -> None:
    controlled_config = adni_section(config)
    private_root = resolve_configured_path(controlled_config["paths"]["private_root"])
    write_json(private_root / controlled_config["paths"]["collection_next_action_report"], report)


def planned_download_fields() -> list[str]:
    return [
        "source_target_key",
        "source_kind",
        "link_text_hash",
        "link_text_sample",
        "download_url_hash",
        "download_url_redacted",
        "filename",
        "planned_status",
    ]


def manifest_output_fields(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "source_kind",
        "download_url",
        "file_id",
        "package_id",
        "modality",
        "visit",
        "subject",
        "phase",
        "filename",
        "size",
        "source_target_key",
        "source_link_text_hash",
    ]
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_generated_manifest(config: dict, rows: list[dict[str, str]]) -> None:
    path = manifest_path(config)
    ensure_directory(path.parent)
    write_tsv(path, manifest_output_fields(rows), rows)


def parse_delimited_manifest_text(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header_line = lines[0]
    delimiter = "\t" if header_line.count("\t") > header_line.count(",") else ","
    if delimiter not in header_line:
        return []
    reader = csv.DictReader(lines, delimiter=delimiter)
    if not reader.fieldnames:
        return []
    return [
        {str(key): "" if value is None else str(value) for key, value in row.items()}
        for row in reader
    ]


def row_text(row: dict[str, str]) -> str:
    return " ".join(str(value) for value in row.values())


def smoke_row_rank(config: dict, row: dict[str, str]) -> tuple[int, str]:
    text = row_text(row)
    denied_patterns = [str(value) for value in collection_first_config(config)["denied_filename_patterns"]]
    if bool(collection_first_config(config)["deny_raw_dicom_or_full_image"]) and any(re.search(pattern, text) for pattern in denied_patterns):
        return (100, text)
    preferred_patterns = [str(value) for value in collection_first_config(config)["preferred_record_patterns"]]
    if any(re.search(pattern, text) for pattern in preferred_patterns):
        return (0, text)
    if re.search(r"(?i)\b(metadata|clinical|csv|study\s*data)\b", text):
        return (1, text)
    if re.search(r"(?i)\b(mri|pet|image|dicom|nifti)\b", text):
        return (20, text)
    return (10, text)


def filter_smoke_manifest_rows(config: dict, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    max_records = int(collection_first_config(config)["max_records"])
    denied_patterns = [str(value) for value in collection_first_config(config)["denied_filename_patterns"]]
    filtered_rows = [
        row
        for row in rows
        if not bool(collection_first_config(config)["deny_raw_dicom_or_full_image"])
        or not any(re.search(pattern, row_text(row)) for pattern in denied_patterns)
    ]
    ranked_rows = sorted(filtered_rows, key=lambda row: smoke_row_rank(config, row))
    return ranked_rows[:max_records]


def link_denied_for_download_manifest(config: dict, link: dict[str, str]) -> bool:
    candidate_text = f"{link.get('text', '')} {link.get('href', '')}"
    denied_patterns = [str(value) for value in collection_first_config(config).get("denied_link_patterns", [])]
    return any(re.search(pattern, candidate_text) for pattern in denied_patterns)


def link_looks_like_direct_download(config: dict, link: dict[str, str]) -> bool:
    candidate_text = f"{link.get('text', '')} {link.get('href', '')}"
    parsed = urlparse(str(link.get("href", "")))
    path_name = Path(parsed.path).name
    if role_for_candidate_text(candidate_text, config):
        return True
    denied_path_names = {str(value).lower() for value in collection_first_config(config).get("denied_navigation_path_names", [])}
    if path_name.lower() in denied_path_names:
        return False
    file_patterns = [str(value) for value in collection_first_config(config).get("direct_download_file_patterns", [])]
    if any(re.search(pattern, candidate_text) for pattern in file_patterns):
        return True
    path_patterns = [str(value) for value in collection_first_config(config).get("direct_download_path_patterns", [])]
    return any(re.search(pattern, parsed.path) for pattern in path_patterns)


def direct_download_link_manifest_rows(config: dict, links: list[dict[str, str]], source_kind: str) -> list[dict[str, str]]:
    default_extension = str(acquisition_config(config)["filename_default_extension"])
    rows: list[dict[str, str]] = []
    for link in links:
        if link_denied_for_download_manifest(config, link) or not link_looks_like_direct_download(config, link):
            continue
        filename = filename_from_link(link, default_extension)
        rows.append(
            {
                "source_kind": source_kind,
                "download_url": link["href"],
                "filename": filename,
                "source_target_key": str(link.get("source_target_key", "")),
                "source_link_text_hash": short_hash(link.get("text", "")),
                "source_link_text_sample": str(link.get("text", ""))[:80],
            }
        )
    return filter_smoke_manifest_rows(config, rows)


def csv_collection_manifest_rows(config: dict, session: requests.Session, csv_links: list[dict[str, str]]) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for link in csv_links:
        if link_denied_for_download_manifest(config, link):
            continue
        response = session.get(link["href"])
        if int(response.status_code) >= 400:
            response.close()
            continue
        rows = parse_delimited_manifest_text(response.text)
        response.close()
        rows = filter_smoke_manifest_rows(config, rows)
        if rows and manifest_record_counts(rows, infer_manifest_columns(rows))["downloadable_record_count"] > 0:
            manifest_rows.extend(
                [
                    {
                        **row,
                        "source_kind": "collection_csv",
                        "source_target_key": str(link["source_target_key"]),
                        "source_link_text_hash": short_hash(link["text"]),
                    }
                    for row in rows
                ]
            )
            continue
        direct_rows = direct_download_link_manifest_rows(config, [link], "direct_csv_download")
        if direct_rows:
            manifest_rows.extend(direct_rows)
    return filter_smoke_manifest_rows(config, manifest_rows)


def filename_from_link(link: dict[str, str], default_extension: str) -> str:
    path_name = Path(urlparse(link["href"]).path).name
    return safe_manifest_filename(path_name or link["text"], default_extension) or f"planned_download.{default_extension}"


def advanced_download_manifest_rows(config: dict, session: requests.Session, advanced_links: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    manifest_rows: list[dict[str, str]] = []
    planned_rows: list[dict[str, str]] = []
    default_extension = str(acquisition_config(config)["filename_default_extension"])
    max_records = int(collection_first_config(config)["max_records"])
    for link in advanced_links:
        response = session.get(link["href"])
        if int(response.status_code) >= 400:
            response.close()
            continue
        page_links = links_from_html(link["href"], response.text)
        response.close()
        download_links = [
            page_link
            for page_link in page_links
            if link_matches_patterns(page_link, [str(value) for value in collection_first_config(config)["candidate_text_patterns"]["download"]])
        ]
        for page_link in download_links:
            filename = filename_from_link(page_link, default_extension)
            planned_rows.append(
                {
                    "source_target_key": str(link["source_target_key"]),
                    "source_kind": "advanced_download_page",
                    "link_text_hash": short_hash(page_link["text"]),
                    "link_text_sample": page_link["text"][:80],
                    "download_url_hash": short_hash(page_link["href"]),
                    "download_url_redacted": redacted_url(page_link["href"]),
                    "filename": filename,
                    "planned_status": "not_downloaded",
                }
            )
            manifest_rows.append(
                {
                    "source_kind": "advanced_download_page",
                    "download_url": page_link["href"],
                    "filename": filename,
                    "source_target_key": str(link["source_target_key"]),
                    "source_link_text_hash": short_hash(page_link["text"]),
                }
            )
            if len(manifest_rows) >= max_records:
                return filter_smoke_manifest_rows(config, manifest_rows), planned_rows
    return filter_smoke_manifest_rows(config, manifest_rows), planned_rows


def write_planned_downloads(config: dict, rows: list[dict[str, str]]) -> None:
    controlled_config = adni_section(config)
    private_root = resolve_configured_path(controlled_config["paths"]["private_root"])
    write_tsv(private_root / controlled_config["paths"]["planned_downloads"], planned_download_fields(), rows)


def deduplicate_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduplicated_rows: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for row in rows:
        key = row.get("table_role") or row.get("download_url") or row.get("file_id") or row.get("package_id") or row_text(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated_rows.append(row)
    return deduplicated_rows


def arc_builder_config(config: dict) -> dict[str, object]:
    return collection_first_config(config).get("arc_builder", {})


def arc_builder_enabled(config: dict) -> bool:
    return bool(arc_builder_config(config).get("enabled", False))


def arc_builder_url(config: dict, path_key: str) -> str:
    builder_config = arc_builder_config(config)
    return urljoin(str(builder_config["base_url"]), str(builder_config[path_key]))


def arc_download_url(config: dict, base_path_key: str, path_name: str) -> str:
    builder_config = arc_builder_config(config)
    return urljoin(str(builder_config["base_url"]), urljoin(str(builder_config[base_path_key]), path_name))


def arc_project(config: dict) -> str:
    return str(collection_first_config(config)["project"])


def xml_root_from_response(response: requests.Response, context: str) -> ET.Element:
    response.raise_for_status()
    root = ET.fromstring(response.text.strip())
    if root.tag in {"invalid-search-id", "error"}:
        raise RuntimeError(f"ADNI ARC Builder API returned {root.tag} for {context}.")
    return root


def arc_get_xml(config: dict, session: requests.Session, path_key: str, params: dict[str, object]) -> ET.Element:
    response = session.get(arc_builder_url(config, path_key), params=params)
    root = xml_root_from_response(response, path_key)
    response.close()
    return root


def arc_post_xml(config: dict, session: requests.Session, path_key: str, payload: dict[str, object]) -> ET.Element:
    response = session.post(arc_builder_url(config, path_key), data=payload)
    root = xml_root_from_response(response, path_key)
    response.close()
    return root


def arc_path_name_from_root(root: ET.Element, context: str) -> str:
    path_element = root.find(".//path")
    path_name = path_element.attrib.get("name", "") if path_element is not None else ""
    if not path_name:
        raise RuntimeError(f"ADNI ARC Builder API did not return a download path for {context}.")
    return path_name


def arc_selected_search_id(config: dict, session: requests.Session) -> str:
    root = arc_get_xml(
        config,
        session,
        "get_searches_path",
        {"project": arc_project(config)},
    )
    patterns = [re.compile(str(value)) for value in arc_builder_config(config).get("selected_search_name_patterns", [])]
    pattern_candidates: list[str] = []
    generic_candidates: list[str] = []
    for search_element in root.iter("search"):
        search_id = str(search_element.attrib.get("id", ""))
        if not search_id or search_id == "-1":
            continue
        generic_candidates.append(search_id)
        search_name = str(search_element.attrib.get("name", ""))
        if patterns and any(pattern.search(search_name) for pattern in patterns):
            if str(search_element.attrib.get("loaded", "")).lower() == "true":
                return search_id
            pattern_candidates.append(search_id)
    if pattern_candidates:
        return pattern_candidates[0]
    if generic_candidates:
        return generic_candidates[0]
    raise RuntimeError("ADNI ARC Builder did not expose a usable cohort search id.")


def arc_variable_category_index(config: dict, session: requests.Session) -> dict[str, dict[str, str]]:
    root = arc_get_xml(
        config,
        session,
        "get_variables_path",
        {"project": arc_project(config)},
    )
    categories: dict[str, dict[str, str]] = {}
    for group in root.iter("group"):
        group_name = str(group.attrib.get("name", "")).strip()
        if not group_name:
            continue
        first_variable = next(group.iter("variable"), None)
        if first_variable is None:
            continue
        categories[group_name] = {
            "name": group_name,
            "description": str(group.attrib.get("description", "")),
            "var_id": str(first_variable.attrib.get("id", "")),
        }
    return categories


def role_manifest_filename(config: dict, table_role: str) -> str:
    return Path(str(role_registry(config)[table_role]["target_relative_path"])).name


def arc_manifest_row(
    config: dict,
    table_role: str,
    source_kind: str,
    source_target_key: str,
    download_url: str,
    filename: str,
    file_id: str,
    size: str,
    note_text: str,
) -> dict[str, str]:
    role_config = role_registry(config)[table_role]
    return {
        "source_kind": source_kind,
        "download_url": download_url,
        "file_id": file_id,
        "package_id": "",
        "modality": str(role_config["modality"]),
        "visit": "",
        "subject": "",
        "phase": "ADNI",
        "filename": filename,
        "size": size,
        "source_target_key": source_target_key,
        "source_link_text_hash": short_hash(note_text),
        "source_link_text_sample": note_text[:80],
        "table_role": table_role,
    }


def arc_table_manifest_rows(config: dict, session: requests.Session, search_id: str) -> list[dict[str, str]]:
    categories = arc_variable_category_index(config, session)
    rows: list[dict[str, str]] = []
    role_categories = arc_builder_config(config).get("role_table_categories", {})
    for table_role in required_table_roles(config):
        for category_name in [str(value) for value in role_categories.get(table_role, [])]:
            category = categories.get(category_name)
            if not category or not category["var_id"]:
                continue
            root = arc_post_xml(
                config,
                session,
                "download_tables_path",
                {
                    "project": arc_project(config),
                    "searchId": search_id,
                    "name": table_role,
                    "varId": category["var_id"],
                },
            )
            path_name = arc_path_name_from_root(root, f"table_role={table_role}")
            rows.append(
                arc_manifest_row(
                    config=config,
                    table_role=table_role,
                    source_kind="arc_builder_download_tables",
                    source_target_key="arc_builder_table_api",
                    download_url=arc_download_url(config, "search_download_base_path", path_name),
                    filename=role_manifest_filename(config, table_role),
                    file_id=category["var_id"],
                    size="",
                    note_text=category_name,
                )
            )
            break
    return rows


def study_file_matches_role(item: ET.Element, category_name: str, role_config: dict[str, object]) -> bool:
    if str(role_config.get("type", "")) and str(item.attrib.get("type", "")) != str(role_config["type"]):
        return False
    if category_name.startswith("_") and not bool(role_config.get("allow_archive", False)):
        return False
    item_name = str(item.attrib.get("name", ""))
    return any(re.search(str(pattern), item_name) for pattern in role_config.get("item_name_patterns", []))


def arc_study_file_manifest_rows(config: dict, session: requests.Session) -> list[dict[str, str]]:
    root = arc_get_xml(
        config,
        session,
        "get_study_categories_path",
        {"project": arc_project(config)},
    )
    rows: list[dict[str, str]] = []
    matched_roles: set[str] = set()
    role_patterns = arc_builder_config(config).get("role_study_file_patterns", {})
    for category in root.findall(".//category"):
        category_name = str(category.attrib.get("name", "")).strip()
        for item in category.findall(".//item"):
            for table_role in required_table_roles(config):
                if table_role in matched_roles:
                    continue
                role_pattern_config = role_patterns.get(table_role)
                if not role_pattern_config or not study_file_matches_role(item, category_name, role_pattern_config):
                    continue
                file_id = str(item.attrib.get("id", ""))
                root = arc_post_xml(
                    config,
                    session,
                    "get_study_data_path",
                    {
                        "project": arc_project(config),
                        "fileId": file_id,
                    },
                )
                path_name = arc_path_name_from_root(root, f"study_file_role={table_role}")
                rows.append(
                    arc_manifest_row(
                        config=config,
                        table_role=table_role,
                        source_kind="arc_builder_study_file",
                        source_target_key="arc_builder_study_file_api",
                        download_url=arc_download_url(config, "study_download_base_path", path_name),
                        filename=role_manifest_filename(config, table_role),
                        file_id=file_id,
                        size=str(item.attrib.get("fileSize", "")),
                        note_text=str(item.attrib.get("name", table_role)),
                    )
                )
                matched_roles.add(table_role)
    return rows


def arc_builder_manifest_rows(config: dict, session: requests.Session) -> list[dict[str, str]]:
    if not arc_builder_enabled(config):
        return []
    search_id = arc_selected_search_id(config, session)
    rows = arc_table_manifest_rows(config, session, search_id)
    rows.extend(arc_study_file_manifest_rows(config, session))
    return filter_smoke_manifest_rows(config, deduplicate_manifest_rows(rows))


def collection_first_manifest_rows(config: dict, session: requests.Session, raw_candidates: dict[str, list[dict[str, str]]]) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    arc_rows = arc_builder_manifest_rows(config, session)
    csv_rows = csv_collection_manifest_rows(config, session, raw_candidates["csv"])
    advanced_rows, planned_rows = advanced_download_manifest_rows(config, session, raw_candidates["advanced_download"])
    direct_rows = direct_download_link_manifest_rows(config, raw_candidates["download"], "direct_download_link")
    combined_rows = filter_smoke_manifest_rows(config, deduplicate_manifest_rows(arc_rows + csv_rows + advanced_rows + direct_rows))
    if combined_rows:
        source_kinds = sorted({row["source_kind"] for row in combined_rows})
        return combined_rows, planned_rows, "+".join(source_kinds)
    return [], planned_rows, "collection_csv_or_advanced_download"


def run_collection_first_no_download_dry_run(config: dict) -> dict[str, object]:
    session = build_authenticated_session(config)
    active_cookie_jar_path = getattr(session, "_adni_cookie_jar_path", None)
    inventory, raw_candidates = collect_ida_navigation_inventory_requests(config, session)
    manifest_rows, planned_rows, source_kind = collection_first_manifest_rows(config, session, raw_candidates)
    browser_engine_used = "requests"
    browser_engine = str(collection_first_config(config).get("browser_engine", "auto"))
    if not manifest_rows and browser_engine in {"auto", "playwright"} and playwright_available():
        inventory, raw_candidates = collect_ida_navigation_inventory_playwright(config, active_cookie_jar_path)
        manifest_rows, planned_rows, source_kind = collection_first_manifest_rows(config, session, raw_candidates)
        browser_engine_used = "playwright"
    write_collection_page_inventory(config, inventory)
    write_planned_downloads(config, planned_rows)
    if not manifest_rows:
        report = playwright_next_action_report(config, inventory, source_kind)
        report["playwright_available"] = playwright_available()
        report["browser_engine_used"] = browser_engine_used
        write_collection_next_action_report(config, report)
        raise RuntimeError(f"ADNI collection-first automation could not locate a usable {report['missing_entry']} entry; next-action report was written.")
    write_generated_manifest(config, manifest_rows)
    manifest_audit = run_manifest_first_no_download_dry_run(config)
    return {
        "source_key": adni_section(config)["source_key"],
        "acquisition_mode": "collection_first",
        "execution_profile": str(collection_first_config(config).get("execution_profile", "")),
        "build_manifest_only": bool(collection_first_config(config)["build_manifest_only"]),
        "download_enabled": bool(collection_first_config(config)["download_enabled"]),
        "browser_engine_used": browser_engine_used,
        "page_count": len(inventory),
        "generated_manifest_path_private": controlled_display_path(manifest_path(config)),
        "generated_manifest_record_count": len(manifest_rows),
        "planned_download_count": len(planned_rows),
        "manifest_source_kind": source_kind,
        "manifest_schema_audit": redacted_manifest_dry_run_summary(manifest_audit),
    }


def redacted_collection_first_summary(audit: dict[str, object]) -> dict[str, object]:
    return {
        "acquisition_mode": audit["acquisition_mode"],
        "execution_profile": audit["execution_profile"],
        "build_manifest_only": audit["build_manifest_only"],
        "download_enabled": audit["download_enabled"],
        "browser_engine_used": audit["browser_engine_used"],
        "page_count": audit["page_count"],
        "generated_manifest_record_count": audit["generated_manifest_record_count"],
        "planned_download_count": audit["planned_download_count"],
        "manifest_source_kind": audit["manifest_source_kind"],
        "manifest_schema_audit": audit["manifest_schema_audit"],
    }


def required_table_roles(config: dict) -> list[str]:
    sync_config = adni_section(config)["sync"]
    scoped_unavailable = {str(value) for value in sync_config.get("scoped_unavailable_table_roles", [])}
    return [str(value) for value in sync_config["required_table_roles"] if str(value) not in scoped_unavailable]


def role_registry(config: dict) -> dict[str, dict[str, object]]:
    return {
        str(role_name): role_config
        for role_name, role_config in adni_section(config)["sync"]["role_registry"].items()
    }


def selected_asset_rows(config: dict, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    sync_config = adni_section(config)["sync"]
    allowed_asset_kinds = set(sync_config["allowed_asset_kinds"])
    allowed_data_domains = set(sync_config["allowed_data_domains"])
    allowed_modalities = set(sync_config["allowed_modalities"])
    declared_roles = set(role_registry(config))
    selected_rows: list[dict[str, str]] = []
    for row in rows:
        if row["include_flag"] != "yes":
            continue
        if row["table_role"] not in declared_roles:
            raise ValueError(f"ADNI asset has a non-declared table role: {row['asset_uid']}")
        if row["asset_kind"] not in allowed_asset_kinds:
            raise ValueError(f"ADNI asset has a non-declared asset kind: {row['asset_uid']}")
        if row["data_domain"] not in allowed_data_domains:
            raise ValueError(f"ADNI asset has a non-declared data domain: {row['asset_uid']}")
        if row["modality"] and row["modality"] not in allowed_modalities:
            raise ValueError(f"ADNI asset has a non-declared modality: {row['asset_uid']}")
        if row["participant_level_flag"] != "yes":
            raise ValueError(f"ADNI controlled-access asset must be marked participant-level: {row['asset_uid']}")
        safe_relative_path(row["target_relative_path"])
        selected_rows.append(row)
    return selected_rows


def extract_hidden_form_fields(response_text: str, field_names: list[str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    input_pattern = re.compile(r"<input\b[^>]*>", flags=re.IGNORECASE)
    name_pattern = re.compile(r"\bname=[\"']([^\"']+)[\"']", flags=re.IGNORECASE)
    value_pattern = re.compile(r"\bvalue=[\"']([^\"']*)[\"']", flags=re.IGNORECASE)
    wanted_names = {str(value) for value in field_names}
    for input_match in input_pattern.finditer(response_text):
        input_text = input_match.group(0)
        name_match = name_pattern.search(input_text)
        if name_match is None or html.unescape(name_match.group(1)) not in wanted_names:
            continue
        value_match = value_pattern.search(input_text)
        payload[html.unescape(name_match.group(1))] = html.unescape(value_match.group(1)) if value_match else ""
    return payload


def validate_login_success(response: requests.Response, auth_config: dict, session: requests.Session) -> None:
    if response.status_code not in set(int(value) for value in auth_config["success_status_codes"]):
        raise RuntimeError(f"ADNI authenticated HTTP login failed with status code {response.status_code}")
    cookie_policy = auth_config.get("cookie_policy", {})
    if bool(cookie_policy.get("require_session_cookie", False)):
        allowed_names = {str(value) for value in cookie_policy["allowed_session_cookie_names"]}
        session_cookie_names = {cookie.name for cookie in session.cookies}
        if not allowed_names & session_cookie_names:
            raise RuntimeError("ADNI authenticated HTTP login did not return a declared session cookie.")
    success_patterns = [re.compile(str(value)) for value in auth_config.get("success_text_patterns", [])]
    if success_patterns and not any(pattern.search(response.text) for pattern in success_patterns):
        raise RuntimeError("ADNI authenticated HTTP login response did not match a declared success marker.")


def configured_session(config: dict) -> requests.Session:
    sync_config = adni_section(config)["sync"]
    auth_config = sync_config["authenticated_http"]
    session = requests.Session()
    session.trust_env = bool(auth_config.get("network_policy", {}).get("use_environment_proxy", True))
    session.headers.update({str(key): str(value) for key, value in auth_config["request_headers"].items()})
    return session


def session_text_matches(response_text: str, pattern_values: list[str]) -> bool:
    return any(re.search(str(pattern_value), response_text) for pattern_value in pattern_values)


def browser_cookie_session_failure_reason(config: dict, session: requests.Session) -> str:
    sync_config = adni_section(config)["sync"]
    auth_config = sync_config["authenticated_http"]
    discovery_config = sync_config["ida_discovery"]
    if not bool(discovery_config["enabled"]):
        return ""
    discovery_required_gate = bool(discovery_config.get("required_gate", True)) and str(acquisition_config(config).get("mode", "")) != "manifest_first"
    download_link_count = 0
    for target_config in discovery_config["discovery_targets"]:
        page_url = str(target_config["url"])
        response = session.get(page_url)
        response.raise_for_status()
        page_link_count = sum(
            1
            for download_url in link_urls_from_html(page_url, response.text, str(discovery_config["link_href_regex"]))
            if role_for_download_url(download_url, config)
        )
        entry_patterns = [str(value) for value in auth_config.get("authenticated_entry_text_patterns", [])]
        if entry_patterns and session_text_matches(response.text, entry_patterns):
            page_link_count += 1
        if page_link_count == 0 and session_text_matches(response.text, [str(value) for value in auth_config.get("login_page_text_patterns", [])]):
            return "ADNI browser cookie jar is not authenticated; re-export cookies from a logged-in IDA browser session."
        download_link_count += page_link_count
    if download_link_count == 0 and discovery_required_gate:
        return "ADNI browser cookie jar did not expose an authenticated IDA discovery entry; re-export cookies from a logged-in browser session."
    return ""


def validate_browser_cookie_session(config: dict, session: requests.Session) -> None:
    failure_reason = browser_cookie_session_failure_reason(config, session)
    if failure_reason:
        raise RuntimeError(failure_reason)


def load_browser_cookie_jar(config: dict) -> MozillaCookieJar:
    return load_browser_cookie_jar_from_path(config, configured_cookie_jar_paths(config)[0])


def build_browser_cookie_session(config: dict) -> requests.Session:
    candidate_errors: list[str] = []
    for cookie_jar_path in configured_cookie_jar_paths(config):
        session = configured_session(config)
        filtered_cookie_jar = CookieJar()
        for cookie in ida_relevant_cookies_for_path(config, cookie_jar_path):
            filtered_cookie_jar.set_cookie(cookie)
        if sum(1 for _ in filtered_cookie_jar) == 0:
            candidate_errors.append(f"{controlled_display_path(cookie_jar_path)}: no configured-domain cookies")
            continue
        session.cookies.update(filtered_cookie_jar)
        failure_reason = browser_cookie_session_failure_reason(config, session)
        if failure_reason:
            candidate_errors.append(f"{controlled_display_path(cookie_jar_path)}: {failure_reason}")
            continue
        setattr(session, "_adni_cookie_jar_path", cookie_jar_path)
        return session
    error_detail = "; ".join(candidate_errors)
    raise RuntimeError(f"ADNI browser cookie jars did not authenticate; re-export cookies from a logged-in IDA browser session. Candidate status: {error_detail}")


def build_password_login_session(config: dict, credentials: dict[str, str]) -> requests.Session:
    sync_config = adni_section(config)["sync"]
    auth_config = sync_config["authenticated_http"]
    session = configured_session(config)
    authentication_kind = str(auth_config["authentication_kind"])
    if authentication_kind not in {"form_post", "form_post_with_csrf"}:
        raise ValueError(f"Unsupported ADNI authenticated HTTP authentication kind: {auth_config['authentication_kind']}")
    login_page = session.get(str(auth_config["login_url"]))
    login_page.raise_for_status()
    cookie_policy = auth_config.get("cookie_policy", {})
    policy_cookie_name = str(cookie_policy.get("accepted_policy_cookie_name", ""))
    if policy_cookie_name:
        session.cookies.set(policy_cookie_name, str(cookie_policy["accepted_policy_cookie_value"]), domain=urlparse(str(auth_config["login_url"])).hostname, path="/")
    form_payload = {}
    hidden_field_names = [str(value) for value in auth_config.get("hidden_field_names", [])]
    if hidden_field_names:
        form_payload.update(extract_hidden_form_fields(login_page.text, hidden_field_names))
        missing_hidden_fields = sorted(set(hidden_field_names) - set(form_payload))
        if missing_hidden_fields:
            raise RuntimeError(f"ADNI authenticated HTTP login page did not expose declared hidden fields: {missing_hidden_fields}")
    if authentication_kind == "form_post_with_csrf":
        form_payload.update(extract_hidden_form_fields(login_page.text, [str(value) for value in auth_config["csrf_field_names"]]))
        if not form_payload:
            raise RuntimeError("ADNI authenticated HTTP login page did not expose any declared CSRF field.")
    form_payload.update(
        {
            str(auth_config["email_field"]): credentials["email"],
            str(auth_config["password_field"]): credentials["password"],
        }
    )
    form_payload.update({str(key): str(value) for key, value in auth_config.get("extra_post_fields", {}).items()})
    submit_url = str(auth_config.get("submit_url", auth_config["login_url"]))
    response = session.post(
        submit_url,
        data=form_payload,
    )
    validate_login_success(response, auth_config, session)
    return session


def build_authenticated_session(config: dict, credentials: dict[str, str] | None = None) -> requests.Session:
    auth_config = adni_section(config)["sync"]["authenticated_http"]
    authentication_order = str(auth_config.get("authentication_order", "password_login"))
    if authentication_order == "browser_cookie_jar_first":
        return build_browser_cookie_session(config)
    if authentication_order == "password_login":
        if credentials is None:
            credentials = load_adni_credentials(config)
        return build_password_login_session(config, credentials)
    raise ValueError(f"Unsupported ADNI authenticated HTTP authentication order: {authentication_order}")


def link_urls_from_html(page_url: str, page_text: str, link_regex: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(link_regex, page_text):
        href = html.unescape(match.group(1))
        parsed = urlparse(href)
        if parsed.scheme in {"mailto", "javascript"}:
            continue
        urls.append(urljoin(page_url, href))
    return urls


def role_for_download_url(download_url: str, config: dict) -> str:
    parsed_path = urlparse(download_url).path
    candidate_text = f"{Path(parsed_path).name} {download_url}"
    return role_for_candidate_text(candidate_text, config)


def discovered_asset_rows(config: dict, session: requests.Session) -> list[dict[str, str]]:
    sync_config = adni_section(config)["sync"]
    discovery_config = sync_config["ida_discovery"]
    if not bool(discovery_config["enabled"]):
        return []
    rows_by_role: dict[str, dict[str, str]] = {}
    registry = role_registry(config)
    for target_config in discovery_config["discovery_targets"]:
        page_url = str(target_config["url"])
        response = session.get(page_url)
        response.raise_for_status()
        for download_url in link_urls_from_html(page_url, response.text, str(discovery_config["link_href_regex"])):
            table_role = role_for_download_url(download_url, config)
            if not table_role or table_role in rows_by_role:
                continue
            role_config = registry[table_role]
            rows_by_role[table_role] = {
                "asset_uid": f"adni_{table_role}",
                "include_flag": "yes",
                "transport_kind": "authenticated_http",
                "asset_kind": str(role_config["asset_kind"]),
                "data_domain": str(role_config["data_domain"]),
                "table_role": table_role,
                "modality": str(role_config["modality"]),
                "adni_cohort": "ADNI",
                "source_url": download_url,
                "source_path": "",
                "target_relative_path": str(role_config["target_relative_path"]),
                "participant_level_flag": "yes",
                "allowed_use": "validation_internal_only",
                "notes": f"ida_discovery:{target_config['target_kind']}",
            }
    return list(rows_by_role.values())


def merged_rows_by_role(manifest_rows: list[dict[str, str]], discovery_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows_by_role = {row["table_role"]: row for row in discovery_rows}
    rows_by_role.update({row["table_role"]: row for row in manifest_rows})
    return list(rows_by_role.values())


def missing_required_roles(config: dict, selected_rows: list[dict[str, str]]) -> list[str]:
    selected_roles = {row["table_role"] for row in selected_rows}
    return sorted(set(required_table_roles(config)) - selected_roles)


def write_missing_role_audit(config: dict, selected_rows: list[dict[str, str]], missing_roles: list[str]) -> None:
    controlled_config = adni_section(config)
    private_root = resolve_configured_path(controlled_config["paths"]["private_root"])
    audit_path = private_root / controlled_config["paths"]["missing_role_audit"]
    write_json(
        audit_path,
        {
            "source_key": controlled_config["source_key"],
            "participant_level_flag": "yes_private_audit",
            "required_table_roles": required_table_roles(config),
            "covered_table_roles": sorted({row["table_role"] for row in selected_rows}),
            "scoped_unavailable_table_roles": [str(value) for value in controlled_config["sync"].get("scoped_unavailable_table_roles", [])],
            "missing_required_table_roles": missing_roles,
        },
    )


def ensure_required_role_coverage(config: dict, selected_rows: list[dict[str, str]]) -> None:
    missing_roles = missing_required_roles(config, selected_rows)
    write_missing_role_audit(config, selected_rows, missing_roles)
    if missing_roles:
        raise RuntimeError(f"Missing required ADNI controlled-access table roles: {', '.join(missing_roles)}")


def write_stream_atomic(source_handle, destination: Path, chunk_size: int) -> None:
    ensure_directory(destination.parent)
    temporary_path = temporary_output_path(destination)
    with temporary_path.open("wb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle, length=chunk_size)
    temporary_path.replace(destination)


def copy_authorized_export(row: dict[str, str], destination: Path, chunk_size: int, reuse_existing_files: bool) -> str:
    if destination.exists() and reuse_existing_files:
        return "reused_cached"
    source_path = resolve_configured_path(row["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Approved ADNI export source is missing for asset {row['asset_uid']}")
    with source_path.open("rb") as source_handle:
        write_stream_atomic(source_handle, destination, chunk_size)
    return "copied"


def download_authenticated_http_asset(
    row: dict[str, str],
    destination: Path,
    session: requests.Session,
    chunk_size: int,
    reuse_existing_files: bool,
) -> str:
    if destination.exists() and reuse_existing_files:
        return "reused_cached"
    response = session.get(row["source_url"], stream=True)
    if response.status_code >= 400:
        response.close()
        raise RuntimeError(f"ADNI authenticated HTTP asset download failed with status code {response.status_code}")
    ensure_directory(destination.parent)
    temporary_path = temporary_output_path(destination)
    with temporary_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)
    response.close()
    temporary_path.replace(destination)
    return "downloaded"


def materialize_asset(
    row: dict[str, str],
    private_root: Path,
    session: requests.Session | None,
    sync_config: dict,
) -> dict[str, object]:
    destination = private_root / safe_relative_path(row["target_relative_path"])
    chunk_size = int(sync_config["download_chunk_size_bytes"])
    reuse_existing_files = bool(sync_config["reuse_existing_files"])
    transport_kind = row["transport_kind"]
    if transport_kind == "local_authorized_export":
        status = copy_authorized_export(row, destination, chunk_size, reuse_existing_files)
    elif transport_kind == "authenticated_http":
        if session is None:
            raise RuntimeError("ADNI authenticated HTTP assets require an authenticated session.")
        status = download_authenticated_http_asset(row, destination, session, chunk_size, reuse_existing_files)
    else:
        raise ValueError(f"Unsupported ADNI asset transport kind: {transport_kind}")
    return {
        "asset_uid": row["asset_uid"],
        "asset_kind": row["asset_kind"],
        "data_domain": row["data_domain"],
        "table_role": row["table_role"],
        "modality": row["modality"],
        "adni_cohort": row["adni_cohort"],
        "local_path_private": controlled_display_path(destination),
        "checksum_sha256": sha256_of_file(destination),
        "size_bytes": destination.stat().st_size,
        "sync_date": today_iso(),
        "status": status,
        "participant_level_flag": row["participant_level_flag"],
        "allowed_use": row["allowed_use"],
        "notes": row["notes"],
    }


def sync_adni_assets(config: dict) -> list[dict[str, object]]:
    controlled_config = adni_section(config)
    sync_config = controlled_config["sync"]
    acquisition_mode = str(acquisition_config(config).get("mode", "approved_asset_manifest"))
    private_root = resolve_configured_path(controlled_config["paths"]["private_root"])
    ensure_directory(private_root)
    if acquisition_mode == "collection_first":
        run_collection_first_no_download_dry_run(config)
        if not bool(collection_first_config(config)["download_enabled"]) or bool(collection_first_config(config)["build_manifest_only"]):
            return []
        manifest_rows = read_ida_export_manifest(config)
        manifest_audit = manifest_schema_audit(config, manifest_rows, "authenticated_cookie_session_valid")
        write_manifest_schema_audit(config, manifest_audit)
        session = build_authenticated_session(config)
        selected_rows = manifest_first_asset_rows(config, manifest_rows, manifest_audit)
        if bool(collection_first_config(config).get("enforce_required_roles_on_download", True)):
            ensure_required_role_coverage(config, selected_rows)
            required_roles = set(required_table_roles(config))
            selected_rows = [row for row in selected_rows if row["table_role"] in required_roles]
            rows_by_role: dict[str, dict[str, str]] = {}
            for row in selected_rows:
                rows_by_role.setdefault(row["table_role"], row)
            selected_rows = list(rows_by_role.values())
        receipt_rows = [
            materialize_asset(row, private_root, session, sync_config)
            for row in selected_rows
        ]
        receipt_fields = sync_config["receipt_fields"]
        write_tsv(private_root / controlled_config["paths"]["download_receipts"], receipt_fields, receipt_rows)
        write_tsv(private_root / controlled_config["paths"]["raw_manifest"], config["schemas"]["adni_raw_manifest_fields"], receipt_rows)
        refresh_checksums(private_root, controlled_config["paths"]["checksum_table"])
        return receipt_rows
    if acquisition_mode == "manifest_first":
        session = build_authenticated_session(config)
        manifest_rows = read_ida_export_manifest(config)
        audit = manifest_schema_audit(config, manifest_rows, "authenticated_cookie_session_valid")
        write_manifest_schema_audit(config, audit)
        selected_rows = manifest_first_asset_rows(config, manifest_rows, audit)
        receipt_rows = [
            materialize_asset(row, private_root, session, sync_config)
            for row in selected_rows
        ]
        receipt_fields = sync_config["receipt_fields"]
        write_tsv(private_root / controlled_config["paths"]["download_receipts"], receipt_fields, receipt_rows)
        write_tsv(private_root / controlled_config["paths"]["raw_manifest"], config["schemas"]["adni_raw_manifest_fields"], receipt_rows)
        refresh_checksums(private_root, controlled_config["paths"]["checksum_table"])
        return receipt_rows
    if acquisition_mode != "approved_asset_manifest":
        raise ValueError(f"Unsupported ADNI acquisition mode: {acquisition_mode}")
    approved_rows = read_approved_asset_rows(config)
    selected_rows = selected_asset_rows(config, approved_rows)
    needs_authenticated_http = any(row["transport_kind"] == "authenticated_http" for row in selected_rows)
    needs_discovery = bool(sync_config["ida_discovery"]["enabled"]) and bool(missing_required_roles(config, selected_rows))
    authentication_order = str(sync_config["authenticated_http"].get("authentication_order", "password_login"))
    needs_password_credentials = bool(sync_config["credential_required"]) and authentication_order == "password_login"
    credentials = load_adni_credentials(config) if needs_password_credentials else {}
    session = build_authenticated_session(config, credentials) if (needs_authenticated_http or needs_discovery) else None
    if needs_discovery and session is not None:
        discovered_rows = selected_asset_rows(config, discovered_asset_rows(config, session))
        selected_rows = selected_asset_rows(config, merged_rows_by_role(selected_rows, discovered_rows))
    ensure_required_role_coverage(config, selected_rows)
    needs_authenticated_http = any(row["transport_kind"] == "authenticated_http" for row in selected_rows)
    if needs_authenticated_http and session is None:
        session = build_authenticated_session(config, credentials)
    receipt_rows = [
        materialize_asset(row, private_root, session, sync_config)
        for row in selected_rows
    ]
    receipt_fields = sync_config["receipt_fields"]
    write_tsv(private_root / controlled_config["paths"]["download_receipts"], receipt_fields, receipt_rows)
    write_tsv(private_root / controlled_config["paths"]["raw_manifest"], config["schemas"]["adni_raw_manifest_fields"], receipt_rows)
    refresh_checksums(private_root, controlled_config["paths"]["checksum_table"])
    return receipt_rows


def main() -> None:
    args = parse_args()
    config = load_controlled_research_config(args.config)
    assert_controlled_governance_approval(config)
    apply_collection_first_profile_override(config, str(args.collection_first_profile))
    assert_controlled_network_credentials(config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    pipeline_logger.progress(
        "adni_sync_started",
        authentication_source="browser_cookie_jar" if str(adni_section(config)["sync"]["authenticated_http"].get("authentication_order", "")) == "browser_cookie_jar_first" else "explicit_configured_form_login",
        logging_policy=adni_section(config)["credentials"]["logging_policy"],
        acquisition_mode=str(acquisition_config(config).get("mode", "approved_asset_manifest")),
    )
    if args.dry_run:
        acquisition_mode = str(acquisition_config(config).get("mode", "approved_asset_manifest"))
        if acquisition_mode == "collection_first":
            audit = run_collection_first_no_download_dry_run(config)
            summary = redacted_collection_first_summary(audit)
            event_name = "adni_collection_first_dry_run_completed"
        else:
            audit = run_manifest_first_no_download_dry_run(config)
            summary = redacted_manifest_dry_run_summary(audit)
            event_name = "adni_manifest_first_dry_run_completed"
        print(json.dumps(summary, sort_keys=True))
        pipeline_logger.progress(
            event_name,
            summary=summary,
        )
        complete_script_logging(
            pipeline_logger,
            pipeline_started_at,
            summary=summary,
        )
        return
    receipt_rows = sync_adni_assets(config)
    status_counts: dict[str, int] = {}
    for row in receipt_rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    pipeline_logger.progress(
        "adni_sync_completed",
        asset_count=len(receipt_rows),
        status_counts=status_counts,
    )
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        asset_count=len(receipt_rows),
        status_counts=status_counts,
    )


if __name__ == "__main__":
    main()
