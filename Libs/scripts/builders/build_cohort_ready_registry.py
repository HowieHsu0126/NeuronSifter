from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import (
    ensure_directory,
    infer_tissue_or_biofluid,
    load_config,
    normalize_text,
    read_tsv,
    resolve_repo_path,
    slugify,
    write_tsv,
)


PAIR_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9 /_-]+):\s*([^:]+?)(?=\s+[A-Za-z][A-Za-z0-9 /_-]+:\s*|$)")
MMSE_PATTERN = re.compile(r"\bmmse\b[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
CDR_PATTERN = re.compile(r"\bcdr\b[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cohort-ready processed catalogs for model-ready public studies.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--source-accessions",
        default="",
        help="Optional comma-separated source_accession subset for targeted rebuilds.",
    )
    return parser.parse_args()


def selected_accessions(accession_text: str) -> set[str]:
    return {value.strip() for value in accession_text.split(",") if value.strip()}


def stable_uid(*parts: str) -> str:
    normalized_parts = [part.strip() for part in parts if part and part.strip()]
    joined = "::".join(normalized_parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    slug_seed = "_".join(normalized_parts[:4]) if normalized_parts else "record"
    return f"{slugify(slug_seed)[:80]}_{digest}"


def normalize_header(header: str) -> str:
    return normalize_text(header).replace(" ", "_")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", newline="")


def path_suffix_label(value: str) -> str:
    suffixes = Path(value).suffixes
    if not suffixes:
        return ""
    if len(suffixes) >= 2 and suffixes[-1] == ".gz" and not suffixes[-2].removeprefix(".").isdigit():
        return "".join(suffixes[-2:])
    return suffixes[-1]


def keyword_present(text: str, keyword: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_keyword = normalize_text(keyword)
    if not normalized_keyword:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword).replace(r'\\ ', r'\\s+')}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def detect_diagnosis_label(text: str, alias_map: dict[str, list[str]]) -> str:
    matches: list[tuple[int, str]] = []
    for diagnosis_label, keywords in alias_map.items():
        for keyword in keywords:
            if keyword_present(text, keyword):
                matches.append((len(normalize_text(keyword)), diagnosis_label))
    if not matches:
        return ""
    matches.sort(reverse=True)
    return matches[0][1]


def canonicalize_diagnosis(text: str, study_diagnosis_label: str, alias_map: dict[str, list[str]]) -> str:
    sample_label = detect_diagnosis_label(text, alias_map)
    if sample_label:
        return sample_label
    return study_diagnosis_label


def infer_case_control(text: str, keyword_groups: dict[str, list[str]]) -> str:
    if any(keyword_present(text, keyword) for keyword in keyword_groups["comparator"]):
        return "comparator"
    if any(keyword_present(text, keyword) for keyword in keyword_groups["control"]):
        return "control"
    if any(keyword_present(text, keyword) for keyword in keyword_groups["case"]):
        return "case"
    return ""


def parse_key_value_pairs(text: str) -> dict[str, str]:
    pairs = {}
    for key, value in PAIR_PATTERN.findall(text):
        pairs[normalize_text(key)] = " ".join(value.split())
    return pairs


def join_key_values(pairs: dict[str, str], preferred_keys: list[str]) -> str:
    ordered_values: list[str] = []
    for preferred_key in preferred_keys:
        normalized_key = normalize_text(preferred_key)
        if normalized_key in pairs and pairs[normalized_key]:
            ordered_values.append(f"{preferred_key}: {pairs[normalized_key]}")
    seen = set()
    deduplicated_values: list[str] = []
    for value in ordered_values:
        normalized_value = normalize_text(value)
        if normalized_value in seen:
            continue
        seen.add(normalized_value)
        deduplicated_values.append(value)
    return "; ".join(deduplicated_values)


def header_lookup(headers: list[str], preferred_names: list[str]) -> str:
    normalized_map = {normalize_header(header): header for header in headers}
    for preferred_name in preferred_names:
        normalized_preferred = normalize_header(preferred_name)
        if normalized_preferred in normalized_map:
            return normalized_map[normalized_preferred]
    for preferred_name in preferred_names:
        normalized_preferred = normalize_header(preferred_name)
        for header in headers:
            if normalized_preferred in normalize_header(header):
                return header
    return ""


def headers_with_keywords(headers: list[str], keywords: list[str]) -> list[str]:
    matches: list[str] = []
    for header in headers:
        normalized_header_value = normalize_header(header)
        if any(normalize_text(keyword).replace(" ", "_") in normalized_header_value for keyword in keywords):
            matches.append(header)
    return matches


def first_non_empty(row: dict[str, str], headers: list[str]) -> str:
    for header in headers:
        if header and row.get(header, "").strip():
            return row[header].strip()
    return ""


def clean_join(values: list[str]) -> str:
    normalized_values = [" ".join(value.split()) for value in values if value and " ".join(value.split())]
    return "; ".join(normalized_values)


def extract_clinical_score(text: str, pairs: dict[str, str]) -> str:
    pair_text = join_key_values(pairs, ["cdr", "mmse", "cerad", "braak"])
    if pair_text:
        return pair_text
    matches: list[str] = []
    mmse_match = MMSE_PATTERN.search(text)
    if mmse_match:
        matches.append(f"MMSE: {mmse_match.group(1)}")
    cdr_match = CDR_PATTERN.search(text)
    if cdr_match:
        matches.append(f"CDR: {cdr_match.group(1)}")
    return "; ".join(matches)


def resolve_local_path(path_value: str) -> Path:
    return resolve_repo_path(path_value)


def choose_tissue(study_tissue: str, text: str, tissue_keyword_map: dict[str, list[str]]) -> str:
    if study_tissue:
        return study_tissue
    return infer_tissue_or_biofluid(text, tissue_keyword_map)


def build_geo_samples(
    study_row: dict[str, str],
    family_soft_receipt: dict[str, str],
    config: dict,
) -> tuple[list[dict[str, object]], str]:
    local_path = resolve_local_path(family_soft_receipt["local_path"])
    alias_map = config["cohort_ready_registry"]["diagnosis_aliases"]
    keyword_groups = config["cohort_ready_registry"]["case_control_keywords"]
    geo_keys = config["cohort_ready_registry"]["geo_characteristic_keys"]
    tissue_keyword_map = config["classification"]["tissue_keyword_map"]
    sample_rows: list[dict[str, object]] = []
    row_count_values: set[str] = set()

    current_sample_id = ""
    current_fields: defaultdict[str, list[str]] = defaultdict(list)
    in_sample_table = False

    def finalize_sample() -> None:
        if not current_sample_id:
            return
        characteristics = current_fields.get("characteristics_ch1", [])
        characteristic_pairs: dict[str, str] = {}
        for characteristic in characteristics:
            characteristic_pairs.update(parse_key_value_pairs(characteristic))
        sample_title = clean_join(current_fields.get("title", []))
        source_name = clean_join(current_fields.get("source_name_ch1", []))
        descriptions = clean_join(current_fields.get("description", []))
        supplementary_urls = clean_join(
            [
                value
                for field_name, values in current_fields.items()
                if field_name.startswith("supplementary_file")
                for value in values
            ]
        )
        sample_text = clean_join([sample_title, source_name, clean_join(characteristics), descriptions])
        sample_rows.append(
            {
                "sample_uid": stable_uid(study_row["source_key"], study_row["source_accession"], current_sample_id),
                "study_uid": study_row["study_uid"],
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "source_sample_id": first_non_empty({"sample_id": current_sample_id}, ["sample_id"]),
                "subject_id": source_name,
                "sample_title": sample_title or current_sample_id,
                "species": clean_join(current_fields.get("organism_ch1", [])) or study_row["species"],
                "tissue_or_biofluid": choose_tissue(study_row["tissue_or_biofluid"], sample_text, tissue_keyword_map),
                "modality": study_row["modality"],
                "sample_context": study_row["sample_context"],
                "study_diagnosis_label": study_row["diagnosis_label"],
                "sample_diagnosis_label": canonicalize_diagnosis(sample_text, study_row["diagnosis_label"], alias_map),
                "diagnosis_detail": sample_text,
                "disease_stage": join_key_values(characteristic_pairs, geo_keys["disease_stage"]),
                "case_control_label": infer_case_control(sample_text, keyword_groups),
                "sex": join_key_values(characteristic_pairs, geo_keys["sex"]),
                "age": join_key_values(characteristic_pairs, geo_keys["age"]),
                "clinical_score": extract_clinical_score(sample_text, characteristic_pairs),
                "analysis_or_platform": clean_join(current_fields.get("platform_id", [])),
                "primary_quantitative_asset_id": "",
                "supplementary_file_url": supplementary_urls,
                "raw_metadata_path": family_soft_receipt["local_path"],
                "note": "geo_family_soft",
            }
        )
        for row_count_value in current_fields.get("data_row_count", []):
            if row_count_value:
                row_count_values.add(row_count_value)

    with open_text(local_path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                finalize_sample()
                current_sample_id = line.split(" = ", 1)[1].strip()
                current_fields = defaultdict(list)
                in_sample_table = False
                continue
            if not current_sample_id:
                continue
            if line == "!sample_table_begin":
                in_sample_table = True
                continue
            if line == "!sample_table_end":
                in_sample_table = False
                continue
            if in_sample_table:
                continue
            if not line.startswith("!Sample_"):
                continue
            field_name, value = line.split(" = ", 1)
            current_fields[field_name.removeprefix("!Sample_")].append(value.strip())
    finalize_sample()

    feature_count = ""
    if len(row_count_values) == 1:
        feature_count = next(iter(row_count_values))
    return sample_rows, feature_count


def count_series_matrix_dimensions(local_path: Path) -> tuple[int, int]:
    sample_count = 0
    feature_count = 0
    in_table = False
    header_consumed = False
    with open_text(local_path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "!series_matrix_table_begin":
                in_table = True
                header_consumed = False
                continue
            if line == "!series_matrix_table_end":
                break
            if not in_table:
                continue
            if not header_consumed:
                sample_count = max(len(line.split("\t")) - 1, 0)
                header_consumed = True
                continue
            if line:
                feature_count += 1
    return sample_count, feature_count


def build_geo_assets(
    study_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
    sample_count: int,
    family_soft_feature_count: str,
) -> list[dict[str, object]]:
    asset_rows: list[dict[str, object]] = []
    for receipt_row in receipt_rows:
        if receipt_row["status"] not in {"downloaded", "reused_cached"}:
            continue
        if receipt_row["asset_name"] not in {"family_soft", "series_matrix"}:
            continue
        feature_count = family_soft_feature_count
        sample_identifier_field = "Sample_geo_accession"
        feature_identifier_field = "ID_REF"
        value_field = "VALUE"
        if receipt_row["asset_name"] == "series_matrix":
            series_sample_count, series_feature_count = count_series_matrix_dimensions(resolve_local_path(receipt_row["local_path"]))
            sample_count_value = series_sample_count
            feature_count = str(series_feature_count)
            sample_identifier_field = "matrix_sample_columns"
            value_field = "matrix_cell_value"
        else:
            sample_count_value = sample_count
        asset_rows.append(
            {
                "quantitative_asset_uid": stable_uid(
                    study_row["source_key"],
                    study_row["source_accession"],
                    receipt_row["asset_name"],
                    receipt_row["local_path"],
                ),
                "study_uid": study_row["study_uid"],
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "asset_name": receipt_row["asset_name"],
                "asset_role": receipt_row["asset_name"],
                "local_availability_flag": "yes",
                "local_path": receipt_row["local_path"],
                "asset_url": receipt_row["asset_url"],
                "modality": study_row["modality"],
                "matrix_format": path_suffix_label(receipt_row["local_path"]),
                "sample_count": sample_count_value,
                "feature_count": feature_count,
                "sample_identifier_field": sample_identifier_field,
                "feature_identifier_field": feature_identifier_field,
                "value_field": value_field,
                "raw_metadata_path": receipt_row["local_path"],
                "note": receipt_row["note"],
            }
        )
    return asset_rows


def build_arrayexpress_samples_and_assets(
    study_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ready_receipts = [row for row in receipt_rows if row["status"] in {"downloaded", "reused_cached"}]
    ready_receipts_by_asset = {
        (row["asset_name"], row["asset_url"]): row
        for row in ready_receipts
        if row["asset_name"] in {"raw_archive", "derived_matrix_archive"}
    }
    sdrf_receipts = [row for row in ready_receipts if row["asset_name"] == "SDRF File"]
    if not sdrf_receipts:
        raise ValueError(f"Missing SDRF File for ArrayExpress study {study_row['source_accession']}")
    sdrf_receipt = sdrf_receipts[0]
    sdrf_rows = read_tsv(resolve_local_path(sdrf_receipt["local_path"]))
    headers = list(sdrf_rows[0].keys())
    alias_map = config["cohort_ready_registry"]["diagnosis_aliases"]
    keyword_groups = config["cohort_ready_registry"]["case_control_keywords"]
    tissue_keyword_map = config["classification"]["tissue_keyword_map"]
    remote_column_map = config["cohort_ready_registry"]["arrayexpress_remote_columns"]

    sample_id_header = header_lookup(headers, ["Sample Name", "Extract Name"])
    subject_id_header = header_lookup(headers, ["Characteristics[individual]"])
    sample_title_headers = [
        header_lookup(headers, ["Description"]),
        header_lookup(headers, ["Comment [Sample_title]"]),
        header_lookup(headers, ["Comment [Sample_source_name]"]),
        header_lookup(headers, ["Comment [Sample_characteristics]"]),
    ]
    species_header = header_lookup(headers, ["Characteristics [organism]", "Characteristics[organism]"])
    tissue_header = header_lookup(headers, ["Characteristics [organism part]", "Characteristics[organism part]"])
    diagnosis_header = header_lookup(
        headers,
        [
            "FactorValue [disease]",
            "FactorValue [disease status]",
            "Characteristics[disease]",
            "Characteristics[disease status]",
        ],
    )
    stage_header = header_lookup(headers, ["Characteristics[disease staging]", "FactorValue [disease staging]"])
    clinical_header = header_lookup(headers, ["Characteristics[clinical information]", "Characteristics [clinical information]"])
    platform_header = header_lookup(headers, ["Array Design REF"])
    sex_headers = headers_with_keywords(headers, ["sex", "gender"])
    age_headers = headers_with_keywords(headers, ["age"])

    sample_rows: list[dict[str, object]] = []
    remote_assets: dict[tuple[str, str], str] = {}
    for row in sdrf_rows:
        sample_title = clean_join([first_non_empty(row, sample_title_headers), row.get(sample_id_header, "").strip()])
        diagnosis_detail = clean_join(
            [
                row.get(diagnosis_header, "").strip(),
                row.get(stage_header, "").strip(),
                row.get(clinical_header, "").strip(),
                sample_title,
            ]
        )
        supplementary_urls = clean_join(
            [
                row.get(header_lookup(headers, remote_column_map["raw_archive"]), "").strip(),
                row.get(header_lookup(headers, remote_column_map["derived_matrix_archive"]), "").strip(),
            ]
        )
        sample_rows.append(
            {
                "sample_uid": stable_uid(study_row["source_key"], study_row["source_accession"], row.get(sample_id_header, "").strip()),
                "study_uid": study_row["study_uid"],
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "source_sample_id": row.get(sample_id_header, "").strip(),
                "subject_id": row.get(subject_id_header, "").strip(),
                "sample_title": sample_title or row.get(sample_id_header, "").strip(),
                "species": row.get(species_header, "").strip() or study_row["species"],
                "tissue_or_biofluid": choose_tissue(
                    study_row["tissue_or_biofluid"],
                    clean_join([row.get(tissue_header, "").strip(), sample_title]),
                    tissue_keyword_map,
                ),
                "modality": study_row["modality"],
                "sample_context": study_row["sample_context"],
                "study_diagnosis_label": study_row["diagnosis_label"],
                "sample_diagnosis_label": canonicalize_diagnosis(diagnosis_detail, study_row["diagnosis_label"], alias_map),
                "diagnosis_detail": diagnosis_detail,
                "disease_stage": row.get(stage_header, "").strip(),
                "case_control_label": infer_case_control(diagnosis_detail, keyword_groups),
                "sex": first_non_empty(row, sex_headers),
                "age": first_non_empty(row, age_headers),
                "clinical_score": row.get(clinical_header, "").strip(),
                "analysis_or_platform": row.get(platform_header, "").strip(),
                "primary_quantitative_asset_id": "",
                "supplementary_file_url": supplementary_urls,
                "raw_metadata_path": sdrf_receipt["local_path"],
                "note": "arrayexpress_sdrf",
            }
        )
        for asset_name, candidate_headers in remote_column_map.items():
            header = header_lookup(headers, candidate_headers)
            asset_url = row.get(header, "").strip()
            if asset_url:
                remote_assets[(asset_name, asset_url)] = sdrf_receipt["local_path"]

    asset_rows: list[dict[str, object]] = []
    for asset_name, asset_url in sorted(remote_assets):
        receipt_row = ready_receipts_by_asset.get((asset_name, asset_url))
        asset_rows.append(
            {
                "quantitative_asset_uid": stable_uid(study_row["source_key"], study_row["source_accession"], asset_name, asset_url),
                "study_uid": study_row["study_uid"],
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "asset_name": asset_name,
                "asset_role": asset_name,
                "local_availability_flag": "yes" if receipt_row is not None else "no",
                "local_path": receipt_row["local_path"] if receipt_row is not None else "",
                "asset_url": asset_url,
                "modality": study_row["modality"],
                "matrix_format": path_suffix_label(receipt_row["local_path"] if receipt_row is not None else asset_url),
                "sample_count": len(sample_rows),
                "feature_count": "",
                "sample_identifier_field": sample_id_header or "Sample Name",
                "feature_identifier_field": "",
                "value_field": "",
                "raw_metadata_path": remote_assets[(asset_name, asset_url)],
                "note": receipt_row["note"] if receipt_row is not None else Path(asset_url).name,
            }
        )
    return sample_rows, asset_rows


def sorted_json_values(payload: dict[str, object]) -> list[dict[str, object]]:
    return [payload[key] for key in sorted(payload, key=lambda value: int(value))]


def build_metabolomics_samples_and_assets(
    study_row: dict[str, str],
    receipt_rows: list[dict[str, str]],
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ready_receipts = [row for row in receipt_rows if row["status"] in {"downloaded", "reused_cached"}]
    receipt_by_name = {row["asset_name"]: row for row in ready_receipts}
    required_assets = {"factors_json", "data_json", "mwtab_txt", "summary_json"}
    missing_assets = required_assets - set(receipt_by_name)
    if missing_assets:
        raise ValueError(f"Missing Metabolomics Workbench assets for {study_row['source_accession']}: {sorted(missing_assets)}")

    factors_payload = json.loads(resolve_local_path(receipt_by_name["factors_json"]["local_path"]).read_text(encoding="utf-8"))
    data_payload = json.loads(resolve_local_path(receipt_by_name["data_json"]["local_path"]).read_text(encoding="utf-8"))
    summary_payload = json.loads(resolve_local_path(receipt_by_name["summary_json"]["local_path"]).read_text(encoding="utf-8"))
    factor_rows = sorted_json_values(factors_payload)
    data_rows = sorted_json_values(data_payload)

    alias_map = config["cohort_ready_registry"]["diagnosis_aliases"]
    keyword_groups = config["cohort_ready_registry"]["case_control_keywords"]
    tissue_keyword_map = config["classification"]["tissue_keyword_map"]
    analysis_label = summary_payload.get("analysis_type", "")
    if not analysis_label and data_rows:
        analysis_label = str(data_rows[0].get("analysis_summary", ""))

    sample_rows: list[dict[str, object]] = []
    for factor_row in factor_rows:
        factor_pairs = parse_key_value_pairs(str(factor_row.get("factors", "")))
        factor_text = clean_join([str(factor_row.get("factors", "")), str(factor_row.get("sample_source", ""))])
        sample_rows.append(
            {
                "sample_uid": stable_uid(
                    study_row["source_key"],
                    study_row["source_accession"],
                    str(factor_row.get("local_sample_id", "")),
                ),
                "study_uid": study_row["study_uid"],
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "source_sample_id": str(factor_row.get("local_sample_id", "")),
                "subject_id": str(factor_row.get("mb_sample_id", "")),
                "sample_title": str(factor_row.get("local_sample_id", "")),
                "species": study_row["species"] or str(summary_payload.get("species", "")),
                "tissue_or_biofluid": choose_tissue(
                    study_row["tissue_or_biofluid"],
                    factor_text,
                    tissue_keyword_map,
                ),
                "modality": study_row["modality"],
                "sample_context": study_row["sample_context"],
                "study_diagnosis_label": study_row["diagnosis_label"],
                "sample_diagnosis_label": canonicalize_diagnosis(factor_text, study_row["diagnosis_label"], alias_map),
                "diagnosis_detail": str(factor_row.get("factors", "")),
                "disease_stage": join_key_values(factor_pairs, ["stage"]),
                "case_control_label": infer_case_control(factor_text, keyword_groups),
                "sex": join_key_values(factor_pairs, ["sex", "gender"]),
                "age": join_key_values(factor_pairs, ["age"]),
                "clinical_score": extract_clinical_score(str(factor_row.get("factors", "")), factor_pairs),
                "analysis_or_platform": analysis_label,
                "primary_quantitative_asset_id": "",
                "supplementary_file_url": str(factor_row.get("raw_data", "")),
                "raw_metadata_path": receipt_by_name["factors_json"]["local_path"],
                "note": "metabolomics_workbench_factors",
            }
        )

    feature_count = len(data_rows)
    sample_count = len(sample_rows)
    asset_rows = [
        {
            "quantitative_asset_uid": stable_uid(
                study_row["source_key"],
                study_row["source_accession"],
                "data_json",
                receipt_by_name["data_json"]["local_path"],
            ),
            "study_uid": study_row["study_uid"],
            "source_key": study_row["source_key"],
            "source_accession": study_row["source_accession"],
            "asset_name": "data_json",
            "asset_role": "data_json",
            "local_availability_flag": "yes",
            "local_path": receipt_by_name["data_json"]["local_path"],
            "asset_url": receipt_by_name["data_json"]["asset_url"],
            "modality": study_row["modality"],
            "matrix_format": path_suffix_label(receipt_by_name["data_json"]["local_path"]),
            "sample_count": sample_count,
            "feature_count": feature_count,
            "sample_identifier_field": "local_sample_id",
            "feature_identifier_field": "metabolite_id",
            "value_field": "DATA",
            "raw_metadata_path": receipt_by_name["factors_json"]["local_path"],
            "note": receipt_by_name["data_json"]["note"],
        },
        {
            "quantitative_asset_uid": stable_uid(
                study_row["source_key"],
                study_row["source_accession"],
                "mwtab_txt",
                receipt_by_name["mwtab_txt"]["local_path"],
            ),
            "study_uid": study_row["study_uid"],
            "source_key": study_row["source_key"],
            "source_accession": study_row["source_accession"],
            "asset_name": "mwtab_txt",
            "asset_role": "mwtab_txt",
            "local_availability_flag": "yes",
            "local_path": receipt_by_name["mwtab_txt"]["local_path"],
            "asset_url": receipt_by_name["mwtab_txt"]["asset_url"],
            "modality": study_row["modality"],
            "matrix_format": path_suffix_label(receipt_by_name["mwtab_txt"]["local_path"]),
            "sample_count": sample_count,
            "feature_count": feature_count,
            "sample_identifier_field": "local_sample_id",
            "feature_identifier_field": "metabolite_name",
            "value_field": "metabolite_intensity",
            "raw_metadata_path": receipt_by_name["factors_json"]["local_path"],
            "note": receipt_by_name["mwtab_txt"]["note"],
        },
    ]
    return sample_rows, asset_rows


def choose_primary_asset_id(
    source_key: str,
    asset_rows: list[dict[str, object]],
    priority_config: dict[str, list[str]],
) -> tuple[str, str]:
    study_assets = asset_rows
    if not study_assets:
        return "", ""
    priority_list = priority_config[source_key]
    ordered_assets = sorted(
        study_assets,
        key=lambda row: (
            priority_list.index(str(row["asset_name"])) if str(row["asset_name"]) in priority_list else len(priority_list),
            0 if row["local_availability_flag"] == "yes" else 1,
            str(row["asset_name"]),
            str(row["asset_url"]),
        ),
    )
    primary_row = ordered_assets[0]
    return str(primary_row["quantitative_asset_uid"]), str(primary_row["local_path"])


def load_receipts_for_studies(study_rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    receipt_cache: dict[str, list[dict[str, str]]] = {}
    receipt_rows_by_study: dict[tuple[str, str], list[dict[str, str]]] = {}
    for study_row in study_rows:
        receipt_table = study_row["raw_receipt_table"]
        if receipt_table not in receipt_cache:
            receipt_cache[receipt_table] = read_tsv(resolve_local_path(receipt_table))
        receipt_rows_by_study[(study_row["source_key"], study_row["source_accession"])] = [
            row
            for row in receipt_cache[receipt_table]
            if row["source_accession"] == study_row["source_accession"]
        ]
    return receipt_rows_by_study


def build_study_rows(
    model_ready_studies: list[dict[str, str]],
    sample_rows: list[dict[str, object]],
    asset_rows: list[dict[str, object]],
    config: dict,
) -> list[dict[str, object]]:
    samples_by_study: dict[str, list[dict[str, object]]] = defaultdict(list)
    assets_by_study: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample_row in sample_rows:
        samples_by_study[str(sample_row["study_uid"])].append(sample_row)
    for asset_row in asset_rows:
        assets_by_study[str(asset_row["study_uid"])].append(asset_row)

    study_summary_rows: list[dict[str, object]] = []
    priority_config = config["cohort_ready_registry"]["primary_asset_priority"]
    primary_asset_by_study: dict[str, tuple[str, str]] = {}

    for study_row in model_ready_studies:
        study_assets = assets_by_study.get(study_row["study_uid"], [])
        primary_asset_id, primary_asset_path = choose_primary_asset_id(
            study_row["source_key"],
            study_assets,
            priority_config,
        )
        primary_asset_by_study[study_row["study_uid"]] = (primary_asset_id, primary_asset_path)
        study_summary_rows.append(
            {
                "study_uid": study_row["study_uid"],
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "title": study_row["title"],
                "modality": study_row["modality"],
                "sample_context": study_row["sample_context"],
                "standardized_sample_count": len(samples_by_study.get(study_row["study_uid"], [])),
                "quantitative_asset_count": len(study_assets),
                "local_quantitative_asset_count": sum(1 for row in study_assets if row["local_availability_flag"] == "yes"),
                "remote_only_quantitative_asset_count": sum(1 for row in study_assets if row["local_availability_flag"] == "no"),
                "primary_quantitative_asset_id": primary_asset_id,
                "primary_quantitative_asset_path": primary_asset_path,
            }
        )

    for sample_row in sample_rows:
        primary_asset_id, _primary_asset_path = primary_asset_by_study[str(sample_row["study_uid"])]
        sample_row["primary_quantitative_asset_id"] = primary_asset_id

    study_summary_rows.sort(key=lambda row: (row["source_key"], row["source_accession"]))
    return study_summary_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    source_accessions = selected_accessions(args.source_accessions)
    processed_root = resolve_repo_path(config["cohort_ready_registry"]["local_root"])
    ensure_directory(processed_root)

    processed_registry_root = resolve_repo_path(config["processed_registry"]["local_root"])
    study_catalog_path = processed_registry_root / config["processed_registry"]["relative_paths"]["study_catalog"]
    study_catalog_rows = read_tsv(study_catalog_path)
    model_ready_studies = [
        row
        for row in study_catalog_rows
        if row["model_ready_flag"] == "yes" and row["source_key"] in config["cohort_ready_registry"]["enabled_sources"]
        and (not source_accessions or row["source_accession"] in source_accessions)
    ]
    pipeline_logger.progress(
        "loaded_cohort_ready_inputs",
        study_catalog_row_count=len(study_catalog_rows),
        model_ready_study_count=len(model_ready_studies),
    )
    receipt_rows_by_study = load_receipts_for_studies(model_ready_studies)

    sample_rows: list[dict[str, object]] = []
    asset_rows: list[dict[str, object]] = []

    for study_row in model_ready_studies:
        study_key = (study_row["source_key"], study_row["source_accession"])
        receipt_rows = receipt_rows_by_study[study_key]
        if study_row["source_key"] == "geo":
            family_soft_receipts = [
                row
                for row in receipt_rows
                if row["asset_name"] == "family_soft" and row["status"] in {"downloaded", "reused_cached"}
            ]
            if not family_soft_receipts:
                raise ValueError(f"Missing family_soft for GEO study {study_row['source_accession']}")
            geo_sample_rows, family_soft_feature_count = build_geo_samples(study_row, family_soft_receipts[0], config)
            geo_asset_rows = build_geo_assets(study_row, receipt_rows, len(geo_sample_rows), family_soft_feature_count)
            sample_rows.extend(geo_sample_rows)
            asset_rows.extend(geo_asset_rows)
            continue
        if study_row["source_key"] == "arrayexpress":
            arrayexpress_sample_rows, arrayexpress_asset_rows = build_arrayexpress_samples_and_assets(study_row, receipt_rows, config)
            sample_rows.extend(arrayexpress_sample_rows)
            asset_rows.extend(arrayexpress_asset_rows)
            continue
        if study_row["source_key"] == "metabolomics_workbench":
            metabolomics_sample_rows, metabolomics_asset_rows = build_metabolomics_samples_and_assets(study_row, receipt_rows, config)
            sample_rows.extend(metabolomics_sample_rows)
            asset_rows.extend(metabolomics_asset_rows)
            continue
        raise ValueError(f"Unsupported cohort-ready source: {study_row['source_key']}")

    study_summary_rows = build_study_rows(model_ready_studies, sample_rows, asset_rows, config)

    sample_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["source_sample_id"]))
    asset_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["asset_name"], row["asset_url"]))

    relative_paths = config["cohort_ready_registry"]["relative_paths"]
    write_tsv(
        processed_root / relative_paths["sample_catalog"],
        config["schemas"]["cohort_ready_sample_fields"],
        sample_rows,
    )
    write_tsv(
        processed_root / relative_paths["quantitative_asset_catalog"],
        config["schemas"]["cohort_ready_asset_fields"],
        asset_rows,
    )
    write_tsv(
        processed_root / relative_paths["study_summary"],
        config["schemas"]["cohort_ready_study_fields"],
        study_summary_rows,
    )
    pipeline_logger.artifact(
        "cohort_ready_registry_written",
        artifact=processed_root,
        sample_row_count=len(sample_rows),
        asset_row_count=len(asset_rows),
        study_row_count=len(study_summary_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=processed_root,
        sample_row_count=len(sample_rows),
        asset_row_count=len(asset_rows),
        study_row_count=len(study_summary_rows),
    )

if __name__ == "__main__":
    main()
