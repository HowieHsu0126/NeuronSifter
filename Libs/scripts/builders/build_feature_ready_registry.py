from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, normalize_text, read_tsv, resolve_repo_path, slugify, write_tsv


GSM_PATTERN = re.compile(r"(GSM\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature-ready matrix catalogs from the cohort-ready registry.")
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


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", newline="")


def parse_tsv_line(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="\t"))


def detect_delimiter(header_line: str) -> str:
    return "\t" if header_line.count("\t") >= header_line.count(",") else ","


def path_suffix_label(value: str) -> str:
    suffixes = Path(value).suffixes
    if not suffixes:
        return ""
    if len(suffixes) >= 2 and suffixes[-1] == ".gz" and not suffixes[-2].removeprefix(".").isdigit():
        return "".join(suffixes[-2:])
    return suffixes[-1]


def sorted_json_values(payload: dict[str, object]) -> list[dict[str, object]]:
    return [payload[key] for key in sorted(payload, key=lambda value: int(value))]


def remove_known_suffixes(value: str) -> str:
    lowered_value = value.lower()
    suffixes = [".cel.gz", ".txt.gz", ".csv.gz", ".tsv.gz", ".cel", ".txt", ".csv", ".tsv"]
    for suffix in suffixes:
        if lowered_value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def identifier_key_pairs(value: str) -> list[tuple[str, str]]:
    stripped_value = value.strip()
    if not stripped_value:
        return []
    pairs: list[tuple[str, str]] = []
    candidates = [
        ("normalized_text", normalize_text(stripped_value)),
        ("path_stem", normalize_text(Path(stripped_value).stem)),
        ("suffix_trimmed", normalize_text(remove_known_suffixes(stripped_value))),
    ]
    for gsm_accession in GSM_PATTERN.findall(stripped_value):
        candidates.append(("gsm_accession", gsm_accession.lower()))
    unique_pairs: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for strategy, key in candidates:
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        unique_pairs.append((strategy, key))
    return unique_pairs


def build_sample_lookup(sample_rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, dict[str, str]]]]:
    samples_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    lookup_buckets: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for sample_row in sample_rows:
        study_uid = sample_row["study_uid"]
        samples_by_study[study_uid].append(sample_row)
        seen_keys: set[str] = set()
        for field_name in ["source_sample_id", "subject_id", "sample_title"]:
            for _strategy, key in identifier_key_pairs(sample_row[field_name]):
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                lookup_buckets[study_uid][key].append(sample_row)

    lookup_by_study: dict[str, dict[str, dict[str, str]]] = {}
    for study_uid, key_rows in lookup_buckets.items():
        lookup_by_study[study_uid] = {}
        for key, rows in key_rows.items():
            unique_rows = {row["sample_uid"]: row for row in rows}
            if len(unique_rows) == 1:
                lookup_by_study[study_uid][key] = next(iter(unique_rows.values()))
    return samples_by_study, lookup_by_study


def match_sample(
    study_uid: str,
    matrix_sample_id: str,
    sample_lookup_by_study: dict[str, dict[str, dict[str, str]]],
) -> tuple[dict[str, str] | None, str]:
    study_lookup = sample_lookup_by_study.get(study_uid, {})
    for strategy, key in identifier_key_pairs(matrix_sample_id):
        if key in study_lookup:
            return study_lookup[key], f"matched_by_{strategy}"
    return None, "unmapped"


def build_matrix_reference(local_path: str, member_path: str, matrix_name: str) -> str:
    reference = local_path
    if member_path:
        reference = f"{reference}::{member_path}"
    return f"{reference}#{matrix_name}"


def choose_matrix_semantics(source_key: str, measurement_label: str, modality: str) -> str:
    normalized_measurement = normalize_text(measurement_label)
    normalized_modality = normalize_text(modality)
    if "p value" in normalized_measurement or "pvalue" in normalized_measurement:
        return "detection_pvalue"
    if "abs call" in normalized_measurement or normalized_measurement == "call":
        return "presence_call"
    if source_key == "metabolomics_workbench":
        return "metabolite_intensity"
    if "methylation" in normalized_modality:
        return "methylation_signal"
    if "transcript" in normalized_modality or "expression" in normalized_modality:
        return "expression_signal"
    return "quantitative_signal"


def infer_feature_type(source_key: str, modality: str) -> str:
    normalized_modality = normalize_text(modality)
    if source_key == "metabolomics_workbench":
        return "metabolite"
    if "methylation" in normalized_modality:
        return "methylation_locus"
    if "transcript" in normalized_modality or "expression" in normalized_modality:
        return "transcript_feature"
    return "molecular_feature"


def non_matrix_row(asset_row: dict[str, str], exclusion_reason: str, note: str) -> dict[str, object]:
    return {
        "quantitative_asset_uid": asset_row["quantitative_asset_uid"],
        "study_uid": asset_row["study_uid"],
        "source_key": asset_row["source_key"],
        "source_accession": asset_row["source_accession"],
        "asset_name": asset_row["asset_name"],
        "local_path": asset_row["local_path"],
        "exclusion_reason": exclusion_reason,
        "note": note,
    }


def feature_rows_for_matrix(
    matrix_uid: str,
    study_row: dict[str, str],
    feature_entries: list[tuple[str, str]],
) -> list[dict[str, object]]:
    feature_type = infer_feature_type(study_row["source_key"], study_row["modality"])
    rows: list[dict[str, object]] = []
    for row_index, (feature_id, feature_label) in enumerate(feature_entries, start=1):
        rows.append(
            {
                "matrix_uid": matrix_uid,
                "study_uid": study_row["study_uid"],
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "feature_id": feature_id,
                "feature_label": feature_label,
                "row_index": row_index,
                "feature_type": feature_type,
                "note": "",
            }
        )
    return rows


def sample_axis_rows_for_matrix(
    matrix_uid: str,
    study_row: dict[str, str],
    sample_identifiers: list[str],
    sample_lookup_by_study: dict[str, dict[str, dict[str, str]]],
) -> tuple[list[dict[str, object]], int, int]:
    rows: list[dict[str, object]] = []
    mapped_count = 0
    unmapped_count = 0
    for column_index, matrix_sample_id in enumerate(sample_identifiers, start=1):
        matched_sample_row, match_status = match_sample(study_row["study_uid"], matrix_sample_id, sample_lookup_by_study)
        if matched_sample_row is None:
            unmapped_count += 1
        else:
            mapped_count += 1
        rows.append(
            {
                "matrix_uid": matrix_uid,
                "study_uid": study_row["study_uid"],
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "matrix_sample_id": matrix_sample_id,
                "column_index": column_index,
                "sample_uid": matched_sample_row["sample_uid"] if matched_sample_row is not None else "",
                "source_sample_id": matched_sample_row["source_sample_id"] if matched_sample_row is not None else "",
                "sample_match_status": match_status,
                "note": "",
            }
        )
    return rows, mapped_count, unmapped_count


def create_matrix_row(
    study_row: dict[str, str],
    asset_row: dict[str, str],
    matrix_name: str,
    matrix_origin: str,
    measurement_label: str,
    member_path: str,
    file_format: str,
    sample_axis_count: int,
    feature_axis_count: int,
    mapped_sample_count: int,
    unmapped_sample_count: int,
    sample_identifier_field: str,
    feature_identifier_field: str,
    feature_label_field: str,
    value_identifier: str,
    note: str,
) -> dict[str, object]:
    matrix_uid = stable_uid(
        study_row["source_key"],
        study_row["source_accession"],
        asset_row["quantitative_asset_uid"],
        matrix_name,
        measurement_label,
    )
    return {
        "matrix_uid": matrix_uid,
        "study_uid": study_row["study_uid"],
        "source_key": study_row["source_key"],
        "source_accession": study_row["source_accession"],
        "quantitative_asset_uid": asset_row["quantitative_asset_uid"],
        "asset_name": asset_row["asset_name"],
        "matrix_name": matrix_name,
        "matrix_origin": matrix_origin,
        "measurement_label": measurement_label,
        "matrix_semantics": choose_matrix_semantics(study_row["source_key"], measurement_label, study_row["modality"]),
        "local_path": asset_row["local_path"],
        "member_path": member_path,
        "matrix_reference": build_matrix_reference(asset_row["local_path"], member_path, matrix_name),
        "file_format": file_format,
        "sample_axis_count": sample_axis_count,
        "feature_axis_count": feature_axis_count,
        "mapped_sample_count": mapped_sample_count,
        "unmapped_sample_count": unmapped_sample_count,
        "sample_identifier_field": sample_identifier_field,
        "feature_identifier_field": feature_identifier_field,
        "feature_label_field": feature_label_field,
        "value_identifier": value_identifier,
        "raw_metadata_path": asset_row["raw_metadata_path"],
        "note": note,
    }


def create_member_row(
    matrix_uid: str,
    study_row: dict[str, str],
    asset_row: dict[str, str],
    member_path: str,
    source_sample_id_hint: str,
    matched_sample_row: dict[str, str] | None,
    sample_match_status: str,
    member_feature_count: int,
    note: str,
) -> dict[str, object]:
    return {
        "member_uid": stable_uid(matrix_uid, member_path or asset_row["local_path"], source_sample_id_hint or "member"),
        "matrix_uid": matrix_uid,
        "study_uid": study_row["study_uid"],
        "source_key": study_row["source_key"],
        "source_accession": study_row["source_accession"],
        "quantitative_asset_uid": asset_row["quantitative_asset_uid"],
        "asset_name": asset_row["asset_name"],
        "local_path": asset_row["local_path"],
        "member_path": member_path,
        "source_sample_id_hint": source_sample_id_hint,
        "sample_uid": matched_sample_row["sample_uid"] if matched_sample_row is not None else "",
        "source_sample_id": matched_sample_row["source_sample_id"] if matched_sample_row is not None else "",
        "sample_match_status": sample_match_status,
        "member_feature_count": member_feature_count,
        "note": note,
    }


def parse_geo_series_matrix(
    study_row: dict[str, str],
    asset_row: dict[str, str],
    sample_lookup_by_study: dict[str, dict[str, dict[str, str]]],
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    local_path = resolve_repo_path(asset_row["local_path"])
    sample_identifiers: list[str] = []
    feature_entries: list[tuple[str, str]] = []
    in_table = False
    with open_text(local_path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "!series_matrix_table_begin":
                in_table = True
                continue
            if line == "!series_matrix_table_end":
                break
            if not in_table:
                continue
            values = parse_tsv_line(line)
            if not sample_identifiers:
                sample_identifiers = [value.strip() for value in values[1:] if value.strip()]
                continue
            if not values or not values[0].strip():
                continue
            feature_id = values[0].strip()
            feature_entries.append((feature_id, feature_id))

    if not feature_entries:
        return [], [], [], [], [
            non_matrix_row(
                asset_row,
                config["feature_ready_registry"]["excluded_asset_reasons"]["empty_series_matrix"],
                "GEO series_matrix table does not contain feature rows.",
            )
        ]

    matrix_name = local_path.name.removesuffix(".txt.gz").removesuffix(".gz")
    matrix_row_template = create_matrix_row(
        study_row,
        asset_row,
        matrix_name,
        "geo_series_matrix",
        "VALUE",
        "",
        path_suffix_label(local_path.name),
        len(sample_identifiers),
        len(feature_entries),
        0,
        0,
        "series_matrix_header",
        "ID_REF",
        "ID_REF",
        "VALUE",
        asset_row["note"],
    )
    sample_axis_rows, mapped_count, unmapped_count = sample_axis_rows_for_matrix(
        matrix_row_template["matrix_uid"],
        study_row,
        sample_identifiers,
        sample_lookup_by_study,
    )
    matrix_row_template["mapped_sample_count"] = mapped_count
    matrix_row_template["unmapped_sample_count"] = unmapped_count
    feature_axis_rows = feature_rows_for_matrix(matrix_row_template["matrix_uid"], study_row, feature_entries)
    member_rows = [
        create_member_row(
            matrix_row_template["matrix_uid"],
            study_row,
            asset_row,
            "",
            "",
            None,
            "not_applicable",
            len(feature_entries),
            "geo_series_matrix_file",
        )
    ]
    return [matrix_row_template], member_rows, sample_axis_rows, feature_axis_rows, []


def classify_arrayexpress_member(zip_file: zipfile.ZipFile, member_name: str) -> str:
    with zip_file.open(member_name) as raw_handle:
        text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(text_handle, delimiter="\t")
        first_row = next(reader, [])
        second_row = next(reader, [])
    if len(first_row) > 4 and len(second_row) > 4:
        return "combined_matrix"
    if len(first_row) >= 2:
        return "sample_table"
    return "unknown"


def parse_arrayexpress_combined_members(
    study_row: dict[str, str],
    asset_row: dict[str, str],
    member_names: list[str],
    sample_lookup_by_study: dict[str, dict[str, dict[str, str]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    matrix_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    sample_axis_rows: list[dict[str, object]] = []
    feature_axis_rows: list[dict[str, object]] = []
    local_path = resolve_repo_path(asset_row["local_path"])

    with zipfile.ZipFile(local_path) as zip_file:
        for member_name in sorted(member_names):
            with zip_file.open(member_name) as raw_handle:
                text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", errors="replace", newline="")
                reader = csv.reader(text_handle, delimiter="\t")
                sample_header = next(reader, [])
                measurement_header = next(reader, [])
                sample_identifiers_by_measurement: dict[str, list[str]] = defaultdict(list)
                for sample_identifier, measurement_label in zip(sample_header[1:], measurement_header[1:]):
                    cleaned_measurement = measurement_label.strip() or "VALUE"
                    sample_identifiers_by_measurement[cleaned_measurement].append(sample_identifier.strip())
                feature_identifier_field = measurement_header[0].strip() if measurement_header else ""
                feature_entries: list[tuple[str, str]] = []
                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    feature_id = row[0].strip()
                    feature_entries.append((feature_id, feature_id))

            for measurement_label, sample_identifiers in sorted(sample_identifiers_by_measurement.items()):
                matrix_name = f"{Path(member_name).stem}_{slugify(measurement_label)}"
                matrix_row = create_matrix_row(
                    study_row,
                    asset_row,
                    matrix_name,
                    "arrayexpress_combined_member",
                    measurement_label,
                    member_name,
                    path_suffix_label(member_name),
                    len(sample_identifiers),
                    len(feature_entries),
                    0,
                    0,
                    sample_header[0].strip() if sample_header else "sample_header",
                    feature_identifier_field or "feature_identifier",
                    feature_identifier_field or "feature_identifier",
                    measurement_label,
                    member_name,
                )
                current_sample_axis_rows, mapped_count, unmapped_count = sample_axis_rows_for_matrix(
                    matrix_row["matrix_uid"],
                    study_row,
                    sample_identifiers,
                    sample_lookup_by_study,
                )
                matrix_row["mapped_sample_count"] = mapped_count
                matrix_row["unmapped_sample_count"] = unmapped_count
                matrix_rows.append(matrix_row)
                sample_axis_rows.extend(current_sample_axis_rows)
                feature_axis_rows.extend(feature_rows_for_matrix(matrix_row["matrix_uid"], study_row, feature_entries))
                member_rows.append(
                    create_member_row(
                        matrix_row["matrix_uid"],
                        study_row,
                        asset_row,
                        member_name,
                        "",
                        None,
                        "not_applicable",
                        len(feature_entries),
                        "arrayexpress_combined_member",
                    )
                )
    return matrix_rows, member_rows, sample_axis_rows, feature_axis_rows


def parse_arrayexpress_sample_table_members(
    study_row: dict[str, str],
    asset_row: dict[str, str],
    member_names: list[str],
    sample_lookup_by_study: dict[str, dict[str, dict[str, str]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    local_path = resolve_repo_path(asset_row["local_path"])
    states: dict[str, dict[str, object]] = {}

    with zipfile.ZipFile(local_path) as zip_file:
        for member_name in sorted(member_names):
            with zip_file.open(member_name) as raw_handle:
                text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", errors="replace", newline="")
                reader = csv.reader(text_handle, delimiter="\t")
                header = next(reader, [])
                if len(header) < 2:
                    continue
                feature_identifier_field = header[0].strip() or "feature_identifier"
                value_headers = [value.strip() for value in header[1:] if value.strip()]
                feature_entries: list[tuple[str, str]] = []
                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    feature_id = row[0].strip()
                    feature_entries.append((feature_id, feature_id))

            source_sample_id_hint = Path(member_name).stem
            if source_sample_id_hint.lower().endswith("_sample_table"):
                source_sample_id_hint = source_sample_id_hint[: -len("_sample_table")]
            matched_sample_row, sample_match_status = match_sample(
                study_row["study_uid"],
                source_sample_id_hint,
                sample_lookup_by_study,
            )

            for value_header in value_headers:
                state = states.setdefault(
                    value_header,
                    {
                        "feature_identifier_field": feature_identifier_field,
                        "feature_entries": feature_entries[:],
                        "reference_feature_ids": [feature_id for feature_id, _feature_label in feature_entries],
                        "feature_seen": {feature_id for feature_id, _feature_label in feature_entries},
                        "sample_identifiers": [],
                        "member_rows": [],
                        "consistent_feature_axis": True,
                    },
                )
                current_feature_ids = [feature_id for feature_id, _feature_label in feature_entries]
                if current_feature_ids != state["reference_feature_ids"]:
                    state["consistent_feature_axis"] = False
                for feature_id, feature_label in feature_entries:
                    if feature_id in state["feature_seen"]:
                        continue
                    state["feature_entries"].append((feature_id, feature_label))
                    state["feature_seen"].add(feature_id)
                state["sample_identifiers"].append(source_sample_id_hint)
                state["member_rows"].append(
                    (
                        member_name,
                        source_sample_id_hint,
                        matched_sample_row,
                        sample_match_status,
                        len(feature_entries),
                    )
                )

    matrix_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    sample_axis_rows: list[dict[str, object]] = []
    feature_axis_rows: list[dict[str, object]] = []

    for value_header, state in sorted(states.items()):
        matrix_name = f"{Path(asset_row['local_path']).name.removesuffix('.zip')}_{slugify(value_header)}"
        note = "arrayexpress_sample_table_members"
        if not state["consistent_feature_axis"]:
            note = f"{note}; feature_axis_union_applied"
        matrix_row = create_matrix_row(
            study_row,
            asset_row,
            matrix_name,
            "arrayexpress_sample_table_members",
            value_header,
            "",
            ".txt",
            len(state["sample_identifiers"]),
            len(state["feature_entries"]),
            0,
            0,
            "member_name_sample_hint",
            state["feature_identifier_field"],
            state["feature_identifier_field"],
            value_header,
            note,
        )
        current_sample_axis_rows, mapped_count, unmapped_count = sample_axis_rows_for_matrix(
            matrix_row["matrix_uid"],
            study_row,
            state["sample_identifiers"],
            sample_lookup_by_study,
        )
        matrix_row["mapped_sample_count"] = mapped_count
        matrix_row["unmapped_sample_count"] = unmapped_count
        matrix_rows.append(matrix_row)
        sample_axis_rows.extend(current_sample_axis_rows)
        feature_axis_rows.extend(feature_rows_for_matrix(matrix_row["matrix_uid"], study_row, state["feature_entries"]))
        for member_name, source_sample_id_hint, matched_sample_row, sample_match_status, feature_count in state["member_rows"]:
            member_rows.append(
                create_member_row(
                    matrix_row["matrix_uid"],
                    study_row,
                    asset_row,
                    member_name,
                    source_sample_id_hint,
                    matched_sample_row,
                    sample_match_status,
                    feature_count,
                    value_header,
                )
            )

    return matrix_rows, member_rows, sample_axis_rows, feature_axis_rows


def parse_arrayexpress_asset(
    study_row: dict[str, str],
    asset_row: dict[str, str],
    sample_lookup_by_study: dict[str, dict[str, dict[str, str]]],
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    local_path = resolve_repo_path(asset_row["local_path"])
    if local_path.suffix.lower() != ".zip":
        return parse_arrayexpress_text_matrix_asset(study_row, asset_row, sample_lookup_by_study, config)
    with zipfile.ZipFile(local_path) as zip_file:
        member_names = [name for name in zip_file.namelist() if not name.endswith("/")]
        combined_members: list[str] = []
        sample_table_members: list[str] = []
        for member_name in member_names:
            member_kind = classify_arrayexpress_member(zip_file, member_name)
            if member_kind == "combined_matrix":
                combined_members.append(member_name)
                continue
            if member_kind == "sample_table":
                sample_table_members.append(member_name)

    if combined_members:
        matrix_rows, member_rows, sample_axis_rows, feature_axis_rows = parse_arrayexpress_combined_members(
            study_row,
            asset_row,
            combined_members,
            sample_lookup_by_study,
        )
        return matrix_rows, member_rows, sample_axis_rows, feature_axis_rows, []
    if sample_table_members:
        matrix_rows, member_rows, sample_axis_rows, feature_axis_rows = parse_arrayexpress_sample_table_members(
            study_row,
            asset_row,
            sample_table_members,
            sample_lookup_by_study,
        )
        return matrix_rows, member_rows, sample_axis_rows, feature_axis_rows, []
    return [], [], [], [], [
        non_matrix_row(
            asset_row,
            config["feature_ready_registry"]["excluded_asset_reasons"]["unsupported_arrayexpress_archive"],
            "ArrayExpress archive does not contain a supported processed-matrix member layout.",
        )
    ]


def parse_arrayexpress_text_matrix_asset(
    study_row: dict[str, str],
    asset_row: dict[str, str],
    sample_lookup_by_study: dict[str, dict[str, dict[str, str]]],
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    local_path = resolve_repo_path(asset_row["local_path"])
    with open_text(local_path) as handle:
        header_line = handle.readline().rstrip("\n")
        if not header_line:
            return [], [], [], [], [
                non_matrix_row(
                    asset_row,
                    config["feature_ready_registry"]["excluded_asset_reasons"]["unsupported_arrayexpress_archive"],
                    "ArrayExpress text matrix is empty.",
                )
            ]
        delimiter = detect_delimiter(header_line)
        reader = csv.reader(io.StringIO(header_line + "\n" + handle.read()), delimiter=delimiter)
        header = next(reader, [])
        if len(header) < 2:
            return [], [], [], [], [
                non_matrix_row(
                    asset_row,
                    config["feature_ready_registry"]["excluded_asset_reasons"]["unsupported_arrayexpress_archive"],
                    "ArrayExpress text matrix does not expose a feature column plus sample columns.",
                )
            ]
        feature_identifier_field = header[0].strip() or "feature_identifier"
        sample_identifiers = [value.strip() for value in header[1:] if value.strip()]
        feature_entries: list[tuple[str, str]] = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            feature_id = row[0].strip()
            feature_entries.append((feature_id, feature_id))

    matrix_name = local_path.stem
    matrix_row = create_matrix_row(
        study_row,
        asset_row,
        matrix_name,
        "arrayexpress_text_matrix",
        "VALUE",
        "",
        path_suffix_label(local_path.name),
        len(sample_identifiers),
        len(feature_entries),
        0,
        0,
        header[1].strip() if len(header) > 1 else "sample_header",
        feature_identifier_field,
        feature_identifier_field,
        "VALUE",
        asset_row["note"],
    )
    sample_axis_rows, mapped_count, unmapped_count = sample_axis_rows_for_matrix(
        matrix_row["matrix_uid"],
        study_row,
        sample_identifiers,
        sample_lookup_by_study,
    )
    matrix_row["mapped_sample_count"] = mapped_count
    matrix_row["unmapped_sample_count"] = unmapped_count
    feature_axis_rows = feature_rows_for_matrix(matrix_row["matrix_uid"], study_row, feature_entries)
    member_rows = [
        create_member_row(
            matrix_row["matrix_uid"],
            study_row,
            asset_row,
            "",
            "",
            None,
            "not_applicable",
            len(feature_entries),
            "arrayexpress_text_matrix",
        )
    ]
    return [matrix_row], member_rows, sample_axis_rows, feature_axis_rows, []


def parse_metabolomics_data_json(
    study_row: dict[str, str],
    asset_row: dict[str, str],
    sample_lookup_by_study: dict[str, dict[str, dict[str, str]]],
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    payload = json.loads(resolve_repo_path(asset_row["local_path"]).read_text(encoding="utf-8"))
    data_rows = sorted_json_values(payload)
    if not data_rows:
        return [], [], [], [], [
            non_matrix_row(
                asset_row,
                config["feature_ready_registry"]["excluded_asset_reasons"]["empty_metabolomics_data_json"],
                "Metabolomics Workbench data_json does not contain metabolite rows.",
            )
        ]

    sample_identifiers: list[str] = []
    seen_samples: set[str] = set()
    feature_entries: list[tuple[str, str]] = []
    for data_row in data_rows:
        feature_id = str(data_row.get("metabolite_id", "")).strip() or str(data_row.get("metabolite_name", "")).strip()
        if not feature_id:
            raise ValueError(f"Metabolomics data row lacks metabolite identifier in {asset_row['local_path']}")
        feature_label = (
            str(data_row.get("refmet_name", "")).strip()
            or str(data_row.get("metabolite_name", "")).strip()
            or feature_id
        )
        feature_entries.append((feature_id, feature_label))
        data_payload = data_row.get("DATA", {})
        if not isinstance(data_payload, dict):
            raise ValueError(f"Metabolomics DATA payload is not a dictionary in {asset_row['local_path']}")
        for sample_identifier in data_payload:
            if sample_identifier in seen_samples:
                continue
            seen_samples.add(sample_identifier)
            sample_identifiers.append(sample_identifier)

    matrix_name = f"{Path(asset_row['local_path']).stem}_{slugify('DATA')}"
    matrix_row = create_matrix_row(
        study_row,
        asset_row,
        matrix_name,
        "metabolomics_data_json",
        "DATA",
        "",
        path_suffix_label(asset_row["local_path"]),
        len(sample_identifiers),
        len(feature_entries),
        0,
        0,
        "DATA_keys",
        "metabolite_id",
        "refmet_name_or_metabolite_name",
        "DATA",
        str(data_rows[0].get("analysis_summary", "")).strip() or asset_row["note"],
    )
    sample_axis_rows, mapped_count, unmapped_count = sample_axis_rows_for_matrix(
        matrix_row["matrix_uid"],
        study_row,
        sample_identifiers,
        sample_lookup_by_study,
    )
    matrix_row["mapped_sample_count"] = mapped_count
    matrix_row["unmapped_sample_count"] = unmapped_count
    feature_axis_rows = feature_rows_for_matrix(matrix_row["matrix_uid"], study_row, feature_entries)
    member_rows = [
        create_member_row(
            matrix_row["matrix_uid"],
            study_row,
            asset_row,
            "",
            "",
            None,
            "not_applicable",
            len(feature_entries),
            "metabolomics_data_json_file",
        )
    ]
    return [matrix_row], member_rows, sample_axis_rows, feature_axis_rows, []


def parse_metabolomics_mwtab_sections(local_path: Path) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    current_section = ""
    current_units = ""
    in_section = False
    section_index = 0
    sample_identifiers: list[str] = []
    feature_entries: list[tuple[str, str]] = []
    with local_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line in {"#MS_METABOLITE_DATA", "#NMR_METABOLITE_DATA"}:
                current_section = line.removeprefix("#")
                current_units = ""
                continue
            if current_section and line.startswith(f"{current_section}:UNITS"):
                parts = line.split("\t", 1)
                current_units = parts[1].strip() if len(parts) == 2 else ""
                continue
            if current_section and line == f"{current_section}_START":
                in_section = True
                sample_identifiers = []
                feature_entries = []
                continue
            if current_section and line == f"{current_section}_END":
                section_index += 1
                sections.append(
                    {
                        "section_name": current_section,
                        "section_index": section_index,
                        "units": current_units,
                        "sample_identifiers": sample_identifiers,
                        "feature_entries": feature_entries,
                    }
                )
                current_section = ""
                current_units = ""
                in_section = False
                continue
            if not in_section:
                continue
            values = line.split("\t")
            if not sample_identifiers:
                if not values or values[0] != "Samples":
                    raise ValueError(f"Expected Samples header in mwtab section for {local_path}")
                sample_identifiers = [value.strip() for value in values[1:] if value.strip()]
                continue
            if values and values[0] == "Factors":
                continue
            if not values or not values[0].strip():
                continue
            feature_id = values[0].strip()
            feature_entries.append((feature_id, feature_id))
    return sections


def parse_metabolomics_mwtab(
    study_row: dict[str, str],
    asset_row: dict[str, str],
    sample_lookup_by_study: dict[str, dict[str, dict[str, str]]],
    config: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    local_path = resolve_repo_path(asset_row["local_path"])
    sections = parse_metabolomics_mwtab_sections(local_path)
    if not sections:
        return [], [], [], [], [
            non_matrix_row(
                asset_row,
                config["feature_ready_registry"]["excluded_asset_reasons"]["missing_metabolomics_mwtab_section"],
                "Metabolomics Workbench mwtab file does not contain an MS_METABOLITE_DATA or NMR_METABOLITE_DATA section.",
            )
        ]

    matrix_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    sample_axis_rows: list[dict[str, object]] = []
    feature_axis_rows: list[dict[str, object]] = []

    for section in sections:
        matrix_name = (
            f"{Path(asset_row['local_path']).stem}_"
            f"{slugify(str(section['section_name']))}_"
            f"section_{section['section_index']}"
        )
        matrix_row = create_matrix_row(
            study_row,
            asset_row,
            matrix_name,
            "metabolomics_mwtab_section",
            str(section["section_name"]),
            "",
            path_suffix_label(asset_row["local_path"]),
            len(section["sample_identifiers"]),
            len(section["feature_entries"]),
            0,
            0,
            "Samples",
            "metabolite_name",
            "metabolite_name",
            str(section["section_name"]),
            (
                f"{section['section_name']} section {section['section_index']}; {str(section['units']).strip()}"
                if str(section["units"]).strip()
                else f"{section['section_name']} section {section['section_index']}"
            ),
        )
        current_sample_axis_rows, mapped_count, unmapped_count = sample_axis_rows_for_matrix(
            matrix_row["matrix_uid"],
            study_row,
            list(section["sample_identifiers"]),
            sample_lookup_by_study,
        )
        matrix_row["mapped_sample_count"] = mapped_count
        matrix_row["unmapped_sample_count"] = unmapped_count
        matrix_rows.append(matrix_row)
        sample_axis_rows.extend(current_sample_axis_rows)
        feature_axis_rows.extend(feature_rows_for_matrix(matrix_row["matrix_uid"], study_row, list(section["feature_entries"])))
        member_rows.append(
            create_member_row(
                matrix_row["matrix_uid"],
                study_row,
                asset_row,
                "",
                "",
                None,
                "not_applicable",
                len(section["feature_entries"]),
                f"{section['section_name']} section {section['section_index']}",
            )
        )

    return matrix_rows, member_rows, sample_axis_rows, feature_axis_rows, []


def build_feature_ready_outputs(config: dict, source_accessions: set[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    processed_root = resolve_repo_path(config["cohort_ready_registry"]["local_root"])
    sample_catalog_path = processed_root / config["cohort_ready_registry"]["relative_paths"]["sample_catalog"]
    asset_catalog_path = processed_root / config["cohort_ready_registry"]["relative_paths"]["quantitative_asset_catalog"]
    study_summary_path = processed_root / config["cohort_ready_registry"]["relative_paths"]["study_summary"]

    sample_rows = read_tsv(sample_catalog_path)
    asset_rows = read_tsv(asset_catalog_path)
    study_rows = read_tsv(study_summary_path)
    enabled_sources = set(config["feature_ready_registry"]["enabled_sources"])

    sample_rows = [row for row in sample_rows if row["source_key"] in enabled_sources]
    asset_rows = [
        row
        for row in asset_rows
        if row["source_key"] in enabled_sources and row["local_availability_flag"] == "yes" and row["local_path"]
    ]
    study_rows = [row for row in study_rows if row["source_key"] in enabled_sources]
    if source_accessions:
        sample_rows = [row for row in sample_rows if row["source_accession"] in source_accessions]
        asset_rows = [row for row in asset_rows if row["source_accession"] in source_accessions]
        study_rows = [row for row in study_rows if row["source_accession"] in source_accessions]

    samples_by_study, sample_lookup_by_study = build_sample_lookup(sample_rows)
    assets_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)
    for asset_row in asset_rows:
        assets_by_study[asset_row["study_uid"]].append(asset_row)

    matrix_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    sample_axis_rows: list[dict[str, object]] = []
    feature_axis_rows: list[dict[str, object]] = []
    non_matrix_rows: list[dict[str, object]] = []

    analyzable_assets = config["feature_ready_registry"]["analyzable_assets"]
    excluded_asset_reasons = config["feature_ready_registry"]["excluded_asset_reasons"]

    for study_row in study_rows:
        study_assets = assets_by_study.get(study_row["study_uid"], [])
        for asset_row in study_assets:
            if asset_row["asset_name"] not in analyzable_assets[study_row["source_key"]]:
                exclusion_reason = excluded_asset_reasons.get(asset_row["asset_name"], "not_selected_for_feature_ready")
                non_matrix_rows.append(non_matrix_row(asset_row, exclusion_reason, asset_row["note"]))
                continue
            if study_row["source_key"] == "geo":
                current_matrix_rows, current_member_rows, current_sample_axis_rows, current_feature_axis_rows, current_non_matrix_rows = parse_geo_series_matrix(
                    study_row,
                    asset_row,
                    sample_lookup_by_study,
                    config,
                )
            elif study_row["source_key"] == "arrayexpress":
                current_matrix_rows, current_member_rows, current_sample_axis_rows, current_feature_axis_rows, current_non_matrix_rows = parse_arrayexpress_asset(
                    study_row,
                    asset_row,
                    sample_lookup_by_study,
                    config,
                )
            elif study_row["source_key"] == "metabolomics_workbench" and asset_row["asset_name"] == "data_json":
                current_matrix_rows, current_member_rows, current_sample_axis_rows, current_feature_axis_rows, current_non_matrix_rows = parse_metabolomics_data_json(
                    study_row,
                    asset_row,
                    sample_lookup_by_study,
                    config,
                )
            elif study_row["source_key"] == "metabolomics_workbench" and asset_row["asset_name"] == "mwtab_txt":
                current_matrix_rows, current_member_rows, current_sample_axis_rows, current_feature_axis_rows, current_non_matrix_rows = parse_metabolomics_mwtab(
                    study_row,
                    asset_row,
                    sample_lookup_by_study,
                    config,
                )
            else:
                current_matrix_rows, current_member_rows, current_sample_axis_rows, current_feature_axis_rows, current_non_matrix_rows = (
                    [],
                    [],
                    [],
                    [],
                    [non_matrix_row(asset_row, "unsupported_feature_ready_asset", asset_row["note"])],
                )

            if not current_matrix_rows and not current_non_matrix_rows:
                current_non_matrix_rows = [non_matrix_row(asset_row, "empty_feature_ready_parse", asset_row["note"])]
            matrix_rows.extend(current_matrix_rows)
            member_rows.extend(current_member_rows)
            sample_axis_rows.extend(current_sample_axis_rows)
            feature_axis_rows.extend(current_feature_axis_rows)
            non_matrix_rows.extend(current_non_matrix_rows)

    study_summary_rows = build_feature_ready_study_summary(
        study_rows,
        samples_by_study,
        matrix_rows,
        sample_axis_rows,
        feature_axis_rows,
        non_matrix_rows,
        config,
    )
    matrix_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["matrix_name"], row["measurement_label"]))
    member_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["matrix_uid"], row["member_path"], row["source_sample_id_hint"]))
    sample_axis_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["matrix_uid"], int(row["column_index"])))
    feature_axis_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["matrix_uid"], int(row["row_index"])))
    non_matrix_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["asset_name"], row["local_path"]))
    return matrix_rows, member_rows, sample_axis_rows, feature_axis_rows, non_matrix_rows, study_summary_rows


def choose_primary_matrix(
    source_key: str,
    study_matrix_rows: list[dict[str, object]],
    config: dict,
) -> tuple[str, str]:
    if not study_matrix_rows:
        return "", ""
    priority_list = config["feature_ready_registry"]["primary_measurement_priority"][source_key]
    ordered_rows = sorted(
        study_matrix_rows,
        key=lambda row: (
            priority_list.index(str(row["measurement_label"]))
            if str(row["measurement_label"]) in priority_list
            else len(priority_list),
            -int(row["mapped_sample_count"]),
            -int(row["feature_axis_count"]),
            str(row["matrix_name"]),
        ),
    )
    primary_row = ordered_rows[0]
    return str(primary_row["matrix_uid"]), str(primary_row["matrix_reference"])


def build_feature_ready_study_summary(
    study_rows: list[dict[str, str]],
    samples_by_study: dict[str, list[dict[str, str]]],
    matrix_rows: list[dict[str, object]],
    sample_axis_rows: list[dict[str, object]],
    feature_axis_rows: list[dict[str, object]],
    non_matrix_rows: list[dict[str, object]],
    config: dict,
) -> list[dict[str, object]]:
    matrix_rows_by_study: dict[str, list[dict[str, object]]] = defaultdict(list)
    sample_axis_rows_by_study: dict[str, list[dict[str, object]]] = defaultdict(list)
    feature_axis_rows_by_study: dict[str, list[dict[str, object]]] = defaultdict(list)
    non_matrix_rows_by_study: dict[str, list[dict[str, object]]] = defaultdict(list)
    mapped_sample_uids_by_study: dict[str, set[str]] = defaultdict(set)
    unmapped_sample_count_by_study: dict[str, int] = defaultdict(int)

    for matrix_row in matrix_rows:
        matrix_rows_by_study[str(matrix_row["study_uid"])].append(matrix_row)
    for sample_axis_row in sample_axis_rows:
        study_uid = str(sample_axis_row["study_uid"])
        sample_axis_rows_by_study[study_uid].append(sample_axis_row)
        if sample_axis_row["sample_uid"]:
            mapped_sample_uids_by_study[study_uid].add(str(sample_axis_row["sample_uid"]))
            continue
        unmapped_sample_count_by_study[study_uid] += 1
    for feature_axis_row in feature_axis_rows:
        feature_axis_rows_by_study[str(feature_axis_row["study_uid"])].append(feature_axis_row)
    for non_matrix_row_value in non_matrix_rows:
        non_matrix_rows_by_study[str(non_matrix_row_value["study_uid"])].append(non_matrix_row_value)

    for study_uid, study_matrix_rows in matrix_rows_by_study.items():
        matrix_ids = {str(matrix_row["matrix_uid"]) for matrix_row in study_matrix_rows}
        for sample_axis_row in sample_axis_rows_by_study.get(study_uid, []):
            if str(sample_axis_row["matrix_uid"]) not in matrix_ids:
                raise ValueError(f"Sample axis row refers to unknown matrix in study {study_uid}")
        for feature_axis_row in feature_axis_rows_by_study.get(study_uid, []):
            if str(feature_axis_row["matrix_uid"]) not in matrix_ids:
                raise ValueError(f"Feature axis row refers to unknown matrix in study {study_uid}")

    study_summary_rows: list[dict[str, object]] = []
    for study_row in study_rows:
        study_uid = study_row["study_uid"]
        study_matrix_rows = matrix_rows_by_study.get(study_uid, [])
        matrix_asset_count = len({str(row["quantitative_asset_uid"]) for row in study_matrix_rows})
        primary_matrix_uid, primary_matrix_reference = choose_primary_matrix(study_row["source_key"], study_matrix_rows, config)
        study_summary_rows.append(
            {
                "study_uid": study_uid,
                "source_key": study_row["source_key"],
                "source_accession": study_row["source_accession"],
                "standardized_sample_count": len(samples_by_study.get(study_uid, [])),
                "matrix_ready_flag": "yes" if study_matrix_rows else "no",
                "matrix_count": len(study_matrix_rows),
                "matrix_asset_count": matrix_asset_count,
                "non_matrix_asset_count": len(non_matrix_rows_by_study.get(study_uid, [])),
                "mapped_sample_uid_count": len(mapped_sample_uids_by_study.get(study_uid, set())),
                "unmapped_matrix_sample_count": unmapped_sample_count_by_study.get(study_uid, 0),
                "feature_axis_row_count": len(feature_axis_rows_by_study.get(study_uid, [])),
                "primary_matrix_uid": primary_matrix_uid,
                "primary_matrix_path": primary_matrix_reference,
            }
        )
    study_summary_rows.sort(key=lambda row: (row["source_key"], row["source_accession"]))
    return study_summary_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    source_accessions = selected_accessions(args.source_accessions)
    feature_ready_root = resolve_repo_path(config["feature_ready_registry"]["local_root"])
    ensure_directory(feature_ready_root)

    matrix_rows, member_rows, sample_axis_rows, feature_axis_rows, non_matrix_rows, study_summary_rows = build_feature_ready_outputs(config, source_accessions)
    pipeline_logger.progress(
        "loaded_feature_ready_inputs",
        matrix_row_count=len(matrix_rows),
        member_row_count=len(member_rows),
        sample_axis_row_count=len(sample_axis_rows),
        feature_axis_row_count=len(feature_axis_rows),
        non_matrix_row_count=len(non_matrix_rows),
        study_row_count=len(study_summary_rows),
    )

    relative_paths = config["feature_ready_registry"]["relative_paths"]
    write_tsv(
        feature_ready_root / relative_paths["matrix_catalog"],
        config["schemas"]["feature_ready_matrix_fields"],
        matrix_rows,
    )
    write_tsv(
        feature_ready_root / relative_paths["member_catalog"],
        config["schemas"]["feature_ready_member_fields"],
        member_rows,
    )
    write_tsv(
        feature_ready_root / relative_paths["sample_axis_catalog"],
        config["schemas"]["feature_ready_sample_axis_fields"],
        sample_axis_rows,
    )
    write_tsv(
        feature_ready_root / relative_paths["feature_axis_catalog"],
        config["schemas"]["feature_ready_feature_axis_fields"],
        feature_axis_rows,
    )
    write_tsv(
        feature_ready_root / relative_paths["non_matrix_asset_catalog"],
        config["schemas"]["feature_ready_non_matrix_fields"],
        non_matrix_rows,
    )
    write_tsv(
        feature_ready_root / relative_paths["study_summary"],
        config["schemas"]["feature_ready_study_fields"],
        study_summary_rows,
    )
    pipeline_logger.artifact(
        "feature_ready_registry_written",
        artifact=feature_ready_root,
        matrix_row_count=len(matrix_rows),
        member_row_count=len(member_rows),
        sample_axis_row_count=len(sample_axis_rows),
        feature_axis_row_count=len(feature_axis_rows),
        non_matrix_row_count=len(non_matrix_rows),
        study_row_count=len(study_summary_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=feature_ready_root,
        matrix_row_count=len(matrix_rows),
        member_row_count=len(member_rows),
        sample_axis_row_count=len(sample_axis_rows),
        feature_axis_row_count=len(feature_axis_rows),
        non_matrix_row_count=len(non_matrix_rows),
        study_row_count=len(study_summary_rows),
    )

if __name__ == "__main__":
    main()
