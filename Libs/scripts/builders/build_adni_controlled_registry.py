from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.controlled_research_config import (
    assert_controlled_governance_approval,
    load_controlled_research_config,
)
from Libs.utils.data.raw_data_registry import ensure_directory, repo_root, sha256_of_file, write_json, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build controlled-access ADNI registries from private local exports.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def adni_section(config: dict) -> dict:
    return config["adni_controlled_access"]


def resolve_configured_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def controlled_paths(config: dict) -> tuple[Path, Path]:
    paths_config = adni_section(config)["paths"]
    return (
        resolve_configured_path(paths_config["private_root"]),
        resolve_configured_path(paths_config["processed_root"]),
    )


def display_path(path: Path) -> str:
    resolved_path = path.resolve()
    root = repo_root()
    if resolved_path.is_relative_to(root):
        return resolved_path.relative_to(root).as_posix()
    return str(resolved_path)


def hash_salt(config: dict) -> str:
    env_name = str(adni_section(config)["credentials"]["hash_salt_env"])
    salt_value = os.environ.get(env_name)
    if not salt_value:
        raise RuntimeError(f"Missing required ADNI hash-salt environment variable: {env_name}")
    return salt_value


def build_governance_evidence_report(config: dict) -> Path:
    private_root, _ = controlled_paths(config)
    paths_config = adni_section(config)["paths"]
    governance_config = adni_section(config)["governance"]
    evidence_config = governance_config["evidence"]
    governance_root = private_root / str(evidence_config["governance_root"])
    document_rows: list[dict[str, object]] = []
    missing_paths: list[str] = []
    for document_config in evidence_config["required_documents"]:
        document_path = governance_root / str(document_config["relative_path"])
        present = document_path.exists()
        if bool(document_config["required"]) and not present:
            missing_paths.append(display_path(document_path))
        document_rows.append(
            {
                "document_id": str(document_config["document_id"]),
                "document_kind": str(document_config["document_kind"]),
                "private_path": display_path(document_path),
                "required": bool(document_config["required"]),
                "present": present,
                "sha256": sha256_of_file(document_path) if present else "",
            }
        )
    report_path = private_root / paths_config["access_governance_report"]
    payload = {
        "source_key": adni_section(config)["source_key"],
        "access_boundary": governance_config["access_boundary"],
        "participant_level_flag": "aggregate_governance_only",
        "document_status": "complete" if not missing_paths else "missing_required_documents",
        "governance_root": display_path(governance_root),
        "required_document_count": len(document_rows),
        "missing_required_document_paths": missing_paths,
        "documents": document_rows,
        "approved_scope_flags": dict(evidence_config["approved_scope_flags"]),
        "storage_policy": dict(evidence_config["storage_policy"]),
        "acknowledgement_publication_policy": dict(evidence_config["acknowledgement_publication_policy"]),
        "raw_redistribution_allowed": bool(governance_config["raw_redistribution_allowed"]),
        "derived_participant_level_redistribution_allowed": bool(governance_config["derived_participant_level_redistribution_allowed"]),
        "public_release_policy": governance_config["public_release_policy"],
    }
    write_json(report_path, payload)
    if missing_paths:
        raise FileNotFoundError(f"Missing required ADNI governance documents: {missing_paths}")
    return report_path


def stable_hash(config: dict, salt_value: str, identifier_kind: str, raw_value: str) -> str:
    prefix_length = int(adni_section(config)["builder"]["hash_prefix_length"])
    digest = hashlib.sha256(f"{salt_value}|adni|{identifier_kind}|{raw_value}".encode("utf-8")).hexdigest()
    return digest[:prefix_length]


def missing_tokens(config: dict) -> set[str]:
    return {str(value).strip().lower() for value in adni_section(config)["builder"]["missing_tokens"]}


def is_missing(value: object, config: dict) -> bool:
    return str(value).strip().lower() in missing_tokens(config)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def first_present_value(row: dict[str, object], columns: list[str], config: dict) -> str:
    for column in columns:
        if column in row and not is_missing(row[column], config):
            return normalize_text(row[column])
    return ""


def existing_column(headers: set[str], columns: list[str]) -> str:
    for column in columns:
        if column in headers:
            return column
    return ""


def require_column(headers: set[str], columns: list[str], semantic_name: str, table_role: str) -> str:
    column = existing_column(headers, columns)
    if not column:
        raise ValueError(f"ADNI {table_role} table is missing the required {semantic_name} column.")
    return column


def read_table(path: Path, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False)


def load_table(config: dict, table_key: str) -> pd.DataFrame:
    controlled_config = adni_section(config)
    private_root, _ = controlled_paths(config)
    table_config = controlled_config["builder"]["table_inputs"][table_key]
    table_path = private_root / table_config["local_relative_path"]
    if not table_path.exists():
        if bool(table_config["required"]):
            raise FileNotFoundError(f"Missing required ADNI controlled-access table for role {table_config['table_role']}.")
        return pd.DataFrame()
    return read_table(table_path, str(table_config["delimiter"]))


def table_key_by_role(config: dict) -> dict[str, str]:
    return {
        str(table_config["table_role"]): str(table_key)
        for table_key, table_config in adni_section(config)["builder"]["table_inputs"].items()
    }


def role_metadata(config: dict, table_role: str) -> dict[str, object]:
    return adni_section(config)["sync"]["role_registry"][table_role]


def generic_subject_column(frame: pd.DataFrame, config: dict, table_role: str) -> str:
    builder_config = adni_section(config)["builder"]
    headers = set(frame.columns)
    if table_role == "apoe_genetics":
        return require_column(headers, builder_config["apoe_table"]["subject_identifier"], "subject identifier", table_role)
    if table_role == "medications":
        return require_column(headers, builder_config["medication_table"]["subject_identifier"], "subject identifier", table_role)
    return require_column(headers, builder_config["generic_table_columns"]["subject_identifier"], "subject identifier", table_role)


def generic_visit_column(frame: pd.DataFrame, config: dict, table_role: str) -> str:
    builder_config = adni_section(config)["builder"]
    headers = set(frame.columns)
    if table_role == "medications":
        return existing_column(headers, builder_config["medication_table"]["visit_code"])
    return existing_column(headers, builder_config["generic_table_columns"]["visit_code"])


def row_groups_by_role(config: dict) -> tuple[dict[tuple[str, str, str], list[dict[str, object]]], dict[tuple[str, str], list[dict[str, object]]]]:
    rows_by_visit: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    rows_by_subject: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for table_key, table_config in adni_section(config)["builder"]["table_inputs"].items():
        table_role = str(table_config["table_role"])
        if table_role == "adni_merge_core":
            continue
        frame = load_table(config, str(table_key))
        if frame.empty:
            continue
        subject_column = generic_subject_column(frame, config, table_role)
        visit_column = generic_visit_column(frame, config, table_role)
        for row in frame.to_dict(orient="records"):
            subject_value = first_present_value(row, [subject_column], config)
            if not subject_value:
                continue
            visit_value = first_present_value(row, [visit_column], config) if visit_column else ""
            if visit_value:
                rows_by_visit[(table_role, subject_value, visit_value)].append(row)
            rows_by_subject[(table_role, subject_value)].append(row)
    return rows_by_visit, rows_by_subject


def rows_for_role(
    table_role: str,
    raw_subject: str,
    visit_code: str,
    rows_by_visit: dict[tuple[str, str, str], list[dict[str, object]]],
    rows_by_subject: dict[tuple[str, str], list[dict[str, object]]],
) -> list[dict[str, object]]:
    visit_rows = rows_by_visit.get((table_role, raw_subject, visit_code), [])
    if visit_rows:
        return visit_rows
    return rows_by_subject.get((table_role, raw_subject), [])


def first_present_from_rows(rows: list[dict[str, object]], columns: list[str], config: dict) -> str:
    for row in rows:
        value = first_present_value(row, columns, config)
        if value:
            return value
    return ""


def first_present_from_role_rows(
    role_rows: dict[str, list[dict[str, object]]],
    table_roles: list[str],
    columns: list[str],
    config: dict,
) -> str:
    for table_role in table_roles:
        value = first_present_from_rows(role_rows.get(table_role, []), columns, config)
        if value:
            return value
    return ""


def normalized_diagnosis(raw_value: str, config: dict) -> str:
    value = normalize_text(raw_value)
    if not value:
        return ""
    lower_value = value.lower()
    normalization_map = adni_section(config)["builder"]["diagnosis_normalization_map"]
    if lower_value in normalization_map:
        return str(normalization_map[lower_value])
    return value


def diagnosis_numeric(diagnosis: str, config: dict) -> str:
    numeric_map = adni_section(config)["builder"]["diagnosis_numeric_map"]
    if diagnosis in numeric_map:
        return str(numeric_map[diagnosis])
    return ""


def site_from_participant_identifier(participant_identifier: str, config: dict) -> str:
    pattern = adni_section(config)["builder"]["site_identifier_from_ptid_regex"]
    match = re.match(pattern, participant_identifier)
    if match is None:
        return ""
    return match.group(1)


def visit_month(visit_code: str, exam_date: str, first_exam_date: pd.Timestamp | None, config: dict) -> str:
    builder_config = adni_section(config)["builder"]
    if exam_date and first_exam_date is not None:
        parsed_date = pd.to_datetime(exam_date, format=str(builder_config["date_format"]))
        month_delta = int(round((parsed_date - first_exam_date).days / 30.4375))
        return str(month_delta)
    visit_map = builder_config["followup_month_from_visit_code"]
    key = visit_code.strip().lower()
    if key in visit_map:
        return str(visit_map[key])
    return ""


def split_for_subject(subject_uid_hash: str, config: dict) -> str:
    split_policy = adni_section(config)["builder"]["split_policy"]
    context = str(split_policy["assignment_hash_context"])
    digest = hashlib.sha256(f"{context}|{subject_uid_hash}".encode("utf-8")).hexdigest()
    ratio = int(digest[:12], 16) / float(16**12)
    train_fraction = float(split_policy["train_fraction"])
    validation_fraction = float(split_policy["validation_fraction"])
    if ratio < train_fraction:
        return "train"
    if ratio < train_fraction + validation_fraction:
        return "validation"
    return "test"


def apoe_lookup(config: dict, salt_value: str) -> dict[str, str]:
    frame = load_table(config, "apoe")
    if frame.empty:
        return {}
    table_config = adni_section(config)["builder"]["apoe_table"]
    headers = set(frame.columns)
    subject_column = require_column(headers, table_config["subject_identifier"], "subject identifier", "apoe")
    apoe_count_column = existing_column(headers, table_config["apoe4_count"])
    genotype_column = existing_column(headers, table_config.get("genotype", []))
    allele_one_column = existing_column(headers, table_config["allele_one"])
    allele_two_column = existing_column(headers, table_config["allele_two"])
    lookup: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        subject_value = first_present_value(row, [subject_column], config)
        if not subject_value:
            continue
        if apoe_count_column:
            apoe_count = first_present_value(row, [apoe_count_column], config)
        elif genotype_column:
            genotype_value = first_present_value(row, [genotype_column], config)
            apoe_count = str(sum(1 for allele in re.findall(r"\d+", genotype_value) if allele == "4"))
        else:
            alleles = [
                first_present_value(row, [allele_one_column], config),
                first_present_value(row, [allele_two_column], config),
            ]
            apoe_count = str(sum(1 for allele in alleles if allele == "4"))
        lookup[stable_hash(config, salt_value, "subject", subject_value)] = apoe_count
    return lookup


def modality_from_text(value: str, config: dict) -> str:
    normalized_value = value.upper()
    for modality, aliases in adni_section(config)["builder"]["modality_aliases"].items():
        if any(str(alias).upper() in normalized_value for alias in aliases):
            return str(modality)
    return ""


def load_imaging_rows(config: dict, salt_value: str) -> tuple[list[dict[str, object]], dict[tuple[str, str], set[str]]]:
    frame = load_table(config, "imaging_inventory")
    if frame.empty:
        return [], defaultdict(set)
    builder_config = adni_section(config)["builder"]
    column_config = builder_config["imaging_inventory_columns"]
    headers = set(frame.columns)
    subject_column = require_column(headers, column_config["subject_identifier"], "subject identifier", "imaging_inventory")
    participant_column = existing_column(headers, column_config["participant_identifier"])
    visit_column = require_column(headers, column_config["visit_code"], "visit code", "imaging_inventory")
    modality_column = existing_column(headers, column_config["modality"])
    image_uid_column = existing_column(headers, column_config["image_uid"])
    sequence_column = existing_column(headers, column_config["sequence"])
    collection_column = existing_column(headers, column_config["collection"])
    path_column = existing_column(headers, column_config["local_relative_path"])
    checksum_column = existing_column(headers, column_config["checksum_sha256"])
    rows: list[dict[str, object]] = []
    modality_by_visit: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_key = adni_section(config)["source_key"]
    cohort_column = existing_column(headers, ["ADNI_COHORT", "COLPROT", "Phase", "phase"])
    for index, row in enumerate(frame.to_dict(orient="records")):
        raw_subject = first_present_value(row, [subject_column], config)
        visit_code = first_present_value(row, [visit_column], config)
        if not raw_subject or not visit_code:
            continue
        participant_identifier = first_present_value(row, [participant_column], config) if participant_column else ""
        raw_modality = first_present_value(row, [modality_column], config) if modality_column else str(role_metadata(config, "t1_inventory")["modality"])
        modality = modality_from_text(raw_modality, config) or str(role_metadata(config, "t1_inventory")["modality"])
        subject_uid_hash = stable_hash(config, salt_value, "subject", raw_subject)
        visit_uid = f"{builder_config['visit_uid_prefix']}_{stable_hash(config, salt_value, 'visit', f'{raw_subject}|{visit_code}')}"
        raw_image_uid = first_present_value(row, [image_uid_column], config) if image_uid_column else f"{raw_subject}|{visit_code}|{index}"
        cohort_value = first_present_value(row, [cohort_column], config) if cohort_column else ""
        modality_by_visit[(subject_uid_hash, visit_code)].add(modality)
        rows.append(
            {
                "image_uid_hash": f"{builder_config['image_uid_prefix']}_{stable_hash(config, salt_value, 'image', raw_image_uid)}",
                "subject_uid_hash": subject_uid_hash,
                "visit_uid": visit_uid,
                "source_key": source_key,
                "adni_cohort": cohort_value,
                "visit_code": visit_code,
                "modality": modality,
                "sequence": first_present_value(row, [sequence_column], config) if sequence_column else "",
                "collection": first_present_value(row, [collection_column], config) if collection_column else "",
                "local_path_private": first_present_value(row, [path_column], config) if path_column else "",
                "checksum_sha256": first_present_value(row, [checksum_column], config) if checksum_column else "",
                "participant_level_flag": "yes",
                "allowed_use": "validation_internal_only",
            }
        )
    return rows, modality_by_visit


def first_exam_dates(frame: pd.DataFrame, subject_column: str, exam_date_column: str, config: dict) -> dict[str, pd.Timestamp]:
    if not exam_date_column:
        return {}
    builder_config = adni_section(config)["builder"]
    dates: dict[str, list[pd.Timestamp]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        subject_value = first_present_value(row, [subject_column], config)
        date_value = first_present_value(row, [exam_date_column], config)
        if subject_value and date_value:
            dates[subject_value].append(pd.to_datetime(date_value, format=str(builder_config["date_format"])))
    return {subject_value: min(values) for subject_value, values in dates.items() if values}


def numeric_or_empty(value: str) -> str:
    if not value:
        return ""
    return str(float(value))


def adjusted_volume_value(raw_value: str, intracranial_volume: str, config: dict) -> str:
    if not raw_value or not intracranial_volume:
        return ""
    reference_volume = float(adni_section(config)["builder"]["icv_adjustment"]["reference_intracranial_volume"])
    return str(float(raw_value) / float(intracranial_volume) * reference_volume)


def feature_value(
    role_rows: dict[str, list[dict[str, object]]],
    feature_config: dict,
    apoe4_count: str,
    active_medication_count: str,
    config: dict,
) -> str:
    if feature_config["feature_name"] == "apoe4_count" and apoe4_count:
        return apoe4_count
    if feature_config["feature_name"] == "active_medication_count":
        return active_medication_count
    table_roles = [str(value) for value in feature_config.get("source_table_roles", ["adni_merge_core"])]
    source_columns = [str(value) for value in feature_config["source_columns"]]
    raw_value = first_present_from_role_rows(role_rows, table_roles, source_columns, config)
    if str(feature_config["normalization_policy"]) == "icv_adjust_then_train_split_zscore":
        intracranial_volume = first_present_from_role_rows(
            role_rows,
            ["mri_freesurfer", "adni_merge_core"],
            ["ICV", "IntracranialVol"],
            config,
        )
        return adjusted_volume_value(raw_value, intracranial_volume, config)
    return raw_value


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def numeric_text(value: str) -> str:
    if not value:
        return ""
    return str(float(value))


def numeric_feature_float(config: dict, value: str) -> float:
    parsing_config = adni_section(config)["builder"]["numeric_value_parsing"]
    numeric_text_value = value.strip()
    separator = str(parsing_config.get("thousands_separator", ""))
    if separator:
        numeric_text_value = numeric_text_value.replace(separator, "")
    if str(parsing_config["censored_bound_policy"]) == "use_reported_boundary_value":
        for prefix in sorted([str(item) for item in parsing_config["censored_bound_prefixes"]], key=len, reverse=True):
            if numeric_text_value.startswith(prefix):
                numeric_text_value = numeric_text_value[len(prefix):].strip()
                break
    if not re.fullmatch(r"[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?", numeric_text_value):
        raise ValueError(f"ADNI controlled numeric feature value is not parseable under configured policy: {value}")
    return float(numeric_text_value)


def build_medication_exposure_rows(
    config: dict,
    salt_value: str,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], int]]:
    frame = load_table(config, "medications")
    if frame.empty:
        return [], {}
    builder_config = adni_section(config)["builder"]
    table_config = builder_config["medication_table"]
    headers = set(frame.columns)
    subject_column = require_column(headers, table_config["subject_identifier"], "subject identifier", "medications")
    visit_column = existing_column(headers, table_config["visit_code"])
    medication_column = require_column(headers, table_config["medication_name"], "medication name", "medications")
    exposure_column = existing_column(headers, table_config["exposure_status"])
    start_column = existing_column(headers, table_config["start_date"])
    rows: list[dict[str, object]] = []
    count_by_visit: dict[tuple[str, str], int] = defaultdict(int)
    source_key = adni_section(config)["source_key"]
    for index, row in enumerate(frame.to_dict(orient="records")):
        raw_subject = first_present_value(row, [subject_column], config)
        visit_code = first_present_value(row, [visit_column], config) if visit_column else ""
        medication_name = first_present_value(row, [medication_column], config)
        if not raw_subject or not medication_name:
            continue
        subject_uid_hash = stable_hash(config, salt_value, "subject", raw_subject)
        visit_uid = f"{builder_config['visit_uid_prefix']}_{stable_hash(config, salt_value, 'visit', f'{raw_subject}|{visit_code}')}"
        sample_uid = f"{builder_config['sample_uid_prefix']}_{stable_hash(config, salt_value, 'sample', f'{raw_subject}|{visit_code}')}"
        count_by_visit[(raw_subject, visit_code)] += 1
        rows.append(
            {
                "exposure_uid_hash": stable_hash(config, salt_value, "medication_exposure", f"{raw_subject}|{visit_code}|{medication_name}|{index}"),
                "subject_uid_hash": subject_uid_hash,
                "visit_uid": visit_uid,
                "sample_uid": sample_uid,
                "source_key": source_key,
                "visit_code": visit_code,
                "medication_name_hash": stable_hash(config, salt_value, "medication_name", medication_name),
                "exposure_status": first_present_value(row, [exposure_column], config) if exposure_column else "recorded",
                "exposure_timepoint": first_present_value(row, [start_column], config) if start_column else "",
                "participant_level_flag": "yes",
                "allowed_use": "validation_internal_only",
            }
        )
    return rows, count_by_visit


def build_domain_inventory_rows(config: dict, salt_value: str) -> list[dict[str, object]]:
    builder_config = adni_section(config)["builder"]
    generic_columns = builder_config["generic_table_columns"]
    rows: list[dict[str, object]] = []
    source_key = adni_section(config)["source_key"]
    excluded_roles = {"adni_merge_core", "visit_diagnosis", "demographics", "cognitive_scores", "medications"}
    for table_key, table_config in builder_config["table_inputs"].items():
        table_role = str(table_config["table_role"])
        if table_role in excluded_roles:
            continue
        frame = load_table(config, str(table_key))
        if frame.empty:
            continue
        role_config = role_metadata(config, table_role)
        headers = set(frame.columns)
        subject_column = generic_subject_column(frame, config, table_role)
        visit_column = generic_visit_column(frame, config, table_role)
        inventory_column = existing_column(headers, generic_columns["inventory_uid"])
        modality_column = existing_column(headers, generic_columns["modality"])
        assay_column = existing_column(headers, generic_columns["assay_name"])
        availability_column = existing_column(headers, generic_columns["availability_flag"])
        for index, row in enumerate(frame.to_dict(orient="records")):
            raw_subject = first_present_value(row, [subject_column], config)
            if not raw_subject:
                continue
            visit_code = first_present_value(row, [visit_column], config) if visit_column else ""
            inventory_value = first_present_value(row, [inventory_column], config) if inventory_column else f"{table_role}|{index}"
            assay_value = first_present_value(row, [assay_column], config) if assay_column else table_role
            modality_value = first_present_value(row, [modality_column], config) if modality_column else str(role_config["modality"])
            modality = modality_from_text(modality_value, config) or str(role_config["modality"])
            rows.append(
                {
                    "inventory_uid_hash": stable_hash(config, salt_value, "domain_inventory", f"{table_role}|{inventory_value}|{index}"),
                    "subject_uid_hash": stable_hash(config, salt_value, "subject", raw_subject),
                    "visit_uid": f"{builder_config['visit_uid_prefix']}_{stable_hash(config, salt_value, 'visit', f'{raw_subject}|{visit_code}')}",
                    "source_key": source_key,
                    "table_role": table_role,
                    "data_domain": str(role_config["data_domain"]),
                    "modality": modality,
                    "visit_code": visit_code,
                    "inventory_label": assay_value,
                    "availability_flag": first_present_value(row, [availability_column], config) if availability_column else "available",
                    "participant_level_flag": "yes",
                    "allowed_use": "validation_internal_only",
                }
            )
    return rows


def build_sample_and_feature_rows(
    config: dict,
    salt_value: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[dict[str, object]]], dict[str, dict[str, str]]]:
    clinical_frame = load_table(config, "clinical_core")
    builder_config = adni_section(config)["builder"]
    column_config = builder_config["column_aliases"]
    headers = set(clinical_frame.columns)
    subject_column = require_column(headers, column_config["subject_identifier"], "subject identifier", "clinical_core")
    participant_column = existing_column(headers, column_config["participant_identifier"])
    visit_column = require_column(headers, column_config["visit_code"], "visit code", "clinical_core")
    diagnosis_column = existing_column(headers, column_config["diagnosis"])
    exam_date_column = existing_column(headers, column_config["exam_date"])
    site_column = existing_column(headers, column_config["site_identifier"])
    cohort_column = existing_column(headers, ["ADNI_COHORT", "COLPROT", "Phase", "phase"])
    first_dates = first_exam_dates(clinical_frame, subject_column, exam_date_column, config)
    apoe_by_subject = apoe_lookup(config, salt_value)
    imaging_rows, imaging_modalities = load_imaging_rows(config, salt_value)
    rows_by_visit, rows_by_subject = row_groups_by_role(config)
    _, medication_count_by_visit = build_medication_exposure_rows(config, salt_value)
    sample_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    samples_by_subject: dict[str, list[dict[str, object]]] = defaultdict(list)
    feature_lookup: dict[str, dict[str, str]] = defaultdict(dict)
    source_key = adni_section(config)["source_key"]
    for row in clinical_frame.to_dict(orient="records"):
        raw_subject = first_present_value(row, [subject_column], config)
        visit_code = first_present_value(row, [visit_column], config)
        if not raw_subject or not visit_code:
            continue
        subject_uid_hash = stable_hash(config, salt_value, "subject", raw_subject)
        visit_uid = f"{builder_config['visit_uid_prefix']}_{stable_hash(config, salt_value, 'visit', f'{raw_subject}|{visit_code}')}"
        sample_uid = f"{builder_config['sample_uid_prefix']}_{stable_hash(config, salt_value, 'sample', f'{raw_subject}|{visit_code}')}"
        split_name = split_for_subject(subject_uid_hash, config)
        role_rows: dict[str, list[dict[str, object]]] = {"adni_merge_core": [row]}
        for table_role in table_key_by_role(config):
            if table_role == "adni_merge_core":
                continue
            role_rows[table_role] = rows_for_role(table_role, raw_subject, visit_code, rows_by_visit, rows_by_subject)
        participant_identifier = first_present_from_role_rows(
            role_rows,
            ["adni_merge_core", "demographics"],
            column_config["participant_identifier"],
            config,
        ) if participant_column else ""
        explicit_site = first_present_from_role_rows(
            role_rows,
            ["adni_merge_core", "demographics"],
            column_config["site_identifier"],
            config,
        ) if site_column else ""
        site_identifier = explicit_site or site_from_participant_identifier(participant_identifier, config)
        site_uid_hash = stable_hash(config, salt_value, "site", site_identifier) if site_identifier else ""
        diagnosis_source_roles = ["adni_merge_core", "visit_diagnosis"]
        diagnosis_columns = column_config["diagnosis"]
        diagnosis = normalized_diagnosis(
            first_present_from_role_rows(role_rows, diagnosis_source_roles, diagnosis_columns, config)
            if diagnosis_column else first_present_from_role_rows(role_rows, ["visit_diagnosis"], diagnosis_columns, config),
            config,
        )
        exam_date = first_present_value(row, [exam_date_column], config) if exam_date_column else ""
        followup_month = visit_month(visit_code, exam_date, first_dates.get(raw_subject), config)
        cohort_value = first_present_value(row, [cohort_column], config) if cohort_column else ""
        apoe4_count = first_present_from_role_rows(
            role_rows,
            ["adni_merge_core", "apoe_genetics"],
            column_config["apoe4_count"],
            config,
        ) or apoe_by_subject.get(subject_uid_hash, "")
        active_medication_count = str(medication_count_by_visit.get((raw_subject, visit_code), 0))
        modality_flags = {modality: False for modality in builder_config["modality_output_order"]}
        modality_flags["clinical"] = True
        if apoe4_count:
            modality_flags["apoe"] = True
            modality_flags["genomics"] = True
        for modality in imaging_modalities.get((subject_uid_hash, visit_code), set()):
            if modality:
                modality_flags[modality] = True
        sample_feature_values: dict[str, str] = {}
        for feature_config in builder_config["feature_definitions"]:
            value = feature_value(role_rows, feature_config, apoe4_count, active_medication_count, config)
            missingness_flag = yes_no(not value)
            if value:
                modality_flags[str(feature_config["modality_flag"])] = True
            feature_rows.append(
                {
                    "sample_uid": sample_uid,
                    "feature_name": feature_config["feature_name"],
                    "feature_value": value,
                    "feature_unit": feature_config["unit"],
                    "feature_timepoint": followup_month,
                    "missingness_flag": missingness_flag,
                    "normalization_policy": feature_config["normalization_policy"],
                }
            )
            sample_feature_values[str(feature_config["feature_name"])] = value
        clinical_score_available = any(sample_feature_values.get(name, "") for name in ["mmse_total", "cdrsb_total", "adas_cog", "faq_total", "npi_total"])
        available_modalities = ";".join(modality for modality in builder_config["modality_output_order"] if modality_flags[modality])
        age_value = first_present_from_role_rows(role_rows, ["adni_merge_core", "demographics"], column_config["age"], config)
        sex_value = first_present_from_role_rows(role_rows, ["adni_merge_core", "demographics"], column_config["sex"], config)
        education_value = first_present_from_role_rows(role_rows, ["adni_merge_core", "demographics"], column_config["education"], config)
        sample_row = {
            "sample_uid": sample_uid,
            "subject_uid_hash": subject_uid_hash,
            "visit_uid": visit_uid,
            "source_key": source_key,
            "adni_cohort": cohort_value,
            "visit_code": visit_code,
            "exam_date_relative": followup_month,
            "diagnosis": diagnosis,
            "diagnosis_numeric": diagnosis_numeric(diagnosis, config),
            "age": age_value,
            "sex": sex_value,
            "education": education_value,
            "apoe4_count": apoe4_count,
            "site_uid_hash": site_uid_hash,
            "available_modalities": available_modalities,
            "clinical_score_available": yes_no(clinical_score_available),
            "mri_available": yes_no(modality_flags["mri"]),
            "pet_amyloid_available": yes_no(modality_flags["pet_amyloid"]),
            "pet_tau_available": yes_no(modality_flags["pet_tau"]),
            "fdg_pet_available": yes_no(modality_flags["fdg_pet"]),
            "csf_available": yes_no(modality_flags["csf"]),
            "plasma_available": yes_no(modality_flags["plasma"]),
            "genomics_available": yes_no(modality_flags["genomics"]),
            "followup_month": followup_month,
            "conversion_event": "no",
            "benchmark_split": split_name,
        }
        sample_rows.append(sample_row)
        samples_by_subject[subject_uid_hash].append(sample_row)
        feature_lookup[sample_uid] = sample_feature_values
    for subject_rows in samples_by_subject.values():
        ordered_rows = sorted(subject_rows, key=lambda item: float(item["followup_month"] or "0"))
        previous_numeric = ""
        for sample_row in ordered_rows:
            current_numeric = str(sample_row["diagnosis_numeric"])
            if previous_numeric and current_numeric and int(float(current_numeric)) > int(float(previous_numeric)):
                sample_row["conversion_event"] = "yes"
            previous_numeric = current_numeric
    return sample_rows, feature_rows, samples_by_subject, feature_lookup


def transition_label(source_dx: str, target_dx: str) -> str:
    if not source_dx or not target_dx:
        return ""
    source_value = int(float(source_dx))
    target_value = int(float(target_dx))
    if target_value > source_value:
        return "progression"
    if target_value < source_value:
        return "improvement"
    return "stable"


def build_transition_rows(
    samples_by_subject: dict[str, list[dict[str, object]]],
    feature_lookup: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject_uid_hash, subject_rows in samples_by_subject.items():
        ordered_rows = sorted(subject_rows, key=lambda item: float(item["followup_month"] or "0"))
        for source_row, target_row in zip(ordered_rows[:-1], ordered_rows[1:]):
            source_month = float(source_row["followup_month"] or "0")
            target_month = float(target_row["followup_month"] or "0")
            delta_month = target_month - source_month
            source_features = feature_lookup[str(source_row["sample_uid"])]
            target_features = feature_lookup[str(target_row["sample_uid"])]
            source_dx = str(source_row["diagnosis_numeric"])
            target_dx = str(target_row["diagnosis_numeric"])
            progression_rate = ""
            if delta_month and source_dx and target_dx:
                progression_rate = str((float(target_dx) - float(source_dx)) / delta_month)
            rows.append(
                {
                    "subject_uid_hash": subject_uid_hash,
                    "source_sample_uid": source_row["sample_uid"],
                    "target_sample_uid": target_row["sample_uid"],
                    "source_visit_uid": source_row["visit_uid"],
                    "target_visit_uid": target_row["visit_uid"],
                    "delta_month": numeric_text(str(delta_month)),
                    "dx_t0": source_row["diagnosis"],
                    "dx_t1": target_row["diagnosis"],
                    "mmse_t0": source_features.get("mmse_total", ""),
                    "mmse_t1": target_features.get("mmse_total", ""),
                    "cdrsb_t0": source_features.get("cdrsb_total", ""),
                    "cdrsb_t1": target_features.get("cdrsb_total", ""),
                    "adas_t0": source_features.get("adas_cog", ""),
                    "adas_t1": target_features.get("adas_cog", ""),
                    "amyloid_t0": source_features.get("amyloid_suvr", ""),
                    "amyloid_t1": target_features.get("amyloid_suvr", ""),
                    "tau_t0": source_features.get("tau_suvr", ""),
                    "tau_t1": target_features.get("tau_suvr", ""),
                    "mri_region_features_t0_path": "",
                    "mri_region_features_t1_path": "",
                    "transition_label": transition_label(source_dx, target_dx),
                    "progression_rate": progression_rate,
                    "benchmark_split": source_row["benchmark_split"],
                }
            )
    return rows


def missingness_audit_payload(config: dict, sample_rows: list[dict[str, object]], feature_rows: list[dict[str, object]]) -> dict[str, object]:
    feature_counts: dict[str, dict[str, int]] = {}
    for row in feature_rows:
        feature_name = str(row["feature_name"])
        feature_counts.setdefault(feature_name, {"present": 0, "missing": 0})
        key = "missing" if row["missingness_flag"] == "yes" else "present"
        feature_counts[feature_name][key] += 1
    modality_counts = {
        key: sum(1 for row in sample_rows if row[f"{key}_available"] == "yes")
        for key in ["mri", "pet_amyloid", "pet_tau", "fdg_pet", "csf", "plasma", "genomics"]
    }
    return {
        "source_key": adni_section(config)["source_key"],
        "participant_level_flag": "aggregate_only",
        "sample_count": len(sample_rows),
        "feature_missingness": feature_counts,
        "modality_available_sample_counts": modality_counts,
    }


def split_audit_payload(config: dict, sample_rows: list[dict[str, object]], transition_rows: list[dict[str, object]]) -> dict[str, object]:
    subjects_by_split: dict[str, set[str]] = defaultdict(set)
    samples_per_split: dict[str, int] = defaultdict(int)
    sites_by_split: dict[str, set[str]] = defaultdict(set)
    site_sample_counts_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cohorts_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    diagnosis_counts_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    modality_missingness_by_split: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"available": 0, "missing": 0}))
    modality_pattern_counts_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    modality_keys = ["mri", "pet_amyloid", "pet_tau", "fdg_pet", "csf", "plasma", "genomics"]
    for row in sample_rows:
        split_name = str(row["benchmark_split"])
        subjects_by_split[split_name].add(str(row["subject_uid_hash"]))
        samples_per_split[split_name] += 1
        if row["site_uid_hash"]:
            sites_by_split[split_name].add(str(row["site_uid_hash"]))
            site_sample_counts_by_split[split_name][str(row["site_uid_hash"])] += 1
        if row["adni_cohort"]:
            cohorts_by_split[split_name][str(row["adni_cohort"])] += 1
        if row["diagnosis"]:
            diagnosis_counts_by_split[split_name][str(row["diagnosis"])] += 1
        pattern_tokens: list[str] = []
        for modality in modality_keys:
            availability_key = f"{modality}_available"
            available = str(row[availability_key]) == "yes"
            modality_missingness_by_split[split_name][modality]["available" if available else "missing"] += 1
            pattern_tokens.append(f"{modality}:{'yes' if available else 'no'}")
        modality_pattern_counts_by_split[split_name][";".join(pattern_tokens)] += 1
    leakage_count = 0
    subject_splits: dict[str, set[str]] = defaultdict(set)
    for row in sample_rows:
        subject_splits[str(row["subject_uid_hash"])].add(str(row["benchmark_split"]))
    for splits in subject_splits.values():
        if len(splits) > 1:
            leakage_count += 1
    site_overlap_matrix: dict[str, dict[str, int]] = {}
    for left_split, left_sites in sites_by_split.items():
        site_overlap_matrix[left_split] = {
            right_split: len(left_sites & right_sites)
            for right_split, right_sites in sites_by_split.items()
        }
    transition_counts: dict[str, int] = defaultdict(int)
    transition_counts_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in transition_rows:
        transition_counts[str(row["transition_label"])] += 1
        transition_counts_by_split[str(row["benchmark_split"])][str(row["transition_label"])] += 1
    site_count_distribution_by_split = {
        split_name: sorted(counts.values(), reverse=True)
        for split_name, counts in site_sample_counts_by_split.items()
    }
    return {
        "split_policy_uid": adni_section(config)["builder"]["split_policy"]["policy_uid"],
        "participant_level_flag": "aggregate_only",
        "subject_holdout_leakage_count": leakage_count,
        "subjects_per_split": {split_name: len(subjects) for split_name, subjects in subjects_by_split.items()},
        "samples_per_split": dict(samples_per_split),
        "unique_site_counts_by_split": {split_name: len(sites) for split_name, sites in sites_by_split.items()},
        "site_sample_count_distribution_by_split": site_count_distribution_by_split,
        "site_overlap_matrix": site_overlap_matrix,
        "cohort_counts_by_split": {split_name: dict(counts) for split_name, counts in cohorts_by_split.items()},
        "adni_phase_distribution_by_split": {split_name: dict(counts) for split_name, counts in cohorts_by_split.items()},
        "diagnosis_counts_by_split": {split_name: dict(counts) for split_name, counts in diagnosis_counts_by_split.items()},
        "diagnosis_transition_counts": dict(transition_counts),
        "diagnosis_transition_counts_by_split": {split_name: dict(counts) for split_name, counts in transition_counts_by_split.items()},
        "modality_missingness_by_split": {
            split_name: {modality: dict(counts) for modality, counts in modality_counts.items()}
            for split_name, modality_counts in modality_missingness_by_split.items()
        },
        "modality_missingness_pattern_counts_by_split": {
            split_name: dict(counts)
            for split_name, counts in modality_pattern_counts_by_split.items()
        },
    }


def write_parquet_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    temporary_path = path.parent / f".{path.name}.{os.getpid()}.part"
    dataframe.to_parquet(temporary_path, index=False, engine="pyarrow")
    temporary_path.replace(path)


def model_input_registry_payload(config: dict) -> dict[str, object]:
    paths_config = adni_section(config)["paths"]
    return {
        "source_key": adni_section(config)["source_key"],
        "benchmark_role": adni_section(config)["benchmark_role"],
        "access_boundary": adni_section(config)["governance"]["access_boundary"],
        "participant_level_policy": adni_section(config)["governance"]["participant_level_policy"],
        "public_release_policy": adni_section(config)["governance"]["public_release_policy"],
        "processed_root": paths_config["processed_root"],
        "artifacts": {
            "sample_observation_registry": paths_config["sample_observation_registry"],
            "longitudinal_transition_table": paths_config["longitudinal_transition_table"],
            "feature_matrix": paths_config["feature_matrix"],
            "imaging_inventory": paths_config["imaging_inventory"],
            "medication_exposure": paths_config["medication_exposure"],
            "domain_inventory": paths_config["domain_inventory"],
            "access_governance_report": paths_config["access_governance_report"],
            "missingness_audit": paths_config["missingness_audit"],
            "split_audit": paths_config["split_audit"],
            "sample_index": paths_config["sample_index"],
            "feature_index": paths_config["feature_index"],
            "matrix_raw": paths_config["matrix_raw"],
            "matrix_norm": paths_config["matrix_norm"],
            "matrix_missing_mask": paths_config["matrix_missing_mask"],
            "train_scaler": paths_config["train_scaler"],
            "transition_label_index": paths_config["transition_label_index"],
            "controlled_application_audit": paths_config["controlled_application_audit"],
        },
        "schemas": {
            "sample_observation": "schemas.adni_sample_observation_fields",
            "longitudinal_transition": "schemas.adni_longitudinal_transition_fields",
            "feature_matrix": "schemas.adni_feature_matrix_fields",
            "imaging_inventory": "schemas.adni_imaging_inventory_fields",
            "medication_exposure": "schemas.adni_medication_exposure_fields",
            "domain_inventory": "schemas.adni_domain_inventory_fields",
            "feature_index": "schemas.adni_feature_index_fields",
        },
    }


def table_role_row_counts(config: dict) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    for table_key, table_config in adni_section(config)["builder"]["table_inputs"].items():
        frame = load_table(config, str(table_key))
        row_counts[str(table_config["table_role"])] = int(frame.shape[0])
    return row_counts


def feature_index_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_config in adni_section(config)["builder"]["feature_definitions"]:
        rows.append(
            {
                "feature_name": str(feature_config["feature_name"]),
                "feature_group": str(feature_config["feature_group"]),
                "feature_unit": str(feature_config["unit"]),
                "modality_flag": str(feature_config["modality_flag"]),
                "normalization_policy": str(feature_config["normalization_policy"]),
                "source_table_roles": ";".join(str(value) for value in feature_config.get("source_table_roles", [])),
                "source_columns": ";".join(str(value) for value in feature_config["source_columns"]),
            }
        )
    return rows


def matrix_from_feature_rows(
    config: dict,
    sample_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    sample_uids = [str(row["sample_uid"]) for row in sample_rows]
    feature_names = [str(row["feature_name"]) for row in adni_section(config)["builder"]["feature_definitions"]]
    sample_index = {sample_uid: index for index, sample_uid in enumerate(sample_uids)}
    feature_index = {feature_name: index for index, feature_name in enumerate(feature_names)}
    matrix = np.full((len(sample_uids), len(feature_names)), np.nan, dtype=np.float64)
    missing_mask = np.ones((len(sample_uids), len(feature_names)), dtype=np.uint8)
    for row in feature_rows:
        sample_uid = str(row["sample_uid"])
        feature_name = str(row["feature_name"])
        value = str(row["feature_value"])
        if sample_uid not in sample_index or feature_name not in feature_index or not value:
            continue
        matrix[sample_index[sample_uid], feature_index[feature_name]] = numeric_feature_float(config, value)
        missing_mask[sample_index[sample_uid], feature_index[feature_name]] = 0
    return feature_names, matrix, missing_mask


def train_only_scaler_payload(
    config: dict,
    sample_rows: list[dict[str, object]],
    feature_names: list[str],
    matrix_raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    builder_config = adni_section(config)["builder"]
    feature_configs = {str(row["feature_name"]): row for row in builder_config["feature_definitions"]}
    train_indices = [index for index, row in enumerate(sample_rows) if row["benchmark_split"] == "train"]
    if not train_indices:
        raise ValueError("ADNI controlled model-input scaler requires at least one train-split sample.")
    matrix_norm = np.zeros_like(matrix_raw, dtype=np.float64)
    scaler_rows: list[dict[str, object]] = []
    imputation_value = float(builder_config["numeric_imputation_value"])
    for feature_index, feature_name in enumerate(feature_names):
        policy = str(feature_configs[feature_name]["normalization_policy"])
        train_values = matrix_raw[train_indices, feature_index]
        present_train_values = train_values[np.isfinite(train_values)]
        if present_train_values.size == 0:
            raise ValueError(f"ADNI controlled model-input feature has no train-split observed values: {feature_name}")
        if policy == "none":
            mean_value = 0.0
            scale_value = 1.0
        else:
            mean_value = float(present_train_values.mean())
            standard_deviation = float(present_train_values.std(ddof=0))
            scale_value = standard_deviation if standard_deviation > 0.0 else 1.0
        present_mask = np.isfinite(matrix_raw[:, feature_index])
        matrix_norm[present_mask, feature_index] = (matrix_raw[present_mask, feature_index] - mean_value) / scale_value
        matrix_norm[~present_mask, feature_index] = imputation_value
        scaler_rows.append(
            {
                "feature_name": feature_name,
                "normalization_policy": policy,
                "fit_split": "train",
                "mean": mean_value,
                "scale": scale_value,
                "train_nonmissing_count": int(present_train_values.size),
            }
        )
    return matrix_norm, {
        "source_key": adni_section(config)["source_key"],
        "participant_level_flag": "yes_private_model_input",
        "fit_split": "train",
        "train_sample_count": len(train_indices),
        "feature_count": len(feature_names),
        "scaler_rows": scaler_rows,
    }


def write_npy_atomic(path: Path, array: np.ndarray) -> None:
    ensure_directory(path.parent)
    temporary_path = path.parent / f".{path.name}.{os.getpid()}.part"
    with temporary_path.open("wb") as handle:
        np.save(handle, array)
    temporary_path.replace(path)


def controlled_application_audit_payload(
    config: dict,
    sample_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
    medication_rows: list[dict[str, object]],
    domain_inventory_rows: list[dict[str, object]],
    scaler_payload: dict[str, object],
) -> dict[str, object]:
    table_row_counts = table_role_row_counts(config)
    return {
        "source_key": adni_section(config)["source_key"],
        "benchmark_role": adni_section(config)["benchmark_role"],
        "participant_level_flag": "aggregate_only",
        "sample_count": len(sample_rows),
        "feature_row_count": len(feature_rows),
        "transition_count": len(transition_rows),
        "medication_exposure_count": len(medication_rows),
        "domain_inventory_count": len(domain_inventory_rows),
        "required_table_roles": [str(value) for value in adni_section(config)["sync"]["required_table_roles"]],
        "required_table_role_count": len(adni_section(config)["sync"]["required_table_roles"]),
        "table_role_row_counts": table_row_counts,
        "train_only_scaler_fit_split": scaler_payload["fit_split"],
        "train_only_scaler_feature_count": scaler_payload["feature_count"],
        "application_benchmark_id": adni_section(config)["application"]["benchmark_id"],
    }


def split_counts(sample_rows: list[dict[str, object]], transition_rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    sample_counts: dict[str, int] = defaultdict(int)
    transition_counts: dict[str, int] = defaultdict(int)
    for row in sample_rows:
        sample_counts[str(row["benchmark_split"])] += 1
    for row in transition_rows:
        transition_counts[str(row["benchmark_split"])] += 1
    return {
        "samples": dict(sample_counts),
        "transitions": dict(transition_counts),
    }


def controlled_bundle_pointer_payload(
    config: dict,
    sample_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
    output_paths: dict[str, Path],
) -> dict[str, object]:
    paths_config = adni_section(config)["paths"]
    private_root, processed_root = controlled_paths(config)
    role_counts = table_role_row_counts(config)
    artifact_keys = [
        "sample_observation_registry",
        "longitudinal_transition_table",
        "feature_matrix",
        "imaging_inventory",
        "medication_exposure",
        "domain_inventory",
        "model_input_registry",
        "controlled_application_audit",
        "state_projection_report",
        "world_state_predictions",
    ]
    return {
        "report_kind": "adni_controlled_bundle_pointer",
        "source_key": adni_section(config)["source_key"],
        "benchmark_role": adni_section(config)["benchmark_role"],
        "access_boundary": "controlled_access_aggregate_pointer",
        "participant_level_flag": "aggregate_only",
        "public_release_policy": adni_section(config)["governance"]["public_release_policy"],
        "participant_level_registry_policy": "do_not_append_adni_participants_to_public_sample_or_model_input_registries",
        "private_roots": {
            "private_root": display_path(private_root),
            "processed_root": display_path(processed_root),
        },
        "private_artifact_paths": {
            artifact_key: display_path(output_paths.get(artifact_key, processed_root / paths_config[artifact_key]))
            for artifact_key in artifact_keys
            if artifact_key in paths_config
        },
        "schema_ids": {
            "sample_observation": "schemas.adni_sample_observation_fields",
            "longitudinal_transition": "schemas.adni_longitudinal_transition_fields",
            "feature_matrix": "schemas.adni_feature_matrix_fields",
            "imaging_inventory": "schemas.adni_imaging_inventory_fields",
            "medication_exposure": "schemas.adni_medication_exposure_fields",
            "domain_inventory": "schemas.adni_domain_inventory_fields",
            "world_state_predictions": "schemas.adni_world_state_prediction_fields",
        },
        "required_role_count": len(adni_section(config)["sync"]["required_table_roles"]),
        "role_counts": role_counts,
        "split_counts": split_counts(sample_rows, transition_rows),
        "sample_count": len(sample_rows),
        "transition_count": len(transition_rows),
        "public_aggregate_outputs": {
            "state_inference_summary": paths_config["state_inference_summary"],
            "longitudinal_validation_summary": paths_config["longitudinal_validation_summary"],
            "missing_modality_robustness": paths_config["missing_modality_robustness"],
            "world_state_factor_anchoring": paths_config["world_state_factor_anchoring"],
            "posterior_calibration_summary": paths_config["posterior_calibration_summary"],
        },
    }


def write_model_input_bundle(
    config: dict,
    processed_root: Path,
    sample_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    transition_rows: list[dict[str, object]],
) -> tuple[dict[str, Path], dict[str, object]]:
    paths_config = adni_section(config)["paths"]
    sample_index_path = processed_root / paths_config["sample_index"]
    feature_index_path = processed_root / paths_config["feature_index"]
    matrix_raw_path = processed_root / paths_config["matrix_raw"]
    matrix_norm_path = processed_root / paths_config["matrix_norm"]
    missing_mask_path = processed_root / paths_config["matrix_missing_mask"]
    train_scaler_path = processed_root / paths_config["train_scaler"]
    transition_label_path = processed_root / paths_config["transition_label_index"]
    write_parquet_atomic(pd.DataFrame(sample_rows, columns=config["schemas"]["adni_sample_observation_fields"]), sample_index_path)
    write_parquet_atomic(pd.DataFrame(feature_index_rows(config), columns=config["schemas"]["adni_feature_index_fields"]), feature_index_path)
    write_parquet_atomic(pd.DataFrame(transition_rows, columns=config["schemas"]["adni_longitudinal_transition_fields"]), transition_label_path)
    feature_names, matrix_raw, missing_mask = matrix_from_feature_rows(config, sample_rows, feature_rows)
    matrix_norm, scaler_payload = train_only_scaler_payload(config, sample_rows, feature_names, matrix_raw)
    write_npy_atomic(matrix_raw_path, matrix_raw)
    write_npy_atomic(matrix_norm_path, matrix_norm)
    write_npy_atomic(missing_mask_path, missing_mask)
    write_json(train_scaler_path, scaler_payload)
    return {
        "sample_index": sample_index_path,
        "feature_index": feature_index_path,
        "matrix_raw": matrix_raw_path,
        "matrix_norm": matrix_norm_path,
        "matrix_missing_mask": missing_mask_path,
        "train_scaler": train_scaler_path,
        "transition_label_index": transition_label_path,
    }, scaler_payload


def build_adni_registry_artifacts(config: dict) -> dict[str, Path]:
    governance_report_path = build_governance_evidence_report(config)
    salt_value = hash_salt(config)
    _, processed_root = controlled_paths(config)
    ensure_directory(processed_root)
    paths_config = adni_section(config)["paths"]
    sample_rows, feature_rows, samples_by_subject, feature_lookup = build_sample_and_feature_rows(config, salt_value)
    transition_rows = build_transition_rows(samples_by_subject, feature_lookup)
    imaging_rows, _ = load_imaging_rows(config, salt_value)
    medication_rows, _ = build_medication_exposure_rows(config, salt_value)
    domain_inventory_rows = build_domain_inventory_rows(config, salt_value)
    output_paths = {
        "sample_observation_registry": processed_root / paths_config["sample_observation_registry"],
        "longitudinal_transition_table": processed_root / paths_config["longitudinal_transition_table"],
        "feature_matrix": processed_root / paths_config["feature_matrix"],
        "imaging_inventory": processed_root / paths_config["imaging_inventory"],
        "medication_exposure": processed_root / paths_config["medication_exposure"],
        "domain_inventory": processed_root / paths_config["domain_inventory"],
        "missingness_audit": processed_root / paths_config["missingness_audit"],
        "split_audit": processed_root / paths_config["split_audit"],
        "model_input_registry": processed_root / paths_config["model_input_registry"],
        "controlled_application_audit": processed_root / paths_config["controlled_application_audit"],
        "access_governance_report": governance_report_path,
        "controlled_bundle_pointer": resolve_configured_path(paths_config["controlled_bundle_pointer"]),
    }
    write_tsv(output_paths["sample_observation_registry"], config["schemas"]["adni_sample_observation_fields"], sample_rows)
    write_tsv(output_paths["longitudinal_transition_table"], config["schemas"]["adni_longitudinal_transition_fields"], transition_rows)
    write_tsv(output_paths["imaging_inventory"], config["schemas"]["adni_imaging_inventory_fields"], imaging_rows)
    write_tsv(output_paths["medication_exposure"], config["schemas"]["adni_medication_exposure_fields"], medication_rows)
    write_tsv(output_paths["domain_inventory"], config["schemas"]["adni_domain_inventory_fields"], domain_inventory_rows)
    write_parquet_atomic(pd.DataFrame(feature_rows, columns=config["schemas"]["adni_feature_matrix_fields"]), output_paths["feature_matrix"])
    bundle_paths, scaler_payload = write_model_input_bundle(config, processed_root, sample_rows, feature_rows, transition_rows)
    output_paths.update(bundle_paths)
    write_json(output_paths["missingness_audit"], missingness_audit_payload(config, sample_rows, feature_rows))
    write_json(output_paths["split_audit"], split_audit_payload(config, sample_rows, transition_rows))
    write_json(output_paths["model_input_registry"], model_input_registry_payload(config))
    write_json(output_paths["controlled_application_audit"], controlled_application_audit_payload(
        config,
        sample_rows,
        feature_rows,
        transition_rows,
        medication_rows,
        domain_inventory_rows,
        scaler_payload,
    ))
    write_json(
        output_paths["controlled_bundle_pointer"],
        controlled_bundle_pointer_payload(config, sample_rows, transition_rows, output_paths),
    )
    return output_paths


def main() -> None:
    args = parse_args()
    config = load_controlled_research_config(args.config)
    assert_controlled_governance_approval(config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    output_paths = build_adni_registry_artifacts(config)
    pipeline_logger.progress(
        "adni_registry_built",
        artifact_count=len(output_paths),
        access_boundary=adni_section(config)["governance"]["access_boundary"],
    )
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact_count=len(output_paths),
    )


if __name__ == "__main__":
    main()
