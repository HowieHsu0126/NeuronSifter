from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.evaluation.alignment_benchmark import (
    load_bundle_artifacts,
    load_model_input_catalog,
    modality_from_sample_index,
    normalized_metabolite_label,
    relative_repo_path,
    tissue_from_sample_index,
    write_npy_atomic,
    write_parquet_atomic,
)
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, normalize_text, resolve_repo_path, write_json, write_tsv


MISSING_TOKENS = {"", "na", "nan", "null", "none", "---"}
ENTREZ_PATTERN = re.compile(r"\b\d{2,10}\b")
GENE_SYMBOL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9@._-]{0,31}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build alignment-ready study bundles from model-input bundles.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--source-accessions",
        default="",
        help="Optional comma-separated source_accession subset for smoke tests.",
    )
    return parser.parse_args()


def selected_accessions(accession_text: str) -> set[str]:
    return {value.strip() for value in accession_text.split(",") if value.strip()}


def filtered_rows(rows: list[dict[str, str]], source_accessions: set[str]) -> list[dict[str, str]]:
    if not source_accessions:
        return rows
    return [row for row in rows if row["source_accession"] in source_accessions]


def normalize_matrix_by_modality(raw_matrix: np.ndarray, modality: str) -> np.ndarray:
    transformed = raw_matrix.copy()
    finite_mask = np.isfinite(transformed)
    if modality == "metabolomics" and np.any(finite_mask) and np.min(transformed[finite_mask]) >= 0:
        transformed[finite_mask] = np.log1p(transformed[finite_mask])
    counts = finite_mask.sum(axis=0).astype(np.int64)
    sums = np.where(finite_mask, transformed, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(transformed.shape[1], dtype=np.float64), where=counts > 0)
    centered = np.where(finite_mask, transformed - means, 0.0)
    variances = np.divide((centered**2).sum(axis=0), counts, out=np.zeros(transformed.shape[1], dtype=np.float64), where=counts > 0)
    stds = np.sqrt(variances)
    normalized = np.full(transformed.shape, np.nan, dtype=np.float64)
    nonzero_mask = (counts > 0) & (stds > 0)
    zero_mask = (counts > 0) & (stds == 0)
    if np.any(nonzero_mask):
        normalized[:, nonzero_mask] = (transformed[:, nonzero_mask] - means[nonzero_mask]) / stds[nonzero_mask]
    if np.any(zero_mask):
        normalized[:, zero_mask] = np.where(finite_mask[:, zero_mask], 0.0, np.nan)
    normalized[~finite_mask] = np.nan
    return normalized


def candidate_column(columns: list[str], preferred_names: list[str]) -> str:
    direct_lookup = {normalize_text(column): column for column in columns}
    for preferred_name in preferred_names:
        normalized_name = normalize_text(preferred_name)
        if normalized_name in direct_lookup:
            return direct_lookup[normalized_name]
    for preferred_name in preferred_names:
        normalized_name = normalize_text(preferred_name)
        for normalized_column, original_column in direct_lookup.items():
            if normalized_name and normalized_name in normalized_column:
                return original_column
    return ""


def tokenized_values(value: str) -> list[str]:
    cleaned = value.replace("///", "|").replace("//", "|").replace(";", "|").replace(",", "|")
    return [token.strip().strip('"').strip("'") for token in cleaned.split("|") if token.strip().strip('"').strip("'")]


def is_missing(value: str) -> bool:
    return normalize_text(value) in MISSING_TOKENS


def is_control_feature_text(value: str) -> bool:
    normalized_value = normalize_text(value)
    return any(keyword in normalized_value for keyword in ["control", "spike", "affx"])


def is_gene_symbol_token(token: str) -> bool:
    if not GENE_SYMBOL_PATTERN.match(token):
        return False
    lowered = token.lower()
    if lowered.startswith(("nm_", "nr_", "xm_", "xr_", "ensg", "chr")):
        return False
    return True


def extract_unique_entrez(value: str) -> str:
    if is_missing(value):
        return ""
    hits = sorted({match.group(0) for match in ENTREZ_PATTERN.finditer(value)})
    if len(hits) != 1:
        return ""
    return hits[0]


def extract_assignment_pair(value: str) -> tuple[str, str]:
    if is_missing(value):
        return "", ""
    pairs: list[tuple[str, str]] = []
    for chunk in value.split("///"):
        tokens = [token.strip() for token in chunk.split("//")]
        if len(tokens) < 5:
            continue
        candidate_symbol = tokens[1]
        candidate_entrez = tokens[4]
        if is_gene_symbol_token(candidate_symbol) and candidate_entrez.isdigit():
            pairs.append((candidate_entrez, candidate_symbol))
    unique_gene_ids = sorted({gene_id for gene_id, _ in pairs})
    if len(unique_gene_ids) != 1:
        return "", ""
    for gene_id, symbol in pairs:
        if gene_id == unique_gene_ids[0]:
            return gene_id, symbol
    return "", ""


def extract_symbol(symbol_text: str, assignment_text: str) -> str:
    for token in tokenized_values(symbol_text):
        if is_gene_symbol_token(token):
            return token
    assignment_tokens = tokenized_values(assignment_text)
    for index, token in enumerate(assignment_tokens):
        if ENTREZ_PATTERN.fullmatch(token):
            for previous_token in reversed(assignment_tokens[max(0, index - 3) : index]):
                if is_gene_symbol_token(previous_token):
                    return previous_token
    for token in assignment_tokens:
        if is_gene_symbol_token(token):
            return token
    return ""


def read_gpl_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        in_table = False
        header: list[str] = []
        rows: list[dict[str, str]] = []
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            if line == "!platform_table_end":
                break
            if not in_table:
                continue
            values = next(csv.reader([line], delimiter="\t"))
            if not header:
                header = values
                continue
            if len(values) < len(header):
                values.extend([""] * (len(header) - len(values)))
            rows.append(dict(zip(header, values)))
    if not header:
        raise ValueError(f"Platform table header missing in {path}")
    return header, rows


def read_adf_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        main_index = lines.index("[main]")
    except ValueError as exc:
        raise ValueError(f"ADF file lacks [main] table: {path}") from exc
    if main_index + 1 >= len(lines):
        raise ValueError(f"ADF file lacks header row after [main]: {path}")
    header = next(csv.reader([lines[main_index + 1]], delimiter="\t"))
    rows: list[dict[str, str]] = []
    for line in lines[main_index + 2 :]:
        if not line.strip():
            continue
        if line.startswith("[") and line.endswith("]"):
            break
        values = next(csv.reader([line], delimiter="\t"))
        if len(values) < len(header):
            values.extend([""] * (len(header) - len(values)))
        rows.append(dict(zip(header, values)))
    return header, rows


def read_plain_table_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = [dict(zip(header, values)) for values in reader]
    return header, rows


def read_gpl_miniml_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    platform_accession = path.name.split("_", 1)[0]
    with tarfile.open(path, "r:gz") as archive:
        member_name = ""
        for current_name in archive.getnames():
            if current_name == f"{platform_accession}-tbl-1.txt":
                member_name = current_name
                break
        if not member_name:
            for current_name in archive.getnames():
                if current_name.endswith("-tbl-1.txt") and current_name.startswith(platform_accession):
                    member_name = current_name
                    break
        if not member_name:
            raise ValueError(f"MINiML archive lacks platform table for {platform_accession}: {path}")
        member_handle = archive.extractfile(member_name)
        if member_handle is None:
            raise ValueError(f"Unable to extract platform table {member_name} from {path}")
        text_handle = io.TextIOWrapper(member_handle, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(text_handle, delimiter="\t")
        first_row = next(reader)
        if first_row and first_row[0].isdigit():
            header = [f"column_{index}" for index in range(len(first_row))]
            rows = [dict(zip(header, first_row))]
        else:
            header = first_row
            rows = []
        for values in reader:
            rows.append(dict(zip(header, values)))
    return header, rows


def gene_reference_maps(config: dict) -> tuple[dict[str, str], dict[str, str]]:
    geo_root = resolve_repo_path(config["sources"]["geo"]["local_root"])
    gene_reference_path = (
        geo_root
        / config["annotation_companion_downloads"]["geo"]["gene_reference_relative_dir"]
        / "Homo_sapiens.gene_info.gz"
    )
    if not gene_reference_path.exists():
        raise FileNotFoundError(f"Missing Homo sapiens gene reference companion: {gene_reference_path}")
    symbol_to_gene_ids: dict[str, set[str]] = defaultdict(set)
    gene_id_to_symbol: dict[str, str] = {}
    with gzip.open(gene_reference_path, "rt", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene_id = row["GeneID"].strip()
            symbol = row["Symbol"].strip()
            if gene_id and symbol and normalize_text(symbol) not in MISSING_TOKENS:
                symbol_to_gene_ids[symbol].add(gene_id)
                gene_id_to_symbol.setdefault(gene_id, symbol)
            synonyms = [token.strip() for token in row.get("Synonyms", "").split("|") if token.strip() and token.strip() != "-"]
            for synonym in synonyms:
                symbol_to_gene_ids[synonym].add(gene_id)
    symbol_to_gene_id = {
        symbol: sorted(gene_ids)[0]
        for symbol, gene_ids in symbol_to_gene_ids.items()
        if len(gene_ids) == 1
    }
    return symbol_to_gene_id, gene_id_to_symbol


def transcript_mapping_from_rows(
    header: list[str],
    rows: list[dict[str, str]],
    config: dict,
    mapping_method: str,
    gene_symbol_lookup: dict[str, str],
    gene_id_to_symbol: dict[str, str],
) -> dict[str, dict[str, str]]:
    transcript_config = config["alignment_ready_registry"]["transcriptomics"]
    symbol_column = candidate_column(header, transcript_config["direct_gene_symbol_columns"])
    entrez_column = candidate_column(header, transcript_config["direct_entrez_columns"])
    assignment_column = candidate_column(header, transcript_config["gene_assignment_columns"])
    probe_column = header[0]
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        probe_id = row.get(probe_column, "").strip()
        if is_missing(probe_id):
            continue
        row_joined_text = " ".join(str(value) for value in row.values())
        canonical_entity_id = ""
        canonical_entity_label = ""
        confidence = ""

        if entrez_column:
            canonical_entity_id = extract_unique_entrez(row.get(entrez_column, ""))
        if symbol_column:
            canonical_entity_label = extract_symbol(row.get(symbol_column, ""), "")
        if not canonical_entity_id and assignment_column:
            canonical_entity_id, assignment_symbol = extract_assignment_pair(row.get(assignment_column, ""))
            if not canonical_entity_id:
                canonical_entity_id = extract_unique_entrez(row.get(assignment_column, ""))
            if not canonical_entity_label:
                canonical_entity_label = assignment_symbol or extract_symbol("", row.get(assignment_column, ""))
        if not canonical_entity_id and not entrez_column and not assignment_column:
            assignment_like_values = [
                str(row.get(column, "")).strip()
                for column in header[1:]
                if " // " in str(row.get(column, "")) and not is_control_feature_text(str(row.get(column, "")))
            ]
            for candidate_value in assignment_like_values:
                canonical_entity_id, assignment_symbol = extract_assignment_pair(candidate_value)
                if canonical_entity_id:
                    canonical_entity_label = canonical_entity_label or assignment_symbol
                    break
            if not canonical_entity_id and not is_control_feature_text(row_joined_text):
                scanned_text = " /// ".join(str(row.get(column, "")).strip() for column in header[1:])
                canonical_entity_id = extract_unique_entrez(scanned_text)
                if not canonical_entity_label:
                    canonical_entity_label = extract_symbol("", scanned_text)
        if is_control_feature_text(row_joined_text) and not canonical_entity_id:
            continue
        if not canonical_entity_id and canonical_entity_label and canonical_entity_label in gene_symbol_lookup:
            canonical_entity_id = gene_symbol_lookup[canonical_entity_label]
        if not canonical_entity_id:
            continue
        if canonical_entity_label.isdigit():
            canonical_entity_label = ""
        canonical_entity_label = gene_id_to_symbol.get(canonical_entity_id, canonical_entity_label)
        if not canonical_entity_label:
            canonical_entity_label = canonical_entity_id
        if entrez_column and symbol_column and row.get(entrez_column, "").strip() and row.get(symbol_column, "").strip():
            confidence = "direct_gene_symbol_and_entrez"
        elif entrez_column and row.get(entrez_column, "").strip():
            confidence = "direct_entrez"
        else:
            confidence = "gene_assignment"
        previous_mapping = mapping.get(probe_id)
        if previous_mapping is not None and previous_mapping["canonical_entity_id"] != canonical_entity_id:
            continue
        mapping[probe_id] = {
            "canonical_entity_id": canonical_entity_id,
            "canonical_entity_label": canonical_entity_label,
            "mapping_method": mapping_method,
            "mapping_confidence": confidence,
            "aggregation_rule": "median_across_source_features",
        }
    return mapping


def transcript_probe_candidates(feature_id: str) -> list[str]:
    normalized_feature_id = feature_id.strip()
    if not normalized_feature_id:
        return []
    candidates = [normalized_feature_id]
    if ":" in normalized_feature_id:
        trailing_probe_id = normalized_feature_id.rsplit(":", 1)[1].strip()
        if trailing_probe_id and trailing_probe_id not in candidates:
            candidates.append(trailing_probe_id)
    return candidates


def transcript_mapping_for_feature_id(
    feature_id: str,
    mapping: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str]] | tuple[None, None]:
    for probe_candidate in transcript_probe_candidates(feature_id):
        if probe_candidate in mapping:
            return probe_candidate, mapping[probe_candidate]
    return None, None


def geo_platform_ids_for_study(source_accession: str, config: dict) -> list[str]:
    geo_root = resolve_repo_path(config["sources"]["geo"]["local_root"])
    family_soft_path = geo_root / "downloads" / "series" / source_accession / f"{source_accession}_family.soft.gz"
    if not family_soft_path.exists():
        raise FileNotFoundError(f"Missing GEO family soft for study {source_accession}: {family_soft_path}")
    platform_ids: list[str] = []
    with gzip.open(family_soft_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Series_platform_id = "):
                platform_id = line.split("=", 1)[1].strip()
                if platform_id and platform_id not in platform_ids:
                    platform_ids.append(platform_id)
    return platform_ids


def arrayexpress_designs_for_study(source_accession: str, config: dict) -> list[str]:
    source_root = resolve_repo_path(config["sources"]["arrayexpress"]["local_root"])
    sdrf_path = source_root / "downloads" / "studies" / source_accession / f"{source_accession}.sdrf.txt"
    if not sdrf_path.exists():
        raise FileNotFoundError(f"Missing ArrayExpress SDRF for study {source_accession}: {sdrf_path}")
    with sdrf_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return sorted({row["Array Design REF"].strip() for row in reader if row.get("Array Design REF", "").strip()})


def annotation_resource_candidates(bundle_row: dict[str, str], config: dict) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if bundle_row["source_key"] == "geo":
        geo_root = resolve_repo_path(config["sources"]["geo"]["local_root"])
        for platform_id in geo_platform_ids_for_study(bundle_row["source_accession"], config):
            path = geo_root / "platform_annotations" / platform_id / f"{platform_id}_family.soft.gz"
            if path.exists():
                candidates.append(
                    {
                        "resource_name": platform_id,
                        "resource_type": "geo_platform",
                        "path": path,
                        "priority": 2,
                    }
                )
        return candidates
    if bundle_row["source_key"] == "arrayexpress":
        array_root = resolve_repo_path(config["sources"]["arrayexpress"]["local_root"])
        alias_lookup = config["annotation_companion_downloads"]["arrayexpress"]["design_to_geo_platform_alias"]
        replacement_lookup = config["annotation_companion_downloads"]["arrayexpress"]["design_replacement_files"]
        for design_accession in arrayexpress_designs_for_study(bundle_row["source_accession"], config):
            design_root = array_root / "platform_annotations" / design_accession
            adf_path = design_root / f"{design_accession}.adf.txt"
            if adf_path.exists():
                candidates.append(
                    {
                        "resource_name": design_accession,
                        "resource_type": "array_design_adf",
                        "path": adf_path,
                        "priority": 1,
                    }
                )
            if design_accession in replacement_lookup:
                replacement_filename = str(replacement_lookup[design_accession]["replacement_filename"])
                replacement_path = design_root / replacement_filename
                if replacement_path.exists():
                    candidates.append(
                        {
                            "resource_name": str(replacement_lookup[design_accession]["replacement_accession"]),
                            "resource_type": "array_design_replacement_annotation",
                            "path": replacement_path,
                            "priority": 4,
                        }
                    )
            if design_accession in alias_lookup:
                alias_platform = str(alias_lookup[design_accession])
                miniml_path = design_root / f"{alias_platform}_family.xml.tgz"
                alias_path = design_root / f"{alias_platform}_family.soft.gz"
                if miniml_path.exists():
                    candidates.append(
                        {
                            "resource_name": alias_platform,
                            "resource_type": "array_design_geo_platform_alias_miniml",
                            "path": miniml_path,
                            "priority": 3,
                        }
                    )
                elif alias_path.exists():
                    candidates.append(
                        {
                            "resource_name": alias_platform,
                            "resource_type": "array_design_geo_platform_alias",
                            "path": alias_path,
                            "priority": 3,
                        }
                    )
        return candidates
    return candidates


def resource_mapping(
    resource: dict[str, object],
    config: dict,
    mapping_cache: dict[str, dict[str, dict[str, str]]],
    gene_symbol_lookup: dict[str, str],
    gene_id_to_symbol: dict[str, str],
) -> dict[str, dict[str, str]]:
    cache_key = f"{resource['resource_type']}::{resource['path']}"
    if cache_key in mapping_cache:
        return mapping_cache[cache_key]
    path = Path(str(resource["path"]))
    resource_type = str(resource["resource_type"])
    if resource_type == "array_design_replacement_annotation":
        header, rows = read_plain_table_rows(path)
    elif resource_type.endswith("miniml"):
        header, rows = read_gpl_miniml_rows(path)
    elif resource_type.endswith("gpl") or resource_type == "geo_platform":
        header, rows = read_gpl_rows(path)
    else:
        header, rows = read_adf_rows(path)
    mapping = transcript_mapping_from_rows(
        header,
        rows,
        config,
        str(resource["resource_type"]),
        gene_symbol_lookup,
        gene_id_to_symbol,
    )
    mapping_cache[cache_key] = mapping
    return mapping


def best_transcript_resource(
    bundle_row: dict[str, str],
    feature_index: pd.DataFrame,
    config: dict,
    mapping_cache: dict[str, dict[str, dict[str, str]]],
    gene_symbol_lookup: dict[str, str],
    gene_id_to_symbol: dict[str, str],
) -> tuple[dict[str, object], dict[str, dict[str, str]]] | tuple[None, None]:
    candidates = annotation_resource_candidates(bundle_row, config)
    if not candidates:
        return None, None
    feature_ids = feature_index["feature_id"].astype(str).tolist()
    best_resource = None
    best_mapping: dict[str, dict[str, str]] | None = None
    best_score: tuple[int, int] = (-1, -1)
    for candidate in candidates:
        try:
            current_mapping = resource_mapping(candidate, config, mapping_cache, gene_symbol_lookup, gene_id_to_symbol)
        except Exception:
            continue
        coverage = sum(
            1
            for feature_id in feature_ids
            if transcript_mapping_for_feature_id(feature_id, current_mapping)[1] is not None
        )
        score = (coverage, int(candidate["priority"]))
        if score > best_score:
            best_score = score
            best_resource = candidate
            best_mapping = current_mapping
    if best_resource is None or best_mapping is None or best_score[0] <= 0:
        return None, None
    return best_resource, best_mapping


def build_transcriptomics_alignment(
    bundle_row: dict[str, str],
    sample_index: pd.DataFrame,
    feature_index: pd.DataFrame,
    raw_matrix: np.ndarray,
    config: dict,
    mapping_cache: dict[str, dict[str, dict[str, str]]],
    gene_symbol_lookup: dict[str, str],
    gene_id_to_symbol: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, object]], list[dict[str, object]], str]:
    selected_resource, mapping = best_transcript_resource(
        bundle_row,
        feature_index,
        config,
        mapping_cache,
        gene_symbol_lookup,
        gene_id_to_symbol,
    )
    if selected_resource is None or mapping is None:
        raise ValueError(f"No transcript annotation resource matched features for {bundle_row['source_accession']}")
    ordered_features = feature_index.sort_values("row_index").reset_index(drop=True)
    positions_by_entity: dict[str, list[int]] = defaultdict(list)
    mapping_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    entity_metadata: dict[str, dict[str, object]] = {}

    for feature_position, feature_row in enumerate(ordered_features.itertuples(index=False)):
        probe_id = str(feature_row.feature_id)
        matched_probe_id, annotation = transcript_mapping_for_feature_id(probe_id, mapping)
        if matched_probe_id is None or annotation is None:
            exclusion_rows.append(
                {
                    "study_uid": bundle_row["study_uid"],
                    "source_key": bundle_row["source_key"],
                    "source_accession": bundle_row["source_accession"],
                    "exclusion_scope": "feature",
                    "original_feature_id": probe_id,
                    "exclusion_reason": "missing_platform_annotation_mapping",
                    "note": str(selected_resource["resource_name"]),
                }
            )
            continue
        entity_id = annotation["canonical_entity_id"]
        positions_by_entity[entity_id].append(feature_position)
        mapping_rows.append(
            {
                "study_uid": bundle_row["study_uid"],
                "source_accession": bundle_row["source_accession"],
                "original_feature_id": probe_id,
                "canonical_entity_id": entity_id,
                "mapping_method": annotation["mapping_method"],
                "mapping_confidence": annotation["mapping_confidence"],
                "aggregation_rule": annotation["aggregation_rule"],
                "note": matched_probe_id if matched_probe_id != probe_id else "",
            }
        )
        if entity_id not in entity_metadata:
            entity_metadata[entity_id] = {
                "canonical_entity_label": annotation["canonical_entity_label"],
                "mapping_method": annotation["mapping_method"],
            }

    if not positions_by_entity:
        raise ValueError(f"No transcript features mapped for {bundle_row['source_accession']}")

    ordered_entity_ids = sorted(positions_by_entity, key=lambda entity_id: min(positions_by_entity[entity_id]))
    aligned_raw = np.full((raw_matrix.shape[0], len(ordered_entity_ids)), np.nan, dtype=np.float64)
    entity_rows: list[dict[str, object]] = []
    for entity_index, entity_id in enumerate(ordered_entity_ids, start=1):
        source_positions = positions_by_entity[entity_id]
        aligned_raw[:, entity_index - 1] = np.nanmedian(raw_matrix[:, source_positions], axis=1)
        ordered_source_feature_ids = [str(ordered_features.iloc[position]["feature_id"]) for position in source_positions]
        entity_rows.append(
            {
                "row_index": entity_index,
                "canonical_entity_id": entity_id,
                "canonical_entity_label": entity_metadata[entity_id]["canonical_entity_label"],
                "entity_type": "gene",
                "namespace": "entrez_gene_id",
                "source_feature_ids": "|".join(ordered_source_feature_ids),
                "source_feature_count": len(ordered_source_feature_ids),
                "mapping_method": entity_metadata[entity_id]["mapping_method"],
                "aggregation_rule": "median_across_source_features",
            }
        )

    aligned_norm = normalize_matrix_by_modality(aligned_raw, "transcriptomics")
    return sample_index, pd.DataFrame(entity_rows), aligned_raw, aligned_norm, mapping_rows, exclusion_rows, relative_repo_path(Path(str(selected_resource["path"])))


def filter_all_nan_entities(
    bundle_row: dict[str, str],
    entity_rows: list[dict[str, object]],
    aligned_raw: np.ndarray,
    mapping_rows: list[dict[str, object]],
    exclusion_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    if aligned_raw.size == 0:
        return entity_rows, aligned_raw, mapping_rows, exclusion_rows
    keep_mask = ~np.isnan(aligned_raw).all(axis=0)
    if bool(np.all(keep_mask)):
        return entity_rows, aligned_raw, mapping_rows, exclusion_rows
    filtered_entity_rows: list[dict[str, object]] = []
    removed_entity_ids: set[str] = set()
    next_row_index = 1
    for entity_position, entity_row in enumerate(entity_rows):
        if keep_mask[entity_position]:
            updated_row = dict(entity_row)
            updated_row["row_index"] = next_row_index
            next_row_index += 1
            filtered_entity_rows.append(updated_row)
            continue
        removed_entity_ids.add(str(entity_row["canonical_entity_id"]))
        source_feature_ids = str(entity_row["source_feature_ids"]).split("|")
        for source_feature_id in source_feature_ids:
            exclusion_rows.append(
                {
                    "study_uid": bundle_row["study_uid"],
                    "source_key": bundle_row["source_key"],
                    "source_accession": bundle_row["source_accession"],
                    "exclusion_scope": "feature",
                    "original_feature_id": source_feature_id,
                    "exclusion_reason": "all_nan_after_aggregation",
                    "note": str(entity_row["canonical_entity_id"]),
                }
            )
    filtered_mapping_rows = [
        mapping_row
        for mapping_row in mapping_rows
        if str(mapping_row["canonical_entity_id"]) not in removed_entity_ids
    ]
    return filtered_entity_rows, aligned_raw[:, keep_mask], filtered_mapping_rows, exclusion_rows


def build_metabolomics_alignment(
    bundle_row: dict[str, str],
    sample_index: pd.DataFrame,
    feature_index: pd.DataFrame,
    raw_matrix: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, object]], list[dict[str, object]], str]:
    ordered_features = feature_index.sort_values("row_index").reset_index(drop=True)
    positions_by_entity: dict[str, list[int]] = defaultdict(list)
    mapping_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    entity_labels: dict[str, str] = {}

    for feature_position, feature_row in enumerate(ordered_features.itertuples(index=False)):
        original_feature_id = str(feature_row.feature_id)
        source_label = str(feature_row.feature_label).strip() or original_feature_id
        canonical_label = normalized_metabolite_label(source_label)
        if not canonical_label:
            exclusion_rows.append(
                {
                    "study_uid": bundle_row["study_uid"],
                    "source_key": bundle_row["source_key"],
                    "source_accession": bundle_row["source_accession"],
                    "exclusion_scope": "feature",
                    "original_feature_id": original_feature_id,
                    "exclusion_reason": "missing_metabolite_label",
                    "note": source_label,
                }
            )
            continue
        positions_by_entity[canonical_label].append(feature_position)
        entity_labels.setdefault(canonical_label, source_label)
        mapping_rows.append(
            {
                "study_uid": bundle_row["study_uid"],
                "source_accession": bundle_row["source_accession"],
                "original_feature_id": original_feature_id,
                "canonical_entity_id": canonical_label,
                "mapping_method": "normalized_metabolite_label",
                "mapping_confidence": "direct_feature_label",
                "aggregation_rule": "median_across_source_features",
            }
        )

    if not positions_by_entity:
        raise ValueError(f"No metabolite features mapped for {bundle_row['source_accession']}")

    ordered_entity_ids = sorted(positions_by_entity, key=lambda entity_id: min(positions_by_entity[entity_id]))
    aligned_raw = np.full((raw_matrix.shape[0], len(ordered_entity_ids)), np.nan, dtype=np.float64)
    entity_rows: list[dict[str, object]] = []
    for entity_index, entity_id in enumerate(ordered_entity_ids, start=1):
        source_positions = positions_by_entity[entity_id]
        aligned_raw[:, entity_index - 1] = np.nanmedian(raw_matrix[:, source_positions], axis=1)
        ordered_source_feature_ids = [str(ordered_features.iloc[position]["feature_id"]) for position in source_positions]
        entity_rows.append(
            {
                "row_index": entity_index,
                "canonical_entity_id": entity_id,
                "canonical_entity_label": entity_labels[entity_id],
                "entity_type": "metabolite",
                "namespace": "normalized_metabolite_label",
                "source_feature_ids": "|".join(ordered_source_feature_ids),
                "source_feature_count": len(ordered_source_feature_ids),
                "mapping_method": "normalized_metabolite_label",
                "aggregation_rule": "median_across_source_features",
            }
        )

    aligned_norm = normalize_matrix_by_modality(aligned_raw, "metabolomics")
    return sample_index, pd.DataFrame(entity_rows), aligned_raw, aligned_norm, mapping_rows, exclusion_rows, "feature_label_normalization"


def bundle_paths(bundle_row: dict[str, str], config: dict) -> dict[str, Path]:
    bundle_root = resolve_repo_path(
        config["alignment_ready_registry"]["bundle_root_template"].format(
            source_key=bundle_row["source_key"],
            source_accession=bundle_row["source_accession"],
        )
    )
    filenames = config["alignment_ready_registry"]["bundle_filenames"]
    return {
        "bundle_root": bundle_root,
        "manifest_path": bundle_root / filenames["manifest"],
        "sample_index_path": bundle_root / filenames["sample_index"],
        "entity_index_path": bundle_root / filenames["entity_index"],
        "matrix_raw_path": bundle_root / filenames["matrix_raw"],
        "matrix_norm_path": bundle_root / filenames["matrix_norm"],
    }


def prune_bundle_root(bundle_path_map: dict[str, Path]) -> None:
    bundle_root = bundle_path_map["bundle_root"]
    ensure_directory(bundle_root)
    expected_names = {path.name for key, path in bundle_path_map.items() if key != "bundle_root"}
    for path in bundle_root.iterdir():
        if path.name in expected_names:
            continue
        if path.name.endswith(".part") or path.is_file():
            path.unlink()


def manifest_payload(
    bundle_row: dict[str, str],
    modality: str,
    tissue_or_biofluid: str,
    annotation_resource: str,
    sample_count: int,
    original_feature_count: int,
    mapped_feature_count: int,
    aligned_entity_count: int,
    bundle_path_map: dict[str, Path],
) -> dict[str, object]:
    return {
        "study_uid": bundle_row["study_uid"],
        "source_key": bundle_row["source_key"],
        "source_accession": bundle_row["source_accession"],
        "modality": modality,
        "tissue_or_biofluid": tissue_or_biofluid,
        "annotation_resource": annotation_resource,
        "sample_count": sample_count,
        "original_feature_count": original_feature_count,
        "mapped_feature_count": mapped_feature_count,
        "aligned_entity_count": aligned_entity_count,
        "bundle_root": relative_repo_path(bundle_path_map["bundle_root"]),
        "manifest_path": relative_repo_path(bundle_path_map["manifest_path"]),
        "sample_index_path": relative_repo_path(bundle_path_map["sample_index_path"]),
        "entity_index_path": relative_repo_path(bundle_path_map["entity_index_path"]),
        "matrix_raw_path": relative_repo_path(bundle_path_map["matrix_raw_path"]),
        "matrix_norm_path": relative_repo_path(bundle_path_map["matrix_norm_path"]),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    source_accessions = selected_accessions(args.source_accessions)
    bundle_rows = filtered_rows(load_model_input_catalog(config), source_accessions)
    pipeline_logger.progress(
        "loaded_alignment_inputs",
        bundle_row_count=len(bundle_rows),
    )
    alignment_root = resolve_repo_path(config["alignment_ready_registry"]["local_root"])
    ensure_directory(alignment_root)

    aligned_matrix_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    source_accessions_by_entity: dict[tuple[str, str], set[str]] = defaultdict(set)
    entity_label_lookup: dict[tuple[str, str], str] = {}
    mapping_cache: dict[str, dict[str, dict[str, str]]] = {}
    gene_symbol_lookup, gene_id_to_symbol = gene_reference_maps(config)

    for bundle_row in bundle_rows:
        sample_index, feature_index, raw_matrix, _ = load_bundle_artifacts(bundle_row)
        modality = modality_from_sample_index(sample_index)
        tissue_or_biofluid = tissue_from_sample_index(sample_index)
        try:
            if modality == "transcriptomics":
                (
                    aligned_sample_index,
                    entity_index,
                    aligned_raw,
                    aligned_norm,
                    current_mapping_rows,
                    current_exclusion_rows,
                    annotation_resource,
                ) = build_transcriptomics_alignment(
                    bundle_row,
                    sample_index,
                    feature_index,
                    raw_matrix,
                    config,
                    mapping_cache,
                    gene_symbol_lookup,
                    gene_id_to_symbol,
                )
                entity_type = "gene"
                namespace = "entrez_gene_id"
            elif modality == "metabolomics":
                (
                    aligned_sample_index,
                    entity_index,
                    aligned_raw,
                    aligned_norm,
                    current_mapping_rows,
                    current_exclusion_rows,
                    annotation_resource,
                ) = build_metabolomics_alignment(bundle_row, sample_index, feature_index, raw_matrix)
                entity_type = "metabolite"
                namespace = config["alignment_ready_registry"]["metabolomics"]["canonical_namespace"]
            else:
                raise ValueError(f"Unsupported modality for alignment: {modality}")
        except Exception as exc:
            failed_bundle_root = bundle_paths(bundle_row, config)["bundle_root"]
            if failed_bundle_root.exists():
                shutil.rmtree(failed_bundle_root)
            exclusion_rows.append(
                {
                    "study_uid": bundle_row["study_uid"],
                    "source_key": bundle_row["source_key"],
                    "source_accession": bundle_row["source_accession"],
                    "exclusion_scope": "study",
                    "original_feature_id": "",
                    "exclusion_reason": "alignment_failed",
                    "note": str(exc),
                }
            )
            continue

        entity_rows, filtered_raw, current_mapping_rows, current_exclusion_rows = filter_all_nan_entities(
            bundle_row,
            entity_index.to_dict("records"),
            aligned_raw,
            current_mapping_rows,
            current_exclusion_rows,
        )
        entity_index = pd.DataFrame(entity_rows, columns=config["schemas"]["alignment_entity_index_fields"])
        aligned_raw = filtered_raw
        aligned_norm = normalize_matrix_by_modality(aligned_raw, modality)

        bundle_path_map = bundle_paths(bundle_row, config)
        prune_bundle_root(bundle_path_map)
        write_parquet_atomic(aligned_sample_index, bundle_path_map["sample_index_path"])
        write_parquet_atomic(entity_index, bundle_path_map["entity_index_path"])
        write_npy_atomic(aligned_raw, bundle_path_map["matrix_raw_path"])
        write_npy_atomic(aligned_norm, bundle_path_map["matrix_norm_path"])
        write_json(
            bundle_path_map["manifest_path"],
            manifest_payload(
                bundle_row,
                modality,
                tissue_or_biofluid,
                annotation_resource,
                aligned_sample_index.shape[0],
                feature_index.shape[0],
                len(current_mapping_rows),
                entity_index.shape[0],
                bundle_path_map,
            ),
        )

        aligned_matrix_rows.append(
            {
                "study_uid": bundle_row["study_uid"],
                "source_key": bundle_row["source_key"],
                "source_accession": bundle_row["source_accession"],
                "modality": modality,
                "tissue_or_biofluid": tissue_or_biofluid,
                "aligned_entity_namespace": namespace,
                "aligned_entity_type": entity_type,
                "original_feature_count": feature_index.shape[0],
                "mapped_feature_count": len(current_mapping_rows),
                "aligned_entity_count": entity_index.shape[0],
                "sample_count": aligned_sample_index.shape[0],
                "alignment_status": "aligned",
                "alignment_note": annotation_resource,
                "bundle_root": relative_repo_path(bundle_path_map["bundle_root"]),
                "manifest_path": relative_repo_path(bundle_path_map["manifest_path"]),
                "sample_index_path": relative_repo_path(bundle_path_map["sample_index_path"]),
                "entity_index_path": relative_repo_path(bundle_path_map["entity_index_path"]),
                "matrix_raw_path": relative_repo_path(bundle_path_map["matrix_raw_path"]),
                "matrix_norm_path": relative_repo_path(bundle_path_map["matrix_norm_path"]),
            }
        )
        mapping_rows.extend(current_mapping_rows)
        exclusion_rows.extend(current_exclusion_rows)
        for entity_row in entity_index.itertuples(index=False):
            entity_key = (str(entity_row.namespace), str(entity_row.canonical_entity_id))
            source_accessions_by_entity[entity_key].add(bundle_row["source_accession"])
            entity_label_lookup.setdefault(entity_key, str(entity_row.canonical_entity_label))

    entity_catalog_rows: list[dict[str, object]] = []
    for namespace, canonical_entity_id in sorted(source_accessions_by_entity):
        entity_catalog_rows.append(
            {
                "canonical_entity_id": canonical_entity_id,
                "canonical_entity_label": entity_label_lookup[(namespace, canonical_entity_id)],
                "entity_type": "gene" if namespace == "entrez_gene_id" else "metabolite",
                "namespace": namespace,
                "source_count": len(source_accessions_by_entity[(namespace, canonical_entity_id)]),
            }
        )

    mapping_rows.sort(key=lambda row: (row["source_accession"], row["canonical_entity_id"], row["original_feature_id"]))
    aligned_matrix_rows.sort(key=lambda row: (row["source_key"], row["source_accession"]))
    exclusion_rows.sort(key=lambda row: (row["source_key"], row["source_accession"], row["exclusion_scope"], row["original_feature_id"]))

    relative_paths = config["alignment_ready_registry"]["relative_paths"]
    write_tsv(
        alignment_root / relative_paths["entity_catalog"],
        config["schemas"]["alignment_entity_fields"],
        entity_catalog_rows,
    )
    write_tsv(
        alignment_root / relative_paths["study_feature_mapping"],
        config["schemas"]["alignment_feature_mapping_fields"],
        mapping_rows,
    )
    write_tsv(
        alignment_root / relative_paths["aligned_matrix_catalog"],
        config["schemas"]["alignment_matrix_fields"],
        aligned_matrix_rows,
    )
    write_tsv(
        alignment_root / relative_paths["alignment_exclusion_catalog"],
        config["schemas"]["alignment_exclusion_fields"],
        exclusion_rows,
    )
    pipeline_logger.artifact(
        "alignment_ready_registry_written",
        artifact=alignment_root,
        entity_row_count=len(entity_catalog_rows),
        mapping_row_count=len(mapping_rows),
        aligned_matrix_row_count=len(aligned_matrix_rows),
        exclusion_row_count=len(exclusion_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=alignment_root,
        entity_row_count=len(entity_catalog_rows),
        mapping_row_count=len(mapping_rows),
        aligned_matrix_row_count=len(aligned_matrix_rows),
        exclusion_row_count=len(exclusion_rows),
    )

if __name__ == "__main__":
    main()
