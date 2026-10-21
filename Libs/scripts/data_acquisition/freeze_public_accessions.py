from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import (
    chunked,
    choose_diagnosis_label,
    contains_any,
    http_get,
    infer_sample_context,
    infer_tissue_or_biofluid,
    load_config,
    normalize_text,
    read_tsv,
    refresh_checksums,
    resolve_repo_path,
    slugify,
    write_json,
    write_tsv,
)


def load_cached_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def request_json(url: str, destination: Path, params: dict | None = None) -> dict:
    if destination.exists():
        return load_cached_json(destination)
    response = http_get(url, params=params)
    payload = response.json()
    write_json(destination, payload)
    return payload


def infer_modality_from_geo(gds_type: str) -> str:
    normalized = normalize_text(gds_type)
    if "sequencing" in normalized:
        return "transcriptomics"
    if "array" in normalized:
        return "transcriptomics"
    if "protein" in normalized:
        return "proteomics"
    if "metabol" in normalized:
        return "metabolomics"
    return "metadata"


def infer_modality_from_text(text: str, fallback: str) -> str:
    normalized = normalize_text(text)
    if "lc ms" in normalized or "gc ms" in normalized or "metabol" in normalized:
        return "metabolomics"
    if "rna seq" in normalized or "transcript" in normalized or "expression" in normalized:
        return "transcriptomics"
    if "proteom" in normalized or "mass spectrometry" in normalized:
        return "proteomics"
    return fallback


def classify_row(text: str, species: str, diagnosis_label: str, config: dict) -> tuple[str, str]:
    normalized_text = normalize_text(text)
    excluded_keywords = config["classification"]["excluded_disease_keywords"]
    if contains_any(normalized_text, excluded_keywords):
        return "no", "non_ad_aging_disease"
    if not diagnosis_label:
        return "no", "query_term_not_matched"
    sample_context = infer_sample_context(
        normalized_text,
        config["classification"]["mechanistic_keywords"],
        config["classification"]["patient_keywords"],
        species,
    )
    if sample_context == "mechanistic_model":
        return "yes", ""
    return "yes", ""


def annotate_accession_row(source_key: str, row: dict[str, str]) -> dict[str, str]:
    annotated_row = dict(row)
    defaults_by_source = {
        "geo": {
            "source_repository": "GEO",
            "release_type": "processed",
            "data_granularity": "sample_level",
            "provenance_level": "publication_curated",
            "world_model_role": "disease_observation",
            "cohort_or_atlas_family": "ad_aging_public",
        },
        "arrayexpress": {
            "source_repository": "ArrayExpress",
            "release_type": "processed",
            "data_granularity": "sample_level",
            "provenance_level": "publication_curated",
            "world_model_role": "disease_observation",
            "cohort_or_atlas_family": "ad_aging_public",
        },
        "metabolomics_workbench": {
            "source_repository": "MetabolomicsWorkbench",
            "release_type": "processed",
            "data_granularity": "sample_level",
            "provenance_level": "publication_curated",
            "world_model_role": "disease_observation",
            "cohort_or_atlas_family": "ad_aging_public",
        },
    }
    annotated_row.update(defaults_by_source[source_key])
    annotated_row["subject_linkable_flag"] = "yes" if annotated_row["sample_context"] == "patient_cohort" else "no"
    return annotated_row


def build_geo_rows(config: dict) -> list[dict[str, str]]:
    query_config = config["query_repositories"]["geo"]
    source_config = config["sources"]["geo"]
    download_root = resolve_repo_path(source_config["local_root"]) / "downloads"
    response_root = download_root / "query_responses"
    response_root.mkdir(parents=True, exist_ok=True)

    unique_results: dict[str, dict[str, str]] = {}
    for query_term in query_config["query_terms"]:
        term_slug = slugify(query_term)
        esearch_path = response_root / f"{term_slug}_esearch.json"
        search_payload = request_json(
            query_config["search_url"],
            esearch_path,
            {
                "db": "gds",
                "retmode": "json",
                "retmax": query_config["max_results_per_query"],
                "term": f"\"{query_term}\"[All Fields] AND {query_config['organism_filter']}[Organism]",
            },
        )
        identifier_list = search_payload["esearchresult"]["idlist"]
        if not identifier_list:
            continue
        esummary_path = response_root / f"{term_slug}_esummary.json"
        if esummary_path.exists():
            summary_payload = load_cached_json(esummary_path)
        else:
            summary_payload = {"result": {"uids": []}}
            for identifier_chunk in chunked(identifier_list, 100):
                chunk_payload = request_json(
                    query_config["summary_url"],
                    response_root / f"{term_slug}_esummary_{identifier_chunk[0]}.json",
                    {
                        "db": "gds",
                        "retmode": "json",
                        "id": ",".join(identifier_chunk),
                    },
                )
                summary_payload["result"]["uids"].extend(chunk_payload["result"]["uids"])
                for uid in chunk_payload["result"]["uids"]:
                    summary_payload["result"][uid] = chunk_payload["result"][uid]
            write_json(esummary_path, summary_payload)
        for uid in summary_payload["result"]["uids"]:
            item = summary_payload["result"][uid]
            accession = item["accession"]
            combined_text = " ".join(
                [
                    item.get("title") or "",
                    item.get("summary") or "",
                    item.get("gdstype") or "",
                ]
            )
            diagnosis_label = choose_diagnosis_label(combined_text, query_config["query_terms"])
            include_flag, exclusion_reason = classify_row(combined_text, item.get("taxon", ""), diagnosis_label, config)
            row = {
                "source_accession": accession,
                "title": item.get("title") or "",
                "species": item.get("taxon") or "",
                "tissue_or_biofluid": infer_tissue_or_biofluid(
                    normalize_text(combined_text),
                    config["classification"]["tissue_keyword_map"],
                ),
                "modality": infer_modality_from_geo(item.get("gdstype", "")),
                "sample_context": infer_sample_context(
                    normalize_text(combined_text),
                    config["classification"]["mechanistic_keywords"],
                    config["classification"]["patient_keywords"],
                    item.get("taxon", ""),
                ),
                "diagnosis_label": diagnosis_label,
                "case_count": "",
                "control_count": "",
                "public_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
                "include_flag": include_flag,
                "exclusion_reason": exclusion_reason,
            }
            annotated_row = annotate_accession_row("geo", row)
            if accession not in unique_results:
                unique_results[accession] = annotated_row
                continue
            if unique_results[accession]["include_flag"] == "no" and annotated_row["include_flag"] == "yes":
                unique_results[accession] = annotated_row
    return sorted(unique_results.values(), key=lambda row: row["source_accession"])


def parse_arrayexpress_species(detail_payload: dict, search_hit: dict) -> str:
    content = " ".join(
        [
            search_hit.get("content") or "",
            json.dumps(detail_payload, ensure_ascii=True),
        ]
    )
    if "Homo sapiens" in content:
        return "Homo sapiens"
    if "Mus musculus" in content:
        return "Mus musculus"
    return ""


def build_arrayexpress_rows(config: dict, geo_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    query_config = config["query_repositories"]["arrayexpress"]
    source_config = config["sources"]["arrayexpress"]
    download_root = resolve_repo_path(source_config["local_root"]) / "downloads"
    response_root = download_root / "query_responses"
    detail_root = download_root / "study_details"
    response_root.mkdir(parents=True, exist_ok=True)
    detail_root.mkdir(parents=True, exist_ok=True)

    geo_accessions = {row["source_accession"] for row in geo_rows}
    accession_index: dict[str, dict[str, object]] = {}
    for query_term in query_config["query_terms"]:
        term_slug = slugify(query_term)
        for page_number in range(1, query_config["max_pages"] + 1):
            query_path = response_root / f"{term_slug}_page_{page_number}.json"
            search_payload = request_json(
                query_config["search_url"],
                query_path,
                {
                    "query": f'collection:ArrayExpress AND "{query_term}"',
                    "pageSize": query_config["page_size"],
                    "page": page_number,
                },
            )
            hits = search_payload.get("hits", [])
            if not hits:
                break
            for hit in hits:
                accession = hit["accession"]
                if accession not in accession_index:
                    accession_index[accession] = {
                        "hit": hit,
                        "matched_terms": {query_term},
                    }
                    continue
                accession_index[accession]["matched_terms"].add(query_term)

    unique_results: dict[str, dict[str, str]] = {}
    for accession, accession_payload in accession_index.items():
        hit = accession_payload["hit"]
        matched_terms = sorted(accession_payload["matched_terms"])
        combined_text = " ".join(
            [
                hit.get("title") or "",
                hit.get("content") or "",
            ]
        )
        diagnosis_label = choose_diagnosis_label(combined_text, query_config["query_terms"])
        if not diagnosis_label and matched_terms:
            diagnosis_label = matched_terms[0]
        if accession.startswith("E-GEOD-"):
            geo_candidate = f"GSE{accession.removeprefix('E-GEOD-')}"
            if geo_candidate in geo_accessions:
                species = parse_arrayexpress_species({}, hit)
                unique_results[accession] = annotate_accession_row(
                    "arrayexpress",
                    {
                    "source_accession": accession,
                    "title": hit.get("title") or "",
                    "species": species,
                    "tissue_or_biofluid": infer_tissue_or_biofluid(
                        normalize_text(combined_text),
                        config["classification"]["tissue_keyword_map"],
                    ),
                    "modality": infer_modality_from_text(combined_text, "metadata"),
                    "sample_context": infer_sample_context(
                        normalize_text(combined_text),
                        config["classification"]["mechanistic_keywords"],
                        config["classification"]["patient_keywords"],
                        species,
                    ),
                    "diagnosis_label": diagnosis_label,
                    "case_count": "",
                    "control_count": "",
                    "public_url": f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}",
                    "include_flag": "no",
                    "exclusion_reason": f"duplicate_mirror_of_{geo_candidate}",
                    },
                )
                continue
        detail_path = detail_root / f"{accession}.json"
        if detail_path.exists():
            detail_payload = load_cached_json(detail_path)
        else:
            detail_payload = request_json(
                query_config["detail_url_template"].format(accession=accession),
                detail_path,
            )
        detail_text = json.dumps(detail_payload, ensure_ascii=True)
        combined_text = " ".join(
            [
                combined_text,
                detail_text,
            ]
        )
        diagnosis_label = choose_diagnosis_label(combined_text, query_config["query_terms"]) or diagnosis_label
        species = parse_arrayexpress_species(detail_payload, hit)
        include_flag, exclusion_reason = classify_row(combined_text, species, diagnosis_label, config)
        row = {
            "source_accession": accession,
            "title": hit.get("title") or "",
            "species": species,
            "tissue_or_biofluid": infer_tissue_or_biofluid(
                normalize_text(combined_text),
                config["classification"]["tissue_keyword_map"],
            ),
            "modality": infer_modality_from_text(combined_text, "metadata"),
            "sample_context": infer_sample_context(
                normalize_text(combined_text),
                config["classification"]["mechanistic_keywords"],
                config["classification"]["patient_keywords"],
                species,
            ),
            "diagnosis_label": diagnosis_label,
            "case_count": "",
            "control_count": "",
            "public_url": f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}",
            "include_flag": include_flag,
            "exclusion_reason": exclusion_reason,
        }
        unique_results[accession] = annotate_accession_row("arrayexpress", row)
    return sorted(unique_results.values(), key=lambda row: row["source_accession"])


def build_metabolomics_rows(config: dict) -> list[dict[str, str]]:
    query_config = config["query_repositories"]["metabolomics_workbench"]
    source_config = config["sources"]["metabolomics_workbench"]
    download_root = resolve_repo_path(source_config["local_root"]) / "downloads"
    response_root = download_root / "query_responses"
    summary_root = download_root / "study_summaries"
    response_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    disease_payload = request_json(
        query_config["disease_index_url"],
        response_root / "disease_index.json",
    )

    species_payload = request_json(
        query_config["species_index_url"],
        response_root / "species_index.json",
    )

    species_lookup = {
        item["Study ID"]: item.get("Latin name") or item.get("Common name", "")
        for item in species_payload.values()
    }
    rows: list[dict[str, str]] = []
    normalized_terms = [normalize_text(term) for term in query_config["query_terms"]]
    for disease_item in disease_payload.values():
        disease_name = disease_item["Disease"]
        normalized_disease = normalize_text(disease_name)
        if not any(term in normalized_disease for term in normalized_terms):
            continue
        study_id = disease_item["Study ID"]
        summary_payload = request_json(
            query_config["summary_url_template"].format(study_id=study_id),
            summary_root / f"{study_id}.json",
        )
        combined_text = " ".join(
            [
                summary_payload.get("study_title") or "",
                summary_payload.get("analysis_type") or "",
                disease_name,
            ]
        )
        diagnosis_label = choose_diagnosis_label(combined_text, query_config["query_terms"])
        species = species_lookup.get(study_id, summary_payload.get("species", ""))
        include_flag, exclusion_reason = classify_row(combined_text, species, diagnosis_label, config)
        rows.append(
            annotate_accession_row(
                "metabolomics_workbench",
                {
                "source_accession": study_id,
                "title": summary_payload.get("study_title") or "",
                "species": species,
                "tissue_or_biofluid": infer_tissue_or_biofluid(
                    normalize_text(combined_text),
                    config["classification"]["tissue_keyword_map"],
                ),
                "modality": infer_modality_from_text(combined_text, "metabolomics"),
                "sample_context": infer_sample_context(
                    normalize_text(combined_text),
                    config["classification"]["mechanistic_keywords"],
                    config["classification"]["patient_keywords"],
                    species,
                ),
                "diagnosis_label": diagnosis_label,
                "case_count": "",
                "control_count": "",
                "public_url": summary_payload.get("study_url", ""),
                "include_flag": include_flag,
                "exclusion_reason": exclusion_reason,
                },
            )
        )
    return sorted(rows, key=lambda row: row["source_accession"])


def build_biccn_public_rows(config: dict) -> list[dict[str, str]]:
    source_config = config["sources"]["biccn_public"]
    rows: list[dict[str, str]] = []
    for row in source_config.get("catalog_rows", []):
        normalized_row = {key: str(value) for key, value in dict(row).items()}
        rows.append(normalized_row)
    return sorted(rows, key=lambda row: row["source_accession"])


def build_catalog_rows(source_key: str, config: dict) -> list[dict[str, str]]:
    source_config = config["sources"][source_key]
    rows: list[dict[str, str]] = []
    for row in source_config.get("catalog_rows", []):
        rows.append({key: str(value) for key, value in dict(row).items()})
    return sorted(rows, key=lambda row: row["source_accession"])


def write_repository_rows(source_key: str, source_rows: list[dict[str, str]], config: dict) -> None:
    source_config = config["sources"][source_key]
    freeze_path = resolve_repo_path(source_config["local_root"]) / "accession_freeze" / "accessions.tsv"
    write_tsv(freeze_path, config["schemas"]["accession_fields"], source_rows)
    refresh_checksums(resolve_repo_path(source_config["local_root"]), source_config["checksum_file"])


def load_existing_rows(source_key: str, config: dict) -> list[dict[str, str]]:
    source_config = config["sources"][source_key]
    freeze_path = resolve_repo_path(source_config["local_root"]) / "accession_freeze" / "accessions.tsv"
    if not freeze_path.exists():
        raise FileNotFoundError(f"Existing accession freeze is missing for {source_key}: {freeze_path}")
    return read_tsv(freeze_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze accession lists for public search repositories.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the public raw data registry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    try:
        geo_rows = build_geo_rows(config)
    except requests.RequestException:
        geo_rows = load_existing_rows("geo", config)
        pipeline_logger.progress("source_refresh_fallback", source_key="geo", reason="blocked_network_environment")
    write_repository_rows("geo", geo_rows, config)
    try:
        arrayexpress_rows = build_arrayexpress_rows(config, geo_rows)
    except requests.RequestException:
        arrayexpress_rows = load_existing_rows("arrayexpress", config)
        pipeline_logger.progress("source_refresh_fallback", source_key="arrayexpress", reason="blocked_network_environment")
    write_repository_rows("arrayexpress", arrayexpress_rows, config)
    try:
        metabolomics_rows = build_metabolomics_rows(config)
    except requests.RequestException:
        metabolomics_rows = load_existing_rows("metabolomics_workbench", config)
        pipeline_logger.progress("source_refresh_fallback", source_key="metabolomics_workbench", reason="blocked_network_environment")
    write_repository_rows("metabolomics_workbench", metabolomics_rows, config)
    biccn_public_rows = build_biccn_public_rows(config)
    write_repository_rows("biccn_public", biccn_public_rows, config)

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
