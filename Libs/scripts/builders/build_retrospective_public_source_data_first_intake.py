"""Preregister and build a bounded source-data-first public CNS evidence queue."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable

from jsonschema import Draft202012Validator
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_public_cns_ranking_intake import (  # noqa: E402
    _binding,
    _canonical_digest,
    _sha256,
)
from Libs.scripts.builders.build_retrospective_public_design_enriched_ranking_intake import (  # noqa: E402
    CNS_MODEL_TERMS,
    _context_family,
    _design_class,
    _named_intervention_entities,
    _objective_family,
)
from Libs.scripts.data_acquisition.download_first_public_batch import (  # noqa: E402
    iter_arrayexpress_file_entries,
)
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
PREREGISTRATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/retrospective_public_source_data_first_preregistration.schema.json"
)
INTAKE_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/retrospective_public_source_data_first_intake.schema.json"
)
RESOURCE_SNAPSHOT_PATH = REPO_ROOT / "Output/audits/source_data_first_resource_snapshot.json"
FROZEN_QUEUE_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_public_ranking_source_queue.json"
REFRESH_QUEUE_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_public_ranking_source_refresh_queue.json"
PRIOR_INTAKE_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake.json"
PRIOR_DESIGN_INTAKE_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_design_enriched_ranking_intake.json"
)
PRIOR_DESIGN_RECOVERY_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_design_enriched_ranking_source_recovery.json"
)
MATERIALIZATION_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_public_evidence_materialization.json"
REPLAY_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_replay_benchmark.json"
ACCESS_BLOCKED_UNIVERSE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_metadata_universe_access_blocked.tsv"
)
ACCESS_BLOCKED_QUEUE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_ranking_queue_access_blocked.tsv"
)
ACCESS_BLOCKED_SNAPSHOT_PATH = (
    REPO_ROOT / "Input/raw/public/retrospective/source_data_first/metadata_snapshot_access_blocked.json"
)
ACCESS_BLOCKED_AUDIT_PAYLOAD_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_source_data_first_intake_access_blocked_payload.json"
)
ACCESS_BLOCKED_INVALIDATION_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_source_data_first_intake_access_blocked.json"
)
SCHEMA_FAILURE_UNIVERSE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_metadata_universe_schema_failure.tsv"
)
SCHEMA_FAILURE_QUEUE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_ranking_queue_schema_failure.tsv"
)
SCHEMA_FAILURE_SNAPSHOT_PATH = (
    REPO_ROOT / "Input/raw/public/retrospective/source_data_first/metadata_snapshot_schema_failure.json"
)
SCHEMA_FAILURE_AUDIT_PAYLOAD_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_source_data_first_intake_schema_failure_payload.json"
)
SCHEMA_FAILURE_INVALIDATION_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_source_data_first_intake_schema_failure.json"
)
PROXY_FAILURE_UNIVERSE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_metadata_universe_proxy_failure.tsv"
)
PROXY_FAILURE_QUEUE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_ranking_queue_proxy_failure.tsv"
)
PROXY_FAILURE_SNAPSHOT_PATH = (
    REPO_ROOT / "Input/raw/public/retrospective/source_data_first/metadata_snapshot_proxy_failure.json"
)
PROXY_FAILURE_AUDIT_PAYLOAD_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_source_data_first_intake_proxy_failure_payload.json"
)
PROXY_FAILURE_INVALIDATION_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_source_data_first_intake_proxy_failure.json"
)

EXCLUSION_TABLES = (
    REPO_ROOT / "Input/manifests/retrospective_public/evidence_materialization.tsv",
    REPO_ROOT / "Input/manifests/retrospective_public/evidence_registry.tsv",
    REPO_ROOT / "Output/audits/retrospective_public_evidence_rejection_catalog.tsv",
    REPO_ROOT / "Input/manifests/retrospective_public/ranking_informative_source_queue.tsv",
    REPO_ROOT / "Input/manifests/retrospective_public/ranking_source_refresh_queue.tsv",
    REPO_ROOT / "Input/manifests/retrospective_public/cns_ranking_intake_queue.tsv",
    REPO_ROOT / "Input/manifests/retrospective_public/design_enriched_ranking_intake_queue.tsv",
)

PRIOR_METADATA_SNAPSHOTS = (
    REPO_ROOT / "Input/raw/public/retrospective/cns_ranking_intake/pubmed_metadata_snapshot.json",
    REPO_ROOT / "Input/raw/public/retrospective/design_enriched_ranking_intake/pubmed_metadata_snapshot.json",
)

UNIVERSE_FIELDS = (
    "asset_uid",
    "repository_family",
    "repository_record_uid",
    "asset_name",
    "asset_type",
    "asset_size_bytes",
    "asset_url",
    "local_path",
    "study_title",
    "publication_uid",
    "doi",
    "release_date",
    "public_access_status",
    "metadata_source",
    "metadata_text_sha256",
    "context_family",
    "objective_uid",
    "study_design_class",
    "named_intervention_entities",
    "named_intervention_entity_count",
    "structured_intervention_signal",
    "shared_control_signal",
    "replication_or_uncertainty_signal",
    "quantitative_asset_signal",
    "eligibility_status",
    "rejection_reason",
    "source_asset_content_retrieved",
    "protected_holdout_access",
)

QUEUE_FIELDS = (
    "intake_uid",
    "queue_sequence",
    "preregistration_digest",
    "parent_state_digest",
    "metadata_universe_digest",
    "asset_uid",
    "repository_family",
    "repository_record_uid",
    "asset_name",
    "asset_type",
    "asset_size_bytes",
    "asset_url",
    "local_path",
    "study_title",
    "publication_uid",
    "doi",
    "release_date",
    "context_family",
    "objective_uid",
    "study_design_class",
    "named_intervention_entities",
    "named_intervention_entity_count",
    "structured_intervention_signal",
    "shared_control_signal",
    "replication_or_uncertainty_signal",
    "ranking_priority",
    "queue_status",
    "source_asset_content_retrieved_before_priority",
    "numeric_outcomes_retrieved_before_priority",
    "effect_direction_used_for_priority",
    "p_values_used_for_priority",
    "winner_statements_used_for_priority",
    "protected_holdout_access",
)

MACHINE_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx", ".json", ".txt", ".zip"}
QUANTITATIVE_NAME_TERMS = (
    "source data",
    "source_data",
    "raw data",
    "raw_data",
    "replicate",
    "measurement",
    "quantitative",
    "spreadsheet",
    "table",
    "results",
    "data",
)
CONTROL_TERMS = ("control", "vehicle", "dmso", "untreated", "sham", "baseline")
REPLICATION_TERMS = (
    "replicate",
    "individual data",
    "source data",
    "source_data",
    "raw data",
    "raw_data",
    "mean",
    "sem",
    "standard error",
    "standard deviation",
    "statistics",
)
STRUCTURED_INTERVENTION_TERMS = (
    "dose",
    "concentration",
    "combination",
    "monotherapy",
    "versus",
    " vs ",
    "screen",
    "compound library",
    "drug library",
)
OUTCOME_BEARING_TERMS = (
    "significant",
    "winner",
    "strongest",
    "improv",
    "mitigat",
    "rescu",
    "benefit",
    "protect",
    "increase",
    "decrease",
    "enhanc",
    "reduc",
    "restore",
    "attenuat",
    "efficac",
    "promot",
    "inhibit",
    "prevent",
    "ameliorat",
    "reverse",
)
EXCLUDED_FILE_NAMES = (
    "manifest",
    "metadata_snapshot",
    "audit",
    "receipt",
    "schema",
    "readme",
    "license",
    "terms",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_policy() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config.get("retrospective_public_source_data_first_intake")
    if not isinstance(policy, dict):
        raise ValueError("Missing retrospective_public_source_data_first_intake configuration")
    authorization = policy["authorization"]
    if len(policy["repository_families"]) != int(authorization["repository_family_cap"]):
        raise ValueError("Repository family count does not match the frozen cap")
    if sum(int(item["asset_cap"]) for item in policy["repository_families"]) > int(
        authorization["metadata_asset_cap"]
    ):
        raise ValueError("Per-family asset caps exceed the total metadata cap")
    return policy


def _output_paths(policy: dict[str, Any]) -> dict[str, Path]:
    paths = {key: (REPO_ROOT / str(value)).resolve() for key, value in policy["outputs"].items()}
    for key, path in paths.items():
        if path != REPO_ROOT and REPO_ROOT not in path.parents:
            raise ValueError(f"Configured source-data-first output escapes repository: {key}")
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _assert_parent_state(policy: dict[str, Any]) -> dict[str, dict[str, object]]:
    expected = policy["parent_state"]
    frozen = _read_json(FROZEN_QUEUE_AUDIT_PATH)
    refresh = _read_json(REFRESH_QUEUE_AUDIT_PATH)
    prior = _read_json(PRIOR_INTAKE_AUDIT_PATH)
    design = _read_json(PRIOR_DESIGN_INTAKE_AUDIT_PATH)
    design_recovery = _read_json(PRIOR_DESIGN_RECOVERY_PATH)
    materialization = _read_json(MATERIALIZATION_AUDIT_PATH)
    replay = _read_json(REPLAY_AUDIT_PATH)
    observed = {
        "status": materialization.get("status"),
        "accepted_records": materialization.get("accepted_record_count"),
        "rejected_records": materialization.get("rejected_record_count"),
        "legal_candidate_sets": materialization.get("legal_candidate_set_count"),
        "independent_public_contexts": materialization.get("independent_context_count"),
        "independent_decision_sensitive_groups": replay.get("independent_decision_sensitive_group_count"),
        "required_groups": replay.get("minimum_independent_decision_sensitive_groups"),
        "publication_first_route_exhausted": design_recovery.get("status") == "not_run_queue_empty",
        "frozen_queue_digest": frozen.get("queue_digest"),
        "refresh_queue_digest": refresh.get("queue_digest"),
        "prior_public_intake_queue_digest": prior.get("queue_digest"),
        "design_enriched_intake_queue_digest": design.get("queue_digest"),
        "protected_holdout_access": replay.get("protected_holdout_access"),
        "split_files_created": replay.get("split_files_created"),
    }
    required = {**expected, "split_files_created": False}
    if observed != required:
        raise ValueError(f"Source-data-first parent state drifted: observed={observed} expected={required}")
    return {
        "resource_snapshot": _binding(RESOURCE_SNAPSHOT_PATH),
        "frozen_queue_audit": _binding(FROZEN_QUEUE_AUDIT_PATH),
        "refresh_queue_audit": _binding(REFRESH_QUEUE_AUDIT_PATH),
        "prior_intake_audit": _binding(PRIOR_INTAKE_AUDIT_PATH),
        "design_enriched_intake_audit": _binding(PRIOR_DESIGN_INTAKE_AUDIT_PATH),
        "design_enriched_recovery": _binding(PRIOR_DESIGN_RECOVERY_PATH),
        "materialization_audit": _binding(MATERIALIZATION_AUDIT_PATH),
        "replay_audit": _binding(REPLAY_AUDIT_PATH),
    }


def _read_table_values(path: Path, fields: Iterable[str]) -> dict[str, set[str]]:
    values = {field: set() for field in fields}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        available = set(reader.fieldnames or ())
        for row in reader:
            for field in fields:
                if field in available and str(row.get(field) or "").strip():
                    values[field].add(str(row[field]).strip())
    return values


def _collect_exclusions() -> dict[str, Any]:
    publication_uids: set[str] = set()
    source_paths: set[str] = set()
    source_hashes: set[str] = set()
    repository_record_uids: set[str] = set()
    for path in EXCLUSION_TABLES:
        values = _read_table_values(
            path,
            ("publication_uid", "source_artifact_path", "source_content_hash", "repository_record_uid"),
        )
        publication_uids.update(values["publication_uid"])
        source_paths.update(values["source_artifact_path"])
        source_hashes.update(values["source_content_hash"])
        repository_record_uids.update(values["repository_record_uid"])
    prior_metadata_pmids: set[str] = set()
    for path in PRIOR_METADATA_SNAPSHOTS:
        snapshot = _read_json(path)
        for query in snapshot.get("search_plan", {}).get("queries", []):
            prior_metadata_pmids.update(str(value) for value in query.get("returned_pmids", []))
    return {
        "publication_uids": sorted(publication_uids),
        "pmids": sorted(
            value.removeprefix("pubmed:")
            for value in publication_uids
            if value.startswith("pubmed:")
        ),
        "dois": sorted(
            value.removeprefix("doi:").lower()
            for value in publication_uids
            if value.startswith("doi:")
        ),
        "source_artifact_paths": sorted(source_paths),
        "source_content_hashes": sorted(source_hashes),
        "repository_record_uids": sorted(repository_record_uids),
        "prior_publication_metadata_pmids": sorted(prior_metadata_pmids, key=int),
        "prior_publication_metadata_count": len(prior_metadata_pmids),
    }


def _schema_errors(payload: dict[str, Any], schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{list(error.absolute_path)}: {error.message}" for error in errors]


def _validate_preregistration(payload: dict[str, Any], policy: dict[str, Any]) -> None:
    errors = _schema_errors(payload, PREREGISTRATION_SCHEMA_PATH)
    if errors:
        raise ValueError("Source-data-first preregistration schema errors: " + "; ".join(errors[:8]))
    digest_payload = dict(payload)
    observed_digest = str(digest_payload.pop("preregistration_digest"))
    if _canonical_digest(digest_payload) != observed_digest:
        raise ValueError("Source-data-first preregistration digest drifted")
    for field in (
        "intake_uid",
        "authorization",
        "parent_state",
        "repository_families",
        "metadata_policy",
        "selection_policy",
        "boundaries",
        "outputs",
    ):
        if payload[field] != policy[field]:
            raise ValueError(f"Source-data-first preregistration {field} drifted")


def preregister(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    if paths["preregistration_manifest"].is_file():
        payload = _read_json(paths["preregistration_manifest"])
        _validate_preregistration(payload, policy)
        return payload
    payload: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-public-source-data-first-preregistration-v1",
        "status": "preregistered_before_repository_metadata_access",
        "intake_uid": policy["intake_uid"],
        "preregistered_at": _now(),
        "authorization": policy["authorization"],
        "parent_state": policy["parent_state"],
        "parent_state_digest": _canonical_digest(policy["parent_state"]),
        "parent_bindings": _assert_parent_state(policy),
        "repository_families": policy["repository_families"],
        "metadata_policy": policy["metadata_policy"],
        "selection_policy": policy["selection_policy"],
        "exclusions": _collect_exclusions(),
        "boundaries": policy["boundaries"],
        "outputs": policy["outputs"],
    }
    payload["preregistration_digest"] = _canonical_digest(payload)
    _validate_preregistration(payload, policy)
    if write_output:
        write_json(paths["preregistration_manifest"], payload)
    return payload


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object] | None = None,
    max_bytes: int = 4 * 1024 * 1024,
) -> tuple[str, dict[str, Any] | None]:
    last_error = ""
    for _ in range(2):
        try:
            response = session.get(url, params=params, timeout=(10, 45))
            response.raise_for_status()
            if len(response.content) > max_bytes:
                return "payload_limit", None
            payload = response.json()
            if not isinstance(payload, dict):
                return "invalid_payload", None
            return "resolved", payload
        except (requests.RequestException, ValueError, TypeError) as exc:
            last_error = f"{type(exc).__name__}:{exc}"
    return f"access_blocked:{last_error}", None


def _file_extension(name: str) -> str:
    lowered = name.lower()
    for extension in sorted(MACHINE_EXTENSIONS, key=len, reverse=True):
        if lowered.endswith(extension):
            return extension
    return Path(lowered).suffix


def _metadata_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(str(term).lower() in text for term in terms)


def _local_inventory(
    family: dict[str, Any],
    exclusions: dict[str, Any],
    *,
    maximum_single_asset_bytes: int,
) -> tuple[str, list[dict[str, Any]]]:
    root = (REPO_ROOT / str(family["root"])).resolve()
    excluded_paths = set(exclusions["source_artifact_paths"])
    rows: list[dict[str, Any]] = []
    visited = 0
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            visited += 1
            if visited > 5000:
                raise ValueError("Local source-data metadata inventory exceeded 5000 paths")
            path = Path(directory) / filename
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative in excluded_paths:
                continue
            extension = _file_extension(filename)
            if extension not in MACHINE_EXTENSIONS:
                continue
            lowered = filename.lower()
            if any(term in lowered for term in EXCLUDED_FILE_NAMES):
                continue
            if not any(term in lowered for term in QUANTITATIVE_NAME_TERMS):
                continue
            stat = path.stat()
            if stat.st_size > maximum_single_asset_bytes:
                continue
            pmc_match = re.search(r"pmc\d+", relative, re.IGNORECASE)
            pmid_match = re.search(r"pmid[_:-]?(\d+)", relative, re.IGNORECASE)
            publication_uid = (
                f"pmc:{pmc_match.group(0).upper()}"
                if pmc_match
                else f"pubmed:{pmid_match.group(1)}"
                if pmid_match
                else ""
            )
            metadata_text = " ".join((relative, filename))
            rows.append(
                {
                    "repository_family": family["family_uid"],
                    "repository_record_uid": path.parent.name,
                    "asset_name": filename,
                    "asset_type": extension.lstrip("."),
                    "asset_size_bytes": stat.st_size,
                    "asset_url": "",
                    "local_path": relative,
                    "study_title": path.parent.name.replace("_", " "),
                    "publication_uid": publication_uid,
                    "doi": "",
                    "release_date": "",
                    "public_access_status": "public_local_snapshot",
                    "metadata_source": "local_path_and_file_stat_only",
                    "metadata_text": metadata_text,
                }
            )
            if len(rows) >= int(family["asset_cap"]):
                return "resolved", rows
    return "resolved", rows


def _identifier_fields(value: Any, output: dict[str, str]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _identifier_fields(child, output)
        return
    if isinstance(value, list):
        for child in value:
            _identifier_fields(child, output)
        return
    text = str(value or "")
    pmid = re.search(r"(?:PMID[:\s]?|pubmed/)(\d{6,9})", text, re.IGNORECASE)
    pmc = re.search(r"PMC\d+", text, re.IGNORECASE)
    doi = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    if pmid and "publication_uid" not in output:
        output["publication_uid"] = f"pubmed:{pmid.group(1)}"
    if pmc and "pmcid" not in output:
        output["pmcid"] = pmc.group(0).upper()
    if doi and "doi" not in output:
        output["doi"] = doi.group(0).rstrip(".,;)").lower()


def _biostudies_inventory(
    session: requests.Session,
    family: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    seen_assets: set[str] = set()
    for query in family["query_plan"]:
        status, payload = _request_json(
            session,
            str(family["search_url"]),
            params={"query": query, "pageSize": 20, "page": 1},
        )
        statuses.append(status)
        if status != "resolved" or payload is None:
            continue
        for hit in payload.get("hits", []):
            if len(rows) >= int(family["asset_cap"]):
                break
            if not isinstance(hit, dict) or hit.get("isPublic") is not True:
                continue
            accession = str(hit.get("accession") or "").strip()
            if not accession:
                continue
            detail_url = str(family["detail_url_template"]).format(accession=accession)
            detail_status, detail = _request_json(session, detail_url, max_bytes=8 * 1024 * 1024)
            statuses.append(detail_status)
            if detail_status != "resolved" or detail is None:
                continue
            identifiers: dict[str, str] = {}
            _identifier_fields(detail, identifiers)
            title = " ".join(str(hit.get("title") or "").split())
            content = " ".join(str(hit.get("content") or "").split())[:4000]
            for entry in iter_arrayexpress_file_entries(detail.get("section", {})):
                if len(rows) >= int(family["asset_cap"]):
                    break
                name = str(entry.get("path") or "").strip()
                if not name:
                    continue
                extension = _file_extension(name)
                if extension not in MACHINE_EXTENSIONS:
                    continue
                asset_uid = f"{family['family_uid']}:{accession}:{name}"
                if asset_uid in seen_assets:
                    continue
                seen_assets.add(asset_uid)
                size = int(entry.get("size") or 0)
                if size > 0 and size > 67_108_864:
                    continue
                metadata_text = " ".join((title, content, accession, name))
                rows.append(
                    {
                        "repository_family": family["family_uid"],
                        "repository_record_uid": accession,
                        "asset_name": name,
                        "asset_type": extension.lstrip("."),
                        "asset_size_bytes": size,
                        "asset_url": str(family["file_url_template"]).format(
                            accession=accession,
                            file_path=name,
                        ),
                        "local_path": "",
                        "study_title": title,
                        "publication_uid": identifiers.get("publication_uid", ""),
                        "doi": identifiers.get("doi", ""),
                        "release_date": str(hit.get("release_date") or ""),
                        "public_access_status": "public",
                        "metadata_source": "biostudies_search_and_detail_file_metadata",
                        "metadata_text": metadata_text,
                    }
                )
    overall_status = "resolved" if statuses and all(status == "resolved" for status in statuses) else "access_blocked"
    return overall_status, rows


def _europe_pmc_inventory(
    session: requests.Session,
    family: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    seen: set[str] = set()
    for query in family["query_plan"]:
        status, payload = _request_json(
            session,
            str(family["search_url"]),
            params={"query": query, "format": "json", "resultType": "lite", "pageSize": 20},
        )
        statuses.append(status)
        if status != "resolved" or payload is None:
            continue
        results = payload.get("resultList", {}).get("result", [])
        for result in results:
            if len(rows) >= int(family["asset_cap"]):
                break
            if not isinstance(result, dict) or str(result.get("hasSuppl") or "N") != "Y":
                continue
            pmcid = str(result.get("pmcid") or "").strip()
            pmid = str(result.get("pmid") or result.get("id") or "").strip()
            if not pmcid or not pmid or pmcid in seen:
                continue
            seen.add(pmcid)
            title = " ".join(str(result.get("title") or "").split())
            rows.append(
                {
                    "repository_family": family["family_uid"],
                    "repository_record_uid": pmcid,
                    "asset_name": f"{pmcid}_supplementary_bundle.zip",
                    "asset_type": "supplementary_bundle",
                    "asset_size_bytes": 0,
                    "asset_url": f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles",
                    "local_path": "",
                    "study_title": title,
                    "publication_uid": f"pubmed:{pmid}",
                    "doi": str(result.get("doi") or "").lower(),
                    "release_date": str(result.get("firstPublicationDate") or result.get("journalInfo", {}).get("printPublicationDate") or ""),
                    "public_access_status": "public_open_access",
                    "metadata_source": "europe_pmc_lite_has_supplement_metadata",
                    "metadata_text": title,
                }
            )
    overall_status = "resolved" if statuses and all(status == "resolved" for status in statuses) else "access_blocked"
    return overall_status, rows


def _asset_uid(record: dict[str, Any]) -> str:
    return "asset:" + _canonical_digest(
        {
            "repository_family": record["repository_family"],
            "repository_record_uid": record["repository_record_uid"],
            "asset_name": record["asset_name"],
            "asset_url": record["asset_url"],
            "local_path": record["local_path"],
        }
    )[:24]


def _assess_asset(
    record: dict[str, Any],
    *,
    policy: dict[str, Any],
    exclusions: dict[str, Any],
) -> dict[str, str]:
    metadata_text = " ".join(str(record.get("metadata_text") or "").split())
    lowered = f" {metadata_text.lower()} "
    title = str(record.get("study_title") or "")
    publication_uid = str(record.get("publication_uid") or "")
    doi = str(record.get("doi") or "").lower()
    local_path = str(record.get("local_path") or "")
    asset_name = str(record.get("asset_name") or "")
    asset_type = str(record.get("asset_type") or "")
    context_family = _context_family(metadata_text) or ""
    objective_family = _objective_family(metadata_text) or ""
    design = _design_class(metadata_text)
    named_entities = _named_intervention_entities(metadata_text)
    structured_signal = _contains_any(lowered, STRUCTURED_INTERVENTION_TERMS)
    shared_control_signal = _contains_any(lowered, CONTROL_TERMS)
    replication_signal = _contains_any(lowered, REPLICATION_TERMS)
    quantitative_signal = (
        f".{asset_type.lower()}" in MACHINE_EXTENSIONS
        and _contains_any(asset_name.lower(), QUANTITATIVE_NAME_TERMS)
    ) or asset_type == "supplementary_bundle"
    rejection_reason = ""
    if publication_uid in exclusions["publication_uids"]:
        rejection_reason = "prior_publication_lineage"
    elif publication_uid.startswith("pubmed:") and publication_uid.removeprefix("pubmed:") in exclusions["prior_publication_metadata_pmids"]:
        rejection_reason = "prior_publication_metadata_universe"
    elif doi and doi in exclusions["dois"]:
        rejection_reason = "prior_doi_lineage"
    elif any(doi.startswith(prefix) for prefix in policy["selection_policy"]["excluded_doi_prefixes"]):
        rejection_reason = "source_policy_excluded"
    elif local_path and local_path in exclusions["source_artifact_paths"]:
        rejection_reason = "prior_source_artifact"
    elif record.get("public_access_status") not in {"public", "public_open_access", "public_local_snapshot"}:
        rejection_reason = "not_public"
    elif _contains_any(title.lower(), OUTCOME_BEARING_TERMS):
        rejection_reason = "outcome_bearing_study_title"
    elif not publication_uid and not record.get("repository_record_uid"):
        rejection_reason = "missing_publication_or_study_identity"
    elif not context_family:
        rejection_reason = "no_disease_injury_or_pathological_context"
    elif not _contains_any(lowered, CNS_MODEL_TERMS):
        rejection_reason = "no_cns_or_neural_model"
    elif not objective_family:
        rejection_reason = "no_functional_endpoint"
    elif design is None:
        rejection_reason = "no_ranking_informative_design"
    elif len(named_entities) < 2 and not structured_signal:
        rejection_reason = "insufficient_intervention_structure"
    elif not shared_control_signal:
        rejection_reason = "no_shared_control_metadata_signal"
    elif not replication_signal:
        rejection_reason = "no_replication_or_uncertainty_metadata_signal"
    elif not quantitative_signal:
        rejection_reason = "no_quantitative_source_data_asset"
    design_class = design[0] if design else ""
    objective_uid = (
        policy["selection_policy"]["objective_uids"].get(objective_family, "")
        if objective_family
        else ""
    )
    return {
        "asset_uid": _asset_uid(record),
        "repository_family": str(record["repository_family"]),
        "repository_record_uid": str(record["repository_record_uid"]),
        "asset_name": asset_name,
        "asset_type": asset_type,
        "asset_size_bytes": str(record.get("asset_size_bytes") or 0),
        "asset_url": str(record.get("asset_url") or ""),
        "local_path": local_path,
        "study_title": title,
        "publication_uid": publication_uid,
        "doi": doi,
        "release_date": str(record.get("release_date") or ""),
        "public_access_status": str(record.get("public_access_status") or ""),
        "metadata_source": str(record.get("metadata_source") or ""),
        "metadata_text_sha256": _metadata_hash(metadata_text),
        "context_family": context_family,
        "objective_uid": objective_uid,
        "study_design_class": design_class,
        "named_intervention_entities": ";".join(named_entities),
        "named_intervention_entity_count": str(len(named_entities)),
        "structured_intervention_signal": str(structured_signal).lower(),
        "shared_control_signal": str(shared_control_signal).lower(),
        "replication_or_uncertainty_signal": str(replication_signal).lower(),
        "quantitative_asset_signal": str(quantitative_signal).lower(),
        "eligibility_status": "eligible" if not rejection_reason else "rejected",
        "rejection_reason": rejection_reason,
        "source_asset_content_retrieved": "false",
        "protected_holdout_access": "0",
    }


def _priority_key(record: dict[str, str], selected: list[dict[str, str]]) -> tuple[object, ...]:
    extension_rank = {
        "csv": 0,
        "tsv": 0,
        "xls": 0,
        "xlsx": 0,
        "zip": 1,
        "json": 2,
        "txt": 2,
        "supplementary_bundle": 3,
    }
    design_rank = {
        "combination_vs_monotherapy": 0,
        "direct_head_to_head": 1,
        "dose_response": 2,
        "multi_arm_screen": 3,
    }
    selected_contexts = {item["context_family"] for item in selected}
    selected_families = {item["repository_family"] for item in selected}
    return (
        extension_rank.get(record["asset_type"], 4),
        design_rank.get(record["study_design_class"], 4),
        -int(record["named_intervention_entity_count"]),
        record["context_family"] in selected_contexts,
        record["repository_family"] in selected_families,
        record["publication_uid"] or record["repository_record_uid"],
        record["asset_uid"],
    )


def _queue_rows(
    eligible: list[dict[str, str]],
    *,
    preregistration: dict[str, Any],
    universe_digest: str,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    remaining = list(eligible)
    cap = int(preregistration["authorization"]["preregistered_source_cap"])
    used_studies: set[str] = set()
    while remaining and len(selected) < cap:
        remaining.sort(key=lambda record: _priority_key(record, selected))
        candidate = remaining.pop(0)
        study_key = candidate["publication_uid"] or candidate["repository_record_uid"]
        if study_key in used_studies:
            continue
        used_studies.add(study_key)
        selected.append(candidate)
    rows: list[dict[str, str]] = []
    for index, record in enumerate(selected, start=1):
        rows.append(
            {
                "intake_uid": str(preregistration["intake_uid"]),
                "queue_sequence": str(index),
                "preregistration_digest": str(preregistration["preregistration_digest"]),
                "parent_state_digest": str(preregistration["parent_state_digest"]),
                "metadata_universe_digest": universe_digest,
                "asset_uid": record["asset_uid"],
                "repository_family": record["repository_family"],
                "repository_record_uid": record["repository_record_uid"],
                "asset_name": record["asset_name"],
                "asset_type": record["asset_type"],
                "asset_size_bytes": record["asset_size_bytes"],
                "asset_url": record["asset_url"],
                "local_path": record["local_path"],
                "study_title": record["study_title"],
                "publication_uid": record["publication_uid"],
                "doi": record["doi"],
                "release_date": record["release_date"],
                "context_family": record["context_family"],
                "objective_uid": record["objective_uid"],
                "study_design_class": record["study_design_class"],
                "named_intervention_entities": record["named_intervention_entities"],
                "named_intervention_entity_count": record["named_intervention_entity_count"],
                "structured_intervention_signal": record["structured_intervention_signal"],
                "shared_control_signal": record["shared_control_signal"],
                "replication_or_uncertainty_signal": record["replication_or_uncertainty_signal"],
                "ranking_priority": str(index),
                "queue_status": "preregistered_for_source_data_recovery",
                "source_asset_content_retrieved_before_priority": "false",
                "numeric_outcomes_retrieved_before_priority": "false",
                "effect_direction_used_for_priority": "false",
                "p_values_used_for_priority": "false",
                "winner_statements_used_for_priority": "false",
                "protected_holdout_access": "0",
            }
        )
    return rows


def _queue_digest(rows: list[dict[str, str]], preregistration: dict[str, Any]) -> str:
    return _canonical_digest(
        {
            "queue_digest_contract": "source_data_intake_envelope_v1",
            "intake_uid": preregistration["intake_uid"],
            "preregistration_digest": preregistration["preregistration_digest"],
            "parent_state_digest": preregistration["parent_state_digest"],
            "records": rows,
        }
    )


def _universe_digest(rows: list[dict[str, str]], preregistration: dict[str, Any]) -> str:
    return _canonical_digest(
        {
            "metadata_universe_digest_contract": "source_data_universe_envelope_v1",
            "intake_uid": preregistration["intake_uid"],
            "preregistration_digest": preregistration["preregistration_digest"],
            "parent_state_digest": preregistration["parent_state_digest"],
            "records": rows,
        }
    )


def archive_access_blocked(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    required = (
        paths["metadata_universe"],
        paths["source_queue"],
        paths["metadata_snapshot"],
        paths["intake_audit"],
    )
    if not all(path.is_file() for path in required):
        raise ValueError("Access-block archival requires a complete discovery attempt")
    audit = _read_json(paths["intake_audit"])
    if audit.get("status") != "repository_metadata_access_blocked":
        raise ValueError("Current source-data-first intake is not access blocked")
    statuses = audit.get("repository_family_status", [])
    if not any(item.get("status") != "resolved" for item in statuses):
        raise ValueError("Access-block audit has no blocked repository family")
    invalidation = {
        "schema_version": "neuronsifter-source-data-first-access-blocked-attempt-v1",
        "status": "invalidated_compute_node_network_access_failure",
        "scientific_result_generated": False,
        "repository_queries_changed": False,
        "repository_families_changed": False,
        "metadata_asset_content_retrieved": False,
        "numeric_outcomes_retrieved": False,
        "protected_holdout_access": 0,
        "family_status": statuses,
        "failed_universe_binding": _binding(paths["metadata_universe"]),
        "failed_queue_binding": _binding(paths["source_queue"]),
        "failed_snapshot_binding": _binding(paths["metadata_snapshot"]),
        "failed_audit_binding": _binding(paths["intake_audit"]),
        "retry_rule": "same_preregistration_same_repository_requests_on_cpu03",
        "invalidated_at": _now(),
    }
    if write_output:
        targets = (
            (paths["metadata_universe"], ACCESS_BLOCKED_UNIVERSE_PATH),
            (paths["source_queue"], ACCESS_BLOCKED_QUEUE_PATH),
            (paths["metadata_snapshot"], ACCESS_BLOCKED_SNAPSHOT_PATH),
            (paths["intake_audit"], ACCESS_BLOCKED_AUDIT_PAYLOAD_PATH),
        )
        for source, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha256(target) != _sha256(source):
                    raise ValueError(f"Existing access-block archive drifted: {target}")
            else:
                shutil.copy2(source, target)
        write_json(ACCESS_BLOCKED_INVALIDATION_PATH, invalidation)
    return invalidation


def archive_schema_failure(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    required = (
        paths["metadata_universe"],
        paths["source_queue"],
        paths["metadata_snapshot"],
        paths["intake_audit"],
    )
    if not all(path.is_file() for path in required):
        raise ValueError("Schema-failure archival requires all partial output paths")
    stale_audit = _read_json(paths["intake_audit"])
    snapshot = _read_json(paths["metadata_snapshot"])
    if stale_audit.get("status") != "repository_metadata_access_blocked":
        raise ValueError("Expected the prior access-blocked audit at the canonical path")
    if not all(
        item.get("status") == "resolved"
        for item in snapshot.get("repository_family_status", [])
    ):
        raise ValueError("Schema-failure snapshot does not contain a fully resolved repository wave")
    invalidation = {
        "schema_version": "neuronsifter-source-data-first-schema-failure-attempt-v1",
        "status": "invalidated_post_metadata_pre_audit_schema_failure",
        "reason": "intake_audit_omitted_required_digest_contract_fields",
        "scientific_result_generated": False,
        "repository_queries_changed": False,
        "repository_families_changed": False,
        "metadata_asset_content_retrieved": False,
        "numeric_outcomes_retrieved": False,
        "protected_holdout_access": 0,
        "partial_metadata_asset_count": snapshot.get("metadata_asset_count"),
        "partial_metadata_eligible_count": snapshot.get("metadata_eligible_count"),
        "partial_queue_record_count": snapshot.get("queue_record_count"),
        "partial_universe_binding": _binding(paths["metadata_universe"]),
        "partial_queue_binding": _binding(paths["source_queue"]),
        "partial_snapshot_binding": _binding(paths["metadata_snapshot"]),
        "stale_access_blocked_audit_binding": _binding(paths["intake_audit"]),
        "retry_rule": "same_preregistration_same_repository_requests_on_cpu03_after_schema_fix",
        "invalidated_at": _now(),
    }
    if write_output:
        targets = (
            (paths["metadata_universe"], SCHEMA_FAILURE_UNIVERSE_PATH),
            (paths["source_queue"], SCHEMA_FAILURE_QUEUE_PATH),
            (paths["metadata_snapshot"], SCHEMA_FAILURE_SNAPSHOT_PATH),
            (paths["intake_audit"], SCHEMA_FAILURE_AUDIT_PAYLOAD_PATH),
        )
        for source, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha256(target) != _sha256(source):
                    raise ValueError(f"Existing schema-failure archive drifted: {target}")
            else:
                shutil.copy2(source, target)
        write_json(SCHEMA_FAILURE_INVALIDATION_PATH, invalidation)
    return invalidation


def archive_proxy_failure(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    required = (
        paths["metadata_universe"],
        paths["source_queue"],
        paths["metadata_snapshot"],
        paths["intake_audit"],
    )
    if not all(path.is_file() for path in required):
        raise ValueError("Proxy-failure archival requires all current output paths")
    audit = _read_json(paths["intake_audit"])
    snapshot = _read_json(paths["metadata_snapshot"])
    statuses = snapshot.get("repository_family_status", [])
    external = [item for item in statuses if item.get("adapter") != "bounded_local_file_metadata_inventory"]
    if audit.get("status") != "repository_metadata_access_blocked":
        raise ValueError("Expected the prior access-blocked audit at the canonical path")
    if not external or any(item.get("status") == "resolved" for item in external):
        raise ValueError("Proxy-failure signature does not cover all external repository families")
    invalidation = {
        "schema_version": "neuronsifter-source-data-first-proxy-failure-attempt-v1",
        "status": "invalidated_compute_node_proxy_failure",
        "reason": "requests_inherited_unreachable_loopback_proxy_on_compute_node",
        "scientific_result_generated": False,
        "repository_queries_changed": False,
        "repository_families_changed": False,
        "metadata_asset_content_retrieved": False,
        "numeric_outcomes_retrieved": False,
        "protected_holdout_access": 0,
        "family_status": statuses,
        "failed_universe_binding": _binding(paths["metadata_universe"]),
        "failed_queue_binding": _binding(paths["source_queue"]),
        "failed_snapshot_binding": _binding(paths["metadata_snapshot"]),
        "stale_access_blocked_audit_binding": _binding(paths["intake_audit"]),
        "retry_rule": "same_preregistration_same_repository_requests_on_cpu03_without_environment_proxy",
        "invalidated_at": _now(),
    }
    if write_output:
        targets = (
            (paths["metadata_universe"], PROXY_FAILURE_UNIVERSE_PATH),
            (paths["source_queue"], PROXY_FAILURE_QUEUE_PATH),
            (paths["metadata_snapshot"], PROXY_FAILURE_SNAPSHOT_PATH),
            (paths["intake_audit"], PROXY_FAILURE_AUDIT_PAYLOAD_PATH),
        )
        for source, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha256(target) != _sha256(source):
                    raise ValueError(f"Existing proxy-failure archive drifted: {target}")
            else:
                shutil.copy2(source, target)
        write_json(PROXY_FAILURE_INVALIDATION_PATH, invalidation)
    return invalidation


def discover(*, write_output: bool = True, retry_access_blocked: bool = False) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    preregistration = preregister(write_output=write_output)
    if paths["intake_audit"].is_file() and not retry_access_blocked:
        return _read_json(paths["intake_audit"])
    if retry_access_blocked and not ACCESS_BLOCKED_INVALIDATION_PATH.is_file():
        raise ValueError("Access-block retry requires a persisted invalidation audit")
    if not retry_access_blocked and any(
        paths[key].exists() for key in ("metadata_universe", "source_queue", "metadata_snapshot")
    ):
        raise ValueError("Partial source-data-first outputs exist without a canonical intake audit")
    _assert_parent_state(policy)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "NeuronSifter-source-data-first-metadata/1.0"})
    family_status: list[dict[str, Any]] = []
    raw_assets: list[dict[str, Any]] = []
    for family in policy["repository_families"]:
        if family["adapter"] == "bounded_local_file_metadata_inventory":
            status, records = _local_inventory(
                family,
                preregistration["exclusions"],
                maximum_single_asset_bytes=int(
                    policy["metadata_policy"]["maximum_single_asset_bytes"]
                ),
            )
        elif family["adapter"] == "biostudies_search_and_detail_metadata":
            status, records = _biostudies_inventory(session, family)
        elif family["adapter"] == "europe_pmc_lite_supplement_metadata":
            status, records = _europe_pmc_inventory(session, family)
        else:
            raise ValueError(f"Unsupported repository adapter: {family['adapter']}")
        family_status.append(
            {
                "family_uid": family["family_uid"],
                "adapter": family["adapter"],
                "status": status,
                "metadata_asset_count": len(records),
                "asset_cap": int(family["asset_cap"]),
            }
        )
        raw_assets.extend(records)
    deduplicated: dict[str, dict[str, Any]] = {}
    for record in raw_assets:
        uid = _asset_uid(record)
        deduplicated.setdefault(uid, record)
    if len(deduplicated) > int(policy["authorization"]["metadata_asset_cap"]):
        raise ValueError("Source-data metadata universe exceeded the preregistered cap")
    universe_rows = [
        _assess_asset(record, policy=policy, exclusions=preregistration["exclusions"])
        for _, record in sorted(deduplicated.items())
    ]
    eligible = [row for row in universe_rows if row["eligibility_status"] == "eligible"]
    universe_digest = _universe_digest(universe_rows, preregistration)
    queue_rows = _queue_rows(
        eligible,
        preregistration=preregistration,
        universe_digest=universe_digest,
    )
    queue_digest = _queue_digest(queue_rows, preregistration)
    access_blocked = any(item["status"] != "resolved" for item in family_status)
    status = (
        "repository_metadata_access_blocked"
        if access_blocked
        else "source_data_queue_frozen"
        if queue_rows
        else "source_data_queue_exhausted"
    )
    next_action = (
        "retry_same_preregistered_repository_metadata_requests"
        if access_blocked
        else "recover_source_data_assets_in_priority_order"
        if queue_rows
        else "stop_source_data_first_queue_exhausted"
    )
    snapshot = {
        "schema_version": "neuronsifter-source-data-first-metadata-snapshot-v1",
        "retrieved_at": _now(),
        "retrieval_mode": "repository_metadata_only",
        "preregistration_binding": {
            "path": paths["preregistration_manifest"].relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(paths["preregistration_manifest"]),
            "preregistration_digest": preregistration["preregistration_digest"],
        },
        "repository_family_status": family_status,
        "metadata_asset_count": len(universe_rows),
        "metadata_eligible_count": len(eligible),
        "queue_record_count": len(queue_rows),
        "metadata_universe_digest": universe_digest,
        "metadata_universe_digest_contract": "source_data_universe_envelope_v1",
        "queue_digest": queue_digest,
        "queue_digest_contract": "source_data_queue_envelope_v1",
        "source_boundaries": {
            "source_asset_content_retrieved": False,
            "numeric_outcomes_retrieved": False,
            "effect_direction_used_for_priority": False,
            "p_values_used_for_priority": False,
            "winner_statements_used_for_priority": False,
            "protected_holdout_access": 0,
        },
        "selection": {
            "repository_family_cap": policy["authorization"]["repository_family_cap"],
            "metadata_asset_cap": policy["authorization"]["metadata_asset_cap"],
            "per_family_asset_cap": policy["authorization"]["per_family_asset_cap"],
            "preregistered_source_cap": policy["authorization"]["preregistered_source_cap"],
            "ranking_order": policy["selection_policy"]["ranking_order"],
        },
    }
    serialized = json.dumps(snapshot, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    snapshot_binding = {
        "path": paths["metadata_snapshot"].relative_to(REPO_ROOT).as_posix(),
        "bytes": len(serialized),
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }
    audit: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-public-source-data-first-intake-v1",
        "status": status,
        "intake_uid": preregistration["intake_uid"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "parent_state_digest": preregistration["parent_state_digest"],
        "repository_family_count": len(policy["repository_families"]),
        "metadata_asset_cap": int(policy["authorization"]["metadata_asset_cap"]),
        "metadata_asset_count": len(universe_rows),
        "metadata_eligible_count": len(eligible),
        "queue_record_count": len(queue_rows),
        "metadata_universe_digest": universe_digest,
        "metadata_universe_digest_contract": "source_data_universe_envelope_v1",
        "queue_digest": queue_digest,
        "queue_digest_contract": "source_data_queue_envelope_v1",
        "metadata_universe_path": paths["metadata_universe"].relative_to(REPO_ROOT).as_posix(),
        "queue_path": paths["source_queue"].relative_to(REPO_ROOT).as_posix(),
        "metadata_snapshot_binding": snapshot_binding,
        "repository_family_status": family_status,
        "funnel": {
            "repository_assets_screened": len(universe_rows),
            "metadata_eligible": len(eligible),
            "preregistered": len(queue_rows),
            "source_data_recovered": 0,
            "formal_abc": 0,
            "legal": 0,
            "decision_sensitive": 0,
        },
        "historical_lineage_unchanged": True,
        "source_asset_content_retrieved_before_queue_freeze": False,
        "numeric_outcomes_retrieved_before_queue_freeze": False,
        "effect_direction_used_for_priority": False,
        "p_values_used_for_priority": False,
        "winner_statements_used_for_priority": False,
        "protected_holdout_access": 0,
        "formal_evidence_admitted": 0,
        "next_action": next_action,
        "queue_records": queue_rows,
        "created_at": _now(),
    }
    errors = _schema_errors(audit, INTAKE_SCHEMA_PATH)
    if errors:
        raise ValueError("Source-data-first intake schema errors: " + "; ".join(errors[:8]))
    if write_output:
        write_tsv(paths["metadata_universe"], list(UNIVERSE_FIELDS), universe_rows)
        write_tsv(paths["source_queue"], list(QUEUE_FIELDS), queue_rows)
        write_json(paths["metadata_snapshot"], snapshot)
        write_json(paths["intake_audit"], audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "preregister",
            "discover",
            "archive-access-blocked",
            "archive-schema-failure",
            "archive-proxy-failure",
        ),
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--retry-access-blocked", action="store_true")
    args = parser.parse_args()
    if args.mode == "preregister":
        result = preregister(write_output=not args.no_write)
    elif args.mode == "archive-access-blocked":
        result = archive_access_blocked(write_output=not args.no_write)
    elif args.mode == "archive-schema-failure":
        result = archive_schema_failure(write_output=not args.no_write)
    elif args.mode == "archive-proxy-failure":
        result = archive_proxy_failure(write_output=not args.no_write)
    else:
        result = discover(
            write_output=not args.no_write,
            retry_access_blocked=args.retry_access_blocked,
        )
    keys = (
        "status",
        "intake_uid",
        "preregistration_digest",
        "parent_state_digest",
        "repository_family_count",
        "metadata_asset_cap",
        "metadata_asset_count",
        "metadata_eligible_count",
        "queue_record_count",
        "metadata_universe_digest",
        "queue_digest",
        "next_action",
    )
    print(json.dumps({key: result[key] for key in keys if key in result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
