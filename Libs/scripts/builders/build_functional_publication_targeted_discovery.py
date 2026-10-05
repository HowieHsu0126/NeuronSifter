"""Run one bounded, outcome-blind metadata-only functional publication wave."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import requests
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/functional_publication_targeted_discovery.schema.json"
SOURCE_METADATA_PATH = REPO_ROOT / "Input/manifests/functional_therapeutic/source_metadata.tsv"
PUBLICATION_AUDIT_PATH = REPO_ROOT / "Output/audits/functional_publication_evidence_recovery.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ENDPOINT_TERMS = {
    "neurite_morphology": ("neurite", "neuritogenic", "morpholog"),
    "electrophysiology": ("electrophysi", "excitability", "electrical"),
    "calcium_activity": ("calcium", "calcium imaging"),
    "viability_survival": ("viability", "survival"),
    "synaptic_phenotype": ("synaptic", "synapse"),
    "pathology_phenotype": ("patholog", "functional recovery", "disease burden"),
}
_INTERVENTION_TERMS = (
    "drug",
    "compound",
    "treatment",
    "therapy",
    "therapeutic",
    "inhibitor",
    "agonist",
    "antagonist",
    "screen",
)
_MULTI_ARM_TERMS = ("dose", "multi", "combination", "versus", "comparison", "screen", "treatments")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes repository: {value}")
    return path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        return [dict(row) for row in reader]


def _request(session: requests.Session, query: str, page_size: int) -> tuple[str, list[dict[str, Any]]]:
    try:
        response = session.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "resultType": "lite", "pageSize": str(page_size)},
            timeout=(10, 30),
        )
        response.raise_for_status()
        payload = response.content
        if len(payload) > 2 * 1024 * 1024:
            return "access_blocked", []
        body = json.loads(payload.decode("utf-8"))
        results = body.get("resultList", {}).get("result", [])
        return "resolved", [dict(item) for item in results if isinstance(item, dict)]
    except (requests.RequestException, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return "access_blocked", []


def _seed_identifiers(rows: list[dict[str, str]], publication_audit: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for row in rows:
        for value in str(row.get("publication_uid", "")).split(";"):
            if value.startswith("pubmed:"):
                identifiers.add(f"pmid:{value.split(':', 1)[1]}")
            elif value.startswith("doi:"):
                identifiers.add(f"doi:{value.split(':', 1)[1].lower()}")
    for record in publication_audit.get("records", []):
        publication_uid = str(record.get("publication_uid") or "")
        doi = str(record.get("doi") or "")
        if publication_uid.startswith("pubmed:"):
            identifiers.add(f"pmid:{publication_uid.split(':', 1)[1]}")
        if doi:
            identifiers.add(f"doi:{doi.lower()}")
    return identifiers


def _endpoint_family(title: str, requested: str) -> str | None:
    lowered = title.lower()
    for family, terms in _ENDPOINT_TERMS.items():
        if any(term in lowered for term in terms):
            return family
    terms = _ENDPOINT_TERMS.get(requested, ())
    return requested if any(term in lowered for term in terms) else None


def _record_status(title: str, endpoint_family: str | None) -> tuple[str, str]:
    lowered = title.lower()
    if endpoint_family is None or not any(term in lowered for term in _INTERVENTION_TERMS):
        return "not_functional_metadata", "single_or_unclear"
    if "dose" in lowered:
        return "candidate_multi_arm_metadata", "dose_response"
    if "combination" in lowered:
        return "candidate_multi_arm_metadata", "combination"
    if any(term in lowered for term in _MULTI_ARM_TERMS):
        return "candidate_multi_arm_metadata", "multi_arm"
    return "metadata_lead_single_or_unclear", "single_or_unclear"


def build(*, session: requests.Session | None = None) -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config["functional_therapeutic_evidence"]
    discovery = policy["source_discovery"]["targeted_discovery"]
    queries = list(discovery["query_registry"])
    if len(queries) != int(discovery["max_queries"]):
        raise ValueError("Targeted discovery query count does not match its frozen budget")
    seed_rows = _read_tsv(SOURCE_METADATA_PATH)
    publication_audit = json.loads(PUBLICATION_AUDIT_PATH.read_text(encoding="utf-8"))
    excluded = _seed_identifiers(seed_rows, publication_audit)
    client = session or requests.Session()
    client.headers.update({"User-Agent": "NeuronSifter-functional-targeted-discovery/1.0 (metadata-only research)"})
    query_records: list[dict[str, Any]] = []
    by_identifier: dict[str, dict[str, Any]] = {}
    for query_spec in queries:
        access_status, results = _request(client, str(query_spec["query"]), int(discovery["max_results_per_query"]))
        query_records.append(
            {
                "query_uid": query_spec["query_uid"],
                "query": query_spec["query"],
                "endpoint_family": query_spec["endpoint_family"],
                "result_count": len(results),
                "access_status": access_status,
            }
        )
        for result in results:
            pmid = str(result.get("pmid") or "").strip()
            doi = str(result.get("doi") or "").strip()
            if pmid:
                identifier = pmid
                identifier_type = "pmid"
                key = f"pmid:{pmid}"
            elif doi:
                identifier = doi
                identifier_type = "doi"
                key = f"doi:{doi.lower()}"
            else:
                continue
            title = " ".join(str(result.get("title") or "").split())
            if not title or key in excluded:
                continue
            endpoint = _endpoint_family(title, str(query_spec["endpoint_family"]))
            status, action_mode = _record_status(title, endpoint)
            current = by_identifier.get(key)
            if current is None:
                current = {
                    "lead_uid": key,
                    "identifier": identifier,
                    "identifier_type": identifier_type,
                    "title": title,
                    "year": int(result["pubYear"]) if str(result.get("pubYear", "")).isdigit() else None,
                    "journal": str(result.get("journalTitle") or "") or None,
                    "query_uids": [],
                    "endpoint_family": endpoint or str(query_spec["endpoint_family"]),
                    "metadata_design_status": status,
                    "expected_action_mode": action_mode,
                    "artifact_status": "not_inspected",
                    "functional_values_admitted": False,
                }
                by_identifier[key] = current
            if query_spec["query_uid"] not in current["query_uids"]:
                current["query_uids"].append(query_spec["query_uid"])
            if status == "candidate_multi_arm_metadata":
                current["metadata_design_status"] = status
                current["expected_action_mode"] = action_mode
    records = sorted(by_identifier.values(), key=lambda item: (item["lead_uid"], item["title"]))
    novel_lead_count = sum(record["metadata_design_status"] == "candidate_multi_arm_metadata" for record in records)
    access_blocked = any(item["access_status"] == "access_blocked" for item in query_records)
    if access_blocked:
        terminal_state = "TARGETED_DISCOVERY_ACCESS_BLOCKED"
        next_route = "repeat_bounded_metadata_query_after_access_recovers"
    elif novel_lead_count:
        terminal_state = "TARGETED_DISCOVERY_COMPLETE_LEADS_REMAIN"
        next_route = "recover_publication_artifacts_for_metadata_leads"
    else:
        terminal_state = "BLOCKED_REQUIRES_EXTERNAL_MULTI_ARM_FUNCTIONAL_OUTCOMES"
        next_route = "prepare_external_multi_arm_functional_outcome_intake"
    audit = {
        "schema_version": "neuronsifter-functional-publication-targeted-discovery-v1",
        "status": "completed_with_access_blocks" if access_blocked else "completed_bounded_metadata_only",
        "campaign_scope": "one_targeted_metadata_only_discovery_wave",
        "terminal_state": terminal_state,
        "next_route": next_route,
        "query_count": len(query_records),
        "result_count": sum(item["result_count"] for item in query_records),
        "novel_lead_count": novel_lead_count,
        "admitted_source_count": 0,
        "observation_count": 0,
        "legal_candidate_set_count": 0,
        "qualified_candidate_set_count": 0,
        "outcome_values_used_for_priority": False,
        "abstract_values_inspected": False,
        "model_predictions_inspected": False,
        "protected_holdout_access": 0,
        "publication_comparison_open": False,
        "queries": query_records,
        "records": records,
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(audit))
    if errors:
        raise ValueError("Targeted discovery schema errors: " + "; ".join(error.message for error in errors))
    manifest_path = _repo_path(str(discovery["outputs"]["manifest"]))
    audit_path = _repo_path(str(discovery["outputs"]["audit"]))
    manifest_rows = [
        {
            **record,
            "query_uids": ";".join(record["query_uids"]),
        }
        for record in records
    ]
    fields = [
        "lead_uid",
        "identifier",
        "identifier_type",
        "title",
        "year",
        "journal",
        "query_uids",
        "endpoint_family",
        "metadata_design_status",
        "expected_action_mode",
        "artifact_status",
        "functional_values_admitted",
    ]
    write_tsv(manifest_path, fields, manifest_rows)
    write_json(audit_path, {**audit, "created_at": _now()})
    return audit


def main() -> int:
    result = build()
    print(json.dumps({key: result[key] for key in ("status", "query_count", "result_count", "novel_lead_count", "admitted_source_count", "legal_candidate_set_count", "qualified_candidate_set_count", "protected_holdout_access", "terminal_state")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
