"""Recover publication-linked functional evidence metadata for the frozen seed queue.

This builder performs a bounded, metadata-first recovery pass.  It resolves GEO
series records, PubMed/Europe PMC metadata, open full-text data-availability
statements, and supplementary-file inventories.  It never admits outcome rows,
digitizes figures, or changes the frozen functional objective namespace.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import argparse
from pathlib import Path
import re
import sys
from typing import Any, Mapping
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET

import requests
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
GRAPH_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/functional_publication_evidence_graph.schema.json"
SOURCE_METADATA_PATH = REPO_ROOT / "Input/manifests/functional_therapeutic/source_metadata.tsv"
GRAPH_PATH = REPO_ROOT / "Input/manifests/functional_therapeutic/publication_evidence_graph.tsv"
AUDIT_PATH = REPO_ROOT / "Output/audits/functional_publication_evidence_recovery.json"

_ACCESSION_RE = re.compile(r"^GSE\d+$")
_PMID_RE = re.compile(r"^\d+$")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EXCLUDED_DATA_TERMS = ("expression", "counts", "count", "rna", "scrna", "fastq", "rawdata", "gene")
_MACHINE_SUFFIXES = (".csv", ".tsv", ".txt", ".xls", ".xlsx", ".zip")


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


def _tokens(value: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(value.lower()) if len(token) > 2}


def _title_match_score(seed_title: str, candidate_title: str) -> float:
    seed = _tokens(seed_title)
    candidate = _tokens(candidate_title)
    if not seed or not candidate:
        return 0.0
    return len(seed & candidate) / len(seed)


def _namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_all(root: ET.Element, name: str) -> list[ET.Element]:
    return [element for element in root.iter() if _namespace(element.tag) == name]


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = " ".join("".join(element.itertext()).split())
    return value or None


def _get_href(element: ET.Element) -> str | None:
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1] in {"href", "url"} and value:
            return value
    return None


def _request(session: requests.Session, url: str, *, params: Mapping[str, str] | None = None, max_bytes: int = 8 * 1024 * 1024) -> tuple[str, bytes | None, str | None]:
    try:
        response = session.get(url, params=params, timeout=(10, 45), allow_redirects=True)
        response.raise_for_status()
        content = response.content
        if len(content) > max_bytes:
            return "download_limit", None, response.url
        return "resolved", content, response.url
    except requests.RequestException:
        return "access_blocked", None, None


def _geo_metadata(session: requests.Session, accession: str) -> dict[str, Any]:
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    status, payload, final_url = _request(
        session,
        url,
        params={"acc": accession, "targ": "self", "form": "xml", "view": "quick"},
        max_bytes=16 * 1024 * 1024,
    )
    result: dict[str, Any] = {
        "status": status,
        "url": final_url or f"{url}?acc={accession}&targ=self&form=xml&view=quick",
        "title": None,
        "pubmed_id": None,
        "supplementary_urls": [],
        "relations": [],
        "summary": None,
        "overall_design": None,
    }
    if payload is None:
        return result
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        result["status"] = "http_error"
        return result
    series = next((element for element in _find_all(root, "Series") if element.attrib.get("iid") == accession), None)
    if series is None:
        result["status"] = "not_found"
        return result
    result["title"] = _text(next((child for child in series if _namespace(child.tag) == "Title"), None))
    result["pubmed_id"] = _text(next((child for child in series if _namespace(child.tag) == "Pubmed-ID"), None))
    result["summary"] = _text(next((child for child in series if _namespace(child.tag) == "Summary"), None))
    result["overall_design"] = _text(next((child for child in series if _namespace(child.tag) == "Overall-Design"), None))
    result["supplementary_urls"] = [
        value.strip()
        for element in series
        if _namespace(element.tag) == "Supplementary-Data"
        for value in [_text(element)]
        if value
    ]
    result["relations"] = [
        {"type": element.attrib.get("type", ""), "target": element.attrib.get("target", "")}
        for element in series
        if _namespace(element.tag) == "Relation" and element.attrib.get("target")
    ]
    return result


def _europe_search(session: requests.Session, query: str, *, page_size: int = 5) -> list[dict[str, Any]]:
    status, payload, _ = _request(
        session,
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": query, "format": "json", "pageSize": str(page_size)},
        max_bytes=2 * 1024 * 1024,
    )
    if status != "resolved" or payload is None:
        return []
    try:
        body = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    return [dict(item) for item in body.get("resultList", {}).get("result", []) if isinstance(item, dict)]


def _publication_metadata(session: requests.Session, pmid: str | None, seed_title: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    resolution = "unresolved"
    if pmid and _PMID_RE.fullmatch(pmid):
        candidates = _europe_search(session, f"EXT_ID:{pmid}", page_size=3)
        if candidates:
            resolution = "resolved_from_geo"
    if not candidates:
        candidates = _europe_search(session, f'TITLE:"{seed_title}"', page_size=5)
        if candidates:
            scored = sorted(candidates, key=lambda item: _title_match_score(seed_title, str(item.get("title", ""))), reverse=True)
            if _title_match_score(seed_title, str(scored[0].get("title", ""))) >= 0.55:
                candidates = [scored[0]]
                resolution = "resolved_by_title_search"
            else:
                candidates = []
    if not candidates:
        return {"resolution": resolution, "metadata": None}
    return {"resolution": resolution, "metadata": candidates[0]}


def _full_text(session: requests.Session, pmcid: str | None) -> dict[str, Any]:
    if not pmcid:
        return {"status": "not_found", "url": None, "data_availability": None, "links": [], "supplementary": []}
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    status, payload, final_url = _request(session, url, max_bytes=16 * 1024 * 1024)
    result: dict[str, Any] = {"status": status, "url": final_url or url, "data_availability": None, "links": [], "supplementary": []}
    if payload is None:
        return result
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        result["status"] = "http_error"
        return result
    for element in root.iter():
        if _namespace(element.tag) == "sec" and element.attrib.get("sec-type") == "data-availability":
            result["data_availability"] = _text(element)
        href = _get_href(element)
        if href:
            result["links"].append({"url": urllib.parse.urljoin(final_url or url, href), "kind": _namespace(element.tag)})
        if _namespace(element.tag) == "supplementary-material":
            href = _get_href(element)
            if href:
                result["supplementary"].append({"url": urllib.parse.urljoin(final_url or url, href), "kind": "supplementary-material"})
    return result


def _supplementary_inventory(session: requests.Session, pmcid: str | None, max_bytes: int) -> dict[str, Any]:
    if not pmcid:
        return {"status": "not_found", "url": None, "files": [], "downloaded_bytes": 0}
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
    status, payload, final_url = _request(session, url, max_bytes=max_bytes)
    result: dict[str, Any] = {"status": status, "url": final_url or url, "files": [], "downloaded_bytes": len(payload or b"")}
    if payload is None:
        return result
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile:
        result["status"] = "http_error"
        return result
    for info in archive.infolist():
        name = info.filename
        content = archive.read(name) if not info.is_dir() and info.file_size <= 8 * 1024 * 1024 else None
        result["files"].append(
            {
                "name": name,
                "size_bytes": int(info.file_size),
                "format": Path(name).suffix.lower().lstrip(".") or "unknown",
                "content_hash": hashlib.sha256(content).hexdigest() if content is not None else None,
            }
        )
    return result


def _machine_candidate(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(_MACHINE_SUFFIXES) and not any(term in lower for term in _EXCLUDED_DATA_TERMS)


def _is_multi_arm(row: Mapping[str, str]) -> bool:
    expected = str(row.get("expected_therapeutic_arm_count", ""))
    if expected.isdigit():
        return int(expected) >= 2
    return str(row.get("expected_action_mode", "")) in {"multi_arm", "dose_response", "schedule_response"} and expected in {"at_least_2", "2+"}


def _therapeutic_arm_status(row: Mapping[str, str]) -> str:
    action_mode = str(row.get("expected_action_mode", "")).lower()
    if "genetic" in action_mode or "construct" in action_mode:
        return "genetic_or_non_therapeutic"
    if _is_multi_arm(row):
        return "multi_arm_confirmed"
    return "single_action"


def _classify(
    row: Mapping[str, str],
    geo: Mapping[str, Any],
    publication: Mapping[str, Any],
    full_text: Mapping[str, Any],
    supplements: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    metadata = publication.get("metadata") or {}
    full_text_text = str(full_text.get("data_availability") or "")
    files = list(supplements.get("files", []))
    machine_files = [item for item in files if _machine_candidate(str(item.get("name", "")))]
    has_functional_text = bool(full_text_text) or bool(row.get("endpoint_family"))
    multi_arm = _is_multi_arm(row)
    publication_chain_resolved = bool(metadata) or full_text.get("status") == "resolved" or supplements.get("status") == "resolved"
    if publication.get("resolution") == "access_blocked" or (
        geo.get("status") in {"access_blocked", "download_limit", "http_error"} and not publication_chain_resolved
    ):
        return "ACCESS_BLOCKED", {"machine": "access_blocked", "replicate": "access_blocked", "uncertainty": "access_blocked"}
    if machine_files and multi_arm:
        uncertainty_text = full_text_text.lower()
        if any(token in uncertainty_text for token in ("standard error", "sem", "standard deviation", "confidence interval")):
            return "FUNCTIONAL_DATA_RECOVERED", {"machine": "confirmed", "replicate": "unresolved", "uncertainty": "confirmed"}
        return "FUNCTIONAL_DATA_PRESENT_BUT_UNCERTAINTY_UNAVAILABLE", {"machine": "confirmed", "replicate": "unresolved", "uncertainty": "missing"}
    if has_functional_text and not multi_arm:
        return "NO_MULTI_ARM_FUNCTIONAL_DATA", {"machine": "not_found", "replicate": "unresolved", "uncertainty": "unresolved"}
    if has_functional_text and files:
        return "FUNCTIONAL_DATA_PRESENT_BUT_NON_MACHINE_READABLE", {"machine": "present_but_not_machine_readable", "replicate": "unresolved", "uncertainty": "unresolved"}
    if metadata or full_text.get("status") == "resolved" or geo.get("status") in {"resolved", "not_found"}:
        return "NO_PUBLIC_FUNCTIONAL_ARTIFACT", {"machine": "not_found", "replicate": "unresolved", "uncertainty": "unresolved"}
    return "ACCESS_BLOCKED", {"machine": "access_blocked", "replicate": "access_blocked", "uncertainty": "access_blocked"}


def _node(node_uid: str, node_type: str, relationship: str, url: str, *, identifier: str | None = None, title: str | None = None, artifact_format: str | None = None, artifact_size_bytes: int | None = None, machine_readable_status: str | None = None, access_status: str = "resolved", content_hash: str | None = None, metadata_source: str) -> dict[str, Any]:
    return {
        "node_uid": node_uid,
        "node_type": node_type,
        "relationship": relationship,
        "identifier": identifier,
        "title": title,
        "url": url,
        "artifact_format": artifact_format,
        "artifact_size_bytes": artifact_size_bytes,
        "machine_readable_status": machine_readable_status,
        "access_status": access_status,
        "content_hash": content_hash,
        "metadata_source": metadata_source,
    }


def build(*, session: requests.Session | None = None) -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config["functional_therapeutic_evidence"]
    recovery = policy["source_discovery"]["publication_recovery"]
    rows = _read_tsv(SOURCE_METADATA_PATH)
    if len(rows) != int(recovery["max_seed_sources"]):
        raise ValueError(f"Expected exactly {recovery['max_seed_sources']} seed rows, got {len(rows)}")
    client = session or requests.Session()
    client.headers.update({"User-Agent": "NeuronSifter-functional-publication-recovery/1.0 (research metadata)"})
    records: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    supplementary_budget = int(recovery["max_total_supplementary_bytes"])
    for row in rows:
        accession = row["study_accession"]
        geo = _geo_metadata(client, accession)
        seed_pmid = geo.get("pubmed_id") or next((value.split(":", 1)[1] for value in str(row.get("publication_uid", "")).split(";") if value.startswith("pubmed:")), None)
        publication = _publication_metadata(client, seed_pmid, row["title"])
        metadata = publication.get("metadata") or {}
        pmid = str(metadata.get("pmid") or seed_pmid or "") or None
        pmcid = str(metadata.get("pmcid") or "") or None
        doi = str(metadata.get("doi") or "") or None
        full_text = _full_text(client, pmcid)
        per_source_budget = min(int(recovery["max_supplementary_bytes_per_source"]), supplementary_budget)
        supplements = _supplementary_inventory(client, pmcid, per_source_budget) if per_source_budget > 0 else {"status": "download_limit", "url": None, "files": [], "downloaded_bytes": 0}
        supplementary_budget = max(0, supplementary_budget - int(supplements.get("downloaded_bytes", 0)))
        status, status_fields = _classify(row, geo, publication, full_text, supplements)
        nodes: list[dict[str, Any]] = [
            _node(
                f"{row['source_uid']}:accession",
                "source_accession",
                "seed_source",
                row["public_url"],
                identifier=accession,
                title=geo.get("title") or row["title"],
                access_status=str(geo.get("status", "access_blocked")),
                metadata_source="NCBI GEO MINiML",
            )
        ]
        for index, relation in enumerate(geo.get("relations", [])):
            target = str(relation.get("target", "")).strip()
            relation_type = str(relation.get("type", "")).strip() or "related_resource"
            if not target:
                continue
            identifier = target.rstrip("/").rsplit("/", 1)[-1] or None
            nodes.append(
                _node(
                    f"{row['source_uid']}:relation:{index}",
                    "linked_repository",
                    f"geo_relation_{relation_type}",
                    target,
                    identifier=identifier,
                    access_status="resolved",
                    metadata_source="NCBI GEO MINiML",
                )
            )
        if pmid or doi or pmcid:
            publication_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (f"https://doi.org/{doi}" if doi else f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/")
            nodes.append(_node(f"{row['source_uid']}:publication", "publication", "associated_publication", publication_url, identifier=publication.get("metadata", {}).get("id") if publication.get("metadata") else pmid, title=metadata.get("title"), access_status="resolved", metadata_source="Europe PMC"))
        if full_text.get("status") == "resolved":
            nodes.append(_node(f"{row['source_uid']}:full_text", "full_text", "publication_full_text", str(full_text["url"]), identifier=pmcid, title=metadata.get("title"), access_status="resolved", metadata_source="Europe PMC fullTextXML"))
        for index, item in enumerate(supplements.get("files", [])):
            name = str(item["name"])
            nodes.append(_node(f"{row['source_uid']}:supplement:{index}", "supplementary_artifact", "publication_supplementary_file", f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles#{urllib.parse.quote(name)}", identifier=name, title=None, artifact_format=item.get("format"), artifact_size_bytes=item.get("size_bytes"), machine_readable_status="confirmed" if _machine_candidate(name) else "present_but_not_machine_readable", access_status="resolved", content_hash=item.get("content_hash"), metadata_source="Europe PMC supplementaryFiles"))
        for index, link in enumerate(full_text.get("links", [])[: int(recovery["max_linked_artifacts_per_source"])]):
            url = str(link["url"])
            if url.endswith("/fullTextXML") or "doi.org" in url or "pubmed" in url:
                continue
            nodes.append(_node(f"{row['source_uid']}:link:{index}", "linked_repository", "explicit_full_text_link", url, access_status="resolved", metadata_source="Europe PMC fullTextXML"))
        record = {
            "source_uid": row["source_uid"],
            "study_accession": accession,
            "status": status,
            "publication_resolution": publication.get("resolution", "unresolved"),
            "publication_uid": f"pubmed:{pmid}" if pmid else (f"doi:{doi}" if doi else None),
            "doi": doi,
            "pmcid": pmcid,
            "functional_artifact_status": status,
            "machine_readable_status": status_fields["machine"],
            "replicate_identity_status": status_fields["replicate"],
            "uncertainty_status": status_fields["uncertainty"],
            "shared_control_status": "stated_not_verified" if row.get("common_control_status") else "unresolved",
            "therapeutic_arm_status": _therapeutic_arm_status(row),
            "data_availability": full_text.get("data_availability"),
            "nodes": nodes,
        }
        records.append(record)
        for item in nodes:
            graph_rows.append(
                {
                    "source_uid": row["source_uid"],
                    "study_accession": accession,
                    "node_uid": item["node_uid"],
                    "node_type": item["node_type"],
                    "relationship": item["relationship"],
                    "identifier": item.get("identifier") or "",
                    "title": item.get("title") or "",
                    "url": item["url"],
                    "publication_uid": record["publication_uid"] or "",
                    "doi": doi or "",
                    "pmcid": pmcid or "",
                    "endpoint_family": row.get("endpoint_family", ""),
                    "objective_uid": row.get("objective_uid", ""),
                    "functional_artifact_status": status,
                    "machine_readable_status": item.get("machine_readable_status") or record["machine_readable_status"],
                    "replicate_identity_status": record["replicate_identity_status"],
                    "uncertainty_status": record["uncertainty_status"],
                    "shared_control_status": record["shared_control_status"],
                    "therapeutic_arm_status": record["therapeutic_arm_status"],
                    "artifact_format": item.get("artifact_format") or "",
                    "artifact_size_bytes": item.get("artifact_size_bytes") if item.get("artifact_size_bytes") is not None else "",
                    "access_status": item["access_status"],
                    "content_hash": item.get("content_hash") or "",
                    "metadata_source": item["metadata_source"],
                    "retrieved_at": _now(),
                }
            )
    statuses = {record["status"] for record in records}
    node_access_statuses = {node["access_status"] for record in records for node in record["nodes"]}
    status = (
        "completed_with_access_blocks"
        if "ACCESS_BLOCKED" in statuses or node_access_statuses & {"access_blocked", "download_limit", "http_error"}
        else "completed_bounded_recovery"
    )
    graph = {
        "schema_version": "neuronsifter-functional-publication-evidence-graph-v1",
        "status": status,
        "terminal_state": "BLOCKED_REQUIRES_EXTERNAL_MULTI_ARM_FUNCTIONAL_OUTCOMES",
        "campaign_scope": "nine_seed_publication_centered_recovery",
        "source_count": len(records),
        "functional_artifact_present_count": sum(
            record["status"]
            in {
                "FUNCTIONAL_DATA_RECOVERED",
                "FUNCTIONAL_DATA_PRESENT_BUT_NON_MACHINE_READABLE",
                "FUNCTIONAL_DATA_PRESENT_BUT_INSUFFICIENT_REPLICATES",
                "FUNCTIONAL_DATA_PRESENT_BUT_UNCERTAINTY_UNAVAILABLE",
            }
            for record in records
        ),
        "publication_resolved_count": sum(record["publication_uid"] is not None for record in records),
        "functional_artifact_recovered_count": sum(record["status"] == "FUNCTIONAL_DATA_RECOVERED" for record in records),
        "machine_readable_candidate_count": sum(record["machine_readable_status"] == "confirmed" for record in records),
        "admission_eligible_source_count": 0,
        "legal_candidate_set_count": 0,
        "qualified_candidate_set_count": 0,
        "outcome_values_used_for_priority": False,
        "model_predictions_inspected": False,
        "figure_digitization_admitted": False,
        "protected_holdout_access": 0,
        "publication_comparison_open": False,
        "records": records,
    }
    schema = json.loads(GRAPH_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(graph))
    if errors:
        raise ValueError("Publication evidence graph schema errors: " + "; ".join(error.message for error in errors))
    fields = list(graph_rows[0]) if graph_rows else ["source_uid", "study_accession", "node_uid"]
    write_tsv(GRAPH_PATH, fields, graph_rows)
    write_json(AUDIT_PATH, {**graph, "created_at": _now(), "next_route": "prepare_external_multi_arm_functional_outcome_intake"})
    return graph


def refresh_existing() -> dict[str, Any]:
    """Recompute deterministic semantic fields without network access."""
    existing = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    rows = _read_tsv(SOURCE_METADATA_PATH)
    source_rows = {row["source_uid"]: row for row in rows}
    records = existing.get("records", [])
    for record in records:
        source_uid = record["source_uid"]
        record["therapeutic_arm_status"] = _therapeutic_arm_status(source_rows[source_uid])
    status = (
        "completed_with_access_blocks"
        if any(
            record.get("status") == "ACCESS_BLOCKED"
            or any(node.get("access_status") in {"access_blocked", "download_limit", "http_error"} for node in record.get("nodes", []))
            for record in records
        )
        else "completed_bounded_recovery"
    )
    graph = {
        key: existing[key]
        for key in (
            "schema_version",
            "campaign_scope",
            "outcome_values_used_for_priority",
            "model_predictions_inspected",
            "figure_digitization_admitted",
            "protected_holdout_access",
            "publication_comparison_open",
        )
        if key in existing
    }
    graph.update(
        {
            "status": status,
            "terminal_state": existing.get("terminal_state", "BLOCKED_REQUIRES_EXTERNAL_MULTI_ARM_FUNCTIONAL_OUTCOMES"),
            "source_count": len(records),
            "functional_artifact_present_count": sum(record.get("status") in {
                "FUNCTIONAL_DATA_RECOVERED",
                "FUNCTIONAL_DATA_PRESENT_BUT_NON_MACHINE_READABLE",
                "FUNCTIONAL_DATA_PRESENT_BUT_INSUFFICIENT_REPLICATES",
                "FUNCTIONAL_DATA_PRESENT_BUT_UNCERTAINTY_UNAVAILABLE",
            } for record in records),
            "publication_resolved_count": sum(record.get("publication_uid") is not None for record in records),
            "functional_artifact_recovered_count": sum(record.get("status") == "FUNCTIONAL_DATA_RECOVERED" for record in records),
            "machine_readable_candidate_count": sum(record.get("machine_readable_status") == "confirmed" for record in records),
            "admission_eligible_source_count": 0,
            "legal_candidate_set_count": 0,
            "qualified_candidate_set_count": 0,
            "records": records,
        }
    )
    schema = json.loads(GRAPH_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(graph))
    if errors:
        raise ValueError("Refreshed publication evidence graph schema errors: " + "; ".join(error.message for error in errors))
    graph_rows = _read_tsv(GRAPH_PATH)
    for row in graph_rows:
        source_uid = row["source_uid"]
        row["therapeutic_arm_status"] = _therapeutic_arm_status(source_rows[source_uid])
    fields = list(graph_rows[0]) if graph_rows else ["source_uid", "study_accession", "node_uid"]
    write_tsv(GRAPH_PATH, fields, graph_rows)
    write_json(
        AUDIT_PATH,
        {
            **graph,
            "created_at": existing.get("created_at", _now()),
            "next_route": "prepare_external_multi_arm_functional_outcome_intake",
        },
    )
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-existing", action="store_true", help="Refresh deterministic fields in the existing graph without network access")
    args = parser.parse_args()
    result = refresh_existing() if args.refresh_existing else build()
    print(json.dumps({key: result[key] for key in ("status", "source_count", "publication_resolved_count", "functional_artifact_recovered_count", "machine_readable_candidate_count", "admission_eligible_source_count", "protected_holdout_access")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
