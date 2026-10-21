"""Frozen mechanistic-belief construct, metadata receipt, and source gate."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
import yaml


CONTRACT_PATH = "Libs/configs/validation/mechanistic_belief_construct_firewall.yaml"
RECEIPT_SCHEMA_PATH = "Libs/configs/schema/mechanistic_belief_cutoff_receipt.schema.json"
AUTHORITY_SCHEMA_PATH = "Libs/configs/schema/mechanistic_belief_construct_authority.schema.json"
RECEIPT_PATH = "Output/audits/mechanistic_belief_cutoff_receipt.json"
AUTHORITY_PATH = "Output/audits/mechanistic_belief_construct_authority.json"

EXPECTED_CONTRACT_SHA256 = "f450ed2e1be9f09fdc1b8e7bc5066b2c68a5307c4af7b4778d5686f36cd3e374"
EXPECTED_RECEIPT_SCHEMA_SHA256 = "0c6e30c56c72b53b1d13db622faf7e458d8fa4e211edeb56c370c9d2b8388555"
EXPECTED_AUTHORITY_SCHEMA_SHA256 = "93edc9324dca165820a6d40475f95ba65e332242f63b07f345ad8b0856be6c70"

CONTRACT_SCHEMA_VERSION = "neuronsifter-mechanistic-belief-construct-firewall-contract-v1"
RECEIPT_SCHEMA_VERSION = "neuronsifter-mechanistic-belief-cutoff-receipt-v1"
AUTHORITY_SCHEMA_VERSION = "neuronsifter-mechanistic-belief-construct-authority-v1"
METHOD_UID = "mechanistic_belief_twin_v1"
STARTING_BRANCH = "main"
STARTING_COMMIT = "9fbc6afc74a527dbbe2b086d08bccc64c85f66bf"
EARLIEST_DATE = "2021-03-03"
CUTOFF = "2021-03-02T23:59:59Z"
GATE_ACCESS_KIND = "mechanism_source_content"

METADATA_PROJECTION_FIELDS = (
    "publication_identity",
    "identifier_type",
    "identifier_value",
    "official_date",
    "official_metadata_authority",
    "requested_url",
    "final_url",
    "retrieved_at",
    "http_status",
    "raw_response_sha256",
    "raw_response_bytes",
    "projected_metadata_sha256",
    "validation_status",
)
FORBIDDEN_PERSISTED_FIELDS = (
    "title",
    "abstract",
    "result",
    "results",
    "result_prose",
    "figure",
    "figures",
    "outcome",
    "outcomes",
    "rank",
    "ranking",
    "label",
    "labels",
    "citation_context",
    "references",
    "raw_body",
    "raw_response",
)
ALLOWED_HOST_SUFFIXES = (
    "ncbi.nlm.nih.gov",
    "elifesciences.org",
    "nature.com",
    "crossref.org",
)

STATE_FACTORS = [
    "sample_treatment_action",
    "exposure_occupancy",
    "cns_microenvironment",
    "electrophysiology",
    "structural_morphology",
]
WORLD_ACTION_FIELDS = ["target_set", "agent_set", "dose", "schedule", "route", "context"]
PRIOR_ROLES = {
    "required": ["canonical_mechanism_prior"],
    "optional_sensitivity": [
        "conservative_support_sensitivity",
        "broad_parameter_sensitivity",
    ],
    "maximum_sensitivity_roles": 2,
    "numerical_priors_populated": False,
    "preferred_sensitivity_role": None,
}
DIAGNOSTIC_FORMS = ["static", "point_estimate", "forced_decision", "full_belief_twin"]
EVIDENCE_GRADES = [
    "DIRECT_MECHANISM",
    "PARAMETER_ANCHOR",
    "BOUNDARY_CONSTRAINT",
    "CONTEXT_ONLY",
    "INADMISSIBLE",
]
SUPPORT_GRADES = ["DIRECT_MECHANISM", "PARAMETER_ANCHOR", "BOUNDARY_CONSTRAINT"]
QUEUE_KEYS = [
    "canonical_source_identity",
    "source_family_identity",
    "intervention_identity",
    "official_release_or_publication_date",
    "source_type",
    "mechanism_relevance_tags",
    "queue_order",
]
TERMINALS = [
    "PROSPECTIVE_MECHANISTIC_BELIEF_TWIN_READY_FOR_EVALUATION",
    "MECHANISM_BASIS_INSUFFICIENT",
    "MECHANISM_COMPILER_NOT_GENERALIZABLE",
    "PRIOR_PREDICTIVE_TWIN_NOT_SCIENTIFICALLY_IDENTIFIABLE",
    "NO_FIT_UNCERTAINTY_NOT_DEFENSIBLE",
    "ACTION_DOMAIN_COVERAGE_INSUFFICIENT",
]
ACCESS_COUNTERS = {
    "six_context_outcome_access": 0,
    "protected_holdout_access": 0,
    "source_content_access": 0,
}
DOWNSTREAM_AUTHORIZATION = {
    "source_queue_creation_authorized": False,
    "source_content_intake_authorized": False,
    "prior_distribution_population_authorized": False,
    "reliability_policy_implementation_authorized": False,
    "world_action_compiler_authorized": False,
    "runtime_implementation_authorized": False,
    "inference_authorized": False,
    "training_authorized": False,
    "gpu_authorized": False,
    "formal_evaluation_authorized": False,
    "six_context_evaluation_authorized": False,
    "protected_holdout_authorized": False,
}
ARTIFACTS_CREATED = {
    "source_queue": False,
    "source_content": False,
    "numerical_prior_distribution": False,
    "reliability_policy": False,
    "world_action_compiler": False,
    "belief_runtime": False,
    "scientific_inference": False,
}
LIMITATIONS = [
    "The authority freezes construct semantics and evidence access only; it contains no mechanism-source content.",
    "Raw official metadata responses are hashed in memory and are not persisted, logged, or reproducible from this receipt alone.",
    "No source queue, numerical prior, reliability policy, compiler, runtime, inference, training, or evaluation was created.",
]

PUBLICATIONS = [
    ("pubmed:42334452", "pubmed", "42334452", "2026-06-23", "NCBI PubMed", "pubmed_esummary_sortpubdate"),
    ("pubmed:42028031", "pubmed", "42028031", "2026-03-30", "NCBI PubMed", "pubmed_esummary_sortpubdate"),
    ("doi:10.7554/eLife.69795", "doi", "10.7554/eLife.69795", "2021-09-22", "eLife official publisher/DOI metadata", "elife_article_published"),
    ("doi:10.1038/s43587-026-01166-3", "doi", "10.1038/s43587-026-01166-3", "2026-07-03", "Nature Aging official publisher/DOI metadata", "crossref_published_date"),
    ("pubmed:40140393", "pubmed", "40140393", "2025-03-26", "NCBI PubMed", "pubmed_esummary_sortpubdate"),
    ("pubmed:33658519", "pubmed", "33658519", "2021-03-03", "NCBI PubMed", "pubmed_esummary_sortpubdate"),
]

HISTORICAL_AUTHORITIES = [
    ("Output/audits/no_task_fit_parent_authority.json", "56eea3c26d292d2c2e5575cf89021bdeea4055bd6c66dfadcbcb3c1a3e86f554", "parent_negative_authorities_bound_with_zero_access"),
    ("Output/audits/no_task_fit_reusable_asset_lineage.json", "d89e9da8616cfadd7a989a4b9473afe1538f3c70bb71b27d14a754ba777519ff", "twenty_four_asset_structural_lineage_is_historical_only"),
    ("Output/audits/no_task_fit_route_arbitration.json", "27701a356794d7ffb79cb20d66a3a2a97f5649327e821716041dca6428a7e7c3", "no_admissible_route_four_routes_zero_admissible"),
    ("Output/audits/no_task_fit_candidate_registry.json", "f84de1577eae1d80ed0016fe5ea9a4cd28882e83133488c978f8abbf35562738", "zero_candidates_and_no_preferred_candidate"),
    ("orchestrator/roadmaps/2026-07-13-00-iclr-2027-design-contract/rev-007/roadmap.md", "cc4b180eee6a055a9dcb5168d3db66d26a5801601b1e494255da9a8cf005fd92", "rev_007_terminal_no_continuation"),
    ("orchestrator/roadmaps/2026-07-13-00-iclr-2027-design-contract/rev-007/verification.md", "9839c477555f5d993aa1c554ae58535f16896adce635c1aebbdd927317f62ccf", "rev_007_verification_unchanged"),
]

FORBIDDEN_FIELD_NAMES = set(FORBIDDEN_PERSISTED_FIELDS)
PROTECTED_PATH_PARTS = {
    "protected", "holdout", "six_context", "six-context", "evaluation", "outcome",
    "outcomes", "results", "ranking", "labels",
}


class ConstructFirewallError(ValueError):
    """Raised when a frozen authority or access boundary fails closed."""


@dataclass(frozen=True)
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    body: bytes
    redirect_count: int = 0


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: object, code: int, msg: str, headers: Mapping[str, str], newurl: str) -> None:
        return None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def canonical_value_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConstructFirewallError(f"{label} must be an object with string keys")
    return value


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConstructFirewallError(f"{label} must be an array")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "*?[]\x00"):
        raise ConstructFirewallError(f"{label} must be a normalized relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or "." in pure.parts or ".." in pure.parts or str(pure) != value:
        raise ConstructFirewallError(f"{label} must be a normalized relative path")
    return value


def _resolve_file(root: Path, relative: str, *, must_exist: bool, label: str) -> Path:
    normalized = _relative_path(relative, label=label)
    root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    current = root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise ConstructFirewallError(f"{label} traverses a symlink")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise ConstructFirewallError(f"missing {label}") from exc
    if not resolved.is_relative_to(root):
        raise ConstructFirewallError(f"{label} escapes declared root")
    if must_exist and not resolved.is_file():
        raise ConstructFirewallError(f"{label} is not a regular file")
    return candidate


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label="contract")
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConstructFirewallError("unable to load frozen contract") from exc


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        return _mapping(json.loads(raw.decode("utf-8")), label=label), raw
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConstructFirewallError(f"unable to load {label}") from exc


def _validate_schema_file(path: Path, expected_hash: str) -> tuple[dict[str, Any], str]:
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ConstructFirewallError("frozen schema identity drift")
    schema, _ = _load_json(path, label="schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ConstructFirewallError("invalid Draft 2020-12 schema") from exc
    return schema, observed


def _expected_publication_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [_mapping(row, label="publication") for row in _sequence(contract.get("publications"), label="publications")]
    observed = [
        (
            row.get("publication_identity"), row.get("identifier_type"), row.get("identifier_value"),
            row.get("official_date"), row.get("official_metadata_authority"), row.get("projection_kind"),
        )
        for row in rows
    ]
    if observed != PUBLICATIONS:
        raise ConstructFirewallError("publication roster drift")
    if len({row[0] for row in observed}) != 6:
        raise ConstructFirewallError("publication identities must be unique")
    return rows


def _expected_construct(contract: Mapping[str, Any]) -> dict[str, Any]:
    construct = _mapping(contract["construct"], label="construct")
    return {
        "state_factors": deepcopy(construct["state_factors"]),
        "state_semantics": construct["state_semantics"],
        "forbidden_state_semantics": deepcopy(construct["forbidden_state_semantics"]),
        "action_input": deepcopy(construct["action_input"]),
        "action_output": deepcopy(construct["action_output"]),
        "mechanism_supports": deepcopy(construct["mechanism_supports"]),
        "mechanism_never_supports": deepcopy(construct["mechanism_never_supports"]),
        "prior_predictive_supports": deepcopy(construct["prior_predictive_supports"]),
        "prior_predictive_forbidden": deepcopy(construct["prior_predictive_forbidden"]),
        "prior_roles": deepcopy(contract["prior_roles"]),
        "diagnostic_forms": deepcopy(contract["diagnostic_forms"]),
        "evidence_policy": {
            "grades": deepcopy(contract["evidence_policy"]["grades"]),
            "structure_or_prior_support_grades": deepcopy(contract["evidence_policy"]["structure_or_prior_support_grades"]),
        },
        "source_limits": deepcopy(contract["source_limits"]),
        "terminal_vocabulary": deepcopy(contract["terminal_vocabulary"]),
        "access_policy": deepcopy(contract["access_policy"]),
        "non_goals": deepcopy(contract["non_goals"]),
    }


def _expected_retrieval_policy() -> dict[str, Any]:
    return {
        "metadata_source": "official",
        "exact_request_count": 6,
        "maximum_concurrent_requests": 1,
        "request_timeout_seconds": 30,
        "maximum_response_bytes": 2097152,
        "maximum_total_response_bytes": 12582912,
        "maximum_redirects": 3,
        "allowed_host_suffixes": list(ALLOWED_HOST_SUFFIXES),
        "discovery_queries_allowed": False,
        "raw_response_persistence_allowed": False,
        "response_value_logging_allowed": False,
    }


def _expected_receipt_access_policy() -> dict[str, Any]:
    return {
        "metadata_projection_fields": list(METADATA_PROJECTION_FIELDS),
        "forbidden_persisted_fields": list(FORBIDDEN_PERSISTED_FIELDS),
        "gate_access_kind": GATE_ACCESS_KIND,
        "metadata_retrieval_authorizes_source_content": False,
    }


def _validate_frozen_metadata_policy(
    contract: Mapping[str, Any], receipt_access_policy: object | None = None
) -> None:
    evidence_policy = _mapping(
        contract.get("evidence_policy"), label="evidence policy"
    )
    if evidence_policy.get("metadata_projection_fields") != list(
        METADATA_PROJECTION_FIELDS
    ):
        raise ConstructFirewallError("metadata projection allowlist drift")
    if evidence_policy.get("forbidden_persisted_fields") != list(
        FORBIDDEN_PERSISTED_FIELDS
    ):
        raise ConstructFirewallError("forbidden metadata field denylist drift")
    if contract.get("retrieval_policy") != _expected_retrieval_policy():
        raise ConstructFirewallError("official metadata host or retrieval policy drift")
    if receipt_access_policy is not None:
        access_policy = _mapping(
            receipt_access_policy, label="receipt metadata access policy"
        )
        if access_policy != _expected_receipt_access_policy():
            raise ConstructFirewallError("receipt metadata access policy drift")


def validate_contract(repository_root: Path, contract_path: Path | str = CONTRACT_PATH) -> tuple[dict[str, Any], bytes]:
    root = repository_root.resolve(strict=True)
    path = Path(contract_path)
    if path.is_absolute():
        try:
            relative = path.resolve(strict=True).relative_to(root).as_posix()
        except ValueError as exc:
            raise ConstructFirewallError("contract escapes repository root") from exc
    else:
        relative = _relative_path(path.as_posix(), label="contract path")
    if relative != CONTRACT_PATH:
        raise ConstructFirewallError(f"contract must be {CONTRACT_PATH}")
    resolved = _resolve_file(root, relative, must_exist=True, label="contract")
    raw = resolved.read_bytes()
    if sha256_bytes(raw) != EXPECTED_CONTRACT_SHA256:
        raise ConstructFirewallError("frozen contract identity drift")
    contract = _load_yaml(resolved)
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION or contract.get("status") != "frozen_before_mechanism_source_content":
        raise ConstructFirewallError("contract version or status drift")
    if contract.get("method") != {"uid": METHOD_UID, "candidate_search_allowed": False, "preferred_method_allowed": False}:
        raise ConstructFirewallError("canonical method identity drift")
    if contract.get("prior_roles") != PRIOR_ROLES or contract.get("diagnostic_forms") != DIAGNOSTIC_FORMS:
        raise ConstructFirewallError("prior roles or diagnostic forms drift")
    evidence_policy = _mapping(contract.get("evidence_policy"), label="evidence policy")
    if evidence_policy.get("grades") != EVIDENCE_GRADES or evidence_policy.get("structure_or_prior_support_grades") != SUPPORT_GRADES:
        raise ConstructFirewallError("evidence grade contract drift")
    _validate_frozen_metadata_policy(contract)
    limits = contract.get("source_limits")
    if not isinstance(limits, dict) or limits != {
        "maximum_source_families": 6,
        "maximum_total_sources": 160,
        "maximum_sources_per_intervention": 12,
        "future_metadata_only_queue_keys": QUEUE_KEYS,
        "queue_rows_created": 0,
    }:
        raise ConstructFirewallError("source limits or queue keys drift")
    if contract.get("terminal_vocabulary") != TERMINALS or contract.get("access_policy") != {"gate_access_kind": GATE_ACCESS_KIND, "metadata_retrieval_authorizes_source_content": False, **ACCESS_COUNTERS}:
        raise ConstructFirewallError("terminal or access policy drift")
    if contract.get("starting_authority") != {"branch": STARTING_BRANCH, "commit": STARTING_COMMIT}:
        raise ConstructFirewallError("starting authority drift")
    if contract.get("downstream_authorization") != {"mechanism_source_content_read_performed": False, **DOWNSTREAM_AUTHORIZATION}:
        raise ConstructFirewallError("downstream authorization drift")
    expected_history = [{"path": path, "sha256": digest, "required_meaning": meaning} for path, digest, meaning in HISTORICAL_AUTHORITIES]
    if contract.get("historical_authorities") != expected_history:
        raise ConstructFirewallError("historical authority roster drift")
    _expected_publication_records(contract)
    cutoff = contract.get("cutoff")
    if cutoff != {"derivation_rule": "one_second_before_earliest_official_publication_date_utc", "earliest_official_date": EARLIEST_DATE, "strictly_before_cutoff": CUTOFF}:
        raise ConstructFirewallError("cutoff contract drift")
    if construct := _expected_construct(contract):
        if construct["state_factors"] != STATE_FACTORS or construct["action_input"] != {"type": "WorldAction", "fields": WORLD_ACTION_FIELDS}:
            raise ConstructFirewallError("construct state or action boundary drift")
    return contract, raw


def validate_historical_authorities(repository_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for relative, expected_hash, meaning in HISTORICAL_AUTHORITIES:
        path = _resolve_file(repository_root, relative, must_exist=True, label="historical authority")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ConstructFirewallError(
                f"historical authority identity drift: {relative} ({observed_hash})"
            )
        records.append({"path": relative, "sha256": expected_hash, "required_meaning": meaning})
    return records


def _host_allowed(url: str, suffixes: Sequence[str]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and bool(host) and any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


def fetch_official_metadata(publication: Mapping[str, Any], policy: Mapping[str, Any]) -> FetchResult:
    requested_url = str(publication["requested_url"])
    suffixes = [str(value) for value in policy["allowed_host_suffixes"]]
    if not _host_allowed(requested_url, suffixes):
        raise ConstructFirewallError("official metadata host is not allowlisted")
    current_url = requested_url
    opener = build_opener(_NoRedirect)
    max_redirects = int(policy["maximum_redirects"])
    for redirect_count in range(max_redirects + 1):
        request = Request(current_url, headers={"User-Agent": "NeuronSifter-MechanisticBelief-Metadata/1.0", "Accept": "application/json,text/html;q=0.8,*/*;q=0.1"})
        try:
            with opener.open(request, timeout=int(policy["request_timeout_seconds"])) as response:
                body = response.read(int(policy["maximum_response_bytes"]) + 1)
                if len(body) > int(policy["maximum_response_bytes"]):
                    raise ConstructFirewallError("official metadata response exceeds byte limit")
                final_url = response.geturl()
                if not _host_allowed(final_url, suffixes):
                    raise ConstructFirewallError("official metadata final host is not allowlisted")
                return FetchResult(requested_url, final_url, int(response.status), body, redirect_count)
        except HTTPError as exc:
            location = exc.headers.get("Location") if exc.headers else None
            if exc.code in {301, 302, 303, 307, 308} and location:
                if redirect_count >= max_redirects:
                    raise ConstructFirewallError("official metadata redirect limit exceeded") from exc
                next_url = urljoin(current_url, location)
                if not _host_allowed(next_url, suffixes):
                    raise ConstructFirewallError("official metadata redirect host is not allowlisted") from exc
                current_url = next_url
                continue
            raise ConstructFirewallError("official metadata request failed") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ConstructFirewallError("official metadata transport failed") from exc
    raise ConstructFirewallError("official metadata redirect limit exceeded")


def _parse_official_date(publication: Mapping[str, Any], body: bytes) -> str:
    kind = publication["projection_kind"]
    try:
        text = body.decode("utf-8")
    except UnicodeError as exc:
        raise ConstructFirewallError("official metadata is not valid UTF-8") from exc
    try:
        if kind == "pubmed_esummary_sortpubdate":
            payload = _mapping(json.loads(text), label="official metadata")
            result = _mapping(payload.get("result"), label="PubMed result")
            uid = str(publication["identifier_value"])
            row = _mapping(result.get(uid), label="PubMed record")
            if str(row.get("uid")) != uid:
                raise ConstructFirewallError("official metadata identifier mismatch")
            value = str(row.get("sortpubdate", ""))
            match = re.fullmatch(r"(\d{4})/(\d{2})/(\d{2})(?: \d{2}:\d{2})?", value)
            if not match:
                raise ConstructFirewallError("official metadata date is unavailable or ambiguous")
            return "-".join(match.groups())
        if kind == "elife_article_published":
            payload = _mapping(json.loads(text), label="official metadata")
            if str(payload.get("id")) != "69795":
                raise ConstructFirewallError("official metadata identifier mismatch")
            value = str(payload.get("published", ""))
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T.*)?", value):
                raise ConstructFirewallError("official metadata date is unavailable or ambiguous")
            return value[:10]
        if kind == "crossref_published_date":
            payload = _mapping(json.loads(text), label="official metadata")
            message = _mapping(payload.get("message"), label="Crossref DOI metadata")
            if str(message.get("DOI", "")).lower() != str(publication["identifier_value"]).lower():
                raise ConstructFirewallError("official metadata identifier mismatch")
            published = _mapping(message.get("published"), label="Crossref published date")
            date_parts = _sequence(published.get("date-parts"), label="Crossref date parts")
            if len(date_parts) != 1 or not isinstance(date_parts[0], list) or len(date_parts[0]) != 3:
                raise ConstructFirewallError("official metadata date is unavailable or ambiguous")
            year, month, day = date_parts[0]
            if not all(isinstance(value, int) for value in (year, month, day)):
                raise ConstructFirewallError("official metadata date is unavailable or ambiguous")
            return f"{year:04d}-{month:02d}-{day:02d}"
    except json.JSONDecodeError as exc:
        raise ConstructFirewallError("official metadata response is malformed") from exc
    raise ConstructFirewallError("unknown official metadata projection kind")


def project_metadata_record(
    publication: Mapping[str, Any],
    response: FetchResult,
    *,
    retrieved_at: str,
    allowed_host_suffixes: Sequence[str],
    maximum_response_bytes: int,
    maximum_redirects: int,
) -> dict[str, Any]:
    if response.requested_url != publication["requested_url"] or response.status_code != 200:
        raise ConstructFirewallError("official metadata request identity or status mismatch")
    if len(response.body) > maximum_response_bytes:
        raise ConstructFirewallError("official metadata response exceeds byte limit")
    if response.redirect_count < 0 or response.redirect_count > maximum_redirects:
        raise ConstructFirewallError("official metadata redirect limit exceeded")
    if not _host_allowed(response.final_url, allowed_host_suffixes):
        raise ConstructFirewallError("official metadata final host is not allowlisted")
    observed_date = _parse_official_date(publication, response.body)
    if observed_date != publication["official_date"]:
        raise ConstructFirewallError("official metadata date drift")
    projection = {
        "publication_identity": publication["publication_identity"],
        "identifier_type": publication["identifier_type"],
        "identifier_value": publication["identifier_value"],
        "official_date": observed_date,
        "official_metadata_authority": publication["official_metadata_authority"],
        "requested_url": publication["requested_url"],
        "final_url": response.final_url,
    }
    return {
        **projection,
        "retrieved_at": retrieved_at,
        "http_status": 200,
        "raw_response_sha256": sha256_bytes(response.body),
        "raw_response_bytes": len(response.body),
        "projected_metadata_sha256": canonical_value_sha256(projection),
        "validation_status": "validated_exact_official_metadata",
    }


def _derive_cutoff(records: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    dates = [date.fromisoformat(str(record["official_date"])) for record in records]
    if len(dates) != 6:
        raise ConstructFirewallError("exactly six publication dates are required")
    earliest = min(dates)
    cutoff = datetime.combine(earliest, datetime.min.time(), tzinfo=UTC) - timedelta(seconds=1)
    return earliest.isoformat(), cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_no_forbidden_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in FORBIDDEN_FIELD_NAMES:
                raise ConstructFirewallError("forbidden persisted metadata field")
            _validate_no_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_no_forbidden_fields(nested)


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    semantic = dict(payload)
    semantic.pop(field, None)
    return canonical_value_sha256(semantic)


def _validate_json_schema(payload: Mapping[str, Any], schema: Mapping[str, Any], *, label: str) -> None:
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        raise ConstructFirewallError(f"{label} schema validation failed")


def validate_cutoff_receipt(*, repository_root: Path, contract_path: Path | str = CONTRACT_PATH, receipt_path: Path | str = RECEIPT_PATH) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    contract, contract_raw = validate_contract(root, contract_path)
    schema_path = _resolve_file(root, RECEIPT_SCHEMA_PATH, must_exist=True, label="receipt schema")
    schema, schema_hash = _validate_schema_file(schema_path, EXPECTED_RECEIPT_SCHEMA_SHA256)
    relative = Path(receipt_path).as_posix() if not Path(receipt_path).is_absolute() else Path(receipt_path).resolve(strict=True).relative_to(root).as_posix()
    if relative != RECEIPT_PATH:
        raise ConstructFirewallError(f"receipt must be {RECEIPT_PATH}")
    receipt, receipt_raw = _load_json(_resolve_file(root, relative, must_exist=True, label="cutoff receipt"), label="cutoff receipt")
    if receipt_raw != canonical_json_bytes(receipt):
        raise ConstructFirewallError("cutoff receipt is not canonical JSON")
    _validate_frozen_metadata_policy(contract, receipt.get("access_policy"))
    _validate_json_schema(receipt, schema, label="cutoff receipt")
    _validate_no_forbidden_fields(receipt)
    if receipt["contract"] != {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)}:
        raise ConstructFirewallError("receipt contract identity drift")
    if receipt["schema"] != {"path": RECEIPT_SCHEMA_PATH, "sha256": schema_hash}:
        raise ConstructFirewallError("receipt schema identity drift")
    if receipt["receipt_sha256"] != _self_hash(receipt, "receipt_sha256"):
        raise ConstructFirewallError("receipt semantic identity drift")
    expected = _expected_publication_records(contract)
    records = receipt["records"]
    for observed, frozen in zip(records, expected, strict=True):
        for key in ("publication_identity", "identifier_type", "identifier_value", "official_date", "official_metadata_authority", "requested_url"):
            if observed[key] != frozen[key]:
                raise ConstructFirewallError("receipt publication projection drift")
        projection = {key: observed[key] for key in ("publication_identity", "identifier_type", "identifier_value", "official_date", "official_metadata_authority", "requested_url", "final_url")}
        if observed["projected_metadata_sha256"] != canonical_value_sha256(projection):
            raise ConstructFirewallError("projected metadata identity drift")
    earliest, cutoff = _derive_cutoff(records)
    if (earliest, cutoff) != (EARLIEST_DATE, CUTOFF) or receipt["derivation"] != {"rule": "one_second_before_earliest_official_publication_date_utc", "earliest_official_date": earliest, "strictly_before_cutoff": cutoff}:
        raise ConstructFirewallError("cutoff derivation drift")
    if receipt["access_counters"] != ACCESS_COUNTERS:
        raise ConstructFirewallError("receipt access counters are not zero")
    return receipt


def _current_checkout(root: Path) -> dict[str, str]:
    try:
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConstructFirewallError("unable to resolve destination checkout provenance") from exc
    if not branch or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ConstructFirewallError("invalid destination checkout provenance")
    return {"branch": branch, "head": head}


def build_receipt(contract: Mapping[str, Any], contract_raw: bytes, *, schema_hash: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    earliest, cutoff = _derive_cutoff(records)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "MECHANISTIC_BELIEF_CUTOFF_RECEIPT_VALID",
        "receipt_uid": "mechanistic_belief_cutoff_receipt",
        "contract": {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)},
        "schema": {"path": RECEIPT_SCHEMA_PATH, "sha256": schema_hash},
        "derivation": {"rule": contract["cutoff"]["derivation_rule"], "earliest_official_date": earliest, "strictly_before_cutoff": cutoff},
        "records": [dict(record) for record in records],
        "access_policy": _expected_receipt_access_policy(),
        "access_counters": deepcopy(ACCESS_COUNTERS),
        "retrieval_summary": {
            "metadata_source": "official",
            "exact_request_count": 6,
            "serial": True,
            "raw_response_persisted": False,
            "total_raw_response_bytes": sum(int(record["raw_response_bytes"]) for record in records),
        },
        "blockers": [],
    }
    receipt["receipt_sha256"] = _self_hash(receipt, "receipt_sha256")
    return receipt


def build_authority(contract: Mapping[str, Any], contract_raw: bytes, receipt: Mapping[str, Any], receipt_raw: bytes, *, receipt_schema_hash: str, authority_schema_hash: str, destination_checkout: Mapping[str, str], historical_authorities: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    authority: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "status": "MECHANISTIC_BELIEF_CONSTRUCT_FIREWALL_FROZEN",
        "authority_uid": "mechanistic_belief_construct_authority",
        "method_uid": METHOD_UID,
        "contract": {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)},
        "schemas": {
            "cutoff_receipt": {"path": RECEIPT_SCHEMA_PATH, "sha256": receipt_schema_hash},
            "construct_authority": {"path": AUTHORITY_SCHEMA_PATH, "sha256": authority_schema_hash},
        },
        "cutoff_receipt": {"path": RECEIPT_PATH, "file_sha256": sha256_bytes(receipt_raw), "receipt_sha256": receipt["receipt_sha256"], "record_count": 6, "strictly_before_cutoff": CUTOFF},
        "metadata_record_bindings": [
            {key: record[key] for key in ("publication_identity", "raw_response_sha256", "raw_response_bytes", "projected_metadata_sha256")}
            for record in receipt["records"]
        ],
        "starting_authority": {"branch": STARTING_BRANCH, "commit": STARTING_COMMIT},
        "destination_checkout": dict(destination_checkout),
        "construct": _expected_construct(contract),
        "historical_authorities": [dict(record) for record in historical_authorities],
        "downstream_authorization": deepcopy(DOWNSTREAM_AUTHORIZATION),
        "artifacts_created": deepcopy(ARTIFACTS_CREATED),
        "limitations": deepcopy(LIMITATIONS),
        "blockers": [],
    }
    authority["authority_sha256"] = _self_hash(authority, "authority_sha256")
    return authority


def validate_construct_authority(*, repository_root: Path, contract_path: Path | str = CONTRACT_PATH, receipt_path: Path | str = RECEIPT_PATH, authority_path: Path | str = AUTHORITY_PATH) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    contract, contract_raw = validate_contract(root, contract_path)
    receipt = validate_cutoff_receipt(repository_root=root, contract_path=contract_path, receipt_path=receipt_path)
    _validate_frozen_metadata_policy(contract, receipt.get("access_policy"))
    receipt_file = _resolve_file(root, RECEIPT_PATH, must_exist=True, label="cutoff receipt")
    receipt_raw = receipt_file.read_bytes()
    receipt_schema, receipt_schema_hash = _validate_schema_file(_resolve_file(root, RECEIPT_SCHEMA_PATH, must_exist=True, label="receipt schema"), EXPECTED_RECEIPT_SCHEMA_SHA256)
    del receipt_schema
    authority_schema, authority_schema_hash = _validate_schema_file(_resolve_file(root, AUTHORITY_SCHEMA_PATH, must_exist=True, label="authority schema"), EXPECTED_AUTHORITY_SCHEMA_SHA256)
    relative = Path(authority_path).as_posix() if not Path(authority_path).is_absolute() else Path(authority_path).resolve(strict=True).relative_to(root).as_posix()
    if relative != AUTHORITY_PATH:
        raise ConstructFirewallError(f"authority must be {AUTHORITY_PATH}")
    authority, authority_raw = _load_json(_resolve_file(root, relative, must_exist=True, label="construct authority"), label="construct authority")
    if authority_raw != canonical_json_bytes(authority):
        raise ConstructFirewallError("construct authority is not canonical JSON")
    _validate_json_schema(authority, authority_schema, label="construct authority")
    _validate_no_forbidden_fields(authority)
    if authority["authority_sha256"] != _self_hash(authority, "authority_sha256"):
        raise ConstructFirewallError("construct authority semantic identity drift")
    if authority["contract"] != {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)}:
        raise ConstructFirewallError("construct authority contract identity drift")
    if authority["schemas"] != {"cutoff_receipt": {"path": RECEIPT_SCHEMA_PATH, "sha256": receipt_schema_hash}, "construct_authority": {"path": AUTHORITY_SCHEMA_PATH, "sha256": authority_schema_hash}}:
        raise ConstructFirewallError("construct authority schema identity drift")
    if authority["cutoff_receipt"] != {"path": RECEIPT_PATH, "file_sha256": sha256_bytes(receipt_raw), "receipt_sha256": receipt["receipt_sha256"], "record_count": 6, "strictly_before_cutoff": CUTOFF}:
        raise ConstructFirewallError("construct authority receipt binding drift")
    expected_bindings = [{key: record[key] for key in ("publication_identity", "raw_response_sha256", "raw_response_bytes", "projected_metadata_sha256")} for record in receipt["records"]]
    if authority["metadata_record_bindings"] != expected_bindings:
        raise ConstructFirewallError("construct authority metadata binding drift")
    if authority["starting_authority"] != {"branch": STARTING_BRANCH, "commit": STARTING_COMMIT} or authority["method_uid"] != METHOD_UID:
        raise ConstructFirewallError("construct authority method or starting identity drift")
    if authority["construct"] != _expected_construct(contract):
        raise ConstructFirewallError("construct semantics drift")
    if authority["historical_authorities"] != validate_historical_authorities(root):
        raise ConstructFirewallError("historical authority binding drift")
    if authority["downstream_authorization"] != DOWNSTREAM_AUTHORIZATION or authority["artifacts_created"] != ARTIFACTS_CREATED:
        raise ConstructFirewallError("downstream closure drift")
    return authority


def _safe_content_path(repository_root: Path, input_root: Path, relative_path: str) -> Path:
    repository_root = repository_root.resolve(strict=True)
    input_root = input_root.resolve(strict=True)
    if not input_root.is_relative_to(repository_root):
        raise ConstructFirewallError("input root escapes repository root")
    normalized = _relative_path(relative_path, label="mechanism source path")
    pure = PurePosixPath(normalized)
    if any(part.lower() in PROTECTED_PATH_PARTS for part in pure.parts):
        raise ConstructFirewallError("mechanism source path targets a protected or evaluation namespace")
    candidate = input_root.joinpath(*pure.parts)
    current = input_root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ConstructFirewallError("mechanism source path traverses a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ConstructFirewallError("mechanism source path does not exist") from exc
    if not resolved.is_relative_to(input_root) or not resolved.is_file():
        raise ConstructFirewallError("mechanism source path escapes input root")
    return candidate


def open_mechanism_source_content(*, repository_root: Path, input_root: Path, relative_path: str, access_kind: str, opener: Callable[[Path], Any]) -> Any:
    contract, _ = validate_contract(repository_root)
    receipt = validate_cutoff_receipt(repository_root=repository_root)
    _validate_frozen_metadata_policy(contract, receipt.get("access_policy"))
    validate_construct_authority(repository_root=repository_root)
    if access_kind != GATE_ACCESS_KIND:
        raise ConstructFirewallError("access kind is not allowlisted")
    path = _safe_content_path(repository_root, input_root, relative_path)
    return opener(path)


def _write_temp(directory: Path, payload: bytes) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix=".mechanistic-belief-", suffix=".tmp", dir=directory, delete=False)
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)
    finally:
        handle.close()


def _install_pair(receipt_path: Path, receipt_raw: bytes, authority_path: Path, authority_raw: bytes, *, replace: Callable[[Path, Path], None] = os.replace) -> None:
    if receipt_path.exists() or authority_path.exists():
        raise ConstructFirewallError("refusing to overwrite non-identical authority output")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_temp = _write_temp(receipt_path.parent, receipt_raw)
    authority_temp = _write_temp(authority_path.parent, authority_raw)
    receipt_installed = False
    try:
        replace(receipt_temp, receipt_path)
        receipt_installed = True
        replace(authority_temp, authority_path)
    except Exception as exc:
        if receipt_installed:
            receipt_path.unlink(missing_ok=True)
        raise ConstructFirewallError("transactional authority installation failed") from exc
    finally:
        receipt_temp.unlink(missing_ok=True)
        authority_temp.unlink(missing_ok=True)


def build_construct_authority(*, repository_root: Path, contract_path: Path | str = CONTRACT_PATH, metadata_source: str = "official", fetcher: Callable[[Mapping[str, Any], Mapping[str, Any]], FetchResult] = fetch_official_metadata, retrieved_at: str | None = None, destination_checkout: Mapping[str, str] | None = None, replace: Callable[[Path, Path], None] = os.replace) -> tuple[dict[str, Any], dict[str, Any], bool]:
    root = repository_root.resolve(strict=True)
    if metadata_source != "official":
        raise ConstructFirewallError("metadata source must be official")
    contract, contract_raw = validate_contract(root, contract_path)
    historical = validate_historical_authorities(root)
    receipt_schema, receipt_schema_hash = _validate_schema_file(_resolve_file(root, RECEIPT_SCHEMA_PATH, must_exist=True, label="receipt schema"), EXPECTED_RECEIPT_SCHEMA_SHA256)
    authority_schema, authority_schema_hash = _validate_schema_file(_resolve_file(root, AUTHORITY_SCHEMA_PATH, must_exist=True, label="authority schema"), EXPECTED_AUTHORITY_SCHEMA_SHA256)
    receipt_path = root / RECEIPT_PATH
    authority_path = root / AUTHORITY_PATH
    if receipt_path.exists() or authority_path.exists():
        if not (receipt_path.exists() and authority_path.exists()):
            raise ConstructFirewallError("partial existing authority output")
        receipt = validate_cutoff_receipt(repository_root=root, contract_path=contract_path)
        authority = validate_construct_authority(repository_root=root, contract_path=contract_path)
        return receipt, authority, False
    timestamp = retrieved_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    policy = _mapping(contract["retrieval_policy"], label="retrieval policy")
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for publication in _expected_publication_records(contract):
        try:
            response = fetcher(publication, policy)
            total_bytes += len(response.body)
            if total_bytes > int(policy["maximum_total_response_bytes"]):
                raise ConstructFirewallError("official metadata total byte limit exceeded")
            records.append(
                project_metadata_record(
                    publication,
                    response,
                    retrieved_at=timestamp,
                    allowed_host_suffixes=policy["allowed_host_suffixes"],
                    maximum_response_bytes=int(policy["maximum_response_bytes"]),
                    maximum_redirects=int(policy["maximum_redirects"]),
                )
            )
        except ConstructFirewallError as exc:
            raise ConstructFirewallError(
                f"official metadata projection failed for {publication['publication_identity']}: {exc}"
            ) from exc
    receipt = build_receipt(contract, contract_raw, schema_hash=receipt_schema_hash, records=records)
    _validate_json_schema(receipt, receipt_schema, label="cutoff receipt")
    _validate_no_forbidden_fields(receipt)
    receipt_raw = canonical_json_bytes(receipt)
    authority = build_authority(contract, contract_raw, receipt, receipt_raw, receipt_schema_hash=receipt_schema_hash, authority_schema_hash=authority_schema_hash, destination_checkout=destination_checkout or _current_checkout(root), historical_authorities=historical)
    _validate_json_schema(authority, authority_schema, label="construct authority")
    _validate_no_forbidden_fields(authority)
    authority_raw = canonical_json_bytes(authority)
    _install_pair(receipt_path, receipt_raw, authority_path, authority_raw, replace=replace)
    return receipt, authority, True


def read_bytes_opener(path: Path) -> bytes:
    return path.read_bytes()
