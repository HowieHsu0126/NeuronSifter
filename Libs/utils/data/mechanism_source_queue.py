"""Immutable, metadata-only mechanism-source queue construction and validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
import yaml

from Libs.utils.data.mechanistic_belief_construct_firewall import (
    canonical_json_bytes,
    canonical_value_sha256,
    sha256_bytes,
    sha256_file,
    validate_construct_authority,
    validate_cutoff_receipt,
)
from Libs.utils.model.mechanistic_belief_prior_reliability import (
    validate_prior_reliability_outputs,
)


CONTRACT_PATH = "Libs/configs/validation/mechanism_source_queue.yaml"
PREREGISTRATION_SCHEMA_PATH = (
    "Libs/configs/schema/mechanism_source_preregistration.schema.json"
)
QUEUE_SCHEMA_PATH = "Libs/configs/schema/mechanism_source_queue.schema.json"
CHRONOLOGY_SCHEMA_PATH = (
    "Libs/configs/schema/mechanism_source_queue_chronology.schema.json"
)
AUTHORIZATION_SCHEMA_PATH = (
    "Libs/configs/schema/mechanism_source_content_authorization.schema.json"
)
PREREGISTRATION_PATH = (
    "Input/manifests/mechanistic_belief/mechanism_source_preregistration.json"
)
QUEUE_PATH = "Input/manifests/mechanistic_belief/mechanism_source_queue.json"
CHRONOLOGY_PATH = "Output/audits/mechanism_source_queue_chronology.json"
AUTHORIZATION_PATH = "Output/audits/mechanism_source_content_authorization.json"

EXPECTED_CONTRACT_SHA256 = "1baca68cef6e5d4b4c4f8944b51f4f74e8a1bdd2a8c34cc64feee45f795660c8"
EXPECTED_SCHEMA_SHA256 = {
    PREREGISTRATION_SCHEMA_PATH: "367134588caa128995b56ad67ae9e1d72d1d12aec6eb964ae974c459bd500900",
    QUEUE_SCHEMA_PATH: "e90bbee2d49c520dd9f3950c0f86a5507726e63d9e92583f3dd7389156ee9205",
    CHRONOLOGY_SCHEMA_PATH: "d0cb9eb95c5b14ad3af9c6c23cba6610de314336ade3ecd27cd0a929221707e0",
    AUTHORIZATION_SCHEMA_PATH: "f60e973ad66a6a5ace8c879c93b93411468cfd8b125ca139677e493e3d2bbf75",
}

CONTRACT_SCHEMA_VERSION = "neuronsifter-mechanism-source-queue-contract-v1"
PREREGISTRATION_SCHEMA_VERSION = "neuronsifter-mechanism-source-preregistration-v1"
QUEUE_SCHEMA_VERSION = "neuronsifter-mechanism-source-queue-v1"
CHRONOLOGY_SCHEMA_VERSION = "neuronsifter-mechanism-source-queue-chronology-v1"
AUTHORIZATION_SCHEMA_VERSION = (
    "neuronsifter-mechanism-source-content-authorization-v1"
)
PROJECTOR_VERSION = "mechanism-source-action-projector-v1"
CUTOFF = "2021-03-02T23:59:59Z"

FAMILY_ROWS = (
    (1, "publication_metadata", "publication", 4, "preregistered_metadata_only", "unavailable"),
    (2, "open_targets_target_metadata", "target_database", 2, "disabled_cutoff_ineligible_release", "cutoff_ineligible_release"),
    (3, "chembl_compound_target_metadata", "compound_target_database", 2, "disabled_source_date_ambiguous", "source_date_ambiguous"),
    (4, "bindingdb_record_metadata", "binding_database", 1, "disabled_cutoff_ineligible_release", "cutoff_ineligible_release"),
    (5, "drugcentral_drug_moa_metadata", "drug_moa_database", 1, "disabled_cutoff_ineligible_release", "cutoff_ineligible_release"),
    (6, "dgidb_drug_gene_metadata", "drug_gene_database", 2, "disabled_source_date_ambiguous", "source_date_ambiguous"),
)
INTERVENTION_ROWS = (
    ("LECANEMAB", "protein_or_biologic"),
    ("DONANEMAB", "protein_or_biologic"),
    ("VERUBECESTAT", "small_molecule"),
    ("ADUCANUMAB", "protein_or_biologic"),
    ("SOLANEZUMAB", "protein_or_biologic"),
    ("TRAMIPROSATE", "small_molecule"),
)
MECHANISM_TAGS = (
    "exposure_occupancy",
    "cns_transport",
    "protein_pathology",
    "neuroinflammation",
    "electrophysiology",
    "structural_morphology",
)
DISPOSITIONS = (
    "eligible",
    "duplicate",
    "unavailable",
    "source_date_ambiguous",
    "access_license_ambiguous",
    "cutoff_ineligible_release",
    "forbidden_metadata",
    "request_failed",
)
DISPOSITION_PRECEDENCE = (
    "forbidden_metadata",
    "request_failed",
    "cutoff_ineligible_release",
    "source_date_ambiguous",
    "access_license_ambiguous",
    "unavailable",
    "duplicate",
    "eligible",
)
ORDERING_FIELDS = (
    "canonical_intervention_uid",
    "mechanism_tag_priority",
    "family_priority",
    "official_date_ascending",
    "source_type",
    "canonical_source_identity",
    "family_slot_ordinal",
)
ZERO_COUNTERS = {
    "six_context_outcome_access": 0,
    "protected_holdout_access": 0,
    "source_content_access": 0,
}
DOWNSTREAM_AUTHORIZATION = {
    "mechanism_basis_sufficiency_authorized": False,
    "numerical_prior_population_authorized": False,
    "world_action_compiler_authorized": False,
    "runtime_implementation_authorized": False,
    "agent_trace_authorized": False,
    "inference_authorized": False,
    "training_authorized": False,
    "gpu_authorized": False,
    "evaluation_authorized": False,
}
REQUEST_POLICY = {
    "maximum_requests": 48,
    "concurrency": 1,
    "timeout_seconds": 30,
    "retries": 0,
    "maximum_response_bytes": 2 * 1024 * 1024,
    "maximum_total_response_bytes": 32 * 1024 * 1024,
    "maximum_redirects": 3,
    "minimum_request_interval_ms": 400,
    "allowed_host_suffixes": ["ncbi.nlm.nih.gov"],
    "raw_response_persistence_allowed": False,
    "response_value_logging_allowed": False,
}
QUEUE_POLICY = {
    "family_cap": 6,
    "global_slot_cap": 160,
    "per_intervention_slot_cap": 12,
    "exact_slots_per_intervention": 12,
    "identity_normalization": {
        "pmid": "digits_only",
        "doi": "lowercase_trimmed",
        "structured_database_records": "family_scoped_no_fuzzy_merge",
    },
    "deduplication_key": [
        "canonical_intervention_uid",
        "canonical_source_identity",
    ],
    "disposition_precedence": list(DISPOSITION_PRECEDENCE),
    "ordering": list(ORDERING_FIELDS),
    "no_refill_after_exclusion": True,
    "no_replacement_after_preregistration": True,
}

FORBIDDEN_FIELD_NAMES = {
    "aggregate_affinity_score",
    "aggregate_target_evidence_score",
    "abstract",
    "benefit",
    "biomarker_endpoint_summary",
    "citation_context",
    "clinical_endpoint_alignment_policy",
    "clinical_endpoint_source_url",
    "clinical_endpoint_summary",
    "efficacy",
    "endpoint_source_url",
    "evaluation_metric",
    "evidence_summary",
    "expected_effect_direction",
    "expected_standardized_effect",
    "figure",
    "figures",
    "note",
    "outcome",
    "outcomes",
    "pkpd_feasibility_score",
    "ranking",
    "rank",
    "result",
    "results",
    "result_prose",
    "selection_reason",
    "supplement",
    "supplements",
    "title",
}
ALLOWED_RESPONSE_FIELDS = {
    "publication_metadata": [
        "esearchresult.idlist",
        "result.uids",
        "result.<pmid>.uid",
        "result.<pmid>.sortpubdate",
        "result.<pmid>.articleids.idtype",
        "result.<pmid>.articleids.value",
    ],
    "open_targets_target_metadata": [],
    "chembl_compound_target_metadata": [],
    "bindingdb_record_metadata": [],
    "drugcentral_drug_moa_metadata": [],
    "dgidb_drug_gene_metadata": [],
}
FORBIDDEN_RESPONSE_FIELDS = (
    "title",
    "abstract",
    "full_text",
    "result_prose",
    "conclusions",
    "efficacy",
    "effect",
    "ranking",
    "figure",
    "supplement",
    "citation_context",
    "outcome",
    "source_content",
)

CODE_POLICY = {
    "cutoff": CUTOFF,
    "families": FAMILY_ROWS,
    "interventions": INTERVENTION_ROWS,
    "mechanism_tags": MECHANISM_TAGS,
    "forbidden_fields": tuple(sorted(FORBIDDEN_FIELD_NAMES)),
    "allowed_response_fields": ALLOWED_RESPONSE_FIELDS,
    "forbidden_response_fields": FORBIDDEN_RESPONSE_FIELDS,
    "disposition_precedence": DISPOSITION_PRECEDENCE,
    "ordering": ORDERING_FIELDS,
    "request_policy": REQUEST_POLICY,
    "queue_policy": QUEUE_POLICY,
    "zero_counters": ZERO_COUNTERS,
    "downstream_authorization": DOWNSTREAM_AUTHORIZATION,
}
EXPECTED_CODE_POLICY_SHA256 = "3b3c4dc6875b3dd37dd281310ea0764a23306d3b7df1a2ceca36aac8b5a53680"


class MechanismSourceQueueError(ValueError):
    """Raised when queue chronology, policy, or metadata validation fails."""


@dataclass(frozen=True)
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    body: bytes
    redirect_count: int = 0


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MechanismSourceQueueError(f"{label} must be an object with string keys")
    return value


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise MechanismSourceQueueError(f"{label} must be an array")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "*?[]\x00"):
        raise MechanismSourceQueueError(f"{label} must be a normalized relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or "." in pure.parts or ".." in pure.parts or str(pure) != value:
        raise MechanismSourceQueueError(f"{label} must be a normalized relative path")
    return value


def _resolve_file(root: Path, relative: str, *, must_exist: bool, label: str) -> Path:
    normalized = _relative_path(relative, label=label)
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*PurePosixPath(normalized).parts)
    current = resolved_root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise MechanismSourceQueueError(f"{label} traverses a symlink")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise MechanismSourceQueueError(f"missing {label}: {relative}") from exc
    if not resolved.is_relative_to(resolved_root):
        raise MechanismSourceQueueError(f"{label} escapes declared root")
    if must_exist and not resolved.is_file():
        raise MechanismSourceQueueError(f"{label} is not a regular file")
    return candidate


def _primary_worktree_root(repository_root: Path) -> Path:
    git_pointer = repository_root / ".git"
    if not git_pointer.is_file():
        return repository_root
    try:
        text = git_pointer.read_text(encoding="utf-8").strip()
        git_dir = Path(text.removeprefix("gitdir: ")).resolve(strict=True)
        common_dir = (git_dir / (git_dir / "commondir").read_text(encoding="utf-8").strip()).resolve(strict=True)
        primary = common_dir.parent.resolve(strict=True)
    except (OSError, UnicodeError) as exc:
        raise MechanismSourceQueueError("unable to resolve primary worktree") from exc
    return primary


def _resolve_source_input(repository_root: Path, relative: str) -> tuple[Path, str]:
    try:
        return _resolve_file(repository_root, relative, must_exist=True, label="source input"), "round_worktree"
    except MechanismSourceQueueError:
        primary = _primary_worktree_root(repository_root)
        if primary == repository_root.resolve(strict=True):
            raise
        return (
            _resolve_file(primary, relative, must_exist=True, label="primary-worktree source input"),
            "primary_worktree_generated_input",
        )


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label=label)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise MechanismSourceQueueError(f"unable to load {label}") from exc


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        return _mapping(json.loads(raw.decode("utf-8")), label=label), raw
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MechanismSourceQueueError(f"unable to load {label}") from exc


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    semantic = dict(payload)
    semantic.pop(field, None)
    return canonical_value_sha256(semantic)


def _validate_no_forbidden_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in FORBIDDEN_FIELD_NAMES:
                raise MechanismSourceQueueError("forbidden metadata field persisted")
            _validate_no_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_no_forbidden_fields(nested)


def _schema(repository_root: Path, relative: str) -> tuple[dict[str, Any], str]:
    path = _resolve_file(repository_root, relative, must_exist=True, label="schema")
    observed_hash = sha256_file(path)
    if observed_hash != EXPECTED_SCHEMA_SHA256[relative]:
        raise MechanismSourceQueueError(f"frozen schema identity drift: {relative}")
    schema, _ = _load_json(path, label="schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise MechanismSourceQueueError("invalid Draft 2020-12 schema") from exc
    return schema, observed_hash


def _schema_validate(payload: Mapping[str, Any], schema: Mapping[str, Any], *, label: str) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path)
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].path)
        raise MechanismSourceQueueError(f"{label} schema validation failed at {location}")


def _code_policy_sha256() -> str:
    return canonical_value_sha256(CODE_POLICY)


def validate_contract(
    repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> tuple[dict[str, Any], bytes]:
    root = repository_root.resolve(strict=True)
    path = Path(contract_path)
    relative = (
        path.resolve(strict=True).relative_to(root).as_posix()
        if path.is_absolute()
        else _relative_path(path.as_posix(), label="contract path")
    )
    if relative != CONTRACT_PATH:
        raise MechanismSourceQueueError(f"contract must be {CONTRACT_PATH}")
    resolved = _resolve_file(root, relative, must_exist=True, label="contract")
    raw = resolved.read_bytes()
    if sha256_bytes(raw) != EXPECTED_CONTRACT_SHA256:
        raise MechanismSourceQueueError("frozen contract identity drift")
    if _code_policy_sha256() != EXPECTED_CODE_POLICY_SHA256:
        raise MechanismSourceQueueError("independent code policy pin drift")
    contract = _load_yaml(resolved, label="mechanism-source queue contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise MechanismSourceQueueError("contract schema version drift")
    if contract.get("status") != "frozen_before_network_and_source_content":
        raise MechanismSourceQueueError("contract status drift")
    if contract.get("method_uid") != "mechanistic_belief_twin_v1":
        raise MechanismSourceQueueError("method identity drift")
    if contract.get("cutoff") != {
        "official_date_at_or_before": CUTOFF,
        "comparison": "inclusive",
    }:
        raise MechanismSourceQueueError("cutoff drift")
    observed_families = [
        (
            row.get("priority"),
            row.get("family_uid"),
            row.get("source_type"),
            row.get("slot_quota_per_intervention"),
            row.get("network_mode"),
            row.get("preregistered_disposition"),
        )
        for row in _sequence(contract.get("families"), label="families")
        if isinstance(row, dict)
    ]
    if observed_families != list(FAMILY_ROWS):
        raise MechanismSourceQueueError("source family roster or order drift")
    projection = _mapping(contract.get("projection"), label="projection")
    observed_interventions = [
        (row.get("intervention_label"), row.get("modality"))
        for row in _sequence(projection.get("interventions"), label="interventions")
        if isinstance(row, dict)
    ]
    if observed_interventions != list(INTERVENTION_ROWS):
        raise MechanismSourceQueueError("six-context intervention roster drift")
    if tuple(contract.get("mechanism_relevance_vocabulary", [])) != MECHANISM_TAGS:
        raise MechanismSourceQueueError("mechanism relevance vocabulary drift")
    if contract.get("request_policy") != REQUEST_POLICY:
        raise MechanismSourceQueueError("request policy drift")
    if contract.get("queue_policy") != QUEUE_POLICY:
        raise MechanismSourceQueueError("queue policy drift")
    if contract.get("access_counters") != ZERO_COUNTERS:
        raise MechanismSourceQueueError("access counters are not zero")
    if contract.get("downstream_authorization") != {
        "source_content_authorized": False,
        **DOWNSTREAM_AUTHORIZATION,
    }:
        raise MechanismSourceQueueError("downstream authorization drift")
    if set(projection.get("forbidden_fields", [])) != FORBIDDEN_FIELD_NAMES:
        raise MechanismSourceQueueError("projection denylist drift")
    response_policy = _mapping(
        contract.get("response_field_policy"), label="response field policy"
    )
    if response_policy != {
        "allowed_by_family": ALLOWED_RESPONSE_FIELDS,
        "forbidden_for_every_family": list(FORBIDDEN_RESPONSE_FIELDS),
    }:
        raise MechanismSourceQueueError("per-family response field policy drift")
    publication_query = _mapping(
        contract.get("publication_query"), label="publication query"
    )
    if publication_query.get("allowed_response_fields") != {
        "search": ["esearchresult.idlist"],
        "summary": ALLOWED_RESPONSE_FIELDS["publication_metadata"][1:],
    }:
        raise MechanismSourceQueueError("publication response allowlist drift")
    return contract, raw


def validate_upstream_authorities(
    repository_root: Path, contract: Mapping[str, Any]
) -> list[dict[str, str]]:
    validate_cutoff_receipt(repository_root=repository_root)
    validate_construct_authority(repository_root=repository_root)
    validate_prior_reliability_outputs(repository_root=repository_root)
    records: list[dict[str, str]] = []
    for item in _sequence(contract.get("upstream_authorities"), label="upstream authorities"):
        row = _mapping(item, label="upstream authority")
        relative = _relative_path(row.get("path"), label="upstream authority path")
        observed = sha256_file(
            _resolve_file(repository_root, relative, must_exist=True, label="upstream authority")
        )
        if observed != row.get("sha256"):
            raise MechanismSourceQueueError(f"upstream authority identity drift: {relative}")
        records.append({"path": relative, "sha256": observed, "validation_status": "validated"})
    if len(records) != 8:
        raise MechanismSourceQueueError("exactly eight upstream bindings are required")
    return records


def _source_input_records(
    repository_root: Path, contract: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Path]]:
    records: list[dict[str, str]] = []
    paths: dict[str, Path] = {}
    source_inputs = _mapping(contract.get("source_inputs"), label="source inputs")
    if len(source_inputs) != 9:
        raise MechanismSourceQueueError("exactly nine source inputs are required")
    for source_uid, value in source_inputs.items():
        row = _mapping(value, label=f"source input {source_uid}")
        relative = _relative_path(row.get("path"), label="source input path")
        path, resolved_from = _resolve_source_input(repository_root, relative)
        observed = sha256_file(path)
        if observed != row.get("sha256"):
            raise MechanismSourceQueueError(f"source input identity drift: {relative}")
        records.append(
            {
                "source_uid": source_uid,
                "path": relative,
                "sha256": observed,
                "resolved_from": resolved_from,
            }
        )
        paths[source_uid] = path
    return records, paths


def _validate_source_declarations(paths: Mapping[str, Path]) -> None:
    data_registry = _load_yaml(paths["data_registry"], label="data registry")
    raw_sources = _mapping(data_registry.get("public_downloads"), label="public downloads")
    required_fragments = {
        "open_targets": "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.03/output/",
        "chembl": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/",
        "bindingdb": "BindingDB_All_202604_tsv.zip",
        "drugcentral": "/2021_09_01/",
        "dgidb": "/releases/latest",
    }
    for source_key, fragment in required_fragments.items():
        if source_key not in raw_sources or fragment not in json.dumps(raw_sources[source_key], sort_keys=True):
            raise MechanismSourceQueueError(f"source declaration drift: {source_key}")
    validation = _load_yaml(paths["validation_config"], label="validation config")
    registry = _mapping(
        _mapping(validation.get("validation"), label="validation config").get(
            "external_adapter_registry"
        ),
        label="external adapter registry",
    )
    adapters = _sequence(registry.get("adapters"), label="external adapters")
    required_adapters = ["open_targets", "chembl", "bindingdb", "drugcentral", "dgidb"]
    observed = [str(row.get("source_key")) for row in adapters if isinstance(row, dict)]
    if [key for key in observed if key in required_adapters] != required_adapters:
        raise MechanismSourceQueueError("external adapter roster drift")
    dgidb = _mapping(
        _mapping(registry.get("materialization"), label="materialization").get("dgidb"),
        label="DGIdb materialization",
    )
    if dgidb.get("api_url") != "https://dgidb.org/api/graphql" or dgidb.get("source_release") != "v.5.0.11":
        raise MechanismSourceQueueError("DGIdb endpoint or release drift")


def _read_tsv_projection(
    path: Path, *, allowed_fields: Sequence[str]
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise MechanismSourceQueueError(f"empty projected catalog: {path.name}") from exc
        if len(header) != len(set(header)):
            raise MechanismSourceQueueError(f"duplicate TSV header: {path.name}")
        missing = [field for field in allowed_fields if field not in header]
        if missing:
            raise MechanismSourceQueueError(f"missing projected fields in {path.name}")
        indexes = [(field, header.index(field)) for field in allowed_fields]
        rows: list[dict[str, str]] = []
        for values in reader:
            if len(values) != len(header):
                raise MechanismSourceQueueError(f"malformed TSV row in {path.name}")
            rows.append({field: values[index] for field, index in indexes})
    return rows


def _source_provenance(
    *, path: str, source_hash: str, fields: Sequence[str], projected_value: object
) -> dict[str, Any]:
    return {
        "path": path,
        "source_file_sha256": source_hash,
        "projected_fields": list(fields),
        "projection_sha256": canonical_value_sha256(projected_value),
        "projector_version": PROJECTOR_VERSION,
    }


def project_context_actions(
    repository_root: Path, contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    source_records, paths = _source_input_records(repository_root, contract)
    _validate_source_declarations(paths)
    source_by_uid = {row["source_uid"]: row for row in source_records}
    projection = _mapping(contract["projection"], label="projection")
    dynamics = _load_yaml(paths["dynamics_config"], label="dynamics config")
    dynamics_entries = _sequence(
        _mapping(dynamics.get("dynamics_ready_registry"), label="dynamics registry").get(
            "case_validation_entries"
        ),
        label="case validation entries",
    )
    context_fields = [field for field in projection["context_fields"] if field != "validation_uid"]
    safe_contexts: dict[str, dict[str, str]] = {}
    for raw in dynamics_entries:
        row = _mapping(raw, label="case validation entry")
        safe = {field: str(row[field]) for field in context_fields}
        validation_uid = hashlib.sha1(
            "::".join(
                [safe["case_study_name"], safe["intervention_label"], safe["endpoint_family"]]
            ).encode("utf-8")
        ).hexdigest()[:12]
        safe_contexts[safe["intervention_label"]] = {
            "validation_uid": validation_uid,
            **safe,
        }

    compound_fields = list(projection["compound_fields"])
    target_fields = list(projection["target_fields"])
    intervention_fields = list(projection["intervention_fields"])
    compounds = _read_tsv_projection(paths["compound_catalog"], allowed_fields=compound_fields)
    targets = _read_tsv_projection(paths["target_catalog"], allowed_fields=target_fields)
    programs = _read_tsv_projection(
        paths["intervention_program_catalog"], allowed_fields=intervention_fields
    )
    compounds_by_label: dict[str, list[dict[str, str]]] = {}
    for row in compounds:
        compounds_by_label.setdefault(row["preferred_name"].strip().upper(), []).append(row)
    targets_by_symbol: dict[str, list[dict[str, str]]] = {}
    for row in targets:
        targets_by_symbol.setdefault(row["target_symbol"].strip(), []).append(row)

    rule = _mapping(projection["reference_action_rule"], label="reference action rule")
    projected: list[dict[str, Any]] = []
    for requested in projection["interventions"]:
        requested_row = _mapping(requested, label="projected intervention")
        label = str(requested_row["intervention_label"])
        modality = str(requested_row["modality"])
        context = safe_contexts.get(label)
        if context is None:
            raise MechanismSourceQueueError(f"missing six-context entry: {label}")
        compound_matches = compounds_by_label.get(label, [])
        if len(compound_matches) != 1:
            raise MechanismSourceQueueError(f"compound identity is not unique: {label}")
        compound = compound_matches[0]
        route_rule = _mapping(rule[modality], label="modality action rule")
        matching_programs = [
            row
            for row in programs
            if row["compound_set"] == compound["compound_uid"]
            and row["route"] == route_rule["route"]
            and row["schedule"] == route_rule["schedule"]
            and row["dose_level_id"] == rule["dose_level_id"]
            and row["dose_multiplier"] == str(rule["dose"])
        ]
        if len(matching_programs) != 1:
            raise MechanismSourceQueueError(f"reference action is not unique: {label}")
        program = matching_programs[0]
        target_rows: list[dict[str, str]] = []
        for symbol in sorted(filter(None, program["target_program_label_set"].split("|"))):
            matches = targets_by_symbol.get(symbol, [])
            if len(matches) != 1:
                raise MechanismSourceQueueError(f"target identity is not unique: {symbol}")
            target_rows.append(matches[0])

        compound_identifiers = [
            {"namespace": "compound_uid", "value": compound["compound_uid"]},
            {
                "namespace": compound["canonical_id_namespace"],
                "value": compound["canonical_compound_id"],
            },
        ]
        for namespace, field in (
            ("chembl_id", "chembl_id"),
            ("drugcentral_struct_id", "drugcentral_struct_id"),
            ("bindingdb_ligand_id", "bindingdb_ligand_id"),
        ):
            identifier = {"namespace": namespace, "value": compound[field]}
            if compound[field] and identifier not in compound_identifiers:
                compound_identifiers.append(identifier)
        target_identifiers: list[dict[str, str]] = []
        for target in target_rows:
            target_identifiers.extend(
                [
                    {"namespace": "target_uid", "value": target["target_uid"]},
                    {"namespace": "canonical_entity_id", "value": target["canonical_entity_id"]},
                    {"namespace": "target_symbol", "value": target["target_symbol"]},
                ]
            )
        safe_projection = {
            **context,
            "intervention_program_uid": program["intervention_program_uid"],
            "regimen_uid": program["regimen_uid"],
            "compound_identifiers": compound_identifiers,
            "target_identifiers": target_identifiers,
            "modality": modality,
            "dose": float(rule["dose"]),
            "dose_unit": str(rule["dose_unit"]),
            "route": program["route"],
            "schedule": program["schedule"],
        }
        safe_projection["source_provenance"] = [
            _source_provenance(
                path=source_by_uid["dynamics_config"]["path"],
                source_hash=source_by_uid["dynamics_config"]["sha256"],
                fields=["validation_uid", *context_fields],
                projected_value=context,
            ),
            _source_provenance(
                path=source_by_uid["compound_catalog"]["path"],
                source_hash=source_by_uid["compound_catalog"]["sha256"],
                fields=compound_fields,
                projected_value=compound,
            ),
            _source_provenance(
                path=source_by_uid["target_catalog"]["path"],
                source_hash=source_by_uid["target_catalog"]["sha256"],
                fields=target_fields,
                projected_value=target_rows,
            ),
            _source_provenance(
                path=source_by_uid["intervention_program_catalog"]["path"],
                source_hash=source_by_uid["intervention_program_catalog"]["sha256"],
                fields=intervention_fields,
                projected_value=program,
            ),
        ]
        safe_projection["access_counters"] = deepcopy(ZERO_COUNTERS)
        _validate_no_forbidden_fields(safe_projection)
        projected.append(safe_projection)
    if len(projected) != 6:
        raise MechanismSourceQueueError("exactly six context/action projections are required")
    return projected, source_records


def _expanded_requests(
    contract: Mapping[str, Any], projections: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    query = _mapping(contract["publication_query"], label="publication query")
    requests: list[dict[str, Any]] = []
    for projection in projections:
        query_string = str(query["query_template"]).format(
            intervention=projection["intervention_label"]
        )
        parameters = [
            ("db", str(query["database"])),
            ("retmode", "json"),
            ("retmax", "4"),
            ("sort", str(query["result_order"])),
            ("term", query_string),
        ]
        search_url = f"{contract['families'][0]['endpoint']}?{urlencode(parameters)}"
        query_uid = "publication-query-" + hashlib.sha256(
            f"{projection['intervention_program_uid']}\n{query_string}\n{search_url}".encode()
        ).hexdigest()[:16]
        requests.append(
            {
                "query_uid": query_uid,
                "family_uid": "publication_metadata",
                "canonical_intervention_uid": projection["intervention_program_uid"],
                "query_string": query_string,
                "search_url": search_url,
                "summary_url_template": (
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    "?db=pubmed&retmode=json&version=2.0&id={pmid_csv}"
                ),
                "maximum_records": 4,
            }
        )
    return requests


def _artifact_paths(contract: Mapping[str, Any]) -> dict[str, str]:
    observed = _mapping(contract.get("artifacts"), label="artifact paths")
    expected = {
        "preregistration": PREREGISTRATION_PATH,
        "queue": QUEUE_PATH,
        "chronology": CHRONOLOGY_PATH,
        "content_authorization": AUTHORIZATION_PATH,
    }
    if observed != expected:
        raise MechanismSourceQueueError("artifact path drift")
    return expected


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_payload(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload, raw = _load_json(path, label=label)
    if raw != canonical_json_bytes(payload):
        raise MechanismSourceQueueError(f"{label} is not canonical JSON")
    return payload, raw


def _write_temp(directory: Path, payload: bytes) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=".mechanism-source-queue-", suffix=".tmp", dir=directory, delete=False
    )
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)
    finally:
        handle.close()


def _install_one(path: Path, raw: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == raw:
            return False
        raise MechanismSourceQueueError("refusing to overwrite non-identical artifact")
    temporary = _write_temp(path.parent, raw)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _install_many(items: Sequence[tuple[Path, bytes]]) -> bool:
    existing = [path.exists() for path, _ in items]
    if any(existing):
        if not all(existing):
            raise MechanismSourceQueueError("partial existing frozen artifact set")
        if all(path.read_bytes() == raw for path, raw in items):
            return False
        raise MechanismSourceQueueError("refusing to overwrite non-identical artifact set")
    for path, _ in items:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporaries = [(path, _write_temp(path.parent, raw)) for path, raw in items]
    installed: list[Path] = []
    try:
        for path, temporary in temporaries:
            os.replace(temporary, path)
            installed.append(path)
    except Exception as exc:
        for path in installed:
            path.unlink(missing_ok=True)
        raise MechanismSourceQueueError("transactional artifact installation failed") from exc
    finally:
        for _, temporary in temporaries:
            temporary.unlink(missing_ok=True)
    return True


def build_preregistration(
    *,
    repository_root: Path,
    contract_path: Path | str = CONTRACT_PATH,
    now: Callable[[], str] = _now,
) -> tuple[dict[str, Any], bool]:
    root = repository_root.resolve(strict=True)
    contract, contract_raw = validate_contract(root, contract_path)
    artifact_paths = _artifact_paths(contract)
    output_path = root / artifact_paths["preregistration"]
    if output_path.exists():
        return validate_preregistration(repository_root=root, contract_path=contract_path), False
    upstream = validate_upstream_authorities(root, contract)
    projections, source_inputs = project_context_actions(root, contract)
    schemas: dict[str, dict[str, str]] = {}
    for key, path in (
        ("preregistration", PREREGISTRATION_SCHEMA_PATH),
        ("queue", QUEUE_SCHEMA_PATH),
        ("chronology", CHRONOLOGY_SCHEMA_PATH),
        ("content_authorization", AUTHORIZATION_SCHEMA_PATH),
    ):
        _, digest = _schema(root, path)
        schemas[key] = {"path": path, "sha256": digest}
    preregistration: dict[str, Any] = {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "status": "MECHANISM_SOURCE_QUEUE_PREREGISTERED",
        "preregistration_uid": "mechanism_source_preregistration",
        "contract": {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)},
        "schemas": schemas,
        "upstream_authorities": upstream,
        "code_policy_sha256": _code_policy_sha256(),
        "source_inputs": source_inputs,
        "context_action_projections": projections,
        "source_families": deepcopy(contract["families"]),
        "expanded_requests": _expanded_requests(contract, projections),
        "request_policy": deepcopy(REQUEST_POLICY),
        "field_policy": {
            "allowed_projection_fields": list(contract["publication_query"]["allowed_response_fields"]["search"])
            + list(contract["publication_query"]["allowed_response_fields"]["summary"]),
            "forbidden_fields": list(contract["projection"]["forbidden_fields"]),
            "allowed_by_family": deepcopy(ALLOWED_RESPONSE_FIELDS),
            "forbidden_for_every_family": list(FORBIDDEN_RESPONSE_FIELDS),
            "raw_response_persistence_allowed": False,
            "response_value_logging_allowed": False,
        },
        "queue_policy": deepcopy(QUEUE_POLICY),
        "cutoff": CUTOFF,
        "access_counters": deepcopy(ZERO_COUNTERS),
        "network_requests_performed": 0,
        "source_content_accessed": False,
        "preregistered_at": now(),
        "blockers": [],
    }
    preregistration["preregistration_sha256"] = _self_hash(
        preregistration, "preregistration_sha256"
    )
    schema, _ = _schema(root, PREREGISTRATION_SCHEMA_PATH)
    _schema_validate(preregistration, schema, label="preregistration")
    _validate_no_forbidden_fields(preregistration)
    created = _install_one(output_path, canonical_json_bytes(preregistration))
    return validate_preregistration(repository_root=root, contract_path=contract_path), created


def validate_preregistration(
    *, repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    contract, contract_raw = validate_contract(root, contract_path)
    artifact_paths = _artifact_paths(contract)
    payload, raw = _canonical_payload(
        _resolve_file(
            root, artifact_paths["preregistration"], must_exist=True, label="preregistration"
        ),
        label="preregistration",
    )
    schema, _ = _schema(root, PREREGISTRATION_SCHEMA_PATH)
    _schema_validate(payload, schema, label="preregistration")
    _validate_no_forbidden_fields(payload)
    if payload["preregistration_sha256"] != _self_hash(payload, "preregistration_sha256"):
        raise MechanismSourceQueueError("preregistration semantic identity drift")
    if payload["contract"] != {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)}:
        raise MechanismSourceQueueError("preregistration contract binding drift")
    expected_schemas = {}
    for key, path in (
        ("preregistration", PREREGISTRATION_SCHEMA_PATH),
        ("queue", QUEUE_SCHEMA_PATH),
        ("chronology", CHRONOLOGY_SCHEMA_PATH),
        ("content_authorization", AUTHORIZATION_SCHEMA_PATH),
    ):
        _, digest = _schema(root, path)
        expected_schemas[key] = {"path": path, "sha256": digest}
    if payload["schemas"] != expected_schemas:
        raise MechanismSourceQueueError("preregistration schema binding drift")
    if payload["upstream_authorities"] != validate_upstream_authorities(root, contract):
        raise MechanismSourceQueueError("upstream authority binding drift")
    projections, source_inputs = project_context_actions(root, contract)
    if payload["source_inputs"] != source_inputs or payload["context_action_projections"] != projections:
        raise MechanismSourceQueueError("six-context projection drift")
    if payload["source_families"] != contract["families"]:
        raise MechanismSourceQueueError("family preregistration drift")
    if payload["expanded_requests"] != _expanded_requests(contract, projections):
        raise MechanismSourceQueueError("expanded query or endpoint drift")
    if payload["request_policy"] != REQUEST_POLICY or payload["queue_policy"] != QUEUE_POLICY:
        raise MechanismSourceQueueError("preregistered policy drift")
    if payload["field_policy"] != {
        "allowed_projection_fields": ALLOWED_RESPONSE_FIELDS["publication_metadata"],
        "forbidden_fields": list(contract["projection"]["forbidden_fields"]),
        "allowed_by_family": ALLOWED_RESPONSE_FIELDS,
        "forbidden_for_every_family": list(FORBIDDEN_RESPONSE_FIELDS),
        "raw_response_persistence_allowed": False,
        "response_value_logging_allowed": False,
    }:
        raise MechanismSourceQueueError("preregistered field policy drift")
    if payload["code_policy_sha256"] != _code_policy_sha256():
        raise MechanismSourceQueueError("preregistered code policy drift")
    if payload["network_requests_performed"] != 0 or payload["access_counters"] != ZERO_COUNTERS:
        raise MechanismSourceQueueError("network or protected access occurred before freeze")
    if payload["source_content_accessed"]:
        raise MechanismSourceQueueError("source content was accessed before queue freeze")
    if raw != canonical_json_bytes(payload):
        raise MechanismSourceQueueError("preregistration byte identity drift")
    return payload


def _host_allowed(url: str, suffixes: Sequence[str]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and bool(host) and any(
        host == suffix or host.endswith(f".{suffix}") for suffix in suffixes
    )


def fetch_metadata(request_spec: Mapping[str, Any], policy: Mapping[str, Any]) -> FetchResult:
    requested_url = str(request_spec["url"])
    suffixes = [str(value) for value in policy["allowed_host_suffixes"]]
    if not _host_allowed(requested_url, suffixes):
        raise MechanismSourceQueueError("metadata host is not allowlisted")
    current_url = requested_url
    opener = build_opener(ProxyHandler({}), _NoRedirect)
    for redirect_count in range(int(policy["maximum_redirects"]) + 1):
        request = Request(
            current_url,
            method="GET",
            headers={str(key): str(value) for key, value in request_spec["headers"].items()},
        )
        try:
            with opener.open(request, timeout=int(policy["timeout_seconds"])) as response:
                body = response.read(int(policy["maximum_response_bytes"]) + 1)
                if len(body) > int(policy["maximum_response_bytes"]):
                    raise MechanismSourceQueueError("metadata response exceeds byte limit")
                final_url = response.geturl()
                if not _host_allowed(final_url, suffixes):
                    raise MechanismSourceQueueError("metadata final host is not allowlisted")
                return FetchResult(
                    requested_url=requested_url,
                    final_url=final_url,
                    status_code=int(response.status),
                    content_type=str(response.headers.get("Content-Type", "")),
                    body=body,
                    redirect_count=redirect_count,
                )
        except HTTPError as exc:
            location = exc.headers.get("Location") if exc.headers else None
            if exc.code in {301, 302, 303, 307, 308} and location:
                if redirect_count >= int(policy["maximum_redirects"]):
                    raise MechanismSourceQueueError("metadata redirect limit exceeded") from exc
                next_url = urljoin(current_url, location)
                if not _host_allowed(next_url, suffixes):
                    raise MechanismSourceQueueError("metadata redirect host is not allowlisted") from exc
                current_url = next_url
                continue
            raise MechanismSourceQueueError("metadata request failed") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise MechanismSourceQueueError("metadata transport failed") from exc
    raise MechanismSourceQueueError("metadata redirect limit exceeded")


def _request_uid(query_uid: str, request_kind: str, url: str) -> str:
    return "metadata-request-" + hashlib.sha256(
        f"{query_uid}\n{request_kind}\n{url}".encode()
    ).hexdigest()[:20]


def _perform_request(
    *,
    query_uid: str,
    request_kind: str,
    url: str,
    headers: Mapping[str, str],
    policy: Mapping[str, Any],
    fetcher: Callable[[Mapping[str, Any], Mapping[str, Any]], FetchResult],
    now: Callable[[], str],
    receipts: list[dict[str, Any]],
    total_bytes: list[int],
) -> tuple[FetchResult | None, str]:
    if len(receipts) >= int(policy["maximum_requests"]):
        raise MechanismSourceQueueError("metadata request budget exceeded")
    if not _host_allowed(url, policy["allowed_host_suffixes"]):
        raise MechanismSourceQueueError("unregistered metadata endpoint")
    request_uid = _request_uid(query_uid, request_kind, url)
    started_at = now()
    try:
        response = fetcher({"url": url, "method": "GET", "headers": dict(headers)}, policy)
        if response.requested_url != url or response.status_code != 200:
            raise MechanismSourceQueueError("metadata response request identity or status mismatch")
        if response.redirect_count < 0 or response.redirect_count > int(policy["maximum_redirects"]):
            raise MechanismSourceQueueError("metadata redirect limit exceeded")
        if not _host_allowed(response.final_url, policy["allowed_host_suffixes"]):
            raise MechanismSourceQueueError("metadata final host is not allowlisted")
        content_type = response.content_type.split(";", 1)[0].strip().lower()
        if content_type not in {"application/json", "text/json"}:
            raise MechanismSourceQueueError("unexpected metadata content type")
        if len(response.body) > int(policy["maximum_response_bytes"]):
            raise MechanismSourceQueueError("metadata response exceeds byte limit")
        if total_bytes[0] + len(response.body) > int(policy["maximum_total_response_bytes"]):
            raise MechanismSourceQueueError("metadata total response budget exceeded")
        total_bytes[0] += len(response.body)
        receipt = {
            "request_uid": request_uid,
            "query_uid": query_uid,
            "request_kind": request_kind,
            "method": "GET",
            "url": url,
            "host": str(urlparse(url).hostname),
            "started_at": started_at,
            "completed_at": now(),
            "http_status": response.status_code,
            "content_type": content_type,
            "response_bytes": len(response.body),
            "response_sha256": sha256_bytes(response.body),
            "status": "success",
        }
        receipts.append(receipt)
        return response, request_uid
    except MechanismSourceQueueError:
        receipts.append(
            {
                "request_uid": request_uid,
                "query_uid": query_uid,
                "request_kind": request_kind,
                "method": "GET",
                "url": url,
                "host": str(urlparse(url).hostname),
                "started_at": started_at,
                "completed_at": now(),
                "http_status": None,
                "content_type": None,
                "response_bytes": 0,
                "response_sha256": None,
                "status": "request_failed",
            }
        )
        return None, request_uid


def _parse_search_ids(body: bytes, maximum_records: int) -> tuple[list[str], int]:
    try:
        payload = _mapping(json.loads(body.decode("utf-8")), label="PubMed search metadata")
        search = _mapping(payload.get("esearchresult"), label="PubMed search result")
        values = _sequence(search.get("idlist"), label="PubMed id list")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MechanismSourceQueueError("PubMed search metadata is malformed") from exc
    identifiers = sorted({str(value) for value in values if re.fullmatch(r"[0-9]+", str(value))}, key=int)
    if len(identifiers) != len(values):
        raise MechanismSourceQueueError("PubMed search returned invalid or duplicate identifiers")
    return identifiers[:maximum_records], max(0, len(identifiers) - maximum_records)


def _parse_pubmed_date(value: object) -> str | None:
    match = re.fullmatch(r"(\d{4})/(\d{2})/(\d{2})(?: \d{2}:\d{2})?", str(value))
    if not match:
        return None
    return "-".join(match.groups()) + "T00:00:00Z"


def _parse_summary_records(
    body: bytes, requested_ids: Sequence[str], response_hash: str, response_bytes: int
) -> list[dict[str, Any]]:
    try:
        payload = _mapping(json.loads(body.decode("utf-8")), label="PubMed summary metadata")
        result = _mapping(payload.get("result"), label="PubMed summary result")
        uids = _sequence(result.get("uids"), label="PubMed summary UID list")
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MechanismSourceQueueError("PubMed summary metadata is malformed") from exc
    if sorted(str(value) for value in uids) != sorted(requested_ids):
        raise MechanismSourceQueueError("PubMed summary identifiers drifted")
    records: list[dict[str, Any]] = []
    for requested_id in requested_ids:
        row = _mapping(result.get(requested_id), label="PubMed summary row")
        if str(row.get("uid")) != requested_id:
            raise MechanismSourceQueueError("PubMed summary identifier mismatch")
        doi: str | None = None
        for identifier in _sequence(row.get("articleids", []), label="PubMed article identities"):
            identity = _mapping(identifier, label="PubMed article identity")
            if identity.get("idtype") == "doi" and isinstance(identity.get("value"), str):
                normalized = identity["value"].strip().lower()
                if normalized:
                    doi = normalized
        official_date = _parse_pubmed_date(row.get("sortpubdate"))
        canonical_identity = f"doi:{doi}" if doi else f"pmid:{requested_id}"
        projected = {
            "identifier_type": "doi" if doi else "pmid",
            "identifier_value": doi if doi else requested_id,
            "pmid": requested_id,
            "doi": doi,
            "official_date": official_date,
            "canonical_source_identity": canonical_identity,
            "public_access_status": "public_metadata",
            "license_status": "public_metadata_terms",
        }
        records.append(
            {
                **projected,
                "metadata_sha256": canonical_value_sha256(projected),
                "response_sha256": response_hash,
                "response_bytes": response_bytes,
            }
        )
    return records


def _placeholder_slot(
    *,
    projection: Mapping[str, Any],
    family: Mapping[str, Any],
    ordinal: int,
    disposition: str,
    query_uid: str | None,
    request_uids: Sequence[str],
) -> dict[str, Any]:
    identity = (
        f"{disposition}:{family['family_uid']}:{projection['intervention_program_uid']}:{ordinal:02d}"
    )
    return {
        "canonical_intervention_uid": projection["intervention_program_uid"],
        "validation_uid": projection["validation_uid"],
        "family_uid": family["family_uid"],
        "family_priority": family["priority"],
        "family_slot_ordinal": ordinal,
        "mechanism_relevance_tags": list(MECHANISM_TAGS),
        "canonical_source_identity": identity,
        "source_type": family["source_type"],
        "official_date": None,
        "disposition": disposition,
        "duplicate_of": None,
        "query_uids": [query_uid] if query_uid else [],
        "request_uids": list(request_uids),
        "metadata": {
            "identifier_type": "unavailable",
            "identifier_value": None,
            "pmid": None,
            "doi": None,
            "public_access_status": "not_checked",
            "license_status": "not_checked",
            "metadata_sha256": None,
            "response_sha256": None,
            "response_bytes": None,
        },
    }


def _identified_placeholder_slot(
    *,
    projection: Mapping[str, Any],
    family: Mapping[str, Any],
    ordinal: int,
    disposition: str,
    query_uid: str,
    request_uids: Sequence[str],
    pmid: str,
    summary_request: Mapping[str, Any],
) -> dict[str, Any]:
    slot = _placeholder_slot(
        projection=projection,
        family=family,
        ordinal=ordinal,
        disposition=disposition,
        query_uid=query_uid,
        request_uids=request_uids,
    )
    canonical_identity = f"pmid:{pmid}"
    projected = {
        "identifier_type": "pmid",
        "identifier_value": pmid,
        "pmid": pmid,
        "doi": None,
        "official_date": None,
        "canonical_source_identity": canonical_identity,
        "public_access_status": "not_checked",
        "license_status": "not_checked",
    }
    slot["canonical_source_identity"] = canonical_identity
    slot["metadata"] = {
        key: projected[key]
        for key in (
            "identifier_type",
            "identifier_value",
            "pmid",
            "doi",
            "public_access_status",
            "license_status",
        )
    }
    slot["metadata"].update(
        {
            "metadata_sha256": canonical_value_sha256(projected),
            "response_sha256": summary_request["response_sha256"],
            "response_bytes": summary_request["response_bytes"],
        }
    )
    return slot


def _publication_slots(
    *,
    projection: Mapping[str, Any],
    family: Mapping[str, Any],
    query: Mapping[str, Any],
    fetcher: Callable[[Mapping[str, Any], Mapping[str, Any]], FetchResult],
    now: Callable[[], str],
    sleeper: Callable[[float], None],
    receipts: list[dict[str, Any]],
    total_bytes: list[int],
) -> tuple[list[dict[str, Any]], int]:
    headers = {"Accept": "application/json", "User-Agent": "NeuronSifter-MechanismSourceQueue/1.0"}
    if receipts:
        sleeper(REQUEST_POLICY["minimum_request_interval_ms"] / 1000.0)
    search_response, search_request_uid = _perform_request(
        query_uid=str(query["query_uid"]),
        request_kind="pubmed_search",
        url=str(query["search_url"]),
        headers=headers,
        policy=REQUEST_POLICY,
        fetcher=fetcher,
        now=now,
        receipts=receipts,
        total_bytes=total_bytes,
    )
    quota = int(family["slot_quota_per_intervention"])
    if search_response is None:
        return [
            _placeholder_slot(
                projection=projection,
                family=family,
                ordinal=index,
                disposition="request_failed",
                query_uid=str(query["query_uid"]),
                request_uids=[search_request_uid],
            )
            for index in range(1, quota + 1)
        ], 0
    try:
        pmids, overflow_count = _parse_search_ids(search_response.body, quota)
    except MechanismSourceQueueError:
        return [
            _placeholder_slot(
                projection=projection,
                family=family,
                ordinal=index,
                disposition="forbidden_metadata",
                query_uid=str(query["query_uid"]),
                request_uids=[search_request_uid],
            )
            for index in range(1, quota + 1)
        ], 0
    if not pmids:
        return [
            _placeholder_slot(
                projection=projection,
                family=family,
                ordinal=index,
                disposition="unavailable",
                query_uid=str(query["query_uid"]),
                request_uids=[search_request_uid],
            )
            for index in range(1, quota + 1)
        ], overflow_count
    summary_url = str(query["summary_url_template"]).format(pmid_csv=",".join(pmids))
    sleeper(REQUEST_POLICY["minimum_request_interval_ms"] / 1000.0)
    summary_response, summary_request_uid = _perform_request(
        query_uid=str(query["query_uid"]),
        request_kind="pubmed_summary",
        url=summary_url,
        headers=headers,
        policy=REQUEST_POLICY,
        fetcher=fetcher,
        now=now,
        receipts=receipts,
        total_bytes=total_bytes,
    )
    if summary_response is None:
        slots = [
            _identified_placeholder_slot(
                projection=projection,
                family=family,
                ordinal=index,
                disposition="request_failed",
                query_uid=str(query["query_uid"]),
                request_uids=[search_request_uid, summary_request_uid],
                pmid=pmid,
                summary_request=receipts[-1],
            )
            for index, pmid in enumerate(pmids, start=1)
        ]
    else:
        try:
            metadata_records = _parse_summary_records(
                summary_response.body,
                pmids,
                sha256_bytes(summary_response.body),
                len(summary_response.body),
            )
        except MechanismSourceQueueError:
            metadata_records = []
            slots = [
                _identified_placeholder_slot(
                    projection=projection,
                    family=family,
                    ordinal=index,
                    disposition="forbidden_metadata",
                    query_uid=str(query["query_uid"]),
                    request_uids=[search_request_uid, summary_request_uid],
                    pmid=pmid,
                    summary_request=receipts[-1],
                )
                for index, pmid in enumerate(pmids, start=1)
            ]
        else:
            slots = []
            for index, metadata in enumerate(
                sorted(
                    metadata_records,
                    key=lambda row: (
                        row["official_date"] or "9999-12-31T00:00:00Z",
                        row["canonical_source_identity"],
                        int(row["pmid"]),
                    ),
                ),
                start=1,
            ):
                disposition = "eligible"
                if metadata["official_date"] is None:
                    disposition = "source_date_ambiguous"
                elif metadata["official_date"] > CUTOFF:
                    disposition = "cutoff_ineligible_release"
                slot = {
                    "canonical_intervention_uid": projection["intervention_program_uid"],
                    "validation_uid": projection["validation_uid"],
                    "family_uid": family["family_uid"],
                    "family_priority": family["priority"],
                    "family_slot_ordinal": index,
                    "mechanism_relevance_tags": list(MECHANISM_TAGS),
                    "canonical_source_identity": metadata["canonical_source_identity"],
                    "source_type": family["source_type"],
                    "official_date": metadata["official_date"],
                    "disposition": disposition,
                    "duplicate_of": None,
                    "query_uids": [query["query_uid"]],
                    "request_uids": [search_request_uid, summary_request_uid],
                    "metadata": {
                        key: metadata[key]
                        for key in (
                            "identifier_type",
                            "identifier_value",
                            "pmid",
                            "doi",
                            "public_access_status",
                            "license_status",
                            "metadata_sha256",
                            "response_sha256",
                            "response_bytes",
                        )
                    },
                }
                slots.append(slot)
    while len(slots) < quota:
        slots.append(
            _placeholder_slot(
                projection=projection,
                family=family,
                ordinal=len(slots) + 1,
                disposition="unavailable",
                query_uid=str(query["query_uid"]),
                request_uids=[search_request_uid],
            )
        )
    return slots, overflow_count


def _slot_sort_key(slot: Mapping[str, Any]) -> tuple[Any, ...]:
    tag_priority = min(MECHANISM_TAGS.index(tag) for tag in slot["mechanism_relevance_tags"])
    return (
        slot["canonical_intervention_uid"],
        tag_priority,
        slot["family_priority"],
        slot["official_date"] or "9999-12-31T00:00:00Z",
        slot["source_type"],
        slot["canonical_source_identity"],
        slot["family_slot_ordinal"],
    )


def _finalize_slots(slots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = [dict(slot) for slot in sorted(slots, key=_slot_sort_key)]
    seen: dict[tuple[str, str], str] = {}
    for queue_order, slot in enumerate(ordered, start=1):
        slot_uid = "mechanism-source-slot-" + hashlib.sha256(
            canonical_json_bytes(
                {
                    "intervention": slot["canonical_intervention_uid"],
                    "family": slot["family_uid"],
                    "identity": slot["canonical_source_identity"],
                    "ordinal": slot["family_slot_ordinal"],
                    "queries": slot["query_uids"],
                }
            )
        ).hexdigest()[:20]
        slot["slot_uid"] = slot_uid
        slot["queue_order"] = queue_order
        key = (slot["canonical_intervention_uid"], slot["canonical_source_identity"])
        if slot["disposition"] == "eligible" and key in seen:
            slot["disposition"] = "duplicate"
            slot["duplicate_of"] = seen[key]
        elif slot["disposition"] == "eligible":
            seen[key] = slot_uid
    return ordered


def _artifact_binding(path: str, raw: bytes, semantic_sha256: str) -> dict[str, str]:
    return {"path": path, "file_sha256": sha256_bytes(raw), "semantic_sha256": semantic_sha256}


def materialize_queue(
    *,
    repository_root: Path,
    contract_path: Path | str = CONTRACT_PATH,
    fetcher: Callable[[Mapping[str, Any], Mapping[str, Any]], FetchResult] = fetch_metadata,
    now: Callable[[], str] = _now,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    root = repository_root.resolve(strict=True)
    contract, contract_raw = validate_contract(root, contract_path)
    _artifact_paths(contract)
    preregistration = validate_preregistration(repository_root=root, contract_path=contract_path)
    prereg_path = _resolve_file(root, PREREGISTRATION_PATH, must_exist=True, label="preregistration")
    prereg_raw = prereg_path.read_bytes()
    output_paths = [root / QUEUE_PATH, root / CHRONOLOGY_PATH, root / AUTHORIZATION_PATH]
    if any(path.exists() for path in output_paths):
        queue, chronology, authorization = validate_frozen_queue(
            repository_root=root, contract_path=contract_path
        )
        return queue, chronology, authorization, False

    families = {row["family_uid"]: row for row in preregistration["source_families"]}
    queries = {
        row["canonical_intervention_uid"]: row
        for row in preregistration["expanded_requests"]
    }
    slots: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    total_bytes = [0]
    overflow_count = 0
    for projection in preregistration["context_action_projections"]:
        publication_slots, overflow = _publication_slots(
            projection=projection,
            family=families["publication_metadata"],
            query=queries[projection["intervention_program_uid"]],
            fetcher=fetcher,
            now=now,
            sleeper=sleeper,
            receipts=receipts,
            total_bytes=total_bytes,
        )
        slots.extend(publication_slots)
        overflow_count += overflow
        for family in preregistration["source_families"][1:]:
            for ordinal in range(1, int(family["slot_quota_per_intervention"]) + 1):
                slots.append(
                    _placeholder_slot(
                        projection=projection,
                        family=family,
                        ordinal=ordinal,
                        disposition=str(family["preregistered_disposition"]),
                        query_uid=None,
                        request_uids=[],
                    )
                )
    final_slots = _finalize_slots(slots)
    counts = Counter(slot["disposition"] for slot in final_slots)
    queue: dict[str, Any] = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "status": "MECHANISM_SOURCE_QUEUE_FROZEN",
        "queue_uid": "mechanism_source_queue",
        "preregistration": _artifact_binding(
            PREREGISTRATION_PATH, prereg_raw, preregistration["preregistration_sha256"]
        ),
        "contract": {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)},
        "schema": {
            "path": QUEUE_SCHEMA_PATH,
            "sha256": EXPECTED_SCHEMA_SHA256[QUEUE_SCHEMA_PATH],
        },
        "cutoff": CUTOFF,
        "slots": final_slots,
        "counts": {
            "slot_count": len(final_slots),
            "intervention_count": 6,
            "family_count": 6,
            "dispositions": {name: counts.get(name, 0) for name in DISPOSITIONS},
        },
        "network_summary": {
            "request_count": len(receipts),
            "successful_request_count": sum(row["status"] == "success" for row in receipts),
            "failed_request_count": sum(row["status"] == "request_failed" for row in receipts),
            "overflow_record_count": overflow_count,
            "total_response_bytes": total_bytes[0],
            "raw_responses_persisted": False,
            "queried_families": ["publication_metadata"] if receipts else [],
        },
        "access_counters": deepcopy(ZERO_COUNTERS),
        "no_refill_after_exclusion": True,
        "content_access_authorized": False,
        "blockers": [],
    }
    queue["queue_sha256"] = _self_hash(queue, "queue_sha256")
    queue_raw = canonical_json_bytes(queue)

    chronology: dict[str, Any] = {
        "schema_version": CHRONOLOGY_SCHEMA_VERSION,
        "status": "MECHANISM_SOURCE_QUEUE_CHRONOLOGY_VALID",
        "chronology_uid": "mechanism_source_queue_chronology",
        "preregistration": _artifact_binding(
            PREREGISTRATION_PATH, prereg_raw, preregistration["preregistration_sha256"]
        ),
        "queue": _artifact_binding(QUEUE_PATH, queue_raw, queue["queue_sha256"]),
        "schema": {
            "path": CHRONOLOGY_SCHEMA_PATH,
            "sha256": EXPECTED_SCHEMA_SHA256[CHRONOLOGY_SCHEMA_PATH],
        },
        "preregistered_at": preregistration["preregistered_at"],
        "first_request_at": receipts[0]["started_at"] if receipts else None,
        "last_request_at": receipts[-1]["completed_at"] if receipts else None,
        "requests": receipts,
        "proofs": {
            "preregistration_preceded_requests": True,
            "only_preregistered_publication_metadata_queried": True,
            "serial_request_order": True,
            "request_budget_respected": True,
            "response_budget_respected": True,
            "queue_order_allowlisted_only": True,
            "slot_caps_respected": True,
            "excluded_slots_retained": True,
            "no_post_content_mutation": True,
            "no_refill": True,
            "raw_responses_persisted": False,
        },
        "access_counters": deepcopy(ZERO_COUNTERS),
        "blockers": [],
    }
    chronology["chronology_sha256"] = _self_hash(chronology, "chronology_sha256")
    chronology_raw = canonical_json_bytes(chronology)

    authorization: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": "MECHANISM_SOURCE_CONTENT_NOT_AUTHORIZED",
        "authorization_uid": "mechanism_source_content_authorization",
        "preregistration": _artifact_binding(
            PREREGISTRATION_PATH, prereg_raw, preregistration["preregistration_sha256"]
        ),
        "queue": _artifact_binding(QUEUE_PATH, queue_raw, queue["queue_sha256"]),
        "chronology": _artifact_binding(
            CHRONOLOGY_PATH, chronology_raw, chronology["chronology_sha256"]
        ),
        "schema": {
            "path": AUTHORIZATION_SCHEMA_PATH,
            "sha256": EXPECTED_SCHEMA_SHA256[AUTHORIZATION_SCHEMA_PATH],
        },
        "content_access_authorized": False,
        "access_counters": deepcopy(ZERO_COUNTERS),
        "downstream_authorization": deepcopy(DOWNSTREAM_AUTHORIZATION),
        "blockers": [],
    }
    authorization["authorization_sha256"] = _self_hash(
        authorization, "authorization_sha256"
    )
    authorization_raw = canonical_json_bytes(authorization)

    for payload, schema_path, label in (
        (queue, QUEUE_SCHEMA_PATH, "queue"),
        (chronology, CHRONOLOGY_SCHEMA_PATH, "chronology"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "content authorization"),
    ):
        schema, _ = _schema(root, schema_path)
        _schema_validate(payload, schema, label=label)
        _validate_no_forbidden_fields(payload)
    created = _install_many(
        [
            (root / QUEUE_PATH, queue_raw),
            (root / CHRONOLOGY_PATH, chronology_raw),
            (root / AUTHORIZATION_PATH, authorization_raw),
        ]
    )
    queue, chronology, authorization = validate_frozen_queue(
        repository_root=root, contract_path=contract_path
    )
    return queue, chronology, authorization, created


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_queue_derived_counts(
    queue: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> None:
    slots = _sequence(queue.get("slots"), label="queue slots")
    counts = _mapping(queue.get("counts"), label="queue counts")
    actual_slot_count = len(slots)
    if actual_slot_count != 72 or actual_slot_count > QUEUE_POLICY["global_slot_cap"]:
        raise MechanismSourceQueueError("frozen queue actual slot count drift")

    expected_interventions = {
        row["intervention_program_uid"]
        for row in preregistration["context_action_projections"]
    }
    intervention_counts = Counter(slot["canonical_intervention_uid"] for slot in slots)
    if set(intervention_counts) != expected_interventions:
        raise MechanismSourceQueueError("queue intervention roster drift")
    if any(
        count != QUEUE_POLICY["exact_slots_per_intervention"]
        or count > QUEUE_POLICY["per_intervention_slot_cap"]
        for count in intervention_counts.values()
    ):
        raise MechanismSourceQueueError("per-intervention slot count or cap drift")

    family_quotas = {
        row["family_uid"]: int(row["slot_quota_per_intervention"])
        for row in preregistration["source_families"]
    }
    if len(family_quotas) != 6 or len(family_quotas) > QUEUE_POLICY["family_cap"]:
        raise MechanismSourceQueueError("queue family count or cap drift")
    family_counts = Counter(slot["family_uid"] for slot in slots)
    expected_family_counts = {
        family_uid: quota * len(expected_interventions)
        for family_uid, quota in family_quotas.items()
    }
    if family_counts != expected_family_counts:
        raise MechanismSourceQueueError("family slot quota drift")
    for intervention_uid in expected_interventions:
        for family_uid, quota in family_quotas.items():
            ordinals = sorted(
                slot["family_slot_ordinal"]
                for slot in slots
                if slot["canonical_intervention_uid"] == intervention_uid
                and slot["family_uid"] == family_uid
            )
            if ordinals != list(range(1, quota + 1)):
                raise MechanismSourceQueueError("family slot ordinal or no-refill drift")

    observed_dispositions = Counter(slot["disposition"] for slot in slots)
    expected_dispositions = {
        name: observed_dispositions.get(name, 0) for name in DISPOSITIONS
    }
    expected_counts = {
        "slot_count": actual_slot_count,
        "intervention_count": len(intervention_counts),
        "family_count": len(family_counts),
        "dispositions": expected_dispositions,
    }
    if counts != expected_counts:
        raise MechanismSourceQueueError("declared queue counts do not match slots")
    if sum(expected_dispositions.values()) != actual_slot_count:
        raise MechanismSourceQueueError("disposition totals do not match slots")


def _validate_disabled_family_placeholders(
    preregistration: Mapping[str, Any], queue: Mapping[str, Any]
) -> None:
    observed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for slot in queue["slots"]:
        if slot["family_uid"] == "publication_metadata":
            continue
        key = (
            str(slot["canonical_intervention_uid"]),
            str(slot["family_uid"]),
            int(slot["family_slot_ordinal"]),
        )
        if key in observed:
            raise MechanismSourceQueueError("duplicate disabled-family slot identity")
        observed[key] = slot

    expected_keys: set[tuple[str, str, int]] = set()
    for projection in preregistration["context_action_projections"]:
        for family in preregistration["source_families"]:
            if family["family_uid"] == "publication_metadata":
                continue
            for ordinal in range(1, int(family["slot_quota_per_intervention"]) + 1):
                key = (
                    str(projection["intervention_program_uid"]),
                    str(family["family_uid"]),
                    ordinal,
                )
                expected_keys.add(key)
                slot = observed.get(key)
                if slot is None:
                    raise MechanismSourceQueueError("missing disabled-family placeholder slot")
                expected = _placeholder_slot(
                    projection=projection,
                    family=family,
                    ordinal=ordinal,
                    disposition=str(family["preregistered_disposition"]),
                    query_uid=None,
                    request_uids=[],
                )
                observed_semantics = {
                    field: slot.get(field) for field in expected
                }
                if observed_semantics != expected:
                    raise MechanismSourceQueueError(
                        "disabled-family placeholder metadata or identity drift"
                    )
                if slot.get("duplicate_of") is not None:
                    raise MechanismSourceQueueError(
                        "disabled-family placeholder duplicate lineage drift"
                    )
    if set(observed) != expected_keys:
        raise MechanismSourceQueueError("unexpected disabled-family placeholder slot")


def _publication_source_order_key(slot: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata = _mapping(slot.get("metadata"), label="publication slot metadata")
    pmid = metadata.get("pmid")
    return (
        slot.get("official_date") or "9999-12-31T00:00:00Z",
        slot.get("canonical_source_identity"),
        int(pmid) if isinstance(pmid, str) and re.fullmatch(r"[0-9]+", pmid) else 0,
    )


def _validate_common_slot_bindings(
    preregistration: Mapping[str, Any], queue: Mapping[str, Any]
) -> None:
    projections = {
        str(row["intervention_program_uid"]): row
        for row in preregistration["context_action_projections"]
    }
    families = {
        str(row["family_uid"]): row for row in preregistration["source_families"]
    }
    queries = {
        str(row["canonical_intervention_uid"]): row
        for row in preregistration["expanded_requests"]
    }
    slots = _sequence(queue.get("slots"), label="queue slots")
    if [slot["queue_order"] for slot in slots] != list(range(1, len(slots) + 1)):
        raise MechanismSourceQueueError("queue order is not contiguous")
    if slots != sorted(slots, key=_slot_sort_key):
        raise MechanismSourceQueueError("queue ordering drift")

    publication_by_intervention: dict[str, list[Mapping[str, Any]]] = {}
    for slot in slots:
        intervention_uid = str(slot.get("canonical_intervention_uid"))
        family_uid = str(slot.get("family_uid"))
        projection = projections.get(intervention_uid)
        family = families.get(family_uid)
        if projection is None or family is None:
            raise MechanismSourceQueueError("slot references an unregistered context or family")
        if (
            slot.get("validation_uid") != projection["validation_uid"]
            or slot.get("family_priority") != family["priority"]
            or slot.get("source_type") != family["source_type"]
            or slot.get("mechanism_relevance_tags") != list(MECHANISM_TAGS)
        ):
            raise MechanismSourceQueueError("slot context or family binding drift")
        ordinal = slot.get("family_slot_ordinal")
        if not isinstance(ordinal, int) or not 1 <= ordinal <= int(
            family["slot_quota_per_intervention"]
        ):
            raise MechanismSourceQueueError("slot family ordinal drift")
        if family_uid == "publication_metadata":
            query = queries.get(intervention_uid)
            if query is None or slot.get("query_uids") != [query["query_uid"]]:
                raise MechanismSourceQueueError("publication slot query binding drift")
            search_uid = _request_uid(
                str(query["query_uid"]), "pubmed_search", str(query["search_url"])
            )
            request_uids = slot.get("request_uids")
            if (
                not isinstance(request_uids, list)
                or len(request_uids) not in {1, 2}
                or request_uids[0] != search_uid
                or len(set(request_uids)) != len(request_uids)
            ):
                raise MechanismSourceQueueError("publication slot request binding drift")
            publication_by_intervention.setdefault(intervention_uid, []).append(slot)
        elif slot.get("query_uids") != [] or slot.get("request_uids") != []:
            raise MechanismSourceQueueError("disabled-family slot has request lineage")

    if set(publication_by_intervention) != set(projections):
        raise MechanismSourceQueueError("publication intervention roster drift")
    for intervention_uid, publication_slots in publication_by_intervention.items():
        query = queries[intervention_uid]
        search_uid = _request_uid(
            str(query["query_uid"]), "pubmed_search", str(query["search_url"])
        )
        pmids = sorted(
            {
                str(_mapping(slot.get("metadata"), label="publication slot metadata")["pmid"])
                for slot in publication_slots
                if _mapping(slot.get("metadata"), label="publication slot metadata").get(
                    "pmid"
                )
                is not None
            },
            key=int,
        )
        summary_uid = None
        if pmids:
            summary_url = str(query["summary_url_template"]).format(
                pmid_csv=",".join(pmids)
            )
            summary_uid = _request_uid(
                str(query["query_uid"]), "pubmed_summary", summary_url
            )
        for slot in publication_slots:
            request_uids = slot["request_uids"]
            expected_request_uids = [search_uid]
            if len(request_uids) == 2:
                if summary_uid is None:
                    raise MechanismSourceQueueError(
                        "publication slot request binding drift"
                    )
                expected_request_uids.append(summary_uid)
            if request_uids != expected_request_uids:
                raise MechanismSourceQueueError(
                    "publication slot request binding drift"
                )
        ordered = sorted(publication_slots, key=_publication_source_order_key)
        for expected_ordinal, slot in enumerate(ordered, start=1):
            if slot["family_slot_ordinal"] != expected_ordinal:
                raise MechanismSourceQueueError(
                    "publication source-to-ordinal assignment drift"
                )


def _validate_request_receipt_identity(
    request: Mapping[str, Any],
    *,
    query_uid: str,
    request_kind: str,
    url: str,
) -> str:
    expected_uid = _request_uid(query_uid, request_kind, url)
    expected_identity = {
        "request_uid": expected_uid,
        "query_uid": query_uid,
        "request_kind": request_kind,
        "method": "GET",
        "url": url,
        "host": "eutils.ncbi.nlm.nih.gov",
    }
    observed_identity = {key: request.get(key) for key in expected_identity}
    if observed_identity != expected_identity:
        raise MechanismSourceQueueError("request identity, type, sequence, or URL drift")
    status = request.get("status")
    if status == "success":
        if (
            request.get("http_status") != 200
            or request.get("content_type") not in {"application/json", "text/json"}
            or not isinstance(request.get("response_bytes"), int)
            or request["response_bytes"] <= 0
            or not isinstance(request.get("response_sha256"), str)
        ):
            raise MechanismSourceQueueError("successful request receipt metadata drift")
    elif status == "request_failed":
        if any(
            request.get(key) is not None
            for key in ("http_status", "content_type", "response_sha256")
        ) or request.get("response_bytes") != 0:
            raise MechanismSourceQueueError("failed request receipt metadata drift")
    else:
        raise MechanismSourceQueueError("request receipt status drift")
    return expected_uid


def _validate_publication_slot_metadata(
    slot: Mapping[str, Any], summary_request: Mapping[str, Any]
) -> None:
    metadata = _mapping(slot.get("metadata"), label="publication slot metadata")
    pmid = metadata.get("pmid")
    if not isinstance(pmid, str) or not re.fullmatch(r"[0-9]+", pmid):
        raise MechanismSourceQueueError("publication slot PMID commitment drift")
    doi = metadata.get("doi")
    if doi is not None and (not isinstance(doi, str) or doi != doi.strip().lower() or not doi):
        raise MechanismSourceQueueError("publication slot DOI normalization drift")
    expected_identity = f"doi:{doi}" if doi else f"pmid:{pmid}"
    expected_identifier_type = "doi" if doi else "pmid"
    expected_identifier_value = doi if doi else pmid
    if (
        slot.get("canonical_source_identity") != expected_identity
        or metadata.get("identifier_type") != expected_identifier_type
        or metadata.get("identifier_value") != expected_identifier_value
    ):
        raise MechanismSourceQueueError("publication slot canonical identity drift")
    projected = {
        "identifier_type": expected_identifier_type,
        "identifier_value": expected_identifier_value,
        "pmid": pmid,
        "doi": doi,
        "official_date": slot.get("official_date"),
        "canonical_source_identity": expected_identity,
        "public_access_status": metadata.get("public_access_status"),
        "license_status": metadata.get("license_status"),
    }
    if metadata.get("metadata_sha256") != canonical_value_sha256(projected):
        raise MechanismSourceQueueError("publication slot metadata identity drift")
    if (
        metadata.get("response_sha256") != summary_request.get("response_sha256")
        or metadata.get("response_bytes") != summary_request.get("response_bytes")
    ):
        raise MechanismSourceQueueError("publication slot response binding drift")


def _validate_unidentified_publication_placeholder(
    slot: Mapping[str, Any], *, disposition: str, request_uids: Sequence[str]
) -> None:
    expected_identity = (
        f"{disposition}:publication_metadata:"
        f"{slot['canonical_intervention_uid']}:{slot['family_slot_ordinal']:02d}"
    )
    expected_metadata = {
        "identifier_type": "unavailable",
        "identifier_value": None,
        "pmid": None,
        "doi": None,
        "public_access_status": "not_checked",
        "license_status": "not_checked",
        "metadata_sha256": None,
        "response_sha256": None,
        "response_bytes": None,
    }
    if (
        slot.get("canonical_source_identity") != expected_identity
        or slot.get("official_date") is not None
        or slot.get("disposition") != disposition
        or slot.get("duplicate_of") is not None
        or slot.get("request_uids") != list(request_uids)
        or slot.get("metadata") != expected_metadata
    ):
        raise MechanismSourceQueueError("publication placeholder metadata or identity drift")


def _validate_identified_publication_failure(
    slot: Mapping[str, Any],
    *,
    disposition: str,
    request_uids: Sequence[str],
    summary_request: Mapping[str, Any],
) -> None:
    metadata = _mapping(slot.get("metadata"), label="failed publication metadata")
    pmid = metadata.get("pmid")
    if not isinstance(pmid, str) or not re.fullmatch(r"[0-9]+", pmid):
        raise MechanismSourceQueueError("failed publication PMID commitment drift")
    projected = {
        "identifier_type": "pmid",
        "identifier_value": pmid,
        "pmid": pmid,
        "doi": None,
        "official_date": None,
        "canonical_source_identity": f"pmid:{pmid}",
        "public_access_status": "not_checked",
        "license_status": "not_checked",
    }
    expected_metadata = {
        key: projected[key]
        for key in (
            "identifier_type",
            "identifier_value",
            "pmid",
            "doi",
            "public_access_status",
            "license_status",
        )
    }
    expected_metadata.update(
        {
            "metadata_sha256": canonical_value_sha256(projected),
            "response_sha256": summary_request["response_sha256"],
            "response_bytes": summary_request["response_bytes"],
        }
    )
    if (
        slot.get("canonical_source_identity") != projected["canonical_source_identity"]
        or slot.get("official_date") is not None
        or slot.get("disposition") != disposition
        or slot.get("duplicate_of") is not None
        or slot.get("request_uids") != list(request_uids)
        or metadata != expected_metadata
    ):
        raise MechanismSourceQueueError("failed publication metadata or identity drift")


def _validate_exact_request_chronology(
    preregistration: Mapping[str, Any],
    queue: Mapping[str, Any],
    chronology: Mapping[str, Any],
) -> None:
    requests = _sequence(chronology.get("requests"), label="chronology requests")
    publication_slots = [
        slot for slot in queue["slots"] if slot["family_uid"] == "publication_metadata"
    ]
    cursor = 0
    for query in preregistration["expanded_requests"]:
        query_uid = str(query["query_uid"])
        intervention_uid = str(query["canonical_intervention_uid"])
        slots = [
            slot
            for slot in publication_slots
            if slot["canonical_intervention_uid"] == intervention_uid
        ]
        if len(slots) != 4 or any(slot["query_uids"] != [query_uid] for slot in slots):
            raise MechanismSourceQueueError("publication slot query lineage drift")
        if cursor >= len(requests):
            raise MechanismSourceQueueError("missing preregistered ESearch receipt")
        search_request = requests[cursor]
        search_uid = _validate_request_receipt_identity(
            search_request,
            query_uid=query_uid,
            request_kind="pubmed_search",
            url=str(query["search_url"]),
        )
        cursor += 1
        pmids = sorted(
            {
                str(slot["metadata"]["pmid"])
                for slot in slots
                if slot["metadata"]["pmid"] is not None
            },
            key=int,
        )
        if search_request["status"] == "request_failed":
            if pmids:
                raise MechanismSourceQueueError("failed ESearch slot lineage drift")
            for slot in slots:
                _validate_unidentified_publication_placeholder(
                    slot, disposition="request_failed", request_uids=[search_uid]
                )
            continue
        if not pmids:
            dispositions = {slot["disposition"] for slot in slots}
            if dispositions not in ({"unavailable"}, {"forbidden_metadata"}):
                raise MechanismSourceQueueError("empty ESearch slot lineage drift")
            disposition = next(iter(dispositions))
            for slot in slots:
                _validate_unidentified_publication_placeholder(
                    slot, disposition=disposition, request_uids=[search_uid]
                )
            continue
        if cursor >= len(requests):
            raise MechanismSourceQueueError("missing committed ESummary receipt")
        summary_url = str(query["summary_url_template"]).format(
            pmid_csv=",".join(pmids)
        )
        summary_request = requests[cursor]
        summary_uid = _validate_request_receipt_identity(
            summary_request,
            query_uid=query_uid,
            request_kind="pubmed_summary",
            url=summary_url,
        )
        cursor += 1
        identified_slots = [slot for slot in slots if slot["metadata"]["pmid"] is not None]
        if len(identified_slots) != len(pmids):
            raise MechanismSourceQueueError("publication PMID commitment cardinality drift")
        identified_dispositions = {slot["disposition"] for slot in identified_slots}
        if summary_request["status"] == "request_failed":
            if identified_dispositions != {"request_failed"}:
                raise MechanismSourceQueueError("failed ESummary slot disposition drift")
        elif identified_dispositions == {"forbidden_metadata"}:
            if any(slot["official_date"] is not None for slot in identified_slots):
                raise MechanismSourceQueueError("malformed ESummary exposed an official date")
        elif any(
            slot["metadata"]["public_access_status"] != "public_metadata"
            or slot["metadata"]["license_status"] != "public_metadata_terms"
            for slot in identified_slots
        ):
            raise MechanismSourceQueueError("successful ESummary access metadata drift")
        for slot in slots:
            if slot["metadata"]["pmid"] is None:
                _validate_unidentified_publication_placeholder(
                    slot, disposition="unavailable", request_uids=[search_uid]
                )
                continue
            if summary_request["status"] == "request_failed":
                _validate_identified_publication_failure(
                    slot,
                    disposition="request_failed",
                    request_uids=[search_uid, summary_uid],
                    summary_request=summary_request,
                )
                continue
            if identified_dispositions == {"forbidden_metadata"}:
                _validate_identified_publication_failure(
                    slot,
                    disposition="forbidden_metadata",
                    request_uids=[search_uid, summary_uid],
                    summary_request=summary_request,
                )
                continue
            if slot["request_uids"] != [search_uid, summary_uid]:
                raise MechanismSourceQueueError("publication slot request lineage drift")
            _validate_publication_slot_metadata(slot, summary_request)
            if summary_request["status"] == "success" and identified_dispositions != {"forbidden_metadata"}:
                expected_disposition = "eligible"
                if slot["official_date"] is None:
                    expected_disposition = "source_date_ambiguous"
                elif slot["official_date"] > CUTOFF:
                    expected_disposition = "cutoff_ineligible_release"
                if slot["disposition"] not in {expected_disposition, "duplicate"}:
                    raise MechanismSourceQueueError("successful ESummary slot disposition drift")
    if cursor != len(requests):
        raise MechanismSourceQueueError("extra or duplicate request receipt")


def validate_frozen_queue(
    *, repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = repository_root.resolve(strict=True)
    contract, contract_raw = validate_contract(root, contract_path)
    preregistration = validate_preregistration(repository_root=root, contract_path=contract_path)
    prereg_raw = _resolve_file(root, PREREGISTRATION_PATH, must_exist=True, label="preregistration").read_bytes()
    queue, queue_raw = _canonical_payload(
        _resolve_file(root, QUEUE_PATH, must_exist=True, label="queue"), label="queue"
    )
    chronology, chronology_raw = _canonical_payload(
        _resolve_file(root, CHRONOLOGY_PATH, must_exist=True, label="chronology"),
        label="chronology",
    )
    authorization, _ = _canonical_payload(
        _resolve_file(root, AUTHORIZATION_PATH, must_exist=True, label="content authorization"),
        label="content authorization",
    )
    for payload, schema_path, label in (
        (queue, QUEUE_SCHEMA_PATH, "queue"),
        (chronology, CHRONOLOGY_SCHEMA_PATH, "chronology"),
        (authorization, AUTHORIZATION_SCHEMA_PATH, "content authorization"),
    ):
        schema, _ = _schema(root, schema_path)
        _schema_validate(payload, schema, label=label)
        _validate_no_forbidden_fields(payload)
    if queue["queue_sha256"] != _self_hash(queue, "queue_sha256"):
        raise MechanismSourceQueueError("queue semantic identity drift")
    if chronology["chronology_sha256"] != _self_hash(chronology, "chronology_sha256"):
        raise MechanismSourceQueueError("chronology semantic identity drift")
    if authorization["authorization_sha256"] != _self_hash(
        authorization, "authorization_sha256"
    ):
        raise MechanismSourceQueueError("authorization semantic identity drift")
    prereg_binding = _artifact_binding(
        PREREGISTRATION_PATH, prereg_raw, preregistration["preregistration_sha256"]
    )
    if queue["preregistration"] != prereg_binding or chronology["preregistration"] != prereg_binding or authorization["preregistration"] != prereg_binding:
        raise MechanismSourceQueueError("preregistration cross-binding drift")
    queue_binding = _artifact_binding(QUEUE_PATH, queue_raw, queue["queue_sha256"])
    if chronology["queue"] != queue_binding or authorization["queue"] != queue_binding:
        raise MechanismSourceQueueError("queue cross-binding drift")
    chronology_binding = _artifact_binding(
        CHRONOLOGY_PATH, chronology_raw, chronology["chronology_sha256"]
    )
    if authorization["chronology"] != chronology_binding:
        raise MechanismSourceQueueError("chronology cross-binding drift")
    if queue["contract"] != {"path": CONTRACT_PATH, "sha256": sha256_bytes(contract_raw)}:
        raise MechanismSourceQueueError("queue contract binding drift")
    _validate_queue_derived_counts(queue, preregistration)
    _validate_common_slot_bindings(preregistration, queue)
    _validate_disabled_family_placeholders(preregistration, queue)
    family_by_uid = {row["family_uid"]: row for row in preregistration["source_families"]}
    seen_eligible: dict[tuple[str, str], str] = {}
    request_uid_set = {request["request_uid"] for request in chronology["requests"]}
    for slot in queue["slots"]:
        expected_slot_uid = "mechanism-source-slot-" + hashlib.sha256(
            canonical_json_bytes(
                {
                    "intervention": slot["canonical_intervention_uid"],
                    "family": slot["family_uid"],
                    "identity": slot["canonical_source_identity"],
                    "ordinal": slot["family_slot_ordinal"],
                    "queries": slot["query_uids"],
                }
            )
        ).hexdigest()[:20]
        if slot["slot_uid"] != expected_slot_uid:
            raise MechanismSourceQueueError("slot identity drift")
        if slot["official_date"] is not None and slot["disposition"] == "eligible" and slot["official_date"] > CUTOFF:
            raise MechanismSourceQueueError("post-cutoff slot marked eligible")
        if slot["family_uid"] != "publication_metadata" and slot["request_uids"]:
            raise MechanismSourceQueueError("non-publication family performed a network request")
        if not set(slot["request_uids"]).issubset(request_uid_set):
            raise MechanismSourceQueueError("slot references an unknown request")
        family = family_by_uid[slot["family_uid"]]
        if slot["source_type"] != family["source_type"]:
            raise MechanismSourceQueueError("slot source type drift")
        if slot["family_uid"] != "publication_metadata" and slot["disposition"] != family["preregistered_disposition"]:
            raise MechanismSourceQueueError("zero-network family disposition drift")
        dedup_key = (slot["canonical_intervention_uid"], slot["canonical_source_identity"])
        if slot["disposition"] == "duplicate":
            if slot["duplicate_of"] != seen_eligible.get(dedup_key):
                raise MechanismSourceQueueError("duplicate lineage drift")
        elif slot["disposition"] == "eligible":
            if dedup_key in seen_eligible or slot["duplicate_of"] is not None:
                raise MechanismSourceQueueError("eligible slot deduplication drift")
            seen_eligible[dedup_key] = slot["slot_uid"]
        elif slot["duplicate_of"] is not None:
            raise MechanismSourceQueueError("non-duplicate slot has duplicate lineage")
    requests = chronology["requests"]
    if len(requests) > 48 or queue["network_summary"]["request_count"] != len(requests):
        raise MechanismSourceQueueError("request budget or count drift")
    if len({request["request_uid"] for request in requests}) != len(requests):
        raise MechanismSourceQueueError("request retry or duplicate request detected")
    successful = sum(request["status"] == "success" for request in requests)
    failed = sum(request["status"] == "request_failed" for request in requests)
    total_response_bytes = sum(request["response_bytes"] for request in requests)
    if queue["network_summary"] != {
        "request_count": len(requests),
        "successful_request_count": successful,
        "failed_request_count": failed,
        "overflow_record_count": queue["network_summary"]["overflow_record_count"],
        "total_response_bytes": total_response_bytes,
        "raw_responses_persisted": False,
        "queried_families": ["publication_metadata"] if requests else [],
    }:
        raise MechanismSourceQueueError("network summary drift")
    _validate_exact_request_chronology(preregistration, queue, chronology)
    if requests:
        preregistered_at = _parse_timestamp(preregistration["preregistered_at"])
        if _parse_timestamp(requests[0]["started_at"]) < preregistered_at:
            raise MechanismSourceQueueError("request preceded preregistration")
        for left, right in zip(requests, requests[1:]):
            if _parse_timestamp(right["started_at"]) < _parse_timestamp(left["completed_at"]):
                raise MechanismSourceQueueError("metadata requests were not serial")
            interval = (
                _parse_timestamp(right["started_at"]) - _parse_timestamp(left["started_at"])
            ).total_seconds()
            if interval < REQUEST_POLICY["minimum_request_interval_ms"] / 1000.0:
                raise MechanismSourceQueueError("minimum metadata request interval drift")
        if chronology["first_request_at"] != requests[0]["started_at"] or chronology["last_request_at"] != requests[-1]["completed_at"]:
            raise MechanismSourceQueueError("chronology request boundary drift")
    if queue["access_counters"] != ZERO_COUNTERS or chronology["access_counters"] != ZERO_COUNTERS or authorization["access_counters"] != ZERO_COUNTERS:
        raise MechanismSourceQueueError("access counters are not zero")
    if authorization["status"] != "MECHANISM_SOURCE_CONTENT_NOT_AUTHORIZED" or authorization["content_access_authorized"]:
        raise MechanismSourceQueueError("source content authorization drift")
    if authorization["downstream_authorization"] != DOWNSTREAM_AUTHORIZATION:
        raise MechanismSourceQueueError("downstream authorization opened")
    return queue, chronology, authorization


def validate_stage(
    *, repository_root: Path, stage: str, contract_path: Path | str = CONTRACT_PATH
) -> dict[str, Any]:
    if stage == "preregistration":
        preregistration = validate_preregistration(
            repository_root=repository_root, contract_path=contract_path
        )
        return {
            "status": "MECHANISM_SOURCE_QUEUE_PREREGISTRATION_VALID",
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "context_count": len(preregistration["context_action_projections"]),
            "family_count": len(preregistration["source_families"]),
            "request_plan_count": len(preregistration["expanded_requests"]),
            "access_counters": preregistration["access_counters"],
        }
    if stage == "frozen":
        queue, chronology, authorization = validate_frozen_queue(
            repository_root=repository_root, contract_path=contract_path
        )
        return {
            "status": "MECHANISM_SOURCE_QUEUE_FROZEN_VALID",
            "queue_sha256": queue["queue_sha256"],
            "chronology_sha256": chronology["chronology_sha256"],
            "authorization_sha256": authorization["authorization_sha256"],
            "slot_count": queue["counts"]["slot_count"],
            "request_count": queue["network_summary"]["request_count"],
            "dispositions": queue["counts"]["dispositions"],
            "access_counters": queue["access_counters"],
            "content_access_authorized": authorization["content_access_authorized"],
        }
    raise MechanismSourceQueueError("stage must be preregistration or frozen")
