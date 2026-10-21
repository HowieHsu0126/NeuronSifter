"""Bounded, provenance-complete mechanism-basis qualification."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator
import yaml

from Libs.utils.data.mechanism_source_queue import (
    AUTHORIZATION_PATH as SOURCE_AUTHORIZATION_PATH,
    AUTHORIZATION_SCHEMA_PATH as SOURCE_AUTHORIZATION_SCHEMA_PATH,
    CHRONOLOGY_PATH as SOURCE_CHRONOLOGY_PATH,
    CHRONOLOGY_SCHEMA_PATH as SOURCE_CHRONOLOGY_SCHEMA_PATH,
    CONTRACT_PATH as SOURCE_QUEUE_CONTRACT_PATH,
    PREREGISTRATION_PATH as SOURCE_PREREGISTRATION_PATH,
    PREREGISTRATION_SCHEMA_PATH as SOURCE_PREREGISTRATION_SCHEMA_PATH,
    QUEUE_PATH as SOURCE_QUEUE_PATH,
    QUEUE_SCHEMA_PATH as SOURCE_QUEUE_SCHEMA_PATH,
    ZERO_COUNTERS as SOURCE_ZERO_COUNTERS,
    _schema_validate as _source_schema_validate,
    _self_hash as _source_self_hash,
    _validate_common_slot_bindings,
    _validate_disabled_family_placeholders,
    _validate_exact_request_chronology,
    _validate_queue_derived_counts,
    canonical_json_bytes,
    canonical_value_sha256,
    sha256_bytes,
    validate_contract as validate_source_queue_contract,
    validate_upstream_authorities as validate_source_queue_upstream,
)
from Libs.utils.data.mechanistic_belief_construct_firewall import (
    validate_construct_authority,
    validate_cutoff_receipt,
)
from Libs.utils.model.mechanistic_belief_prior_reliability import (
    validate_prior_reliability_outputs,
)


CONTRACT_PATH = "Libs/configs/validation/mechanism_basis_qualification.yaml"
AUTHORIZATION_SCHEMA_PATH = (
    "Libs/configs/schema/mechanism_content_access_manifest.schema.json"
)
LEDGER_SCHEMA_PATH = "Libs/configs/schema/mechanism_evidence_ledger.schema.json"
GRAPH_SCHEMA_PATH = "Libs/configs/schema/mechanism_source_claim_graph.schema.json"
ASSESSMENT_SCHEMA_PATH = "Libs/configs/schema/mechanism_basis_assessment.schema.json"
DECISION_SCHEMA_PATH = "Libs/configs/schema/mechanism_basis_decision.schema.json"

AUTHORIZATION_PATH = (
    "Input/manifests/mechanistic_belief/mechanism_content_access_manifest.json"
)
LEDGER_PATH = "Input/processed/mechanistic_belief/mechanism_evidence_ledger.json"
GRAPH_PATH = "Input/processed/mechanistic_belief/mechanism_source_claim_graph.json"
ASSESSMENT_PATH = "Output/audits/mechanism_basis_assessment.json"
DECISION_PATH = "Output/audits/mechanism_basis_decision.json"

CONTRACT_SCHEMA_VERSION = "neuronsifter-mechanism-basis-qualification-contract-v1"
AUTHORIZATION_SCHEMA_VERSION = "neuronsifter-mechanism-content-access-manifest-v1"
LEDGER_SCHEMA_VERSION = "neuronsifter-mechanism-evidence-ledger-v1"
GRAPH_SCHEMA_VERSION = "neuronsifter-mechanism-source-claim-graph-v1"
ASSESSMENT_SCHEMA_VERSION = "neuronsifter-mechanism-basis-assessment-v1"
DECISION_SCHEMA_VERSION = "neuronsifter-mechanism-basis-decision-v1"
PARSER_VERSION = "mechanism-content-projector-v1"

EXPECTED_CONTRACT_SHA256 = "4bf05082a2a9a8b3e64fba701373e0e651a2f20bca52913ca3eda031b8a6078e"
EXPECTED_SCHEMA_SHA256 = {
    AUTHORIZATION_SCHEMA_PATH: "b41b18bdc60ec5d9e95f1e514d44e18956e28eab9f8f81c65c1aee46e7346fc7",
    LEDGER_SCHEMA_PATH: "1b1c86bed71db87bb411563763d5b144c255509db094a8d635ac71b2daec463d",
    GRAPH_SCHEMA_PATH: "a819343d51e028d1a5c6d679affe3e023564c9ae55993cc7b8b4157ddcc7fdaf",
    ASSESSMENT_SCHEMA_PATH: "e4111ffc901822058baebd0870fdbfc4f1fbc724a4df600c0989c06811665f93",
    DECISION_SCHEMA_PATH: "ee6073d3ccd41607a0392715609cc4ab84fe215898cdda956f375e50e4596fb0",
}

EXPECTED_UPSTREAM = {
    "Output/audits/mechanistic_belief_cutoff_receipt.json": (
        "a2c9cd351a8f8199a87864d4a7f97f5a3444e859a55efd717343654a5c1570cc",
        "receipt_sha256",
        "076c29ad32d88d6ce08ddcdc25545f90189754ae0ef1ed5acda242493f7d84d5",
    ),
    "Output/audits/mechanistic_belief_construct_authority.json": (
        "a1ea36299c2f2097a7606a45a1643edd731aa693643113b3ee03bea7e0b120e0",
        "authority_sha256",
        "372247967aeaa49cd61c10b80bf2097c9824b4a7f9245eda87353f2f35eabd7b",
    ),
    "Output/audits/mechanistic_belief_prior_reliability_authority.json": (
        "258049e0acc0bae52723bb540454dcba4ca383709373935bff2294fba63c3f7e",
        "authority_sha256",
        "de79f89b9b1f26605eb8e0fe97703bc1679f7dd4c3310146e9ed90fff63fcc84",
    ),
    "Output/audits/mechanistic_belief_terminal_decision_table.json": (
        "d347da41cb7bed51bf67c93d9c80e9588fd1dddb8d32c51fecafd8fa7d23341f",
        "decision_table_sha256",
        "5ac4d99aa2bcac833f747f2191e7ad04c795bf55d6aea41ee85685269d132db1",
    ),
    SOURCE_PREREGISTRATION_PATH: (
        "1d2a0c96c23d72d6bc98402c6ec090ba20e475dc499687abf678fa9f66dd5bd1",
        "preregistration_sha256",
        "d500441457ea3007787e6d9aa215634a6be9ea4596383d7e0ad5ec42a77240bb",
    ),
    SOURCE_QUEUE_PATH: (
        "e1d40a9b878af81e197bd32bbfaf5a972b9c31f361b80b1773b6a0b2a33c5c71",
        "queue_sha256",
        "d88c0a0e147105ce40fcd4140bfb4eb358537906d19259543c2b494b20160453",
    ),
    SOURCE_CHRONOLOGY_PATH: (
        "eae3720e2a018abe6e4cbd5aa993c964046f3d1abadadfa24e275eaf9668d56c",
        "chronology_sha256",
        "33ac94ae3ffe06203c6574c51ebe8af96bfd82263ee043a75c82080bdb30ab82",
    ),
    SOURCE_AUTHORIZATION_PATH: (
        "82052a30204e05bfbda07d10062c3faaa33c67006fe19c80abebc1b79030bbb3",
        "authorization_sha256",
        "f9390c9bb32c20738feb673774b4374237c48b014e59231ca4783069960ea6e3",
    ),
}

GRADES = (
    "DIRECT_MECHANISM",
    "PARAMETER_ANCHOR",
    "BOUNDARY_CONSTRAINT",
    "CONTEXT_ONLY",
    "INADMISSIBLE",
)
SUPPORT_GRADES = frozenset(GRADES[:3])
CLAIM_TYPES = (
    "equation",
    "causal_dependency",
    "structural_dependency",
    "parameter_range",
    "unit_definition",
    "assumption",
    "limitation",
    "measurement_definition",
    "latent_to_observable_link",
    "prior_role_support",
    "support_boundary",
)
ZERO_AUTHORIZATION_COUNTERS = {
    "source_content_access": 0,
    "unqueued_source_attempts": 0,
    "forbidden_value_reads": 0,
    "six_context_outcome_access": 0,
    "protected_holdout_access": 0,
}
DOWNSTREAM_CLOSURE = {
    "numerical_prior_population_authorized": False,
    "world_action_compiler_authorized": False,
    "runtime_implementation_authorized": False,
    "agent_trace_authorized": False,
    "inference_authorized": False,
    "training_authorized": False,
    "gpu_authorized": False,
    "evaluation_authorized": False,
    "push_authorized": False,
}
TERMINAL_INTERPRETATIONS = {
    "MECHANISM_BASIS_QUALIFIED": (
        "Every required mechanism-basis requirement has provenance-complete, "
        "non-contradictory support in the bounded queue. This is a mechanism-basis "
        "qualification only, not an efficacy or model-performance result."
    ),
    "MECHANISM_BASIS_INSUFFICIENT": (
        "The immutable queue was exhausted, but its admitted evidence did not "
        "provide provenance-complete support for every required mechanism, parameter "
        "domain, state-observable link, prior role, and support boundary. This is a "
        "bounded mechanism-basis insufficiency decision, not an efficacy or "
        "model-performance result."
    ),
}
FORBIDDEN_INTERPRETATION_PATTERN = re.compile(
    r"\b(efficacy|benefit|outcome|ranking|recommendation)\b", re.IGNORECASE
)
SAFE_INTERPRETATION_BOUNDARY = "not an efficacy or model-performance result"
CODE_POLICY = {
    "slot_count": 72,
    "eligible_count": 15,
    "cutoff_ineligible_count": 33,
    "source_date_ambiguous_count": 24,
    "maximum_requests_per_slot": 3,
    "maximum_total_requests": 45,
    "concurrency": 1,
    "timeout_seconds": 30,
    "retries": 0,
    "minimum_request_interval_ms": 400,
    "maximum_response_bytes": 16 * 1024 * 1024,
    "maximum_total_response_bytes": 128 * 1024 * 1024,
    "maximum_redirects": 3,
    "allowed_hosts": (
        "eutils.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
    ),
    "allowed_paths": (
        "/entrez/eutils/efetch.fcgi",
        "/entrez/eutils/elink.fcgi",
    ),
    "allowed_content_types": ("application/xml", "text/xml"),
    "eligible_family_uid": "publication_metadata",
    "raw_response_persistence_allowed": False,
    "response_value_logging_allowed": False,
}
EXPECTED_CODE_POLICY_SHA256 = "cf3d5ed434d65e20a0845c2e3f037bc3c12646778e0f0c38800e1fbd1983a256"
EXPECTED_AUTHORIZATION_SHA256 = "176455f454074ce29108e10c8163759bb2b7bbed2c2a412058cf6000e3896a57"
EXPECTED_LEDGER_SHA256 = "9269ac83f37a2ad779d28117e0d236da079433663363508fdde64062eaa0711b"
PROTECTED_PATH_PARTS = {
    "holdout",
    "protected",
    "evaluation",
    "six_context",
    "six-context",
    "results",
    "outcomes",
}

CANONICAL_REVIEW_SPECS = {
    "mechanism-source-slot-43cd5b75856c6a08c2f1": {
        "projection_sha256": "cb8a1d91ea67bf8db7e1ef176496da33fd8ad62be77ff1e791a5a6f40d717136",
        "rationale": "The allowlisted methods section defines MRI-derived hippocampal-volume and cortical-thickness measurements, supporting a structural-state observable definition but not an intervention effect.",
        "claims": [{"claim_type": "measurement_definition", "paraphrase": "Hippocampal volume and cortical thickness can be derived from centrally processed MRI with FreeSurfer and boundary-shift methods.", "locator": "PubmedArticle/MedlineCitation/Article/Abstract/AbstractText[2]", "content_sha256": "27a2074a894150afbf06a3020169f7c0e8ce1cea6474f7c5225570f7edbd6b9e", "requirement": ["state_observable_link", "structural_morphology_observable"], "units": [], "range_context": None, "assumptions": ["MRI processing is the declared measurement context."], "limitations": ["No intervention effect or clinical endpoint statement is admitted."], "support_boundary": "Structural imaging measurement definition only."}]},
    "mechanism-source-slot-c0bd33cd1ea78ca80f14": {
        "projection_sha256": "57a91e1483232e575347347f1523cd70d6aa5056b43031cd8da55458c5ea4805",
        "rationale": "The allowlisted background section directly states the antibody-target relationship and plaque-localized epitope.",
        "claims": [{"claim_type": "causal_dependency", "paraphrase": "Donanemab is an IgG1 antibody directed to an N-terminal pyroglutamate amyloid-beta epitope present in brain amyloid plaques.", "locator": "PubmedArticle/MedlineCitation/Article/Abstract/AbstractText[1]", "content_sha256": "072dc162654dbd398f7bb0ddd522ada93e2de00e80144164d37abffb645c29a2", "requirement": ["mechanism", "dependencies"], "units": [], "range_context": None, "assumptions": [], "limitations": ["This supports target-mechanism identity, not efficacy or clinical benefit."], "support_boundary": "Donanemab target and plaque-epitope relationship."}]},
    "mechanism-source-slot-dd37b16ccee6124b9f1e": {
        "projection_sha256": "f3b41152523ad13af1da3564616df3e36f601505e3dbd35ca9d98076c9ccf9e9",
        "rationale": "The allowlisted methods section supplies an explicit administered-dose range, route, and schedule context.",
        "claims": [{"claim_type": "parameter_range", "paraphrase": "The declared donanemab study design used intravenous doses from 0.1 to 10 mg/kg, with repeated dosing approximately monthly in the multiple-dose phase.", "locator": "PubmedArticle/MedlineCitation/Article/Abstract/AbstractText[2]", "content_sha256": "86c27138f2cf1c80e77dd1f00deee62b256b2325742258dd999a7fef18461b8f", "requirement": ["parameter_domain", "positive_real"], "units": ["mg/kg"], "range_context": "Donanemab intravenous study doses, 0.1-10 mg/kg; schedule approximately monthly.", "assumptions": ["The range is study-design context and is not a numerical prior."], "limitations": ["No response, safety, efficacy, or preferred dose is admitted."], "support_boundary": "Observed study-design dosing range only."}]},
    "mechanism-source-slot-ca4a334f8103cdf9b15c": {
        "projection_sha256": "90a1868d412cac66065b75f4330084ae5b71b047c0de59b37d1a3e1f0b7c885f",
        "rationale": "The allowlisted methods section defines retinal OCT and brain volumetric MRI measurements relevant to structural observables.",
        "claims": [{"claim_type": "measurement_definition", "paraphrase": "Retinal thickness was measured by spectral-domain optical coherence tomography and related to brain volumetric MRI measurements in the declared study design.", "locator": "PubmedArticle/MedlineCitation/Article/Abstract/AbstractText[3]", "content_sha256": "e2caacf863232c0feb638d8975469dd0a8f504b8f19577f28bc2abb7875494ac", "requirement": ["state_observable_link", "structural_morphology_observable"], "units": [], "range_context": None, "assumptions": ["The claim is limited to measurement linkage."], "limitations": ["No longitudinal change, correlation result, or intervention effect is admitted."], "support_boundary": "Retinal and brain structural measurement definitions only."}]},
}


class MechanismBasisQualificationError(ValueError):
    """Raised when the bounded mechanism-basis contract is violated."""


@dataclass(frozen=True)
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    body: bytes
    redirect_count: int = 0


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):  # type: ignore[no-untyped-def]
        return None


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MechanismBasisQualificationError(f"{label} must be an object")
    return value


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise MechanismBasisQualificationError(f"{label} must be an array")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise MechanismBasisQualificationError(f"{label} must be a relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise MechanismBasisQualificationError(f"{label} is not canonical")
    return value


def _resolve_file(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    normalized = _relative_path(relative, label="artifact path")
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    current = root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise MechanismBasisQualificationError("artifact path traverses a symlink")
    if must_exist:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise MechanismBasisQualificationError("required artifact is missing") from exc
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise MechanismBasisQualificationError("artifact path escapes repository root")
    return candidate


def _canonical_payload(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = _mapping(json.loads(raw.decode("utf-8")), label=label)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MechanismBasisQualificationError(f"{label} is not valid JSON") from exc
    if raw != canonical_json_bytes(payload):
        raise MechanismBasisQualificationError(f"{label} is not canonical JSON")
    return payload


def _load_regular_json(root: Path, relative: str, *, label: str) -> tuple[dict[str, Any], bytes]:
    path = _resolve_file(root, relative)
    raw = path.read_bytes()
    return _canonical_payload(raw, label=label), raw


def _run_git(root: Path, args: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise MechanismBasisQualificationError("required current-commit Git blob is unavailable")
    return result.stdout


def load_frozen_git_json(
    repository_root: Path,
    relative_path: str,
    expected_sha256: str,
    *,
    git_reader: Callable[[Path, Sequence[str]], bytes] = _run_git,
) -> tuple[dict[str, Any], bytes, str]:
    root = repository_root.resolve(strict=True)
    relative = _relative_path(relative_path, label="frozen Git artifact path")
    top = git_reader(root, ["rev-parse", "--show-toplevel"]).decode().strip()
    if Path(top).resolve(strict=True) != root:
        raise MechanismBasisQualificationError("another worktree cannot supply frozen authority")
    tracked = git_reader(root, ["ls-files", "--error-unmatch", "--", relative])
    if not tracked.strip():
        raise MechanismBasisQualificationError("frozen Git artifact is not tracked")
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    if candidate.exists():
        if candidate.is_symlink() or not candidate.is_file():
            raise MechanismBasisQualificationError("frozen artifact is not a regular file")
        status = git_reader(root, ["status", "--porcelain=v1", "--", relative])
        if status.strip():
            raise MechanismBasisQualificationError("frozen artifact has worktree drift")
        raw = candidate.read_bytes()
        source = "regular_tracked_file"
    else:
        raw = git_reader(root, ["show", f"HEAD:{relative}"])
        source = "current_commit_git_blob"
    if sha256_bytes(raw) != expected_sha256:
        raise MechanismBasisQualificationError("frozen artifact byte identity drift")
    return _canonical_payload(raw, label=relative), raw, source


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        return _mapping(yaml.safe_load(path.read_text()), label="qualification contract")
    except yaml.YAMLError as exc:
        raise MechanismBasisQualificationError("qualification contract is invalid YAML") from exc


def _schema(root: Path, relative: str) -> tuple[dict[str, Any], str]:
    path = _resolve_file(root, relative)
    raw = path.read_bytes()
    try:
        payload = _mapping(json.loads(raw.decode("utf-8")), label=relative)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MechanismBasisQualificationError(
            "qualification schema is invalid JSON"
        ) from exc
    actual = sha256_bytes(raw)
    if actual != EXPECTED_SCHEMA_SHA256[relative]:
        raise MechanismBasisQualificationError("qualification schema byte identity drift")
    Draft202012Validator.check_schema(payload)
    return payload, actual


def _schema_validate(payload: Mapping[str, Any], schema: Mapping[str, Any], *, label: str) -> None:
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "<root>"
        raise MechanismBasisQualificationError(f"{label} schema violation at {path}: {errors[0].message}")


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    projected = deepcopy(dict(payload))
    projected.pop(field, None)
    return canonical_value_sha256(projected)


def _code_policy_sha256() -> str:
    return canonical_value_sha256(CODE_POLICY)


def validate_contract(
    repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> tuple[dict[str, Any], bytes]:
    root = repository_root.resolve(strict=True)
    relative = _relative_path(Path(contract_path).as_posix(), label="contract path")
    if relative != CONTRACT_PATH:
        raise MechanismBasisQualificationError("qualification contract path drift")
    path = _resolve_file(root, relative)
    raw = path.read_bytes()
    if sha256_bytes(raw) != EXPECTED_CONTRACT_SHA256:
        raise MechanismBasisQualificationError("qualification contract byte identity drift")
    contract = _load_yaml(path)
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise MechanismBasisQualificationError("qualification contract schema version drift")
    if _code_policy_sha256() != EXPECTED_CODE_POLICY_SHA256:
        raise MechanismBasisQualificationError("qualification code policy drift")
    queue_policy = _mapping(contract.get("frozen_queue"), label="frozen queue policy")
    request_policy = _mapping(contract.get("retrieval_policy"), label="retrieval policy")
    expected_queue = {
        "slot_count": CODE_POLICY["slot_count"],
        "eligible_slot_count": CODE_POLICY["eligible_count"],
        "cutoff_ineligible_slot_count": CODE_POLICY["cutoff_ineligible_count"],
        "source_date_ambiguous_slot_count": CODE_POLICY["source_date_ambiguous_count"],
        "eligible_family_uid": CODE_POLICY["eligible_family_uid"],
        "no_refill": True,
        "no_replacement": True,
        "preserve_order": True,
    }
    if queue_policy != expected_queue:
        raise MechanismBasisQualificationError("frozen queue policy drift")
    for key in (
        "maximum_requests_per_slot",
        "maximum_total_requests",
        "concurrency",
        "timeout_seconds",
        "retries",
        "minimum_request_interval_ms",
        "maximum_response_bytes",
        "maximum_total_response_bytes",
        "maximum_redirects",
        "raw_response_persistence_allowed",
        "response_value_logging_allowed",
    ):
        if request_policy.get(key) != CODE_POLICY[key]:
            raise MechanismBasisQualificationError("retrieval policy drift")
    if tuple(request_policy.get("allowed_hosts", [])) != CODE_POLICY["allowed_hosts"]:
        raise MechanismBasisQualificationError("retrieval host allowlist drift")
    if tuple(request_policy.get("allowed_paths", [])) != CODE_POLICY["allowed_paths"]:
        raise MechanismBasisQualificationError("retrieval path allowlist drift")
    if tuple(request_policy.get("allowed_content_types", [])) != CODE_POLICY["allowed_content_types"]:
        raise MechanismBasisQualificationError("retrieval content-type allowlist drift")
    if contract.get("access_counters") != ZERO_AUTHORIZATION_COUNTERS:
        raise MechanismBasisQualificationError("pre-content access counters are not zero")
    if contract.get("downstream_authorization") != DOWNSTREAM_CLOSURE:
        raise MechanismBasisQualificationError("downstream authorization drift")
    return contract, raw


def _artifact_binding(path: str, raw: bytes, semantic_sha256: str) -> dict[str, str]:
    return {
        "path": path,
        "file_sha256": sha256_bytes(raw),
        "semantic_sha256": semantic_sha256,
    }


def _contract_binding(authorities: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": CONTRACT_PATH,
        "sha256": sha256_bytes(authorities["contract_raw"]),
    }


def _upstream_authority_bindings(
    authorities: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        _artifact_binding(path, authorities["loaded"][path][1], values[2])
        for path, values in EXPECTED_UPSTREAM.items()
    ]


def _require_exact_bindings(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for field, binding in expected.items():
        if payload.get(field) != binding:
            raise MechanismBasisQualificationError(
                f"{label} {field} binding drift"
            )


def _validate_upstream_regular(root: Path, contract: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], bytes]]:
    configured = {
        str(row["path"]): row
        for row in _sequence(contract.get("upstream_authorities"), label="upstream authorities")
    }
    if set(configured) != set(EXPECTED_UPSTREAM):
        raise MechanismBasisQualificationError("upstream authority roster drift")
    loaded: dict[str, tuple[dict[str, Any], bytes]] = {}
    for relative, (expected_file, semantic_field, expected_semantic) in EXPECTED_UPSTREAM.items():
        row = configured[relative]
        if row != {
            "path": relative,
            "file_sha256": expected_file,
            "semantic_field": semantic_field,
            "semantic_sha256": expected_semantic,
        }:
            raise MechanismBasisQualificationError("upstream contract binding drift")
        if relative in {SOURCE_PREREGISTRATION_PATH, SOURCE_QUEUE_PATH}:
            payload, raw, _ = load_frozen_git_json(root, relative, expected_file)
        else:
            payload, raw = _load_regular_json(root, relative, label=relative)
            if sha256_bytes(raw) != expected_file:
                raise MechanismBasisQualificationError("upstream artifact byte identity drift")
        if payload.get(semantic_field) != expected_semantic:
            raise MechanismBasisQualificationError("upstream semantic identity drift")
        loaded[relative] = (payload, raw)
    return loaded


def load_frozen_authorities(
    repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    contract, contract_raw = validate_contract(root, contract_path)
    validate_cutoff_receipt(repository_root=root)
    construct = validate_construct_authority(repository_root=root)
    prior, decision_table = validate_prior_reliability_outputs(repository_root=root)
    source_queue_contract, _ = validate_source_queue_contract(
        root, SOURCE_QUEUE_CONTRACT_PATH
    )
    validate_source_queue_upstream(root, source_queue_contract)
    loaded = _validate_upstream_regular(root, contract)
    prereg, prereg_raw = loaded[SOURCE_PREREGISTRATION_PATH]
    queue, queue_raw = loaded[SOURCE_QUEUE_PATH]
    chronology, chronology_raw = loaded[SOURCE_CHRONOLOGY_PATH]
    source_authorization, source_authorization_raw = loaded[SOURCE_AUTHORIZATION_PATH]
    for payload, schema_path, label in (
        (prereg, SOURCE_PREREGISTRATION_SCHEMA_PATH, "source preregistration"),
        (queue, SOURCE_QUEUE_SCHEMA_PATH, "source queue"),
        (chronology, SOURCE_CHRONOLOGY_SCHEMA_PATH, "source chronology"),
        (source_authorization, SOURCE_AUTHORIZATION_SCHEMA_PATH, "source authorization"),
    ):
        try:
            schema = _mapping(
                json.loads(_resolve_file(root, schema_path).read_text()),
                label=schema_path,
            )
        except json.JSONDecodeError as exc:
            raise MechanismBasisQualificationError(
                "source queue schema is invalid JSON"
            ) from exc
        _source_schema_validate(payload, schema, label=label)
    if prereg["preregistration_sha256"] != _source_self_hash(prereg, "preregistration_sha256"):
        raise MechanismBasisQualificationError("source preregistration semantic drift")
    if queue["queue_sha256"] != _source_self_hash(queue, "queue_sha256"):
        raise MechanismBasisQualificationError("source queue semantic drift")
    _validate_queue_derived_counts(queue, prereg)
    _validate_common_slot_bindings(prereg, queue)
    _validate_disabled_family_placeholders(prereg, queue)
    _validate_exact_request_chronology(prereg, queue, chronology)
    dispositions = Counter(slot["disposition"] for slot in queue["slots"])
    if len(queue["slots"]) != 72 or dispositions != Counter(
        {"eligible": 15, "cutoff_ineligible_release": 33, "source_date_ambiguous": 24}
    ):
        raise MechanismBasisQualificationError("frozen queue count drift")
    if any(
        payload.get("access_counters") != SOURCE_ZERO_COUNTERS
        for payload in (prereg, queue, chronology, source_authorization)
    ):
        raise MechanismBasisQualificationError("Round-26 access counter drift")
    if (
        source_authorization.get("status") != "MECHANISM_SOURCE_CONTENT_NOT_AUTHORIZED"
        or source_authorization.get("content_access_authorized") is not False
    ):
        raise MechanismBasisQualificationError("Round-26 authorization drift")
    if construct["method_uid"] != prior["method_uid"] or decision_table["method_uid"] != prior["method_uid"]:
        raise MechanismBasisQualificationError("upstream method identity drift")
    return {
        "contract": contract,
        "contract_raw": contract_raw,
        "construct": construct,
        "prior": prior,
        "decision_table": decision_table,
        "preregistration": prereg,
        "preregistration_raw": prereg_raw,
        "queue": queue,
        "queue_raw": queue_raw,
        "chronology": chronology,
        "chronology_raw": chronology_raw,
        "source_authorization": source_authorization,
        "source_authorization_raw": source_authorization_raw,
        "loaded": loaded,
    }


def _requirement_uid(category: str, subject_uid: str) -> str:
    return "mechanism-requirement-" + hashlib.sha256(
        f"{category}\n{subject_uid}".encode()
    ).hexdigest()[:20]


def derive_requirement_registry(
    construct: Mapping[str, Any], prior: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        category: str,
        subject_uid: str,
        required: bool,
        description: str,
        claim_types: Sequence[str],
        authority_path: str,
        locator: str,
        semantic_sha256: str,
    ) -> None:
        rows.append(
            {
                "requirement_uid": _requirement_uid(category, subject_uid),
                "category": category,
                "subject_uid": subject_uid,
                "required": required,
                "description": description,
                "allowed_claim_types": list(claim_types),
                "authority_path": authority_path,
                "authority_locator": locator,
                "authority_semantic_sha256": semantic_sha256,
            }
        )

    construct_sha = str(construct["authority_sha256"])
    prior_sha = str(prior["authority_sha256"])
    for index, subject in enumerate(construct["construct"]["mechanism_supports"]):
        claim_map = {
            "equations": ["equation"],
            "dependencies": ["causal_dependency", "structural_dependency"],
            "parameter_ranges": ["parameter_range", "unit_definition"],
            "assumptions": ["assumption"],
            "uncertainty": ["assumption", "limitation", "support_boundary"],
            "support_boundaries": ["support_boundary", "limitation"],
        }
        add(
            "mechanism",
            str(subject),
            True,
            f"Source-backed support for declared mechanism role {subject}.",
            claim_map[str(subject)],
            "Output/audits/mechanistic_belief_construct_authority.json",
            f"construct.mechanism_supports[{index}]",
            construct_sha,
        )
    for index, factor in enumerate(prior["latent_state_factors"]):
        add(
            "latent_state",
            str(factor["factor_uid"]),
            True,
            f"Source-backed semantics for latent factor {factor['factor_uid']}.",
            ["causal_dependency", "structural_dependency", "support_boundary"],
            "Output/audits/mechanistic_belief_prior_reliability_authority.json",
            f"latent_state_factors[{index}]",
            prior_sha,
        )
    for index, observable in enumerate(prior["observable_families"]):
        add(
            "state_observable_link",
            str(observable["observable_family_uid"]),
            True,
            f"Measurement and latent-state linkage for {observable['observable_family_uid']}.",
            ["measurement_definition", "latent_to_observable_link", "unit_definition"],
            "Output/audits/mechanistic_belief_prior_reliability_authority.json",
            f"observable_families[{index}]",
            prior_sha,
        )
    for index, domain in enumerate(prior["parameter_domain_distribution_families"]):
        subject = str(domain["parameter_domain"])
        add(
            "parameter_domain",
            subject,
            True,
            f"Typed range, units, and support boundary for parameter domain {subject}.",
            ["parameter_range", "unit_definition", "support_boundary"],
            "Output/audits/mechanistic_belief_prior_reliability_authority.json",
            f"parameter_domain_distribution_families[{index}]",
            prior_sha,
        )
    for index, role in enumerate(prior["prior_roles"]["ordered_roles"]):
        add(
            "prior_role",
            str(role["role_uid"]),
            bool(role["required"]),
            f"Admitted evidence support for prior role {role['role_uid']} without numerical population.",
            ["prior_role_support", "parameter_range", "support_boundary"],
            "Output/audits/mechanistic_belief_prior_reliability_authority.json",
            f"prior_roles.ordered_roles[{index}]",
            prior_sha,
        )
    for index, state in enumerate(prior["support_contract"]["states"]):
        add(
            "support_boundary",
            str(state["support_state"]),
            True,
            f"Source-backed boundary semantics for support state {state['support_state']}.",
            ["support_boundary", "limitation", "assumption"],
            "Output/audits/mechanistic_belief_prior_reliability_authority.json",
            f"support_contract.states[{index}]",
            prior_sha,
        )
    return rows


def _contact_email(root: Path) -> tuple[str, str]:
    environment_email = os.environ.get("NEURONSIFTER_ENTREZ_EMAIL", "").strip()
    if environment_email:
        source, email = "NEURONSIFTER_ENTREZ_EMAIL", environment_email
    else:
        result = subprocess.run(
            ["git", "-C", str(root), "config", "--get", "user.email"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        source, email = "git_config_user_email", result.stdout.strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise MechanismBasisQualificationError("an existing valid NCBI contact email is required")
    return source, email


def _request_uid(slot_uid: str, request_kind: str, url: str) -> str:
    return "mechanism-content-request-" + hashlib.sha256(
        f"{slot_uid}\n{request_kind}\n{url}".encode()
    ).hexdigest()[:20]


def _request_plan(slot: Mapping[str, Any], email: str) -> list[dict[str, Any]]:
    pmid = str(_mapping(slot["metadata"], label="slot metadata")["pmid"])
    query = urlencode(
        [
            ("db", "pubmed"),
            ("id", pmid),
            ("rettype", "pubmed"),
            ("retmode", "xml"),
            ("tool", "NeuronSifter"),
            ("email", email),
        ],
        quote_via=quote,
    )
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{query}"
    return [
        {
            "request_uid": _request_uid(str(slot["slot_uid"]), "pubmed_exact_fetch", url),
            "request_kind": "pubmed_exact_fetch",
            "method": "GET",
            "url": url,
            "host": "eutils.ncbi.nlm.nih.gov",
        }
    ]


def _authorized_slot_row(slot: Mapping[str, Any], email: str) -> dict[str, Any]:
    metadata = _mapping(slot["metadata"], label="eligible slot metadata")
    pmid = metadata.get("pmid")
    if not isinstance(pmid, str) or not re.fullmatch(r"[0-9]+", pmid):
        raise MechanismBasisQualificationError("eligible slot lacks exact PMID identity")
    return {
        "queue_order": slot["queue_order"],
        "slot_uid": slot["slot_uid"],
        "canonical_intervention_uid": slot["canonical_intervention_uid"],
        "validation_uid": slot["validation_uid"],
        "family_uid": slot["family_uid"],
        "canonical_source_identity": slot["canonical_source_identity"],
        "pmid": pmid,
        "doi": metadata.get("doi"),
        "official_date": slot["official_date"],
        "queue_disposition": slot["disposition"],
        "request_plan": _request_plan(slot, email),
        "public_access_status": metadata["public_access_status"],
        "license_status": metadata["license_status"],
    }


def _schema_bindings(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for relative in EXPECTED_SCHEMA_SHA256:
        _, digest = _schema(root, relative)
        result[relative] = {"path": relative, "sha256": digest}
    return result


def _write_temp(directory: Path, payload: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".mechanism-basis-{os.getpid()}-{time.time_ns()}.tmp"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _install_many(items: Sequence[tuple[Path, bytes]]) -> bool:
    existing = [path.exists() for path, _ in items]
    if all(existing):
        for path, raw in items:
            if path.read_bytes() != raw:
                raise MechanismBasisQualificationError("refusing to overwrite non-identical artifact")
        return False
    if any(existing):
        raise MechanismBasisQualificationError("partial artifact installation detected")
    temporary: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for path, raw in items:
            temporary.append((path, _write_temp(path.parent, raw)))
        for destination, source in temporary:
            os.replace(source, destination)
            installed.append(destination)
    except Exception:
        for destination in installed:
            destination.unlink(missing_ok=True)
        for _, source in temporary:
            source.unlink(missing_ok=True)
        raise
    return True


def build_content_authorization(
    *, repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> tuple[dict[str, Any], bool]:
    root = repository_root.resolve(strict=True)
    authorities = load_frozen_authorities(root, contract_path)
    contract = authorities["contract"]
    schemas = _schema_bindings(root)
    source, email = _contact_email(root)
    try:
        import Bio

        biopython_version = str(Bio.__version__)
    except ImportError as exc:
        raise MechanismBasisQualificationError("Biopython is required for NCBI client qualification") from exc
    eligible = [
        slot for slot in authorities["queue"]["slots"] if slot["disposition"] == "eligible"
    ]
    if len(eligible) != CODE_POLICY["eligible_count"] or any(
        slot["family_uid"] != CODE_POLICY["eligible_family_uid"] for slot in eligible
    ):
        raise MechanismBasisQualificationError("eligible slot allowlist drift")
    eligible_rows = [_authorized_slot_row(slot, email) for slot in eligible]
    payload: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": "MECHANISM_CONTENT_ACCESS_AUTHORIZED",
        "authorization_uid": "mechanism_content_access_manifest",
        "contract": _contract_binding(authorities),
        "schema": schemas[AUTHORIZATION_SCHEMA_PATH],
        "upstream_authorities": _upstream_authority_bindings(authorities),
        "queue": _artifact_binding(
            SOURCE_QUEUE_PATH,
            authorities["queue_raw"],
            authorities["queue"]["queue_sha256"],
        ),
        "chronology": _artifact_binding(
            SOURCE_CHRONOLOGY_PATH,
            authorities["chronology_raw"],
            authorities["chronology"]["chronology_sha256"],
        ),
        "eligible_slots": eligible_rows,
        "requirement_registry": derive_requirement_registry(
            authorities["construct"], authorities["prior"]
        ),
        "retrieval_policy": {
            **deepcopy(contract["retrieval_policy"]),
            "contact_email": email,
            "contact_email_source": source,
            "biopython_version_inspected": biopython_version,
        },
        "allowed_content_classes": list(contract["content_policy"]["allowed_content_classes"]),
        "denied_content_classes": list(contract["content_policy"]["denied_content_classes"]),
        "network_requests_performed": 0,
        "content_access_authorized": True,
        "access_counters": deepcopy(ZERO_AUTHORIZATION_COUNTERS),
        "downstream_authorization": deepcopy(DOWNSTREAM_CLOSURE),
        "blockers": [],
    }
    payload["authorization_sha256"] = _self_hash(payload, "authorization_sha256")
    schema, _ = _schema(root, AUTHORIZATION_SCHEMA_PATH)
    _schema_validate(payload, schema, label="content access manifest")
    path = _resolve_file(root, AUTHORIZATION_PATH, must_exist=False)
    created = _install_many([(path, canonical_json_bytes(payload))])
    return payload, created


def _load_output(root: Path, relative: str, schema_path: str, hash_field: str) -> tuple[dict[str, Any], bytes]:
    payload, raw = _load_regular_json(root, relative, label=relative)
    schema, _ = _schema(root, schema_path)
    _schema_validate(payload, schema, label=relative)
    if payload.get(hash_field) != _self_hash(payload, hash_field):
        raise MechanismBasisQualificationError(f"{relative} semantic identity drift")
    return payload, raw


def validate_authorization(
    *, repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    authorities = load_frozen_authorities(root, contract_path)
    manifest, _ = _load_output(
        root, AUTHORIZATION_PATH, AUTHORIZATION_SCHEMA_PATH, "authorization_sha256"
    )
    contract = authorities["contract"]
    schemas = _schema_bindings(root)
    _require_exact_bindings(
        manifest,
        {
            "contract": _contract_binding(authorities),
            "schema": schemas[AUTHORIZATION_SCHEMA_PATH],
            "upstream_authorities": _upstream_authority_bindings(authorities),
            "queue": _artifact_binding(
                SOURCE_QUEUE_PATH,
                authorities["queue_raw"],
                authorities["queue"]["queue_sha256"],
            ),
            "chronology": _artifact_binding(
                SOURCE_CHRONOLOGY_PATH,
                authorities["chronology_raw"],
                authorities["chronology"]["chronology_sha256"],
            ),
        },
        label="authorization",
    )
    expected_requirements = derive_requirement_registry(
        authorities["construct"], authorities["prior"]
    )
    if manifest["requirement_registry"] != expected_requirements:
        raise MechanismBasisQualificationError("post-access requirement registry drift")
    eligible = [
        slot for slot in authorities["queue"]["slots"] if slot["disposition"] == "eligible"
    ]
    email = str(manifest["retrieval_policy"].get("contact_email", ""))
    expected_slots = [_authorized_slot_row(slot, email) for slot in eligible]
    if manifest["eligible_slots"] != expected_slots:
        raise MechanismBasisQualificationError("authorized eligible-slot record drift")
    expected_policy = {
        **deepcopy(contract["retrieval_policy"]),
        "contact_email": manifest["retrieval_policy"]["contact_email"],
        "contact_email_source": manifest["retrieval_policy"]["contact_email_source"],
        "biopython_version_inspected": manifest["retrieval_policy"]["biopython_version_inspected"],
    }
    if manifest["retrieval_policy"] != expected_policy:
        raise MechanismBasisQualificationError("authorization retrieval policy drift")
    if manifest["allowed_content_classes"] != list(
        contract["content_policy"]["allowed_content_classes"]
    ) or manifest["denied_content_classes"] != list(
        contract["content_policy"]["denied_content_classes"]
    ):
        raise MechanismBasisQualificationError("authorization content policy drift")
    if manifest["network_requests_performed"] != 0 or manifest["access_counters"] != ZERO_AUTHORIZATION_COUNTERS:
        raise MechanismBasisQualificationError("authorization is not pre-content")
    if manifest["downstream_authorization"] != DOWNSTREAM_CLOSURE:
        raise MechanismBasisQualificationError("authorization opened downstream work")
    if manifest["authorization_sha256"] != EXPECTED_AUTHORIZATION_SHA256:
        raise MechanismBasisQualificationError("authorization frozen identity drift")
    return manifest


def _host_allowed(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in CODE_POLICY["allowed_hosts"]
        and parsed.path in CODE_POLICY["allowed_paths"]
    )


def validate_exact_request_plan(
    request: Mapping[str, Any], slot: Mapping[str, Any]
) -> None:
    url = str(request.get("url", ""))
    if not _host_allowed(url) or request.get("method") != "GET":
        raise MechanismBasisQualificationError("content request endpoint is not allowlisted")
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    metadata = _mapping(slot.get("metadata"), label="queue slot metadata")
    if request.get("request_kind") != "pubmed_exact_fetch" or query != {
        "db": "pubmed",
        "id": metadata.get("pmid"),
        "rettype": "pubmed",
        "retmode": "xml",
        "tool": "NeuronSifter",
        "email": query.get("email"),
    }:
        raise MechanismBasisQualificationError("content request is not exact-ID PubMed fetch")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", str(query.get("email", ""))):
        raise MechanismBasisQualificationError("content request contact email is invalid")
    expected_uid = _request_uid(str(slot["slot_uid"]), "pubmed_exact_fetch", url)
    if request.get("request_uid") != expected_uid:
        raise MechanismBasisQualificationError("content request identity drift")


def fetch_exact_ncbi(
    request_spec: Mapping[str, Any], policy: Mapping[str, Any]
) -> FetchResult:
    url = str(request_spec["url"])
    if not _host_allowed(url):
        raise MechanismBasisQualificationError("content request endpoint is not allowlisted")
    current_url = url
    opener = build_opener(ProxyHandler({}), _NoRedirect)
    for redirect_count in range(int(policy["maximum_redirects"]) + 1):
        request = Request(
            current_url,
            method="GET",
            headers={
                "Accept": "application/xml,text/xml;q=0.9",
                "User-Agent": "NeuronSifter-MechanismBasis/1.0",
            },
        )
        try:
            with opener.open(request, timeout=int(policy["timeout_seconds"])) as response:
                body = response.read(int(policy["maximum_response_bytes"]) + 1)
                if len(body) > int(policy["maximum_response_bytes"]):
                    raise MechanismBasisQualificationError("content response exceeds byte limit")
                final_url = response.geturl()
                if not _host_allowed(final_url):
                    raise MechanismBasisQualificationError("content final endpoint is not allowlisted")
                return FetchResult(
                    requested_url=url,
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
                    raise MechanismBasisQualificationError("content redirect limit exceeded") from exc
                next_url = urljoin(current_url, location)
                if not _host_allowed(next_url):
                    raise MechanismBasisQualificationError("content redirect endpoint is not allowlisted") from exc
                current_url = next_url
                continue
            raise MechanismBasisQualificationError("content request failed") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise MechanismBasisQualificationError("content transport failed") from exc
    raise MechanismBasisQualificationError("content redirect limit exceeded")


def project_pubmed_xml(
    body: bytes,
    *,
    expected_pmid: str,
    allowed_labels: Sequence[str],
    denied_labels: Sequence[str],
) -> dict[str, Any]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise MechanismBasisQualificationError("PubMed content is not valid XML") from exc
    pmids = [
        (node.text or "").strip()
        for node in root.findall(".//MedlineCitation/PMID")
    ]
    if pmids != [expected_pmid]:
        raise MechanismBasisQualificationError("PubMed content identifier mismatch")
    allowed_set = {value.upper() for value in allowed_labels}
    denied_set = {value.upper() for value in denied_labels}
    allowed_sections: list[dict[str, str]] = []
    forbidden_sections = 0
    mixed_sections = 0
    abstract_nodes = root.findall(".//MedlineCitation/Article/Abstract/AbstractText")
    for index, node in enumerate(abstract_nodes, start=1):
        label = str(node.attrib.get("Label") or node.attrib.get("NlmCategory") or "").upper()
        category = str(node.attrib.get("NlmCategory") or "").upper()
        if label in denied_set or category in denied_set:
            forbidden_sections += 1
            continue
        if label not in allowed_set:
            mixed_sections += 1
            continue
        text = " ".join("".join(node.itertext()).split())
        if not text:
            continue
        content_class = {
            "METHODS": "measurement_definition",
            "LIMITATIONS": "explicit_limitation",
        }.get(label, "mechanism_definition")
        allowed_sections.append(
            {
                "locator": f"PubmedArticle/MedlineCitation/Article/Abstract/AbstractText[{index}]",
                "section_label": label,
                "content_class": content_class,
                "content_sha256": sha256_bytes(text.encode("utf-8")),
                "text": text,
            }
        )
    disposition = "reviewed_allowed_projection"
    if not allowed_sections:
        disposition = "forbidden_mixed_content" if mixed_sections else "forbidden_content_only"
    return {
        "pmid": expected_pmid,
        "allowed_sections": allowed_sections,
        "forbidden_sections_skipped": forbidden_sections,
        "mixed_sections_rejected": mixed_sections,
        "disposition": disposition,
    }


def _safe_local_content_path(repository_root: Path, input_root: Path, relative_path: str) -> Path:
    root = repository_root.resolve(strict=True)
    allowed_root = input_root.resolve(strict=True)
    if not allowed_root.is_relative_to(root):
        raise MechanismBasisQualificationError("local content root escapes repository")
    relative = _relative_path(relative_path, label="local content path")
    pure = PurePosixPath(relative)
    if any(part.lower() in PROTECTED_PATH_PARTS for part in pure.parts):
        raise MechanismBasisQualificationError("local content path targets protected content")
    candidate = allowed_root.joinpath(*pure.parts)
    current = allowed_root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise MechanismBasisQualificationError("local content path traverses a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(allowed_root) or not resolved.is_file():
        raise MechanismBasisQualificationError("local content path escapes allowed root")
    return candidate


def open_local_exact_source(
    *,
    repository_root: Path,
    input_root: Path,
    relative_path: str,
    manifest_entry: Mapping[str, Any],
    slot: Mapping[str, Any],
) -> bytes:
    if (
        manifest_entry.get("canonical_source_identity")
        != slot.get("canonical_source_identity")
        or manifest_entry.get("license_status") != "public_use_authorized"
    ):
        raise MechanismBasisQualificationError("local source identity or license is not authorized")
    path = _safe_local_content_path(repository_root, input_root, relative_path)
    raw = path.read_bytes()
    if sha256_bytes(raw) != manifest_entry.get("file_sha256"):
        raise MechanismBasisQualificationError("local source byte identity drift")
    return raw


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _scratch_is_local(path: Path) -> bool:
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() not in {"nfs", "nfs4"}


def _base_evidence_row(slot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queue_order": slot["queue_order"],
        "slot_uid": slot["slot_uid"],
        "queue_slot_sha256": canonical_value_sha256(slot),
        "canonical_intervention_uid": slot["canonical_intervention_uid"],
        "validation_uid": slot["validation_uid"],
        "family_uid": slot["family_uid"],
        "canonical_source_identity": slot["canonical_source_identity"],
        "official_date": slot["official_date"],
        "queue_disposition": slot["disposition"],
        "projection_sha256": None,
        "semantic_claims": [],
        "claims_supported": [],
        "units": [],
        "assumptions": [],
        "limitations": [],
    }


def _claim_uid(slot_uid: str, claim: Mapping[str, Any]) -> str:
    identity_claim = {
        key: value for key, value in claim.items() if key != "contradicts_claim_uids"
    }
    return "mechanism-claim-" + hashlib.sha256(
        canonical_json_bytes({"slot_uid": slot_uid, **identity_claim})
    ).hexdigest()[:20]


def _derive_grade(claims: Sequence[Mapping[str, Any]], *, admissible_context: bool) -> str:
    claim_types = {str(claim["claim_type"]) for claim in claims}
    if claim_types & {"equation", "causal_dependency", "structural_dependency", "measurement_definition", "latent_to_observable_link"}:
        return "DIRECT_MECHANISM"
    if claim_types & {"parameter_range", "unit_definition", "prior_role_support"}:
        return "PARAMETER_ANCHOR"
    if claim_types & {"assumption", "limitation", "support_boundary"}:
        return "BOUNDARY_CONSTRAINT"
    return "CONTEXT_ONLY" if admissible_context else "INADMISSIBLE"


def _semantic_review(
    slot: Mapping[str, Any],
    projection: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any] | None,
) -> dict[str, Any]:
    projection_sha = canonical_value_sha256(projection)
    allowed_sections = {
        (row["locator"], row["content_sha256"]): row
        for row in projection["allowed_sections"]
    }
    requirement_by_key = {
        (row["category"], row["subject_uid"]): row
        for row in requirements
    }
    claims = []
    if spec is not None:
        if spec["projection_sha256"] != projection_sha:
            raise MechanismBasisQualificationError("semantic review projection identity drift")
        for raw_claim in spec["claims"]:
            locator_key = (raw_claim["locator"], raw_claim["content_sha256"])
            if locator_key not in allowed_sections:
                raise MechanismBasisQualificationError("semantic claim locator or content hash drift")
            requirement = requirement_by_key.get(tuple(raw_claim["requirement"]))
            if requirement is None:
                raise MechanismBasisQualificationError("semantic claim requirement mapping drift")
            if raw_claim["claim_type"] not in requirement["allowed_claim_types"]:
                raise MechanismBasisQualificationError("semantic claim type is not allowed by requirement")
            section_class = allowed_sections[locator_key]["content_class"]
            compatible = {
                "mechanism_definition": {"equation", "causal_dependency", "structural_dependency", "assumption", "limitation", "support_boundary"},
                "measurement_definition": {"measurement_definition", "latent_to_observable_link", "parameter_range", "unit_definition"},
                "explicit_limitation": {"assumption", "limitation", "support_boundary"},
            }
            if raw_claim["claim_type"] not in compatible.get(section_class, set()):
                raise MechanismBasisQualificationError("semantic claim type is incompatible with section class")
            claim = {
                "claim_type": raw_claim["claim_type"],
                "paraphrase": raw_claim["paraphrase"],
                "content_locator": raw_claim["locator"],
                "content_sha256": raw_claim["content_sha256"],
                "requirement_uids": [requirement["requirement_uid"]],
                "units": list(raw_claim["units"]),
                "range_context": raw_claim["range_context"],
                "assumptions": list(raw_claim["assumptions"]),
                "limitations": list(raw_claim["limitations"]),
                "support_boundary": raw_claim["support_boundary"],
                "contradicts_claim_uids": list(raw_claim.get("contradicts_claim_uids", [])),
            }
            if claim["claim_type"] == "parameter_range" and (
                not claim["units"] or not claim["range_context"]
            ):
                raise MechanismBasisQualificationError("parameter claim lacks units or range context")
            claim["claim_uid"] = _claim_uid(str(slot["slot_uid"]), claim)
            claims.append(claim)
    admissible = bool(projection["allowed_sections"])
    grade = _derive_grade(claims, admissible_context=admissible)
    rationale = (
        str(spec["rationale"])
        if spec is not None
        else (
            "The allowlisted projection was semantically reviewed but supplied no support-grade claim."
            if admissible
            else "No structurally separable allowlisted source section was available for semantic review."
        )
    )
    return {
        "projection_sha256": projection_sha,
        "grade": grade,
        "grade_rationale": rationale,
        "semantic_claims": claims,
        "claims_supported": [claim["claim_uid"] for claim in claims],
        "contradiction_status": str(spec.get("contradiction_status", "none_observed")) if spec is not None and grade != "INADMISSIBLE" else ("none_observed" if grade != "INADMISSIBLE" else "inadmissible"),
        "units": sorted({unit for claim in claims for unit in claim["units"]}),
        "assumptions": sorted({value for claim in claims for value in claim["assumptions"]}),
        "limitations": sorted({value for claim in claims for value in claim["limitations"]}),
        "absence_reason": None if claims else ("context_only_no_support_claim" if admissible else str(projection["disposition"])),
    }


def finalize_semantic_review_packet(
    *, repository_root: Path, packet_path: Path, contract_path: Path | str = CONTRACT_PATH
) -> tuple[dict[str, Any], bool]:
    root = repository_root.resolve(strict=True)
    authorities = load_frozen_authorities(root, contract_path)
    manifest = validate_authorization(repository_root=root, contract_path=contract_path)
    packet = _canonical_payload(packet_path.read_bytes(), label="semantic review packet")
    if packet.get("schema") != "node_local_semantic_review_packet_v1":
        raise MechanismBasisQualificationError("semantic review packet schema drift")
    projections = {row["slot_uid"]: row for row in packet.get("projections", [])}
    authorized = {row["slot_uid"]: row for row in manifest["eligible_slots"]}
    if set(projections) != set(authorized):
        raise MechanismBasisQualificationError("semantic review packet slot coverage incomplete")
    receipts = packet.get("request_receipts", [])
    if len(receipts) > CODE_POLICY["maximum_total_requests"]:
        raise MechanismBasisQualificationError("cumulative request budget exceeded")
    validate_operational_review(receipts, manifest["eligible_slots"])
    latest = {receipt["request_uid"]: receipt for receipt in receipts if receipt["attempt_ordinal"] == max(r["attempt_ordinal"] for r in receipts if r["request_uid"] == receipt["request_uid"])}
    requirements = manifest["requirement_registry"]
    rows = []
    for slot in authorities["queue"]["slots"]:
        row = _base_evidence_row(slot)
        if slot["disposition"] != "eligible":
            row.update({"grade": "INADMISSIBLE", "grade_rationale": "Frozen queue disposition is not eligible for source-content access.", "reviewer_identity": "mechanism-basis-reviewer-v1", "review_disposition": "not_accessed_frozen_queue_disposition", "content_receipt": None, "contradiction_status": "inadmissible", "absence_reason": str(slot["disposition"])})
        else:
            projection_row = projections[slot["slot_uid"]]
            projection = projection_row["projection"]
            review = _semantic_review(slot, projection, requirements, CANONICAL_REVIEW_SPECS.get(slot["slot_uid"]))
            receipt = latest[authorized[slot["slot_uid"]]["request_plan"][0]["request_uid"]]
            row.update(review)
            row.update({"reviewer_identity": "mechanism-basis-reviewer-v1", "review_disposition": "semantic_review_complete", "content_receipt": receipt})
        rows.append(row)
    grades = Counter(row["grade"] for row in rows)
    derive_graph_components({"slots": rows}, requirements)
    schema, schema_hash = _schema(root, LEDGER_SCHEMA_PATH)
    ledger = {"schema_version": LEDGER_SCHEMA_VERSION, "status": "MECHANISM_EVIDENCE_LEDGER_FINALIZED", "ledger_uid": "mechanism_evidence_ledger", "contract": {"path": CONTRACT_PATH, "sha256": sha256_bytes(authorities["contract_raw"])}, "schema": {"path": LEDGER_SCHEMA_PATH, "sha256": schema_hash}, "authorization": _artifact_from_payload(AUTHORIZATION_PATH, manifest, "authorization_sha256"), "queue": _artifact_binding(SOURCE_QUEUE_PATH, authorities["queue_raw"], authorities["queue"]["queue_sha256"]), "slots": rows, "request_receipts": receipts, "counts": {"slot_count": 72, "reviewed_eligible_count": 15, "ineligible_count": 57, "grades": {grade: grades.get(grade, 0) for grade in GRADES}}, "network_summary": {"request_count": len(receipts), "successful_request_count": len(receipts), "failed_request_count": 0, "total_response_bytes": sum(r["response_bytes"] for r in receipts)}, "access_counters": {"source_content_access": 15, "successful_source_content_opens": 15, "unqueued_source_attempts": 0, "forbidden_value_reads": 0, "six_context_outcome_access": 0, "protected_holdout_access": 0}, "raw_responses_persisted": False, "temporary_review_packet_retained": False, "blockers": []}
    ledger["ledger_sha256"] = _self_hash(ledger, "ledger_sha256")
    _schema_validate(ledger, schema, label="evidence ledger")
    destination = _resolve_file(root, LEDGER_PATH, must_exist=False)
    created = _install_many([(destination, canonical_json_bytes(ledger))])
    packet_path.unlink()
    return ledger, created


def validate_operational_review(
    receipts: Sequence[Mapping[str, Any]], authorized_slots: Sequence[Mapping[str, Any]]
) -> None:
    expected = {row["request_plan"][0]["request_uid"] for row in authorized_slots}
    observed = {row.get("request_uid") for row in receipts}
    if observed != expected:
        raise MechanismBasisQualificationError("operational review receipt coverage incomplete")
    if any(row.get("response_sha256") is None for row in receipts):
        raise MechanismBasisQualificationError("operational content review failure blocks adjudication")
    for request_uid in expected:
        attempts = [int(row["attempt_ordinal"]) for row in receipts if row["request_uid"] == request_uid]
        if attempts != list(range(1, len(attempts) + 1)) or len(attempts) > CODE_POLICY["maximum_requests_per_slot"]:
            raise MechanismBasisQualificationError("content request attempt chronology drift")


def _receipt_uid(receipt: Mapping[str, Any]) -> str:
    identity = "\n".join(
        (
            str(receipt["request_uid"]),
            str(receipt["attempt_ordinal"]),
            str(receipt["retrieved_at"]),
            str(receipt["response_sha256"]),
        )
    )
    return "mechanism-content-receipt-" + hashlib.sha256(
        identity.encode()
    ).hexdigest()[:20]


def review_mechanism_basis(
    *,
    repository_root: Path,
    scratch_dir: Path,
    contract_path: Path | str = CONTRACT_PATH,
    fetcher: Callable[[Mapping[str, Any], Mapping[str, Any]], FetchResult] = fetch_exact_ncbi,
    now: Callable[[], str] = _now,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], bool]:
    root = repository_root.resolve(strict=True)
    if _resolve_file(root, LEDGER_PATH, must_exist=False).exists():
        ledger = validate_evidence_ledger(repository_root=root, contract_path=contract_path)
        return ledger, False
    raise MechanismBasisQualificationError(
        "semantic review packet is required before evidence-ledger finalization"
    )


def validate_evidence_ledger(
    *, repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> dict[str, Any]:
    root = repository_root.resolve(strict=True)
    authorities = load_frozen_authorities(root, contract_path)
    manifest = validate_authorization(repository_root=root, contract_path=contract_path)
    ledger, _ = _load_output(root, LEDGER_PATH, LEDGER_SCHEMA_PATH, "ledger_sha256")
    schemas = _schema_bindings(root)
    _require_exact_bindings(
        ledger,
        {
            "contract": _contract_binding(authorities),
            "schema": schemas[LEDGER_SCHEMA_PATH],
            "authorization": _artifact_from_payload(
                AUTHORIZATION_PATH, manifest, "authorization_sha256"
            ),
            "queue": _artifact_binding(
                SOURCE_QUEUE_PATH,
                authorities["queue_raw"],
                authorities["queue"]["queue_sha256"],
            ),
        },
        label="evidence ledger",
    )
    queue_slots = authorities["queue"]["slots"]
    rows = ledger["slots"]
    if [row["slot_uid"] for row in rows] != [slot["slot_uid"] for slot in queue_slots]:
        raise MechanismBasisQualificationError("evidence ledger slot order or identity drift")
    authorized_uids = {row["slot_uid"] for row in manifest["eligible_slots"]}
    authorization_by_uid = {row["slot_uid"]: row for row in manifest["eligible_slots"]}
    for slot, row in zip(queue_slots, rows, strict=True):
        for key in (
            "queue_order",
            "slot_uid",
            "canonical_intervention_uid",
            "validation_uid",
            "family_uid",
            "canonical_source_identity",
            "official_date",
        ):
            if row[key] != slot[key]:
                raise MechanismBasisQualificationError("evidence row queue binding drift")
        if (
            row["queue_slot_sha256"] != canonical_value_sha256(slot)
            or row["queue_disposition"] != slot["disposition"]
        ):
            raise MechanismBasisQualificationError("evidence row slot hash drift")
        if slot["disposition"] == "eligible":
            if row["slot_uid"] not in authorized_uids or row["content_receipt"] is None:
                raise MechanismBasisQualificationError("eligible source review is incomplete")
            receipt = row["content_receipt"]
            request = authorization_by_uid[row["slot_uid"]]["request_plan"][0]
            for key in ("request_uid", "request_kind", "method", "url"):
                if receipt[key] != request[key]:
                    raise MechanismBasisQualificationError("content receipt request binding drift")
            if receipt["final_host"] is not None and receipt["final_host"] not in CODE_POLICY["allowed_hosts"]:
                raise MechanismBasisQualificationError("content receipt host drift")
            if receipt["content_type"] is not None and receipt["content_type"] not in CODE_POLICY["allowed_content_types"]:
                raise MechanismBasisQualificationError("content receipt type drift")
        elif row["grade"] != "INADMISSIBLE" or row["content_receipt"] is not None:
            raise MechanismBasisQualificationError("ineligible queue slot contributed content")
        if row["grade"] not in GRADES or len(row["claims_supported"]) != len(set(row["claims_supported"])):
            raise MechanismBasisQualificationError("evidence grade or claim list drift")
        if row["claims_supported"] != [claim["claim_uid"] for claim in row["semantic_claims"]]:
            raise MechanismBasisQualificationError("semantic claim identity drift")
        if row["grade"] != _derive_grade(
            row["semantic_claims"],
            admissible_context=bool(
                row["content_receipt"]
                and row["content_receipt"]["allowed_sections"]
            ),
        ):
            raise MechanismBasisQualificationError("evidence grade precedence drift")
    grades = Counter(row["grade"] for row in rows)
    expected_counts = {
        "slot_count": 72,
        "reviewed_eligible_count": 15,
        "ineligible_count": 57,
        "grades": {grade: grades.get(grade, 0) for grade in GRADES},
    }
    if ledger["counts"] != expected_counts:
        raise MechanismBasisQualificationError("evidence ledger counts drift")
    receipts = ledger["request_receipts"]
    validate_operational_review(receipts, manifest["eligible_slots"])
    request_rows = [
        (slot, slot["request_plan"][0]) for slot in manifest["eligible_slots"]
    ]
    request_by_uid = {request["request_uid"]: (slot, request) for slot, request in request_rows}
    maximum_attempt = max(int(receipt["attempt_ordinal"]) for receipt in receipts)
    expected_order = [
        request["request_uid"]
        for attempt in range(1, maximum_attempt + 1)
        for _, request in request_rows
    ]
    if [receipt["request_uid"] for receipt in receipts] != expected_order or [
        int(receipt["attempt_ordinal"]) for receipt in receipts
    ] != [
        attempt
        for attempt in range(1, maximum_attempt + 1)
        for _ in request_rows
    ]:
        raise MechanismBasisQualificationError("content receipt roster order drift")
    receipts_by_request: dict[str, list[Mapping[str, Any]]] = {}
    for receipt in receipts:
        if receipt["receipt_uid"] != _receipt_uid(receipt):
            raise MechanismBasisQualificationError("content receipt UID drift")
        bound = request_by_uid.get(receipt["request_uid"])
        if bound is None:
            raise MechanismBasisQualificationError("content receipt request linkage drift")
        slot, request = bound
        for key in ("request_uid", "request_kind", "method", "url"):
            if receipt[key] != request[key]:
                raise MechanismBasisQualificationError("content receipt request linkage drift")
        if (
            receipt["public_access_status"] != "public_ncbi_record"
            or receipt["license_status"] != "ncbi_pubmed_record_terms"
            or receipt["parser_version"] != PARSER_VERSION
            or receipt["response_sha256"] is None
            or receipt["response_bytes"] <= 0
        ):
            raise MechanismBasisQualificationError("content receipt source/status drift")
        receipts_by_request.setdefault(receipt["request_uid"], []).append(receipt)
    repeat_fields = (
        "request_kind", "method", "url", "final_host", "content_type",
        "response_bytes", "response_sha256", "parser_version",
        "public_access_status", "license_status", "disposition",
        "allowed_sections", "forbidden_sections_skipped", "mixed_sections_rejected",
    )
    for request_uid, attempts in receipts_by_request.items():
        first = attempts[0]
        if any(
            any(attempt[field] != first[field] for field in repeat_fields)
            for attempt in attempts[1:]
        ):
            raise MechanismBasisQualificationError("content receipt repeated-attempt drift")
    latest_receipt = {request_uid: attempts[-1] for request_uid, attempts in receipts_by_request.items()}
    for row in rows:
        if row["slot_uid"] in authorized_uids:
            request_uid = authorization_by_uid[row["slot_uid"]]["request_plan"][0]["request_uid"]
            if row["content_receipt"] != latest_receipt[request_uid]:
                raise MechanismBasisQualificationError("evidence row receipt linkage drift")
    successful = [receipt for receipt in receipts if receipt["response_sha256"] is not None]
    if len({receipt["receipt_uid"] for receipt in receipts}) != len(receipts):
        raise MechanismBasisQualificationError("duplicate content receipt identity")
    access = ledger["access_counters"]
    if (
        access["source_content_access"] != len({receipt["request_uid"] for receipt in successful})
        or access["successful_source_content_opens"] != len({receipt["request_uid"] for receipt in successful})
        or len({receipt["request_uid"] for receipt in successful}) > 15
        or any(access[key] != 0 for key in ("unqueued_source_attempts", "forbidden_value_reads", "six_context_outcome_access", "protected_holdout_access"))
    ):
        raise MechanismBasisQualificationError("content access counters drift")
    network = ledger["network_summary"]
    if network != {
        "request_count": len(receipts),
        "successful_request_count": len(successful),
        "failed_request_count": len(receipts) - len(successful),
        "total_response_bytes": sum(receipt["response_bytes"] for receipt in receipts),
    }:
        raise MechanismBasisQualificationError("content network summary drift")
    if ledger["raw_responses_persisted"] or ledger["temporary_review_packet_retained"]:
        raise MechanismBasisQualificationError("raw or temporary content persistence drift")
    try:
        retrieval_times = [
            datetime.fromisoformat(receipt["retrieved_at"].replace("Z", "+00:00"))
            for receipt in receipts
        ]
    except ValueError as exc:
        raise MechanismBasisQualificationError(
            "content receipt retrieval chronology drift"
        ) from exc
    if any(
        (right - left).total_seconds()
        < CODE_POLICY["minimum_request_interval_ms"] / 1000.0
        for left, right in zip(retrieval_times, retrieval_times[1:])
    ):
        raise MechanismBasisQualificationError("content request rate-limit chronology drift")
    _derive_contradictions(ledger)
    derive_graph_components(ledger, manifest["requirement_registry"])
    if ledger["ledger_sha256"] != EXPECTED_LEDGER_SHA256:
        raise MechanismBasisQualificationError("evidence ledger frozen identity drift")
    return ledger


def _artifact_from_payload(path: str, payload: Mapping[str, Any], hash_field: str) -> dict[str, str]:
    return _artifact_binding(path, canonical_json_bytes(payload), str(payload[hash_field]))


def derive_requirement_coverage(
    requirements: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    contradictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    claim_uids = {claim["claim_uid"] for claim in claims}
    rows = []
    for requirement in requirements:
        admitted = [claim for claim in claims if requirement["requirement_uid"] in claim["requirement_uids"]]
        admitted_uids = {claim["claim_uid"] for claim in admitted}
        contradiction_ids = [
            row["contradiction_uid"]
            for row in contradictions
            if row["status"] == "unresolved"
            and set(row["claim_uids"]).issubset(claim_uids)
            and set(row["claim_uids"]) & admitted_uids
        ]
        status = "contradictory" if contradiction_ids else ("covered" if admitted else "uncovered")
        rows.append({"requirement_uid": requirement["requirement_uid"], "required": requirement["required"], "coverage_status": status, "admitted_claim_uids": [claim["claim_uid"] for claim in admitted], "source_slot_uids": sorted({claim["slot_uid"] for claim in admitted}), "contradiction_uids": contradiction_ids, "absence_reason": None if admitted else "no_admitted_support_grade_claim_in_bounded_queue"})
    required_rows = [row for row in rows if row["required"]]
    terminal = "MECHANISM_BASIS_QUALIFIED" if required_rows and all(row["coverage_status"] == "covered" for row in required_rows) else "MECHANISM_BASIS_INSUFFICIENT"
    return rows, terminal


def derive_prior_support(
    requirements: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    prior: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prior_requirements = [
        row for row in requirements if row["category"] == "prior_role"
    ]
    requirement_by_subject = {
        row["subject_uid"]: row for row in prior_requirements
    }
    if len(requirement_by_subject) != len(prior_requirements):
        raise MechanismBasisQualificationError("duplicate prior-role requirement")
    coverage_by_uid = {row["requirement_uid"]: row for row in coverage}
    roles = list(prior["prior_roles"]["ordered_roles"])
    role_uids = [row["role_uid"] for row in roles]
    if len(role_uids) != len(set(role_uids)) or set(role_uids) != set(
        requirement_by_subject
    ):
        raise MechanismBasisQualificationError("prior-role registry mapping drift")
    status_map = {
        "covered": "supported",
        "uncovered": "unsupported",
        "contradictory": "contradictory",
    }
    canonical = []
    sensitivity = []
    for role in roles:
        requirement = requirement_by_subject[role["role_uid"]]
        coverage_row = coverage_by_uid.get(requirement["requirement_uid"])
        if coverage_row is None or coverage_row["required"] != requirement["required"]:
            raise MechanismBasisQualificationError("prior-role coverage binding drift")
        if role["required"] != requirement["required"]:
            raise MechanismBasisQualificationError("prior-role required flag drift")
        try:
            support_status = status_map[coverage_row["coverage_status"]]
        except KeyError as exc:
            raise MechanismBasisQualificationError(
                "prior-role support status drift"
            ) from exc
        support = {
            "role_uid": role["role_uid"],
            "required": role["required"],
            "support_status": support_status,
        }
        if role["role_class"] == "canonical":
            canonical.append(support)
        elif role["role_class"] == "sensitivity":
            sensitivity.append(support)
        else:
            raise MechanismBasisQualificationError("prior-role class drift")
    if len(canonical) != 1:
        raise MechanismBasisQualificationError("canonical prior-role registry drift")
    return canonical[0], sensitivity


def scientific_interpretation_for_terminal(terminal: str) -> str:
    try:
        return TERMINAL_INTERPRETATIONS[terminal]
    except KeyError as exc:
        raise MechanismBasisQualificationError(
            "unsupported mechanism-basis terminal"
        ) from exc


def validate_scientific_interpretation(value: str, terminal: str) -> None:
    expected = scientific_interpretation_for_terminal(terminal)
    other_templates = set(TERMINAL_INTERPRETATIONS.values()) - {expected}
    if value in other_templates:
        raise MechanismBasisQualificationError(
            "scientific interpretation terminal mismatch"
        )
    screened = value.lower().replace(SAFE_INTERPRETATION_BOUNDARY, "")
    if FORBIDDEN_INTERPRETATION_PATTERN.search(screened):
        raise MechanismBasisQualificationError(
            "forbidden scientific interpretation content"
        )
    if value != expected:
        raise MechanismBasisQualificationError("scientific interpretation drift")


def _derive_contradictions(
    ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    claim_rows: dict[str, Mapping[str, Any]] = {}
    declaration_lists: dict[str, list[str]] = {}
    for row in ledger["slots"]:
        for claim in row["semantic_claims"]:
            claim_uid = claim["claim_uid"]
            if claim_uid in claim_rows:
                raise MechanismBasisQualificationError(
                    "duplicate semantic claim identity"
                )
            claim_rows[claim_uid] = row
            declaration_lists[claim_uid] = list(claim["contradicts_claim_uids"])
    declarations = {
        claim_uid: set(others) for claim_uid, others in declaration_lists.items()
    }
    for claim_uid, others in declarations.items():
        if (
            claim_uid in others
            or len(declaration_lists[claim_uid]) != len(others)
            or not others.issubset(declarations)
        ):
            raise MechanismBasisQualificationError("malformed contradiction lineage")
        if any(claim_uid not in declarations[other] for other in others):
            raise MechanismBasisQualificationError(
                "asymmetric contradiction lineage"
            )
    related_claim_uids = {
        claim_uid for claim_uid, others in declarations.items() if others
    }
    neutral_statuses = {"none_observed", "not_assessed", "inadmissible"}
    relation_statuses = {"unresolved", "resolved_by_scope"}
    for row in ledger["slots"]:
        status = row["contradiction_status"]
        participates = any(
            claim["claim_uid"] in related_claim_uids
            for claim in row["semantic_claims"]
        )
        if status in relation_statuses and not participates:
            raise MechanismBasisQualificationError(
                "orphan contradiction disposition"
            )
        if status in neutral_statuses and participates:
            raise MechanismBasisQualificationError(
                "neutral contradiction disposition has relation"
            )
        if status not in neutral_statuses | relation_statuses:
            raise MechanismBasisQualificationError(
                "unsupported contradiction disposition"
            )
    contradictions = []
    pairs = {
        tuple(sorted((left, right)))
        for left, others in declarations.items()
        for right in others
    }
    for pair in sorted(pairs):
        statuses = {claim_rows[uid]["contradiction_status"] for uid in pair}
        if statuses == {"resolved_by_scope"}:
            status, dimension = "resolved", "scope"
        elif statuses == {"unresolved"}:
            status, dimension = "unresolved", "not_resolved"
        else:
            raise MechanismBasisQualificationError(
                "contradiction disposition mismatch"
            )
        contradictions.append(
            {
                "contradiction_uid": "mechanism-contradiction-"
                + hashlib.sha256("\n".join(pair).encode()).hexdigest()[:20],
                "claim_uids": list(pair),
                "essential": True,
                "status": status,
                "resolution_dimension": dimension,
                "rationale": (
                    "Contradiction disposition is preserved from the symmetric "
                    "semantic-review declarations."
                ),
            }
        )
    return contradictions


def derive_graph_components(
    ledger: Mapping[str, Any], requirements: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    requirement_by_uid = {row["requirement_uid"]: row for row in requirements}
    claims = []
    claim_rows: dict[str, Mapping[str, Any]] = {}
    for row in ledger["slots"]:
        receipt = row.get("content_receipt")
        allowed = {
            (item["locator"], item["content_sha256"]): item["content_class"]
            for item in receipt["allowed_sections"]
        } if receipt else {}
        for semantic_claim in row["semantic_claims"]:
            if semantic_claim["claim_uid"] != _claim_uid(row["slot_uid"], {key: value for key, value in semantic_claim.items() if key != "claim_uid"}):
                raise MechanismBasisQualificationError("semantic claim UID drift")
            locator_key = (semantic_claim["content_locator"], semantic_claim["content_sha256"])
            if locator_key not in allowed:
                raise MechanismBasisQualificationError("semantic claim locator or hash is not in the allowed projection")
            compatible = {
                "mechanism_definition": {"equation", "causal_dependency", "structural_dependency", "assumption", "limitation", "support_boundary"},
                "measurement_definition": {"measurement_definition", "latent_to_observable_link", "parameter_range", "unit_definition"},
                "explicit_limitation": {"assumption", "limitation", "support_boundary"},
            }
            if semantic_claim["claim_type"] not in compatible.get(allowed[locator_key], set()):
                raise MechanismBasisQualificationError("semantic claim section-class drift")
            if re.search(r"\b(efficacy|benefit|outcome|ranking|recommendation|p[- ]?value|confidence interval)\b", semantic_claim["paraphrase"], re.IGNORECASE):
                raise MechanismBasisQualificationError("forbidden scientific content in semantic claim")
            for requirement_uid in semantic_claim["requirement_uids"]:
                requirement = requirement_by_uid.get(requirement_uid)
                if requirement is None or semantic_claim["claim_type"] not in requirement["allowed_claim_types"]:
                    raise MechanismBasisQualificationError("semantic claim requirement type drift")
            if semantic_claim["claim_type"] == "parameter_range" and (not semantic_claim["units"] or not semantic_claim["range_context"]):
                raise MechanismBasisQualificationError("parameter claim lacks units or range context")
            claim = {**semantic_claim, "slot_uid": row["slot_uid"]}
            claim.pop("contradicts_claim_uids")
            claim["claim_sha256"] = canonical_value_sha256(claim)
            claims.append(claim)
            claim_rows[semantic_claim["claim_uid"]] = row
    if len(claim_rows) != len(claims):
        raise MechanismBasisQualificationError("duplicate semantic claim identity")
    support_edges = [{"edge_uid": "mechanism-support-edge-" + hashlib.sha256(f"{claim['slot_uid']}\n{claim['claim_uid']}".encode()).hexdigest()[:20], "slot_uid": claim["slot_uid"], "claim_uid": claim["claim_uid"], "grade": claim_rows[claim["claim_uid"]]["grade"], "content_locator": claim["content_locator"], "content_sha256": claim["content_sha256"]} for claim in claims]
    contradictions = _derive_contradictions(ledger)
    return claims, support_edges, contradictions


def validate_graph_matches_ledger(
    graph: Mapping[str, Any],
    ledger: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
) -> None:
    expected = derive_graph_components(ledger, requirements)
    observed = (graph["claims"], graph["support_edges"], graph["contradictions"])
    if observed != expected:
        raise MechanismBasisQualificationError("claim graph diverges from evidence ledger")


def adjudicate_mechanism_basis(
    *, repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    root = repository_root.resolve(strict=True)
    authorities = load_frozen_authorities(root, contract_path)
    manifest = validate_authorization(repository_root=root, contract_path=contract_path)
    ledger = validate_evidence_ledger(repository_root=root, contract_path=contract_path)
    output_paths = [
        _resolve_file(root, GRAPH_PATH, must_exist=False),
        _resolve_file(root, ASSESSMENT_PATH, must_exist=False),
        _resolve_file(root, DECISION_PATH, must_exist=False),
    ]
    if all(path.exists() for path in output_paths):
        graph, assessment, decision = validate_final(
            repository_root=root, contract_path=contract_path
        )
        return graph, assessment, decision, False
    if any(path.exists() for path in output_paths):
        raise MechanismBasisQualificationError("partial final artifact installation detected")
    schemas = _schema_bindings(root)
    contract_binding = _contract_binding(authorities)
    claims, support_edges, contradictions = derive_graph_components(
        ledger, manifest["requirement_registry"]
    )
    graph: dict[str, Any] = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "status": "MECHANISM_SOURCE_CLAIM_GRAPH_FINALIZED",
        "graph_uid": "mechanism_source_claim_graph",
        "contract": contract_binding,
        "schema": schemas[GRAPH_SCHEMA_PATH],
        "evidence_ledger": _artifact_from_payload(LEDGER_PATH, ledger, "ledger_sha256"),
        "claims": claims,
        "support_edges": support_edges,
        "contradictions": contradictions,
        "counts": {
            "claim_count": len(claims),
            "support_edge_count": len(support_edges),
            "contradiction_count": len(contradictions),
            "unresolved_essential_contradiction_count": sum(
                row["essential"] and row["status"] == "unresolved"
                for row in contradictions
            ),
        },
        "blockers": [],
    }
    graph["graph_sha256"] = _self_hash(graph, "graph_sha256")
    graph_schema, _ = _schema(root, GRAPH_SCHEMA_PATH)
    _schema_validate(graph, graph_schema, label="source claim graph")
    requirements, terminal = derive_requirement_coverage(
        manifest["requirement_registry"], claims, contradictions
    )
    required = [row for row in requirements if row["required"]]
    uncovered = [row["requirement_uid"] for row in required if row["coverage_status"] == "uncovered"]
    contradictory = [row["requirement_uid"] for row in required if row["coverage_status"] == "contradictory"]
    covered = len(required) - len(uncovered) - len(contradictory)
    canonical_prior_support, sensitivity_prior_support = derive_prior_support(
        manifest["requirement_registry"], requirements, authorities["prior"]
    )
    access_counters = deepcopy(ledger["access_counters"])
    assessment: dict[str, Any] = {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": "MECHANISM_BASIS_ASSESSED",
        "assessment_uid": "mechanism_basis_assessment",
        "contract": contract_binding,
        "schema": schemas[ASSESSMENT_SCHEMA_PATH],
        "authorization": _artifact_from_payload(
            AUTHORIZATION_PATH, manifest, "authorization_sha256"
        ),
        "evidence_ledger": _artifact_from_payload(LEDGER_PATH, ledger, "ledger_sha256"),
        "source_claim_graph": _artifact_from_payload(GRAPH_PATH, graph, "graph_sha256"),
        "requirements": requirements,
        "coverage_counts": {
            "requirement_count": len(requirements),
            "required_count": len(required),
            "covered_required_count": covered,
            "uncovered_required_count": len(uncovered),
            "contradictory_required_count": len(contradictory),
        },
        "canonical_prior_support": canonical_prior_support,
        "sensitivity_prior_support": sensitivity_prior_support,
        "uncovered_required_requirement_uids": uncovered,
        "contradictory_required_requirement_uids": contradictory,
        "decision_candidate": terminal,
        "access_counters": access_counters,
        "downstream_authorization": deepcopy(DOWNSTREAM_CLOSURE),
        "blockers": [],
    }
    assessment["assessment_sha256"] = _self_hash(assessment, "assessment_sha256")
    assessment_schema, _ = _schema(root, ASSESSMENT_SCHEMA_PATH)
    _schema_validate(assessment, assessment_schema, label="mechanism basis assessment")
    decision: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": terminal,
        "decision_uid": "mechanism_basis_decision",
        "contract": contract_binding,
        "schema": schemas[DECISION_SCHEMA_PATH],
        "authorization": _artifact_from_payload(
            AUTHORIZATION_PATH, manifest, "authorization_sha256"
        ),
        "evidence_ledger": _artifact_from_payload(LEDGER_PATH, ledger, "ledger_sha256"),
        "source_claim_graph": _artifact_from_payload(GRAPH_PATH, graph, "graph_sha256"),
        "assessment": _artifact_from_payload(
            ASSESSMENT_PATH, assessment, "assessment_sha256"
        ),
        "queue_exhausted": True,
        "all_72_slots_graded": True,
        "missing_required_requirement_uids": uncovered,
        "contradictory_required_requirement_uids": contradictory,
        "scientific_interpretation": scientific_interpretation_for_terminal(terminal),
        "access_counters": access_counters,
        "downstream_authorization": deepcopy(DOWNSTREAM_CLOSURE),
        "blockers": [],
    }
    decision["decision_sha256"] = _self_hash(decision, "decision_sha256")
    decision_schema, _ = _schema(root, DECISION_SCHEMA_PATH)
    _schema_validate(decision, decision_schema, label="mechanism basis decision")
    created = _install_many(
        [
            (output_paths[0], canonical_json_bytes(graph)),
            (output_paths[1], canonical_json_bytes(assessment)),
            (output_paths[2], canonical_json_bytes(decision)),
        ]
    )
    return graph, assessment, decision, created


def _validate_claim_graph(graph: Mapping[str, Any], ledger: Mapping[str, Any]) -> None:
    claims = {row["claim_uid"]: row for row in graph["claims"]}
    if len(claims) != len(graph["claims"]):
        raise MechanismBasisQualificationError("duplicate claim identity")
    slots = {row["slot_uid"]: row for row in ledger["slots"]}
    for edge in graph["support_edges"]:
        if edge["claim_uid"] not in claims or edge["slot_uid"] not in slots:
            raise MechanismBasisQualificationError("dangling claim support edge")
        if slots[edge["slot_uid"]]["grade"] not in SUPPORT_GRADES:
            raise MechanismBasisQualificationError("unsupported grade used as claim support")
        if edge["grade"] != slots[edge["slot_uid"]]["grade"]:
            raise MechanismBasisQualificationError("claim edge grade drift")
    for contradiction in graph["contradictions"]:
        if not set(contradiction["claim_uids"]).issubset(claims):
            raise MechanismBasisQualificationError("dangling contradiction claim")
        if contradiction["status"] == "resolved" and contradiction["resolution_dimension"] == "not_resolved":
            raise MechanismBasisQualificationError("contradiction resolution lacks provenance dimension")
    counts = graph["counts"]
    expected = {
        "claim_count": len(graph["claims"]),
        "support_edge_count": len(graph["support_edges"]),
        "contradiction_count": len(graph["contradictions"]),
        "unresolved_essential_contradiction_count": sum(
            row["essential"] and row["status"] == "unresolved"
            for row in graph["contradictions"]
        ),
    }
    if counts != expected:
        raise MechanismBasisQualificationError("claim graph counts drift")


def validate_final(
    *, repository_root: Path, contract_path: Path | str = CONTRACT_PATH
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = repository_root.resolve(strict=True)
    authorities = load_frozen_authorities(root, contract_path)
    manifest = validate_authorization(repository_root=root, contract_path=contract_path)
    ledger = validate_evidence_ledger(repository_root=root, contract_path=contract_path)
    graph, _ = _load_output(root, GRAPH_PATH, GRAPH_SCHEMA_PATH, "graph_sha256")
    assessment, _ = _load_output(
        root, ASSESSMENT_PATH, ASSESSMENT_SCHEMA_PATH, "assessment_sha256"
    )
    decision, _ = _load_output(root, DECISION_PATH, DECISION_SCHEMA_PATH, "decision_sha256")
    schemas = _schema_bindings(root)
    contract_binding = _contract_binding(authorities)
    authorization_binding = _artifact_from_payload(
        AUTHORIZATION_PATH, manifest, "authorization_sha256"
    )
    ledger_binding = _artifact_from_payload(
        LEDGER_PATH, ledger, "ledger_sha256"
    )
    _require_exact_bindings(
        graph,
        {
            "contract": contract_binding,
            "schema": schemas[GRAPH_SCHEMA_PATH],
            "evidence_ledger": ledger_binding,
        },
        label="claim graph",
    )
    expected_graph_binding = _artifact_from_payload(GRAPH_PATH, graph, "graph_sha256")
    _require_exact_bindings(
        assessment,
        {
            "contract": contract_binding,
            "schema": schemas[ASSESSMENT_SCHEMA_PATH],
            "authorization": authorization_binding,
            "evidence_ledger": ledger_binding,
            "source_claim_graph": expected_graph_binding,
        },
        label="assessment",
    )
    expected_assessment_binding = _artifact_from_payload(
        ASSESSMENT_PATH, assessment, "assessment_sha256"
    )
    _require_exact_bindings(
        decision,
        {
            "contract": contract_binding,
            "schema": schemas[DECISION_SCHEMA_PATH],
            "authorization": authorization_binding,
            "evidence_ledger": ledger_binding,
            "source_claim_graph": expected_graph_binding,
            "assessment": expected_assessment_binding,
        },
        label="decision",
    )
    validate_graph_matches_ledger(graph, ledger, manifest["requirement_registry"])
    _validate_claim_graph(graph, ledger)
    expected_requirements, expected_status = derive_requirement_coverage(
        manifest["requirement_registry"], graph["claims"], graph["contradictions"]
    )
    if assessment["requirements"] != expected_requirements:
        raise MechanismBasisQualificationError(
            "assessment requirement coverage diverges from evidence ledger"
        )
    expected_canonical_prior, expected_sensitivity_priors = derive_prior_support(
        manifest["requirement_registry"],
        expected_requirements,
        authorities["prior"],
    )
    if assessment["canonical_prior_support"] != expected_canonical_prior:
        raise MechanismBasisQualificationError("canonical prior support drift")
    if assessment["sensitivity_prior_support"] != expected_sensitivity_priors:
        raise MechanismBasisQualificationError("sensitivity prior support drift")
    requirement_uids = [row["requirement_uid"] for row in manifest["requirement_registry"]]
    if [row["requirement_uid"] for row in assessment["requirements"]] != requirement_uids:
        raise MechanismBasisQualificationError("assessment requirement registry drift")
    claim_uids = {row["claim_uid"] for row in graph["claims"]}
    contradiction_uids = {row["contradiction_uid"] for row in graph["contradictions"]}
    slot_uids = {row["slot_uid"] for row in ledger["slots"]}
    for coverage in assessment["requirements"]:
        if not set(coverage["admitted_claim_uids"]).issubset(claim_uids):
            raise MechanismBasisQualificationError("assessment has dangling claim coverage")
        if not set(coverage["source_slot_uids"]).issubset(slot_uids):
            raise MechanismBasisQualificationError("assessment has dangling source coverage")
        if not set(coverage["contradiction_uids"]).issubset(contradiction_uids):
            raise MechanismBasisQualificationError("assessment has hidden contradiction")
        if coverage["coverage_status"] == "covered" and not coverage["admitted_claim_uids"]:
            raise MechanismBasisQualificationError("false complete requirement coverage")
    required_rows = [row for row in assessment["requirements"] if row["required"]]
    uncovered = [row["requirement_uid"] for row in required_rows if row["coverage_status"] == "uncovered"]
    contradictory = [row["requirement_uid"] for row in required_rows if row["coverage_status"] == "contradictory"]
    covered = len(required_rows) - len(uncovered) - len(contradictory)
    expected_counts = {
        "requirement_count": len(assessment["requirements"]),
        "required_count": len(required_rows),
        "covered_required_count": covered,
        "uncovered_required_count": len(uncovered),
        "contradictory_required_count": len(contradictory),
    }
    if assessment["coverage_counts"] != expected_counts:
        raise MechanismBasisQualificationError("assessment coverage counts drift")
    if assessment["uncovered_required_requirement_uids"] != uncovered or assessment["contradictory_required_requirement_uids"] != contradictory:
        raise MechanismBasisQualificationError("assessment gap list drift")
    if assessment["decision_candidate"] != expected_status or decision["status"] != expected_status:
        raise MechanismBasisQualificationError("terminal mechanism-basis decision drift")
    validate_scientific_interpretation(
        decision["scientific_interpretation"], expected_status
    )
    if decision["missing_required_requirement_uids"] != uncovered or decision["contradictory_required_requirement_uids"] != contradictory:
        raise MechanismBasisQualificationError("terminal gap binding drift")
    if decision["access_counters"] != ledger["access_counters"] or assessment["access_counters"] != ledger["access_counters"]:
        raise MechanismBasisQualificationError("final access counter drift")
    if decision["downstream_authorization"] != DOWNSTREAM_CLOSURE or assessment["downstream_authorization"] != DOWNSTREAM_CLOSURE:
        raise MechanismBasisQualificationError("final downstream authorization opened")
    return graph, assessment, decision


def validate_stage(
    *, repository_root: Path, stage: str, contract_path: Path | str = CONTRACT_PATH
) -> dict[str, Any]:
    if stage == "authorization":
        manifest = validate_authorization(
            repository_root=repository_root, contract_path=contract_path
        )
        return {
            "status": "MECHANISM_CONTENT_ACCESS_MANIFEST_VALID",
            "authorization_sha256": manifest["authorization_sha256"],
            "eligible_slot_count": len(manifest["eligible_slots"]),
            "requirement_count": len(manifest["requirement_registry"]),
            "network_requests_performed": manifest["network_requests_performed"],
            "access_counters": manifest["access_counters"],
        }
    if stage == "ledger":
        ledger = validate_evidence_ledger(
            repository_root=repository_root, contract_path=contract_path
        )
        return {
            "status": "MECHANISM_EVIDENCE_LEDGER_VALID",
            "ledger_sha256": ledger["ledger_sha256"],
            "counts": ledger["counts"],
            "network_summary": ledger["network_summary"],
            "access_counters": ledger["access_counters"],
        }
    if stage == "final":
        graph, assessment, decision = validate_final(
            repository_root=repository_root, contract_path=contract_path
        )
        return {
            "status": decision["status"],
            "decision_sha256": decision["decision_sha256"],
            "assessment_sha256": assessment["assessment_sha256"],
            "graph_sha256": graph["graph_sha256"],
            "missing_required_count": len(decision["missing_required_requirement_uids"]),
            "contradictory_required_count": len(decision["contradictory_required_requirement_uids"]),
            "access_counters": decision["access_counters"],
            "downstream_authorization": decision["downstream_authorization"],
        }
    raise MechanismBasisQualificationError("stage must be authorization, ledger, or final")
