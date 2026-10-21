"""Bind frozen negative authorities for the no-task-fit route.

This builder deliberately performs no route arbitration, inference, training,
calibration, or evaluation. It validates an exact nine-file allowlist, stages
the original bytes, and installs only byte-identical authority copies.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any

from jsonschema import Draft202012Validator
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_SCHEMA_VERSION = "neuronsifter-no-task-fit-parent-authority-contract-v1"
MANIFEST_SCHEMA_VERSION = "neuronsifter-no-task-fit-parent-authority-v1"
MANIFEST_STATUS = "NO_TASK_FIT_PARENT_AUTHORITIES_BOUND"
MANIFEST_UID = "no_task_fit_parent_authority_binding"
NEXT_ACTION = "arbitrate_no_task_fit_candidate_routes"
ARTIFACT_COUNT = 9

MANIFEST_PATH = "Output/audits/no_task_fit_parent_authority.json"
SCHEMA_PATH = "Libs/configs/schema/no_task_fit_parent_authority.schema.json"
CONTRACT_PATH = "Libs/configs/validation/no_task_fit_parent_authority.yaml"

EXPECTED_SOURCE_BRANCH = "round17i-dirty-root-preservation"
EXPECTED_SOURCE_HEAD = "b78a7e924971ac1a07905e9f18b6405b084bb7e6"
EXPECTED_SOURCE_ROLE = "live_root_read_only_negative_authority_source"
EXPECTED_SOURCE_TRACKING_STATE = "untracked"

EXPECTED_ARTIFACT_IDENTITIES: dict[str, tuple[str, str, str]] = {
    "Output/audits/frozen_zero_shot_method_authority.json": (
        "method_checkpoint_negative",
        "neuronsifter-zero-shot-method-authority-v1",
        "NO_FORMAL_ZERO_SHOT_CHECKPOINT",
    ),
    "Output/audits/frozen_zero_shot_method_manifest.json": (
        "method_checkpoint_negative",
        "neuronsifter-frozen-zero-shot-method-manifest-v1",
        "incomplete_not_executable",
    ),
    "Output/audits/zero_shot_protocol_executability_decision.json": (
        "method_checkpoint_negative",
        "neuronsifter-zero-shot-protocol-executability-decision-v1",
        "ZERO_SHOT_PROTOCOL_REMAINS_NON_EXECUTABLE",
    ),
    "Output/audits/prospective_development_data_authority.json": (
        "development_domain_negative",
        "neuronsifter-prospective-development-data-authority-v1",
        "NO_INDEPENDENT_PROSPECTIVE_TRAINING_DOMAIN",
    ),
    "Output/audits/prospective_twin_reconstruction_decision.json": (
        "development_domain_negative",
        "neuronsifter-prospective-twin-reconstruction-decision-v1",
        "NO_INDEPENDENT_PROSPECTIVE_TRAINING_DOMAIN",
    ),
    "Output/audits/prospective_development_data_intake.json": (
        "development_data_negative",
        "neuronsifter-prospective-development-data-intake-v1",
        "development_source_queue_exhausted",
    ),
    "Output/audits/prospective_development_foundation_decision.json": (
        "development_data_negative",
        "neuronsifter-prospective-development-foundation-decision-v1",
        "PUBLIC_PROSPECTIVE_DEVELOPMENT_DATA_INSUFFICIENT",
    ),
    "Output/audits/prospective_protocol_scientific_review.json": (
        "protocol_negative",
        "neuronsifter-prospective-protocol-scientific-review-v1",
        "PROTOCOL_NOT_FORMALLY_EXECUTABLE",
    ),
    "Output/audits/six_context_semantic_exclusion_audit.json": (
        "protocol_exclusion_negative",
        "neuronsifter-six-context-semantic-exclusion-audit-v1",
        "failed_incomplete_semantic_independence",
    ),
}

EXPECTED_FROZEN_ARTIFACT_CONTRACT: dict[str, dict[str, str | int]] = {
    "Output/audits/frozen_zero_shot_method_authority.json": {
        "sha256": "6e59323ad2773bb81417affc64780a2e07185811e71f0d4c0cac57e6cfbf1d21",
        "bytes": 13687,
        "required_fact_count": 14,
        "required_facts_sha256": "ac580a025dc52fec76073641a08bff819c3c8b956ba47bf3703ebcd6510f32f8",
        "record_sha256": "42d8622a505de00a5e9e395f28fa451e3071ae83c468ba59500044500dbaa734",
    },
    "Output/audits/frozen_zero_shot_method_manifest.json": {
        "sha256": "e95b001189ff3944a338d2b100b2aa7ca11143e5a3537615255d344ccf9a8bab",
        "bytes": 7071,
        "required_fact_count": 16,
        "required_facts_sha256": "f123b3e58fa56438e29f15e76d660eb9fd82a937b6ffb938c7a9e299002e3b8e",
        "record_sha256": "d241d0422a487eb0b88ecbea89b6de0df2a2c124cd07e892d9690c7795da0805",
    },
    "Output/audits/zero_shot_protocol_executability_decision.json": {
        "sha256": "7e8e88cbcfe0d46746ff7db3f51623b23066289c3bbba6c2f9978d9550c7de43",
        "bytes": 3047,
        "required_fact_count": 10,
        "required_facts_sha256": "ca63106bb271cedb6f9a2b096c80e27e5923ff1bbedafbe653a8be44d9fc9ae5",
        "record_sha256": "15fc72a7c5d9afcd552b904813b5f573dc1fccd207e92f0750f7475b9c5fb23e",
    },
    "Output/audits/prospective_development_data_authority.json": {
        "sha256": "eb3afeedd4e2dc277bd8538a8ac29b087fa05332d1f4fec75d4c75aa960bd2d8",
        "bytes": 13825,
        "required_fact_count": 5,
        "required_facts_sha256": "5e6cfd068c73eed3e6833bba7565dc09bd90cebdadcc607d515d66e0efed47f5",
        "record_sha256": "5b0d7a100a3a4a0f50e2ed41ab003ae96c28df8f9be3ff9c0ad80bed80161a4b",
    },
    "Output/audits/prospective_twin_reconstruction_decision.json": {
        "sha256": "49e80439be78593c690e7c4f57bd0532305c6eca305656c302c29372d4dbd3e6",
        "bytes": 3025,
        "required_fact_count": 12,
        "required_facts_sha256": "c316458aebecebbe084806139c4ad063b9139da2353d90395e014219cde0c0ca",
        "record_sha256": "4b24771dcb6a57ae27a4e46eac41dc97241d3916dcc47bc9ba60d219114057ce",
    },
    "Output/audits/prospective_development_data_intake.json": {
        "sha256": "9e8ed983302ca9fdf253cce98cf33b50dd95ae64f5ae84fa24c817b359d9f904",
        "bytes": 2894,
        "required_fact_count": 12,
        "required_facts_sha256": "6f41aa41917d3495a66b0f3b76e6f26efeb91543b882c6642cbd5845c08c2e38",
        "record_sha256": "b8addf572287f10e3d8b079fb6c9cf7841b7dc0f96b62117c5f0c88b7f7bd16b",
    },
    "Output/audits/prospective_development_foundation_decision.json": {
        "sha256": "914b701bcfae95baf4dad9dc7208f273e9ea35739d997850ae2eaea2a1618850",
        "bytes": 7785,
        "required_fact_count": 16,
        "required_facts_sha256": "fa8104de5b448f4bef7057daaaac0ff357d4cb5d1fb8f65e0e970704ccd06acf",
        "record_sha256": "f5255fe95e68ff9ee4cfe9f0f1c67e141489d6a341b4eec696678b36dd665f61",
    },
    "Output/audits/prospective_protocol_scientific_review.json": {
        "sha256": "ee647ede716e376e26d3e619431d45bb536178d5a0dc26dc8e794d1b1f46596a",
        "bytes": 5751,
        "required_fact_count": 12,
        "required_facts_sha256": "6d93de5fa538f9bc0ea39f5e6704819e0d360756c301fab18770710b599bd6a6",
        "record_sha256": "e6d83ae7c432b8203ce1613d48ad13e1fc9d3c30bc0b8e4807988902fc2bd1bb",
    },
    "Output/audits/six_context_semantic_exclusion_audit.json": {
        "sha256": "62128285feab3a1fdc19a0c068f01432c53a09ece7459abb0ce189f887fd7b1a",
        "bytes": 3485,
        "required_fact_count": 4,
        "required_facts_sha256": "8717a798448e86be18efbf0cbd824ea282c209ef6a2490869e0f07941668c08e",
        "record_sha256": "c824bb0b1586f59a655d03aeabf5477dee027acc33735fd98f3d04f42b48451c",
    },
}
EXPECTED_ARTIFACTS_SHA256 = (
    "24459f594542311e8cfa7452e7d10043908620ba1cd828d61948fa7782b5be11"
)
EXPECTED_REQUIRED_FACT_COUNT = 101

ALLOWED_CATEGORIES = {
    "method_checkpoint_negative",
    "development_domain_negative",
    "development_data_negative",
    "protocol_negative",
    "protocol_exclusion_negative",
}

TERMINAL_DECISIONS = {
    "formal_zero_shot_checkpoint": "NO_FORMAL_ZERO_SHOT_CHECKPOINT",
    "prospective_training_domain": "NO_INDEPENDENT_PROSPECTIVE_TRAINING_DOMAIN",
    "public_development_data": "PUBLIC_PROSPECTIVE_DEVELOPMENT_DATA_INSUFFICIENT",
}

AGGREGATE_ACCESS = {
    "development_outcome_access": 0,
    "evaluation_outcome_access": 0,
    "six_context_outcome_access": 0,
    "protected_holdout_access": 0,
    "protected_test_access": 0,
    "numeric_outcome_retrieval_observed": False,
    "six_context_files_opened": False,
    "protected_files_opened": False,
}

AUTHORIZATION_CLOSURE = {
    "automatic_second_intake_allowed": False,
    "new_public_data_intake_authorized": False,
    "training_authorized": False,
    "gpu_authorized": False,
    "world_action_compiler_authorized": False,
    "tri_authorized": False,
    "tool_calling_authorized": False,
    "memory_adaptation_authorized": False,
    "agent_in_twin_authorized": False,
    "coevolution_authorized": False,
    "formal_evaluation_authorized": False,
    "six_context_evaluation_authorized": False,
}

LIMITATIONS = [
    "Historical embedded transitive artifact bindings were not revalidated in this round.",
    "This manifest binds negative parent authorities and produces no scientific or performance conclusion.",
    "Live-root Git state is recorded only as a provenance observation and is not a clean or reconstructible lineage claim.",
]

CLOSED_AUTHORITY_KEYS = {
    "agent_in_twin_authorized",
    "automatic_second_intake_allowed",
    "coevolution_authorized",
    "evaluation_context_threshold_fit_allowed",
    "execution_authorized",
    "formal_evaluation_authorized",
    "formal_execution_allowed",
    "gpu_authorized",
    "method_preregistration_allowed",
    "new_data_intake_authorized",
    "new_public_data_intake_authorized",
    "protocol_executable",
    "public_evidence_search_authorized",
    "six_context_evaluation_authorized",
    "tool_calling_authorized",
    "training_authorized",
    "tri_authorized",
    "world_action_authorized",
    "world_action_compiler_authorized",
}

ACCESS_COUNTER_MAP = {
    "development_outcome_access": "development_outcome_access",
    "evaluation_outcome_access_during_review": "evaluation_outcome_access",
    "six_context_outcome_access": "six_context_outcome_access",
    "protected_holdout_access": "protected_holdout_access",
    "protected_test_access": "protected_test_access",
}

CONTRACT_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_uid",
    "artifact_count",
    "source_checkout",
    "manifest",
    "terminal_decisions",
    "aggregate_access",
    "authorization_closure",
    "limitations",
    "next_action",
    "artifacts",
}


class AuthorityBindingError(ValueError):
    """Raised when a frozen authority cannot be bound safely."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def canonical_value_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(raw)


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuthorityBindingError(f"{label} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise AuthorityBindingError(f"{label} keys must be strings")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AuthorityBindingError(
            f"Unable to load YAML contract {path}: {exc}"
        ) from exc
    return _require_mapping(payload, label="contract")


def _load_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuthorityBindingError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    return _require_mapping(payload, label=label)


def _validate_relative_json_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuthorityBindingError(
            f"{label} must be a non-empty repository-relative path"
        )
    if any(character in value for character in "*?[]\x00"):
        raise AuthorityBindingError(
            f"{label} contains a glob or NUL character: {value!r}"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or str(pure) != value
    ):
        raise AuthorityBindingError(
            f"{label} is not a normalized repository-relative path: {value!r}"
        )
    if pure.suffix != ".json":
        raise AuthorityBindingError(f"{label} must name one JSON file: {value!r}")
    return value


def _resolve_repository_path(
    root: Path, relative: str, *, must_exist: bool, label: str
) -> Path:
    normalized = _validate_relative_json_path(relative, label=label)
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*PurePosixPath(normalized).parts)
    current = resolved_root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise AuthorityBindingError(f"{label} traverses a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise AuthorityBindingError(f"Missing {label}: {relative}") from exc
    if not resolved.is_relative_to(resolved_root):
        raise AuthorityBindingError(f"{label} escapes declared root: {relative}")
    if must_exist and not resolved.is_file():
        raise AuthorityBindingError(f"{label} is not a regular file: {relative}")
    if not must_exist and candidate.exists() and not candidate.is_file():
        raise AuthorityBindingError(f"{label} is not a regular file: {relative}")
    return candidate


def _resolve_config_path(
    repository_root: Path, path: Path | str, *, label: str
) -> tuple[Path, str]:
    resolved_root = repository_root.resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    if candidate.is_symlink():
        raise AuthorityBindingError(f"{label} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise AuthorityBindingError(f"Missing {label}: {candidate}") from exc
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise AuthorityBindingError(
            f"{label} must be a regular file beneath repository root: {candidate}"
        )
    return resolved, resolved.relative_to(resolved_root).as_posix()


def _validate_fact_pointer(pointer: object) -> str:
    if not isinstance(pointer, str) or not pointer.startswith("/") or pointer == "/":
        raise AuthorityBindingError(
            f"required fact pointer must be a non-root JSON Pointer: {pointer!r}"
        )
    if "//" in pointer:
        raise AuthorityBindingError(
            f"required fact pointer contains an empty segment: {pointer!r}"
        )
    return pointer


def _pointer_segments(pointer: str) -> list[str]:
    return [
        segment.replace("~1", "/").replace("~0", "~")
        for segment in pointer[1:].split("/")
    ]


def _resolve_pointer(payload: object, pointer: str) -> object:
    current = payload
    for segment in _pointer_segments(pointer):
        if isinstance(current, list):
            if not segment.isdigit():
                raise AuthorityBindingError(
                    f"JSON Pointer array segment is not numeric: {pointer}"
                )
            index = int(segment)
            if index >= len(current):
                raise AuthorityBindingError(f"Missing required JSON Pointer: {pointer}")
            current = current[index]
        elif isinstance(current, dict):
            if segment not in current:
                raise AuthorityBindingError(f"Missing required JSON Pointer: {pointer}")
            current = current[segment]
        else:
            raise AuthorityBindingError(
                f"Required JSON Pointer crosses a scalar: {pointer}"
            )
    return current


def _json_values_equal(observed: object, expected: object) -> bool:
    return type(observed) is type(expected) and observed == expected


def _walk_leaves(
    value: object, path: tuple[str | int, ...] = ()
) -> Iterator[tuple[tuple[str | int, ...], object]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_leaves(item, (*path, index))
    else:
        yield path, value


def validate_contract(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract_map = _require_mapping(contract, label="contract")
    if set(contract_map) != CONTRACT_TOP_LEVEL_KEYS:
        missing = sorted(CONTRACT_TOP_LEVEL_KEYS - set(contract_map))
        extra = sorted(set(contract_map) - CONTRACT_TOP_LEVEL_KEYS)
        raise AuthorityBindingError(
            f"contract fields differ from the frozen shape; missing={missing}, extra={extra}"
        )
    if contract_map["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise AuthorityBindingError("unexpected contract schema_version")
    if contract_map["contract_uid"] != MANIFEST_UID:
        raise AuthorityBindingError("unexpected contract_uid")
    if (
        type(contract_map["artifact_count"]) is not int
        or contract_map["artifact_count"] != ARTIFACT_COUNT
    ):
        raise AuthorityBindingError("contract artifact_count must be exactly nine")

    source_checkout = _require_mapping(
        contract_map["source_checkout"], label="source_checkout"
    )
    expected_source = {
        "role": EXPECTED_SOURCE_ROLE,
        "expected_branch": EXPECTED_SOURCE_BRANCH,
        "expected_head": EXPECTED_SOURCE_HEAD,
        "expected_artifact_tracking_state": EXPECTED_SOURCE_TRACKING_STATE,
    }
    if source_checkout != expected_source:
        raise AuthorityBindingError(
            "source_checkout does not match the frozen live-root observation"
        )

    manifest = _require_mapping(contract_map["manifest"], label="manifest")
    expected_manifest = {
        "path": MANIFEST_PATH,
        "schema_path": SCHEMA_PATH,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": MANIFEST_STATUS,
    }
    if manifest != expected_manifest:
        raise AuthorityBindingError(
            "manifest contract does not match the frozen identity"
        )
    if contract_map["terminal_decisions"] != TERMINAL_DECISIONS:
        raise AuthorityBindingError(
            "terminal_decisions must preserve the exact three negative decisions"
        )
    if contract_map["aggregate_access"] != AGGREGATE_ACCESS:
        raise AuthorityBindingError("aggregate_access must remain closed and zero")
    if contract_map["authorization_closure"] != AUTHORIZATION_CLOSURE:
        raise AuthorityBindingError(
            "authorization_closure must keep every downstream authority closed"
        )
    if contract_map["limitations"] != LIMITATIONS:
        raise AuthorityBindingError(
            "limitations do not match the frozen interpretation boundary"
        )
    if contract_map["next_action"] != NEXT_ACTION:
        raise AuthorityBindingError(
            "contract next_action is not resolvable by the active roadmap"
        )

    raw_artifacts = contract_map["artifacts"]
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != ARTIFACT_COUNT:
        raise AuthorityBindingError(
            "contract artifacts must contain exactly nine records"
        )

    artifacts: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_source_paths: set[str] = set()
    seen_destination_paths: set[str] = set()
    for index, raw_record in enumerate(raw_artifacts):
        record = _require_mapping(raw_record, label=f"artifacts[{index}]")
        required_keys = {
            "authority_uid",
            "category",
            "source_path",
            "destination_path",
            "sha256",
            "bytes",
            "schema_version",
            "status",
            "required_facts",
        }
        if set(record) != required_keys:
            raise AuthorityBindingError(
                f"artifacts[{index}] fields must be exactly {sorted(required_keys)}"
            )
        authority_uid = record["authority_uid"]
        if (
            not isinstance(authority_uid, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", authority_uid) is None
        ):
            raise AuthorityBindingError(f"invalid authority_uid at artifacts[{index}]")
        if authority_uid in seen_uids:
            raise AuthorityBindingError(f"duplicate authority_uid: {authority_uid}")
        seen_uids.add(authority_uid)

        category = record["category"]
        if category not in ALLOWED_CATEGORIES:
            raise AuthorityBindingError(f"unknown authority category: {category!r}")
        source_path = _validate_relative_json_path(
            record["source_path"], label="source_path"
        )
        destination_path = _validate_relative_json_path(
            record["destination_path"], label="destination_path"
        )
        if source_path != destination_path:
            raise AuthorityBindingError(
                f"source and destination paths must be identical: {source_path}"
            )
        if (
            source_path in seen_source_paths
            or destination_path in seen_destination_paths
        ):
            raise AuthorityBindingError(
                f"duplicate source or destination path: {source_path}"
            )
        seen_source_paths.add(source_path)
        seen_destination_paths.add(destination_path)

        if source_path not in EXPECTED_ARTIFACT_IDENTITIES:
            raise AuthorityBindingError(f"non-allowlisted source path: {source_path}")
        expected_category, expected_schema, expected_status = (
            EXPECTED_ARTIFACT_IDENTITIES[source_path]
        )
        if (category, record["schema_version"], record["status"]) != (
            expected_category,
            expected_schema,
            expected_status,
        ):
            raise AuthorityBindingError(f"semantic identity mismatch for {source_path}")

        digest = record["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise AuthorityBindingError(f"invalid SHA-256 for {source_path}")
        byte_count = record["bytes"]
        if type(byte_count) is not int or byte_count <= 0:
            raise AuthorityBindingError(f"invalid byte count for {source_path}")

        required_facts = record["required_facts"]
        if not isinstance(required_facts, list) or not required_facts:
            raise AuthorityBindingError(
                f"required_facts must be a non-empty list for {source_path}"
            )
        seen_pointers: set[str] = set()
        for fact_index, raw_fact in enumerate(required_facts):
            fact = _require_mapping(
                raw_fact, label=f"{source_path}.required_facts[{fact_index}]"
            )
            if set(fact) != {"pointer", "expected"}:
                raise AuthorityBindingError(
                    f"required fact fields must be pointer and expected for {source_path}"
                )
            pointer = _validate_fact_pointer(fact["pointer"])
            if pointer in seen_pointers:
                raise AuthorityBindingError(
                    f"duplicate required fact pointer for {source_path}: {pointer}"
                )
            seen_pointers.add(pointer)

        frozen = EXPECTED_FROZEN_ARTIFACT_CONTRACT[source_path]
        if record["sha256"] != frozen["sha256"]:
            raise AuthorityBindingError(
                f"frozen source SHA-256 changed for {source_path}"
            )
        if record["bytes"] != frozen["bytes"]:
            raise AuthorityBindingError(
                f"frozen source byte count changed for {source_path}"
            )
        if len(required_facts) != frozen["required_fact_count"]:
            raise AuthorityBindingError(
                f"frozen required fact count changed for {source_path}"
            )
        if canonical_value_sha256(required_facts) != frozen["required_facts_sha256"]:
            raise AuthorityBindingError(
                f"frozen required fact set changed for {source_path}"
            )
        if canonical_value_sha256(record) != frozen["record_sha256"]:
            raise AuthorityBindingError(
                f"frozen contract record changed for {source_path}"
            )
        artifacts.append(record)

    if seen_source_paths != set(EXPECTED_ARTIFACT_IDENTITIES):
        raise AuthorityBindingError(
            "contract source paths are not the exact frozen nine-entry allowlist"
        )
    if [record["source_path"] for record in artifacts] != list(
        EXPECTED_ARTIFACT_IDENTITIES
    ):
        raise AuthorityBindingError(
            "contract artifacts must retain the frozen allowlist order"
        )
    required_fact_count = sum(len(record["required_facts"]) for record in artifacts)
    if required_fact_count != EXPECTED_REQUIRED_FACT_COUNT:
        raise AuthorityBindingError(
            f"frozen required fact total changed: expected {EXPECTED_REQUIRED_FACT_COUNT}, observed {required_fact_count}"
        )
    if canonical_value_sha256(artifacts) != EXPECTED_ARTIFACTS_SHA256:
        raise AuthorityBindingError("frozen aggregate artifact contract changed")
    return artifacts


def _validate_closed_facts(payload: Mapping[str, Any], *, label: str) -> None:
    for path, value in _walk_leaves(payload):
        if not path:
            continue
        key = str(path[-1])
        dotted = ".".join(str(part) for part in path)
        if (
            key in CLOSED_AUTHORITY_KEYS or key.endswith("_authorized")
        ) and value is not False:
            raise AuthorityBindingError(f"{label} reopens closed authority at {dotted}")
        if key in ACCESS_COUNTER_MAP and (type(value) is not int or value != 0):
            raise AuthorityBindingError(
                f"{label} has nonzero or invalid access counter at {dotted}"
            )
        if (
            key == "development_numeric_outcomes_retrieved_before_queue_freeze"
            and value is not False
        ):
            raise AuthorityBindingError(
                f"{label} records development outcome retrieval before queue freeze"
            )


def validate_artifact_payload(
    payload: Mapping[str, Any], record: Mapping[str, Any]
) -> list[dict[str, Any]]:
    label = str(record["source_path"])
    if payload.get("schema_version") != record["schema_version"]:
        raise AuthorityBindingError(f"schema_version mismatch for {label}")
    if payload.get("status") != record["status"]:
        raise AuthorityBindingError(f"status mismatch for {label}")
    _validate_closed_facts(payload, label=label)

    results: list[dict[str, Any]] = []
    for fact in record["required_facts"]:
        pointer = fact["pointer"]
        observed = _resolve_pointer(payload, pointer)
        expected = fact["expected"]
        if not _json_values_equal(observed, expected):
            raise AuthorityBindingError(
                f"required fact mismatch for {label} at {pointer}: expected {expected!r}, observed {observed!r}"
            )
        results.append(
            {
                "pointer": pointer,
                "expected": expected,
                "observed": observed,
                "passed": True,
            }
        )
    return results


def _read_source_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _run_git(
    repository: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuthorityBindingError(
            f"Git provenance query failed in {repository}: git {' '.join(args)}"
        ) from exc


def _source_tracking_state(source_root: Path, relative: str) -> str:
    tracked = (
        _run_git(
            source_root, "ls-files", "--error-unmatch", "--", relative, check=False
        ).returncode
        == 0
    )
    status = _run_git(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        relative,
    ).stdout.rstrip("\n")
    if not tracked:
        if status == f"?? {relative}":
            return "untracked"
        raise AuthorityBindingError(
            f"untracked source path has unexpected Git status: {relative}: {status!r}"
        )
    if not status:
        return "tracked_clean"
    code = status[:2]
    if "U" in code or code in {"AA", "DD"}:
        return "tracked_conflicted"
    if code[0] != " ":
        return "tracked_staged"
    return "tracked_modified"


def collect_source_checkout(
    source_root: Path, artifact_paths: Sequence[str]
) -> dict[str, Any]:
    branch = _run_git(source_root, "branch", "--show-current").stdout.strip()
    head = _run_git(source_root, "rev-parse", "HEAD").stdout.strip()
    tracking_states = {
        path: _source_tracking_state(source_root, path) for path in artifact_paths
    }
    return {"branch": branch, "head": head, "tracking_states": tracking_states}


def collect_destination_checkout(repository_root: Path) -> dict[str, Any]:
    return {
        "branch": _run_git(repository_root, "branch", "--show-current").stdout.strip(),
        "head_at_build": _run_git(repository_root, "rev-parse", "HEAD").stdout.strip(),
    }


def _validate_source_checkout(
    provenance: Mapping[str, Any],
    contract: Mapping[str, Any],
    artifact_paths: Sequence[str],
) -> dict[str, Any]:
    branch = provenance.get("branch")
    head = provenance.get("head")
    states = provenance.get("tracking_states")
    if not isinstance(states, dict) or set(states) != set(artifact_paths):
        raise AuthorityBindingError(
            "source Git provenance lacks the exact allowlisted artifact paths"
        )
    expected = contract["source_checkout"]
    branch_matches = branch == expected["expected_branch"]
    head_matches = head == expected["expected_head"]
    tracking_matches = all(
        states[path] == expected["expected_artifact_tracking_state"]
        for path in artifact_paths
    )
    if not branch_matches or not head_matches or not tracking_matches:
        raise AuthorityBindingError(
            "live-root provenance drift: "
            f"branch_matches={branch_matches}, head_matches={head_matches}, tracking_matches={tracking_matches}"
        )
    return {
        "role": expected["role"],
        "branch": branch,
        "head": head,
        "branch_matches_contract": True,
        "head_matches_contract": True,
        "allowlisted_tracking_state_matches_contract": True,
        "tracked_artifact_count": sum(
            state != "untracked" for state in states.values()
        ),
        "untracked_artifact_count": sum(
            state == "untracked" for state in states.values()
        ),
        "provenance_observation_only": True,
        "clean_reconstructible_lineage_claimed": False,
    }


def _aggregate_access(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = dict(AGGREGATE_ACCESS)
    for payload in payloads:
        for path, value in _walk_leaves(payload):
            if not path:
                continue
            key = str(path[-1])
            target = ACCESS_COUNTER_MAP.get(key)
            if target is not None:
                if type(value) is not int or value != 0:
                    raise AuthorityBindingError(
                        f"nonzero access evidence at {'.'.join(str(part) for part in path)}"
                    )
                result[target] += value
            if (
                key == "development_numeric_outcomes_retrieved_before_queue_freeze"
                and value is True
            ):
                result["numeric_outcome_retrieval_observed"] = True
    return result


def _load_and_check_schema(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    schema = _load_json_bytes(raw, label="manifest schema")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise AuthorityBindingError(
            f"invalid Draft 2020-12 manifest schema: {exc}"
        ) from exc
    return schema, sha256_bytes(raw)


def validate_manifest_payload(
    payload: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise AuthorityBindingError(
            f"aggregate manifest schema validation failed: {details}"
        )


def _install_transaction(
    *,
    repository_root: Path,
    staged_root: Path,
    missing_destinations: Sequence[str],
    source_bytes_by_path: Mapping[str, bytes],
    manifest_bytes: bytes,
) -> None:
    installed: list[Path] = []
    manifest_destination = repository_root / MANIFEST_PATH
    if manifest_destination.is_symlink() or (
        manifest_destination.exists() and not manifest_destination.is_file()
    ):
        raise AuthorityBindingError(
            "aggregate manifest destination is not a regular non-symlink path"
        )
    manifest_preexisting = manifest_destination.exists()
    if manifest_preexisting and manifest_destination.read_bytes() != manifest_bytes:
        raise AuthorityBindingError("refusing to overwrite drifted aggregate manifest")
    staged_manifest = staged_root / MANIFEST_PATH
    staged_manifest.parent.mkdir(parents=True, exist_ok=True)
    staged_manifest.write_bytes(manifest_bytes)
    try:
        for relative in missing_destinations:
            staged = staged_root / relative
            destination = repository_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, destination)
            installed.append(destination)
        for relative, expected_bytes in source_bytes_by_path.items():
            destination = repository_root / relative
            if destination.read_bytes() != expected_bytes:
                raise AuthorityBindingError(
                    f"post-install byte mismatch for {relative}"
                )
        if not manifest_preexisting:
            manifest_destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_manifest, manifest_destination)
    except Exception as exc:
        for path in reversed(installed):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if isinstance(exc, AuthorityBindingError):
            raise
        raise AuthorityBindingError(
            f"transactional installation failed: {exc}"
        ) from exc


def build_parent_authority(
    *,
    source_root: Path,
    repository_root: Path,
    contract_path: Path | str,
    source_checkout_provenance: Mapping[str, Any] | None = None,
    destination_checkout_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    repository_root = repository_root.resolve(strict=True)
    if source_root == repository_root:
        raise AuthorityBindingError(
            "source-root and repository-root must be distinct checkouts"
        )

    resolved_contract_path, contract_relative = _resolve_config_path(
        repository_root,
        contract_path,
        label="authority contract",
    )
    if contract_relative != CONTRACT_PATH:
        raise AuthorityBindingError(f"authority contract must be {CONTRACT_PATH}")
    contract_raw = resolved_contract_path.read_bytes()
    contract = _load_yaml(resolved_contract_path)
    records = validate_contract(contract)

    schema_relative = contract["manifest"]["schema_path"]
    schema_path = _resolve_repository_path(
        repository_root,
        schema_relative,
        must_exist=True,
        label="manifest schema",
    )
    schema, schema_sha256 = _load_and_check_schema(schema_path)

    artifact_paths = [record["source_path"] for record in records]
    source_raw_provenance = source_checkout_provenance or collect_source_checkout(
        source_root, artifact_paths
    )
    source_checkout = _validate_source_checkout(
        source_raw_provenance, contract, artifact_paths
    )
    destination_raw_provenance = (
        destination_checkout_provenance or collect_destination_checkout(repository_root)
    )
    destination_branch = destination_raw_provenance.get("branch")
    destination_head = destination_raw_provenance.get("head_at_build")
    if not isinstance(destination_branch, str) or not destination_branch:
        raise AuthorityBindingError("destination checkout branch is missing")
    if (
        not isinstance(destination_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", destination_head) is None
    ):
        raise AuthorityBindingError("destination checkout HEAD is not a Git object id")

    tracking_states = source_raw_provenance["tracking_states"]
    source_payloads: list[dict[str, Any]] = []
    source_bytes_by_path: dict[str, bytes] = {}
    artifact_records: list[dict[str, Any]] = []
    missing_destinations: list[str] = []

    for record in records:
        relative = record["source_path"]
        source_path = _resolve_repository_path(
            source_root, relative, must_exist=True, label="source artifact"
        )
        raw = _read_source_bytes(source_path)
        observed_sha256 = sha256_bytes(raw)
        if observed_sha256 != record["sha256"]:
            raise AuthorityBindingError(f"source SHA-256 mismatch for {relative}")
        if len(raw) != record["bytes"]:
            raise AuthorityBindingError(f"source byte-count mismatch for {relative}")
        payload = _load_json_bytes(raw, label=relative)
        fact_results = validate_artifact_payload(payload, record)
        source_payloads.append(payload)
        source_bytes_by_path[relative] = raw

        destination_path = _resolve_repository_path(
            repository_root,
            record["destination_path"],
            must_exist=False,
            label="destination artifact",
        )
        if destination_path.exists():
            destination_raw = destination_path.read_bytes()
            if destination_raw != raw:
                raise AuthorityBindingError(
                    f"refusing to overwrite drifted destination: {relative}"
                )
        else:
            missing_destinations.append(record["destination_path"])

        artifact_records.append(
            {
                "authority_uid": record["authority_uid"],
                "category": record["category"],
                "source_path": relative,
                "destination_path": record["destination_path"],
                "schema_version": record["schema_version"],
                "status": record["status"],
                "expected_sha256": record["sha256"],
                "source_sha256": observed_sha256,
                "destination_sha256": observed_sha256,
                "expected_bytes": record["bytes"],
                "source_bytes": len(raw),
                "destination_bytes": len(raw),
                "source_tracking_state": tracking_states[relative],
                "source_matches_contract": True,
                "destination_matches_source": True,
                "required_fact_results": fact_results,
            }
        )

    aggregate_access = _aggregate_access(source_payloads)
    if aggregate_access != contract["aggregate_access"]:
        raise AuthorityBindingError(
            "aggregate source access evidence does not match the closed contract"
        )

    manifest: dict[str, Any] = {
        "schema_version": contract["manifest"]["schema_version"],
        "manifest_uid": contract["contract_uid"],
        "status": contract["manifest"]["status"],
        "contract": {
            "path": contract_relative,
            "sha256": sha256_bytes(contract_raw),
            "schema_path": schema_relative,
            "schema_sha256": schema_sha256,
            "artifact_count": len(records),
        },
        "source_checkout": source_checkout,
        "destination_checkout": {
            "branch": destination_branch,
            "head_at_build": destination_head,
            "candidate_commit": None,
            "candidate_commit_status": "uncommitted_round_candidate",
            "source_and_destination_checkout_distinct": True,
        },
        "artifact_count": len(artifact_records),
        "artifacts": artifact_records,
        "terminal_decisions": dict(contract["terminal_decisions"]),
        "aggregate_access": aggregate_access,
        "authorization_closure": dict(contract["authorization_closure"]),
        "artifact_read_scope": {
            "allowed_source_artifact_count": len(artifact_paths),
            "source_artifact_paths_opened": artifact_paths,
            "non_allowlisted_source_artifact_paths_opened": [],
            "six_context_files_opened": False,
            "protected_files_opened": False,
        },
        "transitive_bindings_revalidated": False,
        "source_artifacts_regenerated": False,
        "scientific_conclusion_produced": False,
        "performance_conclusion_produced": False,
        "limitations": list(contract["limitations"]),
        "blockers": [],
        "next_action": contract["next_action"],
    }
    validate_manifest_payload(manifest, schema)
    manifest_bytes = canonical_json_bytes(manifest)

    audit_root = repository_root / "Output" / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".no_task_fit_parent_authority_", dir=audit_root
    ) as temporary:
        staged_root = Path(temporary)
        for relative, raw in source_bytes_by_path.items():
            staged_path = staged_root / relative
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_bytes(raw)
            if staged_path.read_bytes() != raw or sha256_file(
                staged_path
            ) != sha256_bytes(raw):
                raise AuthorityBindingError(
                    f"staged byte verification failed for {relative}"
                )
        _install_transaction(
            repository_root=repository_root,
            staged_root=staged_root,
            missing_destinations=missing_destinations,
            source_bytes_by_path=source_bytes_by_path,
            manifest_bytes=manifest_bytes,
        )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_parent_authority(
            source_root=args.source_root,
            repository_root=args.repository_root,
            contract_path=args.contract,
        )
    except AuthorityBindingError as exc:
        raise SystemExit(f"no-task-fit parent-authority build failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "artifact_count": manifest["artifact_count"],
                "blocker_count": len(manifest["blockers"]),
                "next_action": manifest["next_action"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
