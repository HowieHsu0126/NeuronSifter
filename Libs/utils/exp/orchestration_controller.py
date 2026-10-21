"""Canonical controller primitives for formal NeuronSifter Gate decisions.

This module deliberately has no data, model, training, metric, or scientific-tool
implementation dependencies. It only validates authority and writes a formal Gate
record after an approved validator has produced a passing result.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml


CANONICAL_CONTROLLER_UID = "neuronsifter_orchestration_controller"
GATE_WRITER_AUTHORITY_UID = "GateWriterAuthority"
GATE_SCHEMA_VERSION = "neuronsifter-gate-writer-authority-v1"
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^[a-f0-9]{40,64}$")
ALLOWED_DECISIONS = {"accepted", "blocked", "rejected", "not_evaluated"}
ALLOWED_ASSIMILATION_DECISIONS = {
    "approved",
    "approved_uncertainty_only",
    "approved_conflict_register_only",
    "revise_required",
    "insufficient_evidence",
    "unsupported_domain",
    "rejected_qc",
    "rejected_provenance",
    "rejected_conflict",
}
GIT_STATUS_TIMEOUT_SECONDS = 120


class GateAuthorityError(ValueError):
    """Raised when a caller attempts an unauthorized formal Gate operation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not GIT_COMMIT_PATTERN.fullmatch(value):
        raise GateAuthorityError("A formal Gate decision requires a resolvable Git commit.")
    return value


def git_dirty_status(
    repo_root: Path, *, timeout_seconds: float = GIT_STATUS_TIMEOUT_SECONDS
) -> str:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        # A slow status read must never be interpreted as a clean identity.
        return "dirty"
    if completed.returncode != 0:
        raise GateAuthorityError("A formal Gate decision requires a readable Git dirty status.")
    return "dirty" if completed.stdout.strip() else "clean"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_mapping(item) for item in value]


def load_authority_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise GateAuthorityError(f"GateWriterAuthority config must be a mapping: {path}")
    return dict(payload)


def authority_contract_report(authority: Mapping[str, object]) -> dict[str, Any]:
    """Validate the small, non-active authority config before it can be used."""

    blockers: list[dict[str, str]] = []
    controller = _mapping(authority.get("canonical_controller"))
    formal_gate = _mapping(authority.get("formal_gate"))
    artifact_policy = _mapping(authority.get("artifact_policy"))
    assimilation_authority = _mapping(authority.get("assimilation_authority"))

    if authority.get("schema_version") != "neuronsifter-round11f-controller-authority-v1":
        blockers.append({"code": "authority_schema_version", "message": "Unexpected authority config schema version."})
    if authority.get("authority_uid") != GATE_WRITER_AUTHORITY_UID:
        blockers.append({"code": "authority_uid", "message": "Authority UID must be GateWriterAuthority."})
    if controller.get("component_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append({"code": "canonical_controller_uid", "message": "The configured canonical controller UID is invalid."})
    if formal_gate.get("writer_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append({"code": "canonical_gate_writer_uid", "message": "The formal Gate writer must be the canonical controller."})
    if formal_gate.get("schema_version") != GATE_SCHEMA_VERSION:
        blockers.append({"code": "formal_gate_schema_version", "message": "The formal Gate schema version is invalid."})

    required_responsibilities = {
        "config_resolution",
        "stage_gate_orchestration",
        "artifact_dependency_checks",
        "validator_invocation",
        "terminal_state_aggregation",
        "canonical_gate_decision_writing",
        "assimilation_decision_authorization",
    }
    configured_responsibilities = {str(item) for item in controller.get("responsibilities", [])}
    missing_responsibilities = sorted(required_responsibilities - configured_responsibilities)
    if missing_responsibilities:
        blockers.append(
            {
                "code": "missing_controller_responsibility",
                "message": "Canonical controller responsibilities are incomplete: " + ", ".join(missing_responsibilities),
            }
        )

    validators = _list_of_mappings(formal_gate.get("approved_validators"))
    validator_uids = [str(row.get("validator_uid", "")) for row in validators]
    if not validators or any(not value for value in validator_uids):
        blockers.append({"code": "approved_validators_missing", "message": "At least one named approved validator is required."})
    if len(validator_uids) != len(set(validator_uids)):
        blockers.append({"code": "duplicate_validator_uid", "message": "Approved validator UIDs must be unique."})
    for row in validators:
        stage_uids = row.get("stage_uids")
        if not isinstance(stage_uids, list) or not all(isinstance(stage, str) and stage for stage in stage_uids):
            blockers.append(
                {
                    "code": "invalid_validator_stage_binding",
                    "message": f"Validator {row.get('validator_uid', '<missing>')} has no valid stage binding.",
                }
            )

    for key in ("require_input_artifact_hashes", "require_config_hash", "require_git_commit", "require_lineage_parent"):
        if artifact_policy.get(key) is not True:
            blockers.append({"code": "artifact_policy_required", "message": f"Artifact policy {key} must be true."})
    if artifact_policy.get("latest_pointer_is_evidence") is not False:
        blockers.append({"code": "latest_pointer_policy", "message": "A latest pointer cannot be formal evidence."})
    if artifact_policy.get("markdown_is_status_source") is not False:
        blockers.append({"code": "markdown_status_policy", "message": "Markdown cannot be a formal status source."})

    if assimilation_authority.get("schema_version") != "neuronsifter-round13-assimilation-authority-v1":
        blockers.append({"code": "assimilation_authority_schema", "message": "Assimilation authority schema is invalid."})
    if assimilation_authority.get("controller_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append({"code": "assimilation_controller_uid", "message": "Only the canonical controller may authorize assimilation."})
    assimilation_validators = assimilation_authority.get("approved_validators")
    if not isinstance(assimilation_validators, list) or "round13_action_observation_contract_validator" not in assimilation_validators:
        blockers.append({"code": "assimilation_validator_missing", "message": "Round-13 assimilation validator is not approved."})

    return {
        "schema_version": "neuronsifter-gate-writer-authority-validation-v1",
        "authority_uid": authority.get("authority_uid"),
        "canonical_controller_uid": controller.get("component_uid"),
        "canonical_gate_writer_uid": formal_gate.get("writer_uid"),
        "canonical_controller_count": 1 if controller.get("component_uid") == CANONICAL_CONTROLLER_UID else 0,
        "canonical_gate_writer_count": 1 if formal_gate.get("writer_uid") == CANONICAL_CONTROLLER_UID else 0,
        "approved_validators": validators,
        "assimilation_controller_uid": assimilation_authority.get("controller_uid"),
        "approved_assimilation_validators": assimilation_validators if isinstance(assimilation_validators, list) else [],
        "blockers": blockers,
        "status": "passed" if not blockers else "blocked",
    }


def validator_is_approved(authority: Mapping[str, object], validator_uid: str, stage_uid: str) -> bool:
    formal_gate = _mapping(authority.get("formal_gate"))
    for row in _list_of_mappings(formal_gate.get("approved_validators")):
        if row.get("validator_uid") == validator_uid and stage_uid in row.get("stage_uids", []):
            return True
    return False


def assert_formal_gate_writer(
    authority: Mapping[str, object],
    *,
    controller_uid: str,
    writer_kind: str,
) -> None:
    report = authority_contract_report(authority)
    if report["status"] != "passed":
        raise GateAuthorityError("GateWriterAuthority config is invalid.")
    disallowed = set(_mapping(authority.get("formal_gate")).get("disallowed_writer_kinds", []))
    if writer_kind in disallowed:
        raise GateAuthorityError(f"{writer_kind} is prohibited from writing a formal Gate.")
    if controller_uid != CANONICAL_CONTROLLER_UID:
        raise GateAuthorityError("Only neuronsifter_orchestration_controller can write a formal Gate.")


def assert_assimilation_writer(
    authority: Mapping[str, object],
    *,
    controller_uid: str,
    validator_uid: str,
    writer_kind: str,
) -> None:
    """Reject every noncanonical assimilation decision writer."""

    report = authority_contract_report(authority)
    if report["status"] != "passed":
        raise GateAuthorityError("GateWriterAuthority config is invalid.")
    assimilation_authority = _mapping(authority.get("assimilation_authority"))
    disallowed = set(assimilation_authority.get("disallowed_writer_kinds", []))
    if writer_kind in disallowed:
        raise GateAuthorityError(f"{writer_kind} is prohibited from writing an assimilation decision.")
    if controller_uid != CANONICAL_CONTROLLER_UID:
        raise GateAuthorityError("Only neuronsifter_orchestration_controller can write an assimilation decision.")
    if validator_uid not in assimilation_authority.get("approved_validators", []):
        raise GateAuthorityError("Assimilation decision validator is not approved.")


def build_canonical_assimilation_decision(
    authority: Mapping[str, object],
    *,
    proposal_uid: str,
    controller_uid: str,
    validator_uid: str,
    validator_report: Mapping[str, object],
    prior_state_hash: str,
    input_observation_hashes: Mapping[str, str],
    decision: str,
    reason_codes: list[str],
    approved_update_scope: list[str],
    conflict_handling: Mapping[str, str],
    resulting_state_ref: str | None,
    provenance_ref: str,
    writer_kind: str = "canonical_controller",
) -> dict[str, Any]:
    """Construct a typed AssimilationDecision through canonical authority only."""

    assert_assimilation_writer(
        authority,
        controller_uid=controller_uid,
        validator_uid=validator_uid,
        writer_kind=writer_kind,
    )
    if validator_report.get("validator_uid") != validator_uid or validator_report.get("status") != "passed":
        raise GateAuthorityError("Assimilation decision requires a passing approved validator report.")
    if decision not in ALLOWED_ASSIMILATION_DECISIONS:
        raise GateAuthorityError(f"Unsupported assimilation decision: {decision}")
    if not proposal_uid or not provenance_ref:
        raise GateAuthorityError("Assimilation decision requires proposal and provenance identities.")
    if not SHA256_PATTERN.fullmatch(prior_state_hash):
        raise GateAuthorityError("Assimilation prior_state_hash must be SHA-256.")
    if not input_observation_hashes or not all(SHA256_PATTERN.fullmatch(value) for value in input_observation_hashes.values()):
        raise GateAuthorityError("Assimilation observation hashes must be SHA-256.")
    if decision.startswith("approved") and not approved_update_scope:
        raise GateAuthorityError("Approved assimilation decisions require an explicit update scope.")
    decision_uid = "assimilation_decision_" + sha256_payload(
        {
            "proposal_uid": proposal_uid,
            "decision": decision,
            "prior_state_hash": prior_state_hash,
            "input_observation_hashes": dict(sorted(input_observation_hashes.items())),
        }
    )[:24]
    return {
        "schema_version": "neuronsifter-assimilation-decision-v1",
        "decision_uid": decision_uid,
        "proposal_uid": proposal_uid,
        "controller_uid": controller_uid,
        "validator_uid": validator_uid,
        "prior_state_hash": prior_state_hash,
        "input_observation_hashes": dict(sorted(input_observation_hashes.items())),
        "decision": decision,
        "reason_codes": sorted(set(reason_codes)),
        "approved_update_scope": sorted(set(approved_update_scope)),
        "conflict_handling": dict(sorted(conflict_handling.items())),
        "resulting_state_ref": resulting_state_ref,
        "provenance_ref": provenance_ref,
    }


def build_canonical_gate_decision(
    authority: Mapping[str, object],
    *,
    gate_uid: str,
    stage_uid: str,
    controller_uid: str,
    validator_uid: str,
    validator_report: Mapping[str, object],
    input_artifact_hashes: Mapping[str, str],
    config_hash: str,
    commit: str,
    decision: str,
    blockers: list[dict[str, object]],
    lineage_parent: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a formal Gate only after authority and validator checks pass."""

    assert_formal_gate_writer(authority, controller_uid=controller_uid, writer_kind="canonical_controller")
    if not gate_uid or not stage_uid or not lineage_parent:
        raise GateAuthorityError("Formal Gate records require gate_uid, stage_uid, and lineage_parent.")
    if not validator_is_approved(authority, validator_uid, stage_uid):
        raise GateAuthorityError("The requested validator is not approved for this stage.")
    if validator_report.get("validator_uid") != validator_uid:
        raise GateAuthorityError("Gate validator UID does not match the invoked validator report.")
    if validator_report.get("status") != "passed":
        raise GateAuthorityError("A formal Gate cannot be written from a non-passing validator report.")
    if decision not in ALLOWED_DECISIONS:
        raise GateAuthorityError(f"Unsupported formal Gate decision: {decision}")
    if decision == "accepted" and blockers:
        raise GateAuthorityError("An accepted formal Gate cannot include blockers.")
    if not input_artifact_hashes or not all(SHA256_PATTERN.fullmatch(value) for value in input_artifact_hashes.values()):
        raise GateAuthorityError("Formal Gate input artifacts must have SHA-256 hashes.")
    if not SHA256_PATTERN.fullmatch(config_hash):
        raise GateAuthorityError("Formal Gate config_hash must be SHA-256.")
    if not GIT_COMMIT_PATTERN.fullmatch(commit):
        raise GateAuthorityError("Formal Gate git_commit must be a Git object ID.")

    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_uid": gate_uid,
        "stage_uid": stage_uid,
        "controller_uid": controller_uid,
        "validator_uid": validator_uid,
        "input_artifact_hashes": dict(sorted(input_artifact_hashes.items())),
        "config_hash": config_hash,
        "git_commit": commit,
        "decision": decision,
        "blockers": blockers,
        "created_at": created_at or utc_now(),
        "lineage_parent": lineage_parent,
    }
