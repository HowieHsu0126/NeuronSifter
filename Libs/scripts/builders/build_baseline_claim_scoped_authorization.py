"""Build fail-closed, claim-scoped baseline authorization audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json  # noqa: E402


DEFAULT_CONFIG = "Libs/configs/experiments/baseline_claim_scoped_authorization.yaml"
DEFAULT_AUDIT_ROOT = "Output/audits"
CLAIM_IDS = ("C1", "C2", "C3", "C4", "C5")
OUTPUT_FILENAMES = {
    "dependency_graph": "baseline_claim_dependency_graph.json",
    "authorization": "baseline_claim_scoped_authorization.json",
    "semantic_validation": "baseline_registry_semantic_validation.json",
}
REQUIRED_REQUIREMENTS = {
    "C1": {"corpus", "splits", "baselines"},
    "C2": {"frozen_c1_state_interface", "action_complete_c2_domain", "baselines"},
    "C3": {"frozen_c2_predictions_and_posterior", "tool_environment", "baselines"},
    "C4": {"frozen_validation_predictions", "baseline", "calibration"},
    "C5": {"checkpoints", "deployment_profiles"},
}
REQUIRED_CLAIM_DEPENDENCIES = {"C1": set(), "C2": {"C1"}, "C3": {"C2"}, "C4": set(), "C5": set()}


class ContractError(ValueError):
    """Raised when the authorization contract is structurally unsafe."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed baseline claim-scoped authorization audits.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--audit-root", default=DEFAULT_AUDIT_ROOT)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _nonempty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a mapping")
    return value


def _repo_path(repo_root: Path, path_value: object) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ContractError("Evidence path must be a non-empty repository-relative string")
    path = Path(path_value)
    if path.is_absolute():
        raise ContractError(f"Evidence path must be repository-relative: {path_value}")
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ContractError(f"Evidence path escapes repository root: {path_value}") from exc
    return resolved


def _config_path(config_path: str | Path, repo_root: Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_contract(config_path: str | Path, repo_root: Path) -> Mapping[str, Any]:
    path = _config_path(config_path, repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"Claim-scoped authorization config is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, "Claim-scoped authorization config")


def _output_paths(audit_root: Path) -> set[Path]:
    return {(audit_root / filename).resolve() for filename in OUTPUT_FILENAMES.values()}


def _topological_order(claims: Mapping[str, Mapping[str, Any]]) -> list[str]:
    active: list[str] = []
    visited: set[str] = set()
    order: list[str] = []

    def visit(claim_id: str) -> None:
        if claim_id in active:
            cycle = active[active.index(claim_id):] + [claim_id]
            raise ContractError(f"Claim dependency cycle detected: {' -> '.join(cycle)}")
        if claim_id in visited:
            return
        active.append(claim_id)
        for dependency in claims[claim_id]["depends_on_claims"]:
            visit(str(dependency))
        active.pop()
        visited.add(claim_id)
        order.append(claim_id)

    for claim_id in CLAIM_IDS:
        visit(claim_id)
    return order


def _validate_contract(contract: Mapping[str, Any], repo_root: Path, audit_root: Path) -> list[str]:
    if contract.get("schema_version") != "neuronsifter-baseline-claim-scoped-authorization-config-v1":
        raise ContractError("Unsupported claim-scoped authorization config schema_version")
    authorization = _mapping(contract.get("authorization"), "authorization")
    if authorization.get("global_fail_closed") is not True:
        raise ContractError("global_fail_closed must be true")
    if authorization.get("global_activation_requires_all_claims") is not True:
        raise ContractError("global_activation_requires_all_claims must be true")
    producer = authorization.get("artifact_producer")
    validator = authorization.get("validator_producer")
    if not isinstance(producer, str) or not producer or not isinstance(validator, str) or not validator:
        raise ContractError("authorization must declare artifact_producer and validator_producer")

    claims = _mapping(contract.get("claims"), "claims")
    if set(claims) != set(CLAIM_IDS):
        raise ContractError(f"claims must be exactly {list(CLAIM_IDS)}")
    output_paths = _output_paths(audit_root)
    for claim_id in CLAIM_IDS:
        claim = _mapping(claims[claim_id], f"claims.{claim_id}")
        dependencies = claim.get("depends_on_claims")
        if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
            raise ContractError(f"claims.{claim_id}.depends_on_claims must be a list of claim IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ContractError(f"claims.{claim_id}.depends_on_claims contains duplicates")
        unknown = set(dependencies) - set(CLAIM_IDS)
        if unknown:
            raise ContractError(f"claims.{claim_id}.depends_on_claims contains unknown claim IDs: {sorted(unknown)}")
        required_dependencies = REQUIRED_CLAIM_DEPENDENCIES[claim_id]
        if not required_dependencies <= set(dependencies):
            raise ContractError(f"claims.{claim_id} is missing required upstream dependencies: {sorted(required_dependencies - set(dependencies))}")

        requirements = claim.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            raise ContractError(f"claims.{claim_id}.requirements must be a non-empty list")
        requirement_ids: list[str] = []
        for index, requirement_value in enumerate(requirements):
            requirement = _mapping(requirement_value, f"claims.{claim_id}.requirements[{index}]")
            requirement_id = requirement.get("requirement_id")
            if not isinstance(requirement_id, str) or not requirement_id:
                raise ContractError(f"claims.{claim_id}.requirements[{index}] requires requirement_id")
            requirement_ids.append(requirement_id)
            evidence = _mapping(requirement.get("evidence"), f"claims.{claim_id}.{requirement_id}.evidence")
            evidence_path = _repo_path(repo_root, evidence.get("path"))
            if evidence_path in output_paths:
                raise ContractError(f"Self-attested evidence is forbidden: {evidence['path']} is an authorization output")
            if not _is_sha256(evidence.get("expected_sha256")):
                raise ContractError(f"claims.{claim_id}.{requirement_id} must declare a SHA-256")
            for field in ("expected_schema_version", "expected_status"):
                if not isinstance(evidence.get(field), str) or not evidence[field]:
                    raise ContractError(f"claims.{claim_id}.{requirement_id} must declare {field}")
            provenance = _mapping(evidence.get("provenance"), f"claims.{claim_id}.{requirement_id}.evidence.provenance")
            required_fields = provenance.get("required_fields")
            forbidden_producers = provenance.get("forbidden_producers")
            if not isinstance(required_fields, list) or not required_fields or not all(isinstance(item, str) and item for item in required_fields):
                raise ContractError(f"claims.{claim_id}.{requirement_id} must declare required provenance fields")
            if not isinstance(forbidden_producers, list) or not all(isinstance(item, str) and item for item in forbidden_producers):
                raise ContractError(f"claims.{claim_id}.{requirement_id} must declare forbidden provenance producers")
            if producer not in forbidden_producers or validator not in forbidden_producers:
                raise ContractError(f"claims.{claim_id}.{requirement_id} must forbid authorization self-attestation")
            if provenance.get("self_attested") is True or evidence.get("self_attested") is True:
                raise ContractError(f"Self-attested evidence is forbidden for {claim_id}.{requirement_id}")
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ContractError(f"claims.{claim_id}.requirements contains duplicate requirement IDs")
        missing_requirements = REQUIRED_REQUIREMENTS[claim_id] - set(requirement_ids)
        if missing_requirements:
            raise ContractError(f"claims.{claim_id} is missing required evidence: {sorted(missing_requirements)}")

    typed_claims = {claim_id: _mapping(claims[claim_id], f"claims.{claim_id}") for claim_id in CLAIM_IDS}
    return _topological_order(typed_claims)


def _load_evidence(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _evidence_binding(
    evidence: Mapping[str, Any],
    repo_root: Path,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    path_text = str(evidence["path"])
    path = _repo_path(repo_root, path_text)
    payload = _load_evidence(path)
    observed_sha256 = _sha256_file(path) if path.is_file() else None
    provenance_policy = _mapping(evidence["provenance"], "evidence provenance")
    required_provenance_fields = [str(value) for value in provenance_policy["required_fields"]]
    observed_provenance = {
        field: payload.get(field) if payload is not None else None
        for field in required_provenance_fields
    }
    producer = payload.get("producer") if payload is not None else None
    forbidden_producers = {str(value) for value in provenance_policy["forbidden_producers"]}
    self_attestation_detected = bool(
        producer in forbidden_producers
        or (payload is not None and payload.get("self_attested") is True)
        or (payload is not None and payload.get("attested_by") in {authorization["artifact_producer"], authorization["validator_producer"]})
    )
    checks = {
        "path_exists": path.is_file(),
        "sha256_matches": observed_sha256 == evidence["expected_sha256"],
        "schema_matches": payload is not None and payload.get("schema_version") == evidence["expected_schema_version"],
        "status_matches": payload is not None and payload.get("status") == evidence["expected_status"],
        "provenance_complete": payload is not None and all(_nonempty(value) for value in observed_provenance.values()),
        "not_self_attested": not self_attestation_detected,
    }
    return {
        "path": path_text,
        "expected_sha256": evidence["expected_sha256"],
        "observed_sha256": observed_sha256,
        "expected_schema_version": evidence["expected_schema_version"],
        "observed_schema_version": payload.get("schema_version") if payload is not None else None,
        "expected_status": evidence["expected_status"],
        "observed_status": payload.get("status") if payload is not None else None,
        "provenance_policy": {
            "required_fields": required_provenance_fields,
            "forbidden_producers": sorted(forbidden_producers),
        },
        "observed_provenance": observed_provenance,
        "self_attestation_detected": self_attestation_detected,
        "checks": checks,
        "valid": all(checks.values()),
    }


def _as_build_arguments(config_path: str | Path, repo_root: Path | None) -> tuple[Path, Path]:
    candidate = Path(config_path)
    if candidate.is_dir():
        return candidate.resolve(), (candidate / DEFAULT_CONFIG).resolve()
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    return root.resolve(), _config_path(candidate, root.resolve())


def build(config_path: str | Path = DEFAULT_CONFIG, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Return deterministic authorization audits without writing them to disk."""
    resolved_root, resolved_config = _as_build_arguments(config_path, repo_root)
    audit_root = (resolved_root / DEFAULT_AUDIT_ROOT).resolve()
    contract = _load_contract(resolved_config, resolved_root)
    topological_order = _validate_contract(contract, resolved_root, audit_root)
    authorization_config = _mapping(contract["authorization"], "authorization")
    claims = _mapping(contract["claims"], "claims")

    bindings_by_claim: dict[str, list[dict[str, Any]]] = {}
    for claim_id in CLAIM_IDS:
        claim = _mapping(claims[claim_id], f"claims.{claim_id}")
        bindings_by_claim[claim_id] = [
            {
                "requirement_id": str(requirement["requirement_id"]),
                "evidence": _evidence_binding(_mapping(requirement["evidence"], "evidence"), resolved_root, authorization_config),
            }
            for requirement in claim["requirements"]
        ]

    claim_rows: dict[str, dict[str, Any]] = {}
    for claim_id in topological_order:
        dependencies = [str(value) for value in claims[claim_id]["depends_on_claims"]]
        requirement_rows = bindings_by_claim[claim_id]
        missing_requirements = [row["requirement_id"] for row in requirement_rows if not row["evidence"]["valid"]]
        blocked_dependencies = [dependency for dependency in dependencies if claim_rows[dependency]["status"] != "authorized"]
        authorized = not missing_requirements and not blocked_dependencies
        claim_rows[claim_id] = {
            "claim_id": claim_id,
            "status": "authorized" if authorized else "blocked",
            "authorization_scope": "claim_scoped_only",
            "depends_on_claims": dependencies,
            "blocked_dependency_claims": blocked_dependencies,
            "requirements_ready": not missing_requirements,
            "blocked_requirement_ids": missing_requirements,
            "evidence_bindings": requirement_rows,
            "next_action": "claim_scope_authorized" if authorized else f"materialize_independent_evidence_for_{missing_requirements[0]}" if missing_requirements else f"resolve_upstream_claim_{blocked_dependencies[0]}",
        }

    ordered_claim_rows = [claim_rows[claim_id] for claim_id in CLAIM_IDS]
    unauthorized_claims = [claim_id for claim_id in CLAIM_IDS if claim_rows[claim_id]["status"] != "authorized"]
    self_attested_records = [
        f"{claim_id}.{row['requirement_id']}"
        for claim_id in CLAIM_IDS
        for row in bindings_by_claim[claim_id]
        if row["evidence"]["self_attestation_detected"]
    ]
    all_bindings = [row for claim_id in CLAIM_IDS for row in bindings_by_claim[claim_id]]
    all_evidence_valid = all(row["evidence"]["valid"] for row in all_bindings)
    graph_edges = [
        {"from_claim": dependency, "to_claim": claim_id}
        for claim_id in CLAIM_IDS
        for dependency in claims[claim_id]["depends_on_claims"]
    ]
    dependency_graph = {
        "schema_version": "neuronsifter-baseline-claim-dependency-graph-v1",
        "contract_id": authorization_config["contract_id"],
        "status": "valid",
        "nodes": list(CLAIM_IDS),
        "edges": graph_edges,
        "topological_order": topological_order,
        "cycle_free": True,
        "global_fail_closed": True,
    }
    semantic_status = (
        "invalid_evidence_self_attestation" if self_attested_records
        else "passed" if all_evidence_valid
        else "valid_contract_evidence_blocked"
    )
    semantic_validation = {
        "schema_version": "neuronsifter-baseline-registry-semantic-validation-v1",
        "contract_id": authorization_config["contract_id"],
        "status": semantic_status,
        "config_schema_valid": True,
        "dependency_graph_valid": True,
        "required_requirement_sets_valid": True,
        "evidence_record_count": len(all_bindings),
        "valid_evidence_record_count": sum(row["evidence"]["valid"] for row in all_bindings),
        "self_attested_evidence_records": self_attested_records,
        "evidence_records": [
            {"claim_id": claim_id, "requirement_id": row["requirement_id"], "evidence": row["evidence"]}
            for claim_id in CLAIM_IDS
            for row in bindings_by_claim[claim_id]
        ],
    }
    global_authorized = not unauthorized_claims
    authorization = {
        "schema_version": "neuronsifter-baseline-claim-scoped-authorization-v1",
        "contract_id": authorization_config["contract_id"],
        "status": "authorized" if global_authorized else "blocked",
        "global_fail_closed": True,
        "global_activation_requires_all_claims": True,
        "global_authorized": global_authorized,
        "global_blocked_claim_ids": unauthorized_claims,
        "claim_scoped_authorization": ordered_claim_rows,
        "next_action": "global_claim_scope_authorized" if global_authorized else f"resolve_{unauthorized_claims[0]}_claim_scope",
    }
    return {
        "dependency_graph": dependency_graph,
        "authorization": authorization,
        "semantic_validation": semantic_validation,
    }


def build_all(config_path: str | Path = DEFAULT_CONFIG, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Compatibility alias for callers that use builder-style build_all APIs."""
    return build(config_path, repo_root=repo_root)


def write_all(payload: Mapping[str, Any], audit_root: str | Path = DEFAULT_AUDIT_ROOT, *, repo_root: Path | None = None) -> None:
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    target_root = Path(audit_root)
    if not target_root.is_absolute():
        target_root = root / target_root
    for key, filename in OUTPUT_FILENAMES.items():
        write_json(target_root / filename, payload[key])


def main() -> None:
    args = parse_args()
    payload = build(args.config)
    write_all(payload, args.audit_root)
    print(json.dumps({"status": payload["authorization"]["status"], "global_authorized": payload["authorization"]["global_authorized"]}, sort_keys=True))


if __name__ == "__main__":
    main()
