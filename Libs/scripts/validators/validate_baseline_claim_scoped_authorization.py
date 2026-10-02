"""Validate generated fail-closed baseline claim-scoped authorization audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_baseline_claim_scoped_authorization import (  # noqa: E402
    CLAIM_IDS,
    DEFAULT_AUDIT_ROOT,
    DEFAULT_CONFIG,
    OUTPUT_FILENAMES,
    ContractError,
    build,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate baseline claim-scoped authorization audits.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--audit-root", default=DEFAULT_AUDIT_ROOT)
    return parser.parse_args()


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _audit_root(repo_root: Path, audit_root: str | Path) -> Path:
    root = Path(audit_root)
    return root if root.is_absolute() else repo_root / root


def _graph_errors(graph: Mapping[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    order = graph.get("topological_order")
    if nodes != list(CLAIM_IDS):
        errors.append({"code": "invalid_graph_nodes", "message": "Dependency graph nodes must be C1 through C5 exactly."})
        return errors
    if not isinstance(edges, list) or not isinstance(order, list):
        return [{"code": "invalid_graph_shape", "message": "Dependency graph edges and topological_order are required lists."}]
    adjacency = {claim_id: [] for claim_id in CLAIM_IDS}
    for edge in edges:
        if not isinstance(edge, Mapping):
            errors.append({"code": "invalid_graph_edge", "message": "Dependency graph contains a non-object edge."})
            continue
        source = edge.get("from_claim")
        target = edge.get("to_claim")
        if source not in adjacency or target not in adjacency:
            errors.append({"code": "invalid_graph_edge", "message": f"Unknown graph edge {source!r} -> {target!r}."})
            continue
        adjacency[str(source)].append(str(target))
    active: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in active:
            return True
        if node in visited:
            return False
        active.add(node)
        cyclic = any(visit(child) for child in adjacency[node])
        active.remove(node)
        visited.add(node)
        return cyclic

    if any(visit(node) for node in CLAIM_IDS):
        errors.append({"code": "claim_dependency_cycle", "message": "Generated dependency graph contains a cycle."})
    if set(order) != set(CLAIM_IDS) or len(order) != len(CLAIM_IDS):
        errors.append({"code": "invalid_topological_order", "message": "Generated topological order must list every claim once."})
    else:
        position = {claim_id: index for index, claim_id in enumerate(order)}
        if any(position[str(edge["from_claim"])] >= position[str(edge["to_claim"])] for edge in edges if isinstance(edge, Mapping) and edge.get("from_claim") in position and edge.get("to_claim") in position):
            errors.append({"code": "invalid_topological_order", "message": "Generated topological order violates a dependency edge."})
    if graph.get("cycle_free") is not True or graph.get("status") != "valid":
        errors.append({"code": "invalid_graph_status", "message": "Generated dependency graph must report a valid cycle-free state."})
    return errors


def _authorization_errors(authorization: Mapping[str, Any], semantic: Mapping[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    claims = authorization.get("claim_scoped_authorization")
    if authorization.get("global_fail_closed") is not True or authorization.get("global_activation_requires_all_claims") is not True:
        errors.append({"code": "global_fail_closed_disabled", "message": "Generated authorization must preserve both global fail-closed flags."})
    if not isinstance(claims, list) or [row.get("claim_id") for row in claims if isinstance(row, Mapping)] != list(CLAIM_IDS):
        errors.append({"code": "invalid_claim_authorization_rows", "message": "Generated authorization must contain ordered C1-C5 claim rows."})
        return errors
    unauthorized = [row["claim_id"] for row in claims if row.get("status") != "authorized"]
    expected_global = not unauthorized
    if authorization.get("global_authorized") is not expected_global:
        errors.append({"code": "global_authorization_invariant", "message": "Global authorization must require every claim scope."})
    if authorization.get("status") != ("authorized" if expected_global else "blocked"):
        errors.append({"code": "global_status_invariant", "message": "Global status is inconsistent with claim-scoped authorization."})
    for row in claims:
        for binding in row.get("evidence_bindings", []):
            if not isinstance(binding, Mapping):
                errors.append({"code": "invalid_evidence_binding", "message": f"{row['claim_id']} contains a malformed evidence binding."})
                continue
            evidence = binding.get("evidence")
            if not isinstance(evidence, Mapping):
                errors.append({"code": "invalid_evidence_binding", "message": f"{row['claim_id']}.{binding.get('requirement_id')} has no evidence record."})
                continue
            if evidence.get("self_attestation_detected") is True or evidence.get("checks", {}).get("not_self_attested") is False:
                errors.append({"code": "self_attested_evidence", "message": f"{row['claim_id']}.{binding.get('requirement_id')} is self-attested."})
    if semantic.get("status") == "invalid_evidence_self_attestation" or semantic.get("self_attested_evidence_records"):
        errors.append({"code": "self_attested_evidence", "message": "Semantic validation reports self-attested evidence."})
    return errors


def validate(
    config_path: str | Path = DEFAULT_CONFIG,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Return a non-promotion validation result for generated audit artifacts."""
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    try:
        expected = build(config_path, repo_root=root)
    except (ContractError, FileNotFoundError, TypeError) as exc:
        return {"status": "failed", "blocker_count": 1, "blockers": [{"code": "invalid_contract", "message": str(exc)}]}

    blockers: list[dict[str, str]] = []
    actual: dict[str, Mapping[str, Any]] = {}
    target_root = _audit_root(root, audit_root)
    for key, filename in OUTPUT_FILENAMES.items():
        path = target_root / filename
        payload = _read_json(path)
        if payload is None:
            blockers.append({"code": "missing_or_invalid_generated_artifact", "message": str(path)})
            continue
        actual[key] = payload
        if payload != expected[key]:
            blockers.append({"code": "stale_or_tampered_generated_artifact", "message": filename})

    if "dependency_graph" in actual:
        blockers.extend(_graph_errors(actual["dependency_graph"]))
    if "authorization" in actual and "semantic_validation" in actual:
        blockers.extend(_authorization_errors(actual["authorization"], actual["semantic_validation"]))
    authorization = actual.get("authorization", expected["authorization"])
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "authorization_status": authorization.get("status"),
        "global_authorized": authorization.get("global_authorized"),
    }


def main() -> None:
    args = parse_args()
    report = validate(args.config, args.audit_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
