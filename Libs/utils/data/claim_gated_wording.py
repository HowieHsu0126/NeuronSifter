from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Mapping

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_json, write_tsv


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_COLUMNS = [
    "document_path",
    "line",
    "document_role",
    "claim_id",
    "claim_status",
    "gate_status",
    "gated_term",
    "sentence_context",
    "allowed_qualification",
    "prohibited_completion_language",
    "category",
    "status",
]


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _policy(config: Mapping[str, object]) -> dict[str, object]:
    union = _mapping(config.get("iclr_design_contract_union"))
    policy = _mapping(union.get("g0_claim_wording_audit"))
    if not policy:
        raise KeyError("iclr_design_contract_union.g0_claim_wording_audit")
    return policy


def _claim_statuses(config: Mapping[str, object]) -> dict[str, dict[str, str]]:
    union = _mapping(config.get("iclr_design_contract_union"))
    statuses: dict[str, dict[str, str]] = {}
    for row in union.get("claims", []):
        payload = _mapping(row)
        claim_id = str(payload.get("claim_id", ""))
        if claim_id:
            statuses[claim_id] = {
                "claim_status": str(payload.get("result_status", "")),
                "gate_status": str(payload.get("claim_gate_status", "")),
            }
    return statuses


def _repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _discover_documents(policy: Mapping[str, object]) -> list[tuple[Path, str]]:
    documents: list[tuple[Path, str]] = []
    for row in policy.get("public_paths", []):
        payload = _mapping(row)
        path = REPO_ROOT / str(payload["path"])
        role = str(payload["document_role"])
        if path.is_file():
            documents.append((path, role))
        elif path.is_dir():
            documents.extend((candidate, role) for candidate in sorted(path.rglob("*.md")) if candidate.is_file())
    return documents


def _is_historical(relative_path: str, policy: Mapping[str, object]) -> bool:
    return any(relative_path == prefix or relative_path.startswith(prefix) for prefix in policy.get("historical_path_prefixes", []))


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"[- ]")
    return re.compile(rf"(?<![a-z0-9]){escaped}s?(?![a-z0-9])", re.IGNORECASE)


def _context(lines: list[str], index: int) -> str:
    lower = max(index - 2, 0)
    upper = min(index + 3, len(lines))
    return " ".join(line.strip() for line in lines[lower:upper] if line.strip())


def _find_term(value: str, terms: list[str]) -> str:
    lowered = value.lower()
    for term in terms:
        if term.lower() in lowered:
            return term
    return ""


def _is_gate_passed(status: Mapping[str, str]) -> bool:
    return (
        status.get("claim_status") == "completed_empirical"
        and status.get("gate_status") == "manuscript_allowed"
    )


def _has_project_subject(context: str, document_role: str) -> bool:
    lowered = context.lower()
    return document_role == "current_public" or "neuronsifter" in lowered


def _allowed_qualification(context: str, policy: Mapping[str, object]) -> str:
    lowered = context.lower()
    for term in policy.get("allowed_qualification_terms", []):
        value = str(term)
        if value.lower() in lowered:
            return value
    return ""


def _completion_language(term: str, context: str, policy: Mapping[str, object]) -> str:
    lowered = context.lower()
    foundation_terms = {
        "foundation model",
        "foundation digital twin",
    }
    if term in foundation_terms:
        applicable = [
            "is a validated",
            "has learned",
            "demonstrates",
            "is a reliable",
            "reliable digital twin",
            "validated digital twin",
        ]
        return _find_term(context, applicable)
    if term == "reliable screening":
        return _find_term(context, ["is a reliable"])
    if term in {"reduces screening cost", "reduces wet-lab cost"}:
        return term if term in lowered else ""
    return ""


def _negated_completion(completion: str, context: str) -> bool:
    if not completion:
        return False
    lowered = context.lower()
    index = lowered.find(completion.lower())
    prefix = lowered[max(index - 48, 0):index]
    return any(marker in prefix for marker in ["rather than", "not ", "no ", "without", "do not", "must not", "不是", "未", "不能", "不得"])


def audit_document_text(
    *,
    document_path: str,
    document_role: str,
    text: str,
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    policy = _policy(config)
    statuses = _claim_statuses(config)
    lines = text.splitlines()
    rows: list[dict[str, object]] = []
    historical = _is_historical(document_path, policy)
    for term_row in policy.get("gated_terms", []):
        term_payload = _mapping(term_row)
        term = str(term_payload["term"])
        claim_id = str(term_payload["claim_id"])
        status = statuses[claim_id]
        pattern = _term_pattern(term)
        for line_index, line in enumerate(lines):
            if not pattern.search(line):
                continue
            context = _context(lines, line_index)
            prohibited = _completion_language(term, context, policy)
            qualification = _allowed_qualification(context, policy)
            gate_passed = _is_gate_passed(status)
            generic_reference = not _has_project_subject(line, document_role)
            contract_reference = document_role == "design_contract" and (
                any(token in context.lower() for token in ["claim", "gate", "c1", "c2", "c3", "c4", "c5"])
                or "禁止" in context
                or "允许" in context
            )
            allowed = historical or generic_reference or contract_reference or gate_passed or bool(qualification)
            category = "allowed"
            if prohibited and not _negated_completion(prohibited, context) and not historical and not contract_reference and not gate_passed:
                allowed = False
                category = "prohibited_completion_language"
            elif not allowed:
                category = "missing_pending_gate_qualification"
            rows.append(
                {
                    "document_path": document_path,
                    "line": line_index + 1,
                    "document_role": document_role,
                    "claim_id": claim_id,
                    "claim_status": status["claim_status"],
                    "gate_status": status["gate_status"],
                    "gated_term": term,
                    "sentence_context": context,
                    "allowed_qualification": qualification or ("historical" if historical else "contract_reference" if contract_reference else "generic_reference" if generic_reference else "passed_gate" if gate_passed else ""),
                    "prohibited_completion_language": prohibited,
                    "category": category,
                    "status": "allowed" if allowed else "violation",
                }
            )
    return rows


def build_claim_wording_audit(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    policy = _policy(config)
    rows: list[dict[str, object]] = []
    for path, role in _discover_documents(policy):
        rows.extend(
            audit_document_text(
                document_path=_repo_relative(path),
                document_role=role,
                text=path.read_text(encoding="utf-8"),
                config=config,
            )
        )
    violations = [row for row in rows if row["status"] == "violation"]
    claim_statuses = _claim_statuses(config)
    return {
        "schema_version": str(policy["schema_version"]),
        "status": "pass" if not violations else "blocked",
        "owner": str(policy["owner"]),
        "producer_command": str(policy["producer_command"]),
        "claim_statuses": claim_statuses,
        "documents_scanned": sorted({str(row["document_path"]) for row in rows}),
        "finding_count": len(rows),
        "violation_count": len(violations),
        "violations": violations,
        "allowed_findings": [row for row in rows if row["status"] == "allowed"],
    }, rows


def write_claim_wording_artifacts(report: Mapping[str, object], rows: list[dict[str, object]], config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    output_paths = _mapping(_policy(config).get("output_paths"))
    tsv_path = resolve_repo_path(str(output_paths["tsv"]))
    json_path = resolve_repo_path(str(output_paths["json"]))
    write_tsv(tsv_path, AUDIT_COLUMNS, rows)
    write_json(json_path, report)
    return {"tsv": tsv_path, "json": json_path}
