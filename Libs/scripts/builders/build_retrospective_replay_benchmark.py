"""Materialize a sealed, time-aware retrospective public intervention benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
BENCHMARK_CONFIG_PATH = REPO_ROOT / "Libs/configs/data/retrospective_replay_benchmark.yaml"
BENCHMARK_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/retrospective_replay_benchmark.schema.json"
EVIDENCE_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/retrospective_public_therapeutic_evidence.schema.json"
BUILDER_PATH = REPO_ROOT / "Libs/scripts/builders/build_retrospective_replay_benchmark.py"
OUTPUT_PATH = REPO_ROOT / "Output/audits/retrospective_replay_benchmark.json"
LEAKAGE_AUDIT_PATH = REPO_ROOT / "Output/audits/retrospective_replay_leakage_audit.json"

EVIDENCE_FIELDS = (
    "evidence_uid", "publication_uid", "study_uid", "source_snapshot_uid", "publication_date",
    "public_access_status", "source_content_hash", "source_locator", "measurement_protocol",
    "biological_context_uid", "common_control_uid", "structured_action_uid", "action_family_uid",
    "target_program_uid", "regimen_family_uid", "objective_uid", "evidence_grade", "candidate_set_uid",
    "outcome_uid", "replicate_uid", "numeric_outcome", "arm_mean", "biological_replicate_count", "uncertainty", "uncertainty_kind", "unit",
    "outcome_visibility", "target_retrieval_blocked", "tool_output_only", "evidence_role",
    "extraction_method", "extraction_artifact_hash", "decision_domain_classification", "split_role",
)
SNAPSHOT_FIELDS = (
    "episode_uid", "candidate_set_uid", "knowledge_cutoff", "source_snapshot_uid",
    "queryable_publication_uids", "hidden_publication_uid", "queryable_content_hashes",
    "hidden_content_hashes", "target_outcome_uid", "target_retrieval_blocked", "memory_reset_uid",
)
HIDDEN_FIELDS = (
    "episode_uid", "candidate_set_uid", "knowledge_cutoff", "hidden_publication_uid",
    "hidden_publication_date", "hidden_outcome_uid", "hidden_content_hash", "reveal_policy", "memory_reset_uid",
)
FORMAL_GRADES = {"A", "B", "C"}
MIN_INDEPENDENT_DECISION_SENSITIVE_GROUPS = 3
GROUP_FIELDS = (
    "publication_uid", "source_snapshot_uid", "study_uid", "biological_context_uid",
    "action_family_uid", "target_program_uid", "regimen_family_uid", "objective_uid",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes the repository: {value}")
    return path


def load_contract() -> dict[str, Any]:
    payload = yaml.safe_load(BENCHMARK_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    contract = payload.get("retrospective_replay_benchmark")
    schema = json.loads(BENCHMARK_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or not isinstance(schema, dict):
        raise TypeError("Retrospective replay benchmark contract and schema must be mappings")
    errors = sorted(Draft202012Validator(schema).iter_errors(contract), key=lambda error: list(error.absolute_path))
    if errors:
        detail = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Retrospective replay benchmark schema validation failed: {detail}")
    return contract


def _read_tsv(path: Path, fields: Iterable[str]) -> tuple[list[dict[str, str]], str | None]:
    if not path.is_file():
        return [], "missing"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return [], "missing_header"
        missing = sorted(set(fields) - set(reader.fieldnames))
        if missing:
            return [], f"missing_fields:{','.join(missing)}"
        rows = [{key: str(value or "").strip() for key, value in row.items()} for row in reader]
    return rows, None


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _parse_date(value: str) -> date:
    raw = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return date.fromisoformat(value.strip())


def _finite_number(value: str) -> bool:
    try:
        return value not in {"", "explicit_unknown", "explicit_unavailable"} and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_evidence_rows(rows: list[dict[str, str]]) -> list[str]:
    """Reject malformed evidence before candidate-set qualification or split writes."""

    blockers: list[str] = []
    known_grades = FORMAL_GRADES | {"D", "M"}
    hash_pattern = re.compile(r"^[0-9a-fA-F]{64}$")
    optional_fields = {
        "replicate_uid", "numeric_outcome", "arm_mean", "extraction_method",
        "extraction_artifact_hash", "decision_domain_classification", "split_role",
    }
    for index, row in enumerate(rows, start=2):
        prefix = f"row_{index}"
        for field in EVIDENCE_FIELDS:
            if not row.get(field, "") and field not in optional_fields:
                blockers.append(f"{prefix}:missing_{field}")
        grade = row.get("evidence_grade")
        if grade not in known_grades:
            blockers.append(f"{prefix}:unknown_evidence_grade")
        if row.get("public_access_status") != "public":
            blockers.append(f"{prefix}:non_public_source")
        if row.get("source_content_hash") and not hash_pattern.fullmatch(row["source_content_hash"]):
            blockers.append(f"{prefix}:invalid_source_content_hash")
        try:
            _parse_date(row.get("publication_date", ""))
        except ValueError:
            blockers.append(f"{prefix}:invalid_publication_date")
        if row.get("outcome_visibility") not in {"available_pre_cutoff", "hidden_future"}:
            blockers.append(f"{prefix}:invalid_outcome_visibility")
        if row.get("target_retrieval_blocked", "").lower() not in {"true", "false"}:
            blockers.append(f"{prefix}:invalid_target_retrieval_flag")
        if row.get("tool_output_only", "").lower() not in {"true", "false"}:
            blockers.append(f"{prefix}:invalid_tool_output_flag")
        if row.get("outcome_visibility") == "hidden_future" and row.get("target_retrieval_blocked", "").lower() != "true":
            blockers.append(f"{prefix}:hidden_outcome_not_blocked")
        if grade in {"B", "C"}:
            if not _finite_number(row.get("arm_mean", "")):
                blockers.append(f"{prefix}:summary_mean_missing")
            try:
                if int(row.get("biological_replicate_count", "0")) < 2:
                    blockers.append(f"{prefix}:replicate_count_below_two")
            except ValueError:
                blockers.append(f"{prefix}:invalid_replicate_count")
            if not _finite_number(row.get("uncertainty", "")):
                blockers.append(f"{prefix}:summary_uncertainty_missing")
            if not row.get("uncertainty_kind"):
                blockers.append(f"{prefix}:uncertainty_kind_missing")
        if grade == "A" and (not row.get("replicate_uid") or not _finite_number(row.get("numeric_outcome", ""))):
            blockers.append(f"{prefix}:replicate_measurement_missing")
        if grade == "C" and (not row.get("extraction_method") or not hash_pattern.fullmatch(row.get("extraction_artifact_hash", ""))):
            blockers.append(f"{prefix}:extraction_provenance_missing")
        if grade in {"D", "M"} and row.get("evidence_role") in {"empirical_truth", "benchmark_ground_truth", "retrospective_endpoint_label"}:
            blockers.append(f"{prefix}:non_formal_grade_promoted_to_truth")
    return sorted(set(blockers))


def qualify_candidate_sets(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["candidate_set_uid"], []).append(row)
    candidates: list[dict[str, Any]] = []
    blockers: list[str] = []
    for candidate_set_uid, members in sorted(grouped.items()):
        formal = [row for row in members if row["evidence_grade"] in FORMAL_GRADES]
        controls = {row["common_control_uid"] for row in formal if row["common_control_uid"]}
        action_uids = sorted(
            {
                row["structured_action_uid"]
                for row in formal
                if row["structured_action_uid"] and row["structured_action_uid"] not in controls
            }
        )
        contexts = {tuple(row[field] for field in GROUP_FIELDS) for row in formal}
        controls = {row["common_control_uid"] for row in formal if row["common_control_uid"]}
        valid_uncertainty = all(_finite_number(row["uncertainty"]) for row in formal)
        valid_replicates = all(int(row["biological_replicate_count"] or "0") >= 2 for row in formal)
        public = all(row["public_access_status"] == "public" for row in formal)
        tool_free = all(not _bool(row["tool_output_only"]) for row in formal)
        decision_class = next((row.get("decision_domain_classification", "") for row in members if row.get("decision_domain_classification")), "")
        legal = bool(
            len(formal) == len(members)
            and len(action_uids) >= 2
            and len(contexts) == 1
            and len(controls) == 1
            and valid_uncertainty
            and valid_replicates
            and public
            and tool_free
        )
        qualified = legal and decision_class in {"decision_sensitive", "weakly_decision_sensitive"}
        blockers_for_set: list[str] = []
        if len(action_uids) < 2:
            blockers_for_set.append("fewer_than_two_formal_actions")
        if len(contexts) != 1:
            blockers_for_set.append("context_mismatch")
        if len(controls) != 1:
            blockers_for_set.append("common_control_missing_or_mismatched")
        if not valid_uncertainty:
            blockers_for_set.append("variance_missing_or_non_numeric")
        if not valid_replicates:
            blockers_for_set.append("biological_replicate_count_below_two")
        if not public:
            blockers_for_set.append("non_public_source")
        if not tool_free:
            blockers_for_set.append("tool_output_presented_as_empirical_truth")
        if decision_class not in {"decision_sensitive", "weakly_decision_sensitive"}:
            blockers_for_set.append("decision_sensitivity_not_qualified")
        candidates.append({
            "candidate_set_uid": candidate_set_uid,
            "publication_uids": sorted({row["publication_uid"] for row in members}),
            "source_snapshot_uids": sorted({row["source_snapshot_uid"] for row in members}),
            "action_uids": action_uids,
            "control_uids": sorted(controls),
            "row_count": len(members),
            "evidence_grades": sorted({row["evidence_grade"] for row in members}),
            "decision_classification": decision_class,
            "legal": legal,
            "qualified": qualified,
            "blockers": sorted(set(blockers_for_set)),
            "split_role": next((row["split_role"] for row in members if row["split_role"]), ""),
        })
        blockers.extend(f"{candidate_set_uid}:{item}" for item in blockers_for_set)
    return candidates, blockers


def audit_snapshot_leakage(snapshot_rows: list[dict[str, str]], hidden_rows: list[dict[str, str]]) -> dict[str, Any]:
    blockers: list[str] = []
    hidden_publications: set[str] = set()
    hidden_hashes: set[str] = set()
    snapshot_by_episode: dict[str, dict[str, str]] = {}
    hidden_by_episode: dict[str, dict[str, str]] = {}
    for row in hidden_rows:
        if not row["episode_uid"] or row["episode_uid"] in hidden_by_episode:
            blockers.append(f"duplicate_or_missing_hidden_episode:{row.get('episode_uid', '')}")
        hidden_by_episode[row["episode_uid"]] = row
        hidden_publications.add(row["hidden_publication_uid"])
        hidden_hashes.add(row["hidden_content_hash"])
        if row["reveal_policy"] != "sequential_reveal_after_terminal_decision":
            blockers.append(f"hidden_reveal_policy:{row['episode_uid']}")
        if row["memory_reset_uid"] == "":
            blockers.append(f"memory_reset_missing:{row['episode_uid']}")
    for row in snapshot_rows:
        if not row["episode_uid"] or row["episode_uid"] in snapshot_by_episode:
            blockers.append(f"duplicate_or_missing_snapshot_episode:{row.get('episode_uid', '')}")
        snapshot_by_episode[row["episode_uid"]] = row
    if set(snapshot_by_episode) != set(hidden_by_episode):
        blockers.append("snapshot_hidden_episode_sets_differ")
    for episode_uid in sorted(set(snapshot_by_episode) & set(hidden_by_episode)):
        snapshot = snapshot_by_episode[episode_uid]
        hidden = hidden_by_episode[episode_uid]
        for field in ("candidate_set_uid", "knowledge_cutoff", "hidden_publication_uid", "memory_reset_uid"):
            if snapshot.get(field) != hidden.get(field):
                blockers.append(f"manifest_identity_mismatch:{episode_uid}:{field}")
        if snapshot.get("target_outcome_uid") != hidden.get("hidden_outcome_uid"):
            blockers.append(f"manifest_identity_mismatch:{episode_uid}:target_outcome_uid")
        if hidden.get("hidden_content_hash") not in {value for value in snapshot.get("hidden_content_hashes", "").split(",") if value}:
            blockers.append(f"manifest_identity_mismatch:{episode_uid}:hidden_content_hash")
    for row in snapshot_rows:
        try:
            cutoff = _parse_date(row["knowledge_cutoff"])
        except ValueError:
            blockers.append(f"invalid_cutoff:{row['episode_uid']}")
            continue
        queryable_pubs = {value for value in row["queryable_publication_uids"].split(",") if value}
        queryable_hashes = {value for value in row["queryable_content_hashes"].split(",") if value}
        hidden_for_episode = {row["hidden_publication_uid"]}
        hidden_content_hashes = {value for value in row["hidden_content_hashes"].split(",") if value}
        if not row["source_snapshot_uid"] or not queryable_pubs or not queryable_hashes:
            blockers.append(f"queryable_snapshot_incomplete:{row['episode_uid']}")
        if hidden_for_episode & queryable_pubs or hidden_content_hashes & queryable_hashes:
            blockers.append(f"target_directly_retrievable:{row['episode_uid']}")
        if hidden_publications & queryable_pubs or hidden_hashes & queryable_hashes:
            blockers.append(f"hidden_citation_or_content_overlap:{row['episode_uid']}")
        if not _bool(row["target_retrieval_blocked"]):
            blockers.append(f"target_retrieval_not_blocked:{row['episode_uid']}")
        if row["hidden_publication_uid"] == "" or row["target_outcome_uid"] == "":
            blockers.append(f"hidden_target_identity_missing:{row['episode_uid']}")
        if row["memory_reset_uid"] == "":
            blockers.append(f"memory_reset_missing:{row['episode_uid']}")
        hidden_date = next((item.get("hidden_publication_date") for item in hidden_rows if item.get("hidden_publication_uid") == row["hidden_publication_uid"]), "")
        if hidden_date:
            try:
                if _parse_date(hidden_date) <= cutoff:
                    blockers.append(f"hidden_publication_not_future:{row['episode_uid']}")
            except ValueError:
                blockers.append(f"invalid_hidden_publication_date:{row['episode_uid']}")
    unique = sorted(set(blockers))
    return {
        "status": "passed" if not unique else "failed",
        "blocker_count": len(unique),
        "blockers": unique,
        "snapshot_count": len(snapshot_rows),
        "hidden_outcome_count": len(hidden_rows),
        "direct_target_retrieval_count": sum(item.startswith("target_directly_retrievable:") for item in unique),
    }


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(*, write_output: bool = True) -> dict[str, Any]:
    contract = load_contract()
    config_payload = yaml.safe_load(BENCHMARK_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(config_payload, Mapping):
        raise TypeError("Benchmark config must be a mapping")
    evidence_contract_path = _repo_path(str(contract["evidence_contract"]))
    evidence_contract_payload = yaml.safe_load(evidence_contract_path.read_text(encoding="utf-8")) or {}
    evidence_contract = evidence_contract_payload.get("retrospective_public_therapeutic_evidence")
    evidence_schema = json.loads(EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(evidence_contract, Mapping):
        raise TypeError("Bound retrospective evidence contract is missing")
    evidence_errors = list(Draft202012Validator(evidence_schema).iter_errors(evidence_contract))
    if evidence_errors:
        raise ValueError("Bound retrospective evidence contract does not validate")

    inputs = contract["inputs"]
    evidence_path = _repo_path(str(inputs["evidence_registry"]))
    snapshot_path = _repo_path(str(inputs["tool_access_snapshot_manifest"]))
    hidden_path = _repo_path(str(inputs["hidden_outcome_manifest"]))
    evidence_rows, evidence_error = _read_tsv(evidence_path, EVIDENCE_FIELDS)
    snapshot_rows, snapshot_error = _read_tsv(snapshot_path, SNAPSHOT_FIELDS)
    hidden_rows, hidden_error = _read_tsv(hidden_path, HIDDEN_FIELDS)
    row_blockers = validate_evidence_rows(evidence_rows)
    candidates, candidate_blockers = qualify_candidate_sets(evidence_rows) if not row_blockers else ([], [])
    qualified = [row for row in candidates if row["qualified"]]
    qualified_candidate_uids = {row["candidate_set_uid"] for row in qualified}
    qualified_evidence_rows = [
        row for row in evidence_rows if row["candidate_set_uid"] in qualified_candidate_uids
    ]
    independent_decision_sensitive_groups = {
        (row["publication_uid"], row["biological_context_uid"])
        for row in qualified_evidence_rows
    }
    leakage = audit_snapshot_leakage(snapshot_rows, hidden_rows)
    blockers: list[str] = []
    if evidence_error:
        blockers.append(f"evidence_registry:{evidence_error}")
    blockers.extend(f"evidence_registry:{item}" for item in row_blockers)
    if not qualified:
        blockers.append("no_decision_sensitive_candidate_sets")
    elif len(independent_decision_sensitive_groups) < MIN_INDEPENDENT_DECISION_SENSITIVE_GROUPS:
        blockers.append(
            "insufficient_independent_decision_sensitive_groups:"
            f"{len(independent_decision_sensitive_groups)}<{MIN_INDEPENDENT_DECISION_SENSITIVE_GROUPS}"
        )
    if snapshot_error:
        blockers.append(f"tool_access_snapshot:{snapshot_error}")
    if hidden_error:
        blockers.append(f"hidden_outcome_manifest:{hidden_error}")
    if leakage["status"] != "passed":
        blockers.extend(f"leakage:{item}" for item in leakage["blockers"])
    split_files_created = False
    split_paths = {role: _repo_path(str(value)) for role, value in (("retrospective_train", contract["outputs"]["train_split"]), ("retrospective_validation", contract["outputs"]["validation_split"]), ("retrospective_development_eval", contract["outputs"]["development_eval_split"]))}
    if qualified and not blockers:
        role_groups = {role: [] for role in split_paths}
        for row in evidence_rows:
            role = row["split_role"]
            if not role:
                blockers.append("split_role_assignment_missing")
            elif role not in role_groups:
                blockers.append(f"invalid_split_role:{role}")
            else:
                role_groups[role].append(row)
        if not blockers:
            for role, rows in role_groups.items():
                _write_split(split_paths[role], rows)
            split_files_created = True
    if not qualified:
        status = "blocked_insufficient_public_quantitative_evidence"
    elif leakage["status"] != "passed":
        status = "blocked_retropective_leakage"
    elif blockers:
        status = (
            "qualified_public_candidates_but_split_insufficient"
            if "split_role_assignment_missing" in blockers
            or any(item.startswith("invalid_split_role:") for item in blockers)
            or any(item.startswith("insufficient_independent_decision_sensitive_groups:") for item in blockers)
            else "blocked_benchmark_materialization"
        )
    else:
        status = "qualified_sealed_retrospective_benchmark"
    payload: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-replay-benchmark-audit-v1",
        "contract_version": contract["contract_version"],
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_digest": _canonical_hash(contract),
        "contract_binding": artifact(BENCHMARK_CONFIG_PATH),
        "schema_binding": artifact(BENCHMARK_SCHEMA_PATH),
        "evidence_contract_binding": artifact(evidence_contract_path),
        "implementation_binding": artifact(BUILDER_PATH),
        "input_bindings": {name: artifact(path) for name, path in (("evidence_registry", evidence_path), ("tool_access_snapshot_manifest", snapshot_path), ("hidden_outcome_manifest", hidden_path)) if path.is_file()},
        "evidence_registry_present": evidence_error is None,
        "tool_access_snapshot_present": snapshot_error is None,
        "hidden_outcome_manifest_present": hidden_error is None,
        "candidate_set_count": len(candidates),
        "legal_candidate_set_count": sum(bool(row["legal"]) for row in candidates),
        "decision_sensitive_candidate_set_count": len(qualified),
        "independent_publication_count": len({publication for row in qualified for publication in row["publication_uids"]}),
        "independent_decision_sensitive_group_count": len(independent_decision_sensitive_groups),
        "minimum_independent_decision_sensitive_groups": MIN_INDEPENDENT_DECISION_SENSITIVE_GROUPS,
        "candidate_sets": candidates,
        "candidate_blockers": sorted(set(candidate_blockers)),
        "leakage_audit": leakage,
        "split_files_created": split_files_created,
        "split_policy_enforced": True,
        "objective_selection_outcome_blind": True,
        "current_live_web_search_allowed": False,
        "protected_holdout_access": 0,
        "publication_claim_eligible": False,
        "world_action_authorized": False,
        "agent_in_twin_authorized": False,
        "coevolution_authorized": False,
        "blockers": sorted(set(blockers)),
        "next_route": (
            "materialize_public_evidence_registry"
            if not qualified
            else (
                "materialize_independent_public_contexts_and_sealed_snapshots"
                if any(
                    item.startswith("insufficient_independent_decision_sensitive_groups:")
                    for item in blockers
                )
                else "materialize_sealed_snapshots_and_independent_split_roles"
            )
            if blockers
            else "formal_digital_twin_qualification"
        ),
    }
    if write_output and OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if existing.get("contract_digest") != payload["contract_digest"]:
            raise FileExistsError("Retrospective replay benchmark audit exists with a different contract digest")
        if (
            existing.get("implementation_binding") == payload["implementation_binding"]
            and existing.get("input_bindings") == payload["input_bindings"]
        ):
            return existing
    if write_output:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LEAKAGE_AUDIT_PATH.write_text(json.dumps(leakage, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(write_output=not args.no_write), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
