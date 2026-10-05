from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from Libs.utils.data.raw_data_registry import load_config_with_provenance
from Libs.utils.data.public_mainline_isolation import validate_public_mainline_isolation_payload


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_CLAIMS = [
    ("C1", "public_foundation_state_transfer"),
    ("C2", "structured_stochastic_world_action"),
    ("C3", "cost_aware_multi_tool_acquisition"),
    ("C4", "tri_gated_reliable_screening"),
    ("C5", "scale_aware_throughput_support"),
]
EXPECTED_SCHEMA_FAMILIES = {
    "acquisition_request",
    "checkpoint_manifest",
    "claim_metric",
    "cost_estimate",
    "observation",
    "public_source_manifest",
    "run_manifest",
    "split_manifest",
    "structured_action",
    "tool_request",
    "tool_result",
    "trajectory",
}
EXPECTED_ENUMS = {
    "document_status": ["canonical", "extends", "superseded", "historical", "reference-only"],
    "source_status": ["active_contract", "evidence_reference", "historical_reference", "generated_manifest"],
    "task_status": ["not_started", "blocked", "in_progress", "implemented", "empirically_validated", "accepted"],
    "result_status": ["completed_empirical", "failed_empirical", "blocked_preflight", "requires_experiment_result"],
    "claim_gate_status": [
        "proposed",
        "contract_frozen",
        "implementation_complete",
        "empirical_results_available",
        "statistical_gate_passed",
        "integrity_gate_passed",
        "manuscript_allowed",
        "blocked_preflight",
        "failed_empirical",
        "failed_statistics",
        "failed_integrity",
    ],
    "precedence": ["canonical_design", "execution_guidance", "evidence_basis", "generated_manifest", "historical_only"],
}
STATE_TOP_LEVEL_FIELDS = {
    "contract_version",
    "base_branch",
    "roadmap_id",
    "roadmap_revision",
    "roadmap_dir",
    "controller_stage",
    "max_parallel_rounds",
    "active_rounds",
    "roadmap_update",
    "resume_errors",
    "retry",
}
ROUND_STATE_FIELDS = {"round_id", "stage", "branch", "worktree_path", "resume_error"}
CONTROLLER_STAGES = {"dispatch-rounds", "update-roadmap", "done", "blocked"}
ROUND_STAGES = {"plan", "implement", "review", "finalize-round", "done", "blocked"}
LEGACY_PUBLIC_NAME_PATTERN = re.compile(r"mind[-_ ]?dino", re.IGNORECASE)


def resolve_repo_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def relative_repo_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _block(blockers: list[dict[str, object]], code: str, message: str, **context: object) -> None:
    blockers.append({"code": code, "message": message, "context": context})


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [_mapping(row) for row in value]


def _dotted_value(payload: Mapping[str, object], dotted_path: str) -> object:
    current: object = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _public_legacy_values(payload: object, path: str = "") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_path = f"{path}.{key}" if path else str(key)
            findings.extend(_public_legacy_values(value, child_path))
        return findings
    if isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_public_legacy_values(value, f"{path}[{index}]"))
        return findings
    if isinstance(payload, str) and LEGACY_PUBLIC_NAME_PATTERN.search(payload):
        findings.append({"path": path, "value": payload})
    return findings


def _roadmap_inventory(roadmap_path: Path) -> dict[str, object]:
    text = roadmap_path.read_text(encoding="utf-8")
    milestone_ids: list[str] = []
    direction_ids: list[str] = []
    direction_owners: dict[str, str] = {}
    current_milestone = ""
    for line in text.splitlines():
        if line.startswith("Milestone id:"):
            current_milestone = line.split(":", 1)[1].strip()
            milestone_ids.append(current_milestone)
            continue
        stripped = line.strip()
        if stripped.startswith("- Direction id:"):
            direction_id = stripped.split(":", 1)[1].strip()
            direction_ids.append(direction_id)
            if direction_id not in direction_owners:
                direction_owners[direction_id] = current_milestone
    return {
        "milestone_ids": milestone_ids,
        "direction_ids": direction_ids,
        "direction_owners": direction_owners,
        "duplicate_milestone_ids": _duplicates(milestone_ids),
        "duplicate_direction_ids": _duplicates(direction_ids),
        "text": text,
    }


def _inactive_roadmap_ownership_registry(
    union: Mapping[str, object],
    direction_owners: Mapping[str, object],
    active_lineage: Mapping[str, object],
    blockers: list[dict[str, object]],
) -> tuple[set[tuple[str, str]], list[dict[str, str]]]:
    raw_registry = union.get("inactive_roadmap_ownership_registry")
    if not isinstance(raw_registry, list):
        _block(
            blockers,
            "invalid_inactive_roadmap_ownership_registry",
            "Inactive future roadmap ownership must be an explicit list.",
        )
        return set(), []

    active_pair = (
        str(active_lineage.get("milestone_id", "")),
        str(active_lineage.get("direction_id", "")),
    )
    inactive_pairs: set[tuple[str, str]] = set()
    normalized_rows: list[dict[str, str]] = []
    for index, raw_row in enumerate(raw_registry):
        row = _mapping(raw_row)
        milestone_id = str(row.get("owning_milestone_id", ""))
        direction_id = str(row.get("owning_direction_id", ""))
        ownership_status = str(row.get("ownership_status", ""))
        pair = (milestone_id, direction_id)
        if not milestone_id or not direction_id or ownership_status != "inactive_future_roadmap":
            _block(
                blockers,
                "invalid_inactive_roadmap_ownership",
                "Inactive future roadmap ownership requires a milestone, direction, and inactive_future_roadmap status.",
                index=index,
                ownership=row,
            )
            continue
        if pair in inactive_pairs:
            _block(
                blockers,
                "duplicate_inactive_roadmap_ownership",
                "An inactive future roadmap ownership pair is declared more than once.",
                milestone_id=milestone_id,
                direction_id=direction_id,
            )
            continue
        if pair == active_pair:
            _block(
                blockers,
                "inactive_roadmap_ownership_is_active",
                "The active-lineage ownership pair cannot be registered as inactive future work.",
                milestone_id=milestone_id,
                direction_id=direction_id,
            )
            continue
        if direction_owners.get(direction_id) != milestone_id:
            _block(
                blockers,
                "invalid_inactive_roadmap_ownership",
                "An inactive future ownership pair must identify a direction in the active roadmap revision.",
                milestone_id=milestone_id,
                direction_id=direction_id,
            )
            continue
        inactive_pairs.add(pair)
        normalized_rows.append(
            {
                "owning_milestone_id": milestone_id,
                "owning_direction_id": direction_id,
                "ownership_status": ownership_status,
            }
        )
    return inactive_pairs, sorted(normalized_rows, key=lambda row: (row["owning_milestone_id"], row["owning_direction_id"]))


def _union(config: Mapping[str, object]) -> dict[str, object]:
    payload = config.get("iclr_design_contract_union")
    if not isinstance(payload, Mapping):
        raise KeyError("iclr_design_contract_union")
    return dict(payload)


def validate_config_stack_payload(
    config: Mapping[str, object],
    provenance: Mapping[str, object],
    config_path: str | Path,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    union = _union(config)
    runtime_owner = _mapping(union.get("runtime_owner"))
    owner_file = str(runtime_owner.get("owner_file", ""))
    entrypoint = str(runtime_owner.get("config_entrypoint", ""))
    include_order = [str(value) for value in provenance.get("include_order", [])]
    include_graph = _mapping(provenance.get("include_graph"))
    leaf_owners = _mapping(provenance.get("leaf_owners"))
    actual_entrypoint = str(provenance.get("entrypoint", ""))

    if actual_entrypoint != entrypoint:
        _block(blockers, "config_entrypoint_mismatch", "Resolved config entrypoint does not match the union owner.", expected=entrypoint, actual=actual_entrypoint)
    owner_count = include_order.count(owner_file)
    if owner_count != 1:
        _block(blockers, "union_owner_include_count", "The union owner must be included exactly once.", owner_file=owner_file, count=owner_count)

    forbidden_roots = [str(value).rstrip("/") for value in runtime_owner.get("forbidden_runtime_include_roots", [])]
    preview_includes = sorted(
        include_path
        for include_path in include_order
        if any(include_path == root or include_path.startswith(f"{root}/") for root in forbidden_roots)
    )
    if preview_includes:
        _block(blockers, "design_preview_runtime_include", "Design preview files cannot enter the runtime include graph.", paths=preview_includes)

    owned_prefixes = [str(value) for value in runtime_owner.get("owned_path_prefixes", [])]
    owned_leaf_paths: dict[str, list[str]] = {}
    for prefix in owned_prefixes:
        matching = {
            str(path): [str(owner) for owner in owners]
            for path, owners in leaf_owners.items()
            if str(path) == prefix or str(path).startswith(f"{prefix}.")
        }
        if not matching:
            _block(blockers, "missing_union_owned_path", "A configured union-owned path has no leaf provenance.", prefix=prefix)
        for path, owners in matching.items():
            owned_leaf_paths[path] = owners
            if owners != [owner_file]:
                _block(blockers, "union_leaf_owner_conflict", "A union-owned leaf does not have exactly one canonical owner.", path=path, owners=owners, expected_owner=owner_file)

    schemas = _mapping(config.get("schemas"))
    for family in _list_of_mappings(union.get("schema_families")):
        field_key = str(family.get("field_union_key", ""))
        leaf_path = f"schemas.{field_key}"
        owners = [str(value) for value in leaf_owners.get(leaf_path, [])]
        if field_key not in schemas:
            _block(blockers, "missing_schema_field_union", "A declared schema field union is absent from schemas.yaml.", schema_family=family.get("schema_family"), field_union_key=field_key)
        if owners != ["Libs/configs/schema/schemas.yaml"]:
            _block(blockers, "schema_field_union_owner_conflict", "A schema field union must have one schemas.yaml owner.", path=leaf_path, owners=owners)

    source_rows = _list_of_mappings(union.get("source_of_truth_registry"))
    source_domains = [str(row.get("domain_id", "")) for row in source_rows]
    for domain_id in _duplicates(source_domains):
        _block(blockers, "competing_source_of_truth", "A fact domain has multiple canonical sources.", domain_id=domain_id)

    _, isolation_report, remediation_report = validate_public_mainline_isolation_payload(config, provenance, config_path)
    blockers.extend(deepcopy(isolation_report["blockers"]))

    config_file = resolve_repo_path(config_path)
    schema_version = str(_mapping(union.get("audit_contract")).get("schema_versions", {}).get("config_stack", ""))
    status = str(_mapping(union.get("audit_contract")).get("pass_status", "pass")) if not blockers else "blocked"
    return {
        "schema_version": schema_version,
        "status": status,
        "config_path": relative_repo_path(config_file),
        "config_sha256": sha256_file(config_file),
        "include_order": include_order,
        "include_graph": include_graph,
        "union_owner": owner_file,
        "union_owner_include_count": owner_count,
        "leaf_owner_count": len(leaf_owners),
        "union_owned_leaf_count": len(owned_leaf_paths),
        "union_owned_leaf_owners": owned_leaf_paths,
        "preview_runtime_includes": preview_includes,
        "public_mainline_isolation": {
            "schema_version": isolation_report["schema_version"],
            "status": isolation_report["status"],
            "output_path": "Output/audits/public_mainline_isolation_audit.json",
            "controlled_reachability_count": isolation_report["controlled_reachability_count"],
            "config_graph_sha256": isolation_report["config_graph_sha256"],
            "isolation_audit_sha256": remediation_report["isolation_audit_sha256"],
            "blocker_count": isolation_report["blocker_count"],
        },
        "blockers": blockers,
        "blocker_count": len(blockers),
    }


def build_config_stack_contract_report(config_path: str | Path) -> dict[str, object]:
    config, provenance = load_config_with_provenance(config_path)
    return validate_config_stack_payload(config, provenance, config_path)


def build_design_coverage_report(config: Mapping[str, object]) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    union = _union(config)
    coverage = _mapping(_mapping(config.get("validation")).get("design_contract_coverage"))
    docs_root = resolve_repo_path(coverage.get("docs_root", ""))
    discovered = sorted(relative_repo_path(path) for path in docs_root.rglob("*.md") if path.is_file()) if docs_root.is_dir() else []
    declarations = _list_of_mappings(coverage.get("documents"))
    declared = [str(row.get("path", "")) for row in declarations]
    duplicates = _duplicates(declared)
    missing = sorted(set(discovered) - set(declared))
    stale = sorted(set(declared) - set(discovered))
    expected_count = int(coverage.get("expected_document_count", 0))

    if not docs_root.is_dir():
        _block(blockers, "missing_design_root", "The configured ICLR design root is missing.", docs_root=relative_repo_path(docs_root))
    if len(discovered) != expected_count:
        _block(blockers, "discovered_document_count_mismatch", "Discovered Markdown count differs from the configured contract.", expected=expected_count, actual=len(discovered))
    if duplicates:
        _block(blockers, "duplicate_document_declaration", "Design documents must have exactly one declaration.", paths=duplicates)
    if missing:
        _block(blockers, "missing_document_declaration", "Discovered design documents lack declarations.", paths=missing)
    if stale:
        _block(blockers, "stale_document_declaration", "Declarations point to absent design documents.", paths=stale)

    contract_ids = {str(row.get("contract_id", "")) for row in _list_of_mappings(union.get("contract_registry"))}
    validator_ids = {str(row.get("validator_id", "")) for row in _list_of_mappings(union.get("validator_registry"))}
    consumer_ids = {str(row.get("consumer_id", "")) for row in _list_of_mappings(union.get("consumer_registry"))}
    requirement_ids = {str(row.get("requirement_id", "")) for row in _list_of_mappings(union.get("mandatory_requirements"))}
    precedence_rows = {str(row.get("precedence_id", "")): row for row in _list_of_mappings(union.get("precedence_registry"))}
    conflict_ids = {str(row.get("conflict_id", "")) for row in _list_of_mappings(union.get("conflict_registry"))}
    allowed_document_statuses = {str(value) for value in coverage.get("allowed_document_statuses", [])}
    allowed_source_statuses = {str(value) for value in coverage.get("allowed_source_statuses", [])}
    required_fields = {
        "path",
        "document_status",
        "source_status",
        "precedence_id",
        "canonical_contract_ids",
        "owning_milestone_id",
        "owning_direction_id",
        "downstream_consumer_ids",
        "responsible_validator_ids",
        "mandatory_requirement_ids",
        "conflicts",
        "supersedes",
    }
    validated_documents: list[dict[str, object]] = []
    for row in declarations:
        path = str(row.get("path", ""))
        missing_fields = sorted(field for field in required_fields if field not in row)
        if missing_fields:
            _block(blockers, "document_metadata_missing", "A document declaration lacks required metadata.", path=path, fields=missing_fields)
        document_status = str(row.get("document_status", ""))
        source_status = str(row.get("source_status", ""))
        precedence_id = str(row.get("precedence_id", ""))
        if document_status not in allowed_document_statuses:
            _block(blockers, "unknown_document_status", "A document declaration uses an unknown status.", path=path, status=document_status)
        if source_status not in allowed_source_statuses:
            _block(blockers, "unknown_source_status", "A document declaration uses an unknown source status.", path=path, status=source_status)
        if precedence_id not in precedence_rows:
            _block(blockers, "unknown_precedence", "A document declaration references an unknown precedence id.", path=path, precedence_id=precedence_id)
        for contract_id in [str(value) for value in row.get("canonical_contract_ids", [])]:
            if contract_id not in contract_ids:
                _block(blockers, "unknown_contract_reference", "A document references an unknown canonical contract.", path=path, contract_id=contract_id)
        for validator_id in [str(value) for value in row.get("responsible_validator_ids", [])]:
            if validator_id not in validator_ids:
                _block(blockers, "unknown_validator_reference", "A document references an unknown validator.", path=path, validator_id=validator_id)
        for consumer_id in [str(value) for value in row.get("downstream_consumer_ids", [])]:
            if consumer_id not in consumer_ids:
                _block(blockers, "unknown_consumer_reference", "A document references an unknown consumer.", path=path, consumer_id=consumer_id)
        doc_requirements = [str(value) for value in row.get("mandatory_requirement_ids", [])]
        if not doc_requirements:
            _block(blockers, "document_requirement_missing", "Every document declaration must cover a mandatory requirement.", path=path)
        for requirement_id in doc_requirements:
            if requirement_id not in requirement_ids:
                _block(blockers, "unknown_requirement_reference", "A document references an unknown requirement.", path=path, requirement_id=requirement_id)
        conflicts = _list_of_mappings(row.get("conflicts"))
        if not conflicts:
            _block(blockers, "document_conflict_resolution_missing", "Every declaration must state a conflict or no-conflict resolution.", path=path)
        for conflict in conflicts:
            conflict_id = str(conflict.get("conflict_id", ""))
            resolution = str(conflict.get("resolution", "")).strip()
            if conflict_id not in conflict_ids:
                _block(blockers, "unknown_conflict_reference", "A declaration references an unknown conflict id.", path=path, conflict_id=conflict_id)
            if not resolution:
                _block(blockers, "empty_conflict_resolution", "A declaration conflict resolution is empty.", path=path, conflict_id=conflict_id)
        supersedes = [str(value) for value in row.get("supersedes", [])]
        if source_status != "active_contract" and supersedes:
            _block(blockers, "historical_override", "Historical, generated, or evidence documents cannot supersede contracts.", path=path, supersedes=supersedes)
        precedence = precedence_rows.get(precedence_id, {})
        if supersedes and not bool(precedence.get("can_supersede_canonical", False)):
            _block(blockers, "forbidden_document_supersession", "The declaration precedence cannot supersede canonical contracts.", path=path, precedence_id=precedence_id)
        validated = deepcopy(row)
        validated["coverage_status"] = "resolved" if path in discovered and path not in duplicates else "blocked"
        validated_documents.append(validated)

    audit_schema = _mapping(union.get("audit_contract"))
    status = str(audit_schema.get("pass_status", "pass")) if not blockers else "blocked"
    return {
        "schema_version": str(_mapping(audit_schema.get("schema_versions")).get("design_coverage", "")),
        "status": status,
        "contract_status": "design_contract_coverage_satisfied" if not blockers else "design_contract_coverage_blocked",
        "docs_root": str(coverage.get("docs_root", "")),
        "expected_document_count": expected_count,
        "discovered_document_count": len(discovered),
        "declared_document_count": len(declarations),
        "resolved_declaration_count": sum(1 for row in validated_documents if row["coverage_status"] == "resolved"),
        "duplicate_declarations": duplicates,
        "missing_declarations": missing,
        "stale_declarations": stale,
        "documents": sorted(validated_documents, key=lambda row: str(row.get("path", ""))),
        "blockers": blockers,
        "blocker_count": len(blockers),
    }


def _state_source_path(union: Mapping[str, object]) -> Path:
    for row in _list_of_mappings(union.get("source_of_truth_registry")):
        if str(row.get("domain_id", "")) == "controller_state":
            return resolve_repo_path(row.get("canonical_source", ""))
    raise KeyError("controller_state source of truth")


def build_orchestrator_state_report(
    config: Mapping[str, object],
    state_path: str | Path | None = None,
    state_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    union = _union(config)
    configured_state_path = resolve_repo_path(state_path) if state_path is not None else _state_source_path(union)
    state_bytes = configured_state_path.read_bytes() if state_payload is None else json.dumps(state_payload, sort_keys=True).encode("utf-8")
    before_hash = hashlib.sha256(state_bytes).hexdigest()
    state = json.loads(state_bytes.decode("utf-8")) if state_payload is None else dict(state_payload)
    if not isinstance(state, dict):
        state = {}
        _block(blockers, "orchestrator_state_not_mapping", "orchestrator/state.json must contain a JSON object.")

    actual_fields = set(state)
    missing_fields = sorted(STATE_TOP_LEVEL_FIELDS - actual_fields)
    extra_fields = sorted(actual_fields - STATE_TOP_LEVEL_FIELDS)
    if missing_fields:
        _block(blockers, "orchestrator_state_missing_fields", "Controller state lacks required fields.", fields=missing_fields)
    if extra_fields:
        _block(blockers, "orchestrator_state_extra_fields", "Controller state contains retired or unknown fields.", fields=extra_fields)
    if state.get("contract_version") != "orchestrator-v2":
        _block(blockers, "orchestrator_contract_version", "Controller state must use orchestrator-v2.", actual=state.get("contract_version"))
    if state.get("controller_stage") not in CONTROLLER_STAGES:
        _block(blockers, "orchestrator_controller_stage", "Controller stage is invalid.", actual=state.get("controller_stage"))
    max_parallel = state.get("max_parallel_rounds")
    if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or max_parallel < 1:
        _block(blockers, "orchestrator_parallel_limit", "max_parallel_rounds must be a positive integer.", actual=max_parallel)
    active_rounds = state.get("active_rounds", [])
    if not isinstance(active_rounds, list):
        _block(blockers, "orchestrator_active_rounds_type", "active_rounds must be a list.")
        active_rounds = []
    for index, round_row in enumerate(active_rounds):
        if not isinstance(round_row, Mapping):
            _block(blockers, "orchestrator_round_shape", "A live round record is not a mapping.", index=index)
            continue
        round_fields = set(round_row)
        if round_fields != ROUND_STATE_FIELDS:
            _block(blockers, "orchestrator_round_fields", "A live round record has missing or extra fields.", index=index, missing=sorted(ROUND_STATE_FIELDS - round_fields), extra=sorted(round_fields - ROUND_STATE_FIELDS))
        if round_row.get("stage") not in ROUND_STAGES:
            _block(blockers, "orchestrator_round_stage", "A live round stage is invalid.", index=index, actual=round_row.get("stage"))
        for key in ["round_id", "branch", "worktree_path"]:
            if not str(round_row.get(key, "")).strip():
                _block(blockers, "orchestrator_round_value", "A live round required value is empty.", index=index, field=key)

    contract_lineage = _mapping(union.get("active_lineage"))
    lineage = _mapping(union.get("controller_lineage")) or contract_lineage
    for field in ["roadmap_id", "roadmap_revision", "roadmap_dir"]:
        if str(state.get(field, "")) != str(lineage.get(field, "")):
            _block(blockers, "orchestrator_lineage_mismatch", "Controller state and installed union lineage differ.", field=field, state_value=state.get(field), union_value=lineage.get(field))
    roadmap_dir = resolve_repo_path(state.get("roadmap_dir", ""))
    roadmap_path = roadmap_dir / "roadmap.md"
    verification_path = roadmap_dir / "verification.md"
    roadmap_inventory: dict[str, object] = {"milestone_ids": [], "direction_ids": [], "direction_owners": {}}
    if not roadmap_path.is_file():
        _block(blockers, "missing_active_roadmap", "The active roadmap file is missing.", path=relative_repo_path(roadmap_path))
    else:
        roadmap_inventory = _roadmap_inventory(roadmap_path)
        if roadmap_inventory["duplicate_milestone_ids"]:
            _block(blockers, "duplicate_roadmap_milestone_id", "The active roadmap has duplicate milestone ids.", ids=roadmap_inventory["duplicate_milestone_ids"])
        if roadmap_inventory["duplicate_direction_ids"]:
            _block(blockers, "duplicate_roadmap_direction_id", "The active roadmap has duplicate direction ids.", ids=roadmap_inventory["duplicate_direction_ids"])
        if "## Milestones" not in str(roadmap_inventory.get("text", "")):
            _block(blockers, "roadmap_required_section", "The active roadmap lacks its Milestones section.")
    if not verification_path.is_file():
        _block(blockers, "missing_active_verification", "The active verification contract is missing.", path=relative_repo_path(verification_path))
    else:
        verification_text = verification_path.read_text(encoding="utf-8")
        for heading in ["## Baseline Checks", "## Alignment Checks", "## Task-Specific Checks", "## Manual Checks", "## Roadmap Overrides"]:
            if heading not in verification_text:
                _block(blockers, "verification_required_section", "The active verification contract lacks a required section.", heading=heading)
    selected_milestone = str(lineage.get("milestone_id", ""))
    selected_direction = str(lineage.get("direction_id", ""))
    if selected_milestone not in set(roadmap_inventory.get("milestone_ids", [])):
        _block(blockers, "unknown_selected_milestone", "The selected milestone is absent from the active roadmap.", milestone_id=selected_milestone)
    direction_owners = _mapping(roadmap_inventory.get("direction_owners"))
    if selected_direction not in direction_owners:
        _block(blockers, "unknown_selected_direction", "The selected direction is absent from the active roadmap.", direction_id=selected_direction)
    elif direction_owners[selected_direction] != selected_milestone:
        _block(blockers, "selected_direction_owner_mismatch", "The selected direction belongs to a different milestone.", direction_id=selected_direction, expected_milestone=selected_milestone, actual_milestone=direction_owners[selected_direction])

    after_hash = hashlib.sha256(configured_state_path.read_bytes()).hexdigest() if state_payload is None else hashlib.sha256(json.dumps(state_payload, sort_keys=True).encode("utf-8")).hexdigest()
    if after_hash != before_hash:
        _block(blockers, "orchestrator_state_mutated", "State validation mutated controller-owned state.", before=before_hash, after=after_hash)
    audit_contract = _mapping(union.get("audit_contract"))
    return {
        "schema_version": str(_mapping(audit_contract.get("schema_versions")).get("orchestrator_state", "")),
        "status": str(audit_contract.get("pass_status", "pass")) if not blockers else "blocked",
        "state_path": relative_repo_path(configured_state_path),
        "state_sha256_before": before_hash,
        "state_sha256_after": after_hash,
        "state_mutated": before_hash != after_hash,
        "roadmap_id": state.get("roadmap_id"),
        "roadmap_revision": state.get("roadmap_revision"),
        "roadmap_dir": state.get("roadmap_dir"),
        "selected_milestone_id": selected_milestone,
        "selected_direction_id": selected_direction,
        "active_round_count": len(active_rounds),
        "blockers": blockers,
        "blocker_count": len(blockers),
    }


def _find_cycles(graph: Mapping[str, list[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    visited: set[str] = set()
    active: list[str] = []

    def visit(node: str) -> None:
        if node in active:
            start = active.index(node)
            cycle = [*active[start:], node]
            if cycle not in cycles:
                cycles.append(cycle)
            return
        if node in visited:
            return
        active.append(node)
        for target in graph.get(node, []):
            if target in graph:
                visit(target)
        active.pop()
        visited.add(node)

    for node in graph:
        visit(node)
    return cycles


def _validate_schema_families(
    config: Mapping[str, object],
    union: Mapping[str, object],
    validator_ids: set[str],
    consumer_ids: set[str],
    blockers: list[dict[str, object]],
) -> None:
    family_rows = _list_of_mappings(union.get("schema_families"))
    family_ids = [str(row.get("schema_family", "")) for row in family_rows]
    for duplicate in _duplicates(family_ids):
        _block(blockers, "duplicate_schema_family", "A schema family is declared more than once.", schema_family=duplicate)
    if set(family_ids) != EXPECTED_SCHEMA_FAMILIES:
        _block(blockers, "schema_family_set_mismatch", "The union must contain exactly the 12 design schema families.", expected=sorted(EXPECTED_SCHEMA_FAMILIES), actual=sorted(set(family_ids)))
    schema_fields = _mapping(config.get("schemas"))
    for row in family_rows:
        family_id = str(row.get("schema_family", ""))
        source_path = resolve_repo_path(row.get("design_source", ""))
        if not source_path.is_file():
            _block(blockers, "missing_design_schema", "A design schema input is missing.", schema_family=family_id, path=relative_repo_path(source_path))
            continue
        actual_hash = sha256_file(source_path)
        if actual_hash != str(row.get("design_source_sha256", "")):
            _block(blockers, "design_schema_hash_mismatch", "A design schema hash differs from the frozen input.", schema_family=family_id, expected=row.get("design_source_sha256"), actual=actual_hash)
        schema_payload = json.loads(source_path.read_text(encoding="utf-8"))
        expected_required = [str(value) for value in schema_payload.get("required", [])]
        declared_required = [str(value) for value in row.get("required_fields", [])]
        if declared_required != expected_required:
            _block(blockers, "schema_required_fields_mismatch", "Canonical required fields differ from the design schema.", schema_family=family_id, expected=expected_required, actual=declared_required)
        field_key = str(row.get("field_union_key", ""))
        field_union = [str(value) for value in schema_fields.get(field_key, [])]
        property_fields = [str(value) for value in _mapping(schema_payload.get("properties"))]
        if set(field_union) != set(property_fields) or len(field_union) != len(property_fields):
            _block(blockers, "schema_field_union_mismatch", "Canonical field union differs from the design schema properties.", schema_family=family_id, expected=property_fields, actual=field_union)
        declared_enums = _mapping(row.get("enums"))
        expected_enums = {
            property_name: [str(value) for value in property_payload.get("enum", [])]
            for property_name, property_payload in _mapping(schema_payload.get("properties")).items()
            if isinstance(property_payload, Mapping) and "enum" in property_payload
        }
        normalized_declared_enums = {str(key): [str(value) for value in values] for key, values in declared_enums.items()}
        if normalized_declared_enums != expected_enums:
            _block(blockers, "schema_enum_mismatch", "Canonical enums differ from the design schema.", schema_family=family_id, expected=expected_enums, actual=normalized_declared_enums)
        expected_nested = {
            property_name: [str(value) for value in property_payload.get("required", [])]
            for property_name, property_payload in _mapping(schema_payload.get("properties")).items()
            if isinstance(property_payload, Mapping) and property_payload.get("required")
        }
        declared_nested = {str(key): [str(value) for value in values] for key, values in _mapping(row.get("nested_required_fields")).items()}
        if declared_nested != expected_nested:
            _block(blockers, "schema_nested_required_mismatch", "Nested required fields differ from the design schema.", schema_family=family_id, expected=expected_nested, actual=declared_nested)
        if str(row.get("canonical_runtime_owner", "")) != "Libs/configs/schema/schemas.yaml":
            _block(blockers, "schema_runtime_owner", "Every design schema family must bind to schemas.yaml.", schema_family=family_id, actual=row.get("canonical_runtime_owner"))
        binding = str(row.get("current_compatibility_binding", ""))
        try:
            _dotted_value(config, binding)
        except KeyError:
            _block(blockers, "schema_compatibility_binding", "A current schema compatibility binding does not resolve.", schema_family=family_id, binding=binding)
        if str(row.get("validator_id", "")) not in validator_ids:
            _block(blockers, "schema_validator_reference", "A schema family references an unknown validator.", schema_family=family_id, validator_id=row.get("validator_id"))
        for consumer_id in [str(value) for value in row.get("downstream_consumer_ids", [])]:
            if consumer_id not in consumer_ids:
                _block(blockers, "schema_consumer_reference", "A schema family references an unknown consumer.", schema_family=family_id, consumer_id=consumer_id)


def build_union_consistency_report(
    config: Mapping[str, object],
    provenance: Mapping[str, object] | None = None,
    config_path: str | Path = "Libs/configs/config_stack.yaml",
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    union = _union(config)
    audit_contract = _mapping(union.get("audit_contract"))
    runtime_owner = _mapping(union.get("runtime_owner"))
    owner_file = str(runtime_owner.get("owner_file", ""))
    if str(union.get("schema_version", "")) != "iclr-design-contract-union-v1":
        _block(blockers, "union_schema_version", "The union schema version is not the frozen version.", actual=union.get("schema_version"))
    if str(union.get("contract_version", "")) != "iclr-2027-design-contract-v2":
        _block(blockers, "union_contract_version", "The union contract version is not the frozen ICLR-v2 version.", actual=union.get("contract_version"))

    claims = _list_of_mappings(union.get("claims"))
    claim_ids = [str(row.get("claim_id", "")) for row in claims]
    duplicate_claim_ids = _duplicates(claim_ids)
    if duplicate_claim_ids:
        _block(blockers, "unresolved_claim_id_conflict", "Claim ids are declared more than once.", claim_ids=duplicate_claim_ids)
    actual_claims = [(str(row.get("claim_id", "")), str(row.get("meaning", ""))) for row in claims]
    if actual_claims != EXPECTED_CLAIMS:
        _block(blockers, "unresolved_claim_id_conflict", "C1-C5 ids or meanings differ from the frozen migration.", expected=EXPECTED_CLAIMS, actual=actual_claims)
    for row in claims:
        if row.get("result_status") != "requires_experiment_result":
            _block(blockers, "empirical_result_promotion", "Contract installation cannot promote a C1-C5 result.", claim_id=row.get("claim_id"), status=row.get("result_status"))
        if row.get("claim_gate_status") != "contract_frozen":
            _block(blockers, "claim_gate_promotion", "Contract installation can only freeze claim contracts.", claim_id=row.get("claim_id"), status=row.get("claim_gate_status"))

    enums = _mapping(union.get("enums"))
    for enum_name, expected_values in EXPECTED_ENUMS.items():
        actual_values = [str(value) for value in enums.get(enum_name, [])]
        if actual_values != expected_values:
            _block(blockers, "union_enum_mismatch", "A frozen union enum differs from the contract.", enum_name=enum_name, expected=expected_values, actual=actual_values)

    policies = _mapping(union.get("policies"))
    if policies.get("public_mainline_only") is not True or policies.get("controlled_access_allowed") is not False:
        _block(blockers, "public_mainline_policy", "The public mainline must exclude controlled-access data.")
    required_exclusions = {"ADNI", "ROSMAP_controlled", "AMP_AD_controlled", "dbGaP", "UK_Biobank"}
    if not required_exclusions.issubset({str(value) for value in policies.get("excluded_controlled_sources", [])}):
        _block(blockers, "controlled_source_exclusion", "The controlled source exclusion registry is incomplete.", required=sorted(required_exclusions))
    if policies.get("calibration_source_role") != "validation" or policies.get("threshold_selection_source_role") != "validation":
        _block(blockers, "validation_only_calibration", "Calibration and threshold selection must use validation only.")
    forbidden_evidence = {str(value) for value in policies.get("full_acceptance_forbidden_evidence", [])}
    required_forbidden = {"mock", "unit_fixture", "cached_standin", "synthetic_substitute", "silent_fallback"}
    if not required_forbidden.issubset(forbidden_evidence):
        _block(blockers, "acceptance_standin_policy", "Full acceptance stand-in exclusions are incomplete.", missing=sorted(required_forbidden - forbidden_evidence))
    if policies.get("mandatory_missing_input_policy") != "fail_closed":
        _block(blockers, "silent_fallback_policy", "Missing mandatory inputs must fail closed.")
    if policies.get("checkpoint_job_classes_allowed_during_contract_installation") != []:
        _block(blockers, "unauthorized_contract_round_jobs", "The contract-installation round cannot authorize implementation or empirical jobs.")

    public_payload = {"public_identity": union.get("public_identity"), "claims": union.get("claims")}
    legacy_findings = _public_legacy_values(public_payload)
    if legacy_findings:
        _block(blockers, "public_legacy_model_name", "A public union field exposes a legacy model name.", findings=legacy_findings)
    public_identity = _mapping(union.get("public_identity"))
    if public_identity.get("external_model_name") != "NeuronSifter":
        _block(blockers, "public_model_identity", "External model identity must be NeuronSifter.", actual=public_identity.get("external_model_name"))

    validator_rows = _list_of_mappings(union.get("validator_registry"))
    validator_ids = [str(row.get("validator_id", "")) for row in validator_rows]
    for duplicate in _duplicates(validator_ids):
        _block(blockers, "duplicate_validator_id", "A validator id is declared more than once.", validator_id=duplicate)
    validator_id_set = set(validator_ids)
    for row in validator_rows:
        entrypoint = resolve_repo_path(row.get("entrypoint", ""))
        if not entrypoint.is_file():
            _block(blockers, "missing_validator_entrypoint", "A responsible validator entrypoint is missing.", validator_id=row.get("validator_id"), path=relative_repo_path(entrypoint))
        for required_field in ["validator_id", "entrypoint", "output_path", "scope"]:
            if not str(row.get(required_field, "")).strip():
                _block(blockers, "validator_metadata_missing", "A validator registry row is incomplete.", validator_id=row.get("validator_id"), field=required_field)
    consumer_rows = _list_of_mappings(union.get("consumer_registry"))
    consumer_ids = [str(row.get("consumer_id", "")) for row in consumer_rows]
    for duplicate in _duplicates(consumer_ids):
        _block(blockers, "duplicate_consumer_id", "A consumer id is declared more than once.", consumer_id=duplicate)
    consumer_id_set = set(consumer_ids)

    source_rows = _list_of_mappings(union.get("source_of_truth_registry"))
    source_domains = [str(row.get("domain_id", "")) for row in source_rows]
    for duplicate in _duplicates(source_domains):
        _block(blockers, "competing_source_of_truth", "A fact domain has more than one source of truth.", domain_id=duplicate)
    normalized_sources = [row for row in source_rows if row.get("domain_id") == "normalized_runtime_contract_fields"]
    if len(normalized_sources) != 1 or normalized_sources[0].get("canonical_source") != owner_file:
        _block(blockers, "competing_source_of_truth", "Normalized runtime fields must have exactly one union YAML source.", sources=normalized_sources)

    coverage_report = build_design_coverage_report(config)
    blockers.extend(deepcopy(coverage_report["blockers"]))
    declaration_rows = _list_of_mappings(_mapping(_mapping(config.get("validation")).get("design_contract_coverage")).get("documents"))
    declaration_paths = {str(row.get("path", "")) for row in declaration_rows}

    contract_rows = _list_of_mappings(union.get("contract_registry"))
    contract_ids = [str(row.get("contract_id", "")) for row in contract_rows]
    for duplicate in _duplicates(contract_ids):
        _block(blockers, "duplicate_canonical_ownership", "A canonical contract id has multiple owners.", contract_id=duplicate)
    contract_id_set = set(contract_ids)
    requirement_rows = _list_of_mappings(union.get("mandatory_requirements"))
    requirement_ids = [str(row.get("requirement_id", "")) for row in requirement_rows]
    for duplicate in _duplicates(requirement_ids):
        _block(blockers, "duplicate_requirement_id", "A mandatory requirement id is declared more than once.", requirement_id=duplicate)
    requirement_by_id = {str(row.get("requirement_id", "")): row for row in requirement_rows}
    active_lineage = _mapping(union.get("active_lineage"))
    roadmap_revision = str(active_lineage.get("roadmap_revision", ""))
    roadmap_dir_value = str(active_lineage.get("roadmap_dir", "")).rstrip("/")
    if roadmap_revision and roadmap_dir_value and Path(roadmap_dir_value).name != roadmap_revision:
        _block(
            blockers,
            "active_lineage_revision_directory_mismatch",
            "Active contract lineage revision must match the final roadmap directory component.",
            roadmap_revision=roadmap_revision,
            roadmap_dir=roadmap_dir_value,
        )
    roadmap_dir = resolve_repo_path(roadmap_dir_value)
    roadmap_path = roadmap_dir / "roadmap.md"
    roadmap = _roadmap_inventory(roadmap_path) if roadmap_path.is_file() else {"milestone_ids": [], "direction_owners": {}}
    milestone_ids = set(roadmap.get("milestone_ids", []))
    direction_owners = _mapping(roadmap.get("direction_owners"))
    active_ownership_pair = (
        str(active_lineage.get("milestone_id", "")),
        str(active_lineage.get("direction_id", "")),
    )
    inactive_ownership_pairs, inactive_ownership_rows = _inactive_roadmap_ownership_registry(
        union,
        direction_owners,
        active_lineage,
        blockers,
    )
    active_contract_ids: list[str] = []
    inactive_future_contract_ids: list[str] = []
    supersession_graph: dict[str, list[str]] = {}
    for row in contract_rows:
        contract_id = str(row.get("contract_id", ""))
        required_metadata = [
            "contract_id",
            "config_owner",
            "primary_design_source",
            "mandatory_requirement_ids",
            "downstream_consumer_ids",
            "validator_ids",
            "owning_milestone_id",
            "owning_direction_id",
            "downstream_implementation_status",
            "supersedes",
        ]
        for field in required_metadata:
            value = row.get(field)
            if value is None or value == "" or (isinstance(value, list) and field != "supersedes" and not value):
                _block(blockers, "contract_metadata_missing", "A canonical contract row is incomplete.", contract_id=contract_id, field=field)
        if row.get("config_owner") != owner_file:
            _block(blockers, "duplicate_canonical_ownership", "A canonical contract is assigned outside the single union owner.", contract_id=contract_id, owner=row.get("config_owner"))
        primary_source = str(row.get("primary_design_source", ""))
        if primary_source not in declaration_paths:
            _block(blockers, "contract_primary_source_missing", "A canonical contract primary source lacks a document declaration.", contract_id=contract_id, path=primary_source)
        for requirement_id in [str(value) for value in row.get("mandatory_requirement_ids", [])]:
            requirement = requirement_by_id.get(requirement_id)
            if requirement is None:
                _block(blockers, "unknown_contract_requirement", "A canonical contract references an unknown requirement.", contract_id=contract_id, requirement_id=requirement_id)
            elif str(requirement.get("owning_contract_id", "")) != contract_id:
                _block(blockers, "requirement_owner_conflict", "A requirement is listed by a non-owning contract.", contract_id=contract_id, requirement_id=requirement_id, owner=requirement.get("owning_contract_id"))
        for consumer_id in [str(value) for value in row.get("downstream_consumer_ids", [])]:
            if consumer_id not in consumer_id_set:
                _block(blockers, "unknown_contract_consumer", "A canonical contract references an unknown consumer.", contract_id=contract_id, consumer_id=consumer_id)
        for validator_id in [str(value) for value in row.get("validator_ids", [])]:
            if validator_id not in validator_id_set:
                _block(blockers, "unknown_contract_validator", "A canonical contract references an unknown validator.", contract_id=contract_id, validator_id=validator_id)
        milestone_id = str(row.get("owning_milestone_id", ""))
        direction_id = str(row.get("owning_direction_id", ""))
        ownership_pair = (milestone_id, direction_id)
        if ownership_pair == active_ownership_pair:
            active_contract_ids.append(contract_id)
        elif ownership_pair in inactive_ownership_pairs:
            inactive_future_contract_ids.append(contract_id)
        else:
            if milestone_id not in milestone_ids:
                _block(blockers, "unknown_contract_milestone", "A contract references an unknown roadmap milestone.", contract_id=contract_id, milestone_id=milestone_id)
                _block(
                    blockers,
                    "unregistered_inactive_contract_ownership",
                    "A contract outside the active roadmap is not registered as explicit inactive future ownership.",
                    contract_id=contract_id,
                    milestone_id=milestone_id,
                    direction_id=direction_id,
                )
            if direction_id not in direction_owners or direction_owners.get(direction_id) != milestone_id:
                _block(blockers, "unknown_contract_direction", "A contract references an invalid milestone/direction pair.", contract_id=contract_id, milestone_id=milestone_id, direction_id=direction_id)
            elif milestone_id in milestone_ids:
                _block(
                    blockers,
                    "unregistered_inactive_contract_ownership",
                    "A non-active roadmap contract is not registered as explicit inactive future ownership.",
                    contract_id=contract_id,
                    milestone_id=milestone_id,
                    direction_id=direction_id,
                )
        if row.get("downstream_implementation_status") != "not_started":
            _block(blockers, "implementation_scope_promotion", "Downstream implementation must remain not_started in this checkpoint.", contract_id=contract_id, status=row.get("downstream_implementation_status"))
        supersession_graph[contract_id] = [str(value) for value in row.get("supersedes", [])]
    for cycle in _find_cycles(supersession_graph):
        _block(blockers, "circular_supersession", "Canonical contract supersession contains a cycle.", cycle=cycle)

    contract_requirement_ids = {
        str(requirement_id)
        for row in contract_rows
        for requirement_id in row.get("mandatory_requirement_ids", [])
    }
    document_requirement_ids = {
        str(requirement_id)
        for row in declaration_rows
        for requirement_id in row.get("mandatory_requirement_ids", [])
    }
    for requirement in requirement_rows:
        requirement_id = str(requirement.get("requirement_id", ""))
        owner = str(requirement.get("owning_contract_id", ""))
        if owner not in contract_id_set:
            _block(blockers, "unknown_requirement_owner", "A mandatory requirement has no canonical contract owner.", requirement_id=requirement_id, owning_contract_id=owner)
        if not str(requirement.get("statement", "")).strip():
            _block(blockers, "requirement_statement_missing", "A mandatory requirement has no statement.", requirement_id=requirement_id)
        if requirement_id not in contract_requirement_ids or requirement_id not in document_requirement_ids:
            _block(blockers, "uncovered_mandatory_requirement", "A mandatory implementation or validation requirement is not covered by both contract and document declarations.", requirement_id=requirement_id, contract_covered=requirement_id in contract_requirement_ids, document_covered=requirement_id in document_requirement_ids)

    _validate_schema_families(config, union, validator_id_set, consumer_id_set, blockers)
    for design_input in _list_of_mappings(union.get("design_inputs")):
        input_path = resolve_repo_path(design_input.get("path", ""))
        if not input_path.is_file():
            _block(blockers, "missing_design_input", "A frozen design config input is missing.", path=relative_repo_path(input_path))
            continue
        actual_hash = sha256_file(input_path)
        if actual_hash != str(design_input.get("sha256", "")):
            _block(blockers, "design_input_hash_mismatch", "A frozen design config input hash changed.", path=relative_repo_path(input_path), expected=design_input.get("sha256"), actual=actual_hash)
        if design_input.get("runtime_role") != "reference-only":
            _block(blockers, "design_input_runtime_role", "Preview design inputs must remain reference-only.", path=relative_repo_path(input_path), runtime_role=design_input.get("runtime_role"))

    phase0 = _mapping(union.get("phase0_evidence"))
    for path_key, hash_key in [("repository_inventory_path", "repository_inventory_sha256"), ("claim_migration_path", "claim_migration_sha256")]:
        evidence_path = resolve_repo_path(phase0.get(path_key, ""))
        if not evidence_path.is_file():
            _block(blockers, "missing_phase0_evidence", "A required Phase 0 evidence artifact is missing.", path=relative_repo_path(evidence_path))
            continue
        actual_hash = sha256_file(evidence_path)
        if actual_hash != str(phase0.get(hash_key, "")):
            _block(blockers, "phase0_evidence_hash_mismatch", "A frozen Phase 0 evidence artifact changed.", path=relative_repo_path(evidence_path), expected=phase0.get(hash_key), actual=actual_hash)

    conflict_registry = _list_of_mappings(union.get("conflict_registry"))
    conflict_ids = [str(row.get("conflict_id", "")) for row in conflict_registry]
    for duplicate in _duplicates(conflict_ids):
        _block(blockers, "duplicate_conflict_id", "A conflict id is declared more than once.", conflict_id=duplicate)
    pending_conflicts = [
        {"conflict_id": row.get("conflict_id"), "resolution_class": row.get("resolution_class"), "rule": row.get("rule")}
        for row in conflict_registry
        if "pending" in str(row.get("resolution_class", ""))
    ]
    resolved_conflicts = [
        {"conflict_id": row.get("conflict_id"), "resolution_class": row.get("resolution_class"), "rule": row.get("rule")}
        for row in conflict_registry
        if "pending" not in str(row.get("resolution_class", ""))
    ]

    if provenance is None:
        _, provenance = load_config_with_provenance(config_path)
    graph_report, isolation_report, remediation_report = validate_public_mainline_isolation_payload(config, provenance, config_path)
    blockers.extend(deepcopy(isolation_report["blockers"]))
    config_report = validate_config_stack_payload(config, provenance, config_path)
    blockers.extend(deepcopy(config_report["blockers"]))
    state_report = build_orchestrator_state_report(config)
    blockers.extend(deepcopy(state_report["blockers"]))
    deduplicated_blockers: list[dict[str, object]] = []
    seen_blockers: set[str] = set()
    for blocker in blockers:
        key = json.dumps(blocker, sort_keys=True, ensure_ascii=True)
        if key not in seen_blockers:
            deduplicated_blockers.append(blocker)
            seen_blockers.add(key)
    blockers = deduplicated_blockers

    config_file = resolve_repo_path(config_path)
    claim_migration_path = resolve_repo_path(phase0.get("claim_migration_path", ""))
    bound_audits = {
        "config_stack": {
            "schema_version": config_report["schema_version"],
            "status": config_report["status"],
            "output_path": _mapping(audit_contract.get("output_paths")).get("config_stack"),
            "report_sha256": sha256_payload(config_report),
            "blocker_count": config_report["blocker_count"],
        },
        "design_coverage": {
            "schema_version": coverage_report["schema_version"],
            "status": coverage_report["status"],
            "output_path": _mapping(audit_contract.get("output_paths")).get("design_coverage"),
            "report_sha256": sha256_payload(coverage_report),
            "blocker_count": coverage_report["blocker_count"],
        },
        "orchestrator_state": {
            "schema_version": state_report["schema_version"],
            "status": state_report["status"],
            "output_path": _mapping(audit_contract.get("output_paths")).get("orchestrator_state"),
            "report_sha256": sha256_payload(state_report),
            "blocker_count": state_report["blocker_count"],
        },
        "public_mainline_isolation": {
            "schema_version": isolation_report["schema_version"],
            "status": isolation_report["status"],
            "output_path": _mapping(audit_contract.get("output_paths")).get("public_mainline_isolation"),
            "config_graph_schema_version": graph_report["schema_version"],
            "config_graph_output_path": _mapping(audit_contract.get("output_paths")).get("public_mainline_config_graph"),
            "remediation_schema_version": remediation_report["schema_version"],
            "remediation_output_path": _mapping(audit_contract.get("output_paths")).get("g0_public_mainline_remediation"),
            "controlled_reachability_count": isolation_report["controlled_reachability_count"],
            "report_sha256": sha256_payload(isolation_report),
            "blocker_count": isolation_report["blocker_count"],
        },
    }
    status = str(audit_contract.get("pass_status", "pass")) if not blockers else "blocked"
    return {
        "schema_version": str(_mapping(audit_contract.get("schema_versions")).get("contract_union", "")),
        "status": status,
        "contract_version": union.get("contract_version"),
        "config_path": relative_repo_path(config_file),
        "config_sha256": sha256_file(config_file),
        "claim_migration_path": relative_repo_path(claim_migration_path),
        "claim_migration_sha256": sha256_file(claim_migration_path) if claim_migration_path.is_file() else None,
        "claim_count": len(claims),
        "contract_count": len(contract_rows),
        "mandatory_requirement_count": len(requirement_rows),
        "schema_family_count": len(_list_of_mappings(union.get("schema_families"))),
        "document_count": coverage_report["declared_document_count"],
        "resolved_conflicts": resolved_conflicts,
        "pending_conflicts": pending_conflicts,
        "pending_work": deepcopy(_mapping(union.get("pending_work"))),
        "contract_ownership": {
            "active_roadmap_contract_ids": sorted(active_contract_ids),
            "inactive_future_contract_ids": sorted(inactive_future_contract_ids),
            "inactive_future_ownership_registry": inactive_ownership_rows,
        },
        "bound_audits": bound_audits,
        "blockers": blockers,
        "blocker_count": len(blockers),
    }


def load_union_context(config_path: str | Path) -> tuple[dict[str, object], dict[str, object]]:
    config, provenance = load_config_with_provenance(config_path)
    return dict(config), dict(provenance)
