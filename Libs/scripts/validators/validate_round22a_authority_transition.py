"""Validate Round-22A Tier A/B resolution and Tier C eligibility state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_round22_agent_twin_coevolution_readiness import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    canonical_sha256,
    load_contract,
)
from Libs.scripts.builders.build_round22a_authority_recovery import (  # noqa: E402
    REQUIRED_AUTHORITY_INPUTS,
)
from Libs.scripts.builders.build_round22a_authority_transition import (  # noqa: E402
    OUTPUT,
    PREDECESSOR_OUTPUT,
    ROUND20Q_COMPONENTS,
    TRANSITION_SCHEMA,
    _commit_is_at_or_before_cutover,
)
from Libs.scripts.builders.build_round22r_empirical_prerequisites import C2B_BLOCKED  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _block(blockers: list[dict[str, str]], code: str, message: str) -> None:
    blockers.append({"code": code, "message": message})


def validate(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    contract = load_contract(config_path, schema_path)
    recovery = contract["freeze"]["authority_recovery"]
    try:
        report = _read_json(root / OUTPUT)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": "neuronsifter-round22a-authority-transition-validation-v3",
            "status": "failed",
            "passed": False,
            "blockers": [
                {"code": "transition_artifact_read_failure", "message": str(exc)}
            ],
        }
    if report.get("schema_version") != TRANSITION_SCHEMA:
        _block(
            blockers,
            "transition_schema_drift",
            "Authority recovery resolution schema is invalid.",
        )
    without_uid = dict(report)
    artifact_uid = without_uid.pop("artifact_uid", None)
    if artifact_uid != canonical_sha256(without_uid):
        _block(
            blockers,
            "transition_artifact_uid_mismatch",
            "Transition artifact UID does not bind its payload.",
        )
    configured_cutover = recovery.get("recovery_cutover_git_identity")
    report_cutover = report.get("recovery_cutover_git_identity")
    if not isinstance(configured_cutover, Mapping) or report_cutover != dict(
        configured_cutover
    ):
        _block(
            blockers,
            "recovery_cutover_identity_mismatch",
            "The transition does not bind the configured recovery cutover identity.",
        )
    implementation_bindings = report.get("implementation_bindings")
    if (
        not isinstance(implementation_bindings, list)
        or len(implementation_bindings) != 3
    ):
        _block(
            blockers,
            "implementation_bindings_missing",
            "The hardened transition implementation bindings are missing.",
        )
    else:
        observed_paths: set[str] = set()
        for binding in implementation_bindings:
            if not isinstance(binding, Mapping):
                _block(
                    blockers,
                    "implementation_binding_invalid",
                    "A transition implementation binding is not an object.",
                )
                continue
            relative_path = str(binding.get("path", ""))
            observed_paths.add(relative_path)
            path = REPO_ROOT / relative_path
            observed_sha256 = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file() and not path.is_symlink()
                else None
            )
            if observed_sha256 != binding.get("sha256") or (
                path.stat().st_size if path.is_file() else None
            ) != binding.get("bytes"):
                _block(
                    blockers,
                    "implementation_binding_hash_mismatch",
                    f"Transition implementation binding drifted: {relative_path}",
                )
        expected_paths = {
            "Libs/scripts/builders/build_round22a_authority_transition.py",
            "Libs/scripts/validators/validate_round22a_authority_transition.py",
            "Libs/tests/test_round22a_authority_recovery.py",
        }
        if observed_paths != expected_paths:
            _block(
                blockers,
                "implementation_binding_path_mismatch",
                "Transition implementation binding membership drifted.",
            )
    for artifact_key in (
        "wp1_search_artifact",
        "wp1_inventory_artifact",
        "scheduler_workdir_reconciliation_artifact",
    ):
        binding = report.get(artifact_key)
        if not isinstance(binding, Mapping):
            _block(blockers, "wp1_binding_missing", f"{artifact_key} is missing.")
            continue
        path = root / str(binding.get("path", ""))
        observed = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if observed != binding.get("sha256"):
            _block(blockers, "wp1_binding_hash_mismatch", f"{artifact_key} drifted.")
    scheduler_binding = report.get("scheduler_workdir_reconciliation_artifact")
    if (
        not isinstance(scheduler_binding, Mapping)
        or scheduler_binding.get("complete") is not True
    ):
        _block(
            blockers,
            "scheduler_reconciliation_not_bound",
            "The current recovery generation does not bind complete scheduler evidence.",
        )
    predecessor = report.get("predecessor_lineage")
    if not isinstance(predecessor, Mapping):
        _block(
            blockers,
            "predecessor_lineage_binding_missing",
            "The immutable predecessor lineage binding is missing.",
        )
    else:
        predecessor_path = root / str(predecessor.get("path", ""))
        if predecessor.get("path") != PREDECESSOR_OUTPUT:
            _block(
                blockers,
                "predecessor_lineage_path_drift",
                "The predecessor lineage path drifted.",
            )
        observed = (
            hashlib.sha256(predecessor_path.read_bytes()).hexdigest()
            if predecessor_path.is_file()
            else None
        )
        if predecessor.get(
            "exists"
        ) != predecessor_path.is_file() or observed != predecessor.get("sha256"):
            _block(
                blockers,
                "predecessor_lineage_hash_mismatch",
                "The immutable predecessor lineage binding drifted.",
            )
    tier_a = report.get("tier_a")
    tier_b = report.get("tier_b")
    tier_c = report.get("tier_c")
    if not isinstance(tier_a, Mapping):
        _block(
            blockers,
            "tier_a_state_missing",
            "Tier A resolution state is missing.",
        )
    else:
        if tier_a.get("forensic_scope_complete") is not True:
            _block(
                blockers,
                "tier_a_forensic_scope_incomplete",
                "Tier A does not bind complete registered-root and scheduler evidence.",
            )
        archive_validation = tier_a.get("archive_index_validation")
        configured_indexes = recovery.get("tier_a_archive_indexes", [])
        if (
            not isinstance(archive_validation, Mapping)
            or archive_validation.get("registered_count") != len(configured_indexes)
            or not isinstance(archive_validation.get("results"), list)
            or len(archive_validation.get("results", [])) != len(configured_indexes)
        ):
            _block(
                blockers,
                "tier_a_archive_index_state_invalid",
                "Tier A archive-index adjudication does not match the frozen registry.",
            )
        cutover_commit = (
            str(report_cutover.get("commit", ""))
            if isinstance(report_cutover, Mapping)
            else ""
        )
        common_commits = tier_a.get("common_storage_commits")
        if not isinstance(common_commits, list) or any(
            not _commit_is_at_or_before_cutover(root, commit, cutover_commit)
            for commit in common_commits
        ):
            _block(
                blockers,
                "tier_a_common_commit_after_cutover",
                "Tier A retained a storage commit after the recovery cutover.",
            )
        ready = tier_a.get("ready_for_byte_identical_recovery") is True
        failed = tier_a.get("failed") is True
        if ready and (
            failed
            or tier_a.get("coherent_identity_count") != 1
            or not any(
                isinstance(row, Mapping) and row.get("validated") is True
                for row in tier_a.get("candidate_lineages", [])
            )
        ):
            _block(
                blockers,
                "tier_a_positive_path_invalid",
                "Tier A recovery readiness is not backed by one coherent lineage.",
            )
        if failed and (
            ready
            or tier_a.get("common_storage_commit_count") != 0
            or tier_a.get("inventory_authority_candidate_count") != 0
            or tier_a.get("scheduler_supplemental_candidate_count") != 0
        ):
            _block(
                blockers,
                "tier_a_failure_not_proven",
                "Tier A failure is inconsistent with discovered candidates.",
            )
    per_authority = tier_b.get("per_authority") if isinstance(tier_b, Mapping) else None
    if not isinstance(per_authority, list):
        _block(
            blockers,
            "tier_b_per_authority_missing",
            "Tier B per-authority evidence is missing.",
        )
    else:
        ids = [
            str(row.get("authority_uid", ""))
            for row in per_authority
            if isinstance(row, Mapping)
        ]
        if set(ids) != set(REQUIRED_AUTHORITY_INPUTS) or len(ids) != len(set(ids)):
            _block(
                blockers,
                "tier_b_authority_identity_mismatch",
                "Tier B must adjudicate each canonical authority exactly once.",
            )
        for row in per_authority:
            if not isinstance(row, Mapping):
                _block(
                    blockers,
                    "tier_b_row_invalid",
                    "Tier B evidence row is not an object.",
                )
                continue
            manifest_validation = row.get("tier_b_manifest_validation")
            manifest_valid = isinstance(manifest_validation, Mapping) and (
                manifest_validation.get("validated") is True
            )
            if row.get("tier_b_reconstruction_ready") is not manifest_valid:
                _block(
                    blockers,
                    "tier_b_manifest_decision_mismatch",
                    f"Tier B manifest decision drifted for {row.get('authority_uid')}.",
                )
            if isinstance(manifest_validation, Mapping) and (
                manifest_validation.get("semantic_identity_proven") is not False
                or manifest_validation.get("deterministic_reconstruction_executed")
                is not False
                or manifest_validation.get("reconstruction_inputs_ready")
                is not manifest_valid
            ):
                _block(
                    blockers,
                    "tier_b_preexecution_claim_invalid",
                    f"Tier B pre-execution state is invalid for {row.get('authority_uid')}.",
                )
    registered_manifests = contract["freeze"]["authority_recovery"].get(
        "tier_b_reconstruction_manifests", []
    )
    tier_b_complete = bool(
        isinstance(per_authority, list)
        and per_authority
        and all(
            isinstance(row, Mapping) and row.get("tier_b_reconstruction_ready") is True
            for row in per_authority
        )
    )
    if not isinstance(tier_b, Mapping) or (
        tier_b.get("failed") != (not tier_b_complete)
        or tier_b.get("all_targets_execution_ready") is not tier_b_complete
        or tier_b.get("registered_manifest_count") != len(registered_manifests)
        or tier_b.get("deterministic_reconstruction_started") is not False
    ):
        _block(
            blockers,
            "tier_b_state_invalid",
            "Tier B aggregate state is inconsistent with its verified manifests.",
        )
    if not isinstance(tier_c, Mapping):
        _block(blockers, "tier_c_state_missing", "Tier C state is missing.")
    else:
        machine = tier_c.get("round20q_machine_evidence")
        components = tier_c.get("round20q_required_components")
        contract_validation = tier_c.get("round20q_contract_validation")
        configured_round20q = contract["freeze"]["authority_recovery"].get(
            "round20q_replay_prerequisite"
        )
        registered = isinstance(configured_round20q, Mapping)
        validated = bool(
            isinstance(contract_validation, Mapping)
            and contract_validation.get("fully_validated") is True
        )
        if tier_c.get("round20q_contract_registered") != registered:
            _block(
                blockers,
                "round20q_registry_state_invalid",
                "Round-20Q registry state does not match the frozen config.",
            )
        if validated and (
            not isinstance(contract_validation, Mapping)
            or not _commit_is_at_or_before_cutover(
                root,
                contract_validation.get("manifest_storage_git_commit"),
                str(report_cutover.get("commit", ""))
                if isinstance(report_cutover, Mapping)
                else "",
            )
        ):
            _block(
                blockers,
                "round20q_manifest_after_cutover",
                "A validated Round-20Q root must be stored before the recovery cutover.",
            )
        if not isinstance(machine, Mapping) or machine.get("bounded") is not True:
            _block(
                blockers,
                "round20q_machine_search_invalid",
                "Round-20Q machine evidence search is not bounded.",
            )
        if (
            not isinstance(components, Mapping)
            or set(components) != set(ROUND20Q_COMPONENTS)
            or any(not isinstance(value, bool) for value in components.values())
            or (validated and not all(components.values()))
        ):
            _block(
                blockers,
                "round20q_component_state_invalid",
                "Round-20Q component validation state is inconsistent.",
            )
        expected_eligible = bool(
            isinstance(tier_a, Mapping)
            and tier_a.get("failed") is True
            and isinstance(tier_b, Mapping)
            and tier_b.get("failed") is True
            and validated
        )
        if (
            tier_c.get("round20q_fully_validated") != validated
            or tier_c.get("eligible") != expected_eligible
            or tier_c.get("round21r_replay_started") is not False
            or tier_c.get("gpu_used") is not False
        ):
            _block(
                blockers,
                "tier_c_state_invalid",
                "Tier C eligibility or execution state is inconsistent.",
            )
    authority = report.get("round21_authority")
    static_twin = report.get("static_twin")
    if authority != {"required": 9, "present": 0, "validated": 0, "hash_bound": 0}:
        _block(
            blockers,
            "authority_state_promoted",
            "Round-21 authority state changed during transition assessment.",
        )
    if static_twin != {
        "required_components": 10,
        "present": 0,
        "validated": 0,
        "frozen": False,
    }:
        _block(
            blockers,
            "static_twin_state_promoted",
            "Static Twin state changed during transition assessment.",
        )
    for key in (
        "exact_clean_full_suite_authorized",
        "c3_c4_execution_allowed",
        "coevolution_execution_allowed",
        "canonical_round21_authority_written",
        "round22a_authority_root_written",
    ):
        if report.get(key) is not False:
            _block(
                blockers,
                f"transition_{key}_invalid",
                f"Transition invariant {key}=false was violated.",
            )
    if report.get("c2_b_status") != C2B_BLOCKED:
        _block(
            blockers, "c2_b_status_drift", "C2-B changed during transition assessment."
        )
    if any((root / path).is_file() for path in REQUIRED_AUTHORITY_INPUTS.values()):
        _block(
            blockers,
            "canonical_authority_unexpected",
            "A canonical Round-21 authority appeared during failed recovery.",
        )
    if (root / "Output/audits/round22a_round21_authority_root.json").exists():
        _block(
            blockers,
            "authority_root_unexpected",
            "An unresolved authority root was written.",
        )
    expected_status: str | None = None
    expected_next_action: str | None = None
    if (
        isinstance(tier_a, Mapping)
        and tier_a.get("ready_for_byte_identical_recovery") is True
    ):
        expected_status = "tier_a_coherent_lineage_ready_not_recovered"
        expected_next_action = "recover_validated_byte_identical_tier_a_lineage"
    elif isinstance(tier_a, Mapping) and tier_a.get("failed") is not True:
        expected_status = "tier_a_candidates_require_adjudication"
        expected_next_action = "adjudicate_unresolved_tier_a_candidates"
    elif isinstance(tier_b, Mapping) and tier_b.get("failed") is not True:
        expected_status = "tier_b_reconstruction_inputs_complete_not_executed"
        expected_next_action = "execute_validated_deterministic_tier_b_reconstruction"
    elif isinstance(tier_c, Mapping) and tier_c.get("eligible") is True:
        expected_status = "tier_c_eligible_not_started"
        expected_next_action = (
            "execute_round21r_replay_from_validated_round20q_contract"
        )
    else:
        expected_status = "blocked_round20q_replay_prerequisite"
        expected_next_action = "register_and_validate_complete_round20q_frozen_contract_before_round21r_replay"
    if (
        report.get("status") != expected_status
        or report.get("next_action") != expected_next_action
    ):
        _block(
            blockers,
            "transition_route_state_mismatch",
            "Transition status or next action does not match the verified Tier state.",
        )
    return {
        "schema_version": "neuronsifter-round22a-authority-transition-validation-v3",
        "status": "passed" if not blockers else "failed",
        "passed": not blockers,
        "terminal_transition_status": report.get("status"),
        "tier_a_failed": isinstance(tier_a, Mapping) and tier_a.get("failed") is True,
        "tier_a_ready": isinstance(tier_a, Mapping)
        and tier_a.get("ready_for_byte_identical_recovery") is True,
        "tier_b_failed": isinstance(tier_b, Mapping) and tier_b.get("failed") is True,
        "tier_c_eligible": isinstance(tier_c, Mapping)
        and tier_c.get("eligible") is True,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--root", default=str(REPO_ROOT))
    args = parser.parse_args()
    result = validate(args.config, args.schema, Path(args.root))
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
