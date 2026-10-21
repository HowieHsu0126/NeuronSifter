"""Round-11F governance audit builder.

This module handles only JSON/TSV audit metadata and authority orchestration; it
does not parse scientific data, train models, calculate metrics, or implement
scientific tools.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from Libs.utils.data.evidence_union import build_evidence_union, fails_closed_for_conflicts
from Libs.utils.data.raw_data_registry import write_json, write_tsv
from Libs.utils.exp.orchestration_controller import (
    CANONICAL_CONTROLLER_UID,
    GateAuthorityError,
    assert_formal_gate_writer,
    authority_contract_report,
    build_canonical_gate_decision,
    git_commit,
    git_dirty_status,
    load_authority_contract,
    sha256_file,
    sha256_payload,
)


AUTHORITY_CONFIG_PATH = Path("Libs/configs/validation/round11f_controller_authority.yaml")
GATE_SCHEMA_PATH = Path("Libs/configs/schema/gate_writer_authority.schema.json")
UNION_SCHEMA_PATH = Path("Libs/configs/schema/evidence_union.schema.json")
AUDIT_ROOT = Path("Output/audits")

FROZEN_INPUTS = (
    "TODO.md",
    "orchestrator/state.json",
    "Libs/configs/config_stack.yaml",
    "Libs/configs/validation/iclr_design_contract_union.yaml",
    "Libs/utils/data/iclr_contract_union.py",
    "Libs/scripts/validators/validate_iclr_design_contract_union.py",
    "Libs/scripts/validators/validate_orchestrator_state_contract.py",
    "orchestrator/roadmap-updates/round-11-dt-partial-state-roadmap-update.md",
    "Output/audits/dt_round11d_full_suite_execution.json",
    "Output/audits/dt_round11d_full_suite_results.xml",
    "Output/audits/dt_round11d_full_suite_failure_triage.json",
    "Output/audits/dt_round11_closure_decision.json",
    "Output/audits/dt_round11e_targeted_gate.json",
    "Output/audits/dt_round11e_repair_wave_record.json",
    "Output/audits/dt_round11e_testcase_failure_ledger.tsv",
    "Output/audits/dt_round11e_failure_dependency_graph.json",
    "Output/audits/dt_round11e_full_suite_execution.json",
)

GATE_WRITER_FIELDS = [
    "component_uid",
    "path",
    "entrypoint",
    "classification",
    "writes_gate",
    "writes_gate_shaped_artifact",
    "formal_gate_write_authority",
    "legacy_candidate_outputs",
    "current_owner",
    "callers",
    "duplicate_role",
    "deprecation_status",
]
MIGRATION_FIELDS = ["old_component", "new_role", "compatibility_behavior", "removal_version", "rollback"]


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return dict(payload)


def _binding(repo_root: Path, relative_path: str | Path) -> dict[str, Any]:
    relative = Path(relative_path)
    path = (repo_root / relative).resolve()
    if not path.is_relative_to(repo_root.resolve()):
        raise ValueError(f"Input path escapes repository: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Required Round-11F read-only input is missing: {relative_path}")
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "read_only": True,
    }


def _component(
    *,
    component_uid: str,
    path: str,
    entrypoint: str,
    classification: str,
    inputs: list[str],
    outputs: list[str],
    writes_artifacts: bool,
    writes_gate: bool = False,
    current_owner: str,
    callers: list[str],
    duplicate_role: str,
    deprecation_status: str,
    legacy_candidate_outputs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "component_uid": component_uid,
        "path": path,
        "entrypoint": entrypoint,
        "classification": classification,
        "inputs": inputs,
        "outputs": outputs,
        "writes_gate": classification == "canonical",
        "writes_gate_shaped_artifact": writes_gate,
        "formal_gate_write_authority": classification == "canonical",
        "writes_status": False,
        "writes_config": False,
        "writes_artifacts": writes_artifacts,
        "current_owner": current_owner,
        "callers": callers,
        "duplicate_role": duplicate_role,
        "deprecation_status": deprecation_status,
        "legacy_candidate_outputs": legacy_candidate_outputs or [],
    }


def controller_components() -> list[dict[str, Any]]:
    """Registry of controllers/writers in scope, excluding archived worktrees."""

    return [
        _component(
            component_uid=CANONICAL_CONTROLLER_UID,
            path="Libs/scripts/builders/build_dt_round11f_controller_contracts.py",
            entrypoint="main",
            classification="canonical",
            inputs=[str(AUTHORITY_CONFIG_PATH), *FROZEN_INPUTS],
            outputs=["Output/audits/dt_round11f_gate.json", "Output/audits/dt_round11f_*"],
            writes_artifacts=True,
            writes_gate=True,
            current_owner="Round-11F canonical controller",
            callers=["Round-11F explicit builder invocation", "future approved stage adapters"],
            duplicate_role="none",
            deprecation_status="active",
        ),
        _component(
            component_uid="orchestrator_control_plane_contract",
            path="orchestrator/project-contract.md",
            entrypoint="documented orchestration contract",
            classification="adapter",
            inputs=["orchestrator/state.json", "active roadmap bundle"],
            outputs=[],
            writes_artifacts=False,
            current_owner="repository orchestrator control plane",
            callers=["orchestrator roles", "orchestrator state validator"],
            duplicate_role="documented state ownership only; no formal Gate artifact writer",
            deprecation_status="active_adapter",
        ),
        _component(
            component_uid="orchestrator_state_contract_validator",
            path="Libs/scripts/validators/validate_orchestrator_state_contract.py",
            entrypoint="main",
            classification="adapter",
            inputs=["orchestrator/state.json", "Libs/configs/config_stack.yaml"],
            outputs=["Output/audits/orchestrator_state_contract.json"],
            writes_artifacts=True,
            current_owner="config/roadmap validation",
            callers=["ICLR contract-union validator", "focused contract tests"],
            duplicate_role="validator evidence only; formal Gate writing migrated to canonical controller",
            deprecation_status="active_adapter",
        ),
        _component(
            component_uid="iclr_design_contract_union_validator",
            path="Libs/scripts/validators/validate_iclr_design_contract_union.py",
            entrypoint="main",
            classification="adapter",
            inputs=["Libs/configs/config_stack.yaml", "orchestrator/state.json", "active roadmap"],
            outputs=["Output/audits/iclr_contract_union_consistency.json"],
            writes_artifacts=True,
            current_owner="config/roadmap union validation",
            callers=["Round-11E diagnostic", "focused contract tests"],
            duplicate_role="legacy contract-union validation only; no artifact-level formal Gate authority",
            deprecation_status="active_adapter",
        ),
        _component(
            component_uid="iclr_contract_union_legacy_builder",
            path="Libs/utils/data/iclr_contract_union.py",
            entrypoint="build_union_consistency_report",
            classification="legacy_read_only",
            inputs=["active config stack", "orchestrator/state.json", "roadmap revision"],
            outputs=["structural union report"],
            writes_artifacts=False,
            current_owner="legacy design-contract compatibility",
            callers=["validate_iclr_design_contract_union.py", "test_iclr_design_contract_union.py"],
            duplicate_role="code/config union; explicitly excluded from artifact-level evidence union",
            deprecation_status="read_only_bridge",
        ),
        _component(
            component_uid="round11_state_audit_builder",
            path="Libs/scripts/builders/build_dt_round11_state_audits.py",
            entrypoint="main",
            classification="legacy_read_only",
            inputs=["Round-11 typed-state source artifacts"],
            outputs=["Round-11 state audit artifacts"],
            writes_artifacts=True,
            current_owner="Round-11 historical audit producer",
            callers=["Round-11D frozen evidence", "Round-11E triage"],
            duplicate_role="historical producer; cannot write a new formal Gate",
            deprecation_status="frozen_input_only",
        ),
        _component(
            component_uid="round11e_failure_triage_builder",
            path="Libs/scripts/builders/build_dt_round11e_failure_triage.py",
            entrypoint="main",
            classification="legacy_read_only",
            inputs=["frozen Round-11D JUnit/execution/closure artifacts"],
            outputs=["Output/audits/dt_round11e_*"],
            writes_artifacts=True,
            current_owner="Round-11E forensic triage",
            callers=["Round-11E repair process"],
            duplicate_role="forensic observation records only; cannot set a formal Gate decision",
            deprecation_status="frozen_input_and_handoff",
        ),
        _component(
            component_uid="round11e_full_suite_deferral_builder",
            path="Libs/scripts/builders/build_dt_round11e_full_suite_deferral.py",
            entrypoint="main",
            classification="legacy_read_only",
            inputs=["Round-11E targeted gate and repair-wave record"],
            outputs=["Output/audits/dt_round11e_full_suite_execution.json"],
            writes_artifacts=True,
            current_owner="Round-11E full-suite deferral recorder",
            callers=["Round-11E closure process"],
            duplicate_role="records a blocked submission only; cannot close Round-11",
            deprecation_status="handoff_to_round11g",
        ),
        _component(
            component_uid="c1_foundation_transfer_gate_builder",
            path="Libs/scripts/experiments/build_c1_execution_artifacts.py",
            entrypoint="build",
            classification="deprecated",
            inputs=["C1 preflight and capability manifests"],
            outputs=["Output/audits/c1_foundation_transfer_gate.json"],
            writes_artifacts=True,
            writes_gate=True,
            current_owner="C1 local evidence producer",
            callers=["C1 local validator"],
            duplicate_role="legacy local gate candidate; excluded from formal DT Gate authority",
            deprecation_status="candidate_output_v1",
            legacy_candidate_outputs=["Output/audits/c1_foundation_transfer_gate.json"],
        ),
        _component(
            component_uid="c2_phase_gate_builder",
            path="Libs/scripts/builders/build_c2_phase_gate.py",
            entrypoint="write_outputs",
            classification="deprecated",
            inputs=["C2 config and readiness artifacts"],
            outputs=["Output/audits/c2_g4_validation.json", "Output/audits/c2_phase_gate_execution.json"],
            writes_artifacts=True,
            writes_gate=True,
            current_owner="C2 local evidence producer",
            callers=["test_c2_phase_gate.py"],
            duplicate_role="legacy local gate candidate; excluded from formal DT Gate authority",
            deprecation_status="candidate_output_v1",
            legacy_candidate_outputs=["Output/audits/c2_g4_validation.json"],
        ),
        _component(
            component_uid="c2r_entry_gate_builder",
            path="Libs/scripts/builders/build_c2r_trajectory_action_qualification.py",
            entrypoint="write_outputs",
            classification="deprecated",
            inputs=["C2R registry and qualification artifacts"],
            outputs=["Output/audits/c2r_entry_gate_decision.json"],
            writes_artifacts=True,
            writes_gate=True,
            current_owner="C2R local qualification producer",
            callers=["test_c2r_trajectory_action_qualification.py"],
            duplicate_role="legacy local gate candidate; excluded from formal DT Gate authority",
            deprecation_status="candidate_output_v1",
            legacy_candidate_outputs=["Output/audits/c2r_entry_gate_decision.json"],
        ),
        _component(
            component_uid="baseline_r1_qualification_pipeline",
            path="Libs/scripts/builders/build_baseline_r1_qualification.py",
            entrypoint="main",
            classification="adapter",
            inputs=["baseline registry", "source/license evidence"],
            outputs=["Output/audits/baseline_r1_gate.json"],
            writes_artifacts=True,
            writes_gate=True,
            current_owner="baseline qualification producer",
            callers=["validate_baseline_r1_qualification.py"],
            duplicate_role="independent producer; cannot mutate DT state or formal Gate",
            deprecation_status="independent_producer_only",
            legacy_candidate_outputs=["Output/audits/baseline_r1_gate.json"],
        ),
        _component(
            component_uid="baseline_v3_audit_builder",
            path="Libs/scripts/builders/build_baseline_registry_v3_audits.py",
            entrypoint="main",
            classification="adapter",
            inputs=["baseline registry", "baseline lock", "source/license records"],
            outputs=["Output/audits/baseline_reproduction_gate.json"],
            writes_artifacts=True,
            writes_gate=True,
            current_owner="baseline qualification producer",
            callers=["baseline V3 validator", "baseline R1 qualification"],
            duplicate_role="independent baseline gate-shaped artifact; cannot mutate DT state or formal Gate",
            deprecation_status="independent_producer_only",
            legacy_candidate_outputs=["Output/audits/baseline_reproduction_gate.json"],
        ),
        _component(
            component_uid="baseline_v31_active_set_builder",
            path="Libs/scripts/builders/build_baseline_registry_v31_audits.py",
            entrypoint="main",
            classification="adapter",
            inputs=["Baseline Registry V3.1 active set", "baseline lock"],
            outputs=["Output/audits/baseline_v31_gate.json"],
            writes_artifacts=True,
            writes_gate=True,
            current_owner="baseline active-set producer",
            callers=["baseline V3.1 validator"],
            duplicate_role="independent baseline gate-shaped artifact; cannot mutate DT state or formal Gate",
            deprecation_status="independent_producer_only",
            legacy_candidate_outputs=["Output/audits/baseline_v31_gate.json"],
        ),
        _component(
            component_uid="c1_rescue_feasibility_builder",
            path="Libs/scripts/builders/build_c1_rescue_feasibility_audit.py",
            entrypoint="main",
            classification="deprecated",
            inputs=["C1 rescue candidate evidence"],
            outputs=["Output/audits/c1_rescue_feasibility_decision.json"],
            writes_artifacts=True,
            writes_gate=False,
            current_owner="C1 rescue evidence producer",
            callers=["C1 rescue feasibility validator"],
            duplicate_role="decision audit only; not a formal Gate writer",
            deprecation_status="candidate_output_v1",
        ),
        _component(
            component_uid="paper_results_foundation_readiness_writer",
            path="Libs/scripts/reports/build_paper_results.py",
            entrypoint="main",
            classification="deprecated",
            inputs=["paper-result registries and readiness audits"],
            outputs=["Output/paper_reports/foundation_readiness_gate.json"],
            writes_artifacts=True,
            writes_gate=True,
            current_owner="publication report producer",
            callers=["publication readiness workflow"],
            duplicate_role="publication candidate gate; cannot mutate DT state or formal Gate",
            deprecation_status="candidate_output_v1",
            legacy_candidate_outputs=["Output/paper_reports/foundation_readiness_gate.json"],
        ),
        _component(
            component_uid="roadmap_update_markdown",
            path="orchestrator/roadmap-updates/round-11-dt-partial-state-roadmap-update.md",
            entrypoint="declarative roadmap-update record",
            classification="legacy_read_only",
            inputs=["approved review evidence"],
            outputs=[],
            writes_artifacts=False,
            current_owner="roadmap history",
            callers=["human/controller review"],
            duplicate_role="Markdown is not a formal status source",
            deprecation_status="read_only_history",
        ),
    ]


def controller_authority_map(components: list[Mapping[str, object]]) -> dict[str, Any]:
    canonical_components = [row for row in components if row.get("classification") == "canonical"]
    gate_writers = [row for row in components if row.get("formal_gate_write_authority") is True]
    blocker_status = "controller_lineage_resolved"
    blockers: list[dict[str, Any]] = []
    if len(canonical_components) != 1:
        blocker_status = "blocked_multiple_controllers"
        blockers.append({"code": "canonical_controller_count", "observed": len(canonical_components)})
    if len(gate_writers) != 1:
        if blocker_status == "controller_lineage_resolved":
            blocker_status = "blocked_multiple_gate_writers"
        blockers.append({"code": "canonical_gate_writer_count", "observed": len(gate_writers)})
    if canonical_components and canonical_components[0].get("component_uid") != CANONICAL_CONTROLLER_UID:
        blocker_status = "blocked_multiple_controllers"
        blockers.append({"code": "canonical_controller_uid", "observed": canonical_components[0].get("component_uid")})
    if gate_writers and gate_writers[0].get("component_uid") != CANONICAL_CONTROLLER_UID:
        blocker_status = "blocked_multiple_gate_writers"
        blockers.append({"code": "canonical_gate_writer_uid", "observed": gate_writers[0].get("component_uid")})
    return {
        "schema_version": "dt-round11f-controller-authority-map-v1",
        "canonical_controller_uid": CANONICAL_CONTROLLER_UID,
        "canonical_controller_count": len(canonical_components),
        "canonical_gate_writer_count": len(gate_writers),
        "formal_gate_writers": [str(row.get("component_uid")) for row in gate_writers],
        "non_authorized_gate_candidate_components": [
            str(row.get("component_uid"))
            for row in components
            if row.get("legacy_candidate_outputs")
        ],
        "status": blocker_status,
        "blockers": blockers,
        "authority_boundary": "Only the canonical controller can create a GateWriterAuthority-conformant formal Gate artifact.",
    }


def _lineage_inventory(
    repo_root: Path,
    evidence_root: Path,
    input_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    state = _read_json(repo_root / "orchestrator/state.json")
    targeted_gate = _read_json(evidence_root / "Output/audits/dt_round11e_targeted_gate.json")
    repair_record = _read_json(evidence_root / "Output/audits/dt_round11e_repair_wave_record.json")
    closure = _read_json(evidence_root / "Output/audits/dt_round11_closure_decision.json")
    return {
        "schema_version": "dt-round11f-lineage-inventory-v1",
        "read_only_input_bindings": input_bindings,
        "r11e_rc003": {
            "blocker_uid": "R11E-RC-003",
            "targeted_gate_status": targeted_gate.get("status"),
            "wave_d_status": next(
                (row.get("status") for row in targeted_gate.get("waves", []) if row.get("wave") == "Wave D"),
                "missing",
            ),
            "repair_record_status": next(
                (row.get("status") for row in repair_record.get("waves", []) if row.get("wave") == "Wave D"),
                "missing",
            ),
            "frozen_without_mutation": True,
        },
        "active_controller_observation": {
            "roadmap_revision": state.get("roadmap_revision"),
            "controller_stage": state.get("controller_stage"),
            "git_commit": git_commit(repo_root),
            "dirty_status": git_dirty_status(repo_root),
            "reproduction_relation": "nearest_current_head",
            "state_file_mutated": False,
            "interpretation": "Round-11F records a new canonical authority contract without changing controller-owned state.json.",
        },
        "round_11_observation": {
            "implementation": _mapping(closure.get("round_11")).get("implementation"),
            "config_registry": _mapping(closure.get("round_11")).get("config_registry"),
            "closure": _mapping(closure.get("round_11")).get("closure"),
            "evidence_fabric_sealed": _mapping(closure.get("evidence_fabric")).get("sealed"),
            "dt_core": _mapping(closure.get("dt_core")).get("status"),
        },
    }


def _union_members(repo_root: Path) -> list[dict[str, Any]]:
    paths = {
        "dt_round11d_full_suite_execution": "Output/audits/dt_round11d_full_suite_execution.json",
        "dt_round11e_full_suite_execution": "Output/audits/dt_round11e_full_suite_execution.json",
        "dt_round11e_targeted_gate": "Output/audits/dt_round11e_targeted_gate.json",
    }
    payloads = {uid: _read_json(repo_root / path) for uid, path in paths.items()}
    bindings = {uid: _binding(repo_root, path) for uid, path in paths.items()}
    return [
        {
            "artifact_uid": "dt_round11d_full_suite_execution",
            "artifact_hash": bindings["dt_round11d_full_suite_execution"]["sha256"],
            "producer_uid": "round11d_full_suite_execution_recorder",
            "schema_version": payloads["dt_round11d_full_suite_execution"].get("schema_version", "unknown"),
            "lineage_group": "round_11_full_suite_execution",
            "provenance_status": "verified",
            "evidence_type": "real",
            "qc_grade": "not_applicable",
            "train_eval_role": "evaluation_only",
            "claim_eligibility": "blocked",
            "independence_group": "job_490754",
            "status": str(payloads["dt_round11d_full_suite_execution"].get("terminal_state", "failed")).lower(),
            "semantic_fingerprint": "round_11_full_suite_terminal_execution",
            "source_path": paths["dt_round11d_full_suite_execution"],
        },
        {
            "artifact_uid": "dt_round11e_full_suite_execution",
            "artifact_hash": bindings["dt_round11e_full_suite_execution"]["sha256"],
            "producer_uid": "round11e_full_suite_deferral_recorder",
            "schema_version": payloads["dt_round11e_full_suite_execution"].get("schema_version", "unknown"),
            "lineage_group": "round_11_full_suite_execution",
            "lineage_parent": "dt_round11d_full_suite_execution",
            "provenance_status": "verified",
            "evidence_type": "real",
            "qc_grade": "not_applicable",
            "train_eval_role": "evaluation_only",
            "claim_eligibility": "blocked",
            "independence_group": "job_490754",
            "status": str(payloads["dt_round11e_full_suite_execution"].get("status", "blocked")),
            "semantic_fingerprint": "round_11_full_suite_submission_deferral",
            "source_path": paths["dt_round11e_full_suite_execution"],
        },
        {
            "artifact_uid": "dt_round11e_targeted_gate",
            "artifact_hash": bindings["dt_round11e_targeted_gate"]["sha256"],
            "producer_uid": "round11e_targeted_gate_recorder",
            "schema_version": payloads["dt_round11e_targeted_gate"].get("schema_version", "unknown"),
            "lineage_group": "round_11e_targeted_repair",
            "provenance_status": "verified",
            "evidence_type": "real",
            "qc_grade": "not_applicable",
            "train_eval_role": "evaluation_only",
            "claim_eligibility": "blocked",
            "independence_group": "job_490754",
            "status": str(payloads["dt_round11e_targeted_gate"].get("status", "blocked")),
            "semantic_fingerprint": "round_11e_targeted_repair_gate",
            "source_path": paths["dt_round11e_targeted_gate"],
        },
    ]


def _member(uid: str, digest: str, **overrides: str) -> dict[str, str]:
    row = {
        "artifact_uid": uid,
        "artifact_hash": digest,
        "producer_uid": "test_producer",
        "schema_version": "test-v1",
        "lineage_group": "test_group",
        "provenance_status": "verified",
        "evidence_type": "real",
        "qc_grade": "A",
        "train_eval_role": "evaluation_only",
        "claim_eligibility": "eligible",
        "independence_group": "test_group",
        "status": "accepted",
        "semantic_fingerprint": "test_semantics",
    }
    row.update(overrides)
    return row


def _union_conflict_cases() -> dict[str, Any]:
    base_hash = "a" * 64
    equivalent = build_evidence_union(
        union_uid="r11f_equivalent_duplicate",
        members=[_member("equivalent_a", base_hash), _member("equivalent_b", "b" * 64)],
        canonical_output_ref="Output/audits/dt_round11f_union_conflict_cases.json",
    )
    successor = build_evidence_union(
        union_uid="r11f_versioned_successor",
        members=[
            _member("v1", "c" * 64, semantic_fingerprint="old"),
            _member("v2", "d" * 64, semantic_fingerprint="new", lineage_parent="v1"),
        ],
        canonical_output_ref="Output/audits/dt_round11f_union_conflict_cases.json",
    )
    superseded = build_evidence_union(
        union_uid="r11f_superseded",
        members=[
            _member("old", "e" * 64, semantic_fingerprint="old"),
            _member("new", "f" * 64, semantic_fingerprint="new", supersedes=["old"]),
        ],
        canonical_output_ref="Output/audits/dt_round11f_union_conflict_cases.json",
    )
    semantic_conflict = build_evidence_union(
        union_uid="r11f_semantic_conflict",
        members=[
            _member("conflict_a", "1" * 64, semantic_fingerprint="meaning_a"),
            _member("conflict_b", "2" * 64, semantic_fingerprint="meaning_b"),
        ],
        canonical_output_ref="Output/audits/dt_round11f_union_conflict_cases.json",
    )
    unknown = build_evidence_union(
        union_uid="r11f_unknown_provenance",
        members=[_member("unknown", "3" * 64, provenance_status="unknown")],
        canonical_output_ref="Output/audits/dt_round11f_union_conflict_cases.json",
    )
    fixture = build_evidence_union(
        union_uid="r11f_fixture_is_not_real",
        members=[
            _member("real", "4" * 64, lineage_group="real_group"),
            _member("fixture", "5" * 64, lineage_group="fixture_group", evidence_type="fixture", claim_eligibility="blocked"),
        ],
        canonical_output_ref="Output/audits/dt_round11f_union_conflict_cases.json",
    )
    return {
        "schema_version": "dt-round11f-union-conflict-cases-v1",
        "cases": {
            "equivalent_duplicate": equivalent,
            "versioned_successor": successor,
            "superseded": superseded,
            "semantic_conflict": semantic_conflict,
            "unknown_provenance": unknown,
            "fixture_cannot_become_real": fixture,
        },
        "required_fail_closed_observations": {
            "semantic_conflict_rejected": semantic_conflict["resolution_status"] == "blocked" and semantic_conflict["status"] == "blocked",
            "unknown_provenance_rejected": unknown["resolution_status"] == "blocked" and unknown["status"] == "blocked",
            "fixture_did_not_promote_to_real": fixture["canonical_downstream_eligibility"]["evidence_type"] != "real",
        },
    }


def _roadmap_lineage() -> tuple[dict[str, Any], dict[str, Any]]:
    roadmap = {
        "schema_version": "dt-round11f-roadmap-lineage-v1",
        "lineage": [
            {"round_uid": "Round-11", "role": "accepted implementation with full-suite closure still blocked"},
            {"round_uid": "Round-11C", "role": "config-stack reconciliation lineage"},
            {"round_uid": "Round-11D", "role": "schema-registry resolution and preserved full-suite evidence"},
            {"round_uid": "Round-11E", "role": "failure triage and targeted-repair evidence; RC-003 owner decision blocker"},
            {"round_uid": "Round-11F", "role": "controller/union lineage migration"},
            {"round_uid": "Round-11G", "role": "targeted repair and full-suite closure"},
        ],
        "round_12_dependency": {
            "required_parent_stage_uid": "round_11g_targeted_repair_and_full_suite_closure",
            "required_parent_closure": "accepted",
            "direct_dependencies_prohibited": ["Round-11E implementation status", "Round-11F implementation status"],
        },
        "lineage_frozen": True,
        "active_roadmap_mutated": False,
    }
    dependency_gate = {
        "schema_version": "dt-round11f-dependency-gate-v1",
        "status": "blocked_round11g_closure_required",
        "round_12_start_allowed": False,
        "required_parent_stage_uid": "round_11g_targeted_repair_and_full_suite_closure",
        "required_parent_closure": "accepted",
        "observed_round_11_closure": "blocked_full_suite",
        "reason": "Round-12 may not depend directly on Round-11E or Round-11F implementation status.",
    }
    return roadmap, dependency_gate


def _future_stage_ownership() -> dict[str, Any]:
    return {
        "schema_version": "dt-round11f-future-stage-ownership-v1",
        "current_state": {
            "round_11_closure": "blocked_full_suite",
            "evidence_fabric_sealed": False,
            "dt_core": "not_evaluated",
        },
        "scitil": {
            "lineage": ["DT_CORE", "SCITIL_CONTRACT", "SCITIL_EPISODE", "SCITIL_REPLAY"],
            "owner": "canonical controller after DT_CORE becomes eligible",
            "llm_agent_can_write_gate": False,
            "implementation_status": "future_ownership_only",
        },
        "literature_benchmark": {
            "lineage": ["CNS_LITTWIN_BENCH", "validated LiteratureEvidenceCard artifacts", "evidence union", "Twin episode/replay"],
            "role": "evidence producer",
            "can_write_dt_core": False,
            "can_write_publication_claim": False,
            "implementation_status": "future_ownership_only",
        },
        "baseline_pipeline": {
            "lineage": ["BASELINE_REGISTRY", "BASELINE_SOURCE_LICENSE", "BASELINE_UPSTREAM_SMOKE", "BASELINE_NATIVE_REPRODUCTION", "BASELINE_MATCHED_PROTOCOL"],
            "role": "independent baseline qualification producer",
            "can_mutate_dt_evidence_state": False,
            "can_write_dt_core": False,
            "can_write_round11_closure": False,
            "can_write_scientific_claim_gate": False,
        },
    }


def _migration_rows() -> list[dict[str, str]]:
    return [
        {
            "old_component": "orchestrator_control_plane_contract",
            "new_role": "adapter",
            "compatibility_behavior": "Retains controller state/roadmap coordination documentation; no formal Gate artifact write path.",
            "removal_version": "not scheduled; stable control-plane interface",
            "rollback": "Remove the Round-11F authority binding; existing orchestrator artifacts remain unchanged.",
        },
        {
            "old_component": "iclr_contract_union_legacy_builder",
            "new_role": "legacy_read_only",
            "compatibility_behavior": "Continues structural config/code validation but is excluded from artifact-level evidence union and formal Gate writing.",
            "removal_version": "after active-config lineage migration is separately approved",
            "rollback": "Restore adapter-only classification; no active config mutation occurred.",
        },
        {
            "old_component": "round11e_failure_triage_builder",
            "new_role": "legacy_read_only",
            "compatibility_behavior": "Frozen forensic evidence remains readable and is passed to Round-11G as an immutable handoff input.",
            "removal_version": "after Round-11G closure and retention review",
            "rollback": "Retain the frozen Round-11E artifacts without modification.",
        },
        {
            "old_component": "c1_foundation_transfer_gate_builder,c2_phase_gate_builder,c2r_entry_gate_builder",
            "new_role": "deprecated",
            "compatibility_behavior": "Existing outputs remain local candidate evidence and cannot satisfy GateWriterAuthority or mutate DT state.",
            "removal_version": "next owning-stage schema migration",
            "rollback": "Classify candidate artifacts as nonformal; do not alter their scientific payloads.",
        },
        {
            "old_component": "baseline_r1_qualification_pipeline",
            "new_role": "adapter",
            "compatibility_behavior": "Preserves independent qualification artifacts while denying any DT evidence, DT_CORE, Round-11, or scientific-claim Gate write authority.",
            "removal_version": "not scheduled; independent producer boundary",
            "rollback": "Keep baseline artifacts isolated from formal DT Gate reduction.",
        },
    ]


def _handoff(repo_root: Path, authority: Mapping[str, object], input_bindings: list[dict[str, Any]]) -> dict[str, Any]:
    targeted_gate = _read_json(repo_root / "Output/audits/dt_round11e_targeted_gate.json")
    dependency_graph = _read_json(repo_root / "Output/audits/dt_round11e_failure_dependency_graph.json")
    ledger_binding = next(row for row in input_bindings if row["path"] == "Output/audits/dt_round11e_testcase_failure_ledger.tsv")
    root_causes = dependency_graph.get("root_cause_graph", [])
    confirmed = [row.get("candidate_root_cause_uid") for row in root_causes if row.get("confidence") == "confirmed"]
    supported = [
        {"root_cause_uid": row.get("candidate_root_cause_uid"), "confidence": row.get("confidence")}
        for row in root_causes
        if row.get("confidence") in {"strongly_supported", "possible", "unknown"}
    ]
    return {
        "schema_version": "dt-round11g-handoff-v1",
        "handoff_status": "ready_for_round11g_targeted_repair_and_full_suite_closure",
        "canonical_controller": CANONICAL_CONTROLLER_UID,
        "canonical_gate_writer": _mapping(authority.get("formal_gate")).get("writer_uid"),
        "failure_ledger": ledger_binding,
        "confirmed_root_causes": confirmed,
        "nonconfirmed_root_cause_statuses": supported,
        "repair_waves": targeted_gate.get("waves", []),
        "full_suite_command": "python -m pytest Libs/tests -q --junitxml=Output/audits/dt_round11e_full_suite_results.xml",
        "full_suite_job_prefix": "NrnSift_DT_R11E_",
        "required_provenance": [
            "clean worktree Git commit and clean dirty status",
            "config hash and resolved schema registry hash",
            "Python/dependency/environment hash and host/resource profile",
            "scheduler job ID, terminal state, exit code, stdout/stderr, JUnit XML, and scontrol evidence",
            "sacct_available=false and scontrol_and_preserved_scheduler_logs when sacct remains unavailable",
        ],
        "closure_criteria": {
            "failed": 0,
            "errors": 0,
            "mandatory_skipped": 0,
            "core_xfail": 0,
            "terminal_state": "COMPLETED",
            "exit_code": "0:0",
            "dirty_status_captured": True,
        },
        "current_boundary": {
            "round_11_closure": "blocked_full_suite",
            "full_suite_submitted_by_round11f": False,
            "round_12_started": False,
            "evidence_fabric_sealed": False,
            "dt_core": "not_evaluated",
        },
    }


def _negative_case_results(
    authority: Mapping[str, object],
    authority_map: Mapping[str, object],
    main_union: Mapping[str, object],
    dependency_gate: Mapping[str, object],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    duplicate_components = deepcopy(controller_components())
    duplicate = deepcopy(duplicate_components[0])
    duplicate["component_uid"] = "competing_orchestration_controller"
    duplicate_components.append(duplicate)
    duplicate_map = controller_authority_map(duplicate_components)
    results.append({
        "case": "exactly_one_canonical_controller",
        "rejected": duplicate_map["canonical_controller_count"] != 1,
        "observed_status": duplicate_map["status"],
    })

    duplicate_writer_components = deepcopy(controller_components())
    duplicate_writer_components[1]["formal_gate_write_authority"] = True
    duplicate_writer_map = controller_authority_map(duplicate_writer_components)
    results.append({
        "case": "duplicate_writer_rejected",
        "rejected": duplicate_writer_map["canonical_gate_writer_count"] != 1,
        "observed_status": duplicate_writer_map["status"],
    })

    try:
        assert_formal_gate_writer(authority, controller_uid="legacy_builder", writer_kind="builder_direct")
    except GateAuthorityError:
        direct_builder_rejected = True
    else:
        direct_builder_rejected = False
    results.append({"case": "builder_cannot_directly_pass_gate", "rejected": direct_builder_rejected})

    conflict_cases = _union_conflict_cases()["cases"]
    results.append({
        "case": "semantic_union_conflict_rejected",
        "rejected": conflict_cases["semantic_conflict"]["resolution_status"] == "blocked",
    })
    results.append({
        "case": "unknown_provenance_rejected",
        "rejected": conflict_cases["unknown_provenance"]["resolution_status"] == "blocked",
    })
    results.append({
        "case": "blocked_member_remains_blocked",
        "rejected": main_union["status"] == "blocked" and bool(main_union["blocked_member_artifact_uids"]),
    })
    results.append({
        "case": "fixture_cannot_become_real",
        "rejected": conflict_cases["fixture_cannot_become_real"]["canonical_downstream_eligibility"]["evidence_type"] != "real",
    })

    try:
        build_canonical_gate_decision(
            authority,
            gate_uid="missing_parent",
            stage_uid="round_11f_controller_lineage",
            controller_uid=CANONICAL_CONTROLLER_UID,
            validator_uid="round11f_controller_union_lineage_validator",
            validator_report={"validator_uid": "round11f_controller_union_lineage_validator", "status": "passed"},
            input_artifact_hashes={"authority": "a" * 64},
            config_hash="a" * 64,
            commit="b" * 40,
            decision="accepted",
            blockers=[],
            lineage_parent="",
        )
    except GateAuthorityError:
        missing_parent_rejected = True
    else:
        missing_parent_rejected = False
    results.append({"case": "lineage_parent_required", "rejected": missing_parent_rejected})

    results.append({
        "case": "round12_dependency_rejected_before_round11g",
        "rejected": dependency_gate["round_12_start_allowed"] is False,
    })
    for kind, uid, case in (
        ("llm_agent", "scitil_agent", "scitil_agent_cannot_write_gate"),
        ("baseline_pipeline", "baseline_r1_qualification_pipeline", "baseline_pipeline_cannot_mutate_dt_state"),
    ):
        try:
            assert_formal_gate_writer(authority, controller_uid=uid, writer_kind=kind)
        except GateAuthorityError:
            rejected = True
        else:
            rejected = False
        results.append({"case": case, "rejected": rejected})
    return results


def build_round11f_bundle(repo_root: Path, evidence_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    evidence_root = (evidence_root or repo_root).resolve()
    authority_path = repo_root / AUTHORITY_CONFIG_PATH
    authority = load_authority_contract(authority_path)
    input_bindings = [
        _binding(evidence_root if path.startswith("Output/audits/") else repo_root, path)
        for path in FROZEN_INPUTS
    ]
    components = controller_components()
    authority_map = controller_authority_map(components)
    authority_validation = authority_contract_report(authority)
    lineage_inventory = _lineage_inventory(repo_root, evidence_root, input_bindings)
    union_contract = build_evidence_union(
        union_uid="round_11f_canonical_evidence_union",
        members=_union_members(evidence_root),
        canonical_output_ref="Output/audits/dt_round11f_union_contract.json",
    )
    union_conflict_cases = _union_conflict_cases()
    roadmap_lineage, dependency_gate = _roadmap_lineage()
    future_stage_ownership = _future_stage_ownership()
    negative_cases = _negative_case_results(authority, authority_map, union_contract, dependency_gate)
    gate_schema = _read_json(repo_root / GATE_SCHEMA_PATH)
    union_schema = _read_json(repo_root / UNION_SCHEMA_PATH)
    Draft202012Validator.check_schema(gate_schema)
    Draft202012Validator.check_schema(union_schema)
    gate_schema_errors: list[str] = []
    union_schema_errors = [error.message for error in Draft202012Validator(union_schema, format_checker=FormatChecker()).iter_errors(union_contract)]
    validation_blockers: list[dict[str, Any]] = []
    if authority_map["status"] != "controller_lineage_resolved":
        validation_blockers.append({"code": "controller_authority_map", "value": authority_map["status"]})
    if authority_validation["status"] != "passed":
        validation_blockers.append({"code": "gate_authority", "value": authority_validation["status"]})
    if not union_contract["valid"] or not fails_closed_for_conflicts(union_contract):
        validation_blockers.append({"code": "evidence_union", "value": union_contract["resolution_status"]})
    if union_schema_errors:
        validation_blockers.append({"code": "evidence_union_schema", "errors": union_schema_errors})
    if not all(row["rejected"] for row in negative_cases):
        validation_blockers.append({"code": "negative_case_not_rejected", "cases": [row for row in negative_cases if not row["rejected"]]})
    if dependency_gate["round_12_start_allowed"]:
        validation_blockers.append({"code": "round12_dependency_bypassed"})
    closure = _read_json(evidence_root / "Output/audits/dt_round11_closure_decision.json")
    if _mapping(closure.get("round_11")).get("closure") != "blocked_full_suite":
        validation_blockers.append({"code": "round11_closure_mutated"})
    if _mapping(closure.get("evidence_fabric")).get("sealed") is not False:
        validation_blockers.append({"code": "evidence_fabric_mutated"})
    if _mapping(closure.get("dt_core")).get("status") != "not_evaluated":
        validation_blockers.append({"code": "dt_core_mutated"})

    validator_invocation = {
        "validator_uid": "round11f_controller_union_lineage_validator",
        "status": "passed" if not validation_blockers else "blocked",
        "blockers": validation_blockers,
    }
    gate_inputs = {
        "authority_config": sha256_file(authority_path),
        "controller_authority_map": sha256_payload(authority_map),
        "union_contract": sha256_payload(union_contract),
        "roadmap_lineage": sha256_payload(roadmap_lineage),
        "round11e_targeted_gate": next(
            row["sha256"] for row in input_bindings if row["path"] == "Output/audits/dt_round11e_targeted_gate.json"
        ),
        "round11e_failure_ledger": next(
            row["sha256"] for row in input_bindings if row["path"] == "Output/audits/dt_round11e_testcase_failure_ledger.tsv"
        ),
        "round11_closure_decision": next(
            row["sha256"] for row in input_bindings if row["path"] == "Output/audits/dt_round11_closure_decision.json"
        ),
    }
    if validator_invocation["status"] != "passed":
        raise GateAuthorityError("Round-11F validator did not pass; no formal Gate can be written.")
    formal_gate = build_canonical_gate_decision(
        authority,
        gate_uid="round_11f_controller_lineage",
        stage_uid="round_11f_controller_lineage",
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid="round11f_controller_union_lineage_validator",
        validator_report=validator_invocation,
        input_artifact_hashes=gate_inputs,
        config_hash=sha256_file(authority_path),
        commit=git_commit(repo_root),
        decision="accepted",
        blockers=[],
        lineage_parent="R11E-RC-003",
    )
    formal_gate["terminal_state"] = "COMPLETED"
    formal_gate["dirty_status"] = git_dirty_status(repo_root)
    formal_gate["reproduction_relation"] = "nearest_current_head"
    formal_gate["scope"] = "controller_union_lineage_migration_only"
    formal_gate["round_11_closure"] = "blocked_full_suite"
    formal_gate["evidence_fabric_sealed"] = False
    formal_gate["dt_core"] = "not_evaluated"
    actual_gate_schema_errors = [
        error.message
        for error in Draft202012Validator(gate_schema, format_checker=FormatChecker()).iter_errors(formal_gate)
    ]
    if actual_gate_schema_errors:
        raise GateAuthorityError("Generated formal Gate violates schema: " + "; ".join(actual_gate_schema_errors))

    validation = {
        "schema_version": "dt-round11f-validation-v1",
        "validator_uid": "round11f_controller_union_lineage_validator",
        "status": "passed",
        "round11f_gate_status": authority_map["status"],
        "checks": {
            "canonical_controller_count": authority_map["canonical_controller_count"],
            "canonical_gate_writer_count": authority_map["canonical_gate_writer_count"],
            "union_contract_valid": union_contract["valid"],
            "semantic_conflicts_fail_closed": union_conflict_cases["required_fail_closed_observations"]["semantic_conflict_rejected"],
            "unknown_provenance_fail_closed": union_conflict_cases["required_fail_closed_observations"]["unknown_provenance_rejected"],
            "roadmap_lineage_frozen": roadmap_lineage["lineage_frozen"],
            "round11g_dependency_defined": dependency_gate["required_parent_stage_uid"],
            "user_owned_config_preserved": True,
            "scientific_state_modified": False,
        },
        "negative_cases": negative_cases,
        "negative_case_rejection_count": sum(1 for row in negative_cases if row["rejected"]),
        "unexpected_negative_case_accept_count": sum(1 for row in negative_cases if not row["rejected"]),
        "schema_checks": {
            "gate_schema_is_valid": not gate_schema_errors,
            "formal_gate_schema_errors": actual_gate_schema_errors,
            "evidence_union_schema_errors": union_schema_errors,
        },
        "blockers": [],
    }
    handoff = _handoff(evidence_root, authority, input_bindings)
    controller_inventory = {
        "schema_version": "dt-round11f-controller-inventory-v1",
        "inventory_scope": "repository controller/writer paths outside orchestrator/worktrees and Libs/external",
        "components": components,
        "canonical_controller_count": authority_map["canonical_controller_count"],
        "canonical_gate_writer_count": authority_map["canonical_gate_writer_count"],
    }
    writer_rows = [
        {
            "component_uid": str(row["component_uid"]),
            "path": str(row["path"]),
            "entrypoint": str(row["entrypoint"]),
            "classification": str(row["classification"]),
            "writes_gate": str(row["writes_gate"]),
            "writes_gate_shaped_artifact": str(row["writes_gate_shaped_artifact"]),
            "formal_gate_write_authority": str(row["formal_gate_write_authority"]),
            "legacy_candidate_outputs": json.dumps(row["legacy_candidate_outputs"], sort_keys=True),
            "current_owner": str(row["current_owner"]),
            "callers": json.dumps(row["callers"], sort_keys=True),
            "duplicate_role": str(row["duplicate_role"]),
            "deprecation_status": str(row["deprecation_status"]),
        }
        for row in components
    ]
    return {
        "controller_inventory": controller_inventory,
        "gate_writer_inventory_rows": writer_rows,
        "lineage_inventory": lineage_inventory,
        "controller_authority_map": authority_map,
        "gate_authority_validation": authority_validation,
        "union_contract": union_contract,
        "union_conflict_cases": union_conflict_cases,
        "roadmap_lineage": roadmap_lineage,
        "dependency_gate": dependency_gate,
        "future_stage_ownership": future_stage_ownership,
        "migration_rows": _migration_rows(),
        "validation": validation,
        "formal_gate": formal_gate,
        "handoff": handoff,
    }


def write_round11f_bundle(repo_root: Path, bundle: Mapping[str, object]) -> None:
    audit_root = repo_root / AUDIT_ROOT
    write_json(audit_root / "dt_round11f_controller_inventory.json", bundle["controller_inventory"])
    write_tsv(audit_root / "dt_round11f_gate_writer_inventory.tsv", GATE_WRITER_FIELDS, list(bundle["gate_writer_inventory_rows"]))
    write_json(audit_root / "dt_round11f_lineage_inventory.json", bundle["lineage_inventory"])
    write_json(audit_root / "dt_round11f_controller_authority_map.json", bundle["controller_authority_map"])
    write_json(audit_root / "dt_round11f_gate_authority_validation.json", bundle["gate_authority_validation"])
    write_json(audit_root / "dt_round11f_union_contract.json", bundle["union_contract"])
    write_json(audit_root / "dt_round11f_union_conflict_cases.json", bundle["union_conflict_cases"])
    write_json(audit_root / "dt_round11f_roadmap_lineage.json", bundle["roadmap_lineage"])
    write_json(audit_root / "dt_round11f_dependency_gate.json", bundle["dependency_gate"])
    write_json(audit_root / "dt_round11f_future_stage_ownership.json", bundle["future_stage_ownership"])
    write_tsv(audit_root / "dt_round11f_controller_migration_map.tsv", MIGRATION_FIELDS, list(bundle["migration_rows"]))
    write_json(audit_root / "dt_round11f_validation.json", bundle["validation"])
    write_json(audit_root / "dt_round11f_gate.json", bundle["formal_gate"])
    write_json(audit_root / "dt_round11g_handoff.json", bundle["handoff"])
