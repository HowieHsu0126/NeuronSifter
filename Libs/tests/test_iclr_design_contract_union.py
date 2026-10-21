from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.g0_integrity import _structural_reports  # noqa: E402
from Libs.utils.data.iclr_contract_union import (  # noqa: E402
    build_design_coverage_report,
    build_orchestrator_state_report,
    build_union_consistency_report,
    load_union_context,
)


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
EXPECTED_CLAIMS = {
    "C1": "public_foundation_state_transfer",
    "C2": "structured_stochastic_world_action",
    "C3": "cost_aware_multi_tool_acquisition",
    "C4": "tri_gated_reliable_screening",
    "C5": "scale_aware_throughput_support",
}


def loaded_context() -> tuple[dict[str, object], dict[str, object]]:
    return load_union_context(CONFIG_PATH)


def blocker_codes(report: dict[str, object]) -> set[str]:
    return {str(row["code"]) for row in report["blockers"]}


def test_repository_contract_union_is_zero_blocker_and_binds_all_audits() -> None:
    config, provenance = loaded_context()
    report = build_union_consistency_report(config, provenance, CONFIG_PATH)
    assert report["status"] == "pass"
    assert report["blocker_count"] == 0
    assert report["document_count"] == 39
    assert report["schema_family_count"] == 12
    assert report["claim_count"] == 5
    assert {key: row["status"] for key, row in report["bound_audits"].items()} == {
        "config_stack": "pass",
        "design_coverage": "pass",
        "orchestrator_state": "pass",
        "public_mainline_isolation": "pass",
    }
    assert report["bound_audits"]["public_mainline_isolation"]["controlled_reachability_count"] == 0
    assert all(row["blocker_count"] == 0 for row in report["bound_audits"].values())
    assert all(len(row["report_sha256"]) == 64 for row in report["bound_audits"].values())


def test_claims_and_phase0_migration_hash_are_exact_and_unpromoted() -> None:
    config, _ = loaded_context()
    union = config["iclr_design_contract_union"]
    assert {row["claim_id"]: row["meaning"] for row in union["claims"]} == EXPECTED_CLAIMS
    assert {row["result_status"] for row in union["claims"]} == {"requires_experiment_result"}
    migration_path = REPO_ROOT / union["phase0_evidence"]["claim_migration_path"]
    assert hashlib.sha256(migration_path.read_bytes()).hexdigest() == union["phase0_evidence"]["claim_migration_sha256"]


def test_union_preserves_public_only_validation_only_and_no_standin_policies() -> None:
    config, _ = loaded_context()
    policies = config["iclr_design_contract_union"]["policies"]
    assert policies["public_mainline_only"] is True
    assert policies["controlled_access_allowed"] is False
    assert {"ADNI", "ROSMAP_controlled", "AMP_AD_controlled", "dbGaP", "UK_Biobank"}.issubset(policies["excluded_controlled_sources"])
    assert policies["calibration_source_role"] == "validation"
    assert policies["threshold_selection_source_role"] == "validation"
    assert {"mock", "unit_fixture", "cached_standin", "synthetic_substitute", "silent_fallback"}.issubset(policies["full_acceptance_forbidden_evidence"])
    assert policies["checkpoint_job_classes_allowed_during_contract_installation"] == []


def test_all_twelve_design_schema_field_unions_are_exact() -> None:
    config, provenance = loaded_context()
    report = build_union_consistency_report(config, provenance, CONFIG_PATH)
    assert not {code for code in blocker_codes(report) if code.startswith("schema_") or code.startswith("design_schema_")}
    assert {row["schema_family"] for row in config["iclr_design_contract_union"]["schema_families"]} == {
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


def test_exact_39_design_declarations_have_complete_ownership_metadata() -> None:
    config, _ = loaded_context()
    report = build_design_coverage_report(config)
    assert report["status"] == "pass"
    assert report["declared_document_count"] == report["discovered_document_count"] == report["resolved_declaration_count"] == 39
    assert report["duplicate_declarations"] == []
    assert report["missing_declarations"] == []
    assert report["stale_declarations"] == []
    for row in report["documents"]:
        assert row["canonical_contract_ids"]
        assert row["owning_milestone_id"]
        assert row["owning_direction_id"]
        assert row["downstream_consumer_ids"]
        assert row["responsible_validator_ids"]
        assert row["mandatory_requirement_ids"]
        assert row["conflicts"]
        assert "supersedes" in row


def test_active_g0_and_inactive_future_contract_ownership_remain_distinct() -> None:
    config, provenance = loaded_context()
    union = config["iclr_design_contract_union"]
    report = build_union_consistency_report(config, provenance, CONFIG_PATH)
    ownership = report["contract_ownership"]
    active_lineage = union["active_lineage"]
    active_contracts = [
        row["contract_id"]
        for row in union["contract_registry"]
        if row["owning_milestone_id"] == active_lineage["milestone_id"]
        and row["owning_direction_id"] == active_lineage["direction_id"]
    ]
    inactive_pairs = {
        (row["owning_milestone_id"], row["owning_direction_id"])
        for row in ownership["inactive_future_ownership_registry"]
    }
    assert ownership["active_roadmap_contract_ids"] == sorted(active_contracts)
    assert set(ownership["inactive_future_contract_ids"]) == {
        "data-materialization",
        "integrity-claim-discipline",
        "literature-positioning",
        "model-training-evaluation",
        "package-quality",
        "public-data-governance",
        "publication-gates",
        "reproducibility-release",
        "source-governance",
        "tool-acquisition",
        "tool-adapter-security",
    }
    assert all(pair[0] != active_lineage["milestone_id"] for pair in inactive_pairs)
    assert all(
        (row["owning_milestone_id"], row["owning_direction_id"]) in inactive_pairs
        for row in union["contract_registry"]
        if row["contract_id"] in ownership["inactive_future_contract_ids"]
    )


def test_unregistered_inactive_future_contract_ownership_fails_closed() -> None:
    config, provenance = loaded_context()
    mutated = deepcopy(config)
    registry = mutated["iclr_design_contract_union"]["inactive_roadmap_ownership_registry"]
    registry[:] = [
        row
        for row in registry
        if row["owning_direction_id"] != "implement-foundation-objectives"
    ]
    report = build_union_consistency_report(mutated, provenance, CONFIG_PATH)
    assert {"unknown_contract_milestone", "unknown_contract_direction", "unregistered_inactive_contract_ownership"}.issubset(blocker_codes(report))


def test_duplicate_canonical_ownership_fails_closed() -> None:
    config, provenance = loaded_context()
    mutated = deepcopy(config)
    mutated["iclr_design_contract_union"]["contract_registry"].append(
        deepcopy(mutated["iclr_design_contract_union"]["contract_registry"][0])
    )
    report = build_union_consistency_report(mutated, provenance, CONFIG_PATH)
    assert "duplicate_canonical_ownership" in blocker_codes(report)
    assert report["blocker_count"] > 0


def test_circular_supersession_fails_closed() -> None:
    config, provenance = loaded_context()
    mutated = deepcopy(config)
    contracts = mutated["iclr_design_contract_union"]["contract_registry"]
    contracts[0]["supersedes"] = [contracts[1]["contract_id"]]
    contracts[1]["supersedes"] = [contracts[0]["contract_id"]]
    report = build_union_consistency_report(mutated, provenance, CONFIG_PATH)
    assert "circular_supersession" in blocker_codes(report)


def test_unresolved_claim_id_conflict_fails_closed() -> None:
    config, provenance = loaded_context()
    mutated = deepcopy(config)
    mutated["iclr_design_contract_union"]["claims"].append(
        {
            "claim_id": "C1",
            "meaning": "conflicting_meaning",
            "result_status": "requires_experiment_result",
            "claim_gate_status": "contract_frozen",
        }
    )
    report = build_union_consistency_report(mutated, provenance, CONFIG_PATH)
    assert "unresolved_claim_id_conflict" in blocker_codes(report)


def test_competing_source_of_truth_fails_closed() -> None:
    config, provenance = loaded_context()
    mutated = deepcopy(config)
    mutated["iclr_design_contract_union"]["source_of_truth_registry"].append(
        {
            "domain_id": "normalized_runtime_contract_fields",
            "canonical_source": "Docs/design/iclr/config/ccfa.proposed.yaml",
            "authority": "invalid_competitor",
        }
    )
    report = build_union_consistency_report(mutated, provenance, CONFIG_PATH)
    assert "competing_source_of_truth" in blocker_codes(report)


def test_public_legacy_model_name_fails_closed() -> None:
    config, provenance = loaded_context()
    mutated = deepcopy(config)
    mutated["iclr_design_contract_union"]["public_identity"]["external_model_name"] = "MIND-DiNO"
    report = build_union_consistency_report(mutated, provenance, CONFIG_PATH)
    assert "public_legacy_model_name" in blocker_codes(report)


def test_historical_override_fails_closed() -> None:
    config, provenance = loaded_context()
    mutated = deepcopy(config)
    documents = mutated["validation"]["design_contract_coverage"]["documents"]
    changelog = next(row for row in documents if row["path"].endswith("CHANGELOG.md"))
    changelog["supersedes"] = ["Docs/design/iclr/00_MASTER_ORCHESTRATION.md"]
    report = build_union_consistency_report(mutated, provenance, CONFIG_PATH)
    assert "historical_override" in blocker_codes(report)


def test_uncovered_mandatory_requirement_fails_closed() -> None:
    config, provenance = loaded_context()
    mutated = deepcopy(config)
    for document in mutated["validation"]["design_contract_coverage"]["documents"]:
        document["mandatory_requirement_ids"] = [
            requirement_id
            for requirement_id in document["mandatory_requirement_ids"]
            if requirement_id != "req_package_metadata"
        ]
    report = build_union_consistency_report(mutated, provenance, CONFIG_PATH)
    assert "uncovered_mandatory_requirement" in blocker_codes(report)


def test_missing_and_stale_document_declarations_fail_closed() -> None:
    config, _ = loaded_context()
    missing_config = deepcopy(config)
    removed = missing_config["validation"]["design_contract_coverage"]["documents"].pop()
    missing_report = build_design_coverage_report(missing_config)
    assert "missing_document_declaration" in blocker_codes(missing_report)

    stale_config = deepcopy(config)
    stale_row = deepcopy(removed)
    stale_row["path"] = "Docs/design/iclr/not-a-document.md"
    stale_config["validation"]["design_contract_coverage"]["documents"].append(stale_row)
    stale_report = build_design_coverage_report(stale_config)
    assert "stale_document_declaration" in blocker_codes(stale_report)


def test_unknown_document_status_and_registry_reference_fail_closed() -> None:
    config, _ = loaded_context()
    mutated = deepcopy(config)
    document = mutated["validation"]["design_contract_coverage"]["documents"][0]
    document["document_status"] = "unknown"
    document["responsible_validator_ids"] = ["unknown-validator"]
    report = build_design_coverage_report(mutated)
    assert {"unknown_document_status", "unknown_validator_reference"}.issubset(blocker_codes(report))


def test_orchestrator_state_validation_is_read_only() -> None:
    config, _ = loaded_context()
    state_path = REPO_ROOT / "orchestrator/state.json"
    before = hashlib.sha256(state_path.read_bytes()).hexdigest()
    report = build_orchestrator_state_report(config)
    after = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert report["status"] == "pass"
    assert report["blocker_count"] == 0
    assert report["state_mutated"] is False
    assert before == report["state_sha256_before"] == report["state_sha256_after"] == after


def test_orchestrator_state_rejects_roadmap_family_mismatch_and_revision_regression() -> None:
    config, _ = loaded_context()
    state_path = REPO_ROOT / "orchestrator/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    unknown_family = deepcopy(state)
    unknown_family["roadmap_id"] = "different-roadmap-family"
    unknown_family["roadmap_revision"] = "rev-002"
    report = build_orchestrator_state_report(config, state_payload=unknown_family)
    assert {"orchestrator_lineage_family_mismatch", "orchestrator_roadmap_dir_mismatch"}.issubset(blocker_codes(report))
    assert "orchestrator_lineage_regression" not in blocker_codes(report)

    policy = config["iclr_design_contract_union"]["controller_lineage_policy"]
    assert state["roadmap_id"] == policy["roadmap_id"]
    minimum_revision = int(policy["minimum_roadmap_revision"].split("-")[1])
    state["roadmap_revision"] = f"rev-{minimum_revision - 1:03d}"
    report = build_orchestrator_state_report(config, state_payload=state)
    assert {"orchestrator_lineage_regression", "orchestrator_roadmap_dir_mismatch"}.issubset(blocker_codes(report))
    assert "orchestrator_lineage_family_mismatch" not in blocker_codes(report)


def test_orchestrator_state_rejects_invalid_installed_union_selection() -> None:
    config, _ = loaded_context()
    mutated = deepcopy(config)
    mutated["iclr_design_contract_union"]["active_lineage"]["direction_id"] = "unknown-direction"
    report = build_orchestrator_state_report(mutated)
    assert "unknown_installed_lineage_direction" in blocker_codes(report)


def test_orchestrator_state_rejects_extra_fields_and_invalid_stage() -> None:
    config, _ = loaded_context()
    state_path = REPO_ROOT / "orchestrator/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["retired_merge_ready"] = True
    state["controller_stage"] = "invalid"
    report = build_orchestrator_state_report(config, state_payload=state)
    assert {"orchestrator_state_extra_fields", "orchestrator_controller_stage"}.issubset(blocker_codes(report))


def test_orchestrator_state_rejects_missing_bundle_and_duplicate_roadmap_ids(tmp_path: Path) -> None:
    config, _ = loaded_context()
    state_path = REPO_ROOT / "orchestrator/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["roadmap_dir"] = str(tmp_path / "missing")
    missing_report = build_orchestrator_state_report(config, state_payload=state)
    assert {"missing_active_roadmap", "missing_active_verification"}.issubset(blocker_codes(missing_report))

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "roadmap.md").write_text(
        "# Roadmap\n\n## Milestones\n\n### [pending] One\nMilestone id: duplicate\n- Direction id: same\n"
        "\n### [pending] Two\nMilestone id: duplicate\n- Direction id: same\n",
        encoding="utf-8",
    )
    (bundle / "verification.md").write_text(
        "# Verification\n\n## Baseline Checks\n\n## Alignment Checks\n\n## Task-Specific Checks\n"
        "\n## Manual Checks\n\n## Roadmap Overrides\n",
        encoding="utf-8",
    )
    state["roadmap_dir"] = str(bundle)
    duplicate_report = build_orchestrator_state_report(config, state_payload=state)
    assert {"duplicate_roadmap_milestone_id", "duplicate_roadmap_direction_id"}.issubset(blocker_codes(duplicate_report))


def test_pending_migration_numeric_source_venue_and_empirical_work_stays_visible() -> None:
    config, provenance = loaded_context()
    report = build_union_consistency_report(config, provenance, CONFIG_PATH)
    assert set(report["pending_work"]) == {
        "naming_migration",
        "controlled_branch_isolation",
        "numeric_lineage",
        "source_currency",
        "venue_policy",
        "empirical_claims",
    }
    assert report["pending_work"]["empirical_claims"] == [
        "C1_requires_experiment_result",
        "C2_requires_experiment_result",
        "C3_requires_experiment_result",
        "C4_requires_experiment_result",
        "C5_requires_experiment_result",
    ]
    pending_conflict_ids = {row["conflict_id"] for row in report["pending_conflicts"]}
    resolved_conflict_ids = {row["conflict_id"] for row in report["resolved_conflicts"]}
    assert pending_conflict_ids >= {"D", "V", "E"}
    assert {"L", "P"}.isdisjoint(pending_conflict_ids)
    assert {"L", "P"}.issubset(resolved_conflict_ids)


def test_missing_installed_roadmap_retains_declared_selection_and_blocks(tmp_path: Path) -> None:
    config, _ = loaded_context()
    mutated = deepcopy(config)
    lineage = mutated["iclr_design_contract_union"]["active_lineage"]
    lineage["roadmap_dir"] = str(tmp_path / "absent_installed_bundle")
    state_path = REPO_ROOT / "orchestrator/state.json"
    before = state_path.read_bytes()

    report = build_orchestrator_state_report(mutated)

    assert report["status"] == "blocked"
    assert "missing_installed_lineage_roadmap" in blocker_codes(report)
    assert report["selected_milestone_id"] == lineage["milestone_id"]
    assert report["selected_direction_id"] == lineage["direction_id"]
    assert report["state_mutated"] is False
    assert state_path.read_bytes() == before


def test_controller_policy_preserves_live_state_and_installed_ownership() -> None:
    config, _ = loaded_context()
    union = config["iclr_design_contract_union"]
    policy = union["controller_lineage_policy"]
    installed = deepcopy(union["active_lineage"])
    state = json.loads((REPO_ROOT / "orchestrator/state.json").read_text(encoding="utf-8"))
    report = build_orchestrator_state_report(config)
    assert report["status"] == "pass"
    assert report["blocker_count"] == 0
    assert report["roadmap_id"] == state["roadmap_id"] == policy["roadmap_id"]
    assert report["roadmap_revision"] == state["roadmap_revision"]
    assert report["roadmap_id"] != installed["roadmap_id"]
    assert report["selected_milestone_id"] == installed["milestone_id"]
    assert report["selected_direction_id"] == installed["direction_id"]
    assert union["active_lineage"] == installed
    assert set(config["schemas"]["iclr_controller_lineage_policy_fields"]) == set(policy)
    assert "controller_lineage_policy" in config["schemas"]["iclr_contract_union_optional_fields"]
    assert "controller_lineage_policy" not in config["schemas"]["iclr_contract_union_record_fields"]


def test_orchestrator_state_preserves_legacy_admission_when_policy_is_absent() -> None:
    config, _ = loaded_context()
    union = config["iclr_design_contract_union"]
    union.pop("controller_lineage_policy")
    assert set(config["schemas"]["iclr_contract_union_record_fields"]).issubset(union)
    report = build_orchestrator_state_report(config)
    assert report["status"] == "blocked"
    assert "orchestrator_lineage_family_mismatch" in blocker_codes(report)

    installed = union["active_lineage"]
    state = json.loads((REPO_ROOT / "orchestrator/state.json").read_text(encoding="utf-8"))
    state.update({key: installed[key] for key in ("roadmap_id", "roadmap_revision", "roadmap_dir")})
    report = build_orchestrator_state_report(config, state_payload=state)
    assert report["status"] == "pass"
    assert report["blocker_count"] == 0

    installed_revision = int(installed["roadmap_revision"].split("-")[1])
    state["roadmap_revision"] = f"rev-{installed_revision - 1:03d}"
    report = build_orchestrator_state_report(config, state_payload=state)
    assert "orchestrator_lineage_regression" in blocker_codes(report)


@pytest.mark.parametrize(
    "policy",
    [
        None,
        [],
        {},
        {"minimum_roadmap_revision": "rev-015"},
        {"roadmap_id": "example"},
        {"roadmap_id": "example", "minimum_roadmap_revision": "rev-015", "allow_any_family": True},
        {"roadmap_id": "", "minimum_roadmap_revision": "rev-015"},
        {"roadmap_id": " example ", "minimum_roadmap_revision": "rev-015"},
        {"roadmap_id": True, "minimum_roadmap_revision": "rev-015"},
        {"roadmap_id": "example", "minimum_roadmap_revision": "invalid"},
        {"roadmap_id": "example", "minimum_roadmap_revision": 15},
    ],
    ids=["null", "sequence", "empty", "missing_family", "missing_revision", "extra_field", "empty_family", "padded_family", "boolean_family", "invalid_revision", "numeric_revision"],
)
def test_orchestrator_state_rejects_malformed_controller_policy(policy: object) -> None:
    config, _ = loaded_context()
    config["iclr_design_contract_union"]["controller_lineage_policy"] = policy
    report = build_orchestrator_state_report(config)
    assert report["status"] == "blocked"
    assert "invalid_controller_lineage_policy" in blocker_codes(report)
    blocker = next(row for row in report["blockers"] if row["code"] == "invalid_controller_lineage_policy")
    assert blocker["context"] == {
        "policy_source": "iclr_design_contract_union.controller_lineage_policy",
        "policy_value": policy,
    }


def test_orchestrator_state_preserves_installed_revision_validation() -> None:
    config, _ = loaded_context()
    config["iclr_design_contract_union"]["active_lineage"]["roadmap_revision"] = "invalid"
    report = build_orchestrator_state_report(config)
    assert report["status"] == "blocked"
    assert "orchestrator_revision_invalid" in blocker_codes(report)


def test_g0_structural_consumer_accepts_controller_policy_without_mutating_state() -> None:
    state_path = REPO_ROOT / "orchestrator/state.json"
    before = state_path.read_bytes()
    reports = _structural_reports(CONFIG_PATH)
    assert set(reports) == {"config_stack", "design_coverage", "orchestrator_state", "contract_union"}
    assert {name: report["status"] for name, report in reports.items()} == {
        "config_stack": "pass",
        "design_coverage": "pass",
        "orchestrator_state": "pass",
        "contract_union": "pass",
    }
    assert all(report["blocker_count"] == 0 for report in reports.values())
    assert reports["contract_union"]["bound_audits"]["orchestrator_state"]["status"] == "pass"
    assert reports["orchestrator_state"]["state_mutated"] is False
    assert state_path.read_bytes() == before


def test_union_consumer_propagates_invalid_controller_policy() -> None:
    config, provenance = loaded_context()
    config["iclr_design_contract_union"]["controller_lineage_policy"] = None
    report = build_union_consistency_report(config, provenance, CONFIG_PATH)
    assert report["status"] == "blocked"
    assert "invalid_controller_lineage_policy" in blocker_codes(report)
    assert report["bound_audits"]["orchestrator_state"]["status"] == "blocked"
    assert report["bound_audits"]["orchestrator_state"]["blocker_count"] == 1
