from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.g0_integrity import (
    IMMUTABLE_BLOCKER_IDS,
    _audit_policy,
    _artifact_ownership_check,
    _disposition,
    _expected_dedicated_schemas,
    build_g0_integrity_audit,
)
from Libs.utils.data.raw_data_registry import load_config


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"


def test_g0_integrity_binds_exact_original_blockers_and_preserves_unreconciled_numeric_lineage() -> None:
    report = build_g0_integrity_audit(CONFIG_PATH)

    assert tuple(report["immutable_blocker_ids"]) == IMMUTABLE_BLOCKER_IDS
    assert report["original_blocker_count"] == 4
    assert report["status"] == "blocked"
    assert report["remaining_original_blocker_count"] == 1
    assert report["remaining_original_blocker_ids"] == ["G0-NUMERIC-LINEAGE-DRIFT"]
    dispositions = {row["blocker_id"]: row for row in report["blocker_dispositions"]}
    assert dispositions["G0-NAMING-LEGACY-RESIDUALS"]["status"] == "closed"
    assert dispositions["G0-PUBLIC-MAINLINE-CONTROLLED-INCLUDE"]["status"] == "closed"
    assert dispositions["G0-CLAIM-GATED-WORDING"]["status"] == "closed"
    numeric = dispositions["G0-NUMERIC-LINEAGE-DRIFT"]
    assert numeric["status"] == "blocked"
    assert numeric["metrics"]["unresolved_conflict_ids"] == ["historical_source_manifest_hash_missing"]
    assert numeric["metrics"]["independent_manifest_verified"] is False
    assert report["gate_recommendation"] == {
        "current_stage": "g0_blocked",
        "g0_status": "blocked",
        "phase1_open": False,
        "implementation_enabled": False,
        "next_owner": None,
        "next_direction": None,
    }
    assert report["claim_mapping"] == [
        {
            "claim_id": "C1",
            "meaning": "public_foundation_state_transfer",
            "status": "requires_experiment_result",
            "gate_status": "contract_frozen",
        },
        {
            "claim_id": "C2",
            "meaning": "structured_stochastic_world_action",
            "status": "requires_experiment_result",
            "gate_status": "contract_frozen",
        },
        {
            "claim_id": "C3",
            "meaning": "cost_aware_multi_tool_acquisition",
            "status": "requires_experiment_result",
            "gate_status": "contract_frozen",
        },
        {
            "claim_id": "C4",
            "meaning": "tri_gated_reliable_screening",
            "status": "requires_experiment_result",
            "gate_status": "contract_frozen",
        },
        {
            "claim_id": "C5",
            "meaning": "scale_aware_throughput_support",
            "status": "requires_experiment_result",
            "gate_status": "contract_frozen",
        },
    ]
    resolution = report["mandatory_input_resolutions"]["contract_union_integrity_audit"]
    assert resolution["status"] == "pass"
    assert resolution["required_input_path"] == "Output/audits/contract_union_integrity_audit.json"
    assert resolution["canonical_replacement_path"] == "Output/audits/iclr_contract_union_consistency.json"
    assert resolution["owner_validator_id"] == "contract-union-validator"
    assert resolution["owner_binding_count"] == 1
    assert resolution["legacy_artifact_exists"] is False
    structural_output = json.loads(
        (REPO_ROOT / "Output/audits/iclr_contract_union_consistency.json").read_text(encoding="utf-8")
    )
    assert structural_output["producer_command"] == (
        "python Libs/scripts/validators/validate_iclr_design_contract_union.py "
        "--config "
        "Libs/configs/config_stack.yaml"
    )


def test_numeric_disposition_rejects_unresolved_reconstructed_projection() -> None:
    numeric = {
        "status": "blocked",
        "unresolved_conflict_count": 1,
        "unresolved_conflict_ids": ["historical_reconstructed_projection_unverified"],
        "metrics": [{"metric_name": "raw_materialized_full_microenvironment_trajectory_count"}],
    }
    disposition = _disposition(
        "G0-NUMERIC-LINEAGE-DRIFT",
        deepcopy(numeric),
        "Output/audits/trajectory_numeric_lineage_audit.json",
    )
    assert disposition["status"] == "blocked"
    assert disposition["metrics"]["unresolved_conflict_ids"] == ["historical_reconstructed_projection_unverified"]


def test_numeric_disposition_requires_a_verified_independent_manifest() -> None:
    numeric = {
        "status": "pass",
        "unresolved_conflict_count": 0,
        "unresolved_conflict_ids": [],
        "historical_snapshot": {"independent_manifest": {"verified": False}},
        "metrics": [],
    }
    disposition = _disposition(
        "G0-NUMERIC-LINEAGE-DRIFT",
        numeric,
        "Output/audits/trajectory_numeric_lineage_audit.json",
    )
    assert disposition["status"] == "blocked"


def test_artifact_ownership_rejects_duplicate_validator_outputs() -> None:
    union = {
        "validator_registry": [
            {"validator_id": "one", "output_path": "Output/audits/shared.json"},
            {"validator_id": "two", "output_path": "Output/audits/shared.json"},
        ]
    }
    report = _artifact_ownership_check(union)
    assert report["status"] == "blocked"
    assert report["duplicate_artifact_ownership_count"] == 1


def test_dedicated_artifact_schemas_are_owned_by_the_union_contract() -> None:
    _, audit_contract = _audit_policy(load_config(CONFIG_PATH))

    assert _expected_dedicated_schemas(audit_contract) == {
        "naming": "neuronsifter-g0-naming-remediation-v1",
        "public_mainline": "neuronsifter-g0-public-mainline-remediation-v1",
        "numeric_lineage": "neuronsifter-trajectory-numeric-lineage-audit-v1",
        "claim_wording": "neuronsifter-claim-gated-wording-audit-v1",
    }
