from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from Libs.utils.data.controlled_world_tensorization import ControlledWorldAccessError
from Libs.utils.evaluation.operational_twin_training_feasibility import (
    BLOCKED_TERMINAL,
    OPEN_ROLES,
    QUALIFIED_TERMINAL,
    build_open_data_qualification_audit,
    build_training_feasibility_audit,
    load_tracked_open_manifest,
)


FAMILIES = (
    "electrophysiology_conductance_world",
    "calcium_synaptic_world",
    "molecular_pharmacological_causal_world",
)

ISOLATION_FIELDS = (
    "world_uid",
    "latent_world_uid",
    "latent_generator_lineage_uid",
    "seed_identity",
    "replay_identity",
    "observable_output_identity",
    "protocol_state_identity",
)


def _hex_identity(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def _config() -> dict[str, object]:
    return {
        "operational_twin": {
            "manifests": {
                role: f"Input/processed/controlled_neural_world/open_action_identification/{role}_manifest.json"
                for role in OPEN_ROLES
            },
            "access_policy": {
                "six_context_outcome_access": 0,
                "protected_holdout_access": 0,
                "test_outcome_access": 0,
                "tool_access": 0,
                "mcp_access": 0,
                "agent_trace_access": 0,
                "agent_access": 0,
                "memory_access": 0,
                "coevolution_access": 0,
            },
            "claim_boundary": {
                "training_authorized": False,
                "sufficient_training": False,
                "twin_qualification_pass": False,
                "formal_qualification": False,
                "agent_work_open": False,
                "test_manifest_open": False,
                "tool_gate_open": False,
                "mcp_gate_open": False,
                "agent_trace_gate_open": False,
                "memory_gate_open": False,
                "coevolution_gate_open": False,
            },
        }
    }


def _manifests(*, action_count: int, paired: bool, include_ood: bool) -> dict[str, dict[str, object]]:
    manifests: dict[str, dict[str, object]] = {}
    for role in OPEN_ROLES:
        worlds = []
        pairs = []
        for family_uid in FAMILIES:
            member_uids = []
            for action_index in range(action_count):
                pair_uid = f"{family_uid}_{role}_pair"
                world_uid = f"{family_uid}_{role}_{action_index}"
                row = {
                    "world_uid": world_uid,
                    "latent_world_uid": f"{family_uid}_{role}_latent",
                    "family_uid": family_uid,
                    "role": role,
                    "latent_generator_lineage_uid": f"{family_uid}_{role}_lineage",
                    "seed": 1,
                    "seed_identity": _hex_identity(family_uid, role, "seed"),
                    "action_sequence_identity": f"{family_uid}_action_{action_index}",
                    "action_program_uid": f"{family_uid}_program_{action_index}",
                    "action_binding_identity": f"{family_uid}_binding_{action_index}",
                    "replay_identity": f"{family_uid}_{role}_replay_{action_index}",
                    "observable_output_identity": _hex_identity(family_uid, role, "observable", action_index),
                    "protocol_state_identity": f"{family_uid}_{role}_state_{action_index}",
                    "shared_latent_config_identity": "3" * 64,
                    "biological_context_identity": "4" * 64,
                    "observation_policy_identity": "5" * 64,
                    "support_status": {"status": "unsupported" if include_ood and action_index == action_count - 1 else "supported"},
                    "counterfactual_pair_uid": pair_uid if paired else f"{world_uid}_singleton",
                }
                member_uids.append(world_uid)
                worlds.append(row)
            if paired:
                pairs.append({
                    "counterfactual_pair_uid": f"{family_uid}_{role}_pair",
                    "role": role,
                    "family_uid": family_uid,
                    "latent_world_uid": f"{family_uid}_{role}_latent",
                    "latent_generator_lineage_uid": f"{family_uid}_{role}_lineage",
                    "seed_identity": _hex_identity(family_uid, role, "seed"),
                    "member_world_uids": member_uids,
                    "action_program_uids": [f"{family_uid}_program_{index}" for index in range(action_count)],
                    "approved_pair": True,
                })
        support_cases = []
        if role == "development" and include_ood:
            for family_uid in FAMILIES:
                for status in ("supported", "out_of_distribution", "unsupported"):
                    support_cases.append({"family_uid": family_uid, "support_status": {"status": status}})
        manifests[role] = {
            "worlds": worlds,
            "counterfactual_pairs": pairs,
            "development_support_cases": support_cases,
            "family_counts": {
                family_uid: {"latent_worlds": 1}
                for family_uid in FAMILIES
            },
            "schema_version": "neuronsifter-controlled-world-open-action-manifest-v1",
            "content_identity": role,
        }
    return manifests


def _identities() -> dict[str, dict[str, object]]:
    return {
        role: {
            "path": f"Input/processed/controlled_neural_world/open_action_identification/{role}_manifest.json",
            "git_object_id": role,
        }
        for role in OPEN_ROLES
    }


def _receipt(manifests: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "neuronsifter-controlled-world-open-package-receipt-v1",
        "receipt_uid": "receipt",
        "terminal_decision": "OPEN_ACTION_IDENTIFICATION_PACKAGE_CONSTRUCTED",
        "open_data_qualification": "pending",
        "training_authorized": False,
        "access_counters": {
            "six_context_outcome_access": 0,
            "protected_holdout_access": 0,
            "test_outcome_access": 0,
            "tool_access": 0,
            "mcp_access": 0,
            "agent_trace_access": 0,
            "agent_access": 0,
            "memory_access": 0,
            "coevolution_access": 0,
        },
        "closed_downstream_flags": _config()["operational_twin"]["claim_boundary"],
        "manifests": {role: _identities()[role]["path"] for role in OPEN_ROLES},
        "manifest_content_identities": {role: manifests[role]["content_identity"] for role in OPEN_ROLES},
    }


def test_single_action_supported_only_manifests_fail_closed() -> None:
    audit = build_training_feasibility_audit(
        _config(),
        _manifests(action_count=1, paired=False, include_ood=False),
        _identities(),
        authoritative_date="2026-09-04",
    )
    assert audit["status"] == "BLOCKED"
    assert audit["terminal_decision"] == BLOCKED_TERMINAL
    assert {row["code"] for row in audit["blockers"]} == {
        "training_action_identifiability_absent",
        "development_action_value_estimand_unidentified",
        "development_ood_behavior_unobserved",
    }
    assert audit["claim_flags"]["sufficient_training"] is False
    assert all(value == 0 for value in audit["sealed_material_access"].values())


def test_action_varied_paired_support_and_ood_manifests_qualify_when_replay_valid(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manifests = _manifests(action_count=2, paired=True, include_ood=True)
    monkeypatch.setattr(
        "Libs.utils.evaluation.operational_twin_training_feasibility.validate_open_manifests",
        lambda *unused_args, **unused_kwargs: [],
    )
    audit = build_open_data_qualification_audit(
        _config(),
        manifests,
        _identities(),
        _receipt(manifests),
        {"path": "Input/processed/controlled_neural_world/open_action_identification/package_receipt.json", "sha256": "a" * 64},
        authoritative_date="2026-09-04",
        repository_root=tmp_path,
    )
    assert audit["status"] == "PASS"
    assert audit["terminal_decision"] == QUALIFIED_TERMINAL
    assert audit["training_authorized"] is True
    assert audit["role_lineage_isolation"] == {
        "checked_fields": list(ISOLATION_FIELDS),
        "action_result_identity_field": "protocol_state_identity",
        "status": "PASS",
        "overlaps": [],
    }
    assert audit["closed_downstream_gates"]["sufficient_training"] is False
    assert audit["closed_downstream_gates"]["agent_work_open"] is False


def test_qualification_negative_for_missing_development_ood(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manifests = _manifests(action_count=2, paired=True, include_ood=False)
    monkeypatch.setattr(
        "Libs.utils.evaluation.operational_twin_training_feasibility.validate_open_manifests",
        lambda *unused_args, **unused_kwargs: [],
    )
    audit = build_open_data_qualification_audit(
        _config(),
        manifests,
        _identities(),
        _receipt(manifests),
        {"path": "Input/processed/controlled_neural_world/open_action_identification/package_receipt.json", "sha256": "a" * 64},
        authoritative_date="2026-09-04",
        repository_root=tmp_path,
    )
    assert audit["status"] == "BLOCKED"
    assert audit["terminal_decision"] == "OPEN_TWIN_DEVELOPMENT_DATA_NOT_QUALIFIED_SUPPORT_COVERAGE"
    assert audit["training_authorized"] is False


@pytest.mark.parametrize("field", ISOLATION_FIELDS)
def test_cross_role_identity_overlap_fails_qualification(monkeypatch: pytest.MonkeyPatch, tmp_path, field: str) -> None:
    manifests = _manifests(action_count=2, paired=True, include_ood=True)
    overlapping_value = manifests["training"]["worlds"][0][field]
    manifests["development"]["worlds"][0][field] = overlapping_value
    monkeypatch.setattr(
        "Libs.utils.evaluation.operational_twin_training_feasibility.validate_open_manifests",
        lambda *unused_args, **unused_kwargs: [],
    )
    audit = build_open_data_qualification_audit(
        _config(),
        manifests,
        _identities(),
        _receipt(manifests),
        {"path": "Input/processed/controlled_neural_world/open_action_identification/package_receipt.json", "sha256": "a" * 64},
        authoritative_date="2026-09-04",
        repository_root=tmp_path,
    )
    assert audit["status"] == "BLOCKED"
    assert audit["terminal_decision"] == "OPEN_TWIN_DEVELOPMENT_DATA_NOT_QUALIFIED_LINEAGE_LEAKAGE"
    assert audit["training_authorized"] is False
    assert audit["role_lineage_isolation"]["status"] == "BLOCKED"
    assert audit["role_lineage_isolation"]["action_result_identity_field"] == "protocol_state_identity"
    assert audit["role_lineage_isolation"]["checked_fields"] == list(ISOLATION_FIELDS)
    assert {
        (row["field"], row["left_role"], row["right_role"])
        for row in audit["role_lineage_isolation"]["overlaps"]
    } == {(field, "training", "development")}


def test_unique_pair_labels_do_not_count_as_counterfactual_pairs() -> None:
    manifests = _manifests(action_count=2, paired=True, include_ood=True)
    for role in OPEN_ROLES:
        for row in manifests[role]["worlds"]:
            row["counterfactual_pair_uid"] = row["world_uid"]
    audit = build_training_feasibility_audit(
        _config(),
        manifests,
        _identities(),
        authoritative_date="2026-09-04",
    )
    assert audit["status"] == "BLOCKED"
    assert "development_action_value_estimand_unidentified" in {
        row["code"] for row in audit["blockers"]
    }


def test_nonzero_access_policy_is_rejected() -> None:
    config = _config()
    config["operational_twin"]["access_policy"]["protected_holdout_access"] = 1
    with pytest.raises(ControlledWorldAccessError, match="exactly zero"):
        build_training_feasibility_audit(
            config,
            _manifests(action_count=2, paired=True, include_ood=True),
            _identities(),
            authoritative_date="2026-09-04",
        )


def test_open_claim_flag_is_rejected() -> None:
    config = _config()
    config["operational_twin"]["claim_boundary"]["agent_work_open"] = True
    with pytest.raises(ControlledWorldAccessError, match="downstream-open flags"):
        build_training_feasibility_audit(
            config,
            _manifests(action_count=2, paired=True, include_ood=True),
            _identities(),
            authoritative_date="2026-09-04",
        )


def test_missing_claim_boundary_is_rejected() -> None:
    config = _config()
    del config["operational_twin"]["claim_boundary"]
    with pytest.raises(ControlledWorldAccessError, match="downstream-open flags"):
        build_training_feasibility_audit(
            config,
            _manifests(action_count=2, paired=True, include_ood=True),
            _identities(),
            authoritative_date="2026-09-04",
        )


def test_tracked_loader_rejects_test_or_qualification_path_before_file_access(tmp_path) -> None:
    for path in (
        "Input/processed/controlled_neural_world/benchmark/test_manifest.json",
        "Input/processed/controlled_neural_world/benchmark/twin_qualification_manifest.json",
    ):
        with pytest.raises(ControlledWorldAccessError):
            load_tracked_open_manifest(tmp_path, path, "development")


def test_manifest_identity_must_match_canonical_config_path() -> None:
    identities = deepcopy(_identities())
    identities["development"]["path"] = "substitute/development_manifest.json"
    with pytest.raises(ControlledWorldAccessError, match="canonical config-stack path"):
        build_training_feasibility_audit(
            _config(),
            _manifests(action_count=2, paired=True, include_ood=True),
            identities,
            authoritative_date="2026-09-04",
        )
