"""Focused contract tests for the Round-18 literature replay boundary."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from Libs.scripts.builders.build_dt_round18_literature_replay import (
    _agent_visible_episode,
    _episode_record,
    _outcome_record,
    _predict_episode,
    _source_record,
    _validate_config,
    build_round18_pilot,
)
from Libs.scripts.validators.validate_dt_round18_literature_replay import (
    _config_blockers,
    _cpu_only_placement_valid,
    _frozen_gap_uid_set_blockers,
    gate_state_for_blockers,
)
from Libs.utils.data.literature_replay_contracts import (
    CLAIM_SCOPE,
    canonical_sha256,
    canonical_sequential_event_uid,
    compute_direction_metrics,
    select_earliest_public_timestamp,
    validate_abstention_semantics,
    validate_bundle_isolation,
    validate_claim_boundaries,
    validate_execution_isolation,
    validate_failed_retry_preservation,
    validate_frozen_gap_isolation,
    validate_label_provenance,
    validate_lineage_acyclic,
    validate_memory_risk_disclosure,
    validate_outcome_target,
    validate_public_source,
    validate_public_source_boundary,
    validate_scitil_runtime_trace,
    validate_sequential_reveal,
    validate_split_leakage,
    validate_temporal_ordering,
    validate_unknown_uncertainty,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv


REPO_ROOT = Path(__file__).resolve().parents[2]

MANDATORY_NEGATIVE_TEST_INVENTORY = {
    "controlled-access source resolution": "test_controlled_source_route_fails_closed",
    "future evidence in pre-cutoff bundle": "test_future_pre_cutoff_evidence_fails_closed",
    "held-out bundle readable by Agent process": "test_bundle_and_execution_isolation_fail_closed",
    "ambiguous publication timestamp used as strong episode": (
        "test_ambiguous_timestamp_strong_admission_fails_closed"
    ),
    "narrative claim used as numeric endpoint": "test_narrative_numeric_reconstruction_is_rejected",
    "publication family split across train and held-out": (
        "test_split_leakage_and_duplicate_accession_fail_closed"
    ),
    "duplicate accession treated as independent": (
        "test_split_leakage_and_duplicate_accession_fail_closed"
    ),
    "unknown comparator": "test_unknown_comparator_and_action_identity_fail_closed",
    "unknown action identity": "test_unknown_comparator_and_action_identity_fail_closed",
    "model prediction used to close empirical gap": (
        "test_claim_language_and_frozen_gap_boundaries_fail_closed"
    ),
    "forced answer when evidence is insufficient": "test_abstention_forced_answer_is_rejected",
    "unknown uncertainty replaced by zero": "test_unknown_uncertainty_zero_fails_closed",
    "conflict removed": "test_conflict_removal_is_rejected_by_abstention_schema",
    "causal effect language": "test_claim_language_and_frozen_gap_boundaries_fail_closed",
    "clinical recommendation language": (
        "test_claim_language_and_frozen_gap_boundaries_fail_closed"
    ),
    "memory-risk track reported as contamination-free": (
        "test_memory_risk_cannot_claim_contamination_free"
    ),
    "failed episode overwritten by retry": "test_lineage_and_retry_preserve_failed_attempts",
}


def _hash(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def _parent_artifact_root(relative_path: Path) -> Path:
    if (REPO_ROOT / relative_path).is_file():
        return REPO_ROOT
    git_pointer = (REPO_ROOT / ".git").read_text(encoding="utf-8").strip()
    git_dir = Path(git_pointer.removeprefix("gitdir: ")).resolve()
    common_dir = (
        git_dir / (git_dir / "commondir").read_text(encoding="utf-8").strip()
    ).resolve()
    parent_root = common_dir.parent
    assert (parent_root / relative_path).is_file()
    return parent_root


def _pilot_objects() -> tuple[dict, dict, list[dict]]:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    round18 = config["round18_literature_replay"]
    policy = round18["policies"]["temporal_cutoff"]["policy_version"]
    sources = {
        row["source_uid"]: _source_record(row, policy)
        for row in round18["pilot"]["sources"]
    }
    outcomes = {
        row["outcome_key"]: _outcome_record(row, sources)
        for row in round18["pilot"]["outcomes"]
    }
    episodes = [
        _episode_record(row, sources=sources, outcomes=outcomes)
        for row in round18["pilot"]["episodes"]
    ]
    return sources, outcomes, episodes


def test_round18_section23_mandatory_negative_inventory_is_complete() -> None:
    expected_cases = {
        "controlled-access source resolution",
        "future evidence in pre-cutoff bundle",
        "held-out bundle readable by Agent process",
        "ambiguous publication timestamp used as strong episode",
        "narrative claim used as numeric endpoint",
        "publication family split across train and held-out",
        "duplicate accession treated as independent",
        "unknown comparator",
        "unknown action identity",
        "model prediction used to close empirical gap",
        "forced answer when evidence is insufficient",
        "unknown uncertainty replaced by zero",
        "conflict removed",
        "causal effect language",
        "clinical recommendation language",
        "memory-risk track reported as contamination-free",
        "failed episode overwritten by retry",
    }
    assert set(MANDATORY_NEGATIVE_TEST_INVENTORY) == expected_cases
    assert len(MANDATORY_NEGATIVE_TEST_INVENTORY) == 17
    missing_tests = {
        test_name
        for test_name in MANDATORY_NEGATIVE_TEST_INVENTORY.values()
        if not callable(globals().get(test_name))
    }
    assert missing_tests == set()


def test_round18_config_and_pilot_have_all_tracks() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    assert _validate_config(config) == []
    tracks = {
        row["track"] for row in config["round18_literature_replay"]["pilot"]["episodes"]
    }
    assert tracks == {
        "endpoint_direction_prediction",
        "intervention_ranking",
        "mechanism_discrimination",
        "information_action_selection",
        "sequential_hypothesis_replay",
        "abstention_boundary_compliance",
    }


def test_round18_pilot_frozen_gap_map_binds_actual_round14_ledger(
    tmp_path: Path,
) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    round18 = config["round18_literature_replay"]
    ledger_relative = Path(round18["parent_artifacts"]["round14_frozen_gap_ledger"])
    assert ledger_relative.as_posix() == (
        "Output/audits/dt_round14_frozen_gap_resolution_ledger.tsv"
    )
    parent_root = _parent_artifact_root(ledger_relative)
    ledger_rows = read_tsv(parent_root / ledger_relative)
    expected_gap_uids = {row["gap_uid"] for row in ledger_rows}
    assert len(ledger_rows) == 20
    assert len(expected_gap_uids) == 20

    build_round18_pilot(
        config=config,
        repository_root=REPO_ROOT,
        parent_root=parent_root,
        output_root=tmp_path,
        created_at="2026-07-30T00:00:00Z",
    )
    manifest_path = tmp_path / round18["outputs"]["episode_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gap_map = manifest["frozen_gap_map"]
    materialized_gap_uids = {row["gap_uid"] for row in gap_map}
    allowed_impacts = set(round18["policies"]["frozen_gap"]["allowed_impacts"])

    assert materialized_gap_uids == expected_gap_uids
    assert all(not uid.startswith("frozen_gap_") for uid in materialized_gap_uids)
    assert all(row["resolved"] is False for row in gap_map)
    assert all(row["round18_impact"] in allowed_impacts for row in gap_map)
    assert validate_frozen_gap_isolation(gap_map, expected_count=20) == []


def test_round18_system_and_ablation_contracts_are_behaviorally_distinct() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    round18 = config["round18_literature_replay"]
    contracts = round18["system_contracts"]
    policy_fields = (
        "confidence_scale",
        "skeptical_conflict_policy",
        "ranking_policy",
        "information_action_policy",
        "falsifier_enabled",
        "model_uncertainty_known",
    )
    policies = {
        tuple(contract[field] for field in policy_fields)
        for contract in contracts.values()
    }
    assert set(contracts) == set(round18["systems"])
    assert len(policies) == len(round18["systems"])
    agent_only = contracts["agent_only_offline_proposal_ablation"]
    assert agent_only["direct_tool_execution"] is False
    assert agent_only["direct_state_update"] is False
    assert agent_only["gate_authority"] is False

    sources, _, episodes = _pilot_objects()
    information_episode = _agent_visible_episode(
        next(row for row in episodes if row["track"] == "information_action_selection"),
        sources,
    )
    twin_decision = _predict_episode(
        "twin_only_without_agents",
        information_episode,
        contracts["twin_only_without_agents"],
    )
    full_decision = _predict_episode(
        "full_governed_scitil",
        information_episode,
        contracts["full_governed_scitil"],
    )
    assert twin_decision["selected_information_action"] is None
    assert full_decision["selected_information_action"] is not None
    assert twin_decision["falsifier_invoked"] is False
    assert full_decision["falsifier_invoked"] is False
    mechanism_episode = _agent_visible_episode(
        next(row for row in episodes if row["track"] == "mechanism_discrimination"),
        sources,
    )
    mechanism_decision = _predict_episode(
        "full_governed_scitil",
        mechanism_episode,
        contracts["full_governed_scitil"],
    )
    assert mechanism_decision["falsifier_invoked"] is True
    assert "scitil_runtime_trace" in mechanism_decision


def test_formal_validator_reuses_builder_system_contract_validation() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    invalid = deepcopy(config)
    invalid["round18_literature_replay"]["system_contracts"]["full_governed_scitil"][
        "information_action_policy"
    ] = "unbounded_tool_access"

    codes = {row["code"] for row in _config_blockers(invalid)}
    assert "round18_system_contract_invalid" in codes


def test_full_scitil_trace_cannot_update_state_or_write_a_gate() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    sources, _, episodes = _pilot_objects()
    episode = _agent_visible_episode(
        next(row for row in episodes if row["track"] == "mechanism_discrimination"),
        sources,
    )
    trace = deepcopy(
        _predict_episode(
            "full_governed_scitil",
            episode,
            config["round18_literature_replay"]["system_contracts"]["full_governed_scitil"],
        )["scitil_runtime_trace"]
    )
    trace["agent_direct_state_update"] = True
    codes = {item["code"] for item in validate_scitil_runtime_trace(trace)}
    assert "scitil_runtime_boundary_violation" in codes


def test_formal_validator_requires_exact_parent_frozen_gap_uid_set() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    ledger_relative = Path(
        config["round18_literature_replay"]["parent_artifacts"][
            "round14_frozen_gap_ledger"
        ]
    )
    parent_rows = read_tsv(_parent_artifact_root(ledger_relative) / ledger_relative)
    mapped_rows = [
        {
            "gap_uid": row["gap_uid"],
            "round18_impact": "benchmark_evidence_only",
            "resolved": False,
        }
        for row in parent_rows
    ]
    mapped_rows[0]["gap_uid"] = "unrelated_gap_uid"

    codes = {
        row["code"] for row in _frozen_gap_uid_set_blockers(mapped_rows, parent_rows)
    }
    assert codes == {"frozen_gap_uid_set_mismatch"}


def test_cpu_only_execution_placement_requires_zero_gpu_allocation() -> None:
    assert _cpu_only_placement_valid(
        {"node": "cpu03", "partition": "gpu", "gpu_allocation_count": 0}
    )
    assert _cpu_only_placement_valid(
        {"node": "cpu04", "partition": "cpu", "gpu_allocation_count": 0}
    )
    assert not _cpu_only_placement_valid(
        {"node": "cpu03", "partition": "gpu", "gpu_allocation_count": 1}
    )
    assert not _cpu_only_placement_valid(
        {"node": "cpu03", "partition": "cpu", "gpu_allocation_count": 0}
    )


def test_controlled_source_route_fails_closed() -> None:
    sources, _, _ = _pilot_objects()
    bad = deepcopy(next(iter(sources.values())))
    bad["controlled_access_required"] = True
    assert any(
        item["code"] == "controlled_source_resolution"
        for item in validate_public_source(bad)
    )


def test_unknown_access_status_fails_closed() -> None:
    sources, _, _ = _pilot_objects()
    bad = deepcopy(next(iter(sources.values())))
    bad["access_status"] = "unknown"
    assert any(
        item["code"] == "nonpublic_or_unknown_source"
        for item in validate_public_source(bad)
    )


def test_public_timestamp_records_require_versioned_verification() -> None:
    sources, _, _ = _pilot_objects()
    missing_verification = deepcopy(next(iter(sources.values())))
    missing_verification["public_timestamps"][0].pop("verified", None)
    codes = {item["code"] for item in validate_public_source(missing_verification)}
    assert "public_timestamp_verification_missing" in codes

    invalid_type = deepcopy(next(iter(sources.values())))
    invalid_type["public_timestamps"][0]["timestamp_type"] = "citation_index_date"
    codes = {item["code"] for item in validate_public_source(invalid_type)}
    assert "public_timestamp_type_invalid" in codes


def test_unverified_nonadmissible_timestamp_is_retained_but_not_selected() -> None:
    sources, _, _ = _pilot_objects()
    source = deepcopy(next(iter(sources.values())))
    source["public_timestamps"].insert(
        0,
        {
            "timestamp_type": "publisher_publication_date",
            "timestamp": "2020-01-01T00:00:00Z",
            "version": "unverified_archive_hint",
            "verified": False,
            "admissible": False,
        },
    )
    assert validate_public_source(source) == []
    selected = select_earliest_public_timestamp(source["public_timestamps"])
    assert selected["timestamp"] != "2020-01-01T00:00:00Z"


def test_policy_excluded_source_material_fails_closed() -> None:
    sources, _, _ = _pilot_objects()
    bad = deepcopy(next(iter(sources.values())))
    bad["source_identity"] = "untraceable predatory source"
    assert any(
        item["code"] == "policy_excluded_source"
        for item in validate_public_source(bad)
    )


def test_publisher_name_is_not_a_public_source_exclusion() -> None:
    sources, _, _ = _pilot_objects()
    source = deepcopy(next(iter(sources.values())))
    source["source_identity"] = "public MDPI repository mirror"
    assert validate_public_source(source) == []


def test_future_pre_cutoff_evidence_fails_closed() -> None:
    sources, _, _ = _pilot_objects()
    pre = deepcopy(next(iter(sources.values())))
    pre["selected_public_timestamp"] = "2030-01-01T00:00:00Z"
    blockers = validate_temporal_ordering(
        "2022-01-01T00:00:00Z", pre_cutoff_sources=[pre], held_out_sources=[]
    )
    assert any(
        item["code"] == "future_evidence_in_pre_cutoff_bundle" for item in blockers
    )


def test_ambiguous_timestamp_strong_admission_fails_closed() -> None:
    sources, _, _ = _pilot_objects()
    pre = deepcopy(next(iter(sources.values())))
    pre["selected_public_timestamp"] = "2021-01-01T00:00:00Z"
    pre["temporal_ordering_status"] = "ambiguous"
    pre["strong_admission"] = True
    blockers = validate_temporal_ordering(
        "2022-01-01T00:00:00Z", pre_cutoff_sources=[pre], held_out_sources=[]
    )
    assert any(
        item["code"] == "ambiguous_timestamp_strong_admission" for item in blockers
    )


def test_bundle_and_execution_isolation_fail_closed() -> None:
    visible = {
        "bundle_role": "agent_visible_pre_cutoff_bundle",
        "bundle_uid": "a",
        "content_hash": _hash("a"),
        "mount_path": "/tmp/a",
        "read_permissions": {"agent": True, "evaluator": True},
        "artifact_hashes": [_hash("a")],
    }
    held = {
        "bundle_role": "evaluator_only_held_out_bundle",
        "bundle_uid": "b",
        "content_hash": _hash("b"),
        "mount_path": "/tmp/b",
        "read_permissions": {"agent": True, "evaluator": True},
        "artifact_hashes": [_hash("b")],
    }
    assert any(
        item["code"] == "held_out_bundle_readable_by_agent"
        for item in validate_bundle_isolation(visible, held)
    )
    execution = {
        "agent_process_identity": "agent",
        "evaluator_process_identity": "evaluator",
        "held_out_bundle_readable_by_agent": True,
        "agent_direct_tool_execution": True,
        "access_attempts": [],
    }
    codes = {item["code"] for item in validate_execution_isolation(execution)}
    assert {
        "held_out_bundle_readable_by_agent",
        "agent_execution_isolation_violation",
    } <= codes


def test_narrative_numeric_reconstruction_is_rejected() -> None:
    label = {
        "label_uid": "label",
        "label_class": "data_derived_label",
        "source_refs": ["source"],
        "label_extraction_method": "narrative reconstruction",
        "target_type": "numeric_endpoint",
        "numeric_value": 1.0,
        "adjudication_status": "dual_adjudicated",
    }
    assert any(
        item["code"] == "narrative_claim_as_numeric_truth"
        for item in validate_label_provenance(label)
    )


def test_unknown_comparator_and_action_identity_fail_closed() -> None:
    target = {
        "target_uid": "ranking-target",
        "target_type": "candidate_ranking",
        "measurement": "endpoint direction",
        "unit": "directional category",
        "direction_semantics": "larger rank is preferred",
        "timepoint_or_horizon": "12 weeks",
        "comparator": "unknown",
        "uncertainty": {
            "status": "quantified",
            "value": 0.2,
            "unknown_components": [],
        },
        "source": "public-source",
        "label_extraction_method": "data table extraction",
        "adjudication_status": "dual_adjudicated",
        "action_identity": "unknown",
    }
    codes = {item["code"] for item in validate_outcome_target(target)}
    assert {"outcome_comparator_unknown", "outcome_action_identity_unresolved"} <= codes


def test_public_boundary_negative_controls_are_not_credentials() -> None:
    boundary = {
        "schema_version": "neuronsifter-public-source-boundary-v1",
        "strict_public_only_flag": True,
        "controlled_access_required": False,
        "resolved_controlled_source_count": 0,
        "unknown_access_status_count": 0,
        "negative_controls": {
            "credentialed_source_rejected": True,
            "adni_route_rejected": True,
            "aibl_route_rejected": True,
            "institutional_session_dependency_rejected": True,
            "controlled_clinical_repository_rejected": True,
            "unknown_access_status_rejected": True,
        },
    }
    assert validate_public_source_boundary(boundary) == []


def test_split_leakage_and_duplicate_accession_fail_closed() -> None:
    base = {
        "episode_uid": "e1",
        "content_hash": _hash("same"),
        "independence_groups": {
            field: field + "_1"
            for field in (
                "publication_family",
                "preprint_to_journal_family",
                "accession_family",
                "study_cohort",
                "biological_unit_family",
                "compound_scaffold",
                "target_program",
                "regimen",
                "disease_context",
                "mechanism_family",
                "simulator_model_family",
                "tool_family",
            )
        },
        "leakage_controls": {"frozen": True},
    }
    left = deepcopy(base)
    left["independence_groups"]["partition"] = "development"
    right = deepcopy(base)
    right["episode_uid"] = "e2"
    right["independence_groups"]["partition"] = "final_held_out"
    blockers = validate_split_leakage([left, right])
    codes = {item["code"] for item in blockers}
    assert any(
        item["code"] == "independence_group_split_leakage"
        and item["dimension"] == "publication_family"
        for item in blockers
    )
    assert "duplicate_accession_across_partitions" in codes


def test_memory_risk_cannot_claim_contamination_free() -> None:
    risk = {
        "tracks": {
            "retrieval_isolated_replay": {
                "episode_count": 1,
                "residual_risk": "moderate",
                "risk_disclosure": "risk remains",
            },
            "post_model_cutoff_replay": {
                "episode_count": 1,
                "residual_risk": "moderate",
                "risk_disclosure": "risk remains",
            },
            "memory_risk_stress_track": {
                "episode_count": 1,
                "residual_risk": "high",
                "risk_disclosure": "risk remains",
            },
        },
        "mitigations": {"opaque_episode_ids": True},
        "residual_risk_disclosed": True,
        "contamination_free_claim": True,
    }
    assert any(
        item["code"] == "memory_track_claimed_contamination_free"
        for item in validate_memory_risk_disclosure(risk)
    )


def test_sequential_reveal_mutation_and_uid_are_rejected() -> None:
    event = {
        "schema_version": "neuronsifter-sequential-reveal-event-v1",
        "event_uid": "pending",
        "episode_uid": "episode",
        "step": "T0",
        "sequence_index": 0,
        "timestamp": "2026-07-30T00:00:00Z",
        "input_hashes": {"bundle": _hash("bundle")},
        "output_hashes": {"episode": _hash("episode")},
        "held_out_revealed": False,
        "held_out_accessed": False,
        "immutable": True,
        "preserved": True,
        "prior_event_hash": None,
        "process_identity": "agent",
        "status": "completed",
    }
    event["event_hash"] = canonical_sha256(
        {
            key: value
            for key, value in event.items()
            if key not in {"event_uid", "event_hash"}
        }
    )
    event["event_uid"] = canonical_sequential_event_uid(event)
    events = [deepcopy(event) for _ in range(8)]
    for index, row in enumerate(events):
        row["sequence_index"] = index
        row["step"] = f"T{index}"
        row["prior_event_hash"] = events[index - 1]["event_hash"] if index else None
        row["event_hash"] = canonical_sha256(
            {
                key: value
                for key, value in row.items()
                if key not in {"event_uid", "event_hash"}
            }
        )
        row["event_uid"] = canonical_sequential_event_uid(row)
    events[2]["held_out_accessed"] = True
    assert any(
        item["code"] == "future_outcome_access_before_t5"
        for item in validate_sequential_reveal(events)
    )


def test_abstention_forced_answer_is_rejected() -> None:
    row = {
        "prediction_uid": "p",
        "evidence_status": "insufficient_information",
        "answered": True,
        "predicted_label": "decrease",
        "forced_answer": True,
        "uncertainty": {
            "status": "unknown",
            "value": None,
            "unknown_components": ["support_domain"],
        },
    }
    codes = {item["code"] for item in validate_abstention_semantics(row)}
    assert {
        "forced_answer_prompt_not_rejected",
        "unsupported_episode_answered",
    } <= codes


def test_unknown_uncertainty_zero_fails_closed() -> None:
    uncertainty = {
        "status": "unknown",
        "value": 0,
        "unknown_components": ["model"],
    }
    assert any(
        item["code"] == "unknown_uncertainty_replaced_by_zero"
        for item in validate_unknown_uncertainty(uncertainty, object_uid="prediction")
    )


def test_conflict_removal_is_rejected_by_abstention_schema() -> None:
    schema = json.loads(
        (REPO_ROOT / "Libs/configs/schema/benchmark_abstention.schema.json").read_text(
            encoding="utf-8"
        )
    )
    artifact = {
        "schema_version": "neuronsifter-benchmark-abstention-v1",
        "abstention_uid": "abstention-conflict",
        "episode_uid": "episode-conflict",
        "system_uid": "full_governed_scitil",
        "abstention_required": True,
        "abstained": True,
        "reason": "conflicting_evidence",
        "support_domain_status": "in_domain",
        "evidence_sufficiency": "conflicting",
        "uncertainty": {
            "status": "unknown",
            "value": None,
            "unknown_components": ["mechanism_disagreement"],
        },
        "unknown_components_preserved": True,
        "conflicts_preserved": True,
        "forced_answer_refused": True,
        "unsupported_answer_emitted": False,
        **CLAIM_SCOPE,
        "content_hash": _hash("abstention-conflict"),
        "status": "abstained_correctly",
    }
    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors(artifact)) == []

    artifact["conflicts_preserved"] = False
    errors = list(validator.iter_errors(artifact))
    assert any(
        list(error.absolute_path) == ["conflicts_preserved"]
        and error.validator == "const"
        for error in errors
    )


def test_claim_language_and_frozen_gap_boundaries_fail_closed() -> None:
    causal_claim = {
        **CLAIM_SCOPE,
        "causal_effect_claim": True,
        "claim": "causal effect established",
    }
    causal_codes = {item["code"] for item in validate_claim_boundaries(causal_claim)}
    assert {"causal_language_violation", "prohibited_claim_language"} <= causal_codes

    clinical_claim = {
        **CLAIM_SCOPE,
        "clinical_recommendation": True,
        "claim": "clinical recommendation",
    }
    clinical_codes = {
        item["code"] for item in validate_claim_boundaries(clinical_claim)
    }
    assert {
        "clinical_language_violation",
        "prohibited_claim_language",
    } <= clinical_codes

    gaps = [{"gap_uid": "gap_1", "round18_impact": "resolved_by_correct_prediction"}]
    assert any(
        item["code"] == "frozen_gap_count_drift"
        for item in validate_frozen_gap_isolation(gaps, expected_count=20)
    )
    assert any(
        item["code"] == "benchmark_result_closes_empirical_gap"
        for item in validate_frozen_gap_isolation(
            [
                {
                    "gap_uid": f"gap_{i}",
                    "round18_impact": "resolved_by_correct_prediction",
                }
                for i in range(20)
            ]
        )
    )


def test_lineage_and_retry_preserve_failed_attempts() -> None:
    assert validate_lineage_acyclic([("a", "b"), ("b", "a")])
    attempts = [
        {
            "attempt_uid": "attempt-a",
            "status": "FAILED",
            "preserved": True,
            "failure_class": "leakage",
            "attempt_index": 0,
        },
        {
            "attempt_uid": "attempt-a",
            "status": "COMPLETED",
            "preserved": True,
            "retry_of": "attempt-a",
            "attempt_index": 1,
        },
    ]
    assert any(
        item["code"] == "failed_episode_overwritten_by_retry"
        for item in validate_failed_retry_preservation(attempts)
    )


def test_metrics_are_defined_for_direction_catalog() -> None:
    metrics = compute_direction_metrics(
        ["decrease", "no_clear_effect"],
        ["decrease", "decrease"],
        ["publication_a", "publication_b"],
        probabilities=[
            {
                "increase": 0.02,
                "decrease": 0.90,
                "no_clear_effect": 0.03,
                "mixed_or_context_dependent": 0.03,
                "insufficient_information": 0.02,
            },
            {
                "increase": 0.05,
                "decrease": 0.60,
                "no_clear_effect": 0.20,
                "mixed_or_context_dependent": 0.10,
                "insufficient_information": 0.05,
            },
        ],
    )
    assert metrics["publication_macro_direction_accuracy"] == 0.5
    assert metrics["brier_score"] is not None


def test_gate_taxonomy_keeps_execution_blocked_until_evidence_exists() -> None:
    assert (
        gate_state_for_blockers([{"code": "execution_artifact_missing"}])
        == "blocked_execution_verification"
    )
    assert (
        gate_state_for_blockers([]) == "literature_held_out_sequential_replay_resolved"
    )
