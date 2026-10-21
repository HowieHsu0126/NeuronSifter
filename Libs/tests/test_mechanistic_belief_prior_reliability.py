from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

from jsonschema import Draft202012Validator
import pytest
import yaml

from Libs.utils.data.mechanistic_belief_construct_firewall import (
    canonical_json_bytes,
    sha256_file,
)
from Libs.utils.model import mechanistic_belief_prior_reliability as mpr


DEPENDENCY_PATHS = (
    "Libs/configs/validation/mechanistic_belief_construct_firewall.yaml",
    "Libs/configs/schema/mechanistic_belief_cutoff_receipt.schema.json",
    "Libs/configs/schema/mechanistic_belief_construct_authority.schema.json",
    "Output/audits/mechanistic_belief_cutoff_receipt.json",
    "Output/audits/mechanistic_belief_construct_authority.json",
    "Output/audits/no_task_fit_parent_authority.json",
    "Output/audits/no_task_fit_reusable_asset_lineage.json",
    "Output/audits/no_task_fit_route_arbitration.json",
    "Output/audits/no_task_fit_candidate_registry.json",
    "orchestrator/roadmaps/2026-07-13-00-iclr-2027-design-contract/rev-007/roadmap.md",
    "orchestrator/roadmaps/2026-07-13-00-iclr-2027-design-contract/rev-007/verification.md",
    mpr.CONTRACT_PATH,
    mpr.AUTHORITY_SCHEMA_PATH,
    mpr.DECISION_SCHEMA_PATH,
    mpr.UTILITY_PATH,
    mpr.BUILDER_PATH,
    mpr.VALIDATOR_PATH,
)


@pytest.fixture()
def isolated_repository(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[2]
    for relative in DEPENDENCY_PATHS:
        source = source_root / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return tmp_path


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_canonical_json(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _build(root: Path) -> tuple[dict[str, object], dict[str, object], bool]:
    return mpr.build_prior_reliability_authority(repository_root=root)


def _mutate_contract(case: str, payload: dict[str, object]) -> None:
    if case == "prior_role_removed":
        payload["prior_roles"]["ordered_roles"].pop()
    elif case == "prior_role_reordered":
        payload["prior_roles"]["ordered_roles"].reverse()
    elif case == "prior_role_added":
        payload["prior_roles"]["ordered_roles"].append(
            deepcopy(payload["prior_roles"]["ordered_roles"][-1])
        )
    elif case == "preferred_sensitivity":
        payload["prior_roles"]["preferred_sensitivity_role"] = (
            "broad_parameter_sensitivity"
        )
    elif case == "more_than_two_sensitivity_roles":
        payload["prior_roles"]["maximum_sensitivity_roles"] = 3
    elif case == "domain_family_mapping":
        payload["parameter_domain_distribution_families"][0][
            "distribution_family"
        ] = "lognormal"
    elif case == "numerical_prior":
        payload["prior_roles"]["default_value"] = 0.5
    elif case == "numerical_threshold":
        payload["abstention_contract"]["decision_threshold"] = 0.5
    elif case == "latent_rename":
        payload["latent_state_factors"][1]["factor_uid"] = "clinical_state"
    elif case == "clinical_observable":
        payload["observable_families"][0][
            "scientific_meaning"
        ] = "clinical_endpoint"
    elif case == "comparison_identity_split":
        payload["comparison_forms"]["ordered_forms"][0][
            "method_uid"
        ] = "other_method"
    elif case == "preferred_form":
        payload["comparison_forms"]["preferred_form"] = "full_belief_twin"
    elif case == "uncertainty_removed":
        payload["uncertainty_components"].pop()
    elif case == "uncertainty_reordered":
        payload["uncertainty_components"].reverse()
    elif case == "support_state_changed":
        payload["support_contract"]["states"][0]["support_state"] = "SUPPORTED"
    elif case == "support_precedence_changed":
        payload["support_contract"]["precedence"].reverse()
    elif case == "unsupported_abstention_removed":
        payload["support_contract"]["states"][0]["mandatory_abstention"] = False
    elif case == "ood_abstention_removed":
        payload["support_contract"]["states"][1]["mandatory_abstention"] = False
    elif case == "continuous_fallback_removed":
        payload["abstention_contract"]["default_mode"] = "binary"
    elif case == "learned_confidence_claim":
        payload["claim_closures"]["learned_confidence_claim"] = True
    elif case == "tri_claim":
        payload["claim_closures"]["tri_claim"] = True
    elif case == "efficacy_claim":
        payload["claim_closures"]["efficacy_claim"] = True
    elif case == "pairwise_category_changed":
        payload["pairwise_distributions"]["effect_direction_categories"][0] = (
            "favorable"
        )
    elif case == "pairwise_rank_changed":
        payload["pairwise_distributions"]["rank_distribution"] = (
            "point_score_sort"
        )
    elif case == "pairwise_tie_changed":
        payload["abstention_contract"]["forced_decision_tie_rule"] = "random"
    elif case == "terminal_added":
        payload["terminal_decisions"].append(
            deepcopy(payload["terminal_decisions"][-1])
        )
    elif case == "terminal_removed":
        payload["terminal_decisions"].pop()
    elif case == "terminal_renamed":
        payload["terminal_decisions"][0]["terminal"] = "READY"
    elif case == "terminal_trigger_changed":
        payload["terminal_decisions"][1]["trigger_stage"] = "runtime"
    elif case == "terminal_closure_changed":
        payload["terminal_decisions"][1]["closed_milestone_classes"].pop()
    elif case == "current_terminal_issued":
        payload["current_terminal"] = "MECHANISM_BASIS_INSUFFICIENT"
        payload["terminal_issued_in_this_round"] = True
    elif case == "access_counter_nonzero":
        payload["access_counters"]["six_context_outcome_access"] = 1
    elif case == "queue_reopened":
        payload["downstream_authorization"][
            "source_queue_creation_authorized"
        ] = True
    elif case == "content_reopened":
        payload["downstream_authorization"][
            "source_content_intake_authorized"
        ] = True
    elif case == "compiler_reopened":
        payload["downstream_authorization"][
            "world_action_compiler_authorized"
        ] = True
    elif case == "runtime_reopened":
        payload["downstream_authorization"][
            "runtime_implementation_authorized"
        ] = True
    elif case == "inference_reopened":
        payload["downstream_authorization"]["inference_authorized"] = True
    elif case == "training_reopened":
        payload["downstream_authorization"]["training_authorized"] = True
    elif case == "gpu_reopened":
        payload["downstream_authorization"]["gpu_authorized"] = True
    elif case == "evaluation_reopened":
        payload["downstream_authorization"][
            "formal_evaluation_authorized"
        ] = True
    else:
        raise AssertionError(f"unknown mutation case: {case}")


def test_build_validate_and_idempotent_rerun(isolated_repository: Path) -> None:
    authority, decision_table, created = _build(isolated_repository)
    assert created is True
    assert authority["status"] == (
        "MECHANISTIC_BELIEF_PRIOR_RELIABILITY_SEMANTICS_FROZEN"
    )
    assert decision_table["status"] == (
        "MECHANISTIC_BELIEF_TERMINAL_DECISION_TABLE_FROZEN"
    )
    assert len(decision_table["decisions"]) == 6
    assert decision_table["current_terminal"] is None
    assert decision_table["terminal_issued_in_this_round"] is False
    assert not any(authority["downstream_authorization"].values())
    assert set(authority["access_counters"].values()) == {0}
    observed = mpr.validate_prior_reliability_outputs(
        repository_root=isolated_repository
    )
    assert observed == (authority, decision_table)
    rerun_authority, rerun_table, rerun_created = _build(isolated_repository)
    assert rerun_created is False
    assert rerun_authority == authority
    assert rerun_table == decision_table


def test_schemas_are_valid_draft_2020_12(isolated_repository: Path) -> None:
    for relative in (mpr.AUTHORITY_SCHEMA_PATH, mpr.DECISION_SCHEMA_PATH):
        Draft202012Validator.check_schema(_load_json(isolated_repository / relative))


@pytest.mark.parametrize(
    "relative",
    [
        "Output/audits/mechanistic_belief_cutoff_receipt.json",
        "Output/audits/mechanistic_belief_construct_authority.json",
    ],
)
def test_missing_round24_authority_fails(
    isolated_repository: Path, relative: str
) -> None:
    (isolated_repository / relative).unlink()
    with pytest.raises(mpr.PriorReliabilityError, match="Round-24"):
        _build(isolated_repository)


@pytest.mark.parametrize(
    "relative",
    [
        "Output/audits/mechanistic_belief_cutoff_receipt.json",
        "Output/audits/mechanistic_belief_construct_authority.json",
    ],
)
def test_drifted_round24_authority_fails(
    isolated_repository: Path, relative: str
) -> None:
    path = isolated_repository / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(mpr.PriorReliabilityError, match="Round-24"):
        _build(isolated_repository)


@pytest.mark.parametrize(
    "case",
    [
        "prior_role_removed",
        "prior_role_reordered",
        "prior_role_added",
        "preferred_sensitivity",
        "more_than_two_sensitivity_roles",
        "domain_family_mapping",
        "numerical_prior",
        "numerical_threshold",
        "latent_rename",
        "clinical_observable",
        "comparison_identity_split",
        "preferred_form",
        "uncertainty_removed",
        "uncertainty_reordered",
        "support_state_changed",
        "support_precedence_changed",
        "unsupported_abstention_removed",
        "ood_abstention_removed",
        "continuous_fallback_removed",
        "learned_confidence_claim",
        "tri_claim",
        "efficacy_claim",
        "pairwise_category_changed",
        "pairwise_rank_changed",
        "pairwise_tie_changed",
        "terminal_added",
        "terminal_removed",
        "terminal_renamed",
        "terminal_trigger_changed",
        "terminal_closure_changed",
        "current_terminal_issued",
        "access_counter_nonzero",
        "queue_reopened",
        "content_reopened",
        "compiler_reopened",
        "runtime_reopened",
        "inference_reopened",
        "training_reopened",
        "gpu_reopened",
        "evaluation_reopened",
    ],
)
def test_rehashed_contract_mutation_cannot_bypass_code_pins(
    isolated_repository: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    contract_path = isolated_repository / mpr.CONTRACT_PATH
    payload = _load_yaml(contract_path)
    _mutate_contract(case, payload)
    _write_yaml(contract_path, payload)
    monkeypatch.setattr(
        mpr, "EXPECTED_CONTRACT_SHA256", sha256_file(contract_path)
    )
    with pytest.raises(mpr.PriorReliabilityError):
        _build(isolated_repository)


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [
        ("authority", "claim"),
        ("authority", "method"),
        ("authority", "authorization"),
        ("decision", "terminal"),
        ("decision", "trigger"),
        ("decision", "current"),
    ],
)
def test_self_consistently_rehashed_output_mutation_fails(
    isolated_repository: Path, artifact: str, mutation: str
) -> None:
    _build(isolated_repository)
    path = isolated_repository / (
        mpr.AUTHORITY_PATH if artifact == "authority" else mpr.DECISION_TABLE_PATH
    )
    payload = _load_json(path)
    if mutation == "claim":
        payload["claim_closures"]["learned_confidence_claim"] = True
    elif mutation == "method":
        payload["method_uid"] = "other_method"
    elif mutation == "authorization":
        payload["downstream_authorization"]["runtime_implementation_authorized"] = (
            True
        )
    elif mutation == "terminal":
        payload["decisions"][0]["terminal"] = "READY"
    elif mutation == "trigger":
        payload["decisions"][1]["trigger_stage"] = "runtime"
    elif mutation == "current":
        payload["current_terminal"] = "MECHANISM_BASIS_INSUFFICIENT"
        payload["terminal_issued_in_this_round"] = True
    hash_field = (
        "authority_sha256" if artifact == "authority" else "decision_table_sha256"
    )
    payload[hash_field] = mpr._self_hash(payload, hash_field)
    _write_canonical_json(path, payload)
    with pytest.raises(mpr.PriorReliabilityError):
        mpr.validate_prior_reliability_outputs(repository_root=isolated_repository)


def test_partial_existing_output_is_rejected(isolated_repository: Path) -> None:
    authority_path = isolated_repository / mpr.AUTHORITY_PATH
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(mpr.PriorReliabilityError, match="partial existing"):
        _build(isolated_repository)


def test_drifted_existing_output_is_not_overwritten(isolated_repository: Path) -> None:
    _build(isolated_repository)
    authority_path = isolated_repository / mpr.AUTHORITY_PATH
    before = authority_path.read_bytes()
    authority_path.write_bytes(before + b"\n")
    drifted = authority_path.read_bytes()
    with pytest.raises(mpr.PriorReliabilityError):
        _build(isolated_repository)
    assert authority_path.read_bytes() == drifted


def test_transactional_install_rolls_back(isolated_repository: Path) -> None:
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second install failure")
        source.replace(destination)

    with pytest.raises(mpr.PriorReliabilityError, match="transactional"):
        mpr.build_prior_reliability_authority(
            repository_root=isolated_repository,
            replace=fail_second,
        )
    assert not (isolated_repository / mpr.AUTHORITY_PATH).exists()
    assert not (isolated_repository / mpr.DECISION_TABLE_PATH).exists()


def test_round24_inputs_remain_byte_identical(isolated_repository: Path) -> None:
    before = {
        relative: sha256_file(isolated_repository / relative)
        for relative, *_ in mpr.ROUND24_INPUTS
    }
    _build(isolated_repository)
    after = {
        relative: sha256_file(isolated_repository / relative)
        for relative, *_ in mpr.ROUND24_INPUTS
    }
    assert after == before
    assert before == {
        relative: expected_hash
        for relative, expected_hash, *_ in mpr.ROUND24_INPUTS
    }


def test_forced_decision_preserves_full_form_receipt(isolated_repository: Path) -> None:
    authority, _, _ = _build(isolated_repository)
    forms = {
        row["form_uid"]: row for row in authority["comparison_forms"]["ordered_forms"]
    }
    forced = forms["forced_decision"]
    assert forced["method_uid"] == forms["full_belief_twin"]["method_uid"]
    assert forced["distributional_spread"] == "retained_in_receipt"
    assert "retain_full_form_abstention_receipt" in forced["abstention_behavior"]
    assert authority["pairwise_distributions"][
        "forced_decision_preserves_original_distribution"
    ] is True


def test_support_precedence_and_mandatory_abstention(isolated_repository: Path) -> None:
    authority, _, _ = _build(isolated_repository)
    support = authority["support_contract"]
    assert support["precedence"] == [
        "UNSUPPORTED",
        "OOD",
        "PARTIAL_SUPPORT",
        "SUPPORTED",
    ]
    states = {row["support_state"]: row for row in support["states"]}
    assert states["UNSUPPORTED"]["mandatory_abstention"] is True
    assert states["OOD"]["mandatory_abstention"] is True
    assert authority["abstention_contract"]["default_mode"] == "continuous_only"
    assert authority["abstention_contract"]["binary_rule_frozen"] is False


def test_no_numerical_prior_values_or_current_terminal(isolated_repository: Path) -> None:
    authority, table, _ = _build(isolated_repository)
    mpr._validate_no_numeric_prior_or_threshold(authority)
    mpr._validate_no_numeric_prior_or_threshold(table)
    assert authority["prior_roles"]["numerical_priors_populated"] is False
    assert table["current_terminal"] is None
    assert all(
        row["terminal_issued_in_this_round"] is False for row in table["decisions"]
    )
