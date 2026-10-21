from argparse import Namespace
import json
from pathlib import Path
from unittest.mock import create_autospec

import pytest
import yaml
from jsonschema import Draft202012Validator, ValidationError

from Libs.exp.training import replay_operational_twin_checkpoint, train_operational_twin_devcal
from Libs.utils.evaluation.operational_twin_training_protocol import build_assignments
from Libs.utils.data.controlled_world_tensorization import (
    ControlledWorldAccessError,
    _assert_no_forbidden_fields,
    load_controlled_manifest,
    validate_controlled_manifest_payload,
)


def base_manifest() -> dict[str, object]:
    return {
        "schema_version": "neuronsifter-controlled-world-open-action-manifest-v1",
        "manifest_uid": "test_manifest",
        "role": "training",
        "evidence_class": "controlled_world_non_empirical",
        "independent_unit": "latent_world",
        "family_uids": [
            "electrophysiology_conductance_world",
            "calcium_synaptic_world",
            "molecular_pharmacological_causal_world",
        ],
        "latent_world_count": 1,
        "supported_execution_count": 1,
        "support_case_count": 0,
        "family_counts": {},
        "pairing_authority": {},
        "counterfactual_pairs": [],
        "development_support_cases": [],
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
        "closed_downstream_flags": {
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
        "worlds": [],
        "content_identity": "0" * 64,
    }


@pytest.mark.parametrize("field", ["twin_qualification", "test_outcome", "protected_holdout", "six_context_outcome", "oracle_state", "hidden_utility", "future_observation", "therapeutic_label", "supervised_label", "mechanism_source", "agent_result", "agent_control", "agent_trace"])
def test_prohibited_fields_fail_before_model_execution(field: str) -> None:
    payload = base_manifest()
    payload["worlds"] = [{field: "forbidden"}]
    with pytest.raises(ControlledWorldAccessError):
        validate_controlled_manifest_payload(payload, "training")


def test_canonical_agent_set_is_not_an_agent_result() -> None:
    payload = base_manifest()
    payload["worlds"] = [{"agent_set": ["controlled_stimulator"]}]
    assert validate_controlled_manifest_payload(payload, "training")["worlds"][0]["agent_set"] == ["controlled_stimulator"]


def test_declared_zero_access_counters_are_allowed() -> None:
    payload = base_manifest()
    payload["worlds"] = [{
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
        }
    }]
    loaded = validate_controlled_manifest_payload(payload, "training")
    assert loaded["worlds"][0]["access_counters"]["six_context_outcome_access"] == 0


def test_nonzero_access_counter_is_rejected() -> None:
    payload = base_manifest()
    payload["access_counters"]["six_context_outcome_access"] = 1
    with pytest.raises(ControlledWorldAccessError, match="exactly zero"):
        validate_controlled_manifest_payload(payload, "training")


def test_row_level_nonzero_access_counter_is_rejected() -> None:
    payload = base_manifest()
    payload["worlds"] = [{
        "access_counters": {
            "six_context_outcome_access": 0,
            "protected_holdout_access": 1,
            "test_outcome_access": 0,
            "tool_access": 0,
            "mcp_access": 0,
            "agent_trace_access": 0,
            "agent_access": 0,
            "memory_access": 0,
            "coevolution_access": 0,
        }
    }]
    with pytest.raises(ControlledWorldAccessError, match="exactly zero"):
        validate_controlled_manifest_payload(payload, "training")


def test_undeclared_access_counter_is_rejected() -> None:
    payload = base_manifest()
    payload["access_counters"]["new_counter"] = 0
    with pytest.raises(ControlledWorldAccessError, match="Undeclared access counter"):
        validate_controlled_manifest_payload(payload, "training")


def test_declared_false_claim_boundary_metadata_is_allowed() -> None:
    _assert_no_forbidden_fields({
        "claim_boundary": {
            "implementation_level_contract_only": True,
            "benchmark_qualification_claim": False,
            "empirical_evidence": False,
            "therapeutic_efficacy_claim": False,
            "clinical_validation_claim": False,
        }
    })


def test_true_prohibited_claim_boundary_is_rejected() -> None:
    with pytest.raises(ControlledWorldAccessError, match="Forbidden true claim-boundary flag"):
        _assert_no_forbidden_fields({
            "claim_boundary": {
                "implementation_level_contract_only": True,
                "benchmark_qualification_claim": False,
                "empirical_evidence": False,
                "therapeutic_efficacy_claim": True,
                "clinical_validation_claim": False,
            }
        })


def test_test_manifest_path_is_rejected_without_opening(tmp_path) -> None:
    path = tmp_path / "test_manifest.json"
    with pytest.raises(ControlledWorldAccessError, match="Forbidden"):
        load_controlled_manifest(path, "training")
    assert not path.exists()


def canonical_args() -> Namespace:
    return Namespace(
        config="Libs/configs/model/operational_twin.yaml",
        training_manifest="Input/processed/controlled_neural_world/open_action_identification/training_manifest.json",
        development_manifest="Input/processed/controlled_neural_world/open_action_identification/development_manifest.json",
        calibration_manifest="Input/processed/controlled_neural_world/open_action_identification/calibration_manifest.json",
        package_receipt="Input/processed/controlled_neural_world/open_action_identification/package_receipt.json",
        open_data_qualification="Output/audits/controlled_world_open_data_qualification.json",
        output_root="Output/results/operational_twin/development_calibration",
    )


@pytest.fixture(scope="module")
def canonical_config() -> dict[str, object]:
    root = Path(train_operational_twin_devcal.resolve_project_root())
    return train_operational_twin_devcal.load_config_stack(
        root / train_operational_twin_devcal.CANONICAL_CONFIG_STACK
    )


@pytest.fixture
def unit_training_preflight(
    monkeypatch: pytest.MonkeyPatch, canonical_config: dict[str, object]
):
    """Isolate assignment governance before testing the real qualification guard."""
    root = Path(train_operational_twin_devcal.resolve_project_root())
    assignment = build_assignments(
        canonical_config["operational_twin_training_protocol"], root
    )[0]
    args = canonical_args()
    args.assignment_uid = assignment["assignment_uid"]
    args.method_id = assignment["method_id"]
    args.seed = assignment["seed"]
    # This unit prerequisite supplies no audit or scientific training authority.
    preflight = create_autospec(
        train_operational_twin_devcal.require_assignment_preflight,
        return_value=assignment,
    )
    monkeypatch.setattr(
        train_operational_twin_devcal, "require_assignment_preflight", preflight
    )
    return args, root, preflight


@pytest.mark.parametrize(
    "argument",
    [
        "config",
        "training_manifest",
        "development_manifest",
        "calibration_manifest",
        "package_receipt",
        "open_data_qualification",
        "output_root",
    ],
)
@pytest.mark.parametrize(
    "entrypoint",
    [train_operational_twin_devcal, replay_operational_twin_checkpoint],
)
def test_entrypoints_reject_same_basename_substitution_before_data_or_model_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    entrypoint: object,
    canonical_config: dict[str, object],
) -> None:
    args = canonical_args()
    canonical_value = Path(getattr(args, argument))
    substitute = tmp_path / canonical_value.name
    if argument == "output_root":
        substitute.mkdir()
    else:
        substitute.write_text("{}", encoding="utf-8")
    setattr(args, argument, str(substitute))

    monkeypatch.setattr(
        train_operational_twin_devcal,
        "load_config_stack",
        lambda unused_path: canonical_config,
    )
    monkeypatch.setattr(
        entrypoint,
        "tensorize_manifest",
        lambda *unused_args, **unused_kwargs: pytest.fail("tensorization ran before canonical-path rejection"),
    )
    with pytest.raises(ValueError, match="canonical config-stack path"):
        entrypoint.run(args)


def test_training_guard_fails_before_output_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, canonical_config: dict[str, object], unit_training_preflight) -> None:
    args, root, preflight = unit_training_preflight
    output_root = tmp_path / "closed_training_output"
    missing_qualification = tmp_path / "controlled_world_open_data_qualification.json"
    args.output_root = str(output_root)
    args.open_data_qualification = str(missing_qualification)
    patched_config = dict(canonical_config)
    patched_operational = dict(patched_config["operational_twin"])
    patched_operational["output_root"] = str(output_root)
    patched_references = dict(patched_operational["references"])
    patched_references["future_open_data_qualification"] = str(missing_qualification)
    patched_operational["references"] = patched_references
    patched_config["operational_twin"] = patched_operational
    monkeypatch.setattr(
        train_operational_twin_devcal,
        "load_config_stack",
        lambda unused_path: patched_config,
    )
    monkeypatch.setattr(
        train_operational_twin_devcal,
        "tensorize_manifest",
        lambda *unused_args, **unused_kwargs: pytest.fail("tensorization ran before training authorization"),
    )
    assert not missing_qualification.exists()
    with pytest.raises(train_operational_twin_devcal.OperationalTwinTrainingClosedError, match="qualification is absent"):
        train_operational_twin_devcal.run(args)
    preflight.assert_called_once_with(
        patched_config, root,
        assignment_uid=args.assignment_uid, method_id=args.method_id, seed=args.seed,
        output_root=output_root.resolve(),
    )
    assert not output_root.exists()


def test_positive_qualification_authorizes_preflight_without_training(canonical_config: dict[str, object]) -> None:
    root = Path(train_operational_twin_devcal.resolve_project_root())
    args = canonical_args()
    contract = train_operational_twin_devcal.resolve_operational_twin_run_contract(
        args, root, require_output=False
    )
    train_operational_twin_devcal.require_operational_twin_training_authorized(
        contract["cfg"], contract["paths"]
    )


def test_negative_and_mismatched_qualification_block_preflight(tmp_path: Path, canonical_config: dict[str, object]) -> None:
    root = Path(train_operational_twin_devcal.resolve_project_root())
    args = canonical_args()
    contract = train_operational_twin_devcal.resolve_operational_twin_run_contract(
        args, root, require_output=False
    )
    positive = json.loads(contract["paths"]["future_open_data_qualification"].read_text(encoding="utf-8"))
    for mutation, expected in (
        ({"terminal_decision": "OPEN_TWIN_DEVELOPMENT_DATA_NOT_QUALIFIED_SCHEMA_INVALID", "training_authorized": False, "status": "BLOCKED"}, "not positive"),
        ({"input_content_identities": {"training": "mismatch"}}, "content identity does not match"),
    ):
        payload = dict(positive)
        payload.update(mutation)
        path = tmp_path / f"{expected.split()[0]}_qualification.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        patched_paths = dict(contract["paths"])
        patched_paths["future_open_data_qualification"] = path
        with pytest.raises(train_operational_twin_devcal.OperationalTwinTrainingClosedError, match=expected):
            train_operational_twin_devcal.require_operational_twin_training_authorized(
                contract["cfg"], patched_paths
            )


def _positive_qualification_contract() -> tuple[dict[str, object], dict[str, object]]:
    root = Path(train_operational_twin_devcal.resolve_project_root())
    args = canonical_args()
    contract = train_operational_twin_devcal.resolve_operational_twin_run_contract(
        args, root, require_output=False
    )
    return contract, json.loads(contract["paths"]["future_open_data_qualification"].read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"role_lineage_isolation": None}, "role-lineage isolation proof is missing"),
        ({"role_lineage_isolation": {"checked_fields": ["latent_world_uid"], "action_result_identity_field": "protocol_state_identity", "status": "PASS", "overlaps": []}}, "isolation fields are incomplete"),
        ({"role_lineage_isolation": {"checked_fields": ["latent_world_uid", "world_uid", "latent_generator_lineage_uid", "seed_identity", "replay_identity", "observable_output_identity", "protocol_state_identity"], "action_result_identity_field": "protocol_state_identity", "status": "PASS", "overlaps": []}}, "isolation fields are incomplete"),
        ({"role_lineage_isolation": {"checked_fields": ["world_uid", "latent_world_uid", "latent_generator_lineage_uid", "seed_identity", "replay_identity", "observable_output_identity", "protocol_state_identity"], "action_result_identity_field": "action_binding_identity", "status": "PASS", "overlaps": []}}, "action-result identity binding is invalid"),
        ({"role_lineage_isolation": {"checked_fields": ["world_uid", "latent_world_uid", "latent_generator_lineage_uid", "seed_identity", "replay_identity", "observable_output_identity", "protocol_state_identity"], "action_result_identity_field": "protocol_state_identity", "status": "BLOCKED", "overlaps": []}}, "role-lineage isolation did not pass"),
        ({"role_lineage_isolation": {"checked_fields": ["world_uid", "latent_world_uid", "latent_generator_lineage_uid", "seed_identity", "replay_identity", "observable_output_identity", "protocol_state_identity"], "action_result_identity_field": "protocol_state_identity", "status": "PASS", "overlaps": [{"field": "world_uid", "left_role": "training", "right_role": "development", "overlap_count": 1}]}}, "isolation overlaps are present"),
    ],
)
def test_malformed_positive_qualification_blocks_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_config: dict[str, object],
    mutation: dict[str, object],
    expected: str,
    unit_training_preflight,
) -> None:
    args, root, preflight = unit_training_preflight
    output_root = tmp_path / "should_not_exist"
    qualification_path = tmp_path / "qualification.json"
    args.output_root = str(output_root)
    args.open_data_qualification = str(qualification_path)
    patched_config = dict(canonical_config)
    patched_operational = dict(patched_config["operational_twin"])
    patched_operational["output_root"] = str(output_root)
    patched_references = dict(patched_operational["references"])
    patched_references["future_open_data_qualification"] = str(qualification_path)
    patched_operational["references"] = patched_references
    patched_config["operational_twin"] = patched_operational
    _unused_contract, positive = _positive_qualification_contract()
    payload = dict(positive)
    payload.update(mutation)
    qualification_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        train_operational_twin_devcal,
        "load_config_stack",
        lambda unused_path: patched_config,
    )
    monkeypatch.setattr(
        train_operational_twin_devcal,
        "tensorize_manifest",
        lambda *unused_args, **unused_kwargs: pytest.fail("tensorization ran before malformed qualification rejection"),
    )
    with pytest.raises(train_operational_twin_devcal.OperationalTwinTrainingClosedError, match=expected):
        train_operational_twin_devcal.run(args)
    preflight.assert_called_once_with(
        patched_config, root,
        assignment_uid=args.assignment_uid, method_id=args.method_id, seed=args.seed,
        output_root=output_root.resolve(),
    )
    assert not output_root.exists()


def test_schema_rejects_contradictory_positive_role_isolation() -> None:
    _unused_contract, positive = _positive_qualification_contract()
    schema_path = Path(train_operational_twin_devcal.resolve_project_root()) / "Libs/configs/schema/controlled_world_open_data_qualification.schema.yaml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    validator.validate(positive)
    contradictory = dict(positive)
    contradictory["role_lineage_isolation"] = {
        "checked_fields": [
            "world_uid",
            "latent_world_uid",
            "latent_generator_lineage_uid",
            "seed_identity",
            "replay_identity",
            "observable_output_identity",
            "protocol_state_identity",
        ],
        "action_result_identity_field": "protocol_state_identity",
        "status": "BLOCKED",
        "overlaps": [{"field": "world_uid", "left_role": "training", "right_role": "development", "overlap_count": 1}],
    }
    with pytest.raises(ValidationError):
        validator.validate(contradictory)


def test_package_receipt_identity_binding_uses_git_identity_only(
    monkeypatch: pytest.MonkeyPatch,
    canonical_config: dict[str, object],
) -> None:
    root = Path(train_operational_twin_devcal.resolve_project_root())
    args = canonical_args()
    original_read_text = Path.read_text
    package_receipt = (root / args.package_receipt).resolve()

    def guarded_read_text(path: Path, *read_args: object, **read_kwargs: object) -> str:
        if path.resolve() == package_receipt:
            raise AssertionError("Package receipt content was read")
        return original_read_text(path, *read_args, **read_kwargs)

    def fake_git_output(unused_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
            return "a" * 40
        if arguments == ("rev-parse", "--verify", f"{'a' * 40}:{args.package_receipt}"):
            return "b" * 40
        raise AssertionError(f"Unexpected Git identity command: {arguments}")

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(train_operational_twin_devcal, "_git_output", fake_git_output)
    monkeypatch.setattr(
        train_operational_twin_devcal,
        "load_config_stack",
        lambda unused_path: canonical_config,
    )
    contract = train_operational_twin_devcal.resolve_operational_twin_run_contract(
        args, root, require_output=False
    )
    identity = train_operational_twin_devcal.tracked_reference_identity(
        root, contract["paths"]["open_action_package_receipt"]
    )
    assert identity == {
        "path": args.package_receipt,
        "authority": "git_commit_tree_blob",
        "git_object_id": identity["git_object_id"],
        "git_object_type": "blob",
        "receipt_content_opened": False,
    }


def test_receipt_identity_is_independent_of_head(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(train_operational_twin_devcal.resolve_project_root())
    receipt = root / "Input/processed/controlled_neural_world/open_action_identification/package_receipt.json"
    relative_path = receipt.relative_to(root).as_posix()
    first_commit = "a" * 40
    second_commit = "b" * 40
    receipt_blob = "c" * 40
    commits = iter((first_commit, second_commit))
    calls: list[tuple[str, ...]] = []

    def fake_git_output(unused_root: Path, *arguments: str) -> str:
        calls.append(arguments)
        if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
            return next(commits)
        if arguments in {
            ("rev-parse", "--verify", f"{first_commit}:{relative_path}"),
            ("rev-parse", "--verify", f"{second_commit}:{relative_path}"),
        }:
            return receipt_blob
        raise AssertionError(f"Unexpected Git identity command: {arguments}")

    monkeypatch.setattr(train_operational_twin_devcal, "_git_output", fake_git_output)
    first = train_operational_twin_devcal.tracked_reference_identity(root, receipt)
    second = train_operational_twin_devcal.tracked_reference_identity(root, receipt)

    assert first == second
    assert first == {
        "path": relative_path,
        "authority": "git_commit_tree_blob",
        "git_object_id": receipt_blob,
        "git_object_type": "blob",
        "receipt_content_opened": False,
    }
    assert calls == [
        ("rev-parse", "--verify", "HEAD^{commit}"),
        ("rev-parse", "--verify", f"{first_commit}:{relative_path}"),
        ("rev-parse", "--verify", "HEAD^{commit}"),
        ("rev-parse", "--verify", f"{second_commit}:{relative_path}"),
    ]
