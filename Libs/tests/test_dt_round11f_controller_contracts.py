from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from Libs.utils.data.evidence_union import build_evidence_union
from Libs.utils.exp.orchestration_controller import (
    CANONICAL_CONTROLLER_UID,
    GateAuthorityError,
    assert_formal_gate_writer,
    build_canonical_gate_decision,
    load_authority_contract,
)
from Libs.utils.exp.round11f_controller_lineage import (
    GATE_SCHEMA_PATH,
    UNION_SCHEMA_PATH,
    build_round11f_bundle,
    controller_authority_map,
    controller_components,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_PATH = REPO_ROOT / "Libs/configs/validation/round11f_controller_authority.yaml"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Use a complete local Git fixture and the unchanged historical evidence."""
    repository = tmp_path_factory.mktemp("controller_contract")
    for relative in (
        "TODO.md",
        "orchestrator/state.json",
        "Libs/configs/config_stack.yaml",
        "Libs/configs/validation/iclr_design_contract_union.yaml",
        "Libs/utils/data/iclr_contract_union.py",
        "Libs/scripts/validators/validate_iclr_design_contract_union.py",
        "Libs/scripts/validators/validate_orchestrator_state_contract.py",
        "orchestrator/roadmap-updates/round-11-dt-partial-state-roadmap-update.md",
        "Libs/configs/validation/round11f_controller_authority.yaml",
        "Libs/configs/schema/gate_writer_authority.schema.json",
        "Libs/configs/schema/evidence_union.schema.json",
    ):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)

    # The bundle records this genuine fixture commit. The imported validators
    # and all historical audit bodies still come from the approved outer stage.
    for arguments in (
        ["init", "--quiet"],
        ["add", "--all"],
        ["-c", "user.name=NeuronSifter software fixture",
         "-c", "user.email=fixture@invalid.example", "-c", "commit.gpgsign=false",
         "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m",
         "Non-authoritative controller contract inputs"],
    ):
        subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)
    return build_round11f_bundle(repository, evidence_root=REPO_ROOT)


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


def test_exactly_one_canonical_controller_and_gate_writer(bundle: dict[str, object]) -> None:
    authority = bundle["controller_authority_map"]
    assert authority["canonical_controller_uid"] == CANONICAL_CONTROLLER_UID
    assert authority["canonical_controller_count"] == 1
    assert authority["canonical_gate_writer_count"] == 1
    assert authority["formal_gate_writers"] == [CANONICAL_CONTROLLER_UID]


def test_duplicate_controller_and_writer_are_rejected() -> None:
    components = deepcopy(controller_components())
    duplicate_controller = deepcopy(components[0])
    duplicate_controller["component_uid"] = "competing_controller"
    components.append(duplicate_controller)
    assert controller_authority_map(components)["canonical_controller_count"] == 2

    duplicate_writer_components = deepcopy(controller_components())
    duplicate_writer_components[1]["formal_gate_write_authority"] = True
    duplicate_writer_map = controller_authority_map(duplicate_writer_components)
    assert duplicate_writer_map["canonical_gate_writer_count"] == 2
    assert duplicate_writer_map["status"] in {"blocked_multiple_gate_writers", "blocked_multiple_controllers"}


def test_noncanonical_builder_scitil_and_baseline_cannot_write_formal_gate() -> None:
    authority = load_authority_contract(AUTHORITY_PATH)
    for writer_kind, uid in (
        ("builder_direct", "legacy_builder"),
        ("llm_agent", "scitil_agent"),
        ("baseline_pipeline", "baseline_r1_qualification_pipeline"),
    ):
        with pytest.raises(GateAuthorityError):
            assert_formal_gate_writer(authority, controller_uid=uid, writer_kind=writer_kind)


def test_formal_gate_requires_approved_validator_hashes_and_lineage_parent() -> None:
    authority = load_authority_contract(AUTHORITY_PATH)
    kwargs = {
        "authority": authority,
        "gate_uid": "test_gate",
        "stage_uid": "round_11f_controller_lineage",
        "controller_uid": CANONICAL_CONTROLLER_UID,
        "validator_uid": "round11f_controller_union_lineage_validator",
        "validator_report": {"validator_uid": "round11f_controller_union_lineage_validator", "status": "passed"},
        "input_artifact_hashes": {"input": "a" * 64},
        "config_hash": "b" * 64,
        "commit": "c" * 40,
        "decision": "accepted",
        "blockers": [],
        "lineage_parent": "test_parent",
    }
    decision = build_canonical_gate_decision(**kwargs)
    assert decision["lineage_parent"] == "test_parent"
    with pytest.raises(GateAuthorityError):
        build_canonical_gate_decision(**{**kwargs, "lineage_parent": ""})
    with pytest.raises(GateAuthorityError):
        build_canonical_gate_decision(**{**kwargs, "validator_uid": "unapproved_validator"})


def test_union_conflicts_fail_closed_and_preserve_blocked_fixture_and_provenance() -> None:
    semantic_conflict = build_evidence_union(
        union_uid="semantic_conflict",
        members=[
            _member("left", "1" * 64, semantic_fingerprint="left"),
            _member("right", "2" * 64, semantic_fingerprint="right"),
        ],
        canonical_output_ref="test",
    )
    assert semantic_conflict["resolution_status"] == "blocked"
    assert semantic_conflict["status"] == "blocked"

    unknown = build_evidence_union(
        union_uid="unknown_provenance",
        members=[_member("unknown", "3" * 64, provenance_status="unknown")],
        canonical_output_ref="test",
    )
    assert unknown["resolution_status"] == "blocked"

    mixed = build_evidence_union(
        union_uid="fixture_mixed",
        members=[
            _member("real", "4" * 64, lineage_group="real"),
            _member("fixture", "5" * 64, lineage_group="fixture", evidence_type="fixture", claim_eligibility="blocked"),
        ],
        canonical_output_ref="test",
    )
    assert mixed["canonical_downstream_eligibility"]["evidence_type"] != "real"
    assert mixed["canonical_downstream_eligibility"]["claim_eligibility"] == "blocked"


def test_round11f_bundle_is_scoped_and_round12_is_blocked(bundle: dict[str, object]) -> None:
    assert bundle["validation"]["status"] == "passed"
    assert bundle["validation"]["negative_case_rejection_count"] >= 10
    assert bundle["validation"]["unexpected_negative_case_accept_count"] == 0
    assert bundle["dependency_gate"]["round_12_start_allowed"] is False
    assert bundle["formal_gate"]["decision"] == "accepted"
    assert bundle["formal_gate"]["round_11_closure"] == "blocked_full_suite"
    assert bundle["formal_gate"]["evidence_fabric_sealed"] is False
    assert bundle["formal_gate"]["dt_core"] == "not_evaluated"
    assert bundle["handoff"]["current_boundary"]["full_suite_submitted_by_round11f"] is False


def test_gate_and_union_outputs_match_json_schemas(bundle: dict[str, object]) -> None:
    gate_schema = json.loads((REPO_ROOT / GATE_SCHEMA_PATH).read_text(encoding="utf-8"))
    union_schema = json.loads((REPO_ROOT / UNION_SCHEMA_PATH).read_text(encoding="utf-8"))
    gate_errors = list(Draft202012Validator(gate_schema, format_checker=FormatChecker()).iter_errors(bundle["formal_gate"]))
    union_errors = list(Draft202012Validator(union_schema, format_checker=FormatChecker()).iter_errors(bundle["union_contract"]))
    assert gate_errors == []
    assert union_errors == []
