from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest

from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.data.raw_data_registry import load_config_with_provenance

ROOT = Path(__file__).resolve().parents[2]


_TRAINING_PROTOCOL_TEST_IDENTITIES = {'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[candidate_lineage_uid-tampered_lineage-candidate_runtime_identity_mismatch]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                                'parameters': {'field': 'candidate_lineage_uid',
                                                                                                                                                                                                                               'bad_value': 'tampered_lineage',
                                                                                                                                                                                                                               'error_code': 'candidate_runtime_identity_mismatch'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[protocol_uid-tampered_protocol-candidate_runtime_identity_mismatch]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                        'parameters': {'field': 'protocol_uid',
                                                                                                                                                                                                                       'bad_value': 'tampered_protocol',
                                                                                                                                                                                                                       'error_code': 'candidate_runtime_identity_mismatch'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[assignment_uid-tampered_assignment-candidate_assignment_uid_invalid]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                         'parameters': {'field': 'assignment_uid',
                                                                                                                                                                                                                        'bad_value': 'tampered_assignment',
                                                                                                                                                                                                                        'error_code': 'candidate_assignment_uid_invalid'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[attempt_uid-tampered_attempt-candidate_attempt_uid_invalid]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                'parameters': {'field': 'attempt_uid',
                                                                                                                                                                                                               'bad_value': 'tampered_attempt',
                                                                                                                                                                                                               'error_code': 'candidate_attempt_uid_invalid'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[method_id-linear_or_delta-candidate_assignment_identity_mismatch]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                      'parameters': {'field': 'method_id',
                                                                                                                                                                                                                     'bad_value': 'linear_or_delta',
                                                                                                                                                                                                                     'error_code': 'candidate_assignment_identity_mismatch'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[seed-999999-candidate_assignment_identity_mismatch]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                        'parameters': {'field': 'seed',
                                                                                                                                                                                                       'bad_value': 999999,
                                                                                                                                                                                                       'error_code': 'candidate_assignment_identity_mismatch'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[candidate_manifest_hash-0000000000000000000000000000000000000000000000000000000000000000-candidate_runtime_identity_mismatch]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                                                                                  'parameters': {'field': 'candidate_manifest_hash',
                                                                                                                                                                                                                                                                                 'bad_value': '0000000000000000000000000000000000000000000000000000000000000000',
                                                                                                                                                                                                                                                                                 'error_code': 'candidate_runtime_identity_mismatch'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[assignment_manifest_hash-1111111111111111111111111111111111111111111111111111111111111111-candidate_runtime_identity_mismatch]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                                                                                   'parameters': {'field': 'assignment_manifest_hash',
                                                                                                                                                                                                                                                                                  'bad_value': '1111111111111111111111111111111111111111111111111111111111111111',
                                                                                                                                                                                                                                                                                  'error_code': 'candidate_runtime_identity_mismatch'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[command_catalog_hash_value-2222222222222222222222222222222222222222222222222222222222222222-candidate_runtime_identity_mismatch]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                                                                                     'parameters': {'field': 'command_catalog_hash_value',
                                                                                                                                                                                                                                                                                    'bad_value': '2222222222222222222222222222222222222222222222222222222222222222',
                                                                                                                                                                                                                                                                                    'error_code': 'candidate_runtime_identity_mismatch'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[preflight_uid-tampered_preflight-candidate_runtime_identity_mismatch]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                          'parameters': {'field': 'preflight_uid',
                                                                                                                                                                                                                         'bad_value': 'tampered_preflight',
                                                                                                                                                                                                                         'error_code': 'candidate_runtime_identity_mismatch'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root[output_root-wrong_attempt_root-candidate_attempt_output_path_invalid]': {'function_name': 'test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root',
                                                                                                                                                                                                          'parameters': {'field': 'output_root',
                                                                                                                                                                                                                         'bad_value': 'wrong_attempt_root',
                                                                                                                                                                                                                         'error_code': 'candidate_attempt_output_path_invalid'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_runtime_rejects_valid_nonmanifest_attempt_root': {'function_name': 'test_candidate_runtime_rejects_valid_nonmanifest_attempt_root',
                                                                                                                          'parameters': {}}}

_TRAINING_PROTOCOL_GATE_TEST_IDENTITIES = {'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_every_runtime_identity_drift[assignment_manifest_hash]': {'function_name': 'test_candidate_execution_gate_rejects_every_runtime_identity_drift',
                                                                                                                                                         'parameters': {'field': 'assignment_manifest_hash'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_every_runtime_identity_drift[assignment_uid]': {'function_name': 'test_candidate_execution_gate_rejects_every_runtime_identity_drift',
                                                                                                                                               'parameters': {'field': 'assignment_uid'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_every_runtime_identity_drift[attempt_uid]': {'function_name': 'test_candidate_execution_gate_rejects_every_runtime_identity_drift',
                                                                                                                                            'parameters': {'field': 'attempt_uid'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_every_runtime_identity_drift[candidate_lineage_uid]': {'function_name': 'test_candidate_execution_gate_rejects_every_runtime_identity_drift',
                                                                                                                                                      'parameters': {'field': 'candidate_lineage_uid'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_every_runtime_identity_drift[candidate_manifest_hash]': {'function_name': 'test_candidate_execution_gate_rejects_every_runtime_identity_drift',
                                                                                                                                                        'parameters': {'field': 'candidate_manifest_hash'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_every_runtime_identity_drift[command_catalog_hash]': {'function_name': 'test_candidate_execution_gate_rejects_every_runtime_identity_drift',
                                                                                                                                                     'parameters': {'field': 'command_catalog_hash'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_every_runtime_identity_drift[output_root]': {'function_name': 'test_candidate_execution_gate_rejects_every_runtime_identity_drift',
                                                                                                                                            'parameters': {'field': 'output_root'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_every_runtime_identity_drift[protocol_uid]': {'function_name': 'test_candidate_execution_gate_rejects_every_runtime_identity_drift',
                                                                                                                                             'parameters': {'field': 'protocol_uid'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_missing_authority_or_receipt[authority_path-candidate_execution_authority_missing]': {'function_name': 'test_candidate_execution_gate_rejects_missing_authority_or_receipt',
                                                                                                                                                                                     'parameters': {'missing_path': 'authority_path',
                                                                                                                                                                                                    'error_code': 'candidate_execution_authority_missing'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_missing_authority_or_receipt[receipt_path-candidate_resource_receipt_missing]': {'function_name': 'test_candidate_execution_gate_rejects_missing_authority_or_receipt',
                                                                                                                                                                                'parameters': {'missing_path': 'receipt_path',
                                                                                                                                                                                               'error_code': 'candidate_resource_receipt_missing'}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_rejected_and_tampered_authority': {'function_name': 'test_candidate_execution_gate_rejects_rejected_and_tampered_authority',
                                                                                                                                  'parameters': {}},
 'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_execution_gate_rejects_tampered_and_stale_receipt': {'function_name': 'test_candidate_execution_gate_rejects_tampered_and_stale_receipt',
                                                                                                                             'parameters': {}}}

_TRAINING_PROTOCOL_COMMAND_GATE_TEST_IDENTITIES = {'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_command_runner_requires_execution_gate_before_dispatch': {'function_name': 'test_candidate_command_runner_requires_execution_gate_before_dispatch', 'parameters': {}}}
_TRAINING_PROTOCOL_COMMAND_CATALOG_TEST_IDENTITIES = {'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_command_runner_rejects_catalog_tamper_and_bad_task_indexes_before_dispatch': {'function_name': 'test_candidate_command_runner_rejects_catalog_tamper_and_bad_task_indexes_before_dispatch', 'parameters': {}}}
_TRAINING_PROTOCOL_COMMAND_DEFAULT_TEST_IDENTITIES = {'Libs/tests/test_operational_twin_training_protocol.py::test_candidate_command_runner_main_defaults_to_configured_successor_before_authority_gate': {'function_name': 'test_candidate_command_runner_main_defaults_to_configured_successor_before_authority_gate', 'parameters': {}}}

_TRAINING_PROTOCOL_WITNESS_BOOTSTRAP = r"""
import json as _tp_json
import stat as _tp_stat
assert _tp_stat.S_ISFIFO(os.fstat(0).st_mode), "test witness stdin must be a pipe"
_tp_raw = sys.stdin.buffer.read(4097)
assert 0 < len(_tp_raw) <= 4096 and _tp_raw.endswith(b"\n") and _tp_raw.count(b"\n") == 1
def _tp_unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate test witness field")
        result[key] = value
    return result
_training_protocol_call_witness = _tp_json.loads(_tp_raw.decode("utf-8"), object_pairs_hook=_tp_unique_pairs)
_tp_packet = _training_protocol_call_witness
assert type(_tp_packet) is dict and set(_tp_packet) == set(_tp_expected) | {"child_pid", "program_sha256"}
for _tp_key, _tp_value in _tp_expected.items():
    assert type(_tp_packet[_tp_key]) is type(_tp_value) and _tp_packet[_tp_key] == _tp_value
assert type(_tp_packet["child_pid"]) is int and _tp_packet["child_pid"] == os.getpid()
assert _tp_packet["parent_pid"] == os.getppid()
assert _tp_packet["version"] == 1 and 1 <= _tp_packet["call_ordinal"] <= 12
assert _tp_packet["slurm_job_id"] == os.environ["SLURM_JOB_ID"]
assert _tp_packet["phase"] == "call" and os.environ.get("PYTEST_CURRENT_TEST") == _tp_packet["nodeid"] + " (call)"
assert _tp_packet["root"] == os.getcwd() == sys.argv[1]
assert _tp_packet["nodeid"] in _tp_allowed
_tp_case = _tp_allowed[_tp_packet["nodeid"]]
assert _tp_packet["function_name"] == _tp_packet["function_qualname"] == _tp_case["function_name"]
assert type(_tp_packet["parameters"]) is dict and set(_tp_packet["parameters"]) == set(_tp_case["parameters"])
for _tp_key, _tp_value in _tp_case["parameters"].items():
    assert type(_tp_packet["parameters"][_tp_key]) is type(_tp_value) and _tp_packet["parameters"][_tp_key] == _tp_value
assert type(_tp_packet["program_sha256"]) is str and len(_tp_packet["program_sha256"]) == 64
assert all(character in "0123456789abcdef" for character in _tp_packet["program_sha256"])
print(_tp_json.dumps({"event": "test_call_witness_accepted", "witness": _tp_packet}, sort_keys=True), flush=True)
"""

def _training_protocol_witness_bootstrap(metadata):
    """Validate the parent-observed test call before guarded project imports."""
    caller_group = metadata.get("caller_group", "runtime_identity")
    if caller_group == "runtime_identity":
        allowed = _TRAINING_PROTOCOL_TEST_IDENTITIES
    elif caller_group == "gate_metadata":
        allowed = _TRAINING_PROTOCOL_GATE_TEST_IDENTITIES
    elif caller_group == "command_gate_preflight":
        allowed = _TRAINING_PROTOCOL_COMMAND_GATE_TEST_IDENTITIES
        assert metadata["call_ordinal"] == 1, "single command gate caller"
    elif caller_group == "command_catalog_preflight":
        allowed = _TRAINING_PROTOCOL_COMMAND_CATALOG_TEST_IDENTITIES
        assert metadata["call_ordinal"] == 1, "single command catalog caller"
    elif caller_group == "command_default_preflight":
        allowed = _TRAINING_PROTOCOL_COMMAND_DEFAULT_TEST_IDENTITIES
        assert metadata["call_ordinal"] == 1, "single command default caller"
    else:
        raise ValueError("unknown training protocol caller group")
    assert metadata["nodeid"] in allowed, "cross-group training protocol caller"
    return (
        "import json as _tp_json\n"
        + "_tp_expected = _tp_json.loads(" + repr(json.dumps(metadata, sort_keys=True)) + ")\n"
        + "_tp_allowed = _tp_json.loads(" + repr(json.dumps(allowed, sort_keys=True)) + ")\n"
        + _TRAINING_PROTOCOL_WITNESS_BOOTSTRAP
    )


def _phase_marker(component: str, event: str, started: float | None = None) -> float:
    now = time.perf_counter()
    elapsed = "" if started is None else f" elapsed_s={now - started:.3f}"
    print(f"R51_TEST_PHASE component={component} event={event}{elapsed}", flush=True)
    return now


@pytest.fixture(scope="module")
def config():
    return load_config_stack(ROOT / "Libs/configs/config_stack.yaml")


@pytest.fixture(scope="module")
def frozen(config):
    return handoff.authenticate_frozen_successor_package(config, ROOT)


def test_real_frozen_git_replay_keeps_original_predicates_and_identity_boundary(config, frozen):
    gate = frozen["approved_package"]
    assert gate["status"] == "PASS"
    assert gate["candidate_dispatch_authorized"] is False
    assert gate["execution_authority"] is None
    assert gate["resource_environment_receipt"] is None
    assert gate["scientific_twin_result"] == "NOT_RUN"
    assert set(gate["access_counters"].values()) == {0}
    allowed = config[handoff.CONFIG_KEY]["input_identity_allowlist"]
    assert {row["path"] for row in frozen["input_identities"]} == set(allowed)
    assert {row["path"] for row in frozen["input_read_events"]} == set(allowed)
    for row in frozen["input_identities"]:
        assert row["git_object_id"] == allowed[row["path"]]
        assert len(row["sha256"]) == 64 and row["size_bytes"] > 0
    for row in frozen["input_read_events"]:
        assert row["ref"] in config[handoff.CONFIG_KEY]["input_identity_refs"]
    assert gate["review_binding_uid"] == config[handoff.CONFIG_KEY]["frozen_package"]["review_binding_uid"]
    assignments = gate["package"]["assignment_manifest"]
    commands = gate["package"]["assignment_commands"]
    assert assignments["assignment_count"] == 15
    assert assignments["attempt_slot_count"] == 30
    assert [row["command_index"] for row in commands["commands"]] == list(range(30))
    assert assignments["seeds"] == [360101, 360211, 360307]


@pytest.mark.parametrize("field", ["input_identity_refs", "input_identity_allowlist", "frozen_package", "review_contract", "ownership_policy", "access_counters"])
def test_frozen_authority_declaration_mutation_rejected_by_canonical_schema(config, field):
    bad = deepcopy(config)
    bad[handoff.CONFIG_KEY][field] = {}
    with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_config_contract_drift"):
        handoff.authenticate_frozen_successor_package(bad, ROOT)


@pytest.mark.parametrize("path", ["../outside", "Output/../outside", "Output//alias", "/absolute", "Output/./alias", "Output\\alias"])
def test_lexical_path_aliases_rejected(tmp_path, path):
    with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_path_"):
        handoff._path(tmp_path, path)


def test_symlink_ancestor_leaf_hardlink_and_occupied_leaf_rejected(tmp_path):
    (tmp_path / "target").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "target", target_is_directory=True)
    with pytest.raises(handoff.OperationalTwinProtocolError, match="symlink"):
        handoff._path(tmp_path, "alias/file")
    (tmp_path / "source").write_text("owned")
    (tmp_path / "leaf").symlink_to(tmp_path / "source")
    with pytest.raises(handoff.OperationalTwinProtocolError, match="symlink"):
        handoff._path(tmp_path, "leaf")
    with pytest.raises(handoff.OperationalTwinProtocolError, match="occupied"):
        handoff._path(tmp_path, "source", absent=True)
    os.link(tmp_path / "source", tmp_path / "hardlink")
    with pytest.raises(handoff.OperationalTwinProtocolError, match="hardlink"):
        handoff._path(tmp_path, "hardlink")


@pytest.mark.parametrize("spec", ["HEAD:Input/forbidden.json", "wrongref:Input/processed/controlled_neural_world/open_action_identification/training_manifest.json"])
def test_actual_audit_hook_stops_extra_input_or_reference_before_git_read(config, tmp_path, spec):
    contract = config[handoff.CONFIG_KEY]
    boundary = {"inputs": contract["input_identity_allowlist"], "refs": contract["input_identity_refs"], "head": contract["frozen_package"]["reviewed_commit"]}
    prefix = handoff._FROZEN_REPLAY.split("from Libs.utils.exp_utils import load_config_stack", 1)[0]
    script = prefix + "\nimport subprocess\nsubprocess.run(['git', 'show', sys.argv[3]], check=True)\n"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(tmp_path), json.dumps(boundary), spec], capture_output=True, text=True)
    assert result.returncode != 0
    assert "handoff_frozen_input_reference_forbidden" in result.stderr


def test_rejected_real_review_cannot_authorize_handoff(isolated_reviewed_handoff, tmp_path):
    _run_rejected_handoff_control(
        isolated_reviewed_handoff, tmp_path,
        r"""
with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_review_not_approved"):
    handoff.require_approved_successor_execution_handoff(config, ROOT)
""",
        stage='committed rejected handoff approval',
    )


def test_unknown_config_field_rejected(config):
    bad = deepcopy(config)
    bad[handoff.CONFIG_KEY]["approved"] = True
    with pytest.raises(handoff.OperationalTwinProtocolError, match="keyset"):
        handoff._contract(bad)


def test_runtime_resolver_rejects_real_rejected_handoff_review(isolated_reviewed_handoff, tmp_path):
    _run_rejected_handoff_control(
        isolated_reviewed_handoff, tmp_path,
        r"""
from Libs.utils.evaluation.operational_twin_candidate_execution import resolve_candidate_execution_protocol
with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_review_not_approved"):
    resolve_candidate_execution_protocol(config, ROOT)
""",
        stage='committed rejected handoff runtime resolver',
    )


def test_runtime_resolver_has_no_predecessor_fallback(config):
    from Libs.utils.evaluation.operational_twin_candidate_execution import resolve_candidate_execution_protocol
    bad = deepcopy(config)
    bad["operational_twin_candidate_execution"]["schema_version"] = "neuronsifter-operational-twin-candidate-execution-v3"
    with pytest.raises(handoff.OperationalTwinProtocolError, match="candidate_predecessor_execution_closed"):
        resolve_candidate_execution_protocol(bad, ROOT)


def test_runtime_current_handoff_selection_is_not_execution_authority(config):
    from Libs.utils.evaluation.operational_twin_candidate_execution import require_future_execution_selection
    round_dir = ROOT / config[handoff.CONFIG_KEY]["review_contract"]["path"]
    with pytest.raises(handoff.OperationalTwinProtocolError, match="candidate_execution_future_selection_required"):
        require_future_execution_selection(config, ROOT, round_dir.parent)


def test_runtime_array_groups_preserve_every_frozen_command(config):
    execution = config["operational_twin_candidate_execution"]
    path = execution["frozen_artifacts"]["assignment_commands"]["path"]
    commands = json.loads((ROOT / path).read_text())["commands"]
    seen = []
    for group in execution["array_groups"]:
        requirements = {key: group[key] for key in ("partition", "nodelist", "cpus_per_task", "memory", "gres", "exclusive")}
        for conditional, indices in ((False, group["initial_command_indices"]), (True, group["recovery_command_indices"])):
            for index in indices:
                command = commands[index]
                assert command["method_id"] == group["method_id"]
                assert command["conditional"] is conditional
                assert command["resource_requirements"] == requirements
                assert command["wall_time"] == group["wall_time"]
                seen.append(index)
    assert sorted(seen) == list(range(30))


def test_callers_parse_all_thirty_frozen_training_and_evaluation_commands(config, monkeypatch):
    from Libs.exp.training import train_operational_twin_devcal as trainer
    from Libs.exp.training.run_operational_twin_candidate_attempt import _evaluation_argv
    from Libs.exp.evaluation import evaluate_operational_twin_development as evaluator
    path = config["operational_twin_candidate_execution"]["frozen_artifacts"]["assignment_commands"]["path"]
    commands = json.loads((ROOT / path).read_text())["commands"]
    for command in commands:
        monkeypatch.setattr(sys, "argv", ["training", *command["argv"][2:]])
        training = trainer.parse_args()
        evaluation_argv = _evaluation_argv(command)
        monkeypatch.setattr(sys, "argv", ["evaluation", *evaluation_argv[3:]])
        evaluation = evaluator.parse_args()
        for parsed in (training, evaluation):
            assert parsed.protocol_scope == "candidate-lineage"
            assert parsed.candidate_lineage_uid == command["candidate_lineage_uid"]
            assert parsed.assignment_uid == command["assignment_uid"]
            assert parsed.attempt_uid == command["attempt_uid"]
            assert parsed.method_id == command["method_id"]
            assert parsed.seed == command["seed"]
            assert parsed.calibration_execution_authority == training.calibration_execution_authority


@pytest.mark.parametrize("consumer", ["worker", "runner", "trainer", "evaluator"])
def test_callers_reject_real_unapproved_handoff_before_attempt_or_input_work(isolated_reviewed_handoff, tmp_path, consumer):
    _run_rejected_handoff_control(
        isolated_reviewed_handoff, tmp_path,
        "consumer = " + repr(consumer) + "\n" + r"""
with pytest.MonkeyPatch.context() as monkeypatch:
    from Libs.exp.training import train_operational_twin_devcal as trainer
    from Libs.exp.training import run_operational_twin_candidate_attempt as worker
    from Libs.exp.training import run_operational_twin_assignment_command as runner
    from Libs.exp.evaluation import evaluate_operational_twin_development as evaluator
    path = config["operational_twin_candidate_execution"]["frozen_artifacts"]["assignment_commands"]["path"]
    command = json.loads((ROOT / path).read_text())["commands"][0]
    monkeypatch.setattr(sys, "argv", ["training", *command["argv"][2:]])
    args = trainer.parse_args()
    observed = {"enabled": True, "input_reads": 0}
    def boundary(event, arguments):
        if not observed["enabled"] or event != "open" or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        candidate = Path(os.fsdecode(arguments[0])).absolute()
        if candidate.is_relative_to(ROOT / "Input"):
            observed["input_reads"] += 1
            raise PermissionError("unexpected_Input_read_before_handoff_approval")
    sys.addaudithook(boundary)
    try:
        with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_review_not_approved"):
            if consumer == "worker":
                worker.run_attempt(0)
            elif consumer == "runner":
                runner._run_candidate_command(ROOT / path, 0)
            elif consumer == "trainer":
                trainer.run(args)
            else:
                evaluator._candidate_evaluation_context(args, config, ROOT, ROOT / command["output_root"])
        assert observed["input_reads"] == 0
        assert not os.path.lexists(ROOT / command["output_root"])
    finally:
        observed["enabled"] = False
""",
        stage='committed rejected handoff caller',
    )


def test_ownership_exclusive_writer_keeps_first_publication(tmp_path):
    from Libs.utils.evaluation.operational_twin_candidate_execution import write_json_exclusive
    path = tmp_path / "declared/receipt.json"
    write_json_exclusive(path, {"status": "non_authoritative_fixture"})
    original = path.read_bytes()
    with pytest.raises(handoff.OperationalTwinProtocolError, match="artifact_exists"):
        write_json_exclusive(path, {"status": "replacement"})
    assert path.read_bytes() == original
    assert path.stat().st_nlink == 1


@pytest.mark.parametrize("component", ["ancestor", "leaf", "hardlink"])
def test_ownership_exclusive_writer_rejects_link_collisions(tmp_path, component):
    from Libs.utils.evaluation.operational_twin_candidate_execution import write_json_exclusive
    real = tmp_path / "real"
    real.mkdir()
    source = real / "source.json"
    source.write_text("preserve")
    if component == "ancestor":
        (tmp_path / "alias").symlink_to(real, target_is_directory=True)
        path = tmp_path / "alias/receipt.json"
    elif component == "leaf":
        path = real / "receipt.json"
        path.symlink_to(source)
    else:
        path = real / "receipt.json"
        os.link(source, path)
    with pytest.raises((handoff.OperationalTwinProtocolError, OSError)):
        write_json_exclusive(path, {"status": "non_authoritative_fixture"})
    assert source.read_text() == "preserve"
    assert not (real / "receipt.json").exists() if component == "ancestor" else path.is_symlink() or path.stat().st_nlink == 2


@pytest.mark.parametrize("intruder", ["foreign.json", "evaluation/foreign.json", "checkpoints/best_dev.pt"])
def test_ownership_declared_attempt_tree_rejects_foreign_or_no_fit_checkpoint(config, tmp_path, intruder):
    from Libs.utils.evaluation.operational_twin_candidate_execution import _check_declared_attempt_tree
    relative = "Output/results/operational_twin/candidate_lineages/fixture/assignment/attempt"
    path = tmp_path / relative / intruder
    path.parent.mkdir(parents=True)
    path.write_text("non-authoritative metadata fixture; not a model")
    with pytest.raises(handoff.OperationalTwinProtocolError, match="foreign_child|no_fit_checkpoint"):
        _check_declared_attempt_tree(
            config, tmp_path,
            {"method_id": "persistence_no_action", "checkpoint_creation_allowed": False},
            {"output_root": relative},
        )


def test_ownership_descriptor_claim_remains_exclusive(tmp_path):
    from Libs.utils.evaluation.operational_twin_training_protocol import _claim_candidate_attempt_root
    relative = "Output/results/operational_twin/candidate_lineages/fixture/assignment/attempt"
    path = _claim_candidate_attempt_root(tmp_path, relative)
    assert path == tmp_path / relative and path.is_dir()
    with pytest.raises(handoff.OperationalTwinProtocolError, match="output_collision"):
        _claim_candidate_attempt_root(tmp_path, relative)


@pytest.mark.parametrize("mode", ["initial", "recovery"])
def test_submitter_rejects_unapproved_handoff_before_any_scheduler_or_write(isolated_reviewed_handoff, tmp_path, mode):
    _run_rejected_handoff_control(
        isolated_reviewed_handoff, tmp_path,
        "mode = " + repr(mode) + "\n" + r"""
from Libs.scripts.builders.submit_operational_twin_candidate_execution import submit_arrays
calls = []
def capture(argv, root):
    calls.append(argv)
    raise AssertionError("unexpected scheduler call before handoff approval")
with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_review_not_approved"):
    submit_arrays(config, ROOT, mode=mode, command_capture=capture)
assert calls == []
outputs = config["operational_twin_candidate_execution"]["output_paths"]
for name in ("dispatch_claim", "recovery_dispatch_claim", "dispatch_manifest", "recovery_dispatch_manifest"):
    assert not os.path.lexists(ROOT / outputs[name])
""",
        stage='committed rejected handoff submitter',
    )


@pytest.mark.parametrize("mode", ["initial", "recovery"])
def test_submission_wrapper_carries_exact_binding_without_changing_catalog(config, mode):
    from Libs.scripts.builders.submit_operational_twin_candidate_execution import build_array_submission_argv
    execution = config["operational_twin_candidate_execution"]
    catalog = ROOT / execution["frozen_artifacts"]["assignment_commands"]["path"]
    original = catalog.read_bytes()
    binding = {
        "NRNSIFT_HANDOFF_UID": "non_authoritative_local_parser_fixture",
        "NRNSIFT_SOURCE_COMMIT": "parser_fixture_only",
        "NRNSIFT_COMMAND_CATALOG_HASH": execution["frozen_artifacts"]["assignment_commands"]["internal_hash"],
    }
    indices = []
    for group in execution["array_groups"]:
        selected = group[mode + "_command_indices"]
        argv = build_array_submission_argv(ROOT, execution, group, mode=mode, indices=selected, binding=binding)
        wrapper = argv[argv.index("--wrap") + 1]
        syntax = subprocess.run(["bash", "-n"], input=wrapper, text=True, capture_output=True)
        assert syntax.returncode == 0, syntax.stderr
        assert all("export " + key + "=" + value in wrapper for key, value in binding.items())
        assert 'mktemp -d "/tmp/NrnSift_otcand_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.XXXXXX"' in wrapper
        assert ("export CUDA_VISIBLE_DEVICES=" in wrapper) is (group["gres"] is None)
        assert "--exclusive" not in argv
        assert "--hold" in argv
        assert f"--mem={group['memory']}" in argv
        indices.extend(selected)
    assert indices == list(range(15)) if mode == "initial" else indices == list(range(15, 30))
    assert catalog.read_bytes() == original


def test_submission_rejects_cross_group_and_duplicate_indices(config):
    from Libs.scripts.builders.submit_operational_twin_candidate_execution import build_array_submission_argv
    execution = config["operational_twin_candidate_execution"]
    group = execution["array_groups"][0]
    for indices in ([0, 0], [1, 0], [3], [15]):
        with pytest.raises(handoff.OperationalTwinProtocolError, match="indices_invalid"):
            build_array_submission_argv(ROOT, execution, group, mode="initial", indices=indices, binding={})


def _isolated_python(root, source, *, stage="maintained fixture", test_call_witness=None):
    """Run maintained code in the test repository, never through ROOT imports."""
    if test_call_witness is None:
        marker = os.environ.get("PYTEST_CURRENT_TEST", "")
        assert marker not in {nodeid + " (call)" for nodeid in (*_TRAINING_PROTOCOL_TEST_IDENTITIES, *_TRAINING_PROTOCOL_GATE_TEST_IDENTITIES, *_TRAINING_PROTOCOL_COMMAND_GATE_TEST_IDENTITIES, *_TRAINING_PROTOCOL_COMMAND_CATALOG_TEST_IDENTITIES)}, "missing original TP call witness"
        bootstrap = "import os, sys; os.chdir(sys.argv[1]); sys.path.insert(0, sys.argv[1]);\n"
        started = time.perf_counter()
        print(f"isolated_handoff start: {stage}", flush=True)
        # Isolated mode ignores PYTHONPYCACHEPREFIX; -B alone still reads caches.
        with tempfile.TemporaryDirectory(prefix="neuronsifter_handoff_pycache_", dir=os.environ.get("TMPDIR")) as cache_prefix:
            process = subprocess.Popen(
                [sys.executable, "-I", "-B", "-X", "pycache_prefix=" + cache_prefix,
                 "-c", bootstrap + source, str(root)],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            assert process.stdout is not None
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                elapsed = time.perf_counter() - started
                print(f"R51_TEST_CHILD stage={stage} elapsed_s={elapsed:.3f}: {line}", end="", flush=True)
            process.stdout.close()
            returncode = process.wait()
            output = "".join(output_lines)
            print(f"isolated_handoff finish: {stage}; rc={returncode}; seconds={time.perf_counter() - started:.3f}", flush=True)
            assert returncode == 0, output[-14000:]
            return json.loads(output.strip().splitlines()[-1])
    metadata = test_call_witness.child_metadata(root, source, stage)
    bootstrap = "import os, sys; os.chdir(sys.argv[1]); sys.path.insert(0, sys.argv[1]);\n"
    program = bootstrap + _training_protocol_witness_bootstrap(metadata) + source
    started = time.perf_counter()
    print(f"isolated_handoff start: {stage}", flush=True)
    process = None
    completed = False
    output_lines = []
    with tempfile.TemporaryDirectory(prefix="neuronsifter_handoff_pycache_", dir=os.environ.get("TMPDIR")) as cache_prefix:
        try:
            argv = (sys.executable, "-I", "-B", "-X", "pycache_prefix=" + cache_prefix,
                    "-c", program, str(root))
            test_call_witness.before_spawn(argv)
            process = subprocess.Popen(
                argv, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            test_call_witness.send_receipt(process)
            assert process.stdout is not None
            for line in process.stdout:
                output_lines.append(line)
                elapsed = time.perf_counter() - started
                print(f"R51_TEST_CHILD stage={stage} elapsed_s={elapsed:.3f}: {line}", end="", flush=True)
                test_call_witness.receive_line(process, line)
            process.stdout.close()
            returncode = process.wait()
            output = "".join(output_lines)
            print(f"isolated_handoff finish: {stage}; rc={returncode}; seconds={time.perf_counter() - started:.3f}", flush=True)
            assert returncode == 0, output[-14000:]
            result = json.loads(output.strip().splitlines()[-1])
            test_call_witness.child_reaped(process, result)
            completed = True
            return result
        finally:
            if not completed:
                # The observer can own a spawned child before this local receives
                # Popen's return value; cleanup must also survive profiler failure.
                test_call_witness.cleanup_child(
                    process,
                    interrupted=isinstance(sys.exc_info()[1], (KeyboardInterrupt, SystemExit, GeneratorExit)),
                )



@pytest.fixture(scope="module")
def isolated_reviewed_handoff(tmp_path_factory, config):
    """Real commits and actual source checks; intentionally non-authoritative."""
    import shutil
    from copy import deepcopy

    fixture_started = _phase_marker("isolated_reviewed_handoff", "START")
    root = tmp_path_factory.mktemp("non_authoritative_handoff")
    assert not root.is_relative_to(ROOT)
    contract = config[handoff.CONFIG_KEY]
    frozen_contract = contract["frozen_package"]
    # Current imports can extend the frozen handoff's historical source list.
    # Bind only this disposable prospective source contract to the actual closure.
    canonical, provenance = load_config_with_provenance(ROOT / contract["config_entrypoint"])
    assert canonical == config
    fixture_contract = deepcopy(contract)
    fixture_contract["source_bindings"] = sorted(
        set(provenance["include_order"])
        | set(contract["source_metadata_bindings"])
        | set(handoff._source_closure(ROOT, "HEAD", contract["source_roots"], current=True))
    )
    paths = set(fixture_contract["source_bindings"])
    paths.update(frozen_contract["publications"].values())
    paths.update(frozen_contract[key] for key in ("review_path", "closeout_path", "pre_change_path", "source_gate_path"))
    paths.add(contract["evidence_inputs"]["pre_change_identity_manifest"])
    # Every copied path is explicit code/config or previously authenticated metadata.
    assert not any(path.startswith("Input/") for path in paths)
    for relative in sorted(paths):
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _phase_marker("isolated_reviewed_handoff", "EXPLICIT_FILES_COPIED", fixture_started)
    def git(*argv, input=None):
        return subprocess.run(["git", *argv], cwd=root, input=input, text=True,
                              capture_output=True, check=True).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "NeuronSifter non-authoritative fixture")
    git("config", "user.email", "fixture@invalid.example")
    objects = handoff._git(ROOT, "rev-parse", "--git-path", "objects").decode().strip()
    (root / ".git/objects/info/alternates").write_text(str((ROOT / objects).resolve()) + "\n")
    _phase_marker("isolated_reviewed_handoff", "GIT_INITIALIZED", fixture_started)
    fixture_contract["source_checks"] = [
        {"name": "fixture_python_syntax",
         "argv": [sys.executable, "-m", "py_compile", *sorted(path for path in paths if path.endswith(".py"))]},
        {"name": "fixture_actual_source_adjudication",
         "argv": [sys.executable, "-m", "Libs.scripts.validators.validate_operational_twin_successor_execution_handoff", "--mode", "source"]},
    ]
    contract_path = "Libs/configs/exp/operational_twin_successor_execution_handoff.yaml"
    schema_path = "Libs/configs/schema/operational_twin_successor_execution_handoff.schema.yaml"
    (root / contract_path).write_text(json.dumps({handoff.CONFIG_KEY: fixture_contract}, indent=2) + "\n")
    fixture_schema = deepcopy(config[handoff.SCHEMA_KEY])
    fixture_schema["required_config_values"] = fixture_contract
    (root / schema_path).write_text(json.dumps({handoff.SCHEMA_KEY: fixture_schema}, indent=2) + "\n")
    git("add", "--", *sorted(paths))
    tree = git("write-tree")
    parent = handoff._git(ROOT, "rev-parse", "HEAD").decode().strip()
    source_commit = git("commit-tree", tree, "-p", parent, "-m", "Non-authoritative test source snapshot")
    git("update-ref", "HEAD", source_commit)
    assert git("status", "--porcelain", "--untracked-files=no") == ""
    _phase_marker("isolated_reviewed_handoff", "SOURCE_COMMIT_READY", fixture_started)
    receipt = _isolated_python(root, """
import json
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.scripts.validators.validate_operational_twin_successor_execution_handoff import collect_source_gate
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
receipt = collect_source_gate(config, root)
if receipt["status"] != "PASS":
    contract = config["operational_twin_successor_execution_handoff"]
    report_path = root / contract["evidence_inputs"]["source_gate"]
    diagnostic = {"event": "isolated_source_gate_blocked_diagnostic", "report_path": str(report_path),
        "report": report_path.read_text() if report_path.is_file() and not report_path.is_symlink()
            else {"missing_or_nonregular_path": str(report_path)}, "checks": []}
    for check in receipt["checks"]:
        log_path = root / check["log_path"]
        diagnostic["checks"].append({"name": check["name"], "returncode": check["returncode"],
            "log_path": str(log_path), "combined_stdout_stderr": log_path.read_text()
                if log_path.is_file() and not log_path.is_symlink()
                else {"missing_or_nonregular_path": str(log_path)}})
    print(json.dumps(diagnostic), flush=True)
print(json.dumps({"status": receipt["status"], "checks": [(row["name"], row["returncode"]) for row in receipt["checks"]]}))
""", stage="source gate")
    assert receipt["status"] == "PASS", receipt
    _phase_marker("isolated_reviewed_handoff", "SOURCE_GATE_READY", fixture_started)
    publication = _isolated_python(root, """
import json
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.scripts.builders.build_operational_twin_training_protocol import _build_candidate_successor_execution_handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
contract = config["operational_twin_successor_execution_handoff"]
result = _build_candidate_successor_execution_handoff(
    config, contract["authoritative_date"],
    pre_change_identity_manifest_path=root / contract["evidence_inputs"]["pre_change_identity_manifest"],
    validation_evidence_path=root / contract["evidence_inputs"]["source_gate"])
print(json.dumps(result))
""", stage="inert handoff publication")
    assert publication["candidate_dispatch_authorized"] is False
    _phase_marker("isolated_reviewed_handoff", "PUBLICATION_READY", fixture_started)
    gate_path = fixture_contract["evidence_inputs"]["source_gate"]
    gate = json.loads((root / gate_path).read_text())
    publication_paths = list(fixture_contract["output_paths"].values())
    git("add", "--", *publication_paths, gate_path, *[row["log_path"] for row in gate["checks"]])
    git("commit", "-q", "-m", "Non-authoritative inert handoff artifacts")
    artifact_commit = git("rev-parse", "HEAD")
    _phase_marker("isolated_reviewed_handoff", "ARTIFACT_COMMIT_READY", fixture_started)
    identity = _isolated_python(root, """
import json
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation.operational_twin_successor_execution_handoff import publication_binding_uid, validate_materialized_handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
manifest = validate_materialized_handoff(config, root)
print(json.dumps({"publication_binding": publication_binding_uid(config, root),
                  "handoff_uid": manifest["handoff_uid"],
                  "frozen_package_binding_uid": manifest["frozen_package_binding_uid"],
                  "terminal_decision": manifest["terminal_decision"]}))
""", stage="materialized handoff validation")
    review = {key: value for key, value in fixture_contract["review_contract"].items() if key != "path"}
    review["evidence_summary"] = "Non-authoritative isolated test review: " + " ".join([
        identity["terminal_decision"], identity["handoff_uid"], identity["publication_binding"],
        source_commit, identity["frozen_package_binding_uid"],
    ])
    review["roadmap_closeout"] = {
        "mode": "status-only", "status_changes": [], "completion_pointers": [],
        "history_entries": [], "semantic_update_required_reason": None,
    }
    review_path = root / fixture_contract["review_contract"]["path"]
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    git("config", "user.name", "NeuronSifter isolated reviewer fixture")
    git("add", "--", review_path.relative_to(root).as_posix())
    git("commit", "-q", "-m", "Non-authoritative independent-stage review fixture")
    _phase_marker("isolated_reviewed_handoff", "END", fixture_started)
    return {"root": root, "source_commit": source_commit,
            "artifact_commit": artifact_commit, "review_commit": git("rev-parse", "HEAD"),
            "handoff_uid": identity["handoff_uid"]}


def test_isolated_real_git_handoff_approval_is_inert_and_resolves_all_slots(isolated_reviewed_handoff):
    fixture = isolated_reviewed_handoff
    result = _isolated_python(fixture["root"], """
import json
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation.operational_twin_candidate_execution import resolve_candidate_execution_protocol
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
resolved = resolve_candidate_execution_protocol(config, root)
print(json.dumps({"handoff_uid": resolved.binding["handoff_uid"],
                  "review_commit": resolved.binding["handoff_review_commit"],
                  "assignment_count": len(resolved.assignment_manifest["assignments"]),
                  "command_count": len(resolved.command_catalog["commands"]),
                  "dispatch": resolved.handoff["candidate_dispatch_authorized"],
                  "scientific_twin": resolved.handoff["scientific_twin_result"],
                  "counters": resolved.handoff["access_counters"]}))
""", stage="approved protocol resolution")
    assert result["handoff_uid"] == fixture["handoff_uid"]
    assert result["review_commit"] == fixture["review_commit"]
    assert result["assignment_count"] == 15 and result["command_count"] == 30
    assert result["dispatch"] is False and result["scientific_twin"] == "NOT_RUN"
    assert set(result["counters"].values()) == {0}


def test_frozen_replay_cache_rechecks_current_publication_bytes(isolated_reviewed_handoff):
    root = isolated_reviewed_handoff["root"]
    config = load_config_stack(root / "Libs/configs/config_stack.yaml")
    first = handoff.authenticate_frozen_successor_package(config, root)
    publication = root / config[handoff.CONFIG_KEY]["frozen_package"]["publications"]["assignment_manifest"]
    original = publication.read_bytes()
    publication.write_bytes(original + b"\n")
    try:
        with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_committed_bytes_drift"):
            handoff.authenticate_frozen_successor_package(config, root)
    finally:
        publication.write_bytes(original)
    assert handoff.authenticate_frozen_successor_package(config, root) == first


def test_isolated_reviewed_handoff_current_dependency_closure(isolated_reviewed_handoff):
    fixture = isolated_reviewed_handoff
    result = _isolated_python(fixture["root"], "source_commit = " + repr(fixture["source_commit"]) + "\n" + """
import json
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
identity = handoff.executing_source_identity(config, root, source_commit)
paths = [row["path"] for row in identity["source_files"]]
assert "Libs/utils/exp/experiment_manifests.py" in paths
assert not any(path.startswith("Input/") for path in paths)
print(json.dumps({"status": "PASS", "source_files": len(paths)}))
""", stage="current fixture dependency closure")
    assert result["status"] == "PASS" and result["source_files"] > 0


def test_isolated_reviewed_handoff_missing_current_dependency_rejected(isolated_reviewed_handoff, tmp_path):
    fixture = isolated_reviewed_handoff
    root = _copy_reviewed_handoff(fixture, tmp_path / "missing_current_dependency")
    (root / "Libs/utils/exp/experiment_manifests.py").unlink()
    result = _isolated_python(root, "source_commit = " + repr(fixture["source_commit"]) + "\n" + """
import json
from pathlib import Path
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
from Libs.utils.data.raw_data_registry import load_config
root = Path.cwd()
config = load_config(root / "Libs/configs/config_stack.yaml")
try:
    handoff.executing_source_identity(config, root, source_commit)
except (handoff.OperationalTwinProtocolError, FileNotFoundError) as error:
    print(json.dumps({"status": "REJECTED", "error": str(error)}))
else:
    raise AssertionError("Missing required current dependency was accepted")
""", stage="missing current fixture dependency")
    assert result["status"] == "REJECTED"
    assert "experiment_manifests.py" in result["error"]


def test_loaded_source_accepts_current_dataclass_fields(isolated_reviewed_handoff):
    result = _isolated_python(isolated_reviewed_handoff["root"], """
import json
from pathlib import Path
from Libs.models.modules.world_state import JointWorldStatePosterior
from Libs.models.modules.native_frozen_rhs_solver import NativeRHSContinuation
from Libs.utils.model.native_training_selection import NativeTrainingSelection
from Libs.utils.model.native_held_source import HeldCNSLaw
from Libs.utils.model.native_conditional_readout import AdaptiveReadoutReference
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
handoff._check_loaded_source(root, config[handoff.CONFIG_KEY]["source_bindings"])
print(json.dumps({"status": "PASS", "classes": [JointWorldStatePosterior.__name__,
    NativeRHSContinuation.__name__, NativeTrainingSelection.__name__, HeldCNSLaw.__name__, AdaptiveReadoutReference.__name__]}))
""", stage="actual source-declared dataclass fields")
    assert result == {"status": "PASS", "classes": ["JointWorldStatePosterior", "NativeRHSContinuation", "NativeTrainingSelection", "HeldCNSLaw", "AdaptiveReadoutReference"]}


@pytest.mark.parametrize("field_name,attribute", [
    ("weights", "init"), ("weights", "repr"), ("weights", "compare"),
    ("weights", "default"), ("history", "default_factory"),
])
def test_loaded_source_rejects_current_dataclass_field_metadata(isolated_reviewed_handoff, field_name, attribute):
    result = _isolated_python(isolated_reviewed_handoff["root"],
        "field_name = " + repr(field_name) + "\nattribute = " + repr(attribute) + "\n" + """
import dataclasses, json
from pathlib import Path
from Libs.models.modules.world_state import JointWorldStatePosterior
from Libs.utils.model.native_training_selection import NativeTrainingSelection
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
owner = JointWorldStatePosterior if field_name == "weights" else NativeTrainingSelection
field = owner.__dataclass_fields__[field_name]
original = getattr(field, attribute)
setattr(field, attribute, tuple if attribute == "default_factory" else 0 if attribute == "default" else True)
try:
    handoff._check_loaded_source(root, config[handoff.CONFIG_KEY]["source_bindings"])
except handoff.OperationalTwinProtocolError as error:
    print(json.dumps({"status": "REJECTED", "error": str(error)}))
else:
    raise AssertionError("Changed actual dataclass field was accepted")
finally:
    setattr(field, attribute, original)
""", stage="actual dataclass metadata mutation")
    assert result["status"] == "REJECTED" and "handoff_dataclass_field_source_drift" in result["error"]


@pytest.mark.parametrize("surface", ["factory_closure", "defaults", "kwdefaults"])
def test_loaded_source_rejects_current_generated_dataclass_execution(isolated_reviewed_handoff, surface):
    result = _isolated_python(isolated_reviewed_handoff["root"], "surface = " + repr(surface) + "\n" + """
import json, types
from pathlib import Path
from Libs.utils.model.native_training_selection import NativeTrainingSelection
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
original = NativeTrainingSelection.__init__
assert NativeTrainingSelection.__dataclass_fields__["history"].default_factory is list
if surface == "factory_closure":
    def cell(value):
        return (lambda: value).__closure__[0]
    factory_cells = [index for index, old in enumerate(original.__closure__)
        if old.cell_contents is list]
    assert len(factory_cells) == 1
    closure = tuple(cell(tuple) if index == factory_cells[0] else old
        for index, old in enumerate(original.__closure__))
    changed = types.FunctionType(original.__code__, original.__globals__, original.__name__,
        original.__defaults__, closure)
    changed.__kwdefaults__ = original.__kwdefaults__
elif surface == "defaults":
    changed = types.FunctionType(original.__code__, original.__globals__, original.__name__,
        original.__defaults__[:-1] + ("changed",), original.__closure__)
    changed.__kwdefaults__ = original.__kwdefaults__
else:
    changed = types.FunctionType(original.__code__, original.__globals__, original.__name__,
        original.__defaults__, original.__closure__)
    changed.__kwdefaults__ = {"history": tuple}
NativeTrainingSelection.__init__ = changed
try:
    handoff._check_loaded_source(root, config[handoff.CONFIG_KEY]["source_bindings"])
except handoff.OperationalTwinProtocolError as error:
    print(json.dumps({"status": "REJECTED", "error": str(error)}))
else:
    raise AssertionError("Changed generated dataclass execution was accepted")
finally:
    NativeTrainingSelection.__init__ = original
""", stage="actual generated dataclass execution mutation")
    assert result["status"] == "REJECTED"
    expected = "closure_drift" if surface == "factory_closure" else "defaults_drift"
    assert "handoff_dataclass_generated_" + expected in result["error"]


def test_loaded_source_rejects_inherited_dataclass_metadata_shadow(isolated_reviewed_handoff):
    result = _isolated_python(isolated_reviewed_handoff["root"], """
import json
from pathlib import Path
from Libs.utils.model.native_held_source import HeldCNSLaw
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
assert "__dataclass_fields__" not in vars(HeldCNSLaw)
HeldCNSLaw.__dataclass_fields__ = dict(HeldCNSLaw.__dataclass_fields__)
try:
    handoff._check_loaded_source(root, config[handoff.CONFIG_KEY]["source_bindings"])
except handoff.OperationalTwinProtocolError as error:
    print(json.dumps({"status": "REJECTED", "error": str(error)}))
else:
    raise AssertionError("Undeclared subclass dataclass metadata was accepted")
finally:
    del HeldCNSLaw.__dataclass_fields__
""", stage="inherited dataclass metadata shadow")
    assert result["status"] == "REJECTED" and "handoff_dataclass_field_source_drift" in result["error"]



@pytest.mark.parametrize("surface", ["dataclass_base", "ordinary_base", "base_and_module_alias"])
def test_loaded_source_rejects_inherited_source_base_identity(surface):
    result = _isolated_python(ROOT, "surface = " + repr(surface) + "\n" + """
import json, types
from pathlib import Path
from Libs.utils.model import native_held_source as held
from Libs.utils.model import controlled_cns_source as cns
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
original_bases, original_alias = held.HeldCNSLaw.__bases__, held.cns
original_rhs = cns.ControlledLaw.rhs
assert original_bases == (cns.ControlledLaw,) and original_alias is cns
if surface == "ordinary_base":
    replacement = type("UnregisteredLaw", (object,), {})
else:
    namespace = {key: value for key, value in vars(cns.ControlledLaw).items()
                 if key not in {"__dict__", "__weakref__"}}
    replacement = type("UnregisteredLaw", cns.ControlledLaw.__bases__, namespace)
    replacement.rhs = lambda *args, **kwargs: None
    assert replacement.__dataclass_fields__ is cns.ControlledLaw.__dataclass_fields__
    assert replacement.rhs is not original_rhs
if surface == "base_and_module_alias":
    alias = types.ModuleType(cns.__name__)
    vars(alias).update(vars(cns))
    alias.ControlledLaw = replacement
    held.cns = alias
held.HeldCNSLaw.__bases__ = (replacement,)
try:
    assert cns.ControlledLaw.rhs is original_rhs
    source_paths = handoff._source_closure(root, "HEAD", config[handoff.CONFIG_KEY]["source_roots"], current=True)
    try:
        handoff._check_loaded_source(root, source_paths)
    except handoff.OperationalTwinProtocolError as error:
        expected = "handoff_source_base_alias_drift" if surface == "base_and_module_alias" else "handoff_source_base_identity_drift"
        assert expected in str(error), str(error)
        print(json.dumps({"status": "REJECTED", "surface": surface, "error": str(error)}))
    else:
        raise AssertionError("Changed actual inherited source base was accepted")
finally:
    held.HeldCNSLaw.__bases__ = original_bases
    held.cns = original_alias
    assert cns.ControlledLaw.rhs is original_rhs
""", stage="actual inherited source base mutation")
    assert result["status"] == "REJECTED" and result["surface"] == surface


def test_loaded_source_rejects_current_dataclass_default_constant(isolated_reviewed_handoff):
    result = _isolated_python(isolated_reviewed_handoff["root"], """
import json
from pathlib import Path
from Libs.utils.model import native_conditional_readout as readout
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
original = readout.ADAPTIVE_READOUT_LAW
readout.ADAPTIVE_READOUT_LAW = "changed"
try:
    handoff._check_loaded_source(root, config[handoff.CONFIG_KEY]["source_bindings"])
except handoff.OperationalTwinProtocolError as error:
    print(json.dumps({"status": "REJECTED", "error": str(error)}))
else:
    raise AssertionError("Changed source-declared dataclass default constant was accepted")
finally:
    readout.ADAPTIVE_READOUT_LAW = original
""", stage="source dataclass default constant mutation")
    assert result["status"] == "REJECTED" and "handoff_dataclass_default_constant_drift" in result["error"]


def _copy_reviewed_handoff(fixture, destination):
    import shutil
    root = destination / "non_authoritative_execution"
    shutil.copytree(fixture["root"], root)
    return root


def _run_rejected_handoff_control(isolated_reviewed_handoff, tmp_path, source, *, stage):
    """Run a rejection control against its own committed review and copied modules."""
    root = _copy_reviewed_handoff(isolated_reviewed_handoff, tmp_path)
    expected_parent = isolated_reviewed_handoff["review_commit"]
    result = _isolated_python(root, "approved_fixture_commit = " + repr(expected_parent) + "\n" + r"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
from Libs.utils.exp_utils import load_config_stack

ROOT = Path.cwd()
config = load_config_stack(ROOT / "Libs/configs/config_stack.yaml")

def git(*argv):
    return subprocess.run(["git", *argv], cwd=ROOT, text=True,
                          capture_output=True, check=True).stdout.strip()

assert git("rev-parse", "HEAD") == approved_fixture_commit
assert git("status", "--porcelain", "--untracked-files=no") == ""
# The maintained approval path must pass before this fixture can test rejection.
approved = handoff.require_approved_successor_execution_handoff(config, ROOT)
assert approved["handoff_review_commit"] == approved_fixture_commit
assert approved["candidate_dispatch_authorized"] is False
assert approved["scientific_twin_result"] == "NOT_RUN"
review_relative = config[handoff.CONFIG_KEY]["review_contract"]["path"]
review_path = ROOT / review_relative
approved_bytes = review_path.read_bytes()
approved_review = json.loads(approved_bytes)
assert approved_review["decision"] == "approved"
assert approved["handoff_review_identity"]["sha256"] == hashlib.sha256(approved_bytes).hexdigest()

# This is a new, non-authoritative fixture commit. Canonical review bytes stay unchanged.
review = dict(approved_review)
review["decision"] = "rejected"
review["retry_target"] = "implement"
review["required_changes"] = [
    "Problem: this non-authoritative control fixture withholds handoff approval. "
    "Suggested fix: obtain an independent approved fixture review before dispatch."
]
review.pop("roadmap_closeout")
assert review["schema_version"] == "review-record-v4"
assert review["retry_target"] == "implement" and review["required_changes"]
assert "roadmap_closeout" not in review
assert {key: value for key, value in review.items()
        if key not in {"decision", "retry_target", "required_changes"}} == {
    key: value for key, value in approved_review.items()
    if key not in {"decision", "roadmap_closeout"}
}
review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
rejected_bytes = review_path.read_bytes()
git("add", "--", review_relative)
assert git("diff", "--cached", "--name-only").splitlines() == [review_relative]
git("commit", "-q", "-m", "Non-authoritative committed rejected handoff control")
rejected_commit = git("rev-parse", "HEAD")
assert rejected_commit != approved_fixture_commit
assert git("rev-parse", "HEAD^") == approved_fixture_commit
assert git("diff", "--name-only", "HEAD^", "HEAD").splitlines() == [review_relative]
assert subprocess.check_output(["git", "show", "HEAD:" + review_relative], cwd=ROOT) == rejected_bytes
review_identity = handoff._identity(ROOT, "HEAD", review_relative, current=True)
assert review_identity["git_object_id"] == git("rev-parse", "HEAD:" + review_relative)
assert review_identity["sha256"] == hashlib.sha256(rejected_bytes).hexdigest()
assert json.loads(rejected_bytes)["decision"] == "rejected"
""" + source + r"""
assert review_path.read_bytes() == rejected_bytes
assert git("rev-parse", "HEAD") == rejected_commit
assert git("status", "--porcelain", "--untracked-files=no") == ""
print(json.dumps({"approved_review_commit": approved_fixture_commit,
                  "rejected_review_commit": rejected_commit,
                  "rejected_review_parent": git("rev-parse", "HEAD^"),
                  "review_identity": review_identity}))
""", stage=stage)
    assert result["approved_review_commit"] == result["rejected_review_parent"] == expected_parent
    assert result["rejected_review_commit"] != expected_parent
    return result



_SOURCE_GATE_RETENTION_FIXTURE = r"""
import hashlib, json, os, subprocess
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
from Libs.scripts.validators.validate_operational_twin_successor_execution_handoff import collect_source_gate
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
contract = config[handoff.CONFIG_KEY]
gate_path = root / contract["evidence_inputs"]["source_gate"]
def git(*args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
preserved = {}
for name in ("focused_handoff_tests", "canonical_source_adjudication", "changed_source_lint", "git_diff_check"):
    path = handoff._path(root, (gate_path.parent / (name + ".log")).relative_to(root).as_posix(), absent=True)
    with path.open("xb") as stream:
        stream.write(("Non-authoritative retained log fixture: " + name + "\n").encode())
    preserved[path.relative_to(root).as_posix()] = path.read_bytes()
def retain_gate(label):
    data = gate_path.read_bytes()
    receipt = json.loads(data)
    for check in receipt["checks"]:
        path = root / check["log_path"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == check["log_sha256"]
        preserved[check["log_path"]] = content
    retained = handoff._path(root, gate_path.with_name(label + "_retained_source_gate.json").relative_to(root).as_posix(), absent=True)
    with retained.open("xb") as stream:
        stream.write(data)
    git("add", "--", gate_path.relative_to(root).as_posix(), *[row["log_path"] for row in receipt["checks"]])
    gate_path.unlink()
    git("add", "-A", "--", gate_path.relative_to(root).as_posix(), retained.relative_to(root).as_posix(), *preserved)
    git("commit", "-q", "-m", "Retain non-authoritative source-gate evidence")
    assert git("status", "--porcelain", "--untracked-files=no") == ""
retain_gate("initial")
"""


def test_source_gate_logs_preserve_history_across_run_identities(isolated_reviewed_handoff, tmp_path):
    root = _copy_reviewed_handoff(isolated_reviewed_handoff, tmp_path)
    result = _isolated_python(root, _SOURCE_GATE_RETENTION_FIXTURE + r"""
runs = []
for label in ("first", "second"):
    source = git("rev-parse", "HEAD")
    receipt = collect_source_gate(config, root)
    assert receipt["status"] == "PASS"
    paths = []
    for check in receipt["checks"]:
        expected = gate_path.parent / (check["name"] + ".source_" + source + ".job_" + os.environ["SLURM_JOB_ID"] + ".log")
        assert root / check["log_path"] == expected
        assert len(source) == 40
        assert hashlib.sha256(expected.read_bytes()).hexdigest() == check["log_sha256"]
        paths.append(check["log_path"])
    assert all((root / path).read_bytes() == data for path, data in preserved.items())
    runs.append({"source": source, "paths": paths})
    if label == "first":
        retain_gate(label)
assert runs[0]["source"] != runs[1]["source"]
assert not set(runs[0]["paths"]) & set(runs[1]["paths"])
assert all((root / path).read_bytes() == data for path, data in preserved.items())
print(json.dumps({"runs": 2, "preserved_logs": len(preserved), "disjoint": True}))
""", stage="source-gate log retention")
    assert result["runs"] == 2 and result["disjoint"] is True
    assert result["preserved_logs"] >= 8


@pytest.mark.parametrize("collision", ["occupied", "leaf_symlink", "ancestor_symlink"])
def test_source_gate_log_destination_refuses_collisions(isolated_reviewed_handoff, tmp_path, collision):
    root = _copy_reviewed_handoff(isolated_reviewed_handoff, tmp_path)
    result = _isolated_python(root, _SOURCE_GATE_RETENTION_FIXTURE + "collision = " + repr(collision) + "\n" + r"""
source = git("rev-parse", "HEAD")
check = contract["source_checks"][0]
log = gate_path.parent / (check["name"] + ".source_" + source + ".job_" + os.environ["SLURM_JOB_ID"] + ".log")
if collision == "ancestor_symlink":
    original_parent = gate_path.parent
    target = root / "retained_fixture_logs"
    original_parent.rename(target)
    original_parent.symlink_to(target, target_is_directory=True)
    git("add", "-A", "--", original_parent.relative_to(root).as_posix(), target.relative_to(root).as_posix())
    git("commit", "-q", "-m", "Non-authoritative log-ancestor collision")
else:
    target = log if collision == "occupied" else root / "retained_fixture_target"
    with target.open("xb") as stream:
        stream.write(b"first writer remains unchanged")
    if collision == "leaf_symlink":
        log.symlink_to(target)
    target_bytes = target.read_bytes()
try:
    collect_source_gate(config, root)
except handoff.OperationalTwinProtocolError as error:
    assert ("occupied" if collision == "occupied" else "symlink") in str(error), error
else:
    raise AssertionError("source-gate log collision was accepted")
assert not gate_path.exists()
if collision != "ancestor_symlink":
    assert target.read_bytes() == target_bytes
assert all((root / path).read_bytes() == data for path, data in preserved.items())
print(json.dumps({"collision": collision, "rejected": True, "history_preserved": True}))
""", stage="source-gate log collision")
    assert result == {"collision": collision, "rejected": True, "history_preserved": True}


def _prepare_isolated_execution(root):
    return _isolated_python(root, """
import json, subprocess
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation.operational_twin_candidate_execution import (
    resolve_candidate_execution_protocol, validate_candidate_calibration_execution_authority,
    build_candidate_pre_dispatch_audit, write_json_exclusive,
)
from Libs.scripts.builders.build_operational_twin_calibration_execution_authority import build_authority
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
handoff = config["operational_twin_successor_execution_handoff"]
round_id = "round-52-execute-operational-twin-candidate-lineage-retry"
relative = "orchestrator/rounds/" + round_id
lineage = {key: handoff["review_contract"][key] for key in (
    "roadmap_id", "roadmap_revision", "roadmap_dir", "milestone_id")}
lineage.update(round_id=round_id, direction_id="execute-operational-twin-candidate-lineage-retry",
               extracted_item_id="non-authoritative-successor-execution-fixture")
selection = {"schema_version": "selection-record-v1", **lineage,
             "summary": "Non-authoritative isolated scheduler-boundary test",
             "scheduler": {"dispatch_mode": "serial", "max_parallel_rounds": 1}}
plan_record = {"schema_version": "round-plan-record-v1", **lineage, "plan_path": relative + "/plan.md",
               "worker_mode": "none", "workers": [], "integration": None}
round_dir = root / relative
round_dir.mkdir(parents=True)
(round_dir / "selection-record.json").write_text(json.dumps(selection))
(round_dir / "round-plan-record.json").write_text(json.dumps(plan_record))
(round_dir / "plan.md").write_text("# Non-authoritative test execution selection\\n\\nNo real scheduler/model operation is authorized.\\n")
subprocess.run(["git", "add", "--", relative], check=True)
subprocess.run(["git", "commit", "-q", "-m", "Non-authoritative later execution selection"], check=True)
resolved = resolve_candidate_execution_protocol(config, root)
authority = build_authority(round_dir, protocol_scope="candidate-lineage", root=root)
authority_path = root / resolved.execution["output_paths"]["calibration_execution_authority"]
write_json_exclusive(authority_path, authority)
blockers = validate_candidate_calibration_execution_authority(
    config, root, authority_path, resolved_context=resolved)
assert not blockers, blockers
audit = build_candidate_pre_dispatch_audit(config, root)
assert audit["status"] == "PASS", audit["blockers"]
write_json_exclusive(root / resolved.execution["output_paths"]["pre_dispatch_audit"], audit)
print(json.dumps({"status": "PASS", "round_id": authority["source_round_id"],
                  "source_binding_count": len(authority["source_files"]),
                  "source_commit": authority["source_commit"]}))
""", stage="later selection and calibration")


@pytest.fixture(scope="module")
def isolated_execution_ready(tmp_path_factory, isolated_reviewed_handoff):
    """Generate actual later authority once; each consumer revalidates a fresh copy."""
    fixture_started = _phase_marker("isolated_execution_ready", "START")
    destination = tmp_path_factory.mktemp("non_authoritative_execution_ready")
    root = _copy_reviewed_handoff(isolated_reviewed_handoff, destination)
    _phase_marker("isolated_execution_ready", "REPOSITORY_COPIED", fixture_started)
    prepared = _prepare_isolated_execution(root)
    _phase_marker("isolated_execution_ready", "END", fixture_started)
    return {"root": root, "prepared": prepared}


def test_isolated_later_selection_calibration_and_exclusive_dispatch(isolated_execution_ready, tmp_path):
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    prepared = isolated_execution_ready["prepared"]
    assert prepared["round_id"] == "round-52-execute-operational-twin-candidate-lineage-retry"
    assert prepared["source_binding_count"] >= 114
    result = _isolated_python(root, """
import json
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation.operational_twin_candidate_execution import resolve_candidate_execution_protocol
from Libs.scripts.builders.submit_operational_twin_candidate_execution import (submit_arrays, require_candidate_dispatch_manifest, require_candidate_release_manifest, require_candidate_release_intent)
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
resolved = resolve_candidate_execution_protocol(config, root)
calls = []
def scheduler_boundary(argv, current_root):
    assert current_root == root
    calls.append(argv)
    if argv[0] == "sinfo":
        return "cpu04|cpu*|mix|(null)\\ngpu03|gpu1|mix|gpu:4"
    if argv[0] == "sbatch":
        assert "--hold" in argv
        return str(765000 + len(calls))
    assert argv[:2] == ["scontrol", "release"]
    dispatch = json.loads((root / resolved.execution["output_paths"]["dispatch_manifest"]).read_text())
    require_candidate_dispatch_manifest(config, root, dispatch, mode="initial", resolved_context=resolved)
    assert len(dispatch["initial_arrays"]) == 5
    row = next(row for row in dispatch["initial_arrays"] if row["job_id"] == argv[2])
    require_candidate_release_intent(config, root, resolved, dispatch, row)
    return ""
dispatch = submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)
require_candidate_dispatch_manifest(config, root, dispatch, mode="initial", resolved_context=resolved)
release = require_candidate_release_manifest(config, root, resolved, dispatch)
assert release["status"] == "RELEASED"
assert [row["job_id"] for row in release["attempts"]] == [row["job_id"] for row in dispatch["initial_arrays"]]
count_before_duplicate = len(calls)
try:
    submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)
except ValueError:
    pass
else:
    raise AssertionError("duplicate dispatcher was admitted")
assert len(calls) == count_before_duplicate
assert all(not (root / row["output_root"]).exists() for assignment in resolved.assignment_manifest["assignments"] for row in assignment["attempts"])
print(json.dumps({"status": dispatch["status"], "arrays": len(dispatch["initial_arrays"]),
                  "indices": dispatch["selected_indices"], "scheduler_boundary_calls": len(calls),
                  "released_arrays": len(release["attempts"]), "production_calls": 0, "attempt_roots": 0}))
""", stage="exclusive initial dispatch")
    assert result == {
        "status": "INITIAL_ARRAYS_SUBMITTED", "arrays": 5, "indices": list(range(15)),
        "scheduler_boundary_calls": 11, "released_arrays": 5, "production_calls": 0, "attempt_roots": 0,
    }


def test_loaded_source_accepts_real_no_grad_wrapper_and_class_descriptors(config):
    # Authenticate real runtime imports in a fresh interpreter. Other collected
    # tests remain outside the production closure and must not be whitelisted.
    result = _isolated_python(ROOT, """
import json
from pathlib import Path
from Libs.exp.training import train_operational_twin_devcal
from Libs.models.modules.world_action_encoder import WorldActionEncoder
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
assert hasattr(train_operational_twin_devcal.evaluate, "__wrapped__")
assert WorldActionEncoder.__dict__
source_paths = handoff._source_closure(root, "HEAD", config[handoff.CONFIG_KEY]["source_roots"], current=True)
handoff._check_loaded_source(root, source_paths)
print(json.dumps({"status": "PASS"}))
""", stage="current loaded runtime source acceptance")
    assert result == {"status": "PASS"}


@pytest.mark.parametrize("surface", ["body", "wrapper", "wrapper_closure", "class_method", "generated_method"])
def test_loaded_source_rejects_actual_body_wrapper_closure_and_method_drift(config, surface):
    result = _isolated_python(ROOT, "surface = " + repr(surface) + "\n" + """
import json, types
from pathlib import Path
from Libs.exp.training import train_operational_twin_devcal as trainer
from Libs.models.modules.world_action_encoder import WorldActionEncoder
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
if surface == "body":
    trainer.evaluate.__wrapped__ = lambda *args, **kwargs: None
elif surface == "wrapper":
    def replacement(*args, **kwargs):
        return None
    replacement.__wrapped__ = trainer.evaluate.__wrapped__
    trainer.evaluate = replacement
elif surface == "wrapper_closure":
    original = trainer.evaluate
    def cell(value):
        return (lambda: value).__closure__[0]
    cells = tuple(cell(None) if isinstance(item.cell_contents, types.FunctionType) else item for item in original.__closure__)
    changed = types.FunctionType(original.__code__, original.__globals__, original.__name__, original.__defaults__, cells)
    changed.__wrapped__ = original.__wrapped__
    trainer.evaluate = changed
elif surface == "class_method":
    WorldActionEncoder.__init__ = lambda *args, **kwargs: None
else:
    from Libs.utils.evaluation.operational_twin_training_protocol import ResolvedCandidateExecutionProtocol
    ResolvedCandidateExecutionProtocol.__init__ = lambda *args, **kwargs: None
source_paths = handoff._source_closure(root, "HEAD", config[handoff.CONFIG_KEY]["source_roots"], current=True)
try:
    handoff._check_loaded_source(root, source_paths)
except handoff.OperationalTwinProtocolError as error:
    assert "stale_loaded_" in str(error) or "wrapper_closure_drift" in str(error), str(error)
    print(json.dumps({"status": "REJECTED", "surface": surface, "error": str(error)}))
else:
    raise AssertionError("Actual loaded-source mutation was accepted: " + surface)
""", stage="current loaded runtime source mutation")
    assert result["status"] == "REJECTED" and result["surface"] == surface


def _reference_per_path_source_closure(root: Path, ref: str, roots, *, current: bool = False) -> list[str]:
    """Independent per-path regression oracle; never used by production."""
    pending, observed = list(roots), set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.startswith("Libs/") or not path.endswith(".py"):
            handoff._fail("handoff_source_root_invalid", path)
        content = handoff._path(root, path).read_bytes() if current else handoff._blob(root, ref, path)
        observed.add(path)
        modules: set[str] = set()
        for node in ast.walk(ast.parse(content, filename=path)):
            if isinstance(node, ast.Import):
                modules.update(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = path[:-3].split("/")[:-node.level]
                    module = ".".join([*package, *([] if not node.module else [node.module])])
                else:
                    module = node.module or ""
                modules.add(module)
                modules.update(f"{module}.{item.name}" for item in node.names)
        for module in modules:
            if module.startswith(("utils.", "models.", "scripts.", "exp.")):
                module = "Libs." + module
            if not module.startswith("Libs."):
                continue
            parts = module.split(".")
            candidates = ["/".join(parts) + ".py", "/".join(parts) + "/__init__.py"]
            candidates.extend("/".join(parts[:i]) + "/__init__.py" for i in range(1, len(parts)))
            for candidate in candidates:
                if current:
                    exists = handoff._path(root, candidate).is_file()
                else:
                    result = subprocess.run(["git", "cat-file", "-e", f"{ref}:{candidate}"],
                                            cwd=root, capture_output=True, check=False)
                    exists = result.returncode == 0
                if exists and candidate not in observed:
                    pending.append(candidate)
    return sorted(observed)


@pytest.mark.parametrize("view", ["frozen", "executing"])
def test_source_closure_batch_equivalence_and_timing(isolated_reviewed_handoff, monkeypatch, view):
    root = isolated_reviewed_handoff["root"]
    contract = load_config_stack(root / "Libs/configs/config_stack.yaml")[handoff.CONFIG_KEY]
    ref = contract["frozen_package"]["source_commit"] if view == "frozen" else isolated_reviewed_handoff["source_commit"]
    roots = contract["frozen_view_source_roots"] if view == "frozen" else contract["source_roots"]
    actual_run, actual_blob = subprocess.run, handoff._blob
    samples = {}
    for mode, operation in (("per_path", _reference_per_path_source_closure), ("batch", handoff._source_closure)):
        probes, metadata, contents = {}, {}, {}
        calls = 0
        def observe_run(argv, *args, **kwargs):
            nonlocal calls
            result = actual_run(argv, *args, **kwargs)
            if argv[:3] == ["git", "cat-file", "-e"]:
                calls += 1
                probes[argv[3]] = result.returncode == 0
            elif argv[:2] == ["git", "cat-file"] and argv[2].startswith("--batch-check="):
                calls += 1
                specifications = kwargs["input"].splitlines()
                for specification, line in zip(specifications, result.stdout.splitlines(), strict=True):
                    present = line != specification + " missing"
                    probes[specification] = present
                    if present:
                        object_id, object_type, size = line.split()
                        metadata[specification] = (object_id, object_type, int(size))
            return result
        def observe_blob(current_root, current_ref, path):
            content = actual_blob(current_root, current_ref, path)
            assert current_root == root and current_ref == ref
            contents[current_ref + ":" + path] = {
                "git_object_id": hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest(),
                "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            }
            return content
        started = time.perf_counter()
        with monkeypatch.context() as instrumentation:
            instrumentation.setattr(subprocess, "run", observe_run)
            instrumentation.setattr(handoff, "_blob", observe_blob)
            closure = operation(root, ref, roots)
        samples[mode] = {"closure": closure, "probes": probes, "metadata": metadata,
                         "contents": contents, "calls": calls, "seconds": time.perf_counter() - started}
    baseline, batched = samples["per_path"], samples["batch"]
    assert baseline["closure"] == batched["closure"]
    assert baseline["probes"] == batched["probes"]
    assert baseline["contents"] == batched["contents"]
    assert set(batched["metadata"]) == {key for key, present in baseline["probes"].items() if present}
    for specification, (object_id, object_type, size) in batched["metadata"].items():
        content = baseline["contents"][specification]
        assert (object_id, object_type, size) == (content["git_object_id"], "blob", content["size_bytes"])
    assert batched["calls"] < baseline["calls"]
    print(json.dumps({
        "source_closure_equivalence": view, "ref": ref,
        "source_count": len(batched["closure"]), "candidate_count": len(batched["probes"]),
        "present_count": len(batched["metadata"]), "identities_equal": True,
        "per_path_calls": baseline["calls"], "batch_calls": batched["calls"],
        "per_path_seconds": baseline["seconds"], "batch_seconds": batched["seconds"],
    }), flush=True)


def _syntax_fixture_git(root, *argv):
    return subprocess.run(
        ["git", *argv], cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip()


def _syntax_fixture_commit(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _syntax_fixture_git(root, "init", "-q")
    _syntax_fixture_git(root, "config", "user.name", "Non-authoritative syntax fixture")
    _syntax_fixture_git(root, "config", "user.email", "fixture@invalid.example")
    _syntax_fixture_git(root, "add", "--", *files)
    _syntax_fixture_git(root, "commit", "-q", "-m", "Non-authoritative syntax fixture")
    return _syntax_fixture_git(root, "rev-parse", "HEAD")


def _observe_syntax_parser(monkeypatch):
    # A closure-free delegating observer keeps the supported context constant.
    namespace = {"actual_parse": ast.parse, "calls": []}
    exec(
        "def parse(source, filename='<unknown>', mode='exec', *, type_comments=False):\n"
        "    calls.append((source, filename))\n"
        "    return actual_parse(source, filename=filename, mode=mode, type_comments=type_comments)\n",
        namespace,
    )
    monkeypatch.setattr(ast, "parse", namespace["parse"])
    monkeypatch.setattr(handoff, "_SOURCE_SYNTAX_CONTEXT", None)
    return namespace


def test_source_syntax_reuse_matches_uncached_closure(tmp_path, monkeypatch):
    root = tmp_path / "syntax"
    roots = ["Libs/pkg/importer.py"]
    files = {
        roots[0]: (
            b"import Libs.absolute as alias\n"
            b"import utils.helper, models.helper, scripts.helper, exp.helper\n"
            b"from . import sibling\nfrom ..shared import value\n"
            b"if True:\n    from Libs.package import *\n"
            b"def nested():\n    from Libs.nested import thing\n"
        ),
        "Libs/pkg/__init__.py": b"from . import sibling\n",
        **{path: b"" for path in (
            "Libs/__init__.py", "Libs/absolute.py", "Libs/utils/helper.py",
            "Libs/models/helper.py", "Libs/scripts/helper.py", "Libs/exp/helper.py",
            "Libs/pkg/sibling.py", "Libs/shared.py", "Libs/package/__init__.py",
            "Libs/nested.py",
        )},
    }
    ref = _syntax_fixture_commit(root, files)
    expected = _reference_per_path_source_closure(root, ref, roots)
    assert expected == sorted(files)
    assert _reference_per_path_source_closure(root, ref, roots, current=True) == expected
    parser = _observe_syntax_parser(monkeypatch)
    reads, probes, existence = [], [], []
    actual_read, actual_probe, actual_exists = Path.read_bytes, handoff._source_candidate_objects, Path.is_file

    def read(path):
        reads.append(path)
        return actual_read(path)

    def probe(current_root, current_ref, paths):
        probes.append((current_root, current_ref, tuple(paths)))
        return actual_probe(current_root, current_ref, paths)

    def exists(path):
        existence.append(path)
        return actual_exists(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(handoff, "_source_candidate_objects", probe)
    monkeypatch.setattr(Path, "is_file", exists)
    for current in (True, False):
        monkeypatch.setattr(handoff, "_SOURCE_SYNTAX_CONTEXT", None)
        parser["calls"].clear()
        reads.clear()
        probes.clear()
        existence.clear()
        assert handoff._source_closure(root, ref, roots, current=current) == expected
        assert len(parser["calls"]) == len(expected)
        cold = (list(reads), list(probes), list(existence))
        assert handoff._source_closure(root, ref, roots, current=current) == expected
        assert len(parser["calls"]) == len(expected)
        assert (reads, probes, existence) == tuple(items * 2 for items in cold)
        assert len(cold[0]) == (len(expected) if current else 0)
        assert len(cold[1]) == (0 if current else len(expected))
        assert bool(cold[2]) is current


def test_source_syntax_reuse_rechecks_same_mtime_bytes_and_relative_path(tmp_path):
    first, second = "Libs/one/importer.py", "Libs/two/importer.py"
    before, after = b"from . import alpha\n", b"from . import bravo\n"
    for relative, content in {
        first: before, second: before,
        "Libs/one/alpha.py": b"", "Libs/one/bravo.py": b"", "Libs/two/alpha.py": b"",
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    assert len(before) == len(after)
    assert handoff._source_closure(tmp_path, "", [first], current=True) == ["Libs/one/alpha.py", first]
    timestamp = (tmp_path / first).stat()
    (tmp_path / first).write_bytes(after)
    os.utime(tmp_path / first, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    assert (tmp_path / first).stat().st_mtime_ns == timestamp.st_mtime_ns
    assert (tmp_path / first).stat().st_size == timestamp.st_size
    changed = handoff._source_closure(tmp_path, "", [first], current=True)
    assert changed == _reference_per_path_source_closure(tmp_path, "", [first], current=True)
    assert changed == ["Libs/one/bravo.py", first]
    assert handoff._source_syntax_candidates(before, first) != handoff._source_syntax_candidates(before, second)
    assert handoff._source_closure(tmp_path, "", [second], current=True) == (
        _reference_per_path_source_closure(tmp_path, "", [second], current=True)
    ) == ["Libs/two/alpha.py", second]


def test_source_syntax_reuse_rechecks_candidate_presence(tmp_path, monkeypatch):
    root = tmp_path / "changing"
    importer = "Libs/importer.py"
    content = b"from Libs.optional import value\n"
    absent_ref = _syntax_fixture_commit(root, {importer: content})
    assert handoff._source_closure(root, absent_ref, [importer], current=True) == [importer]
    additions = {"Libs/optional.py": b"", "Libs/optional/__init__.py": b"", "Libs/optional/value.py": b""}
    for relative, body in additions.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    _syntax_fixture_git(root, "add", "--", *additions)
    _syntax_fixture_git(root, "commit", "-q", "-m", "Optional candidates appear")
    present_ref = _syntax_fixture_git(root, "rev-parse", "HEAD")
    present = sorted([importer, *additions])
    assert handoff._source_closure(root, present_ref, [importer], current=True) == present
    other = tmp_path / "other"
    other_ref = _syntax_fixture_commit(other, {importer: content})
    assert handoff._source_closure(other, other_ref, [importer], current=True) == [importer]
    for relative in additions:
        (root / relative).unlink()
    _syntax_fixture_git(root, "add", "-u", "--", *additions)
    _syntax_fixture_git(root, "commit", "-q", "-m", "Optional candidates disappear")
    removed_ref = _syntax_fixture_git(root, "rev-parse", "HEAD")
    assert len({absent_ref, present_ref, removed_ref}) == 3
    assert handoff._source_closure(root, removed_ref, [importer], current=True) == [importer]
    actual_probe, probes = handoff._source_candidate_objects, []

    def probe(current_root, ref, paths):
        result = actual_probe(current_root, ref, paths)
        probes.append((current_root, ref, result))
        return result

    monkeypatch.setattr(handoff, "_source_candidate_objects", probe)
    for current_root, ref, expected in (
        (root, present_ref, present), (root, absent_ref, [importer]),
        (other, other_ref, [importer]), (root, removed_ref, [importer]),
        (root, present_ref, present),
    ):
        before = len(probes)
        assert handoff._source_closure(current_root, ref, [importer]) == expected
        assert expected == _reference_per_path_source_closure(current_root, ref, [importer])
        assert len(probes) - before == len(expected)
        importer_probe = probes[before][2]
        assert (importer_probe["Libs/optional.py"] is not None) is (expected == present)
    optional = root / "Libs/optional.py"
    optional.symlink_to(root / importer)
    with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_path_symlink"):
        handoff._source_closure(root, removed_ref, [importer], current=True)
    optional.unlink()
    os.link(root / importer, optional)
    with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_evidence_hardlink"):
        handoff._source_closure(root, removed_ref, [importer], current=True)


def test_source_syntax_reuse_context_and_error_lifecycle(monkeypatch):
    parser = _observe_syntax_parser(monkeypatch)
    source, path = b"import Libs.example\n", "Libs/importer.py"
    helper = handoff._source_syntax_candidates
    expected = helper(source, path)
    assert helper(source, path) == expected
    assert len(parser["calls"]) == 1
    observer = parser["parse"]
    for field, value in (
        ("__defaults__", ("changed-default", "exec")),
        ("__kwdefaults__", {"type_comments": True}),
        ("__code__", observer.__code__.replace(co_name="changed_parser")),
    ):
        monkeypatch.setattr(observer, field, value)
        before = len(parser["calls"])
        assert helper(source, path) == expected
        assert len(parser["calls"]) == before + 1
    monkeypatch.setattr(helper, "__code__", helper.__code__.replace(co_name="changed_helper"))
    before = len(parser["calls"])
    assert helper(source, path) == expected
    assert len(parser["calls"]) == before + 1
    walker = {"actual_walk": ast.walk, "calls": []}
    exec("def walk(node):\n    calls.append(node)\n    return actual_walk(node)\n", walker)
    monkeypatch.setattr(ast, "walk", walker["walk"])
    assert helper(source, path) == expected
    assert helper(source, path) == expected
    assert len(walker["calls"]) == 1
    monkeypatch.setattr(walker["walk"], "__code__", walker["walk"].__code__.replace(co_name="changed_walker"))
    assert helper(source, path) == expected
    assert len(walker["calls"]) == 2
    monkeypatch.setattr(ast, "walk", walker["actual_walk"])
    before = len(parser["calls"])
    assert helper(source, path) == expected
    assert len(parser["calls"]) == before + 1
    # Unsupported mutable defaults cannot retain an apparently equal result.
    monkeypatch.setattr(observer, "__kwdefaults__", {"type_comments": []})
    for _ in range(2):
        assert helper(source, path) == expected
        assert handoff._SOURCE_SYNTAX_CACHE == {}
    monkeypatch.setattr(observer, "__kwdefaults__", {"type_comments": False})
    for invalid in (b"from Libs. import\n", b"# coding: unknown-source-codec\n", b"\xff\n"):
        with pytest.raises(SyntaxError) as reference:
            parser["actual_parse"](invalid, filename=path)
        for _ in range(2):
            with pytest.raises(type(reference.value)) as actual:
                helper(invalid, path)
            assert actual.value.args == reference.value.args
            assert actual.value.filename == reference.value.filename == path
            assert (invalid, path) not in handoff._SOURCE_SYNTAX_CACHE
    assert helper(source, path) == expected
    assert (source, path) in handoff._SOURCE_SYNTAX_CACHE


def test_source_syntax_reuse_results_and_capacity(monkeypatch):
    parser = _observe_syntax_parser(monkeypatch)
    monkeypatch.setattr(handoff, "_SOURCE_SYNTAX_ENTRY_LIMIT", 2)
    monkeypatch.setattr(handoff, "_SOURCE_SYNTAX_PAYLOAD_LIMIT", 4096)
    monkeypatch.setattr(handoff, "_SOURCE_SYNTAX_CANDIDATE_LIMIT", 6)
    helper, path = handoff._source_syntax_candidates, "Libs/importer.py"
    source = b"import Libs.first\n"
    expected = helper(source, path)
    assert type(expected) is tuple and expected == tuple(sorted(set(expected)))
    with pytest.raises(TypeError):
        expected[0] = "changed"
    assert helper(source, path) == expected
    for name in (b"second", b"third"):
        helper(b"import Libs." + name + b"\n", path)
    assert len(handoff._SOURCE_SYNTAX_CACHE) == 1
    before = len(parser["calls"])
    assert helper(source, path) == expected
    assert len(parser["calls"]) == before + 1
    # Each entry fits, but their combined retained payload forces eviction.
    one_entry = handoff._SOURCE_SYNTAX_PAYLOAD_BYTES // 2
    monkeypatch.setattr(handoff, "_SOURCE_SYNTAX_PAYLOAD_LIMIT", one_entry + 100)
    handoff._SOURCE_SYNTAX_CACHE.clear()
    monkeypatch.setattr(handoff, "_SOURCE_SYNTAX_PAYLOAD_BYTES", 0)
    assert helper(source, path) == expected
    helper(b"import Libs.second\n", path)
    assert len(handoff._SOURCE_SYNTAX_CACHE) == 1
    oversized = b"#" + b"x" * 4096 + b"\n" + source
    for _ in range(2):
        assert helper(oversized, path) == expected
        assert (oversized, path) not in handoff._SOURCE_SYNTAX_CACHE
    before = len(parser["calls"])
    monkeypatch.setattr(handoff, "_SOURCE_SYNTAX_CANDIDATE_LIMIT", 2)
    for _ in range(2):
        assert helper(b"import Libs.other\n", path)
    assert len(parser["calls"]) == before + 2
    assert (b"import Libs.other\n", path) not in handoff._SOURCE_SYNTAX_CACHE
    retained = sum(
        sys.getsizeof(key) + sys.getsizeof(key[0]) + sys.getsizeof(key[1])
        + sys.getsizeof(value) + sum(sys.getsizeof(candidate) for candidate in value)
        for key, value in handoff._SOURCE_SYNTAX_CACHE.items()
    )
    assert retained == handoff._SOURCE_SYNTAX_PAYLOAD_BYTES <= handoff._SOURCE_SYNTAX_PAYLOAD_LIMIT
    assert len(handoff._SOURCE_SYNTAX_CACHE) <= handoff._SOURCE_SYNTAX_ENTRY_LIMIT


def test_source_syntax_reuse_warm_loaded_source_contract(tmp_path):
    """Component coverage: exact maintained bodies in an isolated Git package."""
    source = Path(handoff.__file__).read_text()
    tree = ast.parse(source)
    names = {"_fail", "_source_syntax_candidates", "_check_loaded_source"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    declarations = [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and any(isinstance(target, ast.Name) and target.id.startswith("_SOURCE_SYNTAX_")
                            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    assert {node.name for node in definitions} == names
    protocol = (ROOT / "Libs/utils/evaluation/operational_twin_training_protocol.py").read_text()
    error_class = next(node for node in ast.parse(protocol).body
                       if isinstance(node, ast.ClassDef) and node.name == "OperationalTwinProtocolError")
    component = (
        "from __future__ import annotations\n"
        "import ast, builtins, dataclasses, inspect, sys, types\n"
        "from pathlib import Path, PurePosixPath\n"
        "from typing import Mapping, Sequence\n\n"
        + ast.get_source_segment(protocol, error_class) + "\n\n"
        + "\n\n".join(ast.get_source_segment(source, node) for node in [*declarations, *definitions]) + "\n"
    )
    root = tmp_path / "component"
    relative = "Libs/syntax_component.py"
    _syntax_fixture_commit(root, {"Libs/__init__.py": b"", relative: component.encode()})
    script = r'''
import importlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from Libs import syntax_component as component
bound = ["Libs/__init__.py", "Libs/syntax_component.py"]
helper = component._source_syntax_candidates
assert not hasattr(helper, "__wrapped__")
expected = helper(b"import Libs.optional\n", "Libs/example.py")
assert helper(b"import Libs.optional\n", "Libs/example.py") == expected
assert component._SOURCE_SYNTAX_CACHE
checked = []
def observe(frame, event, arg):
    if (event == "call" and frame.f_code.co_name == "check_function"
            and frame.f_locals.get("qualified") == "_source_syntax_candidates"):
        checked.append(frame.f_locals["qualified"])
sys.setprofile(observe)
try:
    component._check_loaded_source(root, bound)
finally:
    sys.setprofile(None)
assert checked == ["_source_syntax_candidates"]
rejected = []
def reject(label, expected_error):
    try:
        component._check_loaded_source(root, bound)
    except component.OperationalTwinProtocolError as error:
        assert expected_error in str(error), str(error)
        rejected.append(label)
    else:
        raise AssertionError("loaded-source drift accepted: " + label)
code = helper.__code__
helper.__code__ = code.replace(co_name="drifted_helper")
try:
    reject("body", "handoff_stale_loaded_function")
finally:
    helper.__code__ = code
def wrapper(*args, **kwargs):
    return helper(*args, **kwargs)
wrapper.__wrapped__ = helper
component._source_syntax_candidates = wrapper
try:
    reject("wrapper", "handoff_unexpected_loaded_wrapper")
finally:
    component._source_syntax_candidates = helper
path = root / "Libs/syntax_component.py"
original = path.read_bytes()
path.write_bytes(original.replace(b"return candidates", b"return ()", 1))
try:
    reject("source", "handoff_stale_loaded_function")
finally:
    path.write_bytes(original)
importlib.reload(component)
assert component._SOURCE_SYNTAX_CACHE == {}
assert component._SOURCE_SYNTAX_CONTEXT is None
assert component._SOURCE_SYNTAX_PAYLOAD_BYTES == 0
component._check_loaded_source(root, bound)
print(json.dumps({"checked_helper": checked, "rejected": rejected, "reload_empty": True}))
'''
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(root)],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == {
        "checked_helper": ["_source_syntax_candidates"],
        "rejected": ["body", "wrapper", "source"], "reload_empty": True,
    }


def test_source_batch_preserves_optional_absence_and_required_object_failures(tmp_path):
    root = tmp_path / "source_metadata_only"
    (root / "Libs/tree.py").mkdir(parents=True)
    content = b"from Libs.optional import value\n"
    (root / "Libs/module.py").write_bytes(content)
    (root / "Libs/tree.py/member").write_text("not Python source")
    def git(*argv):
        return subprocess.run(["git", *argv], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "Non-authoritative source metadata fixture")
    git("config", "user.email", "fixture@invalid.example")
    git("add", "--", "Libs/module.py", "Libs/tree.py/member")
    git("commit", "-q", "-m", "Non-authoritative explicit Git object fixture")
    ref = git("rev-parse", "HEAD")
    metadata = handoff._source_candidate_objects(root, ref, ["Libs/module.py", "Libs/missing.py", "Libs/tree.py"])
    assert metadata["Libs/module.py"] == (git("rev-parse", ref + ":Libs/module.py"), "blob", len(content))
    assert metadata["Libs/missing.py"] is None
    assert metadata["Libs/tree.py"] == (git("rev-parse", ref + ":Libs/tree.py"), "tree", int(git("cat-file", "-s", ref + ":Libs/tree.py")))
    for operation in (_reference_per_path_source_closure, handoff._source_closure):
        assert operation(root, ref, ["Libs/module.py"]) == ["Libs/module.py"]
        with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_git_evidence_unavailable"):
            operation(root, ref, ["Libs/missing.py"])
        with pytest.raises(SyntaxError):
            operation(root, ref, ["Libs/tree.py"])


def test_exact_identity_batch_matches_per_path_and_rejects_current_drift(tmp_path):
    started = _phase_marker("exact_identity_batch", "START")
    root = tmp_path / "exact_identity_batch"
    first = root / "Libs/first.py"
    second = root / "Libs/config.yaml"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"value = 1\n")
    second.write_bytes(b"setting: exact\n")

    def git(*argv):
        return subprocess.run(
            ["git", *argv], cwd=root, text=True, capture_output=True, check=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Non-authoritative exact identity fixture")
    git("config", "user.email", "fixture@invalid.example")
    git("add", "--", "Libs/first.py", "Libs/config.yaml")
    git("commit", "-q", "-m", "Non-authoritative exact identity fixture")
    ref = git("rev-parse", "HEAD")
    _phase_marker("exact_identity_batch", "FIXTURE_COMMITTED", started)
    paths = ["Libs/config.yaml", "Libs/first.py"]
    expected = [handoff._identity(root, ref, path, current=True) for path in paths]
    _phase_marker("exact_identity_batch", "PER_PATH_READY", started)
    assert handoff._identities(root, ref, paths, current=True) == expected
    _phase_marker("exact_identity_batch", "BATCH_READY", started)

    first.write_bytes(b"value = 2\n")
    with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_committed_bytes_drift"):
        handoff._identities(root, ref, paths, current=True)
    first.write_bytes(b"value = 1\n")
    with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_identity_path_duplicate"):
        handoff._identities(root, ref, [paths[0], paths[0]], current=False)
    with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_identity_path_invalid"):
        handoff._identities(root, ref, ["Input/unopened.json"], current=False)
    _phase_marker("exact_identity_batch", "END", started)


def test_immutable_blob_cache_rechecks_worktree_bytes_and_new_commits(tmp_path, monkeypatch):
    root = tmp_path / "immutable_blob_cache"
    path = root / "Libs/module.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"value = 1\n")

    def git(*argv):
        return subprocess.run(
            ["git", *argv], cwd=root, text=True, capture_output=True, check=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Non-authoritative immutable cache fixture")
    git("config", "user.email", "fixture@invalid.example")
    git("add", "--", "Libs/module.py")
    git("commit", "-q", "-m", "Non-authoritative immutable cache fixture")
    first_ref = git("rev-parse", "HEAD")
    handoff._IMMUTABLE_BLOB_CACHE.clear()
    handoff._IMMUTABLE_OBJECT_CACHE.clear()
    actual_git = handoff._git
    calls = []

    def observe_git(current_root, *argv):
        calls.append(argv)
        return actual_git(current_root, *argv)

    monkeypatch.setattr(handoff, "_git", observe_git)
    first = handoff._identity(root, first_ref, "Libs/module.py", current=True)
    assert handoff._identity(root, first_ref, "Libs/module.py", current=True) == first
    assert calls == [
        ("show", first_ref + ":Libs/module.py"),
        ("rev-parse", "--verify", first_ref + ":Libs/module.py"),
    ]
    path.write_bytes(b"value = 2\n")
    with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_committed_bytes_drift"):
        handoff._identity(root, first_ref, "Libs/module.py", current=True)
    git("add", "--", "Libs/module.py")
    git("commit", "-q", "-m", "Non-authoritative immutable cache successor")
    second_ref = git("rev-parse", "HEAD")
    second = handoff._identity(root, second_ref, "Libs/module.py", current=True)
    assert second["git_object_id"] != first["git_object_id"]
    assert second["sha256"] != first["sha256"]
    handoff._IMMUTABLE_BLOB_CACHE.clear()
    handoff._IMMUTABLE_OBJECT_CACHE.clear()


@pytest.mark.parametrize("field,value", [
    ("ref", "HEAD"),
    ("ref", "--batch-all-objects"),
    ("ref", "a" * 40 + "\nHEAD"),
    ("ref", "a" * 39 + "\0"),
    ("path", "Libs/module.py\nHEAD:Input/unopened.py"),
    ("path", "Libs/module.py\r"),
    ("path", "Libs/module.py\0"),
    ("path", "Libs/module.py\t"),
    ("path", "Libs/../Input/unopened.py"),
    ("path", "Libs//module.py"),
    ("path", "Libs/module.py:Input/unopened.py"),
    ("path", "Libs\\module.py"),
    ("path", "Input/unopened.py"),
])
def test_source_batch_rejects_nonexact_query_before_git(tmp_path, field, value):
    ref, path = "a" * 40, "Libs/module.py"
    if field == "ref":
        ref = value
    else:
        path = value
    observation = {"enabled": True, "subprocess_calls": 0}
    def boundary(event, arguments):
        if observation["enabled"] and event == "subprocess.Popen":
            observation["subprocess_calls"] += 1
            raise PermissionError("unexpected Git query for a nonexact source candidate")
    sys.addaudithook(boundary)
    try:
        with pytest.raises(handoff.OperationalTwinProtocolError, match="handoff_source_candidate_invalid"):
            handoff._source_candidate_objects(tmp_path, ref, [path])
        assert observation["subprocess_calls"] == 0
    finally:
        observation["enabled"] = False


@pytest.mark.parametrize("scenario", [
    "submission_error", "duplicate_job", "submission_receipt_collision",
    "manifest_collision", "release_error", "release_unknown", "release_receipt_collision",
])
def test_isolated_held_submission_failure_preserves_owned_evidence(isolated_execution_ready, tmp_path, scenario):
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    result = _isolated_python(root, "scenario = " + repr(scenario) + "\n" + """
import json, subprocess
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation.operational_twin_candidate_execution import resolve_candidate_execution_protocol
from Libs.scripts.builders.submit_operational_twin_candidate_execution import (
    submit_arrays, require_candidate_dispatch_manifest, require_candidate_release_manifest,
    _release_owned_arrays,
)
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
resolved = resolve_candidate_execution_protocol(config, root)
paths = resolved.execution["output_paths"]
execution_root = root / paths["execution_root"]
dispatch_path = root / paths["dispatch_manifest"]
calls, returned_ids, release_ids = [], [], []
collision_path = None
def collide(path):
    global collision_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("non-authoritative foreign collision sentinel")
    collision_path = path
def scheduler_boundary(argv, current_root):
    assert current_root == root
    calls.append(argv)
    if argv[0] == "sinfo":
        return "cpu04|cpu*|mix|(null)\\ngpu03|gpu1|mix|gpu:4"
    if argv[0] == "sbatch":
        assert "--hold" in argv and not release_ids
        ordinal = sum(call[0] == "sbatch" for call in calls)
        if scenario == "submission_error" and ordinal == 3:
            raise subprocess.CalledProcessError(1, argv, output="", stderr="non-authoritative submission failure")
        if scenario == "submission_receipt_collision" and ordinal == 2:
            collide(execution_root / "dispatch_receipts/initial_linear_or_delta.json")
        if scenario == "manifest_collision" and ordinal == 5:
            collide(dispatch_path)
        job_id = returned_ids[0] if scenario == "duplicate_job" and ordinal == 2 else str(765100 + ordinal)
        returned_ids.append(job_id)
        return job_id
    assert argv[:2] == ["scontrol", "release"]
    assert dispatch_path.is_file()
    release_ids.append(argv[2])
    assert release_ids == returned_ids[:len(release_ids)]
    if scenario == "release_error" and len(release_ids) == 3:
        raise subprocess.CalledProcessError(1, argv, output="", stderr="non-authoritative release failure")
    if scenario == "release_unknown" and len(release_ids) == 1:
        raise OSError("non-authoritative unknown external release outcome")
    if scenario == "release_receipt_collision" and len(release_ids) == 1:
        collide(execution_root / "release_receipts/initial_persistence_no_action.json")
    return ""
try:
    submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)
except (ValueError, OSError, subprocess.CalledProcessError):
    pass
else:
    raise AssertionError("injected external/publication failure was admitted")
if collision_path is not None:
    assert collision_path.read_text() == "non-authoritative foreign collision sentinel"
assert (root / paths["dispatch_claim"]).is_file()
assert all(not (root / attempt["output_root"]).exists() for assignment in resolved.assignment_manifest["assignments"] for attempt in assignment["attempts"])
if scenario == "manifest_collision":
    assert len(returned_ids) == 5 and release_ids == []
    for group in resolved.execution["array_groups"]:
        assert (execution_root / ("dispatch_receipts/initial_" + group["group_id"] + ".json")).is_file()
    status, released = "MANIFEST_PUBLICATION_REJECTED", 0
else:
    dispatch = json.loads(dispatch_path.read_text())
    require_candidate_dispatch_manifest(config, root, dispatch, mode="initial", resolved_context=resolved)
    if scenario.startswith("release_"):
        release = require_candidate_release_manifest(config, root, resolved, dispatch)
        assert release["status"] == "PARTIAL_RELEASE_FAILED"
        if scenario == "release_error":
            assert [row["status"] for row in release["attempts"]] == ["RELEASED", "RELEASED", "FAILED"]
            assert len(release_ids) == 3
        elif scenario == "release_unknown":
            assert [row["status"] for row in release["attempts"]] == ["UNKNOWN"]
            assert release["attempts"][0]["returncode"] is None
        else:
            assert [row["status"] for row in release["attempts"]] == ["RELEASED"]
            assert release["attempts"][0]["receipt"] is None
        status, released = release["status"], sum(row["status"] == "RELEASED" for row in release["attempts"])
    else:
        assert dispatch["status"] == "PARTIAL_SUBMISSION_FAILED" and release_ids == []
        assert dispatch["submission_failure"]["error"]
        if scenario == "submission_receipt_collision":
            assert dispatch["submission_failure"]["job_id"] == returned_ids[-1]
        status, released = dispatch["status"], 0
    before = len(calls)
    try:
        _release_owned_arrays(config, root, resolved, dispatch_path, command_capture=scheduler_boundary)
    except (ValueError, OSError):
        pass
    else:
        raise AssertionError("duplicate/partial release was admitted")
    assert len(calls) == before
print(json.dumps({"scenario": scenario, "status": status, "returned_ids": returned_ids,
                  "release_calls": release_ids, "released": released, "production_calls": 0, "attempt_roots": 0}))
""", stage="held submission failure: " + scenario)
    assert result["scenario"] == scenario
    assert result["production_calls"] == result["attempt_roots"] == 0

def test_worker_starts_from_release_intent_before_resource_capture(isolated_execution_ready, tmp_path):
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    result = _isolated_python(root, """
import json, os, sys
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_candidate_execution as runtime
from Libs.utils.evaluation.operational_twin_training_protocol import (OperationalTwinProtocolError, candidate_runtime_slurm_identity, CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS, CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS, candidate_execution_binding_keys, _exact_candidate_execution_bindings)
from Libs.scripts.builders.submit_operational_twin_candidate_execution import (
    require_candidate_dispatch_manifest, require_candidate_release_manifest,
    submit_arrays, worker_binding,
)
from Libs.exp.training import run_operational_twin_candidate_attempt as worker
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
resolved = runtime.resolve_candidate_execution_protocol(config, root)
calls, observed, stopped, rejected = [], [], [], []
unpublished = [
    {key: None for key in candidate_execution_binding_keys(keys, resolved)}
    for keys in (CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS, CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS)
]
for envelope in unpublished:
    envelope["successor_execution_binding"] = dict(resolved.binding)
try:
    _exact_candidate_execution_bindings(*unpublished, resolved_context=resolved)
except OperationalTwinProtocolError as error:
    assert "candidate_execution_runtime_dispatch_binding_mismatch" in str(error)
else:
    raise AssertionError("unpublished allocation diagnostic was upgraded to execution authority")
class ResourceBoundary(Exception):
    pass
def trace(frame, event, value):
    if frame.f_code is runtime.require_candidate_dispatch_binding.__code__ and event == "return":
        if isinstance(value, dict):
            observed.append(value)
    if frame.f_code is runtime._capture_command.__code__ and event == "call":
        raise ResourceBoundary("genuine worker stopped before resource capture")
    return trace
def boundary(argv, current_root):
    assert current_root == root
    calls.append(argv)
    if argv[0] == "sinfo":
        return "cpu04|cpu*|mix|(null)\\ngpu03|gpu1|mix|gpu:4"
    if argv[0] == "sbatch":
        assert "--hold" in argv
        return str(765000 + len(calls))
    assert argv[:2] == ["scontrol", "release"]
    dispatch = json.loads((root / resolved.execution["output_paths"]["dispatch_manifest"]).read_text())
    row = next(row for row in dispatch["initial_arrays"] if row["job_id"] == argv[2])
    if row["command_indices"][0] != 0:
        return ""
    assert not (root / dispatch["release_manifest_path"]).exists()
    receipt = root / resolved.execution["output_paths"]["execution_root"] / "release_receipts" / ("initial_" + row["group_id"] + ".json")
    assert not receipt.exists()
    env = worker_binding(root, resolved, dispatch["source_commit"])
    env.update(SLURM_JOB_ID="865002", SLURM_ARRAY_JOB_ID=row["job_id"],
               SLURM_ARRAY_TASK_ID="0", SLURM_JOB_NAME=row["job_name"],
               SLURM_JOB_PARTITION="cpu", SLURM_JOB_NODELIST="cpu04",
               SLURM_CPUS_PER_TASK="2", SLURM_MEM_PER_NODE="8192",
               SLURM_JOB_GRES="", CUDA_VISIBLE_DEVICES="")
    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        assignment = resolved.assignment_manifest["assignments"][0]
        attempt = assignment["attempts"][0]
        identity = candidate_runtime_slurm_identity()
        for change, expected in [
            ({"slurm_array_task_id": "1", "resource_query_job_id": row["job_id"] + "_1"}, "candidate_dispatch_command_index_mismatch"),
            ({"slurm_array_job_id": "765999", "resource_query_job_id": "765999_0"}, "candidate_dispatch_runtime_array_mismatch"),
            ({"slurm_job_name": "NrnSift_foreign"}, "candidate_dispatch_runtime_array_mismatch"),
        ]:
            try:
                runtime.require_candidate_dispatch_binding(
                    config, root, resolved, assignment=assignment, attempt=attempt,
                    slurm={**identity, **change}, source_commit=dispatch["source_commit"],
                    validate_current_runtime=False)
            except OperationalTwinProtocolError as error:
                assert expected in str(error), str(error)
                rejected.append(expected)
            else:
                raise AssertionError("foreign runtime dispatch binding was admitted")
        old_trace = sys.gettrace()
        sys.settrace(trace)
        try:
            worker.run_attempt(0)
        except ResourceBoundary:
            stopped.append(0)
        else:
            raise AssertionError("worker crossed the terminating resource boundary")
        finally:
            sys.settrace(old_trace)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert len(observed) >= 2
    assert all(binding == observed[0] for binding in observed)
    proof = observed[0]
    assert proof["external_job_id"] == row["job_id"] + "_0"
    assert proof["slurm_job_id"] == "865002"
    assert proof["external_job_id"] != proof["slurm_job_id"]
    assert proof["array_submission_receipt"] == row["submission_receipt"]
    assert proof["release_intent"]["path"].endswith("initial_" + row["group_id"] + ".json")
    assert not receipt.exists()
    assert not (root / dispatch["release_manifest_path"]).exists()
    return ""
dispatch = submit_arrays(config, root, mode="initial", command_capture=boundary)
assert stopped == [0]
assert len(rejected) == 3
assert all(not (root / slot["output_root"]).exists()
           for assignment in resolved.assignment_manifest["assignments"] for slot in assignment["attempts"])
assert all(not (root / slot[field]).exists()
           for assignment in resolved.assignment_manifest["assignments"] for slot in assignment["attempts"]
           for field in ("execution_authority", "resource_environment_receipt"))
print(json.dumps({"status": "PRE_RESOURCE_BOUNDARY_VERIFIED", "worker_stops": len(stopped),
                  "negative_joins": len(rejected), "arrays": len(dispatch["initial_arrays"]),
                  "production_calls": 0, "attempt_roots": 0, "runtime_gates": 0}))
""", stage="actual worker at release intent")
    assert result == {
        "status": "PRE_RESOURCE_BOUNDARY_VERIFIED", "worker_stops": 1,
        "negative_joins": 3, "arrays": 5, "production_calls": 0,
        "attempt_roots": 0, "runtime_gates": 0,
    }


def test_retained_calibration_source_survives_metadata_only_commit(isolated_execution_ready, tmp_path):
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    result = _isolated_python(root, """
import json, subprocess
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation.operational_twin_candidate_execution import (
    resolve_candidate_execution_protocol, validate_candidate_calibration_execution_authority,
)
from Libs.exp.training.train_operational_twin_devcal import require_calibration_execution_authority
from Libs.utils.evaluation.operational_twin_training_protocol import OperationalTwinProtocolError
root = Path.cwd()
print("R51_CALLER_PHASE component=retained_calibration event=START", flush=True)
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
resolved = resolve_candidate_execution_protocol(config, root)
print("R51_CALLER_PHASE component=retained_calibration event=RESOLVED", flush=True)
path = root / resolved.execution["output_paths"]["calibration_execution_authority"]
authority = json.loads(path.read_text())
source = authority["source_commit"]
kwargs = dict(method_id="full_uncertainty_aware_action_conditioned",
              candidate_lineage_uid=resolved.protocol["candidate_lineage_uid"],
              protocol_uid=resolved.protocol["protocol_uid"], resolved_context=resolved)
before = require_calibration_execution_authority(config, root, path, **kwargs)
print("R51_CALLER_PHASE component=retained_calibration event=ORIGIN_AUTHORITY_VERIFIED", flush=True)
subprocess.run(["git", "commit", "-q", "--allow-empty", "-m",
                "Non-authoritative retained-source metadata-only descendant"], check=True)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert head != source
print("R51_CALLER_PHASE component=retained_calibration event=METADATA_DESCENDANT_COMMITTED", flush=True)
assert not validate_candidate_calibration_execution_authority(
    config, root, path, source_commit=source, resolved_context=resolved)
after = require_calibration_execution_authority(
    config, root, path, source_commit=source, **kwargs)
assert after == before
try:
    require_calibration_execution_authority(config, root, path, **kwargs)
except OperationalTwinProtocolError as error:
    assert "source_commit" in str(error), str(error)
else:
    raise AssertionError("live calibration accepted a stale source commit")

# The execution role is explicit; the authority is never rewritten to HEAD.
authority_bytes = path.read_bytes()
assert not validate_candidate_calibration_execution_authority(
    config, root, path, source_commit=source, executing_source_commit=head,
    resolved_context=resolved)
assert require_calibration_execution_authority(
    config, root, path, source_commit=source, executing_source_commit=head, **kwargs) == before
assert not validate_candidate_calibration_execution_authority(
    config, root, path, executing_source_commit=head, resolved_context=resolved)
assert require_calibration_execution_authority(
    config, root, path, executing_source_commit=head, **kwargs) == before
print("R51_CALLER_PHASE component=retained_calibration event=EXPLICIT_EXECUTING_BINDING_VERIFIED", flush=True)
from Libs.utils.evaluation.operational_twin_candidate_execution import (
    build_candidate_pre_dispatch_audit, require_candidate_calibration_source_bindings,
    write_json_exclusive,
)
from Libs.scripts.builders.submit_operational_twin_candidate_execution import (
    prepare_submission, submit_arrays, require_candidate_dispatch_manifest,
)
from Libs.scripts.validators import validate_operational_twin_calibration_execution_authority as authority_cli
import copy, sys
audit = build_candidate_pre_dispatch_audit(config, root)
assert audit["status"] == "PASS", audit["blockers"]
print("R51_CALLER_PHASE component=retained_calibration event=PREDISPATCH_AUDIT_BUILT", flush=True)
assert audit["source_commit"] == head
audit_path = root / resolved.execution["output_paths"]["pre_dispatch_audit"]
# This is a new initial-dispatch fixture at its metadata descendant. Only its
# untracked, unused pre-dispatch audit is rebuilt; permission stays immutable.
assert subprocess.run(["git", "cat-file", "-e", "HEAD:" + audit_path.relative_to(root).as_posix()],
                      capture_output=True).returncode != 0
audit_path.unlink()
write_json_exclusive(audit_path, audit)
prepared = prepare_submission(config, root, mode="initial")
assert prepared["source_commit"] == head
print("R51_CALLER_PHASE component=retained_calibration event=SUBMISSION_PREPARED", flush=True)
calls = []
def scheduler_boundary(argv, current_root):
    assert current_root == root
    calls.append(argv)
    if argv[0] == "sinfo":
        return "cpu04|cpu*|mix|(null)\\ngpu03|gpu1|mix|gpu:4"
    if argv[0] == "sbatch":
        assert "--hold" in argv
        return str(695000 + len(calls))
    assert argv[:2] == ["scontrol", "release"]
    return ""
dispatch = submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)
assert dispatch["source_commit"] == head != authority["source_commit"]
require_candidate_dispatch_manifest(config, root, dispatch, mode="initial", resolved_context=resolved)
print("R51_CALLER_PHASE component=retained_calibration event=INITIAL_DISPATCH_VALIDATED", flush=True)
old_argv = sys.argv
sys.argv = [authority_cli.__file__, "--protocol-scope", "candidate-lineage"]
try:
    assert authority_cli.main() == 0
finally:
    sys.argv = old_argv
assert path.read_bytes() == authority_bytes
print("R51_CALLER_PHASE component=retained_calibration event=AUTHORITY_CLI_VALIDATED", flush=True)
negative_checks = 0
def rejected(action, *, fragment=None):
    global negative_checks
    try:
        action()
    except (ValueError, KeyError) as error:
        if fragment is not None:
            assert fragment in str(error), error
        negative_checks += 1
    else:
        raise AssertionError("calibration source adversary was accepted")
bad_dispatch = copy.deepcopy(dispatch)
bad_dispatch["source_commit"] = source
rejected(lambda: require_candidate_dispatch_manifest(
    config, root, bad_dispatch, mode="initial", resolved_context=resolved),
    fragment="candidate_dispatch_claim_binding_mismatch")
rejected(lambda: require_calibration_execution_authority(
    config, root, path, source_commit=head, executing_source_commit=head, **kwargs),
    fragment="source_commit_mismatch")
tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
unrelated = subprocess.check_output(
    ["git", "commit-tree", tree], input="Non-authoritative unrelated ancestry fixture\\n", text=True).strip()
for execution_source in (unrelated, "f" * 40, ""):
    rejected(lambda execution_source=execution_source: require_calibration_execution_authority(
        config, root, path, source_commit=source, executing_source_commit=execution_source, **kwargs))
foreign = {**authority, "source_commit": unrelated}
rejected(lambda: require_candidate_calibration_source_bindings(
    config, root, foreign, resolved, authority["execution_selection"],
    source_commit=unrelated, executing_source_commit=head))
missing_origin = {key: value for key, value in authority.items() if key != "source_commit"}
rejected(lambda: require_candidate_calibration_source_bindings(
    config, root, missing_origin, resolved, authority["execution_selection"],
    source_commit=source, executing_source_commit=head))
changed_config = copy.deepcopy(config)
changed_config["operational_twin_candidate_execution"]["closed_flags"]["scientific_twin_result"] = "INVALID_FIXTURE"
rejected(lambda: require_candidate_calibration_source_bindings(
    changed_config, root, authority, resolved, authority["execution_selection"],
    source_commit=source, executing_source_commit=head),
    fragment="handoff_noncanonical_config")
changed_path = root / "Libs/exp/training/train_operational_twin_devcal.py"
original = changed_path.read_bytes()
changed_path.write_bytes(original + b"\\n# Non-authoritative calibration-source drift fixture.\\n")
try:
    rejected(lambda: require_calibration_execution_authority(
        config, root, path, source_commit=source, executing_source_commit=head, **kwargs),
        fragment="drift")
    subprocess.run(["git", "add", "--", changed_path.relative_to(root).as_posix()], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Non-authoritative committed source drift fixture"], check=True)
    drift_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rejected(lambda: require_calibration_execution_authority(
        config, root, path, source_commit=source, executing_source_commit=drift_head, **kwargs),
        fragment="drift")
finally:
    changed_path.write_bytes(original)
    subprocess.run(["git", "add", "--", changed_path.relative_to(root).as_posix()], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Restore non-authoritative fixture source bytes"], check=True)
assert negative_checks == 10
print("R51_CALLER_PHASE component=retained_calibration event=ADVERSARIES_REJECTED", flush=True)
assert path.read_bytes() == authority_bytes
assert all(not (root / slot["output_root"]).exists()
           for assignment in resolved.assignment_manifest["assignments"] for slot in assignment["attempts"])

print(json.dumps({"status": "RETAINED_ORIGIN_AND_EXPLICIT_EXECUTING_CALLERS_VERIFIED",
                  "metadata_only_descendant": head != source, "binding_unchanged": before == after,
                  "finite_execution_callers": 5, "source_adversaries": negative_checks,
                  "production_calls": 0, "attempt_roots": 0}))
""", stage="retained calibration source")
    assert result == {
        "status": "RETAINED_ORIGIN_AND_EXPLICIT_EXECUTING_CALLERS_VERIFIED",
        "metadata_only_descendant": True, "binding_unchanged": True,
        "finite_execution_callers": 5, "source_adversaries": 10,
        "production_calls": 0, "attempt_roots": 0,
    }


_PRECOMPUTE_IMPORTS_FIXTURE = r"""
import ast, hashlib, inspect, json, os, platform, re, socket, stat, subprocess, sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import torch
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_candidate_execution as runtime
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
from Libs.utils.evaluation import operational_twin_training_protocol as protocol
from Libs.scripts.builders.submit_operational_twin_candidate_execution import (
    require_candidate_dispatch_manifest, require_candidate_release_manifest,
    submit_arrays, worker_binding,
)
from Libs.exp.training import train_operational_twin_devcal as trainer
from Libs.exp.training import run_operational_twin_assignment_command as runner
from Libs.exp.training.run_operational_twin_candidate_attempt import _evaluation_argv
from Libs.exp.evaluation import evaluate_operational_twin_development as evaluator
root = Path.cwd().resolve()
"""

_PRECOMPUTE_READ_BOUNDARY_FIXTURE = r"""prefix = "Input/processed/controlled_neural_world/open_action_identification/"
input_rows = {
    prefix + "training_manifest.json": ("4989996c548016a63014d788d03df8d2b212d9be", 443340),
    prefix + "development_manifest.json": ("962436ca4ae3ad8bfd19b735c00a15e51f9354f9", 936070),
    prefix + "calibration_manifest.json": ("bff0f1422f2812788a1a09757000d7cca4250e7f", 444747),
    prefix + "package_receipt.json": ("f45302de166c6f2e5d2e008386e3515eb3850673", 2849),
}
refs = (
    "bb02a4849994fd22ff1de6b2932fcb6d22c06351",
    "a31757c9b2e40c05e9d2b273d38c36eaa6928d73",
    "33bbfc5ea12dd38f25ad27dd17b1253d84a1e039",
    "f08d72e5b2094d85caddab39723f616b1206b19c",
)
qualification_path = "Output/audits/controlled_world_open_data_qualification.json"
qualification_blob = "43eb74206d81ffe5563e035f33a5abdde506f6a8"
phase = "transport"
observed_inputs, observed_transport, guarded_entries, stops = [], [], [], []
receipt_path = prefix + "package_receipt.json"
guard_code = trainer.require_operational_twin_training_authorized.__code__
def in_guard():
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is guard_code:
            return True
        frame = frame.f_back
    return False
def audit(event, args):
    if event == "socket.connect":
        raise PermissionError("precompute_network_forbidden")
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).absolute()
        if path.suffix in (".pt", ".pth", ".ckpt", ".safetensors"):
            raise PermissionError("precompute_checkpoint_access_forbidden")
        if "Input" in path.parts:
            relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else ""
            if relative not in input_rows or path != root / relative or ".." in path.parts:
                raise PermissionError("precompute_extra_input_forbidden")
            flags = int(args[2])
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if phase == "transport":
                observed_transport.append({"operation": "write" if writing else "verify", "path": relative})
            elif writing or relative == receipt_path or not in_guard():
                raise PermissionError("precompute_input_use_forbidden")
            else:
                observed_inputs.append({"operation": "guard_read", "path": relative})
    if event == "subprocess.Popen":
        argv = args[1]
        if not isinstance(argv, (tuple, list)):
            raise PermissionError("precompute_shell_forbidden")
        if Path(str(argv[0])).name == "git":
            if "show" in argv:
                ref, separator, path = str(argv[-1]).partition(":")
                if separator and path.startswith("Input/"):
                    if phase != "transport" or ref not in refs or path not in input_rows:
                        raise PermissionError("precompute_input_git_use_forbidden")
                    observed_transport.append({"operation": "git_show", "ref": ref, "path": path})
            if "cat-file" in argv and "blob" in argv:
                raise PermissionError("precompute_unscoped_blob_forbidden")
        elif "-c" not in argv or argv[argv.index("-c") + 1] != handoff._FROZEN_REPLAY:
            raise PermissionError("precompute_external_process_forbidden")
"""

_PRECOMPUTE_STOP_BOUNDARY_FIXTURE = r"""def seed_line(module):
    tree = ast.parse(Path(module.__file__).read_text())
    run_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    return next(node.lineno for node in ast.walk(run_node)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "_seed_everything")
trainer_stop, evaluator_stop = seed_line(trainer), seed_line(evaluator)
class BeforeSeed(BaseException):
    pass
stop_kind = None
forbidden_computation_names = frozenset(("_seed_everything", "tensorize_manifest", "build_method_model"))
trace_names = frozenset((guard_code.co_name, trainer.run.__code__.co_name, evaluator.run.__code__.co_name)) | forbidden_computation_names
def trace(frame, event, value):
    if frame.f_code.co_name not in trace_names:
        return None
    if frame.f_code is guard_code:
        if event == "call":
            guarded_entries.append("identity")
        return trace
    if frame.f_code is trainer.run.__code__:
        if event == "line" and frame.f_lineno == trainer_stop:
            stops.append("trainer")
            if stop_kind == "interruption":
                raise InterruptedError("Non-authoritative pre-seed infrastructure boundary")
            raise BeforeSeed("non-authoritative trainer pre-seed boundary")
        return trace
    if frame.f_code is evaluator.run.__code__:
        if event == "line" and frame.f_lineno == evaluator_stop:
            stops.append("evaluator")
            raise BeforeSeed("non-authoritative evaluator pre-seed boundary")
        return trace
    if event == "call" and frame.f_code.co_name in forbidden_computation_names:
        raise AssertionError("scientific computation boundary crossed")
    return None
"""

_SCHEDULER_RECEIPT_READ_BOUNDARY_FIXTURE = r"""
# This boundary covers the scheduler-receipt consumer, not arbitrary programs.
from collections import Counter
from pathlib import PurePosixPath
import shutil, time
phase = "identity_guard"
_ce_identity_audit = audit
_ce_reads = Counter()
_ce_denials = []
_ce_expected_denial = False
_ce_active_batch = None
_ce_views = {}
_ce_sources, _ce_metadata, _ce_refs = set(), set(), set(refs) | {"HEAD", expected_source_commit}
_ce_execution_root = _ce_scheduler_root = ""
_ce_nested_calls = 0
_ce_scratch = Path(os.environ["TMPDIR"]).resolve()
_ce_cleanup_source = Path("/share/home/hwxu/miniconda3/lib/python3.13/shutil.py")
assert Path(shutil.__file__) == _ce_cleanup_source
_ce_shutil_bytes = _ce_cleanup_source.read_bytes()
assert len(_ce_shutil_bytes) == 57463 and hashlib.sha256(_ce_shutil_bytes).hexdigest() == "ce110d23884b9bdc75f7b1654dbd2d7cf5d6762046a4cf8c316b96fd937d1e0b"
_ce_cleanup_module = compile(_ce_shutil_bytes, str(_ce_cleanup_source), "exec", dont_inherit=True, optimize=sys.flags.optimize)
_ce_cleanup_codes = [value for value in _ce_cleanup_module.co_consts if isinstance(value, type(_ce_cleanup_module)) and value.co_name == "_rmtree_safe_fd"]
assert len(_ce_cleanup_codes) == 1
_ce_cleanup_open_code = _ce_cleanup_codes[0]

# Admit local hostname metadata only from the two unchanged maintained callers.
_ce_hostname_events = []
_ce_hostname_source_path = root / "Libs/utils/evaluation/operational_twin_candidate_execution.py"
_ce_hostname_source = _ce_hostname_source_path.read_bytes()
assert len(_ce_hostname_source) == 104082
_ce_hostname_source_sha256 = hashlib.sha256(_ce_hostname_source).hexdigest()
assert _ce_hostname_source_sha256 == 'e43b7d8cf49c1766c6209946e8f75c7a6af6939aaa07419a603e27b78ee8a3e7'
assert Path(runtime.__file__).resolve() == _ce_hostname_source_path
_ce_hostname_module = compile(_ce_hostname_source, str(_ce_hostname_source_path), "exec", dont_inherit=True, optimize=sys.flags.optimize)
_ce_hostname_tree = ast.parse(_ce_hostname_source)
_ce_hostname_callers = []
for _ce_name in ("build_and_publish_candidate_execution_gate", "build_live_candidate_resource_probe"):
    _ce_function = getattr(runtime, _ce_name)
    _ce_codes = [value for value in _ce_hostname_module.co_consts if isinstance(value, type(_ce_hostname_module)) and value.co_name == _ce_name]
    _ce_nodes = [node for node in _ce_hostname_tree.body if isinstance(node, ast.FunctionDef) and node.name == _ce_name]
    assert len(_ce_codes) == len(_ce_nodes) == 1
    _ce_calls = [node for node in ast.walk(_ce_nodes[0]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "socket" and node.func.attr == "gethostname"]
    assert len(_ce_calls) == 1 and not _ce_calls[0].args and not _ce_calls[0].keywords
    assert _ce_function.__code__ == _ce_codes[0] and _ce_function.__globals__ is runtime.__dict__
    _ce_hostname_callers.append((_ce_name, _ce_function, _ce_codes[0], _ce_calls[0].lineno))
_ce_hostname_callers = tuple(_ce_hostname_callers)
_ce_hostname_builtin = socket.gethostname
assert inspect.isbuiltin(_ce_hostname_builtin) and _ce_hostname_builtin.__module__ == "_socket" and _ce_hostname_builtin.__name__ == "gethostname"
_ce_case_marker = os.environ.get("PYTEST_CURRENT_TEST", "")
_ce_case_nodeid, _ce_case_separator, _ce_case_phase = _ce_case_marker.rpartition(" (")
assert _ce_case_separator and _ce_case_phase == "call)"
if _ce_case_nodeid.startswith("Libs/tests/test_operational_twin_candidate_execution.py::"):
    assert "_training_protocol_call_witness" not in globals()
else:
    assert "_training_protocol_call_witness" in globals()
    assert _ce_case_nodeid in _tp_allowed
    assert _training_protocol_call_witness["nodeid"] == _ce_case_nodeid
    assert _training_protocol_call_witness["phase"] == "call"
    assert _training_protocol_call_witness["child_pid"] == os.getpid()
    assert _training_protocol_call_witness["parent_pid"] == os.getppid()
    assert _training_protocol_call_witness["slurm_job_id"] == os.environ["SLURM_JOB_ID"]
_ce_case_job_id = os.environ["SLURM_JOB_ID"]
assert _ce_case_job_id.isascii() and _ce_case_job_id.isdigit()
_ce_case_pid = os.getpid()

def _ce_frame(function):
    function = getattr(function, "__wrapped__", function)
    code = getattr(function, "__code__", None)
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is code:
            return frame
        frame = frame.f_back
    return None

def _ce_deny(kind):
    raise PermissionError("scheduler_receipt_boundary_forbidden: " + kind)

def _ce_admit_local_hostname(event, args, caller):
    if event != "socket.gethostname" or type(args) is not tuple or args:
        _ce_deny("hostname_event_shape")
    if runtime.socket is not socket or socket.gethostname is not _ce_hostname_builtin:
        _ce_deny("hostname_builtin_changed")
    for name, function, code, line in _ce_hostname_callers:
        if (getattr(runtime, name) is function and function.__code__ is caller.f_code
                and caller.f_code == code and caller.f_globals is runtime.__dict__
                and caller.f_lineno == line and caller.f_locals.get("root") == root
                and _ce_frame(function) is caller):
            record = {"event": "candidate_local_hostname_admission", "operation": event,
                      "admission": "ADMITTED_BEFORE_RETURN", "argument_count": 0,
                      "sequence": len(_ce_hostname_events) + 1,
                      "test_nodeid": _ce_case_nodeid, "test_phase": "call",
                      "caller": name, "source_path": _ce_hostname_source_path.relative_to(root).as_posix(),
                      "source_sha256": _ce_hostname_source_sha256, "source_line": line,
                      "pid": _ce_case_pid, "job_id": _ce_case_job_id, "root": str(root),
                      "monotonic_ns": time.monotonic_ns()}
            _ce_hostname_events.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            return
    _ce_deny("hostname_caller_not_bound")

def _ce_relative(path):
    return isinstance(path, str) and bool(path) and not any(c in path for c in "\n\r\0\t:\\") and not PurePosixPath(path).is_absolute() and str(PurePosixPath(path)) == path and ".." not in PurePosixPath(path).parts

def _ce_ref(ref):
    if ref in _ce_refs:
        return True
    for function in (handoff.executing_source_identity, handoff.require_approved_successor_execution_handoff):
        frame = _ce_frame(function)
        if frame is not None and frame.f_locals.get("root") == root:
            for key in ("source_commit", "head_commit", "review_commit", "artifact_commit"):
                if frame.f_locals.get(key) == ref:
                    _ce_refs.add(ref)
                    return True
    return False

def _ce_payload(ref, path):
    if not _ce_ref(ref) or not _ce_relative(path) or path.startswith("Input/") or path.endswith((".pt", ".pth", ".ckpt", ".safetensors")):
        _ce_deny("payload_ref_or_path")
    if path not in _ce_sources | _ce_metadata:
        _ce_deny("unlisted_payload:" + path)

_ce_writer_directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_ce_writer_leaf_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_ce_writer_fdopen_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC
assert (_ce_writer_directory_flags, _ce_writer_leaf_flags, _ce_writer_fdopen_flags) == (720896, 655553, 524865)

def _ce_writer_binding():
    writer = _ce_frame(runtime.write_bytes_exclusive)
    if writer is None or writer.f_code is not runtime.write_bytes_exclusive.__code__ or writer.f_globals is not runtime.__dict__:
        _ce_deny("writer_callsite")
    path, target = writer.f_locals.get("path"), writer.f_locals.get("absolute")
    if (not isinstance(path, Path) or not isinstance(target, Path) or not path.is_absolute()
            or path != target or ".." in path.parts or not root.is_relative_to(_ce_scratch)
            or root.resolve(strict=True) != root or not target.is_relative_to(root)):
        _ce_deny("writer_target_scope")
    relative = target.relative_to(root).as_posix()
    if (not _ce_relative(relative) or "Input" in target.relative_to(root).parts
            or target.suffix in {".pt", ".pth", ".ckpt", ".safetensors"}
            or relative in _ce_sources
            or relative.startswith("Libs/") and target.suffix in {".py", ".yaml", ".yml"}
            or not (relative in _ce_metadata or _ce_execution_root and relative.startswith(_ce_execution_root + "/") and target.suffix == ".json")):
        _ce_deny("writer_target_scope:" + repr(relative))
    return writer, target

def _ce_writer_parent(fd, expected_parent, target):
    if type(fd) is not int or fd < 0:
        _ce_deny("writer_descriptor")
    try:
        identity = os.fstat(fd)
        spelling = os.readlink("/proc/self/fd/" + str(fd))
        parent = Path(spelling)
        if (not stat.S_ISDIR(identity.st_mode) or not parent.is_absolute() or spelling.endswith(" (deleted)")
                or parent != expected_parent or parent.resolve(strict=True) != parent or not target.is_relative_to(parent)):
            _ce_deny("writer_descriptor_scope")
        current = parent.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            _ce_deny("writer_descriptor_identity")
    except PermissionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _ce_deny("writer_descriptor_unavailable")
    return parent

def _ce_writer_ancestor(frame, raw, fd):
    writer, target = _ce_writer_binding()
    opened, parent = writer.f_locals.get("opened"), writer.f_locals.get("current_parent")
    if (frame.f_code is not protocol._open_or_create_candidate_ancestor.__code__ or frame.f_globals is not protocol.__dict__
            or runtime._open_or_create_candidate_ancestor is not protocol._open_or_create_candidate_ancestor
            or frame.f_locals.get("component") != raw or frame.f_locals.get("parent_fd") != fd
            or writer.f_locals.get("component") != raw or writer.f_locals.get("parent_fd") != fd
            or not opened or opened[-1] != fd or not isinstance(parent, Path)
            or raw in {"", ".", ".."} or Path(raw).name != raw):
        _ce_deny("writer_ancestor_callsite")
    parent = _ce_writer_parent(fd, parent, target)
    next_path = parent / raw
    if next_path == target or not target.is_relative_to(next_path):
        _ce_deny("writer_ancestor_component")
    try:
        identity = os.stat(raw, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISDIR(identity.st_mode):
            _ce_deny("writer_ancestor_not_directory")
    return target, next_path

def _ce_writer_open(raw, mode, flags, frame):
    writer, target = _ce_writer_binding()
    if mode is not None:
        _ce_deny("writer_open_mode")
    if flags == _ce_writer_directory_flags:
        if frame is writer:
            if raw != target.anchor or writer.f_locals.get("opened") != [] or "parent_fd" in writer.f_locals:
                _ce_deny("writer_root_callsite")
            path, operation = Path(raw), "root"
        else:
            target, path = _ce_writer_ancestor(frame, raw, frame.f_locals.get("parent_fd"))
            operation = "ancestor"
    elif flags == _ce_writer_leaf_flags:
        parent, fd, opened = writer.f_locals.get("current_parent"), writer.f_locals.get("parent_fd"), writer.f_locals.get("opened")
        if frame is not writer or raw != target.name or parent != target.parent or not opened or opened[-1] != fd:
            _ce_deny("writer_leaf_callsite")
        _ce_writer_parent(fd, parent, target)
        try:
            existing = os.stat(raw, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(existing.st_mode):
                _ce_deny("writer_leaf_symlink")
        path, operation = target, "leaf"
    else:
        _ce_deny("writer_open_flags")
    return path, {"purpose": "metadata_writer", "operation": operation, "target": target}

def _ce_writer_owned_leaf(writer, target):
    parent, fd, opened = writer.f_locals.get("current_parent"), writer.f_locals.get("parent_fd"), writer.f_locals.get("opened")
    identity = writer.f_locals.get("identity")
    if parent != target.parent or not opened or opened[-1] != fd or identity is None:
        _ce_deny("writer_leaf_ownership")
    _ce_writer_parent(fd, parent, target)
    current = os.stat(target.name, dir_fd=fd, follow_symlinks=False)
    if (not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
            or (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino)):
        _ce_deny("writer_leaf_identity")
    return identity

def _ce_claim_directory(raw, frame, *, creating=False, directory_fd=None):
    claim = _ce_frame(protocol._claim_candidate_attempt_root)
    if (claim is None or claim.f_code is not protocol._claim_candidate_attempt_root.__code__
            or claim.f_globals is not protocol.__dict__ or claim.f_locals.get("root") != root):
        _ce_deny("attempt_claim_callsite")
    target, declared = claim.f_locals.get("output_root"), claim.f_locals.get("declared_path")
    if (not isinstance(target, Path) or not isinstance(declared, Path) or declared.is_absolute()
            or ".." in declared.parts or target != root / declared
            or not root.is_relative_to(_ce_scratch) or root.resolve(strict=True) != root
            or declared.as_posix() + "/attempt_ownership.json" not in _ce_attempt_metadata):
        _ce_deny("attempt_claim_target_scope")
    opened = claim.f_locals.get("opened_fds")
    if frame is claim and raw == str(root):
        if creating or opened != [] or "parent_fd" in claim.f_locals:
            _ce_deny("attempt_claim_root_callsite")
        path, operation = root, "root"
    else:
        fd = claim.f_locals.get("parent_fd")
        if not opened or opened[-1] != fd or creating and directory_fd != fd:
            _ce_deny("attempt_claim_descriptor")
        parent = root.joinpath(*declared.parts[:len(opened) - 1])
        _ce_writer_parent(fd, parent, target)
        if frame is claim:
            if (raw != declared.name or parent != target.parent
                    or claim.f_locals.get("leaf") != raw or claim.f_locals.get("leaf_created") is not (not creating)):
                _ce_deny("attempt_claim_leaf_callsite")
            path, operation = target, "leaf"
        else:
            if (frame.f_code is not protocol._open_or_create_candidate_ancestor.__code__
                    or frame.f_globals is not protocol.__dict__ or frame.f_back is not claim
                    or frame.f_locals.get("parent_fd") != fd or frame.f_locals.get("component") != raw
                    or claim.f_locals.get("component") != raw
                    or len(opened) >= len(declared.parts) or raw != declared.parts[len(opened) - 1]):
                _ce_deny("attempt_claim_ancestor_callsite")
            path, operation = parent / raw, "ancestor"
        if not target.is_relative_to(path):
            _ce_deny("attempt_claim_ancestor_scope")
    return path, {"purpose": "attempt_claim", "operation": operation, "target": target}

def _ce_claim_open(raw, mode, flags, frame):
    if mode is not None or flags != _ce_writer_directory_flags:
        _ce_deny("attempt_claim_open_arguments")
    return _ce_claim_directory(raw, frame)

def _ce_claim_mkdir(raw, mode, directory_fd, frame):
    if mode != 0o755 or type(directory_fd) is not int:
        _ce_deny("attempt_claim_mkdir_arguments")
    return _ce_claim_directory(raw, frame, creating=True, directory_fd=directory_fd)

def _ce_open_path(raw, mode, flags):
    raw_path = os.fsdecode(raw)
    if flags == _ce_writer_directory_flags and _ce_frame(protocol._claim_candidate_attempt_root) is not None:
        return _ce_claim_open(raw_path, mode, flags, sys._getframe(2))
    if flags in {_ce_writer_directory_flags, _ce_writer_leaf_flags}:
        return _ce_writer_open(raw_path, mode, flags, sys._getframe(2))
    path = Path(raw_path)
    cleanup_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC
    if path.is_absolute() or mode is not None or flags != cleanup_flags:
        return path.resolve(), None
    # Audit arguments omit dir_fd; accept it only from this pinned direct caller.
    frame = sys._getframe(2)
    if (frame.f_code != _ce_cleanup_open_code or frame.f_code.co_filename != str(_ce_cleanup_source)
            or frame.f_lineno != 670 or frame.f_globals is not shutil.__dict__
            or frame.f_locals.get("name") != raw_path or frame.f_locals.get("func") is not os.open):
        _ce_deny("cleanup_callsite")
    fd = frame.f_locals.get("dirfd")
    if type(fd) is not int or fd < 0:
        _ce_deny("cleanup_descriptor")
    if raw_path in {"", ".", ".."} or path.name != raw_path:
        _ce_deny("cleanup_relative_name")
    try:
        identity = os.fstat(fd)
        fd_target = os.readlink("/proc/self/fd/" + str(fd))
        if not stat.S_ISDIR(identity.st_mode) or not Path(fd_target).is_absolute() or fd_target.endswith(" (deleted)"):
            _ce_deny("cleanup_descriptor")
        parent = Path(fd_target).resolve(strict=True)
        current = parent.stat()
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            _ce_deny("cleanup_descriptor_identity")
        view = next((view for view in _ce_views if parent.is_relative_to(view) and view.is_relative_to(_ce_scratch)), None)
        if view is None:
            _ce_deny("cleanup_scope")
        target = (parent / raw_path).resolve(strict=True)
        if not target.is_relative_to(view):
            _ce_deny("cleanup_scope")
        if not stat.S_ISDIR(os.stat(raw_path, dir_fd=fd, follow_symlinks=False).st_mode):
            _ce_deny("cleanup_target_not_directory")
    except PermissionError:
        raise
    except (OSError, RuntimeError, ValueError):
        _ce_deny("cleanup_descriptor_unavailable")
    return target, {"purpose": "registered_view_cleanup", "view": view}

def _ce_file(path, writing, directory_open, cleanup_context=None):
    if cleanup_context is not None and cleanup_context.get("purpose") == "attempt_claim":
        if writing or not directory_open:
            _ce_deny("attempt_claim_open_intent")
        _ce_reads[("attempt_claim_open_event", cleanup_context["operation"], str(path))] += 1
        return
    if cleanup_context is not None and cleanup_context.get("purpose") == "metadata_writer":
        operation = cleanup_context["operation"]
        if writing != (operation == "leaf"):
            _ce_deny("writer_open_intent")
        _ce_reads[("metadata_writer_open_event", operation, str(path))] += 1
        return
    if path.is_relative_to(root):
        relative = path.relative_to(root).as_posix()
        if relative in _ce_sources or relative.startswith("Libs/") and path.suffix in {".py", ".yaml", ".yml"}:
            if writing:
                _ce_deny("source_write")
            return
        if relative in _ce_metadata or _ce_execution_root and relative.startswith(_ce_execution_root + "/") and path.suffix == ".json":
            return
        if "config" in globals():
            contract = config[handoff.CONFIG_KEY]
            parent = PurePosixPath(contract["evidence_inputs"]["source_gate"]).parent
            names = [row["name"] for row in contract["source_checks"]]
            if PurePosixPath(relative).parent == parent and any(re.fullmatch(re.escape(name) + r"\.source_[0-9a-f]{40}\.job_[0-9]+\.log", path.name) for name in names):
                if writing:
                    _ce_deny("source_gate_log_write")
                return
        declared = _ce_sources | _ce_metadata
        if directory_open and (relative == "." or any(value.startswith(relative + "/") for value in declared) or _ce_execution_root and relative.startswith(_ce_execution_root)):
            return
        _ce_deny("unlisted_repository_file:" + relative)
    for view, allowed in _ce_views.items():
        if path.is_relative_to(view):
            relative = path.relative_to(view).as_posix()
            if (cleanup_context is not None and cleanup_context["purpose"] == "registered_view_cleanup"
                    and cleanup_context["view"] == view and directory_open and not writing and path.is_dir()):
                _ce_reads[("owned_view_cleanup_directory", str(view), relative)] += 1
                return
            if relative in allowed or relative in {".git/HEAD", ".git/config", ".git/objects/info/alternates"}:
                return
            if directory_open and (relative == "." or any(value.startswith(relative + "/") for value in allowed | {".git/objects/info/alternates"})):
                return
            _ce_deny("unlisted_private_view_file:" + relative)
    if "Input" in path.parts or "Output" in path.parts:
        _ce_deny("external_payload")
    if path.is_relative_to(_ce_scratch) or path == Path("/dev/null"):
        return
    if writing:
        _ce_deny("external_write")
    if path.is_relative_to(Path(sys.prefix)) or str(path).startswith(("/usr/lib/", "/lib/", "/usr/share/zoneinfo/", "/proc/", "/sys/", "/etc/ssl/")) or path == Path("/etc/localtime"):
        return
    _ce_deny("external_read")

def _ce_git(argv, cwd):
    global _ce_nested_calls
    current = Path(cwd or os.getcwd()).resolve()
    authentication = _ce_frame(handoff.authenticate_frozen_successor_package)
    if argv[0] != "git":
        if authentication is None or authentication.f_locals.get("root") != root:
            _ce_deny("non_git_process")
        view = authentication.f_locals.get("view")
        boundary = {"inputs": config[handoff.CONFIG_KEY]["input_identity_allowlist"], "refs": config[handoff.CONFIG_KEY]["input_identity_refs"], "head": config[handoff.CONFIG_KEY]["frozen_package"]["reviewed_commit"]}
        expected = [sys.executable, "-I", "-B", "-c", handoff._FROZEN_REPLAY, str(view), json.dumps(boundary)]
        if list(argv) != expected or current != view or view not in _ce_views:
            _ce_deny("unbound_frozen_replay")
        _ce_nested_calls += 1
        return
    tail = list(argv[1:])
    if authentication is not None and authentication.f_locals.get("root") == root:
        view = authentication.f_locals.get("view")
        if tail == ["init", "-q", str(view)] and current == root and view.is_relative_to(_ce_scratch):
            stage = set(authentication.f_locals["stage"])
            for path in stage:
                _ce_payload(config[handoff.CONFIG_KEY]["frozen_package"]["reviewed_commit"], path)
            _ce_views[view] = stage
            return
        if tail == ["update-ref", "HEAD", config[handoff.CONFIG_KEY]["frozen_package"]["reviewed_commit"]] and current == view and view in _ce_views:
            return
    if current != root:
        _ce_deny("git_cwd")
    if tail in (["cat-file", "--batch"], ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"]):
        if _ce_active_batch is None or tail != _ce_active_batch["tail"]:
            _ce_deny("unregistered_batch")
        return
    if len(tail) == 2 and tail[0] == "show" and ":" in tail[1]:
        ref, path = tail[1].split(":", 1)
        _ce_payload(ref, path)
        frames = ((handoff._blob, "ref", "path"), (handoff._cached_blob, "ref", "path"), (protocol._git_blob_bytes, None, "relative"), (protocol._git_commit_path_bytes, "commit", "relative"))
        if not any((frame := _ce_frame(function)) is not None and frame.f_locals.get(path_key) == path and (ref == "HEAD" if ref_key is None else frame.f_locals.get(ref_key) == ref) for function, ref_key, path_key in frames):
            _ce_deny("payload_frame")
        _ce_reads[("git_payload", ref, path)] += 1
        return
    if len(tail) == 3 and tail[:2] == ["rev-parse", "--verify"] and ":" in tail[2]:
        ref, path = tail[2].split(":", 1)
        if not _ce_ref(ref) or not _ce_relative(path):
            _ce_deny("object_selector")
        if path.startswith("Input/") and (path not in input_rows or ref not in refs):
            _ce_deny("input_object_selector")
        _ce_reads[("git_object", ref, path)] += 1
        return
    handoff_frame = _ce_frame(handoff._git)
    if tail == ["rev-parse", "HEAD"]:
        frame = _ce_frame(trainer.require_calibration_execution_authority)
        if frame is not None and frame.f_locals.get("candidate_required", {}).get("source_round_id") is not None:
            context = frame.f_locals.get("resolved_context")
            if (frame.f_code is not trainer.require_calibration_execution_authority.__code__ or frame.f_globals is not trainer.__dict__
                    or frame.f_locals.get("root") != root or frame.f_locals.get("candidate_mode") is not True or context is None
                    or frame.f_locals.get("candidate_lineage_uid") != context.protocol["candidate_lineage_uid"]
                    or frame.f_locals.get("protocol_uid") != context.protocol["protocol_uid"]
                    or frame.f_locals["candidate_required"]["source_round_id"] != frame.f_locals["actual_selection"]["selection"]["round_id"]):
                _ce_deny("calibration_authority_head_binding")
            _ce_reads[("git_metadata", "trainer_calibration_authority", "HEAD")] += 1
            return
    if tail in (["rev-parse", "HEAD"], ["status", "--porcelain", "--untracked-files=no"], ["rev-parse", "--git-path", "objects"]):
        functions = (runtime._git_commit, runtime.require_clean_tracked_source, runtime.build_candidate_pre_dispatch_audit, protocol._current_git_commit)
        if handoff_frame is None and not any(_ce_frame(function) is not None for function in functions) and not any(frame.f_code.co_name == "prepare_submission" and frame.f_code.co_filename == str(root / "Libs/scripts/builders/submit_operational_twin_candidate_execution.py") for frame in (item.frame for item in inspect.stack(context=0))):
            _ce_deny("metadata_frame")
        return
    if len(tail) == 4 and tail[:2] == ["merge-base", "--is-ancestor"]:
        frame = _ce_frame(protocol._require_executing_commit)
        if frame is not None:
            context, source = frame.f_locals.get("resolved_context"), frame.f_locals.get("source")
            if (frame.f_code is not protocol._require_executing_commit.__code__ or frame.f_globals is not protocol.__dict__
                    or frame.f_locals.get("root") != root or context is None
                    or source is not context.handoff["manifest"]["executing_source"]
                    or tail[2:] != [str(source["source_commit"]), frame.f_locals.get("commit")]
                    or not all(_ce_ref(ref) for ref in tail[2:])):
                _ce_deny("executing_commit_ancestry_binding")
            _ce_reads[("git_ancestry", tail[2], tail[3])] += 1
            return
        if _ce_frame(handoff._ancestor) is None and _ce_frame(runtime.require_candidate_calibration_source_bindings) is None:
            _ce_deny("ancestry_frame")
        return
    if tail[:4] == ["diff", "--cached", "--name-only", "--"]:
        frame = _ce_frame(handoff.executing_source_identity)
        if frame is None or tail[4:] != list(frame.f_locals["required"]) or not set(tail[4:]).issubset(_ce_sources):
            _ce_deny("source_index_diff")
        return
    if len(tail) >= 6 and tail[:2] == ["diff", "--quiet"] and tail[4] == "--":
        frame = _ce_frame(runtime.require_candidate_calibration_source_bindings)
        if frame is None or tail[2:4] != [frame.f_locals["expected_source"], frame.f_locals["revision"]] or tail[5:] != list(frame.f_locals["expected"]):
            _ce_deny("calibration_source_diff")
        for path in tail[5:]:
            if path == "Input":
                _ce_deny("calibration_diff_path:" + repr(path))
            try:
                _ce_payload(tail[2], path)
            except PermissionError:
                _ce_deny("calibration_diff_path:" + repr(path))
        return
    if tail[:5] == ["log", "-1", "--format=%H", "HEAD", "--"] and len(tail) == 6:
        frame = _ce_frame(handoff.require_approved_successor_execution_handoff)
        if frame is None or tail[5] != frame.f_locals["review_path"]:
            _ce_deny("review_log")
        return
    _ce_deny("unapproved_git_form")

def _ce_audit(event, args):
    try:
        _ce_identity_audit(event, args)
        if event.startswith("socket."):
            _ce_admit_local_hostname(event, args, sys._getframe(1))
        if event in {"os.system", "os.posix_spawn", "os.fork", "pty.spawn"}:
            _ce_deny("network_or_process")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            mode, flags = args[1], int(args[2])
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            path, cleanup_context = _ce_open_path(args[0], mode, flags)
            directory_open = mode is None and (
                flags == os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC
                or flags == _ce_writer_directory_flags and cleanup_context is not None
                and cleanup_context.get("purpose") == "attempt_claim")
            _ce_file(path, writing, directory_open, cleanup_context)
        if event == "open" and type(args[0]) is int and _ce_frame(runtime.write_bytes_exclusive) is not None:
            writer, target = _ce_writer_binding()
            frame = _ce_frame(os.fdopen)
            if (frame is None or frame.f_code is not os.fdopen.__code__ or frame.f_globals is not os.__dict__
                    or args[0] != writer.f_locals.get("descriptor") or args[1:] != ("w", _ce_writer_fdopen_flags)):
                _ce_deny("writer_fdopen_callsite")
            identity = _ce_writer_owned_leaf(writer, target)
            current = os.fstat(args[0])
            fd_path = os.readlink("/proc/self/fd/" + str(args[0]))
            if fd_path != str(target) or (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
                _ce_deny("writer_fdopen_identity")
            _ce_reads[("metadata_writer_fdopen", "", str(target))] += 1
        if event == "os.mkdir" and _ce_frame(protocol._claim_candidate_attempt_root) is not None:
            if len(args) != 3:
                _ce_deny("attempt_claim_mkdir_arguments")
            path, context = _ce_claim_mkdir(os.fsdecode(args[0]), args[1], args[2], sys._getframe(1))
            _ce_reads[("attempt_claim_mkdir_event", context["operation"], str(path))] += 1
        elif event == "os.mkdir" and (_ce_frame(runtime.write_bytes_exclusive) is not None or _ce_frame(protocol._open_or_create_candidate_ancestor) is not None):
            frame = sys._getframe(1)
            if len(args) != 3 or args[1] != 0o755:
                _ce_deny("writer_mkdir_arguments")
            target, path = _ce_writer_ancestor(frame, os.fsdecode(args[0]), args[2])
            _ce_reads[("metadata_writer_mkdir_event", "", str(path))] += 1
        if event == "os.remove" and _ce_frame(runtime.write_bytes_exclusive) is not None:
            writer, target = _ce_writer_binding()
            if sys._getframe(1) is not writer or len(args) != 2 or args[0] != target.name or args[1] != writer.f_locals.get("parent_fd"):
                _ce_deny("writer_cleanup_callsite")
            _ce_writer_owned_leaf(writer, target)
            _ce_reads[("metadata_writer_owned_leaf_cleanup", "", str(target))] += 1
        if event == "subprocess.Popen":
            if not isinstance(args[1], (tuple, list)):
                _ce_deny("shell_process")
            _ce_git(args[1], args[2])
    except PermissionError as error:
        if not _ce_expected_denial:
            _ce_denials.append(str(error))
        raise

_ce_real_run = subprocess.run
def _ce_checked_run(*args, **kwargs):
    global _ce_active_batch
    argv = args[0] if args else kwargs.get("args")
    forms = (["git", "cat-file", "--batch"], ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"])
    if list(argv) not in forms:
        return _ce_real_run(*args, **kwargs)
    payload = list(argv) == forms[0]
    frame = _ce_frame(handoff._identities if payload else handoff._source_candidate_objects)
    if frame is None or Path(kwargs.get("cwd", "")).resolve() != root or frame.f_locals.get("root") != root:
        _ce_deny("batch_frame_or_root")
    ref, paths = frame.f_locals["ref"], list(frame.f_locals["paths"])
    if payload:
        if paths != frame.f_locals["ordered"]:
            _ce_deny("batch_order")
        for path in paths:
            _ce_payload(ref, path)
    elif not _ce_ref(ref) or any(not _ce_relative(path) or not path.startswith("Libs/") or not path.endswith(".py") for path in paths):
        _ce_deny("source_probe_batch")
    expected = "\n".join(ref + ":" + path for path in paths) + "\n"
    if kwargs.get("input") != (expected.encode() if payload else expected):
        _ce_deny("batch_stdin")
    _ce_active_batch = {"tail": list(argv)[1:], "ref": ref, "paths": paths}
    for path in paths:
        _ce_reads[("batch_payload" if payload else "batch_object", ref, path)] += 1
    try:
        return _ce_real_run(*args, **kwargs)
    finally:
        _ce_active_batch = None

sys.addaudithook(_ce_audit)
subprocess.run = _ce_checked_run
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
contract = config[handoff.CONFIG_KEY]
_ce_sources = set(contract["source_bindings"]) | set(contract["frozen_view_source_roots"])
_ce_sources.update(config["operational_twin_successor_lineage_protocol"]["source_bindings"])
assert all(_ce_relative(path) and not path.startswith(("Input/", "Output/")) for path in _ce_sources)
frozen_package = contract["frozen_package"]
_ce_refs.update(frozen_package[key] for key in ("source_commit", "artifact_commit", "reviewed_commit", "ancestry_merge"))
_ce_metadata.update(frozen_package[key] for key in ("review_path", "closeout_path", "pre_change_path", "source_gate_path"))
_ce_metadata.update(frozen_package["publications"].values())
_ce_metadata.update(contract["output_paths"].values())
_ce_metadata.update(contract["evidence_inputs"].values())
_ce_metadata.add(contract["review_contract"]["path"])
execution = config["operational_twin_candidate_execution"]
_ce_metadata.update(execution["output_paths"].values())
_ce_metadata.update(row["path"] for row in execution["frozen_artifacts"].values())
_ce_metadata.update("orchestrator/rounds/" + expected_round_id + "/" + name for name in ("selection-record.json", "round-plan-record.json", "plan.md"))
_ce_execution_root = execution["output_paths"]["execution_root"]
_ce_scheduler_root = execution["output_paths"]["scheduler_receipts_root"]
assert all(_ce_relative(path) and not path.startswith("Input/") for path in _ce_metadata)
reader_blobs = {
    "Libs/utils/evaluation/operational_twin_successor_lineage_protocol.py": "8432698bb1af74ecab7337a82f24aad546b00d29",
    "Libs/utils/evaluation/operational_twin_training_protocol.py": "d1fe55048ea0bee45ec2bfbeb6c8881d6ebbcd46",
}
for ref in (frozen_package["source_commit"], frozen_package["reviewed_commit"]):
    for path, blob in reader_blobs.items():
        assert subprocess.check_output(["git", "rev-parse", "--verify", ref + ":" + path], cwd=root, text=True).strip() == blob
_ce_negative_controls = []
_ce_expected_denial = True
try:
    for forbidden in (root / receipt_path, root / prefix / "unlisted_manifest.json"):
        try:
            forbidden.read_text()
        except PermissionError:
            _ce_negative_controls.append("opaque_receipt" if forbidden == root / receipt_path else "extra_input")
        else:
            raise AssertionError("protected physical read was admitted")
    try:
        subprocess.check_output(["git", "show", "HEAD:" + prefix + "training_manifest.json"], cwd=root)
    except PermissionError:
        _ce_negative_controls.append("mutable_head_input_git")
    else:
        raise AssertionError("mutable HEAD Input read was admitted")
finally:
    _ce_expected_denial = False
assert not observed_inputs and not observed_transport
"""

_PRECOMPUTE_IDENTITY_FIXTURE = (
    _PRECOMPUTE_IMPORTS_FIXTURE
    + 'config = load_config_stack(root / "Libs/configs/config_stack.yaml")\nresolved = runtime.resolve_candidate_execution_protocol(config, root)\nfixture_mode = globals().get("FIXTURE_MODE", "initial_producer")\nassert fixture_mode in {"initial_producer", "recovery_consumer"}\n'
    + _PRECOMPUTE_READ_BOUNDARY_FIXTURE
    + "sys.addaudithook(audit)\n"
    + 'def git(*args):\n    return subprocess.check_output(["git", *args], cwd=root)\ndef verified_payload(path, data, expected_blob, expected_size=None):\n    identity = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\\0" + data).hexdigest()\n    if identity != expected_blob or (expected_size is not None and len(data) != expected_size):\n        raise ValueError("precompute_authenticated_transport_mismatch: " + path)\n    return hashlib.sha256(data).hexdigest()\nfor ref in refs:\n    for path, (blob, size) in input_rows.items():\n        fields, actual_path = git("ls-tree", ref, "--", path).decode().strip().split("\\t")\n        assert fields.split() == ["100644", "blob", blob] and actual_path == path\n        assert int(git("cat-file", "-s", ref + ":" + path)) == size\nfrozen = handoff.authenticate_frozen_successor_package(config, root)\nfrozen_identities = {row["path"]: row for row in frozen["input_identities"]}\nfor path, (blob, size) in input_rows.items():\n    data = git("show", refs[-1] + ":" + path)\n    digest = verified_payload(path, data, blob, size)\n    if path in frozen_identities:\n        assert digest == frozen_identities[path]["sha256"]\n    destination = handoff._path(root, path, absent=True)\n    destination.parent.mkdir(parents=True, exist_ok=True)\n    with destination.open("xb") as stream:\n        stream.write(data)\n    assert stat.S_ISREG(destination.stat().st_mode) and destination.stat().st_nlink == 1\n    assert verified_payload(path, destination.read_bytes(), blob, size) == digest\n    del data\nfields, actual_path = git("ls-tree", refs[-1], "--", qualification_path).decode().strip().split("\\t")\nassert fields.split() == ["100644", "blob", qualification_blob] and actual_path == qualification_path\nqualification = git("show", refs[-1] + ":" + qualification_path)\nverified_payload(qualification_path, qualification, qualification_blob)\ndestination = handoff._path(root, qualification_path, absent=True)\ndestination.parent.mkdir(parents=True, exist_ok=True)\nwith destination.open("xb") as stream:\n    stream.write(qualification)\ndel qualification\nfor bad in (b"", b"{}", b\'{"status":"PASS"}\'):\n    try:\n        verified_payload(receipt_path, bad, input_rows[receipt_path][0], input_rows[receipt_path][1])\n    except ValueError:\n        pass\n    else:\n        raise AssertionError("dummy receipt transport accepted")\n    try:\n        verified_payload(qualification_path, bad, qualification_blob)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError("fabricated qualification accepted")\nsource = (root / "Libs/exp/training/train_operational_twin_devcal.py").read_text()\nguard_node = next(node for node in ast.parse(source).body\n                  if isinstance(node, ast.FunctionDef) and node.name == "require_operational_twin_training_authorized")\nassert hashlib.sha256(ast.get_source_segment(source, guard_node).encode()).hexdigest() == "fd00db2d44dd484d481ad534b2977d036f850848e93613027a4f647198f24028"\nphase = "identity_guard"\nfor forbidden in (root / receipt_path, root / prefix / "unlisted_manifest.json"):\n    try:\n        forbidden.read_text()\n    except PermissionError:\n        pass\n    else:\n        raise AssertionError("receipt-content or additional Input read accepted")\ntry:\n    git("show", "HEAD:" + prefix + "training_manifest.json")\nexcept PermissionError:\n    pass\nelse:\n    raise AssertionError("mutable current Input blob substituted for frozen staging")\nfixture_path = root / "Libs/tests/fixtures/operational_twin_resource_receipts/slurm_array_element_resource_receipt.json"\nresource_fixture = json.loads(fixture_path.read_text())\nassert hashlib.sha256(resource_fixture["resource_stdout"].encode()).hexdigest() == resource_fixture["resource_stdout_sha256"]\nassert hashlib.sha256(resource_fixture["sinfo_stdout"].encode()).hexdigest() == resource_fixture["sinfo_stdout_sha256"]\ncalls = []\ndef scheduler_boundary(argv, current_root):\n    assert current_root == root\n    calls.append(argv)\n    if argv[0] == "sinfo":\n        return "cpu04|cpu*|mix|(null)\\ngpu03|gpu1|mix|gpu:4"\n    if argv[0] == "sbatch":\n        assert "--hold" in argv\n        return str(775000 + len(calls))\n    assert argv[:2] == ["scontrol", "release"]\n    return ""\nif fixture_mode == "initial_producer":\n    dispatch = submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)\n    for array in dispatch["initial_arrays"]:\n        for option in array["submission_argv"]:\n            if option.startswith(("--output=", "--error=")):\n                for index in array["command_indices"]:\n                    path = Path(option.split("=", 1)[1].replace("%A", array["job_id"]).replace("%a", str(index)))\n                    with path.open("x") as stream:\n                        stream.write("Non-authoritative scheduler-boundary log fixture; no model result.\\n")\nelse:\n    dispatch_path = root / resolved.execution["output_paths"]["dispatch_manifest"]\n    dispatch = require_candidate_dispatch_manifest(\n        config, root, json.loads(dispatch_path.read_text()), mode="initial",\n        resolved_context=resolved,\n    )\n    initial_release = require_candidate_release_manifest(config, root, resolved, dispatch)\n    assert initial_release["status"] == "RELEASED"\ncontroller_binding_environment = {\n    key: os.environ.get(key) for key in worker_binding(root, resolved, dispatch["source_commit"])\n}\n@contextmanager\ndef controller_environment():\n    prior = {key: os.environ.get(key) for key in controller_binding_environment}\n    try:\n        for key, value in controller_binding_environment.items():\n            if value is None:\n                os.environ.pop(key, None)\n            else:\n                os.environ[key] = value\n        # Both placement readers use the same retained synthetic hostname.\n        # Preserve the remaining live uname fields and restore both APIs on exit.\n        retained_hostname = resource_fixture["environment"]["hostname"]\n        live_uname = os.uname()\n        retained_uname = os.uname_result((live_uname.sysname, retained_hostname,\n            live_uname.release, live_uname.version, live_uname.machine))\n        with patch.object(socket, "gethostname", return_value=retained_hostname), \\\n                patch.object(os, "uname", return_value=retained_uname):\n            yield\n    finally:\n        for key, value in prior.items():\n            if value is None:\n                os.environ.pop(key, None)\n            else:\n                os.environ[key] = value\ndef prepare_runtime(index, *, runtime_dispatch=None):\n    runtime_dispatch = dispatch if runtime_dispatch is None else runtime_dispatch\n    command = resolved.command_catalog["commands"][index]\n    row = next(row for row in runtime_dispatch[runtime_dispatch["mode"] + "_arrays"] if index in row["command_indices"])\n    raw_job = str(875000 + index)\n    os.environ.update(worker_binding(root, resolved, runtime_dispatch["source_commit"]))\n    updates = dict(SLURM_JOB_ID=raw_job, SLURM_ARRAY_JOB_ID=row["job_id"],\n                   SLURM_ARRAY_TASK_ID=str(index), SLURM_JOB_NAME=row["job_name"],\n                   SLURM_JOB_PARTITION="cpu", SLURM_JOB_NODELIST="cpu04",\n                   SLURM_CPUS_PER_TASK="2", SLURM_MEM_PER_NODE="8192",\n                   SLURM_JOB_GRES="", CUDA_VISIBLE_DEVICES="")\n    os.environ.update(updates)\n    for key in ("SLURM_MEM_PER_CPU", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS"):\n        os.environ.pop(key, None)\n    raw = resource_fixture["resource_stdout"]\n    for key, value in (("JobId", raw_job), ("ArrayJobId", row["job_id"]),\n                       ("ArrayTaskId", str(index)), ("JobName", row["job_name"])):\n        raw, count = re.subn(r"(?<!\\S)" + re.escape(key) + r"=\\S+", key + "=" + value, raw)\n        assert count == 1\n    def capture(argv, current_root):\n        assert current_root == root\n        if argv[0] == "sinfo":\n            return resource_fixture["sinfo_stdout"]\n        assert argv == ["scontrol", "--oneliner", "show", "job", row["job_id"] + "_" + str(index)]\n        return raw\n    return command, capture\ndef activate(index):\n    command, capture = prepare_runtime(index)\n    gate = runtime.build_and_publish_candidate_execution_gate(\n        config, root, task_index=index, command_capture=capture)\n    assert gate["runtime_dispatch_binding"]["external_job_id"] != gate["runtime_dispatch_binding"]["slurm_job_id"]\n    return command, gate\n'
    + _PRECOMPUTE_STOP_BOUNDARY_FIXTURE
)


def _scheduler_receipt_guarded_source(source, *, source_commit, round_id):
    """Compose the existing identity/preseed predicates before the CE consumer."""
    import textwrap
    marker = 'config = load_config_stack(root / "Libs/configs/config_stack.yaml")\n'
    assert _SCHEDULER_RECEIPT_READ_BOUNDARY_FIXTURE.count(marker) == 1
    setup, consumer = _SCHEDULER_RECEIPT_READ_BOUNDARY_FIXTURE.split(marker, 1)
    prefix = (
        "expected_source_commit = " + repr(source_commit) + "\n"
        + "expected_round_id = " + repr(round_id) + "\nimport pytest\nfrom Libs.utils.evaluation import operational_twin_candidate_evidence\n"
        + _PRECOMPUTE_IMPORTS_FIXTURE
        + _PRECOMPUTE_READ_BOUNDARY_FIXTURE
        + _PRECOMPUTE_STOP_BOUNDARY_FIXTURE
        + setup
        + "_ce_prior_trace = sys.gettrace()\nsys.settrace(trace)\ntry:\n"
    )
    suffix = """\nfinally:
    sys.settrace(_ce_prior_trace)
    subprocess.run = _ce_real_run
assert sys.gettrace() is _ce_prior_trace
assert not _ce_denials, _ce_denials
assert not stops and not guarded_entries
assert _ce_nested_calls >= 1
nested_events = [event for key, value in handoff._FROZEN_REPLAY_CACHE.items() if key[0] == str(root.absolute()) for event in value["input_read_events"]]
assert {row["path"] for row in nested_events} == set(contract["input_identity_allowlist"])
assert all(row["ref"] in contract["input_identity_refs"] for row in nested_events)
assert not observed_inputs and not observed_transport
print(json.dumps({"event": "scheduler_receipt_read_boundary", "test_nodeid": _ce_case_nodeid, "test_phase": "call", "pid": _ce_case_pid, "job_id": _ce_case_job_id, "root": str(root), "local_hostname_events": _ce_hostname_events, "source_pathset": sorted(_ce_sources), "metadata_pathset": sorted(_ce_metadata), "registered_refs": sorted(_ce_refs), "read_counts": [{"operation": op, "ref": ref, "path": path, "count": count} for (op, ref, path), count in sorted(_ce_reads.items())], "nested_input_events": nested_events, "negative_controls": _ce_negative_controls, "unexpected_denials": 0, "trace_restored": True, **({"test_call_witness": _training_protocol_call_witness} if "_training_protocol_call_witness" in globals() else {})}))
result["guard"] = {"test_nodeid": _ce_case_nodeid, "test_phase": "call", "pid": _ce_case_pid, "job_id": _ce_case_job_id, "root": str(root), "local_hostname_events": _ce_hostname_events, "negative_controls": _ce_negative_controls, "nested_authentications": _ce_nested_calls, "nested_input_paths": len({row["path"] for row in nested_events}), "physical_input_reads": len(observed_inputs), "transport_events": len(observed_transport), "unexpected_denials": 0, "trace_restored": True, "scientific_calls": len(stops)}
if "_training_protocol_call_witness" in globals():
    result["guard"]["test_call_witness"] = _training_protocol_call_witness
    result["test_call_witness"] = _training_protocol_call_witness
print(json.dumps(result))
"""
    current_attempt_metadata = r"""
_ce_current_resolved = runtime.resolve_candidate_execution_protocol(config, root)
assert len(_ce_current_resolved.assignment_manifest["assignments"]) == 15
assert len(_ce_current_resolved.command_catalog["commands"]) == 30
_ce_attempt_metadata = set()
for assignment in _ce_current_resolved.assignment_manifest["assignments"]:
    for attempt in assignment["attempts"]:
        artifacts = runtime.candidate_execution_artifact_paths(_ce_current_resolved.protocol, assignment, attempt)
        assert set(artifacts) == {"execution_authority", "resource_environment_receipt"}
        _ce_attempt_metadata.update(artifacts.values())
        for name in ("attempt_ownership.json", "terminal_status.json", "execution_phase_receipt.json"):
            path = handoff._path(root, attempt["output_root"] + "/" + name)
            _ce_attempt_metadata.add(path.relative_to(root).as_posix())
assert len(_ce_attempt_metadata) == 150
assert all(_ce_relative(path) and path.startswith("Output/") and path.endswith(".json")
    and not {"checkpoints", "evaluation", "provenance"}.intersection(PurePosixPath(path).parts)
    for path in _ce_attempt_metadata)
_ce_metadata.update(_ce_attempt_metadata)
_ce_attempt_root = _ce_current_resolved.assignment_manifest["assignments"][0]["attempts"][0]["output_root"]
_ce_prior_expected_denial = _ce_expected_denial
_ce_expected_denial = True
try:
    for leaf in ("unlisted.json", "checkpoints/model.pt", "evaluation/summary.json", "provenance/runtime_provenance.json"):
        target = root / _ce_attempt_root / leaf
        assert not target.exists()
        try:
            target.read_bytes()
        except PermissionError:
            pass
        else:
            raise AssertionError("undeclared attempt payload read was admitted: " + leaf)
finally:
    _ce_expected_denial = _ce_prior_expected_denial
print(json.dumps({"event": "candidate_attempt_metadata_boundary", "declared_file_count": len(_ce_attempt_metadata),
    "assignment_count": 15, "attempt_slot_count": 30, "undeclared_payload_rejections": 4}), flush=True)
"""
    return prefix + textwrap.indent(marker + consumer + "\n" + current_attempt_metadata + "\n" + source, "    ") + suffix


_TRAINER_PRESEED_OWNERSHIP_FIXTURE = r"""
catalog_path = root / resolved.protocol["output_paths"]["assignment_commands"]
real_run = subprocess.run
interrupted_owners = []
for index in (0, 1):
    command, gate = activate(index)
    stop_kind = "interruption"
    delegated = []
    def launch(argv, *args, **kwargs):
        if list(argv) == command["argv"]:
            delegated.append(list(argv))
            old_argv = sys.argv
            sys.argv = [command["argv"][1], *command["argv"][2:]]
            old_trace = sys.gettrace()
            sys.settrace(trace)
            try:
                trainer.main()
            except InterruptedError as error:
                assert "Non-authoritative pre-seed" in str(error)
                return subprocess.CompletedProcess(argv, 173)
            finally:
                sys.settrace(old_trace)
                sys.argv = old_argv
            raise AssertionError("trainer returned without its terminating pre-seed interruption")
        return real_run(argv, *args, **kwargs)
    with patch.object(subprocess, "run", side_effect=launch):
        assert runner._run_candidate_command(catalog_path, index) == 173
    assert delegated == [command["argv"]]
    output = root / command["output_root"]
    terminal = json.loads((output / "terminal_status.json").read_text())
    assert terminal["status"] == "FAILED" and terminal["failure_class"] == "infrastructure_interruption"
    owner = runtime.require_candidate_attempt_ownership(config, root, resolved, command["attempt_uid"])
    assert owner["runtime_dispatch_binding"] == gate["runtime_dispatch_binding"]
    interrupted_owners.append(owner)
    assert not (output / "checkpoints").exists()
    assert not (output / "metrics").exists()
    assert not (output / "evaluation").exists()
assert len(interrupted_owners) == 2
assert interrupted_owners[0]["attempt_uid"] != interrupted_owners[1]["attempt_uid"]
assert stops == ["trainer", "trainer"]
assert len(guarded_entries) == 2
assert len(observed_inputs) == 6
assert {row["path"] for row in observed_inputs} == set(input_rows) - {receipt_path}
assert all(not (root / slot["output_root"]).exists()
           for assignment in resolved.assignment_manifest["assignments"][2:]
           for slot in assignment["attempts"])
print(json.dumps({
    "status": "TRAINER_PRESEED_AND_SECOND_OWNER_VERIFIED",
    "trainer_stops": 2, "identity_guard_reads": len(observed_inputs),
    "authenticated_input_ref_tuples": len(refs) * len(input_rows),
    "receipt_content_guard_reads": 0, "owned_failed_attempts": 2,
    "scientific_results": 0, "production_calls": 0,
}))
"""

_EVALUATOR_PRESEED_METADATA_FIXTURE = r"""
command, gate = activate(0)
stop_kind = "metadata_only"
old_argv = sys.argv
sys.argv = [command["argv"][1], *command["argv"][2:]]
args = trainer.parse_args()
sys.argv = old_argv
old_trace = sys.gettrace()
sys.settrace(trace)
try:
    trainer.run(args)
except BeforeSeed:
    pass
else:
    raise AssertionError("trainer escaped its pre-seed boundary")
finally:
    sys.settrace(old_trace)
output = root / command["output_root"]
assert not (output / "terminal_status.json").exists()
assignment = resolved.assignment_manifest["assignments"][0]
execution_context = args._candidate_execution_context
identity = args._candidate_runtime_identity
initialization = args._candidate_initialization_receipt
environment = execution_context["resource_receipt_payload"]["environment"]
scheduler = execution_context["resource_receipt_payload"]["slurm"]
device = {
    **environment, "resource_profile": assignment["resource_profile"], "device_type": "cpu",
    "slurm_partition": scheduler["slurm_job_partition"],
    "slurm_nodelist": scheduler["slurm_job_nodelist"],
    "slurm_job_gpus": scheduler["slurm_job_gpus"],
    "slurm_gpus_on_node": scheduler["slurm_gpus_on_node"],
    "slurm_step_gpus": scheduler["slurm_step_gpus"],
}
# This separate isolated case supplies terminal metadata only. It produces no
# model result, metrics, checkpoint, or executed catalog and never relabels the
# interrupted trainer test's retained evidence.
metadata = trainer._bind_candidate_runtime_payload(
    {"schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
     "status": "COMPLETED", "method_id": assignment["method_id"], "seed": assignment["seed"],
     "started_at": args._candidate_started_at,
     "finished_at": datetime.now(timezone.utc).isoformat(),
     "scheduler_identity": scheduler, "device_provenance": device,
     "calibration_execution_authority": None,
     "access_flags": trainer.EXPECTED_ACCESS_POLICY,
     "claim_flags": trainer.EXPECTED_CLAIM_BOUNDARY},
    identity, initialization, execution_context, command["output_root"])
trainer._write_attempt_status(output, metadata, candidate_lineage=True)
runtime.write_json_exclusive(output / "provenance/runtime_provenance.json", metadata)
assert not protocol.validate_assignment_runtime_evidence(assignment, metadata, metadata, resolved_context=resolved)
runtime.require_candidate_attempt_ownership(config, root, resolved, command["attempt_uid"])
evaluation_argv = _evaluation_argv(command)
old_argv = sys.argv
sys.argv = [evaluation_argv[2], *evaluation_argv[3:]]
args = evaluator.parse_args()
sys.argv = old_argv
old_trace = sys.gettrace()
sys.settrace(trace)
try:
    evaluator.run(args)
except BeforeSeed:
    pass
else:
    raise AssertionError("evaluator escaped its pre-seed boundary")
finally:
    sys.settrace(old_trace)
assert stops == ["trainer", "evaluator"]
assert len(guarded_entries) == 2
assert len(observed_inputs) == 6
assert {row["path"] for row in observed_inputs} == set(input_rows) - {receipt_path}
assert not (output / "checkpoints").exists()
assert not (output / "metrics").exists()
assert not (output / "evaluation").exists()
print(json.dumps({
    "status": "EVALUATOR_IDENTITY_OWNERSHIP_TERMINAL_PRESEED_VERIFIED",
    "trainer_stops": 1, "evaluator_stops": 1, "identity_guard_reads": len(observed_inputs),
    "authenticated_input_ref_tuples": len(refs) * len(input_rows),
    "receipt_content_guard_reads": 0, "metadata_only_fixture": True,
    "scientific_results": 0, "production_calls": 0,
}))
"""


_WORKER_PRESEED_RETAINED_FIXTURE = r"""
with controller_environment():

    from Libs.exp.training import run_operational_twin_candidate_attempt as worker
    from Libs.utils.evaluation import operational_twin_candidate_evidence as evidence
    command, capture = prepare_runtime(0)
    assignment = resolved.assignment_manifest["assignments"][0]
    attempt = assignment["attempts"][0]
    assert command["attempt_uid"] == attempt["attempt_uid"]
    training_argv = [
        sys.executable, "-m", "Libs.exp.training.run_operational_twin_assignment_command",
        "--protocol-scope", "candidate-lineage", "--task-index", "0",
    ]
    real_run = subprocess.run
    delegated = []
    resource_calls = []
    stop_kind = "interruption"
    def launch(argv, *args, **kwargs):
        argv = list(argv)
        if argv[0] in ("sinfo", "scontrol"):
            resource_calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout=capture(argv, root), stderr="")
        if argv == training_argv:
            delegated.append("runner")
            old_argv = sys.argv
            sys.argv = [argv[2], *argv[3:]]
            try:
                return subprocess.CompletedProcess(argv, runner.main())
            finally:
                sys.argv = old_argv
        if argv == command["argv"]:
            delegated.append("trainer")
            old_argv, old_trace = sys.argv, sys.gettrace()
            sys.argv = [argv[1], *argv[2:]]
            sys.settrace(trace)
            try:
                trainer.main()
            except InterruptedError as error:
                assert "Non-authoritative pre-seed" in str(error)
                return subprocess.CompletedProcess(argv, 173)
            finally:
                sys.settrace(old_trace)
                sys.argv = old_argv
            raise AssertionError("trainer escaped its terminating pre-seed boundary")
        return real_run(argv, *args, **kwargs)
    with patch.object(subprocess, "run", side_effect=launch):
        assert worker.run_attempt(0) == 173
    assert delegated == ["runner", "trainer"]
    assert len(resource_calls) == 2
    output = root / command["output_root"]
    terminal_path = output / "terminal_status.json"
    phase_path = output / worker.PHASE_RECEIPT_NAME
    attempt_receipt_path = output / "attempt_receipt.json"
    failure_path = output / "failures" / (assignment["method_id"] + "_failures.jsonl")
    terminal = json.loads(terminal_path.read_text())
    phase_receipt = json.loads(phase_path.read_text())
    attempt_receipt = json.loads(attempt_receipt_path.read_text())
    failure_rows = [json.loads(line) for line in failure_path.read_text().splitlines() if line]
    assert terminal["status"] == "FAILED"
    assert terminal["failure_class"] == "infrastructure_interruption"
    assert attempt_receipt == terminal
    assert failure_rows == [terminal]
    assert phase_receipt["status"] == "FAIL"
    assert phase_receipt["terminal_decision"] == "OPERATIONAL_TWIN_CANDIDATE_ATTEMPT_TRAINING_FAILED"
    assert phase_receipt["phases"] == [
        {"phase": "training_and_calibration", "argv": training_argv, "returncode": 173}
    ]
    owner = runtime.require_candidate_attempt_ownership(config, root, resolved, command["attempt_uid"])
    assert owner["runtime_dispatch_binding"] == phase_receipt["runtime_dispatch_binding"]
    assert phase_receipt["scheduler_identity"] == terminal["scheduler_identity"]
    assert phase_receipt["terminal_status"] == evidence.artifact_identity(root, terminal_path)
    assert set(phase_receipt["terminal_status"]) == evidence.ARTIFACT_IDENTITY_KEYS
    assert set(owner["runtime_dispatch_binding"]["dispatch_manifest"]) == evidence.ARTIFACT_IDENTITY_KEYS
    for name, base_keys in (("execution_authority", protocol.CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS),
                            ("resource_environment_receipt", protocol.CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS)):
        assert set(owner[name]) == (base_keys - {"git_object_id"}) | {"size_bytes", "successor_execution_binding", "runtime_dispatch_binding"}
        assert protocol.candidate_execution_binding_keys(base_keys, None) == base_keys
    authority_payload = json.loads((root / command["execution_authority"]).read_text())
    assert set(authority_payload["resource_environment_receipt_identity"]) == evidence.ARTIFACT_IDENTITY_KEYS | {"receipt_uid"}
    assert authority_payload["approved_protocol_review_identity"]["git_object_id"] is not None
    assert phase_receipt["evaluation_summary"] is None
    assert stops == ["trainer"] and len(guarded_entries) == 1 and len(observed_inputs) == 3
    accounting_fixture = json.loads((root / "Libs/tests/fixtures/operational_twin_resource_receipts/slurm_array_terminal_accounting_receipt.json").read_text())
    assert hashlib.sha256(accounting_fixture["stdout"].encode()).hexdigest() == accounting_fixture["stdout_sha256"]
    recorded_rows = evidence.parse_sacct_rows(accounting_fixture["stdout"])
    recorded = evidence._array_terminal_rows(
        recorded_rows, job_id=accounting_fixture["source_job_id"],
        command_indices=accounting_fixture["command_indices"])
    assert list(recorded.values()) == accounting_fixture["expected_terminal_rows"]
    # Only the external accounting boundary is adapted to this genuinely
    # interrupted fixture; no scientific outcome or executed catalog is supplied.
    scheduler_row = dict(recorded[0])
    scheduler_row.update(
        job_id=owner["runtime_dispatch_binding"]["external_job_id"],
        job_id_raw=owner["runtime_dispatch_binding"]["slurm_job_id"],
        job_name=terminal["scheduler_identity"]["slurm_job_name"],
        state="FAILED", exit_code="173:0")
    source_commit = owner["source_commit"]
    def materialized(scheduler):
        return evidence.validate_materialized_attempt(
            config, root, assignment=assignment, attempt=attempt, command=command,
            scheduler_evidence={"terminal": scheduler, "receipt": None},
            source_commit=source_commit, candidate_manifest=resolved.candidate_manifest,
            assignment_manifest=resolved.assignment_manifest,
            command_catalog=resolved.command_catalog, preflight=resolved.preflight,
            resolved_context=resolved)
    row, blockers = materialized(scheduler_row)
    assert not blockers, blockers
    assert row["status"] == "FAILED"
    assert row["failure_class"] == "infrastructure_interruption"
    assert row["checkpoint"] is None and row["evaluation_summary"] is None
    retained_paths = [
        resolved.execution["output_paths"]["dispatch_manifest"],
        command["resource_environment_receipt"], command["execution_authority"],
        command["output_root"] + "/attempt_ownership.json",
        attempt_receipt_path.relative_to(root).as_posix(),
        terminal_path.relative_to(root).as_posix(), phase_path.relative_to(root).as_posix(),
        failure_path.relative_to(root).as_posix(),
    ]
    retained = {path: (root / path).read_bytes() for path in retained_paths}
    for path in retained:
        assert subprocess.run(["git", "cat-file", "-e", "HEAD:" + path],
                              cwd=root, capture_output=True, check=False).returncode != 0
    git("add", "--", *retained)
    git("commit", "-m", "Retain guarded operational attempt metadata")
    assert git("rev-parse", "HEAD").decode().strip() != source_commit
    assert all((root / path).read_bytes() == data for path, data in retained.items())
    assert runtime.require_candidate_attempt_ownership(config, root, resolved, command["attempt_uid"]) == owner
    retained_row, retained_blockers = materialized(scheduler_row)
    assert not retained_blockers, retained_blockers
    assert retained_row == row
    for raw in (scheduler_row["job_id"], str(int(scheduler_row["job_id_raw"]) + 100)):
        bad_scheduler = {**scheduler_row, "job_id_raw": raw}
        bad_row, bad_blockers = materialized(bad_scheduler)
        assert "candidate_attempt_scheduler_identity_mismatch" in {item["code"] for item in bad_blockers}
        assert bad_row["status"] != "COMPLETED"
    for path in (retained_paths[0], terminal_path.relative_to(root).as_posix()):
        target = root / path
        original = retained[path]
        target.write_bytes(original + b"\n")
        try:
            runtime.require_candidate_attempt_ownership(config, root, resolved, command["attempt_uid"])
        except protocol.OperationalTwinProtocolError:
            pass
        else:
            raise AssertionError("changed retained artifact bytes were accepted: " + path)
        finally:
            target.write_bytes(original)
    assert all((root / path).read_bytes() == data for path, data in retained.items())
    assert len(observed_inputs) == 3 and stops == ["trainer"]
    assert not (output / "checkpoints").exists()
    assert not (output / "metrics").exists()
    assert not (output / "evaluation").exists()
    assert all(not (root / slot["output_root"]).exists()
               for assignment_row in resolved.assignment_manifest["assignments"]
               for slot in assignment_row["attempts"]
               if slot["attempt_uid"] != command["attempt_uid"])

restored_worker_binding = {key: os.environ.get(key) for key in controller_binding_environment}
assert restored_worker_binding == controller_binding_environment
print(json.dumps({"status": "WORKER_BINDING_ENVIRONMENT_RESTORED",
    "declared_controller_values": restored_worker_binding}, sort_keys=True), flush=True)
print(json.dumps({
    "status": "WORKER_PRESEED_PHASE_AND_RETAINED_IDENTITIES_VERIFIED",
    "trainer_stops": 1, "identity_guard_reads": 3,
    "authenticated_input_ref_tuples": len(refs) * len(input_rows),
    "receipt_content_guard_reads": 0, "owned_failed_attempts": 1,
    "worker_phase_receipts": 1, "same_byte_publication_checked": True,
    "raw_identity_rejections": 2, "changed_byte_rejections": 2,
    "scientific_results": 0, "production_calls": 0,
}))
"""


def test_precompute_identity_fixture_scripts_compile():
    precompute_tree = ast.parse(_PRECOMPUTE_IDENTITY_FIXTURE)
    imported_names = {
        alias.asname or alias.name.split(".", 1)[0]
        for node in precompute_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert {
        "require_candidate_dispatch_manifest",
        "require_candidate_release_manifest",
        "submit_arrays",
        "worker_binding",
    }.issubset(imported_names)
    for source in (
        _TRAINER_PRESEED_OWNERSHIP_FIXTURE,
        _EVALUATOR_PRESEED_METADATA_FIXTURE,
        _WORKER_PRESEED_RETAINED_FIXTURE + _SCHEDULER_ACCOUNTING_FIXTURE + _RECOVERY_READY_PRODUCER_FIXTURE,
    ):
        compile(_PRECOMPUTE_IDENTITY_FIXTURE + source, "<isolated-precompute-fixture>", "exec")
    compile(
        "FIXTURE_MODE = 'recovery_consumer'\nEXPECTED_RECOVERY_READY_STATE_JSON = '{}'\n"
        + _RECOVERY_READY_STATE_VERIFIER_FIXTURE
        + _PRECOMPUTE_IDENTITY_FIXTURE
        + _RECOVERY_FRESHNESS_OBSERVER_FIXTURE
        + _SCHEDULER_ACCOUNTING_FIXTURE
        + _RECOVERY_CONSUMER_RETENTION_FIXTURE,
        "<isolated-recovery-consumer-fixture>", "exec",
    )
    recovery_tree = ast.parse(_RECOVERY_CONSUMER_RETENTION_FIXTURE)

    def call_name(call):
        value = call.func
        parts = []
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))

    def keyword(call, name):
        return next(item.value for item in call.keywords if item.arg == name)

    recovery_calls = sorted(
        (node for node in ast.walk(recovery_tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    attempt_context_calls = [
        call for call in recovery_calls
        if call_name(call) == "runtime.resolve_candidate_attempt_context"
    ]
    assert len(attempt_context_calls) == 2
    assert all(
        len(call.args) >= 3
        and isinstance(call.args[2], ast.Constant)
        and call.args[2].value == 15
        for call in attempt_context_calls
    )
    protocol_resolution = next(
        call for call in recovery_calls
        if call_name(call) == "runtime.resolve_candidate_execution_protocol"
    )
    ownership_calls = [
        call for call in recovery_calls
        if call_name(call) == "runtime.require_candidate_attempt_ownership"
    ]
    materialized_call = next(
        call for call in recovery_calls
        if call_name(call) == "evidence.validate_materialized_attempt"
    )
    stale_authority_call = next(
        call for call in recovery_calls
        if call_name(call) == "protocol.require_candidate_execution_authority"
        and any(
            item.arg == "validate_current_runtime"
            and isinstance(item.value, ast.Constant)
            and item.value.value is False
            for item in call.keywords
        )
    )
    worker_call = next(
        call for call in recovery_calls if call_name(call) == "worker.run_attempt"
    )
    assert (
        protocol_resolution.lineno
        < ownership_calls[0].lineno
        < materialized_call.lineno
        < stale_authority_call.lineno
        < attempt_context_calls[0].lineno
        < worker_call.lineno
        < ownership_calls[-1].lineno
        < attempt_context_calls[-1].lineno
    )

    def assignment_call(name):
        for node in ast.walk(recovery_tree):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
                and isinstance(node.value, ast.Call)
            ):
                return node.value
        raise AssertionError(f"missing semantic fixture assignment: {name}")

    stale_try = next(
        node
        for node in recovery_tree.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Call)
            and call_name(child) == "protocol.require_candidate_execution_authority"
            and any(
                item.arg == "execution_authority_path"
                and isinstance(item.value, ast.Subscript)
                and isinstance(item.value.value, ast.Name)
                and item.value.value.id == "initial_command"
                for item in child.keywords
            )
            for child in ast.walk(node)
        )
    )
    assert any(
        isinstance(child, ast.Constant) and child.value == "candidate_resource_receipt_stale"
        for child in ast.walk(stale_try)
    )
    preflight_call = assignment_call("prewarmed_candidate")
    assert call_name(preflight_call) == "protocol.require_candidate_assignment_preflight"
    assert isinstance(keyword(preflight_call, "require_recovery_predecessor"), ast.Constant)
    assert keyword(preflight_call, "require_recovery_predecessor").value is True

    assert [
        call for call in recovery_calls
        if call_name(call) == "protocol.require_candidate_execution_authority"
    ] == [stale_authority_call]
    assert not any(
        call_name(call) in (
            "runtime.build_and_publish_candidate_execution_gate",
            "observe_worker_gate",
        )
        for call in recovery_calls
    )
    real_gate_binding = next(
        node for node in recovery_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "real_worker_gate_builder"
            for target in node.targets
        )
    )
    assert ast.dump(real_gate_binding.value) == ast.dump(ast.parse(
        "worker.build_and_publish_candidate_execution_gate", mode="eval"
    ).body)
    gate_spy = next(
        node for node in recovery_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "observe_worker_gate"
    )
    builder_calls = [
        node for node in ast.walk(gate_spy)
        if isinstance(node, ast.Call)
        and call_name(node) == "real_worker_gate_builder"
    ]
    assert len(builder_calls) == 1
    assert ast.dump(builder_calls[0]) == ast.dump(ast.parse(
        "real_worker_gate_builder(*args, **kwargs)", mode="eval"
    ).body)
    builder_assignment = next(
        node for node in gate_spy.body
        if isinstance(node, ast.Assign) and node.value is builder_calls[0]
    )
    assert len(builder_assignment.targets) == 1
    assert isinstance(builder_assignment.targets[0], ast.Name)
    assert builder_assignment.targets[0].id == "gate"
    gate_returns = [
        node for node in ast.walk(gate_spy) if isinstance(node, ast.Return)
    ]
    assert len(gate_returns) == 1
    assert isinstance(gate_returns[0].value, ast.Name)
    assert gate_returns[0].value.id == "gate"

    gate_with = next(
        node
        for node in ast.walk(recovery_tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and call_name(item.context_expr) == "patch.object"
            and len(item.context_expr.args) >= 2
            and isinstance(item.context_expr.args[0], ast.Name)
            and item.context_expr.args[0].id == "worker"
            and isinstance(item.context_expr.args[1], ast.Constant)
            and item.context_expr.args[1].value == "build_and_publish_candidate_execution_gate"
            and isinstance(keyword(item.context_expr, "side_effect"), ast.Name)
            and keyword(item.context_expr, "side_effect").id == "observe_worker_gate"
            for item in node.items
        )
    )
    worker_run = next(
        child
        for child in ast.walk(gate_with)
        if isinstance(child, ast.Call)
        and call_name(child) == "worker.run_attempt"
        and len(child.args) == 1
        and isinstance(child.args[0], ast.Constant)
        and child.args[0].value == 15
    )
    semantic_order = [
        stale_try.lineno,
        preflight_call.lineno,
        real_gate_binding.lineno,
        gate_spy.lineno,
        gate_with.lineno,
        worker_run.lineno,
    ]
    assert semantic_order == sorted(semantic_order)

    observer_tree = ast.parse(_RECOVERY_FRESHNESS_OBSERVER_FIXTURE)
    observer_calls = [node for node in ast.walk(observer_tree) if isinstance(node, ast.Call)]
    observer_names = {call_name(node) for node in observer_calls}
    assert {"datetime.now", "time.monotonic", "sys.setprofile", "sys.settrace"} <= observer_names
    assert not any(name.startswith("patch.") for name in observer_names)
    for call in observer_calls:
        if call_name(call) == "print":
            assert isinstance(keyword(call, "flush"), ast.Constant)
            assert keyword(call, "flush").value is True
    observer_scope = next(node for node in observer_tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == "observe_recovery_freshness")
    cleanup = next(node for node in observer_scope.body if isinstance(node, ast.Try))
    assert any(isinstance(node, ast.Raise) and node.exc is None
               for handler in cleanup.handlers for node in ast.walk(handler))
    cleanup_calls = {call_name(node) for node in ast.walk(ast.Module(body=cleanup.finalbody, type_ignores=[]))
                     if isinstance(node, ast.Call)}
    assert {"sys.setprofile", "sys.settrace"} <= cleanup_calls
    assert "worker_gate_build_calls" in _RECOVERY_CONSUMER_RETENTION_FIXTURE
    assert "worker_gate_phase_receipt_matches" in _RECOVERY_CONSUMER_RETENTION_FIXTURE
    assert "fresh_receipt_age_within_window" not in _RECOVERY_CONSUMER_RETENTION_FIXTURE
    assert "consume_prebuilt_gate" not in _RECOVERY_CONSUMER_RETENTION_FIXTURE
    assert "prebuilt_gate_revalidations" not in _RECOVERY_CONSUMER_RETENTION_FIXTURE
    assert any(isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Call)
        and call_name(item.context_expr) == "observe_recovery_freshness"
        for item in node.items) for node in recovery_tree.body)
    observer_with = next(
        node for node in recovery_tree.body
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and call_name(item.context_expr) == "observe_recovery_freshness"
            for item in node.items
        )
    )
    assert any(node is gate_with for node in ast.walk(observer_with))



@pytest.mark.parametrize("builder_fails", [False, True])
def test_recovery_worker_gate_spy_forwards_original_call_once(builder_fails):
    """Check wrapper forwarding only; no production gate or authority is created."""
    recovery_tree = ast.parse(_RECOVERY_CONSUMER_RETENTION_FIXTURE)
    gate_spy = next(
        node for node in recovery_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "observe_worker_gate"
    )
    config_marker, root_marker, result_marker = object(), object(), object()
    expected_error = RuntimeError("non-authoritative wrapper fixture")
    forwarded = []

    def original_builder(*args, **kwargs):
        forwarded.append((args, kwargs))
        if builder_fails:
            raise expected_error
        return result_marker

    namespace = {
        "root": root_marker,
        "real_worker_gate_builder": original_builder,
        "worker_gate_calls": [],
        "worker_gate_results": [],
    }
    exec(
        compile(
            ast.Module(body=[gate_spy], type_ignores=[]),
            "<worker-gate-forwarding-fixture>",
            "exec",
        ),
        namespace,
    )
    spy = namespace["observe_worker_gate"]
    if builder_fails:
        with pytest.raises(RuntimeError) as raised:
            spy(config_marker, root_marker, task_index=15)
        assert raised.value is expected_error
        assert namespace["worker_gate_results"] == []
    else:
        assert spy(config_marker, root_marker, task_index=15) is result_marker
        assert namespace["worker_gate_results"] == [result_marker]

    with pytest.raises(AssertionError):
        spy(config_marker, root_marker, task_index=15)
    assert len(forwarded) == len(namespace["worker_gate_calls"]) == 1
    args, kwargs = forwarded[0]
    assert len(args) == 2
    assert args[0] is config_marker and args[1] is root_marker
    assert kwargs == {"task_index": 15}


@pytest.mark.parametrize("profile_calls", [True, False])
def test_recovery_freshness_observer_records_real_clocks_and_restores_hooks(capsys, profile_calls):
    from contextlib import contextmanager
    from datetime import datetime, timezone
    from Libs.exp.training import run_operational_twin_assignment_command as runner
    from Libs.exp.training import train_operational_twin_devcal as trainer
    from Libs.utils.evaluation import operational_twin_candidate_execution as runtime
    from Libs.utils.evaluation import operational_twin_training_protocol as protocol
    from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff

    namespace = dict(contextmanager=contextmanager, datetime=datetime, timezone=timezone,
                     json=json, sys=sys, runtime=runtime, protocol=protocol, runner=runner, trainer=trainer, handoff=handoff)
    exec(_RECOVERY_FRESHNESS_OBSERVER_FIXTURE, namespace)
    original_profile, original_trace = sys.getprofile(), sys.gettrace()
    def previous_profile(frame, event, value):
        return None
    def previous_trace(frame, event, value):
        return None
    before_utc, before_monotonic = datetime.now(timezone.utc), time.monotonic()
    expected = RuntimeError("original observer-scope exception")
    sys.setprofile(previous_profile)
    sys.settrace(previous_trace)
    try:
        with pytest.raises(RuntimeError) as raised:
            with namespace["observe_recovery_freshness"](profile_calls=profile_calls):
                if profile_calls:
                    assert sys.getprofile() is not previous_profile
                else:
                    assert sys.getprofile() is previous_profile
                parsed = protocol._parse_utc_timestamp(before_utc.isoformat(), label="created_at")
                assert parsed == before_utc
                namespace["freshness_event"]("receipt_identity", stage="unit_observation",
                                             receipt={"receipt_uid": "non_authoritative_identity",
                                                      "created_at": before_utc.isoformat()})
                raise expected
        after_utc, after_monotonic = datetime.now(timezone.utc), time.monotonic()
        assert raised.value is expected
        assert sys.getprofile() is previous_profile and sys.gettrace() is previous_trace
    finally:
        sys.setprofile(original_profile)
        sys.settrace(original_trace)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {"call_enter", "call_exit", "receipt_identity", "exception", "hooks_restored"} <= {row["event"] for row in events}
    assert all(before_utc <= datetime.fromisoformat(row["utc"]) <= after_utc for row in events)
    assert all(before_monotonic <= row["monotonic_seconds"] <= after_monotonic for row in events)
    assert events[-1]["profile_restored"] is True and events[-1]["trace_restored"] is True
    assert next(row for row in events if row["event"] == "receipt_identity")["created_at"] == before_utc.isoformat()

    assert events[-1]["profile_calls"] is profile_calls
    assert events[-1]["observation_scope"] == (
        "selected_call_profiling" if profile_calls else "explicit_events_only"
    )
    if not profile_calls:
        summary = next(row for row in events if row["event"] == "latency_summary")
        assert summary["aggregates"] == []
        assert events[-1]["observed_costs"]["selected_event_count"] == 0
        assert events[-1]["observed_costs"]["delegated_profile_event_count"] == 0



@pytest.mark.parametrize("profile_calls", [True, False])
def test_recovery_freshness_observer_preserves_original_timestamp_error(capsys, profile_calls):
    from contextlib import contextmanager
    from datetime import datetime, timezone
    from Libs.exp.training import run_operational_twin_assignment_command as runner
    from Libs.exp.training import train_operational_twin_devcal as trainer
    from Libs.utils.evaluation import operational_twin_candidate_execution as runtime
    from Libs.utils.evaluation import operational_twin_training_protocol as protocol
    from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff

    namespace = dict(contextmanager=contextmanager, datetime=datetime, timezone=timezone,
                     json=json, sys=sys, runtime=runtime, protocol=protocol, runner=runner, trainer=trainer, handoff=handoff)
    exec(_RECOVERY_FRESHNESS_OBSERVER_FIXTURE, namespace)
    previous_profile, previous_trace = sys.getprofile(), sys.gettrace()
    with pytest.raises(protocol.OperationalTwinProtocolError):
        with namespace["observe_recovery_freshness"](profile_calls=profile_calls):
            protocol._parse_utc_timestamp("invalid", label="created_at")
    assert sys.getprofile() is previous_profile and sys.gettrace() is previous_trace
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    error = next(row for row in events if row["event"] == "exception")
    assert error["stage"] == "timestamp_parser"
    assert error["exception_type"] == "OperationalTwinProtocolError"
    assert isinstance(error["source_line"], int)


def test_recovery_freshness_observer_retains_terminal_events_after_saturation(capsys):
    from contextlib import contextmanager
    from datetime import datetime, timezone
    from Libs.exp.training import run_operational_twin_assignment_command as runner
    from Libs.exp.training import train_operational_twin_devcal as trainer
    from Libs.utils.evaluation import operational_twin_candidate_execution as runtime
    from Libs.utils.evaluation import operational_twin_training_protocol as protocol
    from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff

    namespace = dict(contextmanager=contextmanager, datetime=datetime, timezone=timezone,
                     json=json, sys=sys, runtime=runtime, protocol=protocol, runner=runner, trainer=trainer, handoff=handoff)
    exec(_RECOVERY_FRESHNESS_OBSERVER_FIXTURE, namespace)
    previous_profile, previous_trace = sys.getprofile(), sys.gettrace()
    timestamp = datetime.now(timezone.utc).isoformat()
    with pytest.raises(protocol.OperationalTwinProtocolError) as unobserved:
        protocol._parse_utc_timestamp("invalid", label="created_at")
    with pytest.raises(protocol.OperationalTwinProtocolError) as observed:
        with namespace["observe_recovery_freshness"]():
            for _ in range(300):
                assert protocol._parse_utc_timestamp(timestamp, label="created_at").isoformat() == timestamp
            protocol._parse_utc_timestamp("invalid", label="created_at")
    assert observed.value.args == unobserved.value.args
    assert sys.getprofile() is previous_profile and sys.gettrace() is previous_trace
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summaries = [row for row in events if row["event"] == "latency_summary"]
    assert len(summaries) == 1
    summary = summaries[0]
    parser_rows = [row for row in summary["aggregates"] if row["stage"] == "timestamp_parser"]
    assert sum(row["call_count"] for row in parser_rows) == 301
    assert sum(row["return_count"] for row in parser_rows) == 301
    assert summary["accounting_complete"] is True and summary["unreturned_calls"] == []
    assert abs(summary["exclusive_selected_seconds"] - summary["outermost_completed_seconds"]) < 1e-9
    assert not any(row["event"] in ("call_enter", "call_exit") and row["stage"] == "timestamp_parser" for row in events)
    assert not any(row["event"] == "freshness_event_limit_reached" for row in events)
    assert len(json.dumps(events).encode()) < 65536
    assert [row["event"] for row in events[-2:]] == ["exception", "hooks_restored"]
    assert events[-2]["stage"] == "timestamp_parser"
    assert events[-2]["exception_type"] == "OperationalTwinProtocolError"
    assert isinstance(events[-2]["source_line"], int)
    assert events[-1]["profile_restored"] is True and events[-1]["trace_restored"] is True


def test_recovery_freshness_observer_retains_terminal_fields_after_saturation(capsys):
    """Check observer field extraction only, with no production-validator verdict."""
    from contextlib import contextmanager
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from Libs.exp.training import run_operational_twin_assignment_command as runner  # noqa: F401 - Preserve real module initialization; observation uses a local stand-in.
    from Libs.exp.training import train_operational_twin_devcal as trainer
    from Libs.utils.evaluation import operational_twin_candidate_execution as runtime
    from Libs.utils.evaluation import operational_twin_training_protocol as protocol
    from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff

    expected = RuntimeError("non-authoritative observer field extraction fixture")
    captured = {}

    def observation_only_frame(now, receipt):
        created_at = protocol._parse_utc_timestamp(receipt["created_at"], label="created_at")
        sinfo_captured_at = protocol._parse_utc_timestamp(receipt["sinfo_captured_at"], label="sinfo_captured_at")
        resource_captured_at = protocol._parse_utc_timestamp(receipt["resource_captured_at"], label="resource_captured_at")
        check_time = datetime.now(timezone.utc)
        maximum_age = 1800
        current_source_commit = "non_authoritative_observer_field_fixture"
        unrelated_local = "unrelated_observer_fixture_value_must_not_be_logged"
        captured.update(check_time=check_time, maximum_age=maximum_age,
                        current_source_commit=current_source_commit,
                        timestamps={"created_at": created_at, "sinfo_captured_at": sinfo_captured_at,
                                    "resource_captured_at": resource_captured_at},
                        excluded_value=unrelated_local)
        raise expected

    # Select a local observation frame without patching production modules or
    # replacing any predicate on a production validation path.
    observed_protocol = SimpleNamespace(
        require_candidate_assignment_preflight=protocol.require_candidate_assignment_preflight,
        require_candidate_execution_authority=observation_only_frame,
        _parse_utc_timestamp=protocol._parse_utc_timestamp,
        _require_executing_commit=protocol._require_executing_commit,
    )
    def observation_only_receipt(*, captured_at, receipt):
        return receipt

    def observation_only_runner(receipt):
        return observed_protocol.require_candidate_execution_authority(None, receipt)

    observed_runtime = SimpleNamespace(
        build_and_publish_candidate_execution_gate=runtime.build_and_publish_candidate_execution_gate,
        resolve_candidate_execution_protocol=runtime.resolve_candidate_execution_protocol,
        resolve_candidate_attempt_context=runtime.resolve_candidate_attempt_context,
        build_candidate_resource_receipt=observation_only_receipt,
        require_candidate_dispatch_binding=runtime.require_candidate_dispatch_binding,
    )
    observed_runner = SimpleNamespace(_run_candidate_command=observation_only_runner)
    namespace = dict(contextmanager=contextmanager, datetime=datetime, timezone=timezone,
                     json=json, sys=sys, runtime=observed_runtime, protocol=observed_protocol,
                     runner=observed_runner, trainer=trainer, handoff=handoff)
    exec(_RECOVERY_FRESHNESS_OBSERVER_FIXTURE, namespace)
    previous_profile, previous_trace = sys.getprofile(), sys.gettrace()
    receipt = {"receipt_uid": "non_authoritative_observer_fields_only"}
    for field in ("created_at", "sinfo_captured_at", "resource_captured_at"):
        receipt[field] = datetime.now(timezone.utc).isoformat()
    before_utc, before_monotonic = datetime.now(timezone.utc), time.monotonic()
    with pytest.raises(RuntimeError) as raised:
        with namespace["observe_recovery_freshness"]():
            for _ in range(300):
                protocol._parse_utc_timestamp(receipt["created_at"], label="created_at")
            observed_runtime.build_candidate_resource_receipt(
                captured_at=datetime.now(timezone.utc), receipt=receipt)
            observed_runner._run_candidate_command(receipt)
    after_utc, after_monotonic = datetime.now(timezone.utc), time.monotonic()
    assert raised.value is expected
    assert sys.getprofile() is previous_profile and sys.gettrace() is previous_trace
    output = capsys.readouterr().out
    events = [json.loads(line) for line in output.splitlines()]
    summary = next(row for row in events if row["event"] == "latency_summary")
    parser_rows = [row for row in summary["aggregates"] if row["stage"] == "timestamp_parser"]
    assert sum(row["call_count"] for row in parser_rows) == 303
    assert sum(row["return_count"] for row in parser_rows) == 303
    assert summary["accounting_complete"] is True and summary["unreturned_calls"] == []
    assert abs(summary["exclusive_selected_seconds"] - summary["outermost_completed_seconds"]) < 1e-9
    anchors = [row for row in events if row["event"] == "anchor"]
    assert [row["anchor"] for row in anchors] == [
        "receipt_capture", "runner_entry", "runner_authority_call", "runner_authority_return"]
    assert datetime.fromisoformat(anchors[0]["captured_at"]) >= before_utc
    assert anchors[-1]["effective_check_time"] == captured["check_time"].isoformat()
    assert not any(row["event"] in ("call_enter", "call_exit") and row["stage"] == "timestamp_parser" for row in events)
    assert not any(row["event"] == "freshness_event_limit_reached" for row in events)
    assert len(output.encode()) < 65536
    assert [row["event"] for row in events[-2:]] == ["exception", "hooks_restored"]
    terminal = events[-2]
    assert terminal["stage"] == "execution_authority"
    assert terminal["exception_type"] == "RuntimeError" and isinstance(terminal["source_line"], int)
    assert all(terminal[field] == value for field, value in receipt.items())
    assert terminal["supplied_now"] is None
    assert terminal["effective_check_time"] == captured["check_time"].isoformat()
    assert terminal["ages_at_effective_check_seconds"] == {
        field: (captured["check_time"] - value).total_seconds()
        for field, value in captured["timestamps"].items()
    }
    assert terminal["maximum_age_seconds"] == captured["maximum_age"]
    assert terminal["current_source_commit"] == captured["current_source_commit"]
    assert before_utc <= captured["check_time"] <= datetime.fromisoformat(terminal["utc"]) <= after_utc
    assert before_monotonic <= terminal["monotonic_seconds"] <= after_monotonic
    assert events[-1]["profile_restored"] is True and events[-1]["trace_restored"] is True
    assert "unrelated_local" not in terminal and captured["excluded_value"] not in output
    costs = events[-1]["observed_costs"]
    assert costs["selected_event_count"] > 0
    assert all(value >= 0 for value in costs.values())
    assert costs["selected_event_errors"] == costs["delegated_profile_errors"] == 0
    assert "not counterfactual total profiling overhead" in events[-1]["cost_semantics"]


def _precompute_trace_unit_namespace(guard, trainer, evaluator, *, reference=False):
    """Bind exact trace branches to local unit functions; never import production code."""
    from types import SimpleNamespace

    tree = ast.parse(_PRECOMPUTE_STOP_BOUNDARY_FIXTURE)
    trace = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "trace")
    if reference:
        early = trace.body.pop(0)
        assert isinstance(early, ast.If) and isinstance(early.test, ast.Compare)
        assert isinstance(early.test.comparators[0], ast.Name) and early.test.comparators[0].id == "trace_names"
    declarations = [node for node in tree.body if (
        isinstance(node, ast.ClassDef) and node.name == "BeforeSeed"
    ) or (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in {
            "stop_kind", "forbidden_computation_names", "trace_names"
        } for target in node.targets)
    )]
    namespace = dict(guard_code=guard.__code__, trainer=SimpleNamespace(run=trainer),
                     evaluator=SimpleNamespace(run=evaluator), guarded_entries=[], stops=[],
                     trainer_stop=trainer.__code__.co_firstlineno + 1,
                     evaluator_stop=evaluator.__code__.co_firstlineno + 1)
    exec(compile(ast.Module(body=[*declarations, trace], type_ignores=[]), "<unit trace>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("prefilter_enabled", [False, True])
def test_recovery_freshness_observer_rejects_same_name_foreign_code(capsys, prefilter_enabled):
    from types import FunctionType

    fixture = _RECOVERY_FRESHNESS_OBSERVER_FIXTURE
    if not prefilter_enabled:
        names = "    selected_names = frozenset(code.co_name for code in selected)\n"
        early = "        if frame.f_code.co_name not in selected_names:\n            return\n"
        assert fixture.count(names) == fixture.count(early) == 1
        fixture = fixture.replace(names, "", 1).replace(early, "", 1)
    namespace = _recovery_freshness_observer_unit_namespace(fixture)

    def selected():
        return "selected"

    def foreign_body():
        return "foreign"

    foreign = FunctionType(foreign_body.__code__.replace(co_name=selected.__code__.co_name), foreign_body.__globals__)
    assert foreign.__code__ != selected.__code__
    namespace["handoff"].executing_source_identity = selected
    old_profile, old_trace = sys.getprofile(), sys.gettrace()
    sys.setprofile(None)
    try:
        with namespace["observe_recovery_freshness"]():
            assert foreign() == "foreign" and selected() == "selected"
        assert sys.getprofile() is None and sys.gettrace() is old_trace
    finally:
        sys.setprofile(old_profile)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary = next(row for row in records if row["event"] == "latency_summary")
    assert len(summary["aggregates"]) == 1
    row = summary["aggregates"][0]
    assert row["stage"] == "source_identity" and row["call_count"] == row["return_count"] == 1
    assert summary["accounting_complete"] and summary["unreturned_calls"] == []
    assert records[-1]["observed_costs"]["selected_event_count"] == 2


def test_precompute_trace_name_filter_matches_original_branches():
    from types import FunctionType, SimpleNamespace

    def guard():
        return None

    def trainer():
        return "trainer"

    def evaluator():
        return "evaluator"

    def foreign():
        return "foreign"

    trainer = FunctionType(trainer.__code__.replace(co_name="renamed_unit_trainer"), trainer.__globals__)
    evaluator = FunctionType(evaluator.__code__.replace(co_name="renamed_unit_evaluator"), evaluator.__globals__)
    candidate = _precompute_trace_unit_namespace(guard, trainer, evaluator)
    reference = _precompute_trace_unit_namespace(guard, trainer, evaluator, reference=True)
    assert candidate["trace_names"] == {
        guard.__code__.co_name, trainer.__code__.co_name, evaluator.__code__.co_name,
        *candidate["forbidden_computation_names"],
    }
    codes = [guard.__code__, trainer.__code__, evaluator.__code__, foreign.__code__,
             foreign.__code__.replace(co_name=trainer.__code__.co_name)]
    codes += [foreign.__code__.replace(co_name=name) for name in sorted(candidate["forbidden_computation_names"])]
    for code in codes:
        for event in ("call", "line", "return", "exception", "opcode"):
            for stop_kind in (None, "interruption"):
                results = []
                for namespace in (reference, candidate):
                    namespace["stop_kind"] = stop_kind
                    namespace["guarded_entries"].clear()
                    namespace["stops"].clear()
                    line = namespace["evaluator_stop"] if code is evaluator.__code__ else namespace["trainer_stop"]
                    try:
                        value = namespace["trace"](SimpleNamespace(f_code=code, f_lineno=line), event, None)
                    except BaseException as error:
                        result = ("error", type(error).__name__, str(error))
                    else:
                        assert value is None or value is namespace["trace"]
                        result = ("return", value is namespace["trace"])
                    results.append((result, list(namespace["guarded_entries"]), list(namespace["stops"])))
                assert results[0] == results[1]


def test_precompute_trace_intercepts_forbidden_names_before_body_and_restores():
    from types import FunctionType

    body_calls = []

    def guard():
        return None

    def trainer():
        return None

    def evaluator():
        return None

    def forbidden_body():
        body_calls.append("must not execute")

    old_trace = sys.gettrace()
    for reference in (False, True):
        namespace = _precompute_trace_unit_namespace(guard, trainer, evaluator, reference=reference)
        for name in sorted(namespace["forbidden_computation_names"]):
            target = FunctionType(forbidden_body.__code__.replace(co_name=name), forbidden_body.__globals__,
                                  closure=forbidden_body.__closure__)
            sys.settrace(namespace["trace"])
            try:
                with pytest.raises(AssertionError, match="scientific computation boundary crossed"):
                    target()
            finally:
                sys.settrace(old_trace)
            assert body_calls == [] and sys.gettrace() is old_trace
            assert namespace["stops"] == [] and namespace["guarded_entries"] == []


def test_precompute_trace_preserves_selected_stops_original_error_and_restoration():
    body_calls = []
    expected = RuntimeError("original local guard body error")

    def guard():
        raise expected

    def trainer():
        body_calls.append("trainer must stop")

    def evaluator():
        body_calls.append("evaluator must stop")

    old_trace = sys.gettrace()
    for reference in (False, True):
        namespace = _precompute_trace_unit_namespace(guard, trainer, evaluator, reference=reference)
        for name, target in (("trainer", trainer), ("evaluator", evaluator)):
            sys.settrace(namespace["trace"])
            try:
                with pytest.raises(namespace["BeforeSeed"]):
                    target()
            finally:
                sys.settrace(old_trace)
            assert body_calls == [] and namespace["stops"][-1] == name
            assert sys.gettrace() is old_trace
        sys.settrace(namespace["trace"])
        try:
            with pytest.raises(RuntimeError) as raised:
                guard()
        finally:
            sys.settrace(old_trace)
        assert raised.value is expected and namespace["guarded_entries"] == ["identity"]
        assert sys.gettrace() is old_trace


def _recovery_freshness_observer_unit_namespace(observer_fixture=None):
    """Execute the real observer with inert unit-only targets, never production guards."""
    from contextlib import contextmanager
    from datetime import datetime, timezone
    from types import FunctionType, SimpleNamespace

    namespace = dict(contextmanager=contextmanager, datetime=datetime, timezone=timezone,
                     json=json, sys=sys)
    observer_fixture = _RECOVERY_FRESHNESS_OBSERVER_FIXTURE if observer_fixture is None else observer_fixture
    scope = next(node for node in ast.parse(observer_fixture).body
                 if isinstance(node, ast.FunctionDef) and node.name == "observe_recovery_freshness")
    selected = next(node.value for node in scope.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "selected" for target in node.targets))

    def unused_observation_target():
        raise AssertionError("inert observer target must not execute")

    for key in selected.keys:
        target = key.value
        owner = namespace.setdefault(target.value.id, SimpleNamespace())
        code = unused_observation_target.__code__.replace(co_name=target.value.id + "_" + target.attr)
        setattr(owner, target.attr, FunctionType(code, {}))
    exec(observer_fixture, namespace)
    return namespace


def test_recovery_freshness_observer_does_not_meter_unselected_events(capsys):
    from types import SimpleNamespace

    namespace = _recovery_freshness_observer_unit_namespace()
    previous_profile, previous_trace = sys.getprofile(), sys.gettrace()
    clock_calls = []

    def counted_monotonic():
        clock_calls.append(None)
        return time.monotonic()

    def unselected():
        return len(())

    sys.setprofile(None)
    try:
        with namespace["observe_recovery_freshness"]():
            costs = namespace["freshness_observation"]["costs"]
            before = dict(costs)
            original_time = namespace["time"]
            namespace["time"] = SimpleNamespace(monotonic=counted_monotonic)
            try:
                assert unselected() == 0
                profile = sys.getprofile()
                for event in ("c_call", "c_return", "c_exception"):
                    profile(object(), event, None)  # A C event must not inspect f_code.
                assert costs == before
                assert clock_calls == []
            finally:
                namespace["time"] = original_time
        assert sys.getprofile() is None and sys.gettrace() is previous_trace
    finally:
        sys.setprofile(previous_profile)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    costs = events[-1]["observed_costs"]
    assert set(costs) == {"selected_event_count", "selected_event_seconds", "selected_event_errors",
                          "delegated_profile_event_count", "delegated_profile_seconds", "delegated_profile_errors",
                          "serialization_seconds", "emit_seconds"}
    assert costs["selected_event_count"] == costs["delegated_profile_event_count"] == 0
    assert costs["selected_event_seconds"] == costs["delegated_profile_seconds"] == 0.0
    assert events[-1]["profile_restored"] and events[-1]["trace_restored"]


@pytest.mark.parametrize("body_fails", [False, True])
def test_recovery_freshness_observer_preserves_selected_nesting_and_body_errors(capsys, body_fails):
    from datetime import datetime, timezone
    import math

    namespace = _recovery_freshness_observer_unit_namespace()
    expected = RuntimeError("original local observation body error")

    def child():
        if body_fails:
            raise expected
        return len(())

    def parent():
        return child()

    namespace["handoff"].executing_source_identity = parent
    namespace["handoff"]._check_loaded_source = child
    previous_profile, previous_trace = sys.getprofile(), sys.gettrace()
    sys.setprofile(None)
    before_utc, before_monotonic = datetime.now(timezone.utc), time.monotonic()
    try:
        if body_fails:
            with pytest.raises(RuntimeError) as raised:
                with namespace["observe_recovery_freshness"]():
                    parent()
            assert raised.value is expected
        else:
            with namespace["observe_recovery_freshness"]():
                assert parent() == 0
        assert sys.getprofile() is None and sys.gettrace() is previous_trace
    finally:
        sys.setprofile(previous_profile)
    after_utc, after_monotonic = datetime.now(timezone.utc), time.monotonic()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary = next(row for row in events if row["event"] == "latency_summary")
    rows = {row["stage"]: row for row in summary["aggregates"]}
    assert set(rows) == {"source_identity", "loaded_source"}
    assert all(row["call_count"] == row["return_count"] == 1 for row in rows.values())
    assert summary["accounting_complete"] and summary["unreturned_calls"] == []
    assert rows["source_identity"]["selected_child_seconds"] == rows["loaded_source"]["inclusive_seconds"]
    assert rows["source_identity"]["inclusive_seconds"] == summary["outermost_completed_seconds"]
    assert abs(summary["exclusive_selected_seconds"] - summary["outermost_completed_seconds"]) < 1e-9
    assert all(before_utc <= datetime.fromisoformat(row["utc"]) <= after_utc for row in events)
    assert all(before_monotonic <= row["monotonic_seconds"] <= after_monotonic for row in events)
    costs = events[-1]["observed_costs"]
    assert costs["selected_event_count"] == 4 and costs["selected_event_errors"] == 0
    assert costs["delegated_profile_event_count"] == costs["delegated_profile_errors"] == 0
    assert all(math.isfinite(value) and value >= 0 for value in costs.values())
    assert "Selected-event time excludes filtering and prior-hook delegation" in events[-1]["cost_semantics"]
    assert events[-1]["profile_restored"] and events[-1]["trace_restored"]
    if body_fails:
        terminal = next(row for row in events if row["event"] == "exception")
        assert terminal["stage"] == "loaded_source" and terminal["exception_type"] == "RuntimeError"
        assert isinstance(terminal["source_line"], int)
    else:
        assert not any(row["event"] == "exception" for row in events)


def test_recovery_freshness_observer_delegates_selected_unselected_and_c_events(capsys):
    namespace = _recovery_freshness_observer_unit_namespace()
    delegated = []

    def unselected():
        len(())
        try:
            [].index(1)
        except ValueError:
            return 0

    def selected():
        return unselected()

    def previous_profile(frame, event, value):
        if frame.f_code in (selected.__code__, unselected.__code__):
            delegated.append((frame.f_code, event, value.__name__ if event.startswith("c_") else value))

    namespace["handoff"].executing_source_identity = selected
    original_profile, original_trace = sys.getprofile(), sys.gettrace()
    sys.setprofile(previous_profile)
    try:
        assert selected() == 0
        expected = list(delegated)
        delegated.clear()
        with namespace["observe_recovery_freshness"]():
            assert selected() == 0
        assert delegated == expected
        assert {event for _, event, _ in delegated} == {"call", "return", "c_call", "c_return", "c_exception"}
        assert sys.getprofile() is previous_profile and sys.gettrace() is original_trace
    finally:
        sys.setprofile(original_profile)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    costs = events[-1]["observed_costs"]
    assert costs["selected_event_count"] == 2 and costs["selected_event_errors"] == 0
    assert costs["delegated_profile_event_count"] >= len(expected)
    assert costs["delegated_profile_seconds"] >= 0 and costs["delegated_profile_errors"] == 0
    assert events[-1]["profile_restored"] and events[-1]["trace_restored"]


def test_recovery_freshness_observer_preserves_prior_hook_error_and_restores_hooks(capsys):
    namespace = _recovery_freshness_observer_unit_namespace()
    expected = RuntimeError("original previous-profile error")

    def unselected():
        raise AssertionError("prior hook must fail before this body")

    def previous_profile(frame, event, value):
        if frame.f_code is unselected.__code__ and event == "call":
            raise expected

    original_profile, original_trace = sys.getprofile(), sys.gettrace()
    sys.setprofile(previous_profile)
    try:
        with pytest.raises(RuntimeError) as raised:
            with namespace["observe_recovery_freshness"]():
                unselected()
        assert raised.value is expected
        assert sys.getprofile() is previous_profile and sys.gettrace() is original_trace
    finally:
        sys.setprofile(original_profile)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    costs = events[-1]["observed_costs"]
    assert costs["selected_event_count"] == costs["selected_event_errors"] == 0
    assert costs["delegated_profile_event_count"] > 0 and costs["delegated_profile_errors"] == 1
    assert next(row for row in events if row["event"] == "exception")["exception_type"] == "RuntimeError"
    assert events[-1]["profile_restored"] and events[-1]["trace_restored"]


@pytest.mark.parametrize("raw_id", ["547789_0", "547790.batch", "not-an-id", "５４７７９０"])
def test_scheduler_terminal_requires_numeric_internal_raw_identity(raw_id):
    from Libs.utils.evaluation import operational_twin_candidate_evidence as evidence
    from Libs.utils.evaluation.operational_twin_training_protocol import OperationalTwinProtocolError

    fixture = json.loads((
        Path(__file__).parent / "fixtures/operational_twin_resource_receipts/slurm_array_terminal_accounting_receipt.json"
    ).read_text())
    assert hashlib.sha256(fixture["stdout"].encode()).hexdigest() == fixture["stdout_sha256"]
    rows = evidence.parse_sacct_rows(fixture["stdout"])
    selected = evidence._array_terminal_rows(
        rows, job_id=fixture["source_job_id"], command_indices=fixture["command_indices"])
    assert list(selected.values()) == fixture["expected_terminal_rows"]
    next(row for row in rows if row["job_id"] == fixture["source_job_id"] + "_0")["job_id_raw"] = raw_id
    with pytest.raises(OperationalTwinProtocolError, match="candidate_scheduler_internal_identity_invalid"):
        evidence._array_terminal_rows(
            rows, job_id=fixture["source_job_id"], command_indices=fixture["command_indices"])


def test_isolated_runtime_namespace_rejects_foreign_and_unowned_gates(isolated_execution_ready, tmp_path):
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    result = _isolated_python(root, r"""
import json, os, tempfile
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_candidate_execution as runtime
from Libs.utils.evaluation.operational_twin_training_protocol import candidate_execution_artifact_paths, OperationalTwinProtocolError
root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
resolved = runtime.resolve_candidate_execution_protocol(config, root)
assignment = resolved.assignment_manifest["assignments"][0]
selected = assignment["attempts"][0]["attempt_uid"]
command = resolved.command_catalog["commands"][0]
paths = resolved.execution["output_paths"]
rejections = 0
for kind in ("foreign_file", "foreign_directory", "foreign_scheduler", "foreign_log", "orphan_complete", "orphan_partial", "selected_partial", "selected_directory", "hardlink", "symlink", "occupied"):
    with tempfile.TemporaryDirectory(prefix="NrnSift_namespace_negative_") as temporary:
        namespace_root = Path(temporary)
        if kind.startswith("foreign"):
            suffix = {"foreign_file": "foreign.json", "foreign_directory": "foreign",
                      "foreign_scheduler": "scheduler_receipts/initial_999999.json",
                      "foreign_log": "logs/unowned.out"}[kind]
            target = namespace_root / paths["execution_root"] / suffix
            target.parent.mkdir(parents=True)
            if kind == "foreign_directory":
                target.mkdir()
            else:
                target.write_text("non-authoritative foreign evidence")
        else:
            receipt = namespace_root / command["resource_environment_receipt"]
            authority = namespace_root / command["execution_authority"]
            receipt.parent.mkdir(parents=True)
            authority.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text("non-authoritative gate fixture")
            if kind != "selected_partial" and kind != "orphan_partial":
                if kind == "selected_directory":
                    authority.mkdir()
                elif kind == "hardlink":
                    os.link(receipt, authority)
                elif kind == "symlink":
                    authority.symlink_to(receipt)
                else:
                    authority.write_text("non-authoritative gate fixture")
            if kind == "occupied":
                (namespace_root / command["output_root"]).mkdir(parents=True)
        retained = {path: path.read_bytes() for path in namespace_root.rglob("*") if path.is_file() and not path.is_symlink()}
        try:
            runtime.require_candidate_runtime_ownership(
                config, namespace_root, resolved,
                selected_attempt_uid=None if kind.startswith("orphan") else selected)
        except (OperationalTwinProtocolError, OSError):
            rejections += 1
        else:
            raise AssertionError("undeclared or unowned namespace accepted: " + kind)
        assert all(path.read_bytes() == value for path, value in retained.items())
# Namespace admission of a complete selected pair is deliberately not authority
# admission: the maintained authority validator must reject these malformed bytes.
with tempfile.TemporaryDirectory(prefix="NrnSift_selected_gate_") as temporary:
    namespace_root = Path(temporary)
    for key in ("resource_environment_receipt", "execution_authority"):
        target = namespace_root / command[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    runtime.require_candidate_runtime_ownership(config, namespace_root, resolved, selected_attempt_uid=selected)
    try:
        runtime.require_candidate_execution_authority(
            config, namespace_root, assignment=assignment, attempt=assignment["attempts"][0],
            candidate_manifest=resolved.candidate_manifest, assignment_manifest=resolved.assignment_manifest,
            command_catalog=resolved.command_catalog, preflight=resolved.preflight,
            execution_authority_path=command["execution_authority"],
            resource_environment_receipt_path=command["resource_environment_receipt"],
            cuda_available=False, cuda_device_count=0, torch_version="", resolved_context=resolved)
    except (OperationalTwinProtocolError, KeyError, ValueError, OSError):
        rejections += 1
    else:
        raise AssertionError("namespace membership was mistaken for gate authority")
print(json.dumps({"rejections": rejections, "evidence_preserved": True, "production_calls": 0}))
""", stage="finite namespace and orphan gate rejection")
    assert result == {"rejections": 12, "evidence_preserved": True, "production_calls": 0}


@pytest.mark.parametrize("collision", ["command", "internal", "external"])
def test_scheduler_merge_rejects_cross_array_or_phase_identity_reuse(collision):
    from Libs.utils.evaluation.operational_twin_candidate_evidence import merge_scheduler_evidence
    first = {0: {"terminal": {"job_id": "700_0", "job_id_raw": "800"}, "receipt": None}}
    index = 0 if collision == "command" else 15
    other = {index: {"terminal": {
        "job_id": "700_0" if collision == "external" else "701_15",
        "job_id_raw": "800" if collision == "internal" else "801",
    }, "receipt": None}}
    with pytest.raises(handoff.OperationalTwinProtocolError, match="duplicate|reused"):
        merge_scheduler_evidence(first, other)
    assert first == {0: {"terminal": {"job_id": "700_0", "job_id_raw": "800"}, "receipt": None}}


def _scheduler_join_metadata():
    from Libs.utils.evaluation.operational_twin_candidate_evidence import parse_sacct_rows
    fixtures = ROOT / "Libs/tests/fixtures/operational_twin_resource_receipts"
    resource = json.loads((fixtures / "slurm_array_element_resource_receipt.json").read_text())
    accounting = json.loads((fixtures / "slurm_array_terminal_accounting_receipt.json").read_text())
    row = next(value for value in parse_sacct_rows(accounting["stdout"]) if value["job_id"] == "547789_0")
    # Unit metadata for the maintained worker's process protocol, not a
    # materialized model result or an executed-catalog fixture.
    phase = {"status": "PASS", "phases": [
        {"phase": "training_and_calibration", "argv": ["recorded-worker-training"], "returncode": 0},
        {"phase": "open_development_calibration_evaluation", "argv": ["recorded-worker-evaluation"], "returncode": 0},
    ]}
    return row, resource["slurm"], resource["resource_requirements"], phase


@pytest.mark.parametrize("mutation", [
    "job_id", "job_id_raw", "job_name", "partition", "node_list",
    "cpu", "node", "mem", "gpu", "duplicate_tres", "exit", "phase", "completed_failure",
])
def test_scheduler_attempt_join_rejects_contradictory_provenance(mutation):
    from Libs.utils.evaluation.operational_twin_candidate_evidence import (
        classify_recovery_eligibility, validate_scheduler_attempt_evidence,
    )
    row, identity, requirements, phase = _scheduler_join_metadata()
    if mutation in {"job_id", "job_id_raw", "job_name", "partition", "node_list"}:
        row[mutation] = "foreign"
    elif mutation in {"cpu", "node", "mem", "gpu", "duplicate_tres"}:
        row["alloc_tres"] = {
            "cpu": "cpu=1,node=1", "node": "cpu=2,node=2",
            "mem": "cpu=2,node=1,mem=1G", "gpu": "cpu=2,node=1,gres/gpu=1",
            "duplicate_tres": "cpu=2,node=1,cpu=2",
        }[mutation]
    elif mutation == "exit":
        row["exit_code"] = "not-an-exit"
    elif mutation == "phase":
        phase["phases"][0]["returncode"] = 1
    else:
        phase = {"status": "FAIL", "phases": [
            {"phase": "training_and_calibration", "argv": ["recorded-worker-training"], "returncode": 173},
        ]}
    blockers = validate_scheduler_attempt_evidence(row, identity, requirements, phase)
    assert blockers
    assert classify_recovery_eligibility(
        {"status": "FAILED", "failure_class": "infrastructure_interruption", "validation_blockers": blockers},
        infrastructure_failure_class="infrastructure_interruption",
    ) == "BLOCKED_INVALID_PREDECESSOR"


def test_scheduler_attempt_join_preserves_recorded_missing_optional_tres_and_process_failures():
    from Libs.utils.evaluation.operational_twin_candidate_evidence import validate_scheduler_attempt_evidence
    row, identity, requirements, phase = _scheduler_join_metadata()
    assert "mem=" not in row["alloc_tres"] and "gres/" not in row["alloc_tres"]
    assert validate_scheduler_attempt_evidence(row, identity, requirements, phase) == []
    phase = {"status": "FAIL", "phases": [
        {"phase": "training_and_calibration", "argv": ["recorded-worker-training"], "returncode": 173},
    ]}
    for state, exit_code in (("FAILED", "173:0"), ("CANCELLED", "0:15")):
        assert validate_scheduler_attempt_evidence(
            {**row, "state": state, "exit_code": exit_code}, identity, requirements, phase,
        ) == []


@pytest.mark.parametrize("initial_index", [0, 3, 6, 9, 12])
def test_isolated_all_method_authority_routes_use_recorded_resources(isolated_execution_ready, tmp_path, initial_index):
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    result = _isolated_python(root, r"""
import json, os, re, sys
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_candidate_execution as runtime
from Libs.utils.evaluation import operational_twin_training_protocol as protocol
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
from Libs.scripts.builders.submit_operational_twin_candidate_execution import submit_arrays
root = Path.cwd()
def audit(event, args):
    if event == "socket.connect":
        raise PermissionError("recorded_authority_network_forbidden")
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).absolute()
        if "Input" in path.parts or path.suffix in (".pt", ".pth", ".ckpt", ".safetensors"):
            raise PermissionError("recorded_authority_payload_forbidden")
    if event == "subprocess.Popen":
        argv = args[1]
        if Path(str(argv[0])).name == "git":
            return
        if "-c" not in argv or argv[argv.index("-c") + 1] != handoff._FROZEN_REPLAY:
            raise PermissionError("recorded_authority_external_process_forbidden")
sys.addaudithook(audit)
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
resolved = runtime.resolve_candidate_execution_protocol(config, root)
calls = []
def scheduler_boundary(argv, current_root):
    assert current_root == root
    calls.append(argv)
    if argv[0] == "sinfo":
        return "cpu04|cpu*|mix|(null)\ngpu03|gpu1|mix|gpu:4"
    if argv[0] == "sbatch":
        assert "--hold" in argv
        return str(665000 + len(calls))
    assert argv[:2] == ["scontrol", "release"]
    return ""
dispatch = submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)
index = METHOD_INDEX
command = resolved.command_catalog["commands"][index]
capture_calls = []
def no_capture(*args):
    capture_calls.append(args)
    raise AssertionError("unowned recovery reached resource capture")
try:
    runtime.build_and_publish_candidate_execution_gate(
        config, root, task_index=index + 15, command_capture=no_capture)
except protocol.OperationalTwinProtocolError as error:
    assert "recovery" in str(error).lower() or "predecessor" in str(error).lower(), error
else:
    raise AssertionError("missing predecessor opened a method's recovery authority route")
assert capture_calls == []
array = next(row for row in dispatch["initial_arrays"] if index in row["command_indices"])
fixture_name = "cpu04_job_545144.json" if index == 0 else "gpu03_job_545150.json"
fixture = json.loads((root / "Libs/tests/fixtures/operational_twin_resource_receipts" / fixture_name).read_text())
slurm = dict(fixture["slurm"])
slurm.update(slurm_job_id=str(885000 + index), slurm_array_job_id=array["job_id"],
             slurm_array_task_id=str(index), slurm_job_name=array["job_name"])
slurm = protocol.candidate_runtime_slurm_identity(values=slurm)
raw = fixture["resource_stdout"]
for key, value in (("JobId", slurm["slurm_job_id"]), ("JobName", array["job_name"])):
    raw, count = re.subn(r"(?<!\S)" + re.escape(key) + r"=\S+", key + "=" + value, raw)
    assert count == 1
raw = raw.rstrip() + " ArrayJobId=" + array["job_id"] + " ArrayTaskId=" + str(index) + "\n"
def capture(argv, current_root):
    assert current_root == root
    capture_calls.append(argv)
    if argv[0] == "sinfo":
        assert argv == fixture["sinfo_command"]
        return fixture["sinfo_stdout"]
    assert argv == ["scontrol", "--oneliner", "show", "job", slurm["resource_query_job_id"]]
    return raw
gate = runtime.build_and_publish_candidate_execution_gate(
    config, root, task_index=index, command_capture=capture,
    slurm=slurm, environment=fixture["environment"])
assert len(capture_calls) == 2
authority = json.loads((root / command["execution_authority"]).read_text())
assignment = next(row for row in resolved.assignment_manifest["assignments"] if row["assignment_uid"] == command["assignment_uid"])
assert authority["method_id"] == command["method_id"]
assert authority["seed"] == command["seed"] == 360101
assert authority["checkpoint_creation_authorized"] == assignment["checkpoint_creation_allowed"]
assert authority["calibration_fitting_authorized"] == assignment["calibration_eligible"]
assert authority["resource_requirements"] == fixture["resource_requirements"]
assert gate["successor_execution_binding"] == dict(resolved.binding)
assert all(not (root / slot["output_root"]).exists()
           for item in resolved.assignment_manifest["assignments"] for slot in item["attempts"])
print(json.dumps({"method_id": command["method_id"], "initial_index": index,
                  "recovery_index": index + 15, "recovery_blocked": True,
                  "recorded_resource_only": True, "attempt_roots": 0, "production_calls": 0}))
""".replace("METHOD_INDEX", str(initial_index)), stage=f"recorded method authority route {initial_index}")
    assert result == {
        "method_id": ["persistence_no_action", "linear_or_delta", "same_backbone_no_action",
                      "deterministic_action_conditioned", "full_uncertainty_aware_action_conditioned"][initial_index // 3],
        "initial_index": initial_index, "recovery_index": initial_index + 15,
        "recovery_blocked": True, "recorded_resource_only": True,
        "attempt_roots": 0, "production_calls": 0,
    }


def test_scheduler_merge_retains_distinct_initial_and_recovery_rows():
    from Libs.utils.evaluation.operational_twin_candidate_evidence import merge_scheduler_evidence
    initial = {0: {"terminal": {"job_id": "700_0", "job_id_raw": "800"}, "receipt": None}}
    recovery = {15: {"terminal": {"job_id": "701_15", "job_id_raw": "801"}, "receipt": None}}
    assert merge_scheduler_evidence(initial, recovery) == {**initial, **recovery}


_SCHEDULER_ACCOUNTING_FIXTURE = r"""
from Libs.scripts.builders import build_operational_twin_candidate_execution_catalog as catalog_builder
from Libs.scripts.builders.submit_operational_twin_candidate_execution import require_candidate_dispatch_manifest
paths = resolved.execution["output_paths"]
scheduler_paths = [
    root / paths["scheduler_receipts_root"] / ("initial_" + array["job_id"] + ".json")
    for array in dispatch["initial_arrays"]
]
accounting_calls = []
def accounting_boundary(argv, current_root, *, duplicate=False, manifest=None):
    assert current_root == root
    accounting_calls.append(argv)
    active = dispatch if manifest is None else manifest
    array = next(row for row in active[active["mode"] + "_arrays"] if row["job_id"] == argv[2])
    assert argv == ["sacct", "-j", array["job_id"], "--format=" + evidence.SCHEDULER_SACCT_FORMAT, "-n", "-P"]
    group = next(row for row in resolved.execution["array_groups"] if row["group_id"] == array["group_id"])
    lines = []
    for index in array["command_indices"]:
        # Accounting fixtures are not model results: only genuine interrupted
        # worker roots are materialized; all other model evidence stays absent.
        internal = 875000 + (0 if duplicate and index == 3 else index)
        interrupted = index in (0, 15)
        allocation = "cpu=2,mem=8G,node=1" if group["gres"] is None else "cpu=6,mem=32G,node=1,gres/gpu=1"
        lines.append("|".join([
            array["job_id"] + "_" + str(index), str(internal), array["job_name"],
            group["partition"], "FAILED" if interrupted else "CANCELLED",
            "173:0" if interrupted else "0:15", group["nodelist"], allocation, "00:00:01",
        ]))
    return "\n".join(lines) + "\n"
"""


_RECOVERY_READY_PRODUCER_FIXTURE = r"""
try:
    evidence.collect_scheduler_receipts(
        root, resolved.execution, dispatch["initial_arrays"], phase="initial",
        source_commit=dispatch["source_commit"], resolved_context=resolved,
        command_capture=lambda argv, current: accounting_boundary(argv, current, duplicate=True))
except protocol.OperationalTwinProtocolError as error:
    assert "internal_identity_reused" in str(error), error
else:
    raise AssertionError("cross-array internal JobIDRaw reuse was accepted")
assert not any(path.exists() for path in scheduler_paths)
scheduler = evidence.collect_scheduler_receipts(
    root, resolved.execution, dispatch["initial_arrays"], phase="initial",
    source_commit=dispatch["source_commit"], resolved_context=resolved,
    command_capture=accounting_boundary)
assert set(scheduler) == set(range(15))
assert scheduler == evidence.load_scheduler_receipts(
    root, resolved.execution, dispatch["initial_arrays"], phase="initial",
    source_commit=dispatch["source_commit"], resolved_context=resolved)
assert scheduler[0]["terminal"]["job_id_raw"] == owner["runtime_dispatch_binding"]["slurm_job_id"]
recovery = evidence.build_recovery_eligibility(
    config, root, dispatch_manifest=dispatch, scheduler_by_index=scheduler, resolved_context=resolved)
assert recovery["assignment_count"] == 15
assert [row["initial_command_index"] for row in recovery["assignments"] if row["recovery_eligibility"] == "ELIGIBLE"] == [0]
assert all(row["recovery_eligibility"] == "BLOCKED_INVALID_PREDECESSOR" for row in recovery["assignments"][1:])
runtime.write_json_exclusive(root / paths["recovery_eligibility"], recovery)
assert evidence.require_recovery_eligibility(
    config, root, root / paths["recovery_eligibility"], resolved_context=resolved) == recovery
recovery_bytes = (root / paths["recovery_eligibility"]).read_bytes()
forged_recovery = json.loads(recovery_bytes)
forged_recovery["assignments"][1]["recovery_eligibility"] = "ELIGIBLE"
(root / paths["recovery_eligibility"]).write_text(json.dumps(forged_recovery))
try:
    evidence.require_recovery_eligibility(
        config, root, root / paths["recovery_eligibility"], resolved_context=resolved)
except protocol.OperationalTwinProtocolError as error:
    assert "recovery_evidence_drift" in str(error), error
else:
    raise AssertionError("missing predecessor evidence authorized another recovery")
finally:
    (root / paths["recovery_eligibility"]).write_bytes(recovery_bytes)
initial_release = require_candidate_release_manifest(config, root, resolved, dispatch)
assert initial_release["status"] == "RELEASED"
recovery_command = resolved.command_catalog["commands"][15]
assert all(not (root / recovery_command[key]).exists()
           for key in ("resource_environment_receipt", "execution_authority", "output_root"))
runtime.require_candidate_runtime_ownership(config, root, resolved)

# The next process must start from a genuinely clean, committed recovery-ready
# tree. Remove only the exact immutable transport inputs and the explicitly
# generated non-authoritative scheduler log placeholders before committing the
# authenticated metadata needed to classify and dispatch the recovery slot.
for relative in (*input_rows, qualification_path):
    target = root / relative
    assert target.is_file()
    target.unlink()
placeholder_paths = []
for array in dispatch["initial_arrays"]:
    for option in array["submission_argv"]:
        if option.startswith(("--output=", "--error=")):
            for index in array["command_indices"]:
                target = Path(option.split("=", 1)[1].replace("%A", array["job_id"]).replace("%a", str(index)))
                assert target.is_relative_to(root)
                assert target.read_text() == "Non-authoritative scheduler-boundary log fixture; no model result.\n"
                target.unlink()
                placeholder_paths.append(target.relative_to(root).as_posix())
assert len(placeholder_paths) == 30 and len(set(placeholder_paths)) == 30

recovery_ready_paths = {
    paths["calibration_execution_authority"], paths["pre_dispatch_audit"],
    paths["dispatch_claim"], paths["dispatch_manifest"],
    paths["dispatch_release_manifest"], paths["recovery_eligibility"],
    *retained_paths,
    *(path.relative_to(root).as_posix() for path in scheduler_paths),
    *(row["submission_receipt"]["path"] for row in dispatch["initial_arrays"]),
    *(row["intent"]["path"] for row in initial_release["attempts"]),
    *(row["receipt"]["path"] for row in initial_release["attempts"]),
}
assert all((root / relative).is_file() for relative in recovery_ready_paths)
git("add", "--", *sorted(recovery_ready_paths))
git("commit", "-m", "Retain authenticated recovery-ready execution metadata")
raw_dirty = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
dirty_entries = [
    value.decode("utf-8", errors="surrogateescape")
    for value in raw_dirty.split(b"\0") if value
]
diagnosed_dirty = []
for entry in dirty_entries:
    status = entry[:2]
    relative = entry[3:] if len(entry) >= 4 else ""
    if relative in {*input_rows, qualification_path}:
        classification = "ephemeral_transport_not_removed"
    elif relative in placeholder_paths:
        classification = "ephemeral_placeholder_log_not_removed"
    elif relative in recovery_ready_paths:
        classification = "required_authenticated_metadata_not_committed"
    elif relative.startswith(command["output_root"] + "/"):
        classification = "unclassified_attempt_output"
    else:
        classification = "unexpected_fixture_output"
    diagnosed_dirty.append({
        "entry_repr": repr(entry), "status": status,
        "path": relative, "classification": classification,
    })
print(json.dumps({
    "status": "RECOVERY_READY_DIRTY_ENTRIES",
    "dirty_count": len(diagnosed_dirty), "entries": diagnosed_dirty,
}), flush=True)
assert not dirty_entries, diagnosed_dirty
recovery_ready_commit = git("rev-parse", "HEAD").decode().strip()
assert recovery_ready_commit not in (dispatch["source_commit"], source_commit)

# Committed identities alone do not establish recovery-audit consumption.
committed_resolved = runtime.resolve_candidate_execution_protocol(config, root)
accepted_recovery = evidence.require_recovery_eligibility(
    config, root, root / paths["recovery_eligibility"], resolved_context=committed_resolved)
assert accepted_recovery == recovery


def committed_identity(relative):
    data = (root / relative).read_bytes()
    object_id = git("rev-parse", "--verify", "HEAD:" + relative).decode().strip()
    assert object_id == hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    return {"path": relative, "git_object_id": object_id,
            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}

path_identities = [committed_identity(relative) for relative in sorted(recovery_ready_paths)]
state_payload = {"head": recovery_ready_commit, "path_identities": path_identities}
state_hash = protocol.canonical_hash(state_payload)
print(json.dumps({
    "status": "RECOVERY_READY_COMMITTED_STATE_VERIFIED",
    **state_payload, "state_hash": state_hash,
    "initial_dispatch_source_commit": dispatch["source_commit"],
    "recovery_command_index": 15, "eligible_recovery_indices": [15],
    "committed_path_count": len(path_identities), "removed_transport_paths": 5,
    "removed_placeholder_logs": len(placeholder_paths),
    "trainer_stops": 1, "identity_guard_reads": 3,
    "changed_byte_rejections": 2, "raw_identity_rejections": 2,
    "scientific_results": 0, "production_calls": 0,
}))
"""


_RECOVERY_READY_STATE_VERIFIER_FIXTURE = r"""
import hashlib, json, subprocess
from pathlib import Path
from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_candidate_execution as state_runtime
from Libs.utils.evaluation import operational_twin_candidate_evidence as state_evidence
from Libs.utils.evaluation import operational_twin_training_protocol as state_protocol
from Libs.scripts.builders.submit_operational_twin_candidate_execution import (
    require_candidate_dispatch_manifest as require_state_dispatch,
    require_candidate_release_manifest as require_state_release,
)
state_root = Path.cwd().resolve()
expected_state = json.loads(EXPECTED_RECOVERY_READY_STATE_JSON)
assert expected_state["status"] == "RECOVERY_READY_COMMITTED_STATE_VERIFIED"
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=state_root, text=True).strip()
assert head == expected_state["head"]
assert subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=all"], cwd=state_root, text=True,
).strip() == ""

def state_identity(relative):
    data = (state_root / relative).read_bytes()
    object_id = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD:" + relative], cwd=state_root, text=True,
    ).strip()
    assert object_id == hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    return {"path": relative, "git_object_id": object_id,
            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}

observed_state = {
    "head": head,
    "path_identities": [state_identity(row["path"]) for row in expected_state["path_identities"]],
}
assert observed_state == {key: expected_state[key] for key in ("head", "path_identities")}
assert state_protocol.canonical_hash(observed_state) == expected_state["state_hash"]
state_config = load_config_stack(state_root / "Libs/configs/config_stack.yaml")
state_resolved = state_runtime.resolve_candidate_execution_protocol(state_config, state_root)
state_paths = state_resolved.execution["output_paths"]
state_dispatch = require_state_dispatch(
    state_config, state_root,
    json.loads((state_root / state_paths["dispatch_manifest"]).read_text()),
    mode="initial", resolved_context=state_resolved,
)
assert state_dispatch["source_commit"] == expected_state["initial_dispatch_source_commit"]
assert require_state_release(state_config, state_root, state_resolved, state_dispatch)["status"] == "RELEASED"
state_recovery = state_evidence.require_recovery_eligibility(
    state_config, state_root, state_root / state_paths["recovery_eligibility"],
    resolved_context=state_resolved,
)
assert [row["recovery_command_index"] for row in state_recovery["assignments"]
        if row["recovery_eligibility"] == "ELIGIBLE"] == [15]
state_command = state_resolved.command_catalog["commands"][15]
assert all(not (state_root / state_command[key]).exists()
           for key in ("resource_environment_receipt", "execution_authority", "output_root"))
state_initial_command = state_resolved.command_catalog["commands"][0]
state_runtime.require_candidate_attempt_ownership(
    state_config, state_root, state_resolved, state_initial_command["attempt_uid"])
state_prefix = "Input/processed/controlled_neural_world/open_action_identification/"
assert all(not (state_root / (state_prefix + name)).exists() for name in (
    "training_manifest.json", "development_manifest.json", "calibration_manifest.json", "package_receipt.json",
))
assert not (state_root / "Output/audits/controlled_world_open_data_qualification.json").exists()
print(json.dumps({
    "status": "RECOVERY_READY_CROSS_PROCESS_STATE_VERIFIED",
    "head": head, "state_hash": expected_state["state_hash"],
    "committed_path_count": len(observed_state["path_identities"]),
    "clean_before_transport": True, "production_calls": 0,
}))
"""



_RECOVERY_FRESHNESS_OBSERVER_FIXTURE = r"""
import time

freshness_event_count = 0
freshness_observation = None

def freshness_event(event, *, stage, receipt=None, required=False, **scalars):
    global freshness_event_count
    state = freshness_observation
    if state is None:
        raise RuntimeError("freshness_observer_scope_required")
    freshness_event_count += 1
    record = {
        "event": event, "stage": stage,
        "utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        **scalars,
    }
    if isinstance(receipt, dict):
        for field in ("receipt_uid", "created_at", "sinfo_captured_at", "resource_captured_at"):
            value = receipt.get(field)
            if isinstance(value, str):
                record[field] = value
    kind = "terminal" if event in ("exception", "hooks_restored") else "required" if required else "ordinary"
    started = time.monotonic()
    encoded = json.dumps(record, sort_keys=True)
    state["costs"]["serialization_seconds"] += time.monotonic() - started
    byte_count = len(encoded.encode("utf-8")) + 1
    if state["record_counts"][kind] >= {"ordinary": 64, "required": 96, "terminal": 2}[kind]:
        raise RuntimeError("freshness_observer_record_bound_exceeded")
    if state["output_bytes"][kind] + byte_count > {"ordinary": 65536, "required": 196608, "terminal": 32768}[kind]:
        raise RuntimeError("freshness_observer_output_bound_exceeded")
    started = time.monotonic()
    print(encoded, flush=True)
    state["costs"]["emit_seconds"] += time.monotonic() - started
    state["record_counts"][kind] += 1
    state["output_bytes"][kind] += byte_count

def freshness_frame_observation(frame, *, event):
    fields = {"caller": frame.f_back.f_code.co_name if frame.f_back is not None else None}
    if frame.f_back is not None:
        fields["caller_line"] = frame.f_back.f_lineno
    for name in ("now", "check_time", "captured_at"):
        timestamp = frame.f_locals.get(name)
        if isinstance(timestamp, datetime):
            fields[{"now": "supplied_now", "check_time": "effective_check_time", "captured_at": "captured_at"}[name]] = timestamp.isoformat()
    if event in ("call", "exception") and "now" in frame.f_locals and frame.f_locals.get("now") is None:
        fields["supplied_now"] = None
    for name in ("current_source_commit", "resolved_source_commit", "source_commit"):
        source = frame.f_locals.get(name)
        if isinstance(source, str):
            fields[name] = source
    check_time = frame.f_locals.get("check_time")
    if isinstance(check_time, datetime):
        ages = {}
        for name in ("created_at", "sinfo_captured_at", "resource_captured_at"):
            timestamp = frame.f_locals.get(name)
            if isinstance(timestamp, datetime):
                ages[name] = (check_time - timestamp).total_seconds()
        fields["ages_at_effective_check_seconds"] = ages
        maximum_age = frame.f_locals.get("maximum_age")
        if isinstance(maximum_age, int):
            fields["maximum_age_seconds"] = maximum_age
    return frame.f_locals.get("receipt"), fields

@contextmanager
def observe_recovery_freshness(*, profile_calls=True):
    global freshness_event_count, freshness_observation
    if freshness_observation is not None:
        raise RuntimeError("freshness_observer_nested_scope_forbidden")
    selected = {
        runtime.build_and_publish_candidate_execution_gate.__code__: "gate_builder",
        runtime.resolve_candidate_execution_protocol.__code__: "protocol_resolution",
        runtime.resolve_candidate_attempt_context.__code__: "attempt_resolution",
        protocol.require_candidate_assignment_preflight.__code__: "assignment_preflight",
        protocol.require_candidate_execution_authority.__code__: "execution_authority",
        protocol._parse_utc_timestamp.__code__: "timestamp_parser",
        runner._run_candidate_command.__code__: "assignment_runner",
        trainer.run.__code__: "trainer",
        runtime.build_candidate_resource_receipt.__code__: "resource_receipt",
        runtime.require_candidate_dispatch_binding.__code__: "dispatch_binding",
        protocol._require_executing_commit.__code__: "executing_commit",
        handoff.executing_source_identity.__code__: "source_identity",
        handoff._check_loaded_source.__code__: "loaded_source",
    }
    if len(selected) != 13:
        raise RuntimeError("freshness_observer_ambiguous_code_mapping")
    selected_names = frozenset(code.co_name for code in selected)
    previous_profile, previous_trace = sys.getprofile(), sys.gettrace()
    observation_scope = "selected_call_profiling" if profile_calls else "explicit_events_only"
    costs = {"selected_event_count": 0, "selected_event_seconds": 0.0,
             "selected_event_errors": 0, "delegated_profile_event_count": 0,
             "delegated_profile_seconds": 0.0, "delegated_profile_errors": 0,
             "serialization_seconds": 0.0, "emit_seconds": 0.0}
    state = {"costs": costs, "record_counts": dict.fromkeys(("ordinary", "required", "terminal"), 0),
             "output_bytes": dict.fromkeys(("ordinary", "required", "terminal"), 0)}
    freshness_event_count, freshness_observation = 0, state
    rows, stack = {}, []
    outermost_seconds = 0.0
    scope_started = time.monotonic()
    failure = None
    terminal = None

    def profile(frame, event, value):
        nonlocal outermost_seconds
        # Every event still reaches an installed prior hook, including C events.
        if previous_profile is not None:
            delegated_started = time.monotonic()
            costs["delegated_profile_event_count"] += 1
            try:
                previous_profile(frame, event, value)
            except BaseException:
                costs["delegated_profile_errors"] += 1
                raise
            finally:
                costs["delegated_profile_seconds"] += time.monotonic() - delegated_started
        # The usual unselected path does no clock or counter bookkeeping.
        if event not in ("call", "return"):
            return
        if frame.f_code.co_name not in selected_names:
            return
        stage = selected.get(frame.f_code)
        if stage is None:
            return
        selected_event_started = time.monotonic()
        costs["selected_event_count"] += 1
        try:
            observed = time.monotonic()
            caller = frame.f_back
            caller_code = caller.f_code if caller is not None else None
            caller_line = caller.f_lineno if caller is not None else None
            if event == "call":
                key = (frame.f_code, caller_code, caller_line)
                if key not in rows:
                    if len(rows) >= 128:
                        raise RuntimeError("freshness_observer_aggregate_bound_exceeded")
                    rows[key] = {
                        "stage": stage, "function": frame.f_code.co_name,
                        "function_line": frame.f_code.co_firstlineno,
                        "caller": caller_code.co_name if caller_code is not None else None,
                        "caller_line": caller_line, "call_count": 0, "return_count": 0,
                        "inclusive_seconds": 0.0, "exclusive_selected_seconds": 0.0,
                        "selected_child_seconds": 0.0,
                    }
                if len(stack) >= 128:
                    raise RuntimeError("freshness_observer_selected_depth_exceeded")
                rows[key]["call_count"] += 1
                stack.append({"frame": id(frame), "key": key, "entered": observed, "children": 0.0})
            else:
                if not stack or stack[-1]["frame"] != id(frame):
                    raise RuntimeError("freshness_observer_unmatched_return")
                entry = stack.pop()
                row = rows[entry["key"]]
                inclusive = observed - entry["entered"]
                exclusive = inclusive - entry["children"]
                if exclusive < 0:
                    raise RuntimeError("freshness_observer_negative_exclusive_duration")
                row["return_count"] += 1
                row["inclusive_seconds"] += inclusive
                row["exclusive_selected_seconds"] += exclusive
                row["selected_child_seconds"] += entry["children"]
                if stack:
                    stack[-1]["children"] += inclusive
                else:
                    outermost_seconds += inclusive
            # Parser activity only contributes to aggregates, never the anchor budget.
            anchor = None
            if stage == "resource_receipt" and event == "call":
                anchor = "receipt_capture"
            elif stage == "gate_builder" and event == "return":
                anchor = "gate_return"
            elif (stage == "attempt_resolution" and caller_code is not None
                  and caller_code.co_name == "run_attempt" and caller_line == 147):
                anchor = "worker_post_gate_resolve_" + event
            elif stage == "assignment_runner" and event == "call":
                anchor = "runner_entry"
            elif caller_code is runner._run_candidate_command.__code__:
                if stage == "assignment_preflight":
                    anchor = "runner_preflight_" + event
                elif stage == "execution_authority":
                    anchor = "runner_authority_" + event
            if anchor is not None:
                receipt, fields = freshness_frame_observation(frame, event=event)
                if stage == "resource_receipt" and not isinstance(frame.f_locals.get("captured_at"), datetime):
                    raise RuntimeError("freshness_observer_capture_timestamp_missing")
                if stage == "gate_builder" and isinstance(value, dict):
                    receipt = value.get("resource_environment_receipt")
                freshness_event("anchor", stage=stage, receipt=receipt, required=True,
                                anchor=anchor, **fields)
        except BaseException:
            costs["selected_event_errors"] += 1
            raise
        finally:
            costs["selected_event_seconds"] += time.monotonic() - selected_event_started

    try:
        if profile_calls:
            sys.setprofile(profile)
        freshness_event("call_enter", stage="observer", required=True,
                        profile_calls=profile_calls, observation_scope=observation_scope)
        yield
    except BaseException as error:
        failure = error
        stage, line, receipt, fields = "fixture", None, None, {}
        traceback = error.__traceback__
        while traceback is not None:
            if traceback.tb_frame.f_code in selected:
                stage = selected[traceback.tb_frame.f_code]
                line = traceback.tb_lineno
                receipt, fields = freshness_frame_observation(traceback.tb_frame, event="exception")
            traceback = traceback.tb_next
        terminal = dict(stage=stage, receipt=receipt, exception_type=type(error).__name__,
                        source_line=line, **fields)
        raise
    finally:
        try:
            if profile_calls:
                sys.setprofile(previous_profile)
        finally:
            sys.settrace(previous_trace)
        report_error = None
        try:
            scope_ended = time.monotonic()
            aggregates = sorted(rows.values(), key=lambda row: (
                row["stage"], row["function"], row["caller"] or "", row["caller_line"] or -1))
            exclusive_total = sum(row["exclusive_selected_seconds"] for row in aggregates)
            unreturned = [{"stage": rows[entry["key"]]["stage"],
                           "function": rows[entry["key"]]["function"],
                           "caller": rows[entry["key"]]["caller"],
                           "caller_line": rows[entry["key"]]["caller_line"],
                           "observed_partial_seconds": scope_ended - entry["entered"]}
                          for entry in stack]
            if unreturned and failure is None:
                raise RuntimeError("freshness_observer_unreturned_frames_without_error")
            if not unreturned and abs(exclusive_total - outermost_seconds) > max(1e-9, outermost_seconds * 1e-9):
                raise RuntimeError("freshness_observer_duration_reconciliation_failed")
            freshness_event("call_exit", stage="observer", required=True,
                            elapsed_seconds=scope_ended - scope_started)
            freshness_event("latency_summary", stage="observer", required=True,
                            aggregates=aggregates, unreturned_calls=unreturned,
                            accounting_complete=not unreturned,
                            outermost_completed_seconds=outermost_seconds,
                            exclusive_selected_seconds=exclusive_total,
                            duration_semantics=("Instrumented wall time; exclusive removes direct selected children only. Return events include exception unwinding. Unreturned frames have partial observations only."
                                                 if profile_calls else
                                                 "Explicit events and scope wall time only; no per-function profiling or latency attribution is collected."))
        except BaseException as observation_error:
            report_error = observation_error
        try:
            if report_error is not None:
                if terminal is None:
                    terminal = dict(stage="observer", exception_type=type(report_error).__name__, source_line=None)
                else:
                    terminal["observation_error_type"] = type(report_error).__name__
            if terminal is not None:
                freshness_event("exception", **terminal)
            freshness_event("hooks_restored", stage="observer",
                            profile_calls=profile_calls, observation_scope=observation_scope,
                            profile_restored=sys.getprofile() is previous_profile,
                            trace_restored=sys.gettrace() is previous_trace,
                            observed_costs=dict(costs),
                            cost_semantics="Selected-event time excludes filtering and prior-hook delegation; it measures selected Python call/return handling, including in-handler serialization/emission. Selected-event counts/errors cover only handling reached after delegation. Delegated-profile counts/errors cover all prior-hook invocation attempts, including C/unselected events; its time includes invocation bookkeeping. Unselected events without a prior hook are not metered. Counters overlap aggregate wall time and must not be summed. Serialization/emission counters cover prior records only. These are not counterfactual total profiling overhead.",
                            record_counts_before_this_record=dict(state["record_counts"]),
                            output_bytes_before_this_record=dict(state["output_bytes"]))
        except BaseException as terminal_error:
            errors = [error for error in (failure, report_error, terminal_error) if error is not None]
            raise BaseExceptionGroup("freshness terminal reporting failed", errors) from None
        finally:
            freshness_observation = None
        if report_error is not None:
            if failure is not None:
                raise BaseExceptionGroup("freshness observation failed while preserving the original error", [failure, report_error]) from None
            raise report_error
"""

_RECOVERY_CONSUMER_RETENTION_FIXTURE = r"""
from Libs.exp.training import run_operational_twin_candidate_attempt as worker
from Libs.utils.evaluation import operational_twin_candidate_evidence as evidence
paths = resolved.execution["output_paths"]
scheduler = evidence.load_scheduler_receipts(
    root, resolved.execution, dispatch["initial_arrays"], phase="initial",
    source_commit=dispatch["source_commit"], resolved_context=resolved,
)
assert set(scheduler) == set(range(15))
recovery = evidence.require_recovery_eligibility(
    config, root, root / paths["recovery_eligibility"], resolved_context=resolved,
)
assert [row["recovery_command_index"] for row in recovery["assignments"]
        if row["recovery_eligibility"] == "ELIGIBLE"] == [15]
retained_protocol = runtime.resolve_candidate_execution_protocol(config, root)
initial_command_uid = resolved.command_catalog["commands"][0]["attempt_uid"]
initial_commands = [
    row for row in retained_protocol.command_catalog["commands"]
    if row["attempt_uid"] == initial_command_uid
]
assert len(initial_commands) == 1
initial_command = initial_commands[0]
initial_assignments = [
    row for row in retained_protocol.assignment_manifest["assignments"]
    if row["assignment_uid"] == initial_command["assignment_uid"]
]
assert len(initial_assignments) == 1
initial_assignment = initial_assignments[0]
initial_attempts = [
    row for row in initial_assignment["attempts"]
    if row["attempt_uid"] == initial_command["attempt_uid"]
]
assert len(initial_attempts) == 1
initial_attempt = initial_attempts[0]
assert initial_command["command_index"] == 0
assert initial_attempt["attempt_kind"] == "initial"
owner = runtime.require_candidate_attempt_ownership(
    config, root, retained_protocol, initial_attempt["attempt_uid"])
initial_materialized, initial_materialized_blockers = evidence.validate_materialized_attempt(
    config,
    root,
    assignment=initial_assignment,
    attempt=initial_attempt,
    command=initial_command,
    scheduler_evidence=scheduler[0],
    source_commit=owner["source_commit"],
    candidate_manifest=retained_protocol.candidate_manifest,
    assignment_manifest=retained_protocol.assignment_manifest,
    command_catalog=retained_protocol.command_catalog,
    preflight=retained_protocol.preflight,
    resolved_context=retained_protocol,
)
assert not initial_materialized_blockers, initial_materialized_blockers
assert initial_materialized["materialized"] is True
assert initial_materialized["status"] == "FAILED"
assert initial_materialized["failure_class"] == "infrastructure_interruption"
retained_state_bytes = {
    row["path"]: (root / row["path"]).read_bytes()
    for row in expected_state["path_identities"]
}

with controller_environment():
    recovery_dispatch = submit_arrays(config, root, mode="recovery", command_capture=scheduler_boundary)
assert recovery_dispatch["selected_indices"] == [15]
assert recovery_dispatch["source_commit"] == expected_state["head"]
assert recovery_dispatch["source_commit"] != dispatch["source_commit"]
recovery_dispatch = require_candidate_dispatch_manifest(
    config, root, recovery_dispatch, mode="recovery", resolved_context=resolved,
)
recovery_release = require_candidate_release_manifest(config, root, resolved, recovery_dispatch)
assert recovery_release["status"] == "RELEASED"

# A committed recovery-ready input may not drift after the process boundary.
# Dispatch validation must reject the current bytes, then accept the exact
# restored bytes. The mutation is never committed or used to create a gate.
calibration_path = root / paths["calibration_execution_authority"]
calibration_bytes = calibration_path.read_bytes()
calibration_path.write_bytes(calibration_bytes + b"\n")
try:
    require_candidate_dispatch_manifest(
        config, root, recovery_dispatch, mode="recovery", resolved_context=resolved,
    )
except protocol.OperationalTwinProtocolError as error:
    assert "calibration_drift" in str(error), error
else:
    raise AssertionError("committed recovery-ready current-byte drift was accepted")
finally:
    calibration_path.write_bytes(calibration_bytes)
assert subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True,
).strip() == ""
require_candidate_dispatch_manifest(
    config, root, recovery_dispatch, mode="recovery", resolved_context=resolved,
)

command, capture = prepare_runtime(15, runtime_dispatch=recovery_dispatch)
training_argv = [
    sys.executable, "-m", "Libs.exp.training.run_operational_twin_assignment_command",
    "--protocol-scope", "candidate-lineage", "--task-index", "15",
]

# Establish the stale-negative precondition using only retained initial bytes.
# No recovery receipt has been captured or minted at this point.
from datetime import timedelta


def wait_for_retained_resource_expiry(receipt_bytes, maximum_age):
    "Wait for actual expiry; this helper neither creates nor authorizes receipts."
    if not isinstance(receipt_bytes, bytes):
        raise AssertionError("retained_resource_expiry_requires_immutable_bytes")
    if type(maximum_age) is not int or maximum_age <= 0:
        raise AssertionError("retained_resource_expiry_invalid_maximum_age")
    receipt = json.loads(receipt_bytes)
    if not isinstance(receipt, dict):
        raise AssertionError("retained_resource_expiry_invalid_timestamp_record")
    timestamp_fields = ("created_at", "sinfo_captured_at", "resource_captured_at")
    timestamps = {
        field: protocol._parse_utc_timestamp(receipt.get(field), label=field)
        for field in timestamp_fields
    }
    started_at = datetime.now(timezone.utc)
    if any(timestamp > started_at for timestamp in timestamps.values()):
        raise protocol.OperationalTwinProtocolError("candidate_resource_receipt_from_future")
    if timestamps["created_at"] < max(
        timestamps["sinfo_captured_at"], timestamps["resource_captured_at"],
    ):
        raise protocol.OperationalTwinProtocolError("candidate_resource_receipt_timestamp_order_invalid")
    # The unchanged validator rejects when any of the three ages exceeds TTL.
    expiry = min(timestamps.values()) + timedelta(seconds=maximum_age)
    started = time.monotonic()
    wait_intervals = 0
    context = {
        "event": "retained_resource_expiry_precondition",
        "timestamps": {field: receipt[field] for field in timestamp_fields},
        "maximum_age_seconds": maximum_age,
        "expiry_utc": expiry.isoformat(),
        "started_at_utc": started_at.isoformat(),
    }
    print(json.dumps({**context, "status": "CHECKING", "wait_seconds": 0.0}, sort_keys=True), flush=True)
    while True:
        checked_at = datetime.now(timezone.utc)
        if any(timestamp > checked_at for timestamp in timestamps.values()):
            raise protocol.OperationalTwinProtocolError("candidate_resource_receipt_from_future")
        if checked_at > expiry:
            break
        time.sleep(min(30.0, max(0.001, (expiry - checked_at).total_seconds())))
        wait_intervals += 1
    report = {
        **context,
        "status": "EXPIRED",
        "checked_at_utc": checked_at.isoformat(),
        "ages_seconds": {field: (checked_at - timestamp).total_seconds()
                         for field, timestamp in timestamps.items()},
        "wait_seconds": time.monotonic() - started,
        "wait_intervals": wait_intervals,
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


retained_initial_receipt_bytes = retained_state_bytes[initial_command["resource_environment_receipt"]]
assert (root / initial_command["resource_environment_receipt"]).read_bytes() == retained_initial_receipt_bytes
retained_expiry_contract = protocol._candidate_execution_contract(
    protocol._candidate_protocol(config, retained_protocol), retained_protocol,
)
retained_expiry = wait_for_retained_resource_expiry(
    retained_initial_receipt_bytes,
    retained_expiry_contract["maximum_resource_receipt_age_seconds"],
)
assert (root / initial_command["resource_environment_receipt"]).read_bytes() == retained_initial_receipt_bytes

# The retained initial receipt is deliberately old by the time this fresh
# process reaches recovery. Its exact bytes must still fail the unchanged
# freshness predicate; source and runtime checks are otherwise kept strict.
try:
    protocol.require_candidate_execution_authority(
        config,
        root,
        assignment=initial_assignment,
        attempt=initial_attempt,
        candidate_manifest=retained_protocol.candidate_manifest,
        assignment_manifest=retained_protocol.assignment_manifest,
        command_catalog=retained_protocol.command_catalog,
        preflight=retained_protocol.preflight,
        execution_authority_path=initial_command["execution_authority"],
        resource_environment_receipt_path=initial_command["resource_environment_receipt"],
        cuda_available=False,
        cuda_device_count=0,
        torch_version=str(torch.__version__),
        current_source_commit=expected_state["initial_dispatch_source_commit"],
        validate_current_runtime=False,
        resolved_context=retained_protocol,
    )
except protocol.OperationalTwinProtocolError as error:
    assert "candidate_resource_receipt_stale" in str(error), error
else:
    raise AssertionError("retained initial resource receipt was not rejected as stale")
stale_initial_receipt_rejections = 1

# Only the still-absent recovery slot may traverse the fresh preflight.
assert not (root / command["output_root"]).exists()
assert all(not (root / command[key]).exists()
           for key in ("resource_environment_receipt", "execution_authority"))
recovery_context = runtime.resolve_candidate_attempt_context(config, root, 15)
calibration_blockers = runtime.validate_candidate_calibration_execution_authority(
    config, root, calibration_path,
    executing_source_commit=recovery_dispatch["source_commit"],
    resolved_context=recovery_context["resolved_protocol"],
)
assert not calibration_blockers, calibration_blockers
runtime.require_candidate_worker_binding(root, recovery_context["resolved_protocol"], required=True)
prewarmed_dispatch = runtime.require_candidate_dispatch_binding(
    config, root, recovery_context["resolved_protocol"],
    assignment=recovery_context["assignment"], attempt=recovery_context["attempt"],
    source_commit=recovery_dispatch["source_commit"],
)
assert prewarmed_dispatch["command_index"] == 15

# Warm the complete runner preflight before the one-shot recovery capture.
# This uses the maintained predicate and carries no resource authority.
prewarmed_candidate = protocol.require_candidate_assignment_preflight(
    config,
    root,
    candidate_lineage_uid=str(command["candidate_lineage_uid"]),
    protocol_uid=str(command["protocol_uid"]),
    assignment_uid=str(command["assignment_uid"]),
    attempt_uid=str(command["attempt_uid"]),
    method_id=str(command["method_id"]),
    seed=int(command["seed"]),
    output_root=str(command["output_root"]),
    candidate_manifest_hash=str(resolved.command_catalog["candidate_manifest_hash"]),
    assignment_manifest_hash=str(resolved.command_catalog["assignment_manifest_hash"]),
    command_catalog_hash_value=str(resolved.command_catalog["command_catalog_hash"]),
    preflight_uid=str(resolved.command_catalog["preflight_uid"]),
    state_inputs={field: None for field in (
        "resume_input", "history_input", "checkpoint_input",
        "optimizer_state_input", "model_state_input", "calibration_state_input",
    )},
    require_recovery_predecessor=True,
    resolved_context=recovery_context["resolved_protocol"],
)
assert prewarmed_candidate["attempt"]["attempt_uid"] == command["attempt_uid"]
print(json.dumps({
    "status": "RECOVERY_PREVALIDATION_READY_FOR_FRESH_CAPTURE",
    "command_index": 15, "gate_files_absent": True,
    "cross_process_state_hash": expected_state["state_hash"],
    "production_calls": 0,
}), flush=True)

real_run = subprocess.run
delegated = []
fresh_capture_calls = []
late_resource_calls = []
execution_events = []
stop_kind = "interruption"
def fresh_capture(argv, current_root):
    fresh_capture_calls.append(list(argv))
    execution_events.append("sinfo" if argv[0] == "sinfo" else "resource")
    capture_kind = "sinfo" if argv[0] == "sinfo" else "resource"
    freshness_event("capture_enter", stage=capture_kind)
    try:
        return capture(argv, current_root)
    finally:
        freshness_event("capture_exit", stage=capture_kind)

real_worker_gate_builder = worker.build_and_publish_candidate_execution_gate
worker_gate_calls = []
worker_gate_results = []

def observe_worker_gate(*args, **kwargs):
    assert not worker_gate_calls
    assert len(args) == 2 and args[1] == root
    assert kwargs == {"task_index": 15}
    worker_gate_calls.append((args, kwargs))
    gate = real_worker_gate_builder(*args, **kwargs)
    worker_gate_results.append(gate)
    return gate

# Observe the gate built by the worker at its production call site.
with observe_recovery_freshness(
    profile_calls=globals().get("RECOVERY_PROFILE_CALLS", True)
):
    def launch(argv, *args, **kwargs):
        argv = list(argv)
        if argv[0] in ("sinfo", "scontrol"):
            if (
                len(worker_gate_calls) != 1
                or worker_gate_results
                or len(fresh_capture_calls) >= 2
            ):
                late_resource_calls.append(argv)
                raise AssertionError(
                    "resource evidence capture was outside the one worker gate build"
                )
            assert not args
            assert kwargs == {
                "cwd": root,
                "check": True,
                "capture_output": True,
                "text": True,
            }
            assert argv[0] == ("sinfo", "scontrol")[len(fresh_capture_calls)]
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=fresh_capture(argv, kwargs["cwd"]),
                stderr="",
            )
        if argv == training_argv:
            delegated.append("runner")
            execution_events.append("runner")
            old_argv = sys.argv
            sys.argv = [argv[2], *argv[3:]]
            try:
                return subprocess.CompletedProcess(argv, runner.main())
            finally:
                sys.argv = old_argv
        if argv == command["argv"]:
            delegated.append("trainer")
            execution_events.append("trainer")
            old_argv, old_trace = sys.argv, sys.gettrace()
            sys.argv = [argv[1], *argv[2:]]
            sys.settrace(trace)
            try:
                trainer.main()
            except InterruptedError as error:
                assert "Non-authoritative pre-seed" in str(error)
                return subprocess.CompletedProcess(argv, 173)
            finally:
                sys.settrace(old_trace)
                sys.argv = old_argv
            raise AssertionError("trainer escaped its terminating pre-seed boundary")
        return real_run(argv, *args, **kwargs)

    with patch.object(
        worker,
        "build_and_publish_candidate_execution_gate",
        side_effect=observe_worker_gate,
    ), patch.object(subprocess, "run", side_effect=launch):
        assert worker.run_attempt(15) == 173
assert delegated == ["runner", "trainer"]
assert execution_events[:4] == ["sinfo", "resource", "runner", "trainer"]
assert not late_resource_calls
assert len(fresh_capture_calls) == 2
assert len(worker_gate_calls) == len(worker_gate_results) == 1

fresh_gate = worker_gate_results[0]
fresh_receipt_path = root / command["resource_environment_receipt"]
fresh_authority_path = root / command["execution_authority"]
fresh_receipt = json.loads(fresh_receipt_path.read_text())
fresh_authority = json.loads(fresh_authority_path.read_text())
initial_receipt = json.loads(
    (root / initial_command["resource_environment_receipt"]).read_text()
)
fresh_receipt_distinct = fresh_receipt["receipt_uid"] != initial_receipt["receipt_uid"]
assert fresh_receipt_distinct
assert (
    fresh_receipt["attempt_uid"]
    == fresh_authority["attempt_uid"]
    == fresh_gate["attempt_uid"]
    == command["attempt_uid"]
)
assert fresh_gate["task_index"] == 15
assert fresh_receipt["runtime_dispatch_binding"]["command_index"] == 15
assert fresh_gate["source_commit"] == recovery_dispatch["source_commit"]
assert (
    fresh_gate["resource_environment_receipt"]["receipt_uid"]
    == fresh_receipt["receipt_uid"]
)
assert (
    fresh_gate["execution_authority"]["authority_uid"]
    == fresh_authority["authority_uid"]
)
assert (
    fresh_gate["runtime_dispatch_binding"]
    == fresh_authority["runtime_dispatch_binding"]
)
recovery_output = root / command["output_root"]
recovery_owner = runtime.require_candidate_attempt_ownership(
    config, root, resolved, command["attempt_uid"]
)
assert recovery_owner["assignment_uid"] == owner["assignment_uid"]
assert recovery_owner["source_commit"] == recovery_dispatch["source_commit"]
assert (
    json.loads((recovery_output / "terminal_status.json").read_text())["failure_class"]
    == "infrastructure_interruption"
)
recovery_phase_receipt = json.loads(
    (recovery_output / worker.PHASE_RECEIPT_NAME).read_text()
)
assert recovery_phase_receipt["status"] == "FAIL"
assert recovery_phase_receipt["command_index"] == fresh_gate["task_index"]
worker_gate_phase_receipt_matches = all(
    recovery_phase_receipt[field] == fresh_gate[field]
    for field in (
        "source_commit",
        "assignment_uid",
        "attempt_uid",
        "successor_execution_binding",
        "runtime_dispatch_binding",
        "execution_authority",
        "resource_environment_receipt",
    )
)
assert worker_gate_phase_receipt_matches
recovery_scheduler_path = root / paths["scheduler_receipts_root"] / ("recovery_" + recovery_dispatch["recovery_arrays"][0]["job_id"] + ".json")
def reused_recovery_identity(argv, current):
    text = accounting_boundary(argv, current, manifest=recovery_dispatch)
    return text.replace("|875015|", "|875000|")
try:
    evidence.collect_scheduler_receipts(
        root, resolved.execution, recovery_dispatch["recovery_arrays"], phase="recovery",
        source_commit=recovery_dispatch["source_commit"], resolved_context=resolved,
        command_capture=reused_recovery_identity)
except protocol.OperationalTwinProtocolError as error:
    assert "internal_identity_reused" in str(error), error
else:
    raise AssertionError("initial/recovery internal JobIDRaw reuse was accepted")
assert not recovery_scheduler_path.exists()
recovery_scheduler = evidence.collect_scheduler_receipts(
    root, resolved.execution, recovery_dispatch["recovery_arrays"], phase="recovery",
    source_commit=recovery_dispatch["source_commit"], resolved_context=resolved,
    command_capture=lambda argv, current: accounting_boundary(argv, current, manifest=recovery_dispatch))
joined = evidence.merge_scheduler_evidence(scheduler, recovery_scheduler)
assert set(joined) == set(range(15)) | {15}
assert recovery_scheduler == evidence.load_scheduler_receipts(
    root, resolved.execution, recovery_dispatch["recovery_arrays"], phase="recovery",
    source_commit=recovery_dispatch["source_commit"], resolved_context=resolved)
before_duplicate = {path: (root / path).read_bytes() for path in (
    command["resource_environment_receipt"], command["execution_authority"],
    command["output_root"] + "/attempt_ownership.json")}
try:
    runtime.resolve_candidate_attempt_context(config, root, 15)
except protocol.OperationalTwinProtocolError:
    pass
else:
    raise AssertionError("occupied recovery slot could be reclaimed")
assert all((root / path).read_bytes() == value for path, value in before_duplicate.items())

retention = [
    paths["recovery_dispatch_claim"], paths["recovery_dispatch_manifest"],
    paths["recovery_dispatch_release_manifest"],
    *[row["submission_receipt"]["path"] for row in recovery_dispatch["recovery_arrays"]],
    *[row["intent"]["path"] for row in recovery_release["attempts"]],
    *[row["receipt"]["path"] for row in recovery_release["attempts"]],
    *[row["receipt"]["path"] for row in recovery_scheduler.values()],
    *before_duplicate,
    command["output_root"] + "/terminal_status.json",
    command["output_root"] + "/" + worker.PHASE_RECEIPT_NAME,
]
git("add", "--", *retention)
git("commit", "-m", "Retain interrupted successor recovery and exact scheduler receipts")
metadata_commit = git("rev-parse", "HEAD").decode().strip()
assert metadata_commit not in (dispatch["source_commit"], recovery_dispatch["source_commit"])
old_argv = sys.argv
sys.argv = [catalog_builder.__file__]
try:
    with controller_environment():
        assert catalog_builder.main() == 1
finally:
    sys.argv = old_argv
catalog = json.loads((root / paths["candidate_execution_catalog"]).read_text())
assert catalog["assignment_count"] == 15 and len(catalog["assignments"]) == 15
assert catalog["attempt_slot_count"] == 30 and len(catalog["attempt_slots"]) == 30
assert len({row["attempt_uid"] for row in catalog["attempt_slots"]}) == 30
assert catalog["status"] == "FAIL"
assert catalog["terminal_decision"] == resolved.execution["terminal_policy"]["incomplete"]
assert catalog["closed_flags"]["scientific_twin_result"] == "NOT_RUN"
assert set(catalog["access_counters"].values()) == {0}
assert evidence.validate_execution_catalog(config, root, catalog, resolved_context=resolved) == []
materialized_slots = [row for row in catalog["attempt_slots"] if row["materialized"]]
assert len(materialized_slots) == 2
assert all(row["status"] == "FAILED" and row["failure_class"] == "infrastructure_interruption"
           and row["checkpoint"] is None and row["evaluation_summary"] is None
           and row["validation_blockers"] == [] for row in materialized_slots)
assert all(row["status"] != "COMPLETED" for row in catalog["attempt_slots"])
assert all((root / path).read_bytes() == data for path, data in retained_state_bytes.items())
assert stops == ["trainer"] and len(observed_inputs) == 3
runtime.require_candidate_runtime_ownership(config, root, resolved)
print(json.dumps({
    "status": "WORKER_RECOVERY_SCHEDULER_AND_INCOMPLETE_CATALOG_VERIFIED",
    "trainer_stops": expected_state["trainer_stops"] + len(stops),
    "identity_guard_reads": expected_state["identity_guard_reads"] + len(observed_inputs),
    "authenticated_input_ref_tuples": len(refs) * len(input_rows),
    "receipt_content_guard_reads": 0, "owned_failed_attempts": 2,
    "worker_phase_receipts": 2, "same_byte_publication_checked": True,
    "raw_identity_rejections": 4,
    "changed_byte_rejections": expected_state["changed_byte_rejections"],
    "cross_process_current_byte_rejections": 1,
    "cross_process_state_hash": expected_state["state_hash"],
    "cross_process_clean_boundary": True,
    "resource_capture_calls": len(fresh_capture_calls),
    "stale_initial_receipt_rejections": stale_initial_receipt_rejections,
    "fresh_receipt_distinct": fresh_receipt_distinct,
    "worker_gate_build_calls": len(worker_gate_calls),
    "worker_gate_phase_receipt_matches": worker_gate_phase_receipt_matches,
    "assignment_count": 15, "attempt_slot_count": 30, "eligible_recovery_indices": [15],
    "catalog_builder_exit": 1, "catalog_validation_blockers": 0,
    "scientific_results": 0, "production_calls": 0,
}))
"""


def test_worker_precompute_phase_and_retained_artifact_identity(isolated_execution_ready, tmp_path):
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    recovery_ready = _isolated_python(
        root,
        _PRECOMPUTE_IDENTITY_FIXTURE
        + _WORKER_PRESEED_RETAINED_FIXTURE
        + _SCHEDULER_ACCOUNTING_FIXTURE
        + _RECOVERY_READY_PRODUCER_FIXTURE,
        stage="initial worker and committed recovery-ready state",
    )
    assert recovery_ready["status"] == "RECOVERY_READY_COMMITTED_STATE_VERIFIED"
    assert recovery_ready["recovery_command_index"] == 15
    assert recovery_ready["eligible_recovery_indices"] == [15]
    assert recovery_ready["committed_path_count"] == len(recovery_ready["path_identities"])
    assert recovery_ready["removed_transport_paths"] == 5
    assert recovery_ready["removed_placeholder_logs"] == 30
    assert recovery_ready["scientific_results"] == 0
    assert recovery_ready["production_calls"] == 0
    consumer_preamble = (
        "FIXTURE_MODE = 'recovery_consumer'\n"
        "RECOVERY_PROFILE_CALLS = False\n"
        + "EXPECTED_RECOVERY_READY_STATE_JSON = "
        + repr(json.dumps(recovery_ready, sort_keys=True))
        + "\n"
    )
    result = _isolated_python(
        root,
        consumer_preamble
        + _RECOVERY_READY_STATE_VERIFIER_FIXTURE
        + _PRECOMPUTE_IDENTITY_FIXTURE
        + _RECOVERY_FRESHNESS_OBSERVER_FIXTURE
        + _SCHEDULER_ACCOUNTING_FIXTURE
        + _RECOVERY_CONSUMER_RETENTION_FIXTURE,
        stage="fresh recovery worker, scheduler retention and incomplete catalog",
    )
    assert result == {
        "status": "WORKER_RECOVERY_SCHEDULER_AND_INCOMPLETE_CATALOG_VERIFIED",
        "trainer_stops": 2, "identity_guard_reads": 6,
        "authenticated_input_ref_tuples": 16, "receipt_content_guard_reads": 0,
        "owned_failed_attempts": 2, "worker_phase_receipts": 2,
        "same_byte_publication_checked": True, "raw_identity_rejections": 4,
        "assignment_count": 15, "attempt_slot_count": 30, "eligible_recovery_indices": [15],
        "catalog_builder_exit": 1, "catalog_validation_blockers": 0,
        "changed_byte_rejections": 2, "cross_process_current_byte_rejections": 1,
        "cross_process_state_hash": recovery_ready["state_hash"],
        "cross_process_clean_boundary": True, "resource_capture_calls": 2,
        "stale_initial_receipt_rejections": 1, "fresh_receipt_distinct": True,
        "worker_gate_build_calls": 1, "worker_gate_phase_receipt_matches": True,
        "scientific_results": 0, "production_calls": 0,
    }


def test_trainer_precompute_identity_guard_and_second_owner(isolated_execution_ready, tmp_path):
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    result = _isolated_python(
        root, _PRECOMPUTE_IDENTITY_FIXTURE + _TRAINER_PRESEED_OWNERSHIP_FIXTURE,
        stage="genuine trainer identity and second owner",
    )
    assert result == {
        "status": "TRAINER_PRESEED_AND_SECOND_OWNER_VERIFIED",
        "trainer_stops": 2, "identity_guard_reads": 6,
        "authenticated_input_ref_tuples": 16, "receipt_content_guard_reads": 0,
        "owned_failed_attempts": 2, "scientific_results": 0, "production_calls": 0,
    }


def test_evaluator_precompute_identity_guard_and_owned_metadata(isolated_execution_ready, tmp_path):
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    result = _isolated_python(
        root, _PRECOMPUTE_IDENTITY_FIXTURE + _EVALUATOR_PRESEED_METADATA_FIXTURE,
        stage="genuine evaluator identity and owned terminal metadata",
    )
    assert result == {
        "status": "EVALUATOR_IDENTITY_OWNERSHIP_TERMINAL_PRESEED_VERIFIED",
        "trainer_stops": 1, "evaluator_stops": 1, "identity_guard_reads": 6,
        "authenticated_input_ref_tuples": 16, "receipt_content_guard_reads": 0,
        "metadata_only_fixture": True, "scientific_results": 0, "production_calls": 0,
    }


@pytest.fixture(scope="module")
def retained_resource_expiry_unit():
    """Compile only the wait helper; records below are metadata, never authority."""
    from datetime import datetime, timedelta, timezone
    from Libs.utils.evaluation import operational_twin_training_protocol as protocol

    tree = ast.parse(_RECOVERY_CONSUMER_RETENTION_FIXTURE)
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "wait_for_retained_resource_expiry")
    namespace = {
        "datetime": datetime, "timedelta": timedelta, "timezone": timezone,
        "time": time, "json": json, "protocol": protocol,
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]),
                 "<retained-resource-expiry-helper-only>", "exec"), namespace)
    # Read the canonical constraint without loading config or data fixtures.
    protocol_tree = ast.parse(Path(protocol.__file__).read_text())
    canonical_contracts = [node.value for node in ast.walk(protocol_tree)
                           if isinstance(node, ast.Assign) and any(
                               isinstance(target, ast.Name) and target.id == "expected_execution_authority"
                               for target in node.targets)]
    assert len(canonical_contracts) == 1
    maximum_age = ast.literal_eval(next(value for key, value in zip(
        canonical_contracts[0].keys, canonical_contracts[0].values,
    ) if isinstance(key, ast.Constant) and key.value == "maximum_resource_receipt_age_seconds"))
    assert type(maximum_age) is int and maximum_age > 2
    return namespace[helper.name], datetime, timedelta, timezone, protocol, maximum_age


def test_retained_resource_expiry_waits_on_real_clock_and_preserves_bytes(retained_resource_expiry_unit, capsys):
    wait, datetime, timedelta, timezone, _, maximum_age = retained_resource_expiry_unit
    # All three raw metadata timestamps approach the actual canonical expiry.
    # These unit records are not resource receipts and never confer authority.
    timestamp = datetime.now(timezone.utc) - timedelta(seconds=maximum_age - 2)
    metadata = {field: timestamp.isoformat()
                for field in ("created_at", "sinfo_captured_at", "resource_captured_at")}
    payload = json.dumps(metadata, indent=2).encode()
    before = payload[:]
    result = wait(payload, maximum_age)
    checked_at = datetime.fromisoformat(result["checked_at_utc"])
    expiry = timestamp + timedelta(seconds=maximum_age)
    assert payload == before and json.loads(payload) == metadata
    assert result["timestamps"] == metadata
    assert datetime.fromisoformat(result["expiry_utc"]) == expiry
    assert checked_at > expiry and all(age > maximum_age for age in result["ages_seconds"].values())
    # No upper timing bound: scheduler delays must not make this check flaky.
    if datetime.fromisoformat(result["started_at_utc"]) <= expiry:
        assert result["wait_intervals"] >= 1
    assert result["wait_seconds"] >= 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["status"] for row in records] == ["CHECKING", "EXPIRED"]
    assert records[-1] == result


@pytest.mark.parametrize("oldest_field", ["sinfo_captured_at", "resource_captured_at"])
def test_retained_resource_expiry_already_expired_uses_any_timestamp(retained_resource_expiry_unit, oldest_field):
    wait, datetime, timedelta, timezone, _, maximum_age = retained_resource_expiry_unit
    current = datetime.now(timezone.utc)
    metadata = {field: (current - timedelta(seconds=maximum_age - 2)).isoformat()
                for field in ("created_at", "sinfo_captured_at", "resource_captured_at")}
    metadata[oldest_field] = (current - timedelta(seconds=maximum_age + 1)).isoformat()
    payload = json.dumps(metadata).encode()
    result = wait(payload, maximum_age)
    assert result["status"] == "EXPIRED" and result["wait_intervals"] == 0
    assert result["ages_seconds"][oldest_field] > maximum_age
    assert datetime.fromisoformat(result["expiry_utc"]) == current - timedelta(seconds=1)
    assert json.loads(payload) == metadata


@pytest.mark.parametrize("field", ["created_at", "sinfo_captured_at", "resource_captured_at"])
@pytest.mark.parametrize("invalid", ["malformed", "naive", "future", "missing"])
def test_retained_resource_expiry_rejects_invalid_timestamp_data(retained_resource_expiry_unit, field, invalid):
    wait, datetime, timedelta, timezone, protocol, maximum_age = retained_resource_expiry_unit
    current = datetime.now(timezone.utc)
    metadata = {name: (current - timedelta(seconds=maximum_age - 2)).isoformat()
                for name in ("created_at", "sinfo_captured_at", "resource_captured_at")}
    if invalid == "malformed":
        metadata[field] = "invalid-timestamp"
    elif invalid == "naive":
        metadata[field] = current.replace(tzinfo=None).isoformat()
    elif invalid == "future":
        metadata[field] = (current + timedelta(days=1)).isoformat()
    else:
        del metadata[field]
    payload = json.dumps(metadata).encode()
    expected = "from_future" if invalid == "future" else "timestamp_invalid"
    with pytest.raises(protocol.OperationalTwinProtocolError, match=expected):
        wait(payload, maximum_age)
    assert json.loads(payload) == metadata


def test_retained_resource_expiry_rejects_timestamp_order(retained_resource_expiry_unit):
    wait, datetime, timedelta, timezone, protocol, maximum_age = retained_resource_expiry_unit
    current = datetime.now(timezone.utc)
    metadata = {name: (current - timedelta(seconds=maximum_age - 2)).isoformat()
                for name in ("created_at", "sinfo_captured_at", "resource_captured_at")}
    metadata["created_at"] = (current - timedelta(seconds=maximum_age + 1)).isoformat()
    with pytest.raises(protocol.OperationalTwinProtocolError, match="timestamp_order_invalid"):
        wait(json.dumps(metadata).encode(), maximum_age)


@pytest.mark.parametrize("maximum_age", [0, -1, True, 2.0, "2", None])
def test_retained_resource_expiry_rejects_invalid_age_contract(retained_resource_expiry_unit, maximum_age):
    wait, _, _, _, _, _ = retained_resource_expiry_unit
    with pytest.raises(AssertionError, match="invalid_maximum_age"):
        wait(b"{}", maximum_age)


@pytest.mark.parametrize("payload,expected", [
    (bytearray(b"{}"), "requires_immutable_bytes"),
    (b"[]", "invalid_timestamp_record"),
    (b"invalid-json", "Expecting value"),
])
def test_retained_resource_expiry_rejects_invalid_record(retained_resource_expiry_unit, payload, expected):
    wait, _, _, _, _, maximum_age = retained_resource_expiry_unit
    with pytest.raises((AssertionError, json.JSONDecodeError), match=expected):
        wait(payload, maximum_age)


def test_retained_resource_expiry_precondition_keeps_real_clock_and_authority_order():
    tree = ast.parse(_RECOVERY_CONSUMER_RETENTION_FIXTURE)
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "wait_for_retained_resource_expiry")
    assert [arg.arg for arg in helper.args.args] == ["receipt_bytes", "maximum_age"]
    calls = [node for node in ast.walk(helper) if isinstance(node, ast.Call)]
    assert any(ast.unparse(call.func) == "protocol._parse_utc_timestamp" for call in calls)
    assert sum(ast.unparse(call.func) == "datetime.now" for call in calls) == 2
    sleeps = [call for call in calls if ast.unparse(call.func) == "time.sleep"]
    assert len(sleeps) == 1
    assert ast.unparse(sleeps[0].args[0]) == "min(30.0, max(0.001, (expiry - checked_at).total_seconds()))"
    assert any(isinstance(node, ast.Compare) and ast.unparse(node.left) == "checked_at"
               and len(node.ops) == 1 and isinstance(node.ops[0], ast.Gt)
               and ast.unparse(node.comparators[0]) == "expiry" for node in ast.walk(helper))
    assert not any(ast.unparse(call.func).startswith("patch.") for call in calls)
    assignments = {target.id: node for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    retained_bytes = assignments["retained_initial_receipt_bytes"]
    assert ast.unparse(retained_bytes.value) == "retained_state_bytes[initial_command['resource_environment_receipt']]"
    contract = assignments["retained_expiry_contract"].value
    assert ast.unparse(contract) == "protocol._candidate_execution_contract(protocol._candidate_protocol(config, retained_protocol), retained_protocol)"
    wait = assignments["retained_expiry"]
    assert ast.unparse(wait.value) == "wait_for_retained_resource_expiry(retained_initial_receipt_bytes, retained_expiry_contract['maximum_resource_receipt_age_seconds'])"
    stale_try = next(node for node in tree.body if isinstance(node, ast.Try)
                     and any(isinstance(call, ast.Call)
                             and ast.unparse(call.func) == "protocol.require_candidate_execution_authority"
                             for call in ast.walk(node)))
    authority = next(call for call in ast.walk(stale_try) if isinstance(call, ast.Call)
                     and ast.unparse(call.func) == "protocol.require_candidate_execution_authority")
    assert "now" not in {keyword.arg for keyword in authority.keywords}
    assert any(keyword.arg == "validate_current_runtime" and ast.literal_eval(keyword.value) is False
               for keyword in authority.keywords)
    assert any(isinstance(node, ast.Constant) and node.value == "candidate_resource_receipt_stale"
               for node in ast.walk(stale_try))
    worker = next(call for call in ast.walk(tree) if isinstance(call, ast.Call)
                  and ast.unparse(call.func) == "worker.run_attempt")
    assert retained_bytes.lineno < wait.lineno < stale_try.lineno < worker.lineno
    byte_checks = [node for node in tree.body if isinstance(node, ast.Assert)
                   and ast.unparse(node.test) == "(root / initial_command['resource_environment_receipt']).read_bytes() == retained_initial_receipt_bytes"]
    assert len(byte_checks) == 2
    assert byte_checks[0].lineno < wait.lineno < byte_checks[1].lineno < stale_try.lineno
