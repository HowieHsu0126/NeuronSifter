from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from Libs.tests.test_operational_twin_successor_execution_handoff import (
    _EVALUATOR_PRESEED_METADATA_FIXTURE,
    _PRECOMPUTE_IDENTITY_FIXTURE,
    _SCHEDULER_ACCOUNTING_FIXTURE,
    _WORKER_PRESEED_RETAINED_FIXTURE,
    _RECOVERY_READY_PRODUCER_FIXTURE,
    _copy_reviewed_handoff,
    _isolated_python,
    _scheduler_receipt_guarded_source,
    config as config,
    isolated_execution_ready as isolated_execution_ready,
    isolated_reviewed_handoff as isolated_reviewed_handoff,
)

from Libs.utils.evaluation import operational_twin_candidate_execution as execution
from Libs.utils.evaluation import operational_twin_candidate_evidence as evidence
from Libs.utils.evaluation import operational_twin_candidate_retry_authority as retry_authority
from Libs.utils.evaluation.operational_twin_training_protocol import (
    CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V1,
    CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V2,
    CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V3,
    OperationalTwinProtocolError,
    _require_candidate_resource_evidence,
    _require_live_candidate_resource_placement,
    build_candidate_assignment_manifest,
    build_candidate_lineage_manifest,
    build_candidate_preflight_authority,
    candidate_execution_artifact_paths,
    candidate_fresh_initialization_receipt,
    candidate_runtime_slurm_identity,
    load_candidate_protocol_artifacts,
    read_repo_json,
    require_candidate_execution_authority,
)
from Libs.utils.exp_utils import load_config_stack
from Libs.exp.evaluation import evaluate_operational_twin_development as evaluator
from Libs.exp.training import run_operational_twin_candidate_attempt as attempt_runner
from Libs.exp.training.train_operational_twin_devcal import (
    EXPECTED_ACCESS_POLICY,
    EXPECTED_CLAIM_BOUNDARY,
)
from Libs.scripts.builders.build_operational_twin_development_training_commands import (
    build_candidate_commands,
)
from Libs.scripts.builders import (
    submit_operational_twin_candidate_execution as submitter,
)
import ast
from Libs.utils.evaluation.operational_twin_training_protocol import compute_candidate_calibration_authority_uid


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "Libs/configs/config_stack.yaml"
RESOURCE_FIXTURE_ROOT = ROOT / "Libs/tests/fixtures/operational_twin_resource_receipts"


def _config() -> dict[str, object]:
    return load_config_stack(CONFIG_PATH)


def _initial_context(config: dict[str, object]) -> dict[str, object]:
    candidate, assignments, commands, preflight = _candidate_artifacts_for_test(config)
    assignment = dict(assignments["assignments"][0])
    attempt = dict(assignment["attempts"][0])
    command = next(
        dict(row)
        for row in commands["commands"]
        if row["attempt_uid"] == attempt["attempt_uid"]
    )
    return {
        "execution": dict(config["operational_twin_candidate_execution"]),
        "command": command,
        "assignment": assignment,
        "attempt": attempt,
        "candidate_manifest": candidate,
        "assignment_manifest": assignments,
        "command_catalog": commands,
        "preflight": preflight,
    }


def _candidate_artifacts_for_test(
    config: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    protocol = config["operational_twin_candidate_lineage_protocol"]
    candidate_path = ROOT / protocol["output_paths"]["candidate_lineage_manifest"]
    if candidate_path.is_file():
        return load_candidate_protocol_artifacts(config, ROOT)
    candidate = build_candidate_lineage_manifest(config)
    assignments = build_candidate_assignment_manifest(config, candidate)
    commands = build_candidate_commands(config, candidate, assignments)
    preflight = build_candidate_preflight_authority(
        config,
        ROOT,
        candidate,
        assignments,
        commands,
    )
    return candidate, assignments, commands, preflight


def _cpu_slurm_identity() -> dict[str, object]:
    return {
        "slurm_job_id": "900001",
        "slurm_array_job_id": "900001",
        "slurm_array_task_id": "0",
        "slurm_job_name": "NrnSift_candidate_persistence",
        "slurm_job_partition": "cpu",
        "slurm_job_nodelist": "cpu04",
        "slurm_cpus_per_task": "2",
        "slurm_mem_per_node": "8192",
        "slurm_mem_per_cpu": None,
        "slurm_job_gres": None,
        "hostname": "cpu04",
        "slurm_job_gpus": None,
        "slurm_gpus_on_node": None,
        "slurm_step_gpus": None,
        "resource_query_job_id": "900001_0",
    }


def _cpu_environment() -> dict[str, object]:
    return {
        "hostname": "cpu04",
        "python_version": "3.11.0",
        "torch_version": "2.7.0",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_visible_devices": None,
    }


def _scheduler_capture(argv: list[str], root: Path) -> str:
    del root
    if argv[0] == "sinfo":
        return "cpu04|cpu|idle|(null)\n"
    if argv[0] == "scontrol":
        return (
            "JobId=900001 ArrayJobId=900001 ArrayTaskId=0 "
            "Partition=cpu NodeList=cpu04 CPUs/Task=2 "
            "TRES=cpu=2,node=1,billing=2 MinMemoryNode=8G "
            "OverSubscribe=OK\n"
        )
    raise AssertionError(argv)


def _resource_fixture(name: str) -> dict[str, object]:
    return json.loads((RESOURCE_FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _receipt_for_resource_fixture(
    fixture: dict[str, object],
    *,
    schema_version: str = CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V2,
) -> dict[str, object]:
    stdout = str(fixture["resource_stdout"])
    return {
        "schema_version": schema_version,
        "resource_command": fixture["resource_command"],
        "resource_stdout": stdout,
        "resource_stdout_sha256": execution.sha256_text(stdout),
        "resource_allocation": fixture["resource_requirements"],
        "slurm": fixture["slurm"],
    }


@pytest.mark.parametrize(
    "fixture_name",
    ["cpu04_job_545144.json", "gpu03_job_545150.json"],
)
def test_slurm_2205_retained_resource_fixtures_validate(
    fixture_name: str,
) -> None:
    fixture = _resource_fixture(fixture_name)
    receipt = _receipt_for_resource_fixture(fixture)
    _require_candidate_resource_evidence(
        receipt,
        fixture["resource_requirements"],
        ["scontrol", "--oneliner", "show", "job", "{slurm_job_id}"],
    )
    _require_live_candidate_resource_placement(
        fixture["resource_requirements"],
        fixture["slurm"],
        fixture["environment"],
    )


def test_round44_slurm_shape_is_not_accepted_as_v1() -> None:
    fixture = _resource_fixture("cpu04_job_545144.json")
    receipt = _receipt_for_resource_fixture(
        fixture,
        schema_version=CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V1,
    )
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_resource_receipt_resource_mismatch",
    ):
        _require_candidate_resource_evidence(
            receipt,
            fixture["resource_requirements"],
            ["scontrol", "--oneliner", "show", "job", "{slurm_job_id}"],
        )


@pytest.mark.parametrize(
    ("fixture_name", "old", "new", "error_code"),
    [
        (
            "cpu04_job_545144.json",
            " TRES=cpu=2,node=1,billing=2 ",
            " TRES=cpu=2,node=1,billing=2 TRES=cpu=2,node=1,billing=2 ",
            "candidate_resource_receipt_resource_invalid",
        ),
        (
            "cpu04_job_545144.json",
            "TRES=cpu=2,node=1,billing=2",
            "TRES=cpu=2,cpu=2,node=1,billing=2",
            "candidate_resource_receipt_resource_invalid",
        ),
        (
            "cpu04_job_545144.json",
            "MinMemoryNode=8G",
            "MinMemoryNode=8GB",
            "candidate_resource_receipt_resource_invalid",
        ),
        (
            "cpu04_job_545144.json",
            "TRES=cpu=2,node=1,billing=2",
            "TRES=cpu=3,node=1,billing=2",
            "candidate_resource_receipt_resource_mismatch",
        ),
        (
            "cpu04_job_545144.json",
            "OverSubscribe=OK",
            "OverSubscribe=NO",
            "candidate_resource_receipt_exclusivity_mismatch",
        ),
        (
            "cpu04_job_545144.json",
            "OverSubscribe=OK",
            "OverSubscribe=OK TresPerNode=gres:gpu:1",
            "candidate_resource_receipt_gres_mismatch",
        ),
        (
            "gpu03_job_545150.json",
            "TresPerNode=gres:gpu:1",
            "TresPerNode=gres:gpu:2",
            "candidate_resource_receipt_gres_mismatch",
        ),
        (
            "cpu04_job_545144.json",
            "OverSubscribe=OK",
            "OverSubscribe=OK ReqTRES=cpu=2,mem=8G",
            "candidate_resource_receipt_legacy_field_forbidden",
        ),
    ],
)
def test_slurm_2205_resource_parser_rejects_ambiguous_or_drifted_evidence(
    fixture_name: str,
    old: str,
    new: str,
    error_code: str,
) -> None:
    fixture = _resource_fixture(fixture_name)
    receipt = _receipt_for_resource_fixture(fixture)
    assert old in receipt["resource_stdout"]
    receipt["resource_stdout"] = str(receipt["resource_stdout"]).replace(
        old,
        new,
        1,
    )
    receipt["resource_stdout_sha256"] = execution.sha256_text(
        str(receipt["resource_stdout"])
    )
    with pytest.raises(OperationalTwinProtocolError, match=error_code):
        _require_candidate_resource_evidence(
            receipt,
            fixture["resource_requirements"],
            ["scontrol", "--oneliner", "show", "job", "{slurm_job_id}"],
        )


def test_runtime_slurm_identity_derives_normal_and_array_queries() -> None:
    array_values = _cpu_slurm_identity()
    declared_array_query = array_values.pop("resource_query_job_id")
    array_values["slurm_job_id"] = "900002"
    array_values["slurm_array_task_id"] = "7"
    array_identity = candidate_runtime_slurm_identity(values=array_values)
    assert array_identity["resource_query_job_id"] == "900001_7"
    assert array_identity["slurm_job_id"] == "900002"
    assert declared_array_query == "900001_0"

    normal_values = deepcopy(array_values)
    normal_values["slurm_job_id"] = "900003"
    normal_values["slurm_array_job_id"] = None
    normal_values["slurm_array_task_id"] = None
    normal_identity = candidate_runtime_slurm_identity(values=normal_values)
    assert normal_identity["resource_query_job_id"] == "900003"


@pytest.mark.parametrize(
    "mutation",
    [
        {"slurm_job_id": None},
        {"slurm_array_task_id": None},
        {"slurm_array_job_id": None},
        {"slurm_array_job_id": "array"},
        {"slurm_array_task_id": "-1"},
        {"slurm_job_id": "900001_0x"},
        {"resource_query_job_id": "900001_9"},
    ],
)
def test_runtime_slurm_identity_rejects_missing_partial_or_malformed_values(
    mutation: dict[str, object],
) -> None:
    values = _cpu_slurm_identity()
    values.update(mutation)
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_runtime_slurm_identity_invalid",
    ):
        candidate_runtime_slurm_identity(values=values)


def test_real_array_resource_fixture_uses_composite_query_identity() -> None:
    fixture = _resource_fixture("slurm_array_element_resource_receipt.json")
    receipt = _receipt_for_resource_fixture(
        fixture,
        schema_version=CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V3,
    )
    _require_candidate_resource_evidence(
        receipt,
        fixture["resource_requirements"],
        ["scontrol", "--oneliner", "show", "job", "{resource_query_job_id}"],
    )
    assert receipt["slurm"]["resource_query_job_id"] == "547789_0"


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (
            lambda receipt: receipt["resource_command"].__setitem__(-1, "547789"),
            "candidate_resource_receipt_resource_invalid",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "resource_stdout",
                str(receipt["resource_stdout"])
                + str(receipt["resource_stdout"]),
            ),
            "candidate_resource_receipt_resource_invalid",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "resource_stdout",
                str(receipt["resource_stdout"]).replace(
                    "ArrayTaskId=0", "ArrayTaskId=1", 1
                ),
            ),
            "candidate_resource_receipt_resource_mismatch",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "resource_stdout",
                str(receipt["resource_stdout"]).replace(
                    "JobId=547790", "JobId=547791", 1
                ),
            ),
            "candidate_resource_receipt_resource_mismatch",
        ),
    ],
)
def test_array_resource_fixture_rejects_master_multi_record_or_identity_drift(
    mutation: object,
    error_code: str,
) -> None:
    fixture = _resource_fixture("slurm_array_element_resource_receipt.json")
    receipt = _receipt_for_resource_fixture(
        fixture,
        schema_version=CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V3,
    )
    mutation(receipt)
    receipt["resource_stdout_sha256"] = execution.sha256_text(
        str(receipt["resource_stdout"])
    )
    with pytest.raises(OperationalTwinProtocolError, match=error_code):
        _require_candidate_resource_evidence(
            receipt,
            fixture["resource_requirements"],
            [
                "scontrol",
                "--oneliner",
                "show",
                "job",
                "{resource_query_job_id}",
            ],
        )


def test_real_array_terminal_fixture_preserves_external_and_internal_ids() -> None:
    fixture = _resource_fixture("slurm_array_terminal_accounting_receipt.json")
    assert evidence.validate_scheduler_accounting_fixture(fixture) == []
    rows = evidence.parse_sacct_rows(str(fixture["stdout"]))
    selected = evidence._array_terminal_rows(
        rows,
        job_id=str(fixture["source_job_id"]),
        command_indices=fixture["command_indices"],
    )
    assert selected[0]["job_id"] == "547789_0"
    assert selected[0]["job_id_raw"] == "547790"
    assert selected[1]["job_id"] == "547789_1"
    assert selected[1]["job_id_raw"] == "547789"


def test_array_master_row_cannot_satisfy_element_terminal() -> None:
    rows = evidence.parse_sacct_rows(
        "547789|547789|NrnSift_exec_iface_probe|cpu|COMPLETED|0:0|cpu04|"
        "billing=2,cpu=2,node=1|00:00:00\n"
    )
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_scheduler_terminal_missing",
    ):
        evidence._array_terminal_rows(
            rows,
            job_id="547789",
            command_indices=[0],
        )


@pytest.mark.parametrize(
    ("stdout_mutation", "error_code"),
    [
        (
            lambda stdout: stdout
            + "547789_0|999999|duplicate|cpu|COMPLETED|0:0|cpu04|cpu=2|00:00:00\n",
            "candidate_scheduler_terminal_missing",
        ),
        (
            lambda stdout: stdout.replace(
                "547789_1|547789|",
                "547789_1|547790|",
                1,
            ),
            "candidate_scheduler_internal_identity_reused",
        ),
        (
            lambda stdout: "\n".join(
                line
                for line in stdout.splitlines()
                if not line.startswith("547789_1|")
            )
            + "\n",
            "candidate_scheduler_terminal_missing",
        ),
        (
            lambda stdout: stdout.replace(
                "547789_1|547789|NrnSift_exec_iface_probe|cpu|COMPLETED|",
                "547789_1|547789|NrnSift_exec_iface_probe|cpu|RUNNING|",
                1,
            ),
            "candidate_scheduler_not_terminal",
        ),
        (
            lambda stdout: stdout.replace(
                "547789_1|547789|NrnSift_exec_iface_probe|cpu|COMPLETED|0:0|cpu04|billing=2,cpu=2,node=1|00:00:00",
                "547789_1|547789|NrnSift_exec_iface_probe|cpu|COMPLETED|0:0|cpu04|00:00:00",
                1,
            ),
            "candidate_scheduler_receipt_parse_error",
        ),
    ],
)
def test_array_terminal_mapping_rejects_duplicate_reused_missing_or_malformed_rows(
    stdout_mutation: object,
    error_code: str,
) -> None:
    fixture = _resource_fixture("slurm_array_terminal_accounting_receipt.json")
    with pytest.raises(OperationalTwinProtocolError, match=error_code):
        rows = evidence.parse_sacct_rows(stdout_mutation(str(fixture["stdout"])))
        evidence._array_terminal_rows(
            rows,
            job_id=str(fixture["source_job_id"]),
            command_indices=fixture["command_indices"],
        )


def test_scheduler_receipt_reload_reparses_immutable_stdout(
    isolated_execution_ready,
    tmp_path: Path,
) -> None:
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    source_commit = isolated_execution_ready["prepared"]["source_commit"]
    result = _isolated_python(root, _scheduler_receipt_guarded_source(r"""
import json
from pathlib import Path

import pytest

from Libs.utils.exp_utils import load_config_stack
from Libs.utils.evaluation import operational_twin_candidate_evidence as evidence
from Libs.utils.evaluation.operational_twin_candidate_execution import resolve_candidate_execution_protocol
from Libs.utils.evaluation.operational_twin_training_protocol import OperationalTwinProtocolError
from Libs.scripts.builders.submit_operational_twin_candidate_execution import (
    submit_arrays, require_candidate_dispatch_manifest,
    require_candidate_release_manifest, require_candidate_release_intent,
)

root = Path.cwd()
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
resolved = resolve_candidate_execution_protocol(config, root)
scheduler_calls = []
submitted_ids = []

def scheduler_boundary(argv, current_root):
    assert current_root == root
    scheduler_calls.append(list(argv))
    if argv[0] == "sinfo":
        return "cpu04|cpu*|mix|(null)\ngpu03|gpu1|mix|gpu:4"
    if argv[0] == "sbatch":
        assert "--hold" in argv
        job_id = str(865000 + len(submitted_ids))
        submitted_ids.append(job_id)
        return job_id
    assert argv[:2] == ["scontrol", "release"]
    dispatch = json.loads((root / resolved.execution["output_paths"]["dispatch_manifest"]).read_text())
    require_candidate_dispatch_manifest(config, root, dispatch, mode="initial", resolved_context=resolved)
    assert len(dispatch["initial_arrays"]) == 5
    assert dispatch["selected_indices"] == list(range(15))
    array = next(row for row in dispatch["initial_arrays"] if row["job_id"] == argv[2])
    require_candidate_release_intent(config, root, resolved, dispatch, array)
    return ""

dispatch = submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)
require_candidate_dispatch_manifest(config, root, dispatch, mode="initial", resolved_context=resolved)
release = require_candidate_release_manifest(config, root, resolved, dispatch)
assert release["status"] == "RELEASED"
arrays = dispatch["initial_arrays"]
assert len(arrays) == len(release["attempts"]) == len(set(submitted_ids)) == 5
assert dispatch["selected_indices"] == list(range(15))
assert dispatch["source_commit"] == expected_source_commit
assert len(scheduler_calls) == 11
accounting_calls = []

def accounting_boundary(argv, current_root):
    # Synthetic terminal accounting is confined to this non-authoritative fixture.
    assert current_root == root
    assert argv == ["sacct", "-j", argv[2], f"--format={evidence.SCHEDULER_SACCT_FORMAT}", "-n", "-P"]
    accounting_calls.append(argv[2])
    array = next(row for row in arrays if row["job_id"] == argv[2])
    group = next(row for row in resolved.execution["array_groups"]
                 if row["initial_command_indices"] == array["command_indices"])
    tres = [f"cpu={group['cpus_per_task']}", f"mem={group['memory']}", "node=1"]
    if group["gres"] is not None:
        tres.append("gres/gpu=" + str(group["gres"]).rsplit(":", 1)[1])
    return "\n".join("|".join([
        f"{array['job_id']}_{index}", str(875000 + int(index)), array["job_name"],
        group["partition"], "COMPLETED", "0:0", group["nodelist"], ",".join(tres), "00:00:00",
    ]) for index in array["command_indices"]) + "\n"

captured = evidence.collect_scheduler_receipts(
    root, resolved.execution, arrays, phase="initial",
    source_commit=dispatch["source_commit"], command_capture=accounting_boundary,
    resolved_context=resolved,
)
loaded = evidence.load_scheduler_receipts(
    root, resolved.execution, arrays, phase="initial",
    source_commit=dispatch["source_commit"], resolved_context=resolved,
)
assert loaded == captured
assert sorted(captured) == list(range(15))
assert accounting_calls == [array["job_id"] for array in arrays]
assert len({row["terminal"]["job_id_raw"] for row in captured.values()}) == 15
receipt_path = root / resolved.execution["output_paths"]["scheduler_receipts_root"] / f"initial_{arrays[0]['job_id']}.json"
original_bytes = receipt_path.read_bytes()
original_receipt = json.loads(original_bytes)

cached_rows_tamper = json.loads(original_bytes)
cached_rows_tamper["terminal_rows"][0]["state"] = "RUNNING"
assert cached_rows_tamper["stdout"] == original_receipt["stdout"]
assert cached_rows_tamper["stdout_sha256"] == original_receipt["stdout_sha256"]
receipt_path.write_text(json.dumps(cached_rows_tamper), encoding="utf-8")
try:
    with pytest.raises(OperationalTwinProtocolError, match="candidate_scheduler_receipt_invalid: .*terminal rows differ from raw stdout"):
        evidence.load_scheduler_receipts(
            root, resolved.execution, arrays, phase="initial",
            source_commit=dispatch["source_commit"], resolved_context=resolved,
        )
finally:
    receipt_path.write_bytes(original_bytes)
assert evidence.load_scheduler_receipts(
    root, resolved.execution, arrays, phase="initial",
    source_commit=dispatch["source_commit"], resolved_context=resolved,
) == captured

stdout_tamper = json.loads(original_bytes)
stdout_tamper["stdout"] = stdout_tamper["stdout"].replace("COMPLETED", "RUNNING", 1)
assert stdout_tamper["stdout_sha256"] == original_receipt["stdout_sha256"]
receipt_path.write_text(json.dumps(stdout_tamper), encoding="utf-8")
try:
    with pytest.raises(OperationalTwinProtocolError, match="candidate_scheduler_receipt_invalid"):
        evidence.load_scheduler_receipts(
            root, resolved.execution, arrays, phase="initial",
            source_commit=dispatch["source_commit"], resolved_context=resolved,
        )
finally:
    receipt_path.write_bytes(original_bytes)
assert receipt_path.read_bytes() == original_bytes
assert all(not (root / attempt["output_root"]).exists()
           for assignment in resolved.assignment_manifest["assignments"] for attempt in assignment["attempts"])
result = {"accounting_scope": "non_authoritative_fixture", "arrays": len(arrays),
                  "indices": sorted(captured), "scheduler_boundary_calls": len(scheduler_calls),
                  "accounting_boundary_calls": len(accounting_calls), "released_arrays": len(release["attempts"]),
                  "cached_rows_rejected": True, "stale_stdout_digest_rejected": True,
                  "production_calls": 0, "attempt_roots": 0}
""", source_commit=source_commit, round_id=isolated_execution_ready["prepared"]["round_id"]), stage="scheduler receipt reload")
    guard = result.pop("guard")
    assert guard["negative_controls"] == ["opaque_receipt", "extra_input", "mutable_head_input_git"]
    assert guard["nested_authentications"] >= 1
    assert guard["nested_input_paths"] == 3
    assert guard["physical_input_reads"] == guard["transport_events"] == guard["unexpected_denials"] == guard["scientific_calls"] == 0
    assert guard["trace_restored"] is True
    assert result == {
        "accounting_scope": "non_authoritative_fixture", "arrays": 5,
        "indices": list(range(15)), "scheduler_boundary_calls": 11,
        "accounting_boundary_calls": 5, "released_arrays": 5,
        "cached_rows_rejected": True, "stale_stdout_digest_rejected": True,
        "production_calls": 0, "attempt_roots": 0,
    }


def _copy_review(root: Path, config: dict[str, object]) -> None:
    protocol = dict(config["operational_twin_candidate_lineage_protocol"])
    contract = dict(protocol["execution_authority_contract"])
    relative = Path(str(contract["approved_protocol_review_path"]))
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": contract["required_review_schema_version"],
                "round_id": contract["required_review_round_id"],
                "direction_id": contract["required_review_direction_id"],
                "decision": contract["required_review_decision"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


_CANDIDATE_GATE_FIXTURE = r"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from Libs.scripts.builders import build_operational_twin_calibration_execution_authority as calibration_builder
from Libs.exp.training import run_operational_twin_candidate_attempt as attempt_runner

resolved = runtime.resolve_candidate_execution_protocol(config, root)

def prepare_gate():
    # Only scheduler responses are fixtures; dispatch, review, source, resource
    # and attempt validators run unchanged in the isolated reviewed repository.
    calls, submissions = [], []
    def scheduler_boundary(argv, current_root):
        assert current_root == root
        calls.append(list(argv))
        if argv[0] == "sinfo":
            return "cpu04|cpu*|mix|(null)\ngpu03|gpu1|mix|gpu:4"
        if argv[0] == "sbatch":
            assert "--hold" in argv
            submissions.append(list(argv))
            return str(920000 + len(submissions))
        assert argv[:2] == ["scontrol", "release"]
        return ""
    dispatch = submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)
    require_candidate_dispatch_manifest(config, root, dispatch, mode="initial", resolved_context=resolved)
    release = require_candidate_release_manifest(config, root, resolved, dispatch)
    assert len(submissions) == 5 and len(calls) == 11 and release["status"] == "RELEASED"
    context = runtime.resolve_candidate_attempt_context(config, root, 0)
    command = context["command"]
    paths = protocol.candidate_execution_artifact_paths(resolved.protocol, context["assignment"], context["attempt"])
    # The existing guard receives only these two frozen gate destinations.
    _ce_metadata.update(paths.values())
    fixture = json.loads((root / "Libs/tests/fixtures/operational_twin_resource_receipts/slurm_array_element_resource_receipt.json").read_text())
    assert hashlib.sha256(fixture["resource_stdout"].encode()).hexdigest() == fixture["resource_stdout_sha256"]
    assert hashlib.sha256(fixture["sinfo_stdout"].encode()).hexdigest() == fixture["sinfo_stdout_sha256"]
    array = next(row for row in dispatch["initial_arrays"] if 0 in row["command_indices"])
    slurm = dict(fixture["slurm"])
    slurm.update(slurm_job_id="930000", slurm_array_job_id=array["job_id"], slurm_array_task_id="0", slurm_job_name=array["job_name"])
    slurm.pop("resource_query_job_id", None)
    slurm = protocol.candidate_runtime_slurm_identity(values=slurm)
    raw = fixture["resource_stdout"]
    for key, value in (("JobId", slurm["slurm_job_id"]), ("ArrayJobId", array["job_id"]), ("ArrayTaskId", "0"), ("JobName", array["job_name"])):
        raw, count = re.subn(r"(?<!\S)" + re.escape(key) + r"=\S+", key + "=" + value, raw)
        assert count == 1
    resource_calls = []
    def capture(argv, current_root):
        assert current_root == root
        resource_calls.append(list(argv))
        if argv[0] == "sinfo":
            assert argv == fixture["sinfo_command"]
            return fixture["sinfo_stdout"]
        assert argv == ["scontrol", "--oneliner", "show", "job", slurm["resource_query_job_id"]]
        return raw
    return SimpleNamespace(context=context, command=command, paths=paths,
        receipt_path=root / paths["resource_environment_receipt"],
        authority_path=root / paths["execution_authority"],
        attempt_root=root / command["output_root"], slurm=slurm,
        environment=fixture["environment"], capture=capture, resource_calls=resource_calls,
        timestamp=datetime.now(timezone.utc))

def publish_gate(gate, *, capture=None):
    return runtime.build_and_publish_candidate_execution_gate(
        config, root, task_index=0, command_capture=gate.capture if capture is None else capture,
        captured_at=gate.timestamp, slurm=gate.slurm, environment=gate.environment)

def validate_gate(gate):
    context = gate.context
    return protocol.require_candidate_execution_authority(
        config, root, assignment=context["assignment"], attempt=context["attempt"],
        candidate_manifest=context["candidate_manifest"], assignment_manifest=context["assignment_manifest"],
        command_catalog=context["command_catalog"], preflight=context["preflight"],
        execution_authority_path=gate.authority_path, resource_environment_receipt_path=gate.receipt_path,
        cuda_available=gate.environment["cuda_available"], cuda_device_count=gate.environment["cuda_device_count"],
        torch_version=gate.environment["torch_version"], current_source_commit=expected_source_commit,
        now=gate.timestamp, validate_current_runtime=False, resolved_context=resolved)
"""



_CANDIDATE_RUNTIME_METADATA_FIXTURE = r"""
from contextlib import contextmanager
import os, platform, socket, sys, torch
from Libs.exp.training import train_operational_twin_devcal as trainer
from Libs.exp.training import run_operational_twin_assignment_command as command_runner
from Libs.exp.evaluation import evaluate_operational_twin_development as evaluator
from Libs.utils.evaluation import operational_twin_candidate_evidence as evidence
from Libs.scripts.builders.submit_operational_twin_candidate_execution import worker_binding

def prepare_control_gate(index=0):
    gate = prepare_gate()
    if index:
        context = runtime.resolve_candidate_attempt_context(config, root, index)
        assert context["assignment"]["resource_profile"] == "gpu_single"
        dispatch = json.loads((root / resolved.execution["output_paths"]["dispatch_manifest"]).read_text())
        array = next(row for row in dispatch["initial_arrays"] if index in row["command_indices"])
        fixture = json.loads((root / "Libs/tests/fixtures/operational_twin_resource_receipts/gpu03_job_545150.json").read_text())
        slurm = dict(fixture["slurm"])
        slurm.update(slurm_job_id=str(930000 + index), slurm_array_job_id=array["job_id"],
                     slurm_array_task_id=str(index), slurm_job_name=array["job_name"])
        slurm = protocol.candidate_runtime_slurm_identity(values=slurm)
        raw = fixture["resource_stdout"]
        for key, value in (("JobId", slurm["slurm_job_id"]), ("JobName", array["job_name"])):
            raw, count = re.subn(r"(?<!\S)" + re.escape(key) + r"=\S+", key + "=" + value, raw)
            assert count == 1
        raw = raw.replace("JobId=" + slurm["slurm_job_id"],
            "JobId=" + slurm["slurm_job_id"] + " ArrayJobId=" + array["job_id"] + " ArrayTaskId=" + str(index), 1)
        paths = protocol.candidate_execution_artifact_paths(resolved.protocol, context["assignment"], context["attempt"])
        _ce_metadata.update(paths.values())
        def capture(argv, current_root):
            assert current_root == root
            if argv[0] == "sinfo":
                assert argv == fixture["sinfo_command"]
                return fixture["sinfo_stdout"]
            assert argv == ["scontrol", "--oneliner", "show", "job", slurm["resource_query_job_id"]]
            return raw
        gate = SimpleNamespace(context=context, command=context["command"], paths=paths,
            receipt_path=root / paths["resource_environment_receipt"], authority_path=root / paths["execution_authority"],
            attempt_root=root / context["attempt"]["output_root"], slurm=slurm,
            environment=dict(fixture["environment"]), capture=capture, timestamp=datetime.now(timezone.utc))
    gate.environment = {**gate.environment, "python_version": platform.python_version(), "torch_version": str(torch.__version__)}
    _ce_metadata.update(gate.command["output_root"] + "/" + name for name in (
        "attempt_ownership.json", "terminal_status.json", "provenance/runtime_provenance.json",
        "evaluation/summary.json", attempt_runner.PHASE_RECEIPT_NAME))
    return gate

@contextmanager
def control_runtime(gate):
    updates = worker_binding(root, resolved, expected_source_commit)
    updates.update({key.upper(): value for key, value in gate.slurm.items() if key.startswith("slurm_")})
    updates["CUDA_VISIBLE_DEVICES"] = gate.environment.get("cuda_visible_devices")
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        with patch.object(socket, "gethostname", return_value=gate.environment["hostname"]), \
             patch.object(torch.cuda, "is_available", return_value=gate.environment["cuda_available"]), \
             patch.object(torch.cuda, "device_count", return_value=gate.environment["cuda_device_count"]):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def publish_control_gate(gate):
    return runtime.build_and_publish_candidate_execution_gate(config, root,
        task_index=gate.command["command_index"], command_capture=gate.capture,
        captured_at=gate.timestamp, slurm=gate.slurm, environment=gate.environment)

def control_args(gate):
    argv = gate.command["argv"]
    with patch.object(sys, "argv", [argv[1], *argv[2:]]):
        return trainer.parse_args()

def materialize_control_attempt(gate, *, calibration=None):
    published = publish_control_gate(gate)
    bindings = validate_gate(gate)
    assignment, attempt = gate.context["assignment"], gate.context["attempt"]
    output = runtime.claim_candidate_attempt(config, root, resolved, attempt["attempt_uid"],
        bindings, claimed_at=gate.timestamp)
    identity = protocol.candidate_identity_payload(assignment, attempt)
    initialization = protocol.candidate_fresh_initialization_receipt(resolved.protocol, resolved)
    device = {**gate.environment, "resource_profile": assignment["resource_profile"],
        "device_type": "cuda" if assignment["resource_profile"] == "gpu_single" else "cpu",
        "slurm_partition": gate.slurm["slurm_job_partition"], "slurm_nodelist": gate.slurm["slurm_job_nodelist"],
        "slurm_job_gpus": gate.slurm["slurm_job_gpus"], "slurm_gpus_on_node": gate.slurm["slurm_gpus_on_node"],
        "slurm_step_gpus": gate.slurm["slurm_step_gpus"]}
    terminal = trainer._bind_candidate_runtime_payload({
        "schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
        "status": "COMPLETED", "terminal_reason": "no_fit_metrics_completed",
        "method_id": assignment["method_id"], "seed": assignment["seed"],
        "started_at": gate.timestamp.isoformat(), "finished_at": gate.timestamp.isoformat(),
        "budget": assignment["budget"], "epochs_completed": 0, "selected_checkpoint_epoch": None,
        "steps_completed": 0, "trainable_parameter_count": 0, "checkpoint_created": False,
        "calibration_fitting_executed": False, "resource_profile": assignment["resource_profile"],
        "scheduler_identity": gate.slurm, "device_provenance": device,
        "calibration_execution_authority": calibration,
        "access_flags": trainer.EXPECTED_ACCESS_POLICY, "claim_flags": trainer.EXPECTED_CLAIM_BOUNDARY},
        identity, initialization, bindings, attempt["output_root"])
    # These are explicit unit-test completion metadata, with no model or metrics.
    trainer._write_attempt_status(output, terminal, candidate_lineage=True)
    runtime.write_json_exclusive(output / "provenance/runtime_provenance.json", terminal)
    runtime.write_json_exclusive(output / "evaluation/summary.json", {
        "status": "PASS", **identity, "method_id": assignment["method_id"], "seed": assignment["seed"],
        "checkpoint_reselection_performed": False})
    phase = attempt_runner._phase_receipt(context=gate.context, gate=published,
        started_at=gate.timestamp.isoformat(), phases=[
            {"phase": "training_and_calibration", "argv": list(gate.command["argv"]), "returncode": 0},
            {"phase": "open_development_calibration_evaluation", "argv": attempt_runner._evaluation_argv(gate.command), "returncode": 0}], status="PASS",
        terminal_decision="OPERATIONAL_TWIN_CANDIDATE_ATTEMPT_EXECUTED", error=None)
    runtime.write_json_exclusive(output / attempt_runner.PHASE_RECEIPT_NAME, phase)
    runtime.require_candidate_attempt_ownership(config, root, resolved, attempt["attempt_uid"])
    return output, terminal, bindings
"""

def _run_candidate_gate_fixture(isolated_execution_ready, tmp_path: Path, source: str, *, stage: str, test_call_witness=None):
    """Reuse the reviewed isolated repository and unchanged CE read guard."""
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    prepared = isolated_execution_ready["prepared"]
    result = _isolated_python(root, _scheduler_receipt_guarded_source(
        _CANDIDATE_GATE_FIXTURE + source, source_commit=prepared["source_commit"],
        round_id=prepared["round_id"]), stage=stage, test_call_witness=test_call_witness)
    guard = result.pop("guard")
    assert guard["negative_controls"] == ["opaque_receipt", "extra_input", "mutable_head_input_git"]
    assert guard["nested_authentications"] >= 1 and guard["nested_input_paths"] == 3
    assert guard["physical_input_reads"] == guard["transport_events"] == guard["unexpected_denials"] == guard["scientific_calls"] == 0
    assert guard["trace_restored"] is True
    return result


def test_execution_config_matches_frozen_candidate_lineage(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
resolved = runtime.resolve_candidate_execution_protocol(config, root)
assert runtime.validate_candidate_execution_config(config, root, resolved_context=resolved) == []
assert execution["schema_version"] == runtime.EXECUTION_CONFIG_SCHEMA_VERSION
assert execution["interface_repair_contract"] == config[runtime.EXECUTION_SCHEMA_KEY]["required_interface_repair_contract"]
assert execution["terminal_scheduler_contract"] == config[runtime.EXECUTION_SCHEMA_KEY]["required_terminal_scheduler_contract"]
assert resolved.protocol["candidate_lineage_uid"] == execution["candidate_lineage_uid"]
assert len(resolved.assignment_manifest["assignments"]) == 15
assert len(resolved.command_catalog["commands"]) == 30
result = {"config_valid": True, "assignments": 15, "slots": 30}
""", stage="reviewed successor execution config")
    assert result == {"config_valid": True, "assignments": 15, "slots": 30}


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [("initial_index", "candidate_execution_initial_index_coverage_mismatch"),
     ("review_direction", "candidate_execution_review_contract_mismatch"),
     ("access_counter", "candidate_execution_access_boundary_invalid")],
    ids=["initial_index", "review_direction", "access_counter"],
)
def test_execution_config_rejects_contract_drift(isolated_execution_ready, tmp_path: Path, mutation: str, expected_code: str) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path,
        "mutation = " + repr(mutation) + "\nexpected_code = " + repr(expected_code) + "\n" + r"""
changed = deepcopy(config)
value = changed["operational_twin_candidate_execution"]
if mutation == "initial_index":
    value["array_groups"][0]["initial_command_indices"][0] = 1
elif mutation == "review_direction":
    value["protocol_review_contract"]["direction_id"] = "wrong-direction"
else:
    assert mutation == "access_counter"
    value["access_counters"]["sealed_material_access"] = 1
codes = {row["code"] for row in runtime.validate_candidate_execution_config(changed, root, resolved_context=resolved)}
assert expected_code in codes, codes
assert runtime.validate_candidate_execution_config(config, root, resolved_context=resolved) == []
result = {"mutation": mutation, "expected_code": expected_code, "canonical_config_preserved": True}
""", stage="execution config contract drift")
    assert result == {"mutation": mutation, "expected_code": expected_code, "canonical_config_preserved": True}


def test_attempt_gate_payloads_validate_and_are_no_overwrite(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
gate = prepare_gate()
published = publish_gate(gate)
receipt = json.loads(gate.receipt_path.read_text())
authority = json.loads(gate.authority_path.read_text())
assert set(receipt) == protocol.candidate_execution_binding_keys(protocol.CANDIDATE_RESOURCE_RECEIPT_KEYS, resolved)
assert set(authority) == protocol.candidate_execution_binding_keys(protocol.CANDIDATE_EXECUTION_AUTHORITY_KEYS, resolved)
assert receipt["receipt_uid"] == protocol.compute_candidate_resource_receipt_uid(receipt)
assert authority["authority_uid"] == protocol.compute_candidate_execution_authority_uid(authority)
assert published["attempt_uid"] == gate.context["attempt"]["attempt_uid"]
validated = validate_gate(gate)
assert validated["execution_authority"]["authority_uid"] == authority["authority_uid"]
original = gate.receipt_path.read_bytes()
with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_execution_artifact_exists"):
    runtime.write_json_exclusive(gate.receipt_path, receipt)
assert gate.receipt_path.read_bytes() == original and not gate.attempt_root.exists()
result = {"gate_valid": True, "no_overwrite": True, "attempt_roots": 0}
""", stage="successor gate payload and no overwrite")
    assert result == {"gate_valid": True, "no_overwrite": True, "attempt_roots": 0}


@pytest.mark.parametrize("review_state", ["missing", "rejected"])
def test_attempt_gate_review_failure_leaves_no_candidate_artifacts(isolated_execution_ready, tmp_path: Path, review_state: str) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path,
        "review_state = " + repr(review_state) + "\n" + r"""
gate = prepare_gate()
review_path = root / contract["review_contract"]["path"]
original = review_path.read_bytes()
try:
    if review_state == "missing":
        review_path.unlink()
        error_type, message = FileNotFoundError, re.escape(str(review_path))
    else:
        review = json.loads(original)
        review["decision"] = "rejected"
        review_path.write_text(json.dumps(review))
        error_type, message = protocol.OperationalTwinProtocolError, "handoff_committed_bytes_drift"
    with pytest.raises(error_type, match=message):
        publish_gate(gate)
    assert gate.resource_calls == []
    assert not any(path.exists() for path in (gate.receipt_path, gate.authority_path, gate.attempt_root))
finally:
    review_path.write_bytes(original)
assert review_path.read_bytes() == original
result = {"review_state": review_state, "resource_calls": 0, "candidate_artifacts": 0, "review_restored": True}
""", stage="gate review rejection before resource capture")
    assert result == {"review_state": review_state, "resource_calls": 0, "candidate_artifacts": 0, "review_restored": True}


def test_attempt_gate_retains_published_artifacts_after_validation_failure(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
gate = prepare_gate()
real_fsync = os.fsync
flushed_paths, corrupted_uids = [], []
def corrupt_flushed_authority(descriptor):
    real_fsync(descriptor)
    spelling = os.readlink("/proc/self/fd/" + str(descriptor))
    if spelling not in {str(gate.receipt_path), str(gate.authority_path)}:
        return
    flushed_paths.append(Path(spelling))
    if spelling != str(gate.authority_path):
        return
    identity = os.fstat(descriptor)
    current = gate.authority_path.lstat()
    assert stat.S_ISREG(current.st_mode) and current.st_nlink == 1
    assert (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)
    original = gate.authority_path.read_bytes()
    value = json.loads(original)
    uid = value["authority_uid"]
    assert isinstance(uid, str) and uid
    corrupted_uid = uid[:-1] + ("0" if uid[-1] != "0" else "1")
    token = json.dumps(uid).encode()
    replacement = json.dumps(corrupted_uid).encode()
    field = b'"authority_uid": ' + token
    assert original.count(field) == 1 and len(token) == len(replacement)
    offset = original.index(field) + len(b'"authority_uid": ')
    assert original[offset:offset + len(token)] == token
    assert os.pwrite(descriptor, replacement, offset) == len(replacement)
    real_fsync(descriptor)
    assert os.readlink("/proc/self/fd/" + str(descriptor)) == spelling
    current = gate.authority_path.lstat()
    assert (current.st_dev, current.st_ino, current.st_size) == (identity.st_dev, identity.st_ino, len(original))
    assert gate.authority_path.read_bytes() == original[:offset] + replacement + original[offset + len(token):]
    corrupted_uids.append(corrupted_uid)
# Corrupt flushed bytes through the exact live authority descriptor. The genuine
# writer completes its inode checks before the genuine authority validator fails.
with patch.object(os, "fsync", corrupt_flushed_authority):
    with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_execution_authority_uid_mismatch"):
        publish_gate(gate)
assert flushed_paths == [gate.receipt_path, gate.authority_path]
assert len(corrupted_uids) == 1
preserved = {path: path.read_bytes() for path in flushed_paths}
assert json.loads(preserved[gate.authority_path])["authority_uid"] == corrupted_uids[0]
assert not gate.attempt_root.exists()
with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_execution_artifact_exists"):
    publish_gate(gate)
assert all(path.read_bytes() == data for path, data in preserved.items())
result = {"published_gates_retained": 2, "failed_validation": True, "reuse_rejected": True, "attempt_roots": 0}
""", stage="retained gate evidence after validation failure")
    assert result == {"published_gates_retained": 2, "failed_validation": True, "reuse_rejected": True, "attempt_roots": 0}


def test_attempt_gate_retains_receipt_after_authority_publication_failure(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
gate = prepare_gate()
real_fsync = os.fsync
failed_descriptors = []
def fail_authority_flush(descriptor):
    if os.readlink("/proc/self/fd/" + str(descriptor)) == str(gate.authority_path):
        failed_descriptors.append(descriptor)
        raise OSError("non-authoritative authority publication fault")
    return real_fsync(descriptor)
# The unchanged descriptor-owned writer removes only its incomplete leaf.
with patch.object(os, "fsync", fail_authority_flush):
    with pytest.raises(OSError, match="authority publication fault"):
        publish_gate(gate)
assert len(failed_descriptors) == 1
assert gate.receipt_path.is_file() and not gate.authority_path.exists()
assert not gate.attempt_root.exists()
receipt = json.loads(gate.receipt_path.read_text())
assert receipt["receipt_uid"] == protocol.compute_candidate_resource_receipt_uid(receipt)
result = {"published_receipt_retained": True, "incomplete_authority_removed": True, "attempt_roots": 0}
""", stage="incomplete authority publication cleanup")
    assert result == {"published_receipt_retained": True, "incomplete_authority_removed": True, "attempt_roots": 0}


def test_attempt_gate_does_not_remove_preexisting_authority(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
gate = prepare_gate()
gate.authority_path.parent.mkdir(parents=True, exist_ok=True)
gate.authority_path.write_text("pre-existing-authority\n")
with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_attempt_partial_gate"):
    publish_gate(gate)
assert gate.authority_path.read_text() == "pre-existing-authority\n"
assert not gate.receipt_path.exists() and not gate.attempt_root.exists()
assert gate.resource_calls == []
result = {"preexisting_authority_preserved": True, "resource_calls": 0, "attempt_roots": 0}
""", stage="preexisting authority ownership rejection")
    assert result == {"preexisting_authority_preserved": True, "resource_calls": 0, "attempt_roots": 0}


def test_attempt_gate_rejects_creation_after_attempt_root_exists(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
gate = prepare_gate()
gate.attempt_root.mkdir(parents=True)
with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_attempt_output_collision"):
    publish_gate(gate)
assert gate.attempt_root.is_dir()
assert not gate.receipt_path.exists() and not gate.authority_path.exists()
assert gate.resource_calls == []
result = {"preexisting_root_preserved": True, "new_gate_artifacts": 0, "resource_calls": 0}
""", stage="occupied attempt root rejection")
    assert result == {"preexisting_root_preserved": True, "new_gate_artifacts": 0, "resource_calls": 0}


def test_attempt_gate_rejects_ambiguous_live_scontrol_before_publication(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
gate = prepare_gate()
def ambiguous_capture(argv, current_root):
    value = gate.capture(argv, current_root)
    if argv[0] == "scontrol":
        tres = re.findall(r"(?<!\S)TRES=\S+", value)
        assert len(tres) == 1
        return value.rstrip() + " " + tres[0] + "\n"
    return value
with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_resource_receipt_resource_invalid"):
    publish_gate(gate, capture=ambiguous_capture)
assert len(gate.resource_calls) == 2
assert not any(path.exists() for path in (gate.receipt_path, gate.authority_path, gate.attempt_root))
result = {"ambiguous_resource_rejected": True, "resource_calls": 2, "candidate_artifacts": 0}
""", stage="ambiguous resource evidence before publication")
    assert result == {"ambiguous_resource_rejected": True, "resource_calls": 2, "candidate_artifacts": 0}


def test_live_resource_probe_validates_without_candidate_side_effects(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
fixture = json.loads((root / "Libs/tests/fixtures/operational_twin_resource_receipts/cpu04_job_545144.json").read_text())
slurm = fixture["slurm"]
updates = {key.upper(): str(value) for key, value in slurm.items() if key.startswith("slurm_") and value is not None}
updates["CUDA_VISIBLE_DEVICES"] = ""
resource_calls = []
def capture(argv, current_root):
    assert current_root == root
    resource_calls.append(list(argv))
    if argv[0] == "sinfo":
        assert argv == fixture["sinfo_command"]
        return fixture["sinfo_stdout"]
    assert argv == fixture["resource_command"]
    return fixture["resource_stdout"]
# Environment/scheduler fixture seams leave the actual source, normalization,
# placement and no-side-effect validators active.
with patch.dict(os.environ, updates):
    for key, value in slurm.items():
        if key.startswith("slurm_") and value is None:
            os.environ.pop(key.upper(), None)
    payload = runtime.build_live_candidate_resource_probe(config, root, resource_profile="cpu_light", command_capture=capture)
assert payload["status"] == "PASS" and payload["side_effect_count"] == 0
assert payload["resource_profile"] == "cpu_light" and payload["source_commit"] == expected_source_commit
assert len(resource_calls) == 2
result = {"status": payload["status"], "side_effect_count": 0, "resource_profile": payload["resource_profile"]}
""", stage="recorded resource probe with genuine source validation")
    assert result == {"status": "PASS", "side_effect_count": 0, "resource_profile": "cpu_light"}


def test_candidate_calibration_authority_binds_execution_source(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
authority_path = root / execution["output_paths"]["calibration_execution_authority"]
original = authority_path.read_bytes()
rebuilt = calibration_builder.build_authority(root / "orchestrator/rounds" / expected_round_id, protocol_scope="candidate-lineage", root=root)
payload = json.loads(original)
assert rebuilt["authority_uid"] == runtime.compute_candidate_calibration_authority_uid(rebuilt)
assert {key: value for key, value in rebuilt.items() if key not in {"created_at", "authority_uid"}} == {key: value for key, value in payload.items() if key not in {"created_at", "authority_uid"}}
schema = config["operational_twin_calibration_execution_authority_schema"]
assert set(payload) == set(schema["candidate_required_fields"])
assert payload["candidate_lineage_uid"] == resolved.protocol["candidate_lineage_uid"]
assert payload["best_development_checkpoint_restore_required"] is True
assert payload["calibration_state_initialization_allowed"] is False
assert payload["training_state_modification_allowed"] is False
assert payload["authority_uid"] == runtime.compute_candidate_calibration_authority_uid(payload)
assert [row["path"] for row in payload["source_files"]] == runtime.candidate_calibration_source_paths(config, resolved, payload["execution_selection"])
kwargs = dict(method_id="full_uncertainty_aware_action_conditioned", candidate_lineage_uid=payload["candidate_lineage_uid"], protocol_uid=payload["protocol_uid"], resolved_context=resolved)
validated = trainer.require_calibration_execution_authority(config, root, authority_path, **kwargs)
assert validated["authority_uid"] == payload["authority_uid"]
checks = []
def reject_payload(value, message, label):
    try:
        authority_path.write_text(value if isinstance(value, str) else json.dumps(value))
        with pytest.raises(protocol.OperationalTwinProtocolError, match=message):
            trainer.require_calibration_execution_authority(config, root, authority_path, **kwargs)
        checks.append(label)
    finally:
        authority_path.write_bytes(original)
reject_payload("{not-json", "calibration_execution_authority_invalid", "invalid_json")
stale = deepcopy(payload)
stale["source_files"][0]["sha256"] = "0" * 64
stale["authority_uid"] = runtime.compute_candidate_calibration_authority_uid(stale)
reject_payload(stale, "candidate_calibration_authority_source_drift", "source_drift")
with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate lineage or protocol identity drifted"):
    trainer.require_calibration_execution_authority(config, root, authority_path, **{**kwargs, "candidate_lineage_uid": "wrong-lineage"})
checks.append("lineage_drift")
review_drift = deepcopy(payload)
review_drift["approved_protocol_review_identity"]["sha256"] = "0" * 64
reject_payload(review_drift, "approved protocol review identity drifted", "review_identity_drift")
unauthorized = deepcopy(payload)
unauthorized["allowed_method_ids"] = []
reject_payload(unauthorized, "method is not authorized", "unauthorized_method")
review_path = root / execution["protocol_review_contract"]["path"]
review_bytes = review_path.read_bytes()
try:
    review_path.unlink()
    with pytest.raises(FileNotFoundError, match="review-record.json"):
        calibration_builder.build_authority(root / "orchestrator/rounds" / expected_round_id, protocol_scope="candidate-lineage", root=root)
    checks.append("missing_review")
finally:
    review_path.write_bytes(review_bytes)
assert authority_path.read_bytes() == original
assert trainer.require_calibration_execution_authority(config, root, authority_path, **kwargs) == validated
result = {"source_bound": True, "negative_checks": checks, "authority_preserved": True}
""", stage="calibration source and review binding")
    assert result == {"source_bound": True, "negative_checks": ["invalid_json", "source_drift", "lineage_drift", "review_identity_drift", "unauthorized_method", "missing_review"], "authority_preserved": True}


def test_candidate_evaluator_context_rejects_identity_drift(isolated_execution_ready, tmp_path: Path) -> None:
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    source = (
        "import os, time\nimport pytest\n"
        "_context_launch_job_id = os.environ['SLURM_JOB_ID']\n"
        "_context_pytest_case = os.environ['PYTEST_CURRENT_TEST']\n"
        "_context_started_ns = time.monotonic_ns()\n"
        + _PRECOMPUTE_IDENTITY_FIXTURE
        + _EVALUATOR_PRESEED_METADATA_FIXTURE
        + r"""
# Reuse the separate H identity-only precompute contract. Its COMPLETED JSON is
# non-authoritative terminal metadata; no model, metric, checkpoint or fit ran.
context_outcomes = []
actual = evaluator._candidate_evaluation_context(args, config, root, output)
assert actual[0]["assignment_uid"] == assignment["assignment_uid"]
assert actual[4] is None
context_outcomes.append("valid_null_calibration")
calibration_path = args.calibration_execution_authority
command_hash = args.command_catalog_hash
provenance_path = output / "provenance/runtime_provenance.json"
original_provenance = provenance_path.read_bytes()
try:
    args.calibration_execution_authority = "./" + calibration_path
    with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_evaluation_calibration_authority_path_mismatch"):
        evaluator._candidate_evaluation_context(args, config, root, output)
    context_outcomes.append("noncanonical_calibration_path_rejected")

    args.calibration_execution_authority = ""
    with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_evaluation_calibration_authority_missing"):
        evaluator._candidate_evaluation_context(args, config, root, output)
    context_outcomes.append("missing_calibration_path_rejected")

    args.calibration_execution_authority = calibration_path
    provenance = json.loads(original_provenance)
    provenance["calibration_execution_authority"] = {"authority_uid": "forbidden"}
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(protocol.OperationalTwinProtocolError, match="calibration-ineligible methods must retain null authority bindings"):
        evaluator._candidate_evaluation_context(args, config, root, output)
    context_outcomes.append("ineligible_calibration_provenance_rejected")
    provenance_path.write_bytes(original_provenance)

    args.command_catalog_hash = "tampered"
    with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_evaluation_identity_mismatch"):
        evaluator._candidate_evaluation_context(args, config, root, output)
    context_outcomes.append("command_catalog_identity_rejected")
finally:
    args.calibration_execution_authority = calibration_path
    args.command_catalog_hash = command_hash
    provenance_path.write_bytes(original_provenance)
assert provenance_path.read_bytes() == original_provenance
assert args.calibration_execution_authority == calibration_path and args.command_catalog_hash == command_hash
assert stops == ["trainer", "evaluator"] and len(guarded_entries) == 2
assert len(observed_inputs) == 6
assert {row["path"] for row in observed_inputs} == set(input_rows) - {receipt_path}
assert not any((output / name).exists() for name in ("checkpoints", "metrics", "evaluation"))
assert _context_pytest_case.endswith(" (call)")
print(json.dumps({
    "event": "candidate_evaluator_identity_context",
    "test_nodeid": _context_pytest_case[:-7], "test_phase": "call",
    "job_id": _context_launch_job_id, "pid": os.getpid(), "root": str(root),
    "start_ns": _context_started_ns, "end_ns": time.monotonic_ns(),
    "context_outcomes": context_outcomes, "preseed_stops": stops,
    "identity_guard_paths": [row["path"] for row in observed_inputs],
    "identity_guard_reads": len(observed_inputs), "guard_entries": len(guarded_entries),
    "authenticated_input_ref_tuples": len(refs) * len(input_rows),
    "receipt_content_guard_reads": 0, "metadata_only_fixture": True,
    "fixture_state_restored": True, "scientific_results": 0, "production_calls": 0,
}))
"""
    )
    result = _isolated_python(root, source, stage="candidate evaluator identity context and drift")
    assert result["event"] == "candidate_evaluator_identity_context"
    assert result["test_nodeid"].endswith("::test_candidate_evaluator_context_rejects_identity_drift")
    assert result["test_phase"] == "call" and result["root"] == str(root)
    assert result["context_outcomes"] == [
        "valid_null_calibration", "noncanonical_calibration_path_rejected",
        "missing_calibration_path_rejected", "ineligible_calibration_provenance_rejected",
        "command_catalog_identity_rejected",
    ]
    assert result["preseed_stops"] == ["trainer", "evaluator"]
    assert result["identity_guard_reads"] == len(result["identity_guard_paths"]) == 6
    assert result["guard_entries"] == 2 and result["authenticated_input_ref_tuples"] == 16
    assert result["receipt_content_guard_reads"] == result["scientific_results"] == result["production_calls"] == 0
    assert result["metadata_only_fixture"] is result["fixture_state_restored"] is True


def test_candidate_evaluator_binds_valid_uq_calibration_authority_and_rejects_drift(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path,
        _CANDIDATE_RUNTIME_METADATA_FIXTURE + r"""
index = next(row["command_index"] for row in resolved.command_catalog["commands"]
    if row["method_id"] == "full_uncertainty_aware_action_conditioned"
    and row["seed"] == 360101 and row["attempt_kind"] == "initial")
gate = prepare_control_gate(index)
with control_runtime(gate):
    calibration_path = root / resolved.execution["output_paths"]["calibration_execution_authority"]
    calibration_binding = trainer.require_calibration_execution_authority(
        config, root, calibration_path, method_id=gate.context["assignment"]["method_id"],
        candidate_lineage_uid=resolved.protocol["candidate_lineage_uid"],
        protocol_uid=resolved.protocol["protocol_uid"], source_commit=expected_source_commit,
        resolved_context=resolved)
    output_root, terminal, bindings = materialize_control_attempt(gate, calibration=calibration_binding)
    argv = attempt_runner._evaluation_argv(gate.command)
    with patch.object(sys, "argv", [argv[2], *argv[3:]]):
        args = evaluator.parse_args()
    actual = evaluator._candidate_evaluation_context(args, config, root, output_root)
    assert actual[4] == calibration_binding
    provenance_path = output_root / "provenance/runtime_provenance.json"
    original = provenance_path.read_bytes()
    provenance = json.loads(original)
    provenance["calibration_execution_authority"] = {**calibration_binding, "sha256": "b" * 64}
    provenance_path.write_text(json.dumps(provenance))
    try:
        with pytest.raises(protocol.OperationalTwinProtocolError,
                           match="candidate_evaluation_calibration_authority_binding_mismatch"):
            evaluator._candidate_evaluation_context(args, config, root, output_root)
    finally:
        provenance_path.write_bytes(original)
    # Preserve the original callee-error propagation control after the real
    # calibration validator has completed. Its arguments and result are observed.
    previous_profile = sys.getprofile()
    injected = []
    calibration_code = trainer.require_calibration_execution_authority.__code__
    def stale_authority(frame, event, value):
        if previous_profile is not None:
            previous_profile(frame, event, value)
        parent = frame.f_back
        if frame.f_code is calibration_code and parent is not None and parent.f_code is evaluator._candidate_evaluation_context.__code__:
            if event == "call":
                assert frame.f_locals["resolved_context"] is parent.f_locals["resolved"]
                assert dict(frame.f_locals["resolved_context"].binding) == dict(resolved.binding)
            if event == "return":
                assert value == calibration_binding
                injected.append("validated_calibration_return")
                raise protocol.OperationalTwinProtocolError("calibration_execution_authority_stale: source hash drifted")
    sys.setprofile(stale_authority)
    try:
        with pytest.raises(protocol.OperationalTwinProtocolError, match="calibration_execution_authority_stale"):
            evaluator._candidate_evaluation_context(args, config, root, output_root)
    finally:
        sys.setprofile(previous_profile)
    assert injected == ["validated_calibration_return"]
    assert provenance_path.read_bytes() == original
    assert not (output_root / "checkpoints").exists()
    assert not (output_root / "metrics").exists()
result = {"valid_uq_binding": True, "provenance_drift_rejected": True,
          "callee_error_propagated": True, "resolved_context_retained": True,
          "model_or_fit_executed": False}
""", stage="candidate original UQ calibration authority context")
    assert result == {"valid_uq_binding": True, "provenance_drift_rejected": True,
                      "callee_error_propagated": True, "resolved_context_retained": True,
                      "model_or_fit_executed": False}


def test_attempt_wrapper_runs_training_then_evaluation_and_writes_receipt(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path,
        _CANDIDATE_RUNTIME_METADATA_FIXTURE + r"""
gate = prepare_control_gate()
training_argv = [sys.executable, "-m", "Libs.exp.training.run_operational_twin_assignment_command",
                 "--protocol-scope", "candidate-lineage", "--task-index", "0"]
evaluation_argv = attempt_runner._evaluation_argv(gate.command)
calls = []
original_run = subprocess.run
def run_command(argv, **kwargs):
    if argv[0] in {"sinfo", "scontrol"}:
        return subprocess.CompletedProcess(argv, 0, stdout=gate.capture(argv, root), stderr="")
    if argv not in (training_argv, evaluation_argv):
        return original_run(argv, **kwargs)
    assert kwargs["cwd"] == root and kwargs["check"] is False
    calls.append(argv)
    output_root = gate.attempt_root
    if len(calls) == 1:
        assert argv == training_argv
        (output_root / "provenance").mkdir(parents=True)
        (output_root / "terminal_status.json").write_text(json.dumps({"status": "COMPLETED"}))
    else:
        assert len(calls) == 2 and argv == evaluation_argv
        (output_root / "evaluation").mkdir(parents=True)
        (output_root / "evaluation/summary.json").write_text(json.dumps({"status": "PASS"}))
    return SimpleNamespace(returncode=0)
with control_runtime(gate), patch.object(subprocess, "run", run_command):
    assert attempt_runner.run_attempt(0) == 0
    gate.timestamp = datetime.now(timezone.utc)
    actual_gate = validate_gate(gate)
    receipt = json.loads((gate.attempt_root / attempt_runner.PHASE_RECEIPT_NAME).read_text())
    assert receipt["status"] == "PASS"
    assert receipt["terminal_decision"] == "OPERATIONAL_TWIN_CANDIDATE_ATTEMPT_EXECUTED"
    assert [row["phase"] for row in receipt["phases"]] == [
        "training_and_calibration", "open_development_calibration_evaluation"]
    assert "Libs.exp.training.run_operational_twin_assignment_command" in calls[0]
    assert "Libs.exp.evaluation.evaluate_operational_twin_development" in calls[1]
    authority_flag = "--calibration-execution-authority"
    assert gate.command["argv"].count(authority_flag) == 1
    assert calls[1].count(authority_flag) == 1
    expected_authority = gate.command["argv"][gate.command["argv"].index(authority_flag) + 1]
    assert calls[1][calls[1].index(authority_flag) + 1] == expected_authority
    assert receipt["successor_execution_binding"] == dict(resolved.binding)
    assert receipt["execution_authority"] == actual_gate["execution_authority"]
    assert receipt["resource_environment_receipt"] == actual_gate["resource_environment_receipt"]
    assert len(calls) == 2
result = {"phase_order": [row["phase"] for row in receipt["phases"]],
          "child_calls": len(calls), "receipt_status": receipt["status"],
          "calibration_argument_preserved": True, "real_execution_gate": True}
""", stage="candidate original attempt wrapper phase order")
    assert result == {"phase_order": ["training_and_calibration", "open_development_calibration_evaluation"],
                      "child_calls": 2, "receipt_status": "PASS",
                      "calibration_argument_preserved": True, "real_execution_gate": True}


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_attempt_wrapper_rejects_missing_or_duplicate_calibration_authority(isolated_execution_ready, tmp_path: Path, mutation: str) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path,
        "mutation = " + repr(mutation) + "\n" + r"""
command = deepcopy(resolved.command_catalog["commands"][0])
argv = command["argv"]
if mutation == "missing":
    argv.remove("--calibration-execution-authority")
else:
    assert mutation == "duplicate"
    argv.extend(["--calibration-execution-authority", "duplicate.json"])
with pytest.raises(protocol.OperationalTwinProtocolError, match="calibration authority must appear exactly once"):
    attempt_runner._evaluation_argv(command)
result = {"mutation": mutation, "rejected": True}
""", stage="evaluator calibration argument cardinality")
    assert result == {"mutation": mutation, "rejected": True}


def test_evaluation_argv_preserves_successor_calibration_authority(isolated_execution_ready, tmp_path: Path) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
command = resolved.command_catalog["commands"][0]
original = deepcopy(command)
argv = attempt_runner._evaluation_argv(command)
flag = "--calibration-execution-authority"
assert command["argv"].count(flag) == argv.count(flag) == 1
assert argv[argv.index(flag) + 1] == command["argv"][command["argv"].index(flag) + 1]
assert argv[argv.index(flag) + 1] == resolved.protocol["output_paths"]["candidate_calibration_execution_authority"]
assert "Libs.exp.evaluation.evaluate_operational_twin_development" in argv
assert command == original
result = {"calibration_forwarded": True, "catalog_command_preserved": True}
""", stage="canonical evaluator calibration forwarding")
    assert result == {"calibration_forwarded": True, "catalog_command_preserved": True}


def test_submitter_builds_exact_nonexclusive_initial_arrays(
    isolated_execution_ready,
    tmp_path: Path,
) -> None:
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    prepared = isolated_execution_ready["prepared"]
    result = _isolated_python(root, _scheduler_receipt_guarded_source(r"""
from Libs.scripts.builders import submit_operational_twin_candidate_execution as submitter

resolved = runtime.resolve_candidate_execution_protocol(config, root)
catalog_path = root / execution["frozen_artifacts"]["assignment_commands"]["path"]
catalog_bytes = catalog_path.read_bytes()
scheduler_calls, submissions = [], []

def scheduler_boundary(argv, current_root):
    assert current_root == root
    scheduler_calls.append(list(argv))
    if argv[0] == "sinfo":
        return "cpu04|cpu*|mix|(null)\ngpu03|gpu1|mix|gpu:4"
    if argv[0] == "sbatch":
        assert "--hold" in argv
        submissions.append(list(argv))
        return str(910000 + len(submissions))
    assert argv[:2] == ["scontrol", "release"]
    dispatch = json.loads((root / execution["output_paths"]["dispatch_manifest"]).read_text())
    submitter.require_candidate_dispatch_manifest(config, root, dispatch, mode="initial", resolved_context=resolved)
    assert len(dispatch["initial_arrays"]) == 5
    array = next(row for row in dispatch["initial_arrays"] if row["job_id"] == argv[2])
    submitter.require_candidate_release_intent(config, root, resolved, dispatch, array)
    return ""

manifest = submitter.submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)
submitter.require_candidate_dispatch_manifest(config, root, manifest, mode="initial", resolved_context=resolved)
release = submitter.require_candidate_release_manifest(config, root, resolved, manifest)
assert manifest["status"] == "INITIAL_ARRAYS_SUBMITTED"
assert manifest["successor_execution_binding"] == dict(resolved.binding)
assert manifest["source_commit"] == expected_source_commit
assert manifest["selected_indices"] == list(range(15))
assert [row["command_indices"] for row in manifest["initial_arrays"]] == [group["initial_command_indices"] for group in execution["array_groups"]] == [list(range(index, index + 3)) for index in range(0, 15, 3)]
assert len(submissions) == len(manifest["initial_arrays"]) == 5
assert len(scheduler_calls) == 11
assert release["status"] == "RELEASED"
assert [row["job_id"] for row in release["attempts"]] == [row["job_id"] for row in manifest["initial_arrays"]]
assert all("--exclusive" not in argv for argv in submissions)
assert sum("--gres=gpu:1" in argv for argv in submissions) == 4
for group, argv in zip(execution["array_groups"], submissions, strict=True):
    assert f"--partition={group['partition']}" in argv
    assert f"--nodelist={group['nodelist']}" in argv
    assert f"--cpus-per-task={group['cpus_per_task']}" in argv
    assert f"--mem={group['memory']}" in argv and f"--time={group['wall_time']}" in argv
    wrapper = argv[argv.index("--wrap") + 1]
    assert ('export CUDA_VISIBLE_DEVICES=""' in wrapper) is (group["gres"] is None)
    assert "NRNSIFT_HANDOFF_UID=" in wrapper and resolved.binding["handoff_uid"] in wrapper
    assert "NRNSIFT_SOURCE_COMMIT=" in wrapper and expected_source_commit in wrapper
assert catalog_path.read_bytes() == catalog_bytes
assert len(resolved.assignment_manifest["assignments"]) == 15
assert len(resolved.command_catalog["commands"]) == 30
assert all(not os.path.lexists(root / attempt["output_root"])
           for assignment in resolved.assignment_manifest["assignments"] for attempt in assignment["attempts"])
result = {"status": manifest["status"], "arrays": len(submissions),
          "selected_indices": manifest["selected_indices"], "scheduler_boundary_calls": len(scheduler_calls),
          "released_arrays": len(release["attempts"]), "frozen_commands_preserved": True,
          "production_calls": 0, "attempt_roots": 0}
""", source_commit=prepared["source_commit"], round_id=prepared["round_id"]),
        stage="reviewed successor initial arrays")
    guard = result.pop("guard")
    assert guard["negative_controls"] == ["opaque_receipt", "extra_input", "mutable_head_input_git"]
    assert guard["nested_authentications"] >= 1 and guard["nested_input_paths"] == 3
    assert guard["physical_input_reads"] == guard["transport_events"] == guard["unexpected_denials"] == guard["scientific_calls"] == 0
    assert guard["trace_restored"] is True
    assert result == {
        "status": "INITIAL_ARRAYS_SUBMITTED", "arrays": 5,
        "selected_indices": list(range(15)), "scheduler_boundary_calls": 11,
        "released_arrays": 5, "frozen_commands_preserved": True,
        "production_calls": 0, "attempt_roots": 0,
    }


def test_recovery_submission_requires_retained_eligibility_with_bound_arrays(
    isolated_execution_ready,
    tmp_path: Path,
) -> None:
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    prepared = isolated_execution_ready["prepared"]
    result = _isolated_python(root, _scheduler_receipt_guarded_source(r"""
from Libs.scripts.builders import submit_operational_twin_candidate_execution as submitter

resolved = runtime.resolve_candidate_execution_protocol(config, root)
binding = submitter.worker_binding(root, resolved, expected_source_commit)
catalog_path = root / execution["frozen_artifacts"]["assignment_commands"]["path"]
catalog_bytes = catalog_path.read_bytes()
preserved_paths = set(contract["frozen_package"]["publications"].values())
preserved_paths.update(contract["output_paths"].values())
preserved_paths.add(contract["review_contract"]["path"])
preserved_paths.update(execution["output_paths"][key]
                       for key in ("calibration_execution_authority", "pre_dispatch_audit"))
preserved = {path: (root / path).read_bytes() for path in preserved_paths}
indices = []
for group in execution["array_groups"]:
    selected = group["recovery_command_indices"]
    argv = submitter.build_array_submission_argv(
        root, execution, group, mode="recovery", indices=selected, binding=binding)
    assert "--hold" in argv and "--exclusive" not in argv
    assert f"--array={selected[0]}-{selected[-1]}" in argv
    assert f"--partition={group['partition']}" in argv
    assert f"--nodelist={group['nodelist']}" in argv
    assert f"--cpus-per-task={group['cpus_per_task']}" in argv
    assert f"--mem={group['memory']}" in argv and f"--time={group['wall_time']}" in argv
    assert ("--gres=gpu:1" in argv) is (group["gres"] is not None)
    wrapper = argv[argv.index("--wrap") + 1]
    assert binding["NRNSIFT_HANDOFF_UID"] in wrapper
    assert binding["NRNSIFT_SOURCE_COMMIT"] in wrapper
    for index in selected:
        command = resolved.command_catalog["commands"][index]
        assert command["conditional"] is True and command["method_id"] == group["method_id"]
    indices.extend(selected)
assert indices == list(range(15, 30))
assert len(resolved.assignment_manifest["assignments"]) == 15
assert len(resolved.command_catalog["commands"]) == 30
eligibility_path = root / execution["output_paths"]["recovery_eligibility"]
assert not os.path.lexists(eligibility_path)
scheduler_calls = []
def scheduler_boundary(argv, current_root):
    scheduler_calls.append(list(argv))
    raise AssertionError("recovery without retained eligibility reached the scheduler")
noncanonical_path = root / execution["output_paths"]["pre_dispatch_audit"]
assert noncanonical_path != eligibility_path
assert noncanonical_path.read_bytes() == preserved[execution["output_paths"]["pre_dispatch_audit"]]
# The maintained preparation reaches this path check only after genuine
# resolver, clean-source, calibration-authority and runtime-ownership checks.
with pytest.raises(protocol.OperationalTwinProtocolError,
                   match="^candidate_recovery_path_not_canonical$") as rejection:
    submitter.submit_arrays(config, root, mode="recovery",
                            recovery_eligibility_path=noncanonical_path,
                            command_capture=scheduler_boundary)
assert str(rejection.value) == "candidate_recovery_path_not_canonical"
assert scheduler_calls == []
assert not os.path.lexists(eligibility_path)
# Absence at the canonical path remains a separate retained-eligibility control.
with pytest.raises(FileNotFoundError, match=re.escape(str(eligibility_path))):
    submitter.submit_arrays(config, root, mode="recovery", command_capture=scheduler_boundary)
assert scheduler_calls == []
assert catalog_path.read_bytes() == catalog_bytes
assert {path: (root / path).read_bytes() for path in preserved} == preserved
for key in ("dispatch_claim", "recovery_dispatch_claim", "dispatch_manifest", "recovery_dispatch_manifest",
            "recovery_eligibility", "dispatch_release_manifest", "recovery_dispatch_release_manifest"):
    assert not os.path.lexists(root / execution["output_paths"][key])
assert not os.path.lexists(root / execution["output_paths"]["execution_root"])
assert all(not os.path.lexists(root / attempt["output_root"])
           for assignment in resolved.assignment_manifest["assignments"] for attempt in assignment["attempts"])
result = {"recovery_indices": indices, "bound_arrays": len(execution["array_groups"]),
          "retained_eligibility_required": True, "scheduler_boundary_calls": len(scheduler_calls),
          "noncanonical_recovery_path_rejection": str(rejection.value), "preserved_metadata": True,
          "frozen_commands_preserved": True, "production_calls": 0, "attempt_roots": 0}
""", source_commit=prepared["source_commit"], round_id=prepared["round_id"]),
        stage="bound recovery arrays and retained eligibility")
    guard = result.pop("guard")
    assert guard["negative_controls"] == ["opaque_receipt", "extra_input", "mutable_head_input_git"]
    assert guard["nested_authentications"] >= 1 and guard["nested_input_paths"] == 3
    assert guard["physical_input_reads"] == guard["transport_events"] == guard["unexpected_denials"] == guard["scientific_calls"] == 0
    assert guard["trace_restored"] is True
    assert result == {
        "recovery_indices": list(range(15, 30)), "bound_arrays": 5,
        "retained_eligibility_required": True, "scheduler_boundary_calls": 0,
        "noncanonical_recovery_path_rejection": "candidate_recovery_path_not_canonical",
        "preserved_metadata": True,
        "frozen_commands_preserved": True, "production_calls": 0, "attempt_roots": 0,
    }


def test_successor_submitter_requires_committed_handoff_before_scheduler(
    isolated_execution_ready,
    tmp_path: Path,
) -> None:
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    prepared = isolated_execution_ready["prepared"]
    result = _isolated_python(root, _scheduler_receipt_guarded_source(r"""
from Libs.scripts.builders import submit_operational_twin_candidate_execution as submitter

resolved = runtime.resolve_candidate_execution_protocol(config, root)
identity = config["operational_twin_successor_lineage_protocol"]["identity"]
assert resolved.protocol["candidate_lineage_uid"] == execution["candidate_lineage_uid"] == identity["candidate_lineage_uid"]
assert resolved.protocol["protocol_uid"] == execution["protocol_uid"] == identity["protocol_uid"]
assignments = resolved.assignment_manifest["assignments"]
commands = resolved.command_catalog["commands"]
assert len(assignments) == 15
assert [row["command_index"] for row in commands] == list(range(30))
assert len({row["method_id"] for row in assignments}) == 5
assert len({row["seed"] for row in assignments}) == 3
assert all(len(row["attempts"]) == 2 for row in assignments)
assert resolved.handoff["candidate_dispatch_authorized"] is False
assert resolved.handoff["scientific_twin_result"] == "NOT_RUN"
assert set(resolved.handoff["access_counters"].values()) == {0}
paths = execution["output_paths"]
review_path = root / contract["review_contract"]["path"]
review_bytes = review_path.read_bytes()
preserved_paths = set(contract["frozen_package"]["publications"].values())
preserved_paths.update(contract["output_paths"].values())
preserved_paths.update(paths[key] for key in ("calibration_execution_authority", "pre_dispatch_audit"))
preserved = {path: (root / path).read_bytes() for path in preserved_paths}
scheduler_calls = []

class BeforeSchedulerDispatch(RuntimeError):
    pass

def scheduler_boundary(argv, current_root):
    assert current_root == root
    assert argv == ["sinfo", "--noheader", "--nodes=cpu04,gpu03", "--format=%N|%P|%T|%G"]
    scheduler_calls.append(list(argv))
    raise BeforeSchedulerDispatch("stop before external scheduler and dispatch publication")

# The maintained submitter must pass the real resolver, clean-source,
# calibration, ownership, pre-dispatch and all-slot absence checks first.
with pytest.raises(BeforeSchedulerDispatch):
    submitter.submit_arrays(config, root, mode="initial", command_capture=scheduler_boundary)
assert len(scheduler_calls) == 1
rejected = []
for review_state in ("missing", "uncommitted"):
    try:
        if review_state == "missing":
            review_path.unlink()
            expected_error = FileNotFoundError
            expected_message = str(review_path)
        else:
            uncommitted = json.loads(review_bytes)
            assert uncommitted["decision"] == "approved"
            uncommitted["evidence_summary"] += " Uncommitted non-authoritative review replacement."
            review_path.write_text(json.dumps(uncommitted, indent=2, sort_keys=True) + "\n")
            assert review_path.read_bytes() != review_bytes
            expected_error = protocol.OperationalTwinProtocolError
            expected_message = "handoff_committed_bytes_drift"
        for mode in ("initial", "recovery"):
            with pytest.raises(expected_error, match=re.escape(expected_message)):
                submitter.submit_arrays(config, root, mode=mode, command_capture=scheduler_boundary)
            rejected.append([review_state, mode])
            assert len(scheduler_calls) == 1
    finally:
        review_path.write_bytes(review_bytes)
assert review_path.read_bytes() == review_bytes
assert {path: (root / path).read_bytes() for path in preserved} == preserved
for key in ("dispatch_claim", "recovery_dispatch_claim", "dispatch_manifest", "recovery_dispatch_manifest"):
    assert not os.path.lexists(root / paths[key])
assert not os.path.lexists(root / paths["execution_root"])
assert all(not os.path.lexists(root / attempt["output_root"])
           for assignment in assignments for attempt in assignment["attempts"])
result = {"status": "SUCCESSOR_ROUTE_REQUIRES_COMMITTED_HANDOFF",
          "assignments": len(assignments), "slots": len(commands),
          "methods": 5, "seeds": 3, "scheduler_boundary_calls": len(scheduler_calls),
          "review_rejections": rejected, "review_bytes_restored": True,
          "preserved_metadata": True, "production_calls": 0, "attempt_roots": 0}
""", source_commit=prepared["source_commit"], round_id=prepared["round_id"]),
        stage="successor route and committed review boundary")
    guard = result.pop("guard")
    assert guard["negative_controls"] == ["opaque_receipt", "extra_input", "mutable_head_input_git"]
    assert guard["nested_authentications"] >= 1
    assert guard["nested_input_paths"] == 3
    assert guard["physical_input_reads"] == guard["transport_events"] == guard["unexpected_denials"] == guard["scientific_calls"] == 0
    assert guard["trace_restored"] is True
    assert result == {
        "status": "SUCCESSOR_ROUTE_REQUIRES_COMMITTED_HANDOFF",
        "assignments": 15, "slots": 30, "methods": 5, "seeds": 3,
        "scheduler_boundary_calls": 1,
        "review_rejections": [["missing", "initial"], ["missing", "recovery"],
                              ["uncommitted", "initial"], ["uncommitted", "recovery"]],
        "review_bytes_restored": True, "preserved_metadata": True,
        "production_calls": 0, "attempt_roots": 0,
    }


def _published_cpu_gate(
    tmp_path: Path,
    config: dict[str, object],
    context: dict[str, object],
) -> tuple[datetime, dict[str, object]]:
    _copy_review(tmp_path, config)
    timestamp = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)
    receipt = execution.build_candidate_resource_receipt(
        config,
        context=context,
        source_commit="candidate-evidence-source",
        captured_at=timestamp,
        sinfo_stdout=_scheduler_capture(["sinfo"], tmp_path),
        resource_stdout=_scheduler_capture(["scontrol"], tmp_path),
        slurm=_cpu_slurm_identity(),
        environment=_cpu_environment(),
    )
    protocol = config["operational_twin_candidate_lineage_protocol"]
    paths = candidate_execution_artifact_paths(
        protocol,
        context["assignment"],
        context["attempt"],
    )
    receipt_path = tmp_path / paths["resource_environment_receipt"]
    execution.write_json_exclusive(receipt_path, receipt)
    authority = execution.build_candidate_execution_authority(
        config,
        tmp_path,
        context=context,
        source_commit="candidate-evidence-source",
        issued_at=timestamp,
        receipt_path=receipt_path,
        receipt=receipt,
    )
    authority_path = tmp_path / paths["execution_authority"]
    execution.write_json_exclusive(authority_path, authority)
    bindings = require_candidate_execution_authority(
        config,
        tmp_path,
        assignment=context["assignment"],
        attempt=context["attempt"],
        candidate_manifest=context["candidate_manifest"],
        assignment_manifest=context["assignment_manifest"],
        command_catalog=context["command_catalog"],
        preflight=context["preflight"],
        execution_authority_path=authority_path,
        resource_environment_receipt_path=receipt_path,
        cuda_available=False,
        cuda_device_count=0,
        torch_version="2.7.0",
        current_source_commit="candidate-evidence-source",
        now=timestamp,
        validate_current_runtime=False,
    )
    return timestamp, bindings


def _materialize_completed_no_fit_attempt(
    tmp_path: Path,
    config: dict[str, object],
    context: dict[str, object],
    bindings: dict[str, object],
    timestamp: datetime,
) -> dict[str, object]:
    assignment = context["assignment"]
    attempt = context["attempt"]
    attempt_root = tmp_path / str(attempt["output_root"])
    (attempt_root / "provenance").mkdir(parents=True)
    (attempt_root / "evaluation").mkdir(parents=True)
    initialization = execution.candidate_fresh_initialization_receipt(
        config["operational_twin_candidate_lineage_protocol"]
    )
    device = {
        "resource_profile": "cpu_light",
        "slurm_partition": "cpu",
        "slurm_nodelist": "cpu04",
        "hostname": "cpu04",
        "slurm_job_gpus": None,
        "slurm_gpus_on_node": None,
        "slurm_step_gpus": None,
        "device_type": "cpu",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_visible_devices": None,
        "cuda_device_name": None,
        "python_version": "3.11.0",
        "torch_version": "2.7.0",
    }
    scheduler = _cpu_slurm_identity()
    identity = {
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
    }
    terminal = {
        "schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
        **identity,
        "status": "COMPLETED",
        "terminal_reason": "no_fit_metrics_completed",
        "started_at": timestamp.isoformat(),
        "finished_at": timestamp.isoformat(),
        "budget": assignment["budget"],
        "epochs_completed": 0,
        "selected_checkpoint_epoch": None,
        "steps_completed": 0,
        "trainable_parameter_count": 0,
        "checkpoint_created": False,
        "calibration_fitting_executed": False,
        "resource_profile": "cpu_light",
        "device_provenance": device,
        "scheduler_identity": scheduler,
        "access_flags": EXPECTED_ACCESS_POLICY,
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
        "fresh_initialization_receipt": initialization,
        "state_resume_used": False,
        "candidate_execution_authority": bindings["execution_authority"],
        "candidate_resource_environment_receipt": bindings[
            "resource_environment_receipt"
        ],
    }
    provenance = {
        **identity,
        "scheduler_identity": scheduler,
        "device_provenance": device,
        "candidate_execution_authority": bindings["execution_authority"],
        "candidate_resource_environment_receipt": bindings[
            "resource_environment_receipt"
        ],
        "calibration_execution_authority": None,
    }
    phase = {
        "schema_version": attempt_runner.PHASE_RECEIPT_SCHEMA_VERSION,
        "status": "PASS",
        "terminal_decision": "OPERATIONAL_TWIN_CANDIDATE_ATTEMPT_EXECUTED",
        **identity,
        "attempt_kind": attempt["attempt_kind"],
        "command_index": context["command"]["command_index"],
        "source_commit": "candidate-evidence-source",
        "started_at": timestamp.isoformat(),
        "finished_at": timestamp.isoformat(),
        "phases": [],
        "scheduler_identity": scheduler,
        "execution_authority": bindings["execution_authority"],
        "resource_environment_receipt": bindings["resource_environment_receipt"],
        "terminal_status": None,
        "evaluation_summary": None,
        "error": None,
        "access_counters": config["operational_twin_candidate_execution"][
            "access_counters"
        ],
        "closed_flags": config["operational_twin_candidate_execution"]["closed_flags"],
    }
    evaluation_summary = {
        "status": "PASS",
        **identity,
        "checkpoint_reselection_performed": False,
    }
    (attempt_root / "terminal_status.json").write_text(
        json.dumps(terminal), encoding="utf-8"
    )
    (attempt_root / "provenance/runtime_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    (attempt_root / attempt_runner.PHASE_RECEIPT_NAME).write_text(
        json.dumps(phase), encoding="utf-8"
    )
    (attempt_root / "evaluation/summary.json").write_text(
        json.dumps(evaluation_summary), encoding="utf-8"
    )
    return {
        "terminal": {
            "job_id": "900001_0",
            "job_id_raw": "900001",
            "job_name": "NrnSift_candidate_persistence",
            "partition": "cpu",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "node_list": "cpu04",
            "alloc_tres": "billing=2,cpu=2,mem=8G,node=1",
            "elapsed": "00:00:10",
        },
        "receipt": {"path": "scheduler.json", "sha256": "a" * 64, "size_bytes": 1},
    }


def test_materialized_no_fit_attempt_detects_state_reuse_and_tampered_authority(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path,
        _CANDIDATE_RUNTIME_METADATA_FIXTURE + r"""
gate = prepare_control_gate()
with control_runtime(gate):
    output, terminal, bindings = materialize_control_attempt(gate)
    context = gate.context
    scheduler = {"terminal": {
        "job_id": gate.slurm["resource_query_job_id"], "job_id_raw": gate.slurm["slurm_job_id"],
        "job_name": gate.slurm["slurm_job_name"], "partition": "cpu", "state": "COMPLETED",
        "exit_code": "0:0", "node_list": "cpu04", "alloc_tres": "billing=2,cpu=2,mem=8G,node=1",
        "elapsed": "00:00:10"},
        "receipt": evidence.artifact_identity(root, root / "Libs/tests/fixtures/operational_twin_resource_receipts/slurm_array_terminal_accounting_receipt.json")}
    def materialized():
        return evidence.validate_materialized_attempt(config, root,
            assignment=context["assignment"], attempt=context["attempt"], command=context["command"],
            scheduler_evidence=scheduler, source_commit=expected_source_commit,
            candidate_manifest=context["candidate_manifest"], assignment_manifest=context["assignment_manifest"],
            command_catalog=context["command_catalog"], preflight=context["preflight"], resolved_context=resolved)
    row, blockers = materialized()
    assert row["status"] == "COMPLETED"
    assert blockers == []
    original_scheduler = deepcopy(scheduler)
    scheduler["terminal"]["job_id"] = gate.slurm["slurm_array_job_id"] + "_1"
    _, blockers = materialized()
    assert "candidate_attempt_scheduler_identity_mismatch" in {blocker["code"] for blocker in blockers}
    scheduler = deepcopy(original_scheduler)
    scheduler["terminal"]["job_id_raw"] = str(int(gate.slurm["slurm_job_id"]) + 100)
    _, blockers = materialized()
    assert "candidate_attempt_scheduler_identity_mismatch" in {blocker["code"] for blocker in blockers}
    scheduler = deepcopy(original_scheduler)
    terminal_path = output / "terminal_status.json"
    original_terminal = terminal_path.read_bytes()
    original_authority = gate.authority_path.read_bytes()
    # Ownership now authenticates retained file identities before the original
    # field predicates. Inject each field change after that real guard returns;
    # the materialized-attempt validator must still detect the changed evidence.
    def mutate_after_ownership(mutation):
        previous_profile = sys.getprofile()
        observed = []
        def observe(frame, event, value):
            if previous_profile is not None:
                previous_profile(frame, event, value)
            parent = frame.f_back
            if (event == "return" and frame.f_code is runtime.require_candidate_attempt_ownership.__code__
                    and parent is not None and parent.f_code is evidence.validate_materialized_attempt.__code__):
                assert value["attempt_uid"] == gate.command["attempt_uid"]
                assert not observed
                mutation()
                observed.append("authenticated_owner")
        sys.setprofile(observe)
        try:
            result = materialized()
        finally:
            sys.setprofile(previous_profile)
        assert observed == ["authenticated_owner"]
        return result
    try:
        def change_resume():
            value = json.loads(original_terminal)
            value["state_resume_used"] = True
            terminal_path.write_text(json.dumps(value))
        _, blockers = mutate_after_ownership(change_resume)
        assert "candidate_attempt_state_reuse_detected" in {blocker["code"] for blocker in blockers}
        terminal_path.write_bytes(original_terminal)
        def change_authority():
            value = json.loads(original_authority)
            value["method_id"] = "tampered"
            gate.authority_path.write_text(json.dumps(value))
        _, blockers = mutate_after_ownership(change_authority)
        assert "candidate_attempt_execution_gate_invalid" in {blocker["code"] for blocker in blockers}
    finally:
        terminal_path.write_bytes(original_terminal)
        gate.authority_path.write_bytes(original_authority)
    assert materialized()[1] == []
    assert not (output / "checkpoints").exists()
    assert not (output / "metrics").exists()
result = {"positive_completed_metadata": True, "scheduler_rejections": 2,
          "state_reuse_rejected": True, "tampered_authority_rejected": True,
          "retained_bytes_restored": True, "model_or_fit_executed": False}
""", stage="candidate original no-fit retained controls")
    assert result == {"positive_completed_metadata": True, "scheduler_rejections": 2,
                      "state_reuse_rejected": True, "tampered_authority_rejected": True,
                      "retained_bytes_restored": True, "model_or_fit_executed": False}


def test_candidate_resource_receipt_expires_after_frozen_age(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
from datetime import timedelta

gate = prepare_gate()
published = publish_gate(gate)
baseline = validate_gate(gate)
preserved = {path: path.read_bytes() for path in (gate.receipt_path, gate.authority_path)}
receipt = json.loads(preserved[gate.receipt_path])
authority = json.loads(preserved[gate.authority_path])
assert baseline["resource_receipt_payload"] == receipt
assert receipt["receipt_uid"] == protocol.compute_candidate_resource_receipt_uid(receipt)
assert authority["authority_uid"] == protocol.compute_candidate_execution_authority_uid(authority)
for key, payload, path, uid_key, binding_keys in (
    ("resource_environment_receipt", receipt, gate.receipt_path, "receipt_uid",
     protocol.CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS),
    ("execution_authority", authority, gate.authority_path, "authority_uid",
     protocol.CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS),
):
    binding = baseline[key]
    assert set(binding) == protocol.candidate_execution_binding_keys(binding_keys, resolved)
    assert binding[uid_key] == payload[uid_key]
    identity = protocol.candidate_runtime_artifact_identity(root, path, resolved)
    assert identity == {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(preserved[path]).hexdigest(),
        "size_bytes": len(preserved[path]),
    }
    assert {field: binding[field] for field in identity} == identity
assert published["attempt_uid"] == gate.context["attempt"]["attempt_uid"]
timestamps = {
    key: protocol._parse_utc_timestamp(receipt[key], label=key)
    for key in ("created_at", "sinfo_captured_at", "resource_captured_at")
}
assert set(timestamps.values()) == {gate.timestamp}
expiry_execution_contract = protocol._candidate_execution_contract(resolved.protocol, resolved)
maximum_age = expiry_execution_contract["maximum_resource_receipt_age_seconds"]
assert type(maximum_age) is int and maximum_age > 0
delta = datetime.resolution
expired_at = timestamps["created_at"] + timedelta(seconds=maximum_age) + delta
assert (expired_at - timestamps["created_at"]).total_seconds() > maximum_age
commands = gate.context["command_catalog"]["commands"]
assert len(commands) == 30 and len({row["output_root"] for row in commands}) == 30
assert all(not (root / row["output_root"]).exists() for row in commands)
context = gate.context
try:
    with pytest.raises(protocol.OperationalTwinProtocolError) as failure:
        protocol.require_candidate_execution_authority(
            config, root, assignment=context["assignment"], attempt=context["attempt"],
            candidate_manifest=context["candidate_manifest"],
            assignment_manifest=context["assignment_manifest"],
            command_catalog=context["command_catalog"], preflight=context["preflight"],
            execution_authority_path=gate.authority_path,
            resource_environment_receipt_path=gate.receipt_path,
            cuda_available=gate.environment["cuda_available"],
            cuda_device_count=gate.environment["cuda_device_count"],
            torch_version=gate.environment["torch_version"],
            current_source_commit=expected_source_commit,
            now=expired_at, validate_current_runtime=False, resolved_context=resolved,
        )
    assert str(failure.value) == (
        "candidate_resource_receipt_stale: receipt and sinfo capture must be fresh at dispatch."
    )
finally:
    assert validate_gate(gate) == baseline
    assert all(path.read_bytes() == value for path, value in preserved.items())
    assert all(not (root / row["output_root"]).exists() for row in commands)
result = {
    "positive_baselines": 2,
    "expired_rejection": str(failure.value),
    "receipt_timestamps": {key: receipt[key] for key in timestamps},
    "maximum_age_seconds": maximum_age,
    "expiry_delta_microseconds": delta // timedelta(microseconds=1),
    "expired_check_at": expired_at.isoformat(),
    "receipt_uid": receipt["receipt_uid"],
    "authority_uid": authority["authority_uid"],
    "source_commit": expected_source_commit,
    "gate_bytes_unchanged": True,
    "declared_attempt_roots": len(commands),
    "attempt_roots": 0,
}
""", stage="successor resource receipt expiry")
    assert set(result) == {
        "positive_baselines", "expired_rejection", "receipt_timestamps",
        "maximum_age_seconds", "expiry_delta_microseconds", "expired_check_at",
        "receipt_uid", "authority_uid", "source_commit", "gate_bytes_unchanged",
        "declared_attempt_roots", "attempt_roots",
    }
    assert result["positive_baselines"] == 2
    assert result["expired_rejection"] == (
        "candidate_resource_receipt_stale: receipt and sinfo capture must be fresh at dispatch."
    )
    assert result["expiry_delta_microseconds"] == 1
    assert result["gate_bytes_unchanged"] is True
    assert result["declared_attempt_roots"] == 30 and result["attempt_roots"] == 0


def test_checkpoint_evidence_enforces_roles_without_requiring_different_bytes(
    tmp_path: Path,
) -> None:
    config = _config()
    context = _initial_context(config)
    assignment = deepcopy(context["assignment_manifest"]["assignments"][3])
    attempt = assignment["attempts"][0]
    attempt_root = tmp_path / str(attempt["output_root"])
    checkpoint_root = attempt_root / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    terminal = {
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "selected_checkpoint_epoch": 2,
        "epochs_completed": 2,
        "steps_completed": 4,
        "calibration_fitting_executed": False,
    }
    checkpoint = {
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "epoch": 2,
        "step": 4,
        "method_protocol": {"budget": assignment["budget"]},
        "trainable_parameter_count": 1,
        "calibration_scale": None,
    }
    torch.save(checkpoint, checkpoint_root / "best_dev.pt")
    shutil.copyfile(checkpoint_root / "best_dev.pt", checkpoint_root / "last.pt")
    result, blockers = evidence._checkpoint_evidence(
        tmp_path,
        assignment,
        attempt_root,
        terminal,
        {"calibration_execution_authority": None},
    )
    assert blockers == []
    assert result["content_equal"] is True
    (checkpoint_root / "last.pt").unlink()
    _, blockers = evidence._checkpoint_evidence(
        tmp_path,
        assignment,
        attempt_root,
        terminal,
        {"calibration_execution_authority": None},
    )
    assert "candidate_checkpoint_roles_missing" in {
        blocker["code"] for blocker in blockers
    }


_INCOMPLETE_CATALOG_FIXTURE = (
    r"""
from collections import Counter
from Libs.utils.evaluation import operational_twin_candidate_evidence as evidence

gate = prepare_gate()
dispatch_path = root / execution["output_paths"]["dispatch_manifest"]
dispatch = json.loads(dispatch_path.read_text())
"""
    + _SCHEDULER_ACCOUNTING_FIXTURE
    + r"""
# The accounting callback supplies synthetic TEST responses, including cancelled
# GPU-looking rows. It neither observes external jobs nor establishes GPU ownership.
scheduler = evidence.collect_scheduler_receipts(
    root, execution, dispatch["initial_arrays"], phase="initial",
    source_commit=dispatch["source_commit"], command_capture=accounting_boundary,
    resolved_context=resolved,
)
loaded = evidence.load_scheduler_receipts(
    root, execution, dispatch["initial_arrays"], phase="initial",
    source_commit=dispatch["source_commit"], resolved_context=resolved,
)
assert loaded == scheduler
assert len(accounting_calls) == len(scheduler_paths) == 5
assert [row["job_id"] for row in dispatch["initial_arrays"]] == [str(920001 + i) for i in range(5)]
attempt_roots = [root / row["output_root"] for row in resolved.command_catalog["commands"]]
assert len(attempt_roots) == len(set(attempt_roots)) == 30
assert all(not path.exists() for path in attempt_roots)
assert not gate.receipt_path.exists() and not gate.authority_path.exists()
assert gate.resource_calls == []
recovery = evidence.build_recovery_eligibility(
    config, root, dispatch_manifest=dispatch, scheduler_by_index=loaded,
    resolved_context=resolved,
)
recovery_path = root / execution["output_paths"]["recovery_eligibility"]
evidence.write_json_exclusive(recovery_path, recovery)
assert evidence.require_recovery_eligibility(
    config, root, recovery_path, resolved_context=resolved,
) == recovery
baseline_catalog = evidence.build_execution_catalog(
    config, root, dispatch_manifest=dispatch, recovery_eligibility=recovery,
    scheduler_by_index=loaded, resolved_context=resolved,
)
baseline_catalog["dispatch_manifest"] = evidence.artifact_identity(root, dispatch_path)
baseline_catalog["recovery_eligibility"] = evidence.artifact_identity(root, recovery_path)
assert baseline_catalog["recovery_dispatch_manifest"] is None
assert not (root / execution["output_paths"]["recovery_dispatch_manifest"]).exists()
initial = [row for row in baseline_catalog["attempt_slots"] if row["attempt_kind"] == "initial"]
recoveries = [row for row in baseline_catalog["attempt_slots"] if row["attempt_kind"] == "infrastructure_recovery_01"]
assert len(initial) == len(recoveries) == len(recovery["assignments"]) == 15
assert all(row["materialized"] is False for row in baseline_catalog["attempt_slots"])
assert all(row[key] is None for row in baseline_catalog["attempt_slots"] for key in ("attempt_ownership", "execution_authority", "resource_environment_receipt", "terminal_status", "phase_receipt", "runtime_provenance", "evaluation_summary", "checkpoint"))
assert all(row["status"] == "FAILED" and [item["code"] for item in row["validation_blockers"]] == ["candidate_attempt_root_missing"] for row in initial)
assert all(row["status"] == "NOT_MATERIALIZED_NOT_ELIGIBLE" and not row["validation_blockers"] for row in recoveries)
assert all(row["recovery_eligibility"] == "BLOCKED_INVALID_PREDECESSOR" for row in recovery["assignments"])
baseline_summary = {
    "scheduler_evidence_kind": "SYNTHETIC_TEST_RESPONSES",
    "scheduler_receipt_count": len(scheduler_paths),
    "scheduler_terminal_states": dict(Counter(row["terminal"]["state"] for row in loaded.values())),
    "assignment_count": baseline_catalog["assignment_count"],
    "attempt_slot_count": baseline_catalog["attempt_slot_count"],
    "unique_attempt_uids": len({row["attempt_uid"] for row in baseline_catalog["attempt_slots"]}),
    "absent_attempt_roots": sum(not path.exists() for path in attempt_roots),
    "recovery_status": recovery["status"],
    "recovery_terminal_decision": recovery["terminal_decision"],
    "recovery_classifications": dict(Counter(row["recovery_eligibility"] for row in recovery["assignments"])),
    "status": baseline_catalog["status"],
    "terminal_decision": baseline_catalog["terminal_decision"],
    "validation_blockers": dict(Counter(row["code"] for row in baseline_catalog["validation_blockers"])),
    "recovery_dispatch_manifest": baseline_catalog["recovery_dispatch_manifest"],
}

def clean_catalog():
    # Validate the authentic incomplete baseline before each independent mutation.
    assert evidence.validate_execution_catalog(config, root, baseline_catalog, resolved_context=resolved) == []
    return deepcopy(baseline_catalog)

def catalog_blockers(catalog):
    return evidence.validate_execution_catalog(config, root, catalog, resolved_context=resolved)
"""
)


def _run_incomplete_catalog_fixture(isolated_execution_ready, tmp_path: Path, source: str, *, stage: str):
    result = _run_candidate_gate_fixture(
        isolated_execution_ready, tmp_path,
        _INCOMPLETE_CATALOG_FIXTURE + source + r"""
assert all(not path.exists() for path in attempt_roots)
assert not gate.receipt_path.exists() and not gate.authority_path.exists()
assert gate.resource_calls == []
assert evidence.validate_execution_catalog(config, root, baseline_catalog, resolved_context=resolved) == []
result["baseline"] = baseline_summary
""", stage=stage,
    )
    assert result["baseline"] == {
        "scheduler_evidence_kind": "SYNTHETIC_TEST_RESPONSES",
        "scheduler_receipt_count": 5,
        "scheduler_terminal_states": {"FAILED": 1, "CANCELLED": 14},
        "assignment_count": 15,
        "attempt_slot_count": 30,
        "unique_attempt_uids": 30,
        "absent_attempt_roots": 30,
        "recovery_status": "PASS",
        "recovery_terminal_decision": "OPERATIONAL_TWIN_CANDIDATE_RECOVERY_ELIGIBILITY_RESOLVED",
        "recovery_classifications": {"BLOCKED_INVALID_PREDECESSOR": 15},
        "status": "FAIL",
        "terminal_decision": "OPERATIONAL_TWIN_CANDIDATE_LINEAGE_EXECUTION_INCOMPLETE",
        "validation_blockers": {"candidate_attempt_root_missing": 15},
        "recovery_dispatch_manifest": None,
    }
    return result


def test_catalog_validator_rejects_missing_and_duplicate_rows(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_incomplete_catalog_fixture(isolated_execution_ready, tmp_path, r"""
missing = clean_catalog()
missing["attempt_slots"].pop()
assert "candidate_execution_slot_count_mismatch" in {row["code"] for row in catalog_blockers(missing)}
duplicate = clean_catalog()
duplicate["attempt_slots"][1]["attempt_uid"] = duplicate["attempt_slots"][0]["attempt_uid"]
assert "candidate_execution_attempt_duplicate" in {row["code"] for row in catalog_blockers(duplicate)}
result = {"checks": ["candidate_execution_slot_count_mismatch", "candidate_execution_attempt_duplicate"]}
""", stage="reviewed incomplete catalog missing and duplicate slots")
    assert result["checks"] == ["candidate_execution_slot_count_mismatch", "candidate_execution_attempt_duplicate"]


@pytest.mark.parametrize(
    ("field", "expected_code"),
    [
        (
            "candidate_lineage_uid",
            "candidate_execution_catalog_candidate_lineage_uid_mismatch",
        ),
        (
            "protocol_uid",
            "candidate_execution_catalog_protocol_uid_mismatch",
        ),
        (
            "preflight_uid",
            "candidate_execution_catalog_preflight_uid_mismatch",
        ),
        (
            "candidate_manifest_hash",
            "candidate_execution_catalog_candidate_manifest_hash_mismatch",
        ),
        (
            "assignment_manifest_hash",
            "candidate_execution_catalog_assignment_manifest_hash_mismatch",
        ),
        (
            "command_catalog_hash",
            "candidate_execution_catalog_command_catalog_hash_mismatch",
        ),
    ],
)
def test_catalog_validator_rejects_frozen_identity_tampering(
    isolated_execution_ready, tmp_path: Path, field: str, expected_code: str,
) -> None:
    result = _run_incomplete_catalog_fixture(isolated_execution_ready, tmp_path,
        "field = " + repr(field) + "\nexpected_code = " + repr(expected_code) + "\n" + r"""
catalog = clean_catalog()
catalog[field] = "tampered"
assert expected_code in {row["code"] for row in catalog_blockers(catalog)}
result = {"checks": [expected_code]}
""", stage="reviewed incomplete catalog frozen identity")
    assert result["checks"] == [expected_code]


def test_catalog_validator_rejects_assignment_and_attempt_mapping_tampering(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_incomplete_catalog_fixture(isolated_execution_ready, tmp_path, r"""
catalog = clean_catalog()
catalog["assignments"][0]["assignment_uid"] = "tampered-assignment"
catalog["attempt_slots"][0]["output_root"] = "tampered-root"
codes = {row["code"] for row in catalog_blockers(catalog)}
assert "candidate_execution_assignment_rows_mismatch" in codes
assert "candidate_execution_attempt_rows_mismatch" in codes
result = {"checks": ["candidate_execution_assignment_rows_mismatch", "candidate_execution_attempt_rows_mismatch"]}
""", stage="reviewed incomplete catalog assignment and attempt mapping")
    assert result["checks"] == ["candidate_execution_assignment_rows_mismatch", "candidate_execution_attempt_rows_mismatch"]


@pytest.mark.parametrize("binding_field", ["dispatch_manifest", "recovery_eligibility"])
def test_catalog_validator_rejects_referenced_artifact_tampering(
    isolated_execution_ready, tmp_path: Path, binding_field: str,
) -> None:
    result = _run_incomplete_catalog_fixture(isolated_execution_ready, tmp_path,
        "binding_field = " + repr(binding_field) + "\n" + r"""
catalog = clean_catalog()
catalog[binding_field]["sha256"] = "0" * 64
expected_code = "candidate_execution_" + binding_field + "_hash_mismatch"
assert expected_code in {row["code"] for row in catalog_blockers(catalog)}
result = {"checks": [expected_code]}
""", stage="reviewed incomplete catalog referenced artifact identity")
    assert result["checks"] == [f"candidate_execution_{binding_field}_hash_mismatch"]


def test_catalog_validator_rejects_positive_terminal_with_retained_blockers(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_incomplete_catalog_fixture(isolated_execution_ready, tmp_path, r"""
catalog = clean_catalog()
retained_blockers = deepcopy(catalog["validation_blockers"])
assert retained_blockers
catalog["status"] = "PASS"
catalog["terminal_decision"] = execution["terminal_policy"]["positive"]
codes = {row["code"] for row in catalog_blockers(catalog)}
assert catalog["validation_blockers"] == retained_blockers
assert "candidate_execution_status_invalid" in codes
assert "candidate_execution_terminal_invalid" in codes
result = {"checks": ["candidate_execution_status_invalid", "candidate_execution_terminal_invalid"]}
""", stage="reviewed incomplete catalog rejects positive terminal")
    assert result["checks"] == ["candidate_execution_status_invalid", "candidate_execution_terminal_invalid"]


@pytest.mark.parametrize(
    "blocker_code",
    [
        "candidate_attempt_phase_receipt_invalid",
        "candidate_attempt_scheduler_identity_mismatch",
        "candidate_attempt_runtime_binding_mismatch",
        "candidate_checkpoint_identity_mismatch",
        "candidate_attempt_evaluation_identity_mismatch",
        "candidate_best_checkpoint_selection_mismatch",
        "candidate_calibration_role_invalid",
    ],
)
def test_catalog_terminal_classifies_integrity_blockers_as_invalid(
    blocker_code: str,
) -> None:
    execution_config = _config()["operational_twin_candidate_execution"]
    terminal = evidence._catalog_terminal_decision(
        execution_config,
        [{"code": blocker_code, "message": "integrity violation"}],
    )
    assert terminal == execution_config["terminal_policy"]["invalid"]


@pytest.mark.parametrize(
    "blocker_code",
    [
        "candidate_attempt_root_missing",
        "candidate_attempt_terminal_evidence_missing",
        "candidate_attempt_runtime_provenance_missing",
        "candidate_attempt_evaluation_missing",
        "candidate_scheduler_terminal_missing",
        "candidate_eligible_recovery_missing",
    ],
)
def test_catalog_terminal_keeps_missing_evidence_incomplete(
    blocker_code: str,
) -> None:
    execution_config = _config()["operational_twin_candidate_execution"]
    terminal = evidence._catalog_terminal_decision(
        execution_config,
        [{"code": blocker_code, "message": "missing evidence"}],
    )
    assert terminal == execution_config["terminal_policy"]["incomplete"]


@pytest.mark.parametrize(
    ("target", "field", "value", "expected_code"),
    [
        (
            "catalog",
            "source_commit",
            "tampered-source",
            "candidate_execution_catalog_source_commit_mismatch",
        ),
        (
            "assignment",
            "method_id",
            "tampered-method",
            "candidate_execution_assignment_rows_mismatch",
        ),
        (
            "assignment",
            "seed",
            -1,
            "candidate_execution_assignment_rows_mismatch",
        ),
        (
            "attempt",
            "attempt_kind",
            "tampered-kind",
            "candidate_execution_attempt_rows_mismatch",
        ),
        (
            "attempt",
            "command_index",
            -1,
            "candidate_execution_attempt_rows_mismatch",
        ),
        (
            "assignment",
            "final_retained_attempt_uid",
            "tampered-final-attempt",
            "candidate_execution_assignment_rows_mismatch",
        ),
    ],
)
def test_catalog_validator_rejects_required_identity_mutations(
    isolated_execution_ready, tmp_path: Path, target: str, field: str,
    value: object, expected_code: str,
) -> None:
    result = _run_incomplete_catalog_fixture(isolated_execution_ready, tmp_path,
        "target = " + repr(target) + "\nfield = " + repr(field)
        + "\nvalue = " + repr(value) + "\nexpected_code = " + repr(expected_code) + "\n" + r"""
catalog = clean_catalog()
if target == "catalog":
    catalog[field] = value
elif target == "assignment":
    catalog["assignments"][0][field] = value
else:
    assert target == "attempt"
    catalog["attempt_slots"][0][field] = value
blockers = catalog_blockers(catalog)
if target == "catalog":
    assert field == "source_commit"
    # Preserve the original parameter/nodeid while recording that its old
    # post-recomputation code is not reached by the genuine scheduler loader.
    assert expected_code == "candidate_execution_catalog_source_commit_mismatch"
    required = {
        "candidate_execution_catalog_recomputation_failed": "Execution catalog could not be recomputed: candidate_scheduler_dispatch_mismatch",
        "candidate_execution_dispatch_manifest_identity_mismatch": "Dispatch manifest source_commit drifted.",
        "candidate_execution_recovery_identity_mismatch": "Recovery eligibility source_commit drifted.",
    }
    actual = {row["code"]: row["message"] for row in blockers if row["code"] in required}
    assert actual == required
    assert expected_code not in {row["code"] for row in blockers}
    result = {"checks": list(required), "source_commit_rejections": actual,
              "old_post_recomputation_branch": "NOT_EXERCISED"}
else:
    assert expected_code in {row["code"] for row in blockers}
    result = {"checks": [expected_code]}
""", stage="reviewed incomplete catalog required identity")
    if target == "catalog":
        assert result["old_post_recomputation_branch"] == "NOT_EXERCISED"
        assert result["checks"] == ["candidate_execution_catalog_recomputation_failed", "candidate_execution_dispatch_manifest_identity_mismatch", "candidate_execution_recovery_identity_mismatch"]
    else:
        assert result["checks"] == [expected_code]


def test_catalog_validator_rejects_embedded_recovery_dispatch_drift(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_incomplete_catalog_fixture(isolated_execution_ready, tmp_path, r"""
catalog = clean_catalog()
original_bytes = recovery_path.read_bytes()
original_identity = deepcopy(catalog["recovery_eligibility"])
try:
    changed = json.loads(original_bytes)
    changed["dispatch_manifest"]["source_commit"] = "tampered-source"
    recovery_path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    catalog["recovery_eligibility"] = evidence.artifact_identity(root, recovery_path)
    assert "candidate_execution_recovery_dispatch_binding_invalid" in {row["code"] for row in catalog_blockers(catalog)}
finally:
    recovery_path.write_bytes(original_bytes)
    catalog["recovery_eligibility"] = original_identity
    assert recovery_path.read_bytes() == original_bytes
    assert evidence.artifact_identity(root, recovery_path) == original_identity
assert catalog_blockers(catalog) == []
result = {"checks": ["candidate_execution_recovery_dispatch_binding_invalid"], "recovery_bytes_and_identity_restored": True}
""", stage="reviewed incomplete catalog embedded recovery dispatch")
    assert result["checks"] == ["candidate_execution_recovery_dispatch_binding_invalid"]
    assert result["recovery_bytes_and_identity_restored"] is True


def test_extra_attempt_root_is_detected(tmp_path: Path) -> None:
    manifest = {
        "candidate_lineage_uid": "lineage",
        "assignments": [
            {
                "assignment_uid": "assignment",
                "attempts": [{"attempt_uid": "declared"}],
            }
        ],
    }
    unexpected = (
        tmp_path
        / "Output/results/operational_twin/candidate_lineages/lineage/assignments/assignment/attempts/undeclared"
    )
    unexpected.mkdir(parents=True)
    assert evidence._extra_attempt_roots(tmp_path, manifest) == [
        unexpected.relative_to(tmp_path).as_posix()
    ]


@pytest.mark.parametrize(
    ("slot", "expected"),
    [
        ({"status": "COMPLETED", "validation_blockers": []}, "NOT_NEEDED"),
        (
            {
                "status": "FAILED",
                "failure_class": "infrastructure_interruption",
                "validation_blockers": [],
            },
            "ELIGIBLE",
        ),
        (
            {
                "status": "FAILED",
                "failure_class": "divergence_non_finite",
                "validation_blockers": [],
            },
            "NOT_ELIGIBLE_NON_INFRASTRUCTURE_FAILURE",
        ),
        (
            {
                "status": "FAILED",
                "failure_class": "infrastructure_interruption",
                "validation_blockers": [{"code": "invalid", "message": "invalid"}],
            },
            "BLOCKED_INVALID_PREDECESSOR",
        ),
    ],
)
def test_recovery_eligibility_is_fail_closed(
    slot: dict[str, object],
    expected: str,
) -> None:
    assert (
        evidence.classify_recovery_eligibility(
            slot,
            infrastructure_failure_class="infrastructure_interruption",
        )
        == expected
    )


def _write_retry_fixture_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_retry_fixture_path(root: Path, relative: str) -> None:
    source = ROOT / relative
    content = (
        source.read_bytes()
        if source.is_file()
        else subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
        )
    )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _git_retry_fixture(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _commit_retry_fixture(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", message],
        cwd=root,
        check=True,
    )
    return _git_retry_fixture(root, "rev-parse", "HEAD")


@pytest.fixture
def _positive_retry_state(tmp_path, isolated_execution_ready):
    """Build owned recovery evidence at the final case root before execution."""
    root = _copy_reviewed_handoff(isolated_execution_ready, tmp_path)
    ready = _isolated_python(root, _PRECOMPUTE_IDENTITY_FIXTURE + _WORKER_PRESEED_RETAINED_FIXTURE
        + _SCHEDULER_ACCOUNTING_FIXTURE + _RECOVERY_READY_PRODUCER_FIXTURE,
        stage="committed disposable recovery-ready fixture")
    assert ready["status"] == "RECOVERY_READY_COMMITTED_STATE_VERIFIED"
    assert ready["eligible_recovery_indices"] == [15]
    assert ready["scientific_results"] == ready["production_calls"] == 0
    return {"root": root, "prepared": isolated_execution_ready["prepared"]}




def _positive_retry_code(root, package, source):
    result = _isolated_python(root, _scheduler_receipt_guarded_source(
        "from Libs.scripts.builders import submit_operational_twin_candidate_execution as submitter\n"
        + "resolved = runtime.resolve_candidate_execution_protocol(config, root)\n" + source,
        source_commit=package["fixture_source_commit"], round_id=package["fixture_round_id"]),
        stage="current recovery contract")
    guard = result.pop("guard")
    assert guard["negative_controls"] == ["opaque_receipt", "extra_input", "mutable_head_input_git"]
    assert guard["physical_input_reads"] == guard["transport_events"] == guard["unexpected_denials"] == guard["scientific_calls"] == 0
    assert guard["trace_restored"] is True
    return result


def _positive_retry_repository(tmp_path, _positive_retry_state):
    root = _positive_retry_state["root"]
    assert root == tmp_path / "non_authoritative_execution"
    config = load_config_stack(root / "Libs/configs/config_stack.yaml")
    package = {"fixture_source_commit": _positive_retry_state["prepared"]["source_commit"],
               "fixture_round_id": _positive_retry_state["prepared"]["round_id"]}
    recovery_source = r"""
recovery = operational_twin_candidate_evidence.require_recovery_eligibility(
    config, root, root / resolved.execution["output_paths"]["recovery_eligibility"], resolved_context=resolved)
assert len(recovery["assignments"]) == len(resolved.assignment_manifest["assignments"]) == 15
assert len(resolved.command_catalog["commands"]) == 30
assert [row["recovery_command_index"] for row in recovery["assignments"] if row["recovery_eligibility"] == "ELIGIBLE"] == [15]
result = {"assignment_manifest": resolved.assignment_manifest, "assignment_commands": resolved.command_catalog,
          "preflight_authority": resolved.preflight, "candidate_lineage_manifest": resolved.candidate_manifest,
          "execution": resolved.execution, "recovery": recovery}
"""
    package.update(_positive_retry_code(root, package, recovery_source))
    return config, root, package


def _retry_target_paths(package):
    return [str(attempt[field]) for assignment in package["assignment_manifest"]["assignments"]
            for attempt in assignment["attempts"] if attempt["attempt_kind"] == "infrastructure_recovery_01"
            for field in ("output_root", "execution_authority", "resource_environment_receipt")]


def _retry_guard_admission_rejection(root, package, *, expected_codes, expected_details=(), source_digest=False):
    """Capture only known admission exceptions; never label an API witness."""
    guarded = _scheduler_receipt_guarded_source(
        '_retry_submitter.submit_arrays(config, root, mode="recovery", command_capture=lambda argv, cwd: _retry_scheduler_calls.append(argv) or "0")\nresult = {}\n',
        source_commit=package["fixture_source_commit"], round_id=package["fixture_round_id"])
    assertions = [node for node in ast.walk(ast.parse(guarded)) if isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "_ce_hostname_source_sha256"]
    assert len(assertions) == 1
    assertion = assertions[0]
    prefix = "\n".join(("guard_program = " + repr(guarded), "expected_codes = " + repr(tuple(expected_codes)),
        "expected_details = " + repr(tuple(expected_details)), "source_digest = " + repr(source_digest),
        "expected_assertion_line = " + repr(assertion.lineno),
        "expected_digest = " + repr(ast.literal_eval(assertion.test.comparators[0])))) + "\n"
    program = prefix + r"""
import hashlib, json, subprocess, sys
from Libs.scripts.builders import submit_operational_twin_candidate_execution as _retry_submitter
from Libs.utils.evaluation import operational_twin_training_protocol as _retry_protocol
_retry_scheduler_calls, _retry_api_calls, _retry_wrapper_calls = [], [], []
_retry_previous_profile, _retry_previous_trace, _retry_previous_run = sys.getprofile(), sys.gettrace(), subprocess.run
_retry_api_code = _retry_submitter.submit_arrays.__code__
_retry_wrapper_code = _retry_submitter.build_array_submission_argv.__code__
def _retry_observe(frame, event, arg):
    if event == "call" and frame.f_code is _retry_api_code:
        _retry_api_calls.append(True)
    if event == "call" and frame.f_code is _retry_wrapper_code:
        _retry_wrapper_calls.append(frame.f_locals["mode"])
    if _retry_previous_profile is not None:
        _retry_previous_profile(frame, event, arg)
try:
    sys.setprofile(_retry_observe)
    try:
        exec(compile(guard_program, "<retry-guard-admission>", "exec"), globals())
    except (_retry_protocol.OperationalTwinProtocolError, AssertionError) as error:
        if source_digest:
            assert type(error) is AssertionError
            traceback = error.__traceback__
            while traceback.tb_next is not None:
                traceback = traceback.tb_next
            assert traceback.tb_frame.f_code.co_filename == "<retry-guard-admission>"
            assert traceback.tb_lineno == expected_assertion_line
            actual_digest = hashlib.sha256(traceback.tb_frame.f_locals["_ce_hostname_source"]).hexdigest()
            assert actual_digest != expected_digest
            rejection = {"exception_type": "AssertionError", "gate": "pinned_source_digest",
                "expected_digest": expected_digest, "actual_digest": actual_digest}
        else:
            assert type(error) is _retry_protocol.OperationalTwinProtocolError
            parts = str(error).split("; ")
            assert tuple(part.split(":", 1)[0] for part in parts) == expected_codes
            for part, detail in zip(parts, expected_details, strict=True):
                if detail is not None:
                    assert part.partition(":")[2].strip() == detail
            rejection = {"exception_type": type(error).__name__, "message": str(error)}
    else:
        raise AssertionError("invalid fixture passed guard admission")
finally:
    sys.setprofile(_retry_previous_profile)
assert sys.getprofile() is _retry_previous_profile and sys.gettrace() is _retry_previous_trace
assert subprocess.run is _retry_previous_run
assert not globals().get("_ce_denials", [])
assert _retry_scheduler_calls == _retry_api_calls == _retry_wrapper_calls == []
result = {"rejection_stage": "guard_admission", "api_calls": len(_retry_api_calls),
    "wrapper_calls": len(_retry_wrapper_calls), "scheduler_calls": len(_retry_scheduler_calls), "rejection": rejection,
    "profile_restored": True, "trace_restored": True, "subprocess_restored": True,
    "child_exit_semantics": "handled_expected_exception", "negative_controls": globals().get("_ce_negative_controls", [])}
print(json.dumps(result))
"""
    return _isolated_python(root, program, stage="expected retry guard admission rejection")


def _assert_retry_rejected_without_scheduler_or_writes(config, root, package, *, mutation,
    expected_codes=(), expected_details=(), rejection_stage="submission", source_digest=False, call_arguments=""):
    """Prepare first, mutate the disposable fixture outside the consumer, then reject."""
    del config
    baseline = _positive_retry_code(root, package, 'prepared = submitter.prepare_submission(config, root, mode="recovery")\nassert prepared["selected_indices"] == [15]\nassert len(prepared["resolved_protocol"].assignment_manifest["assignments"]) == 15\nassert len(prepared["resolved_protocol"].command_catalog["commands"]) == 30\nresult = {"baseline_prepared": True}\n')
    assert baseline["baseline_prepared"]
    mutation()
    state_before = _git_retry_fixture(root, "status", "--porcelain", "--untracked-files=all")
    targets_before = {name: (root / name).exists() for name in _retry_target_paths(package)}
    assert not any(targets_before.values())
    if not expected_details:
        expected_details = (None,) * len(expected_codes)
    assert len(expected_codes) == len(expected_details)
    if rejection_stage == "guard_admission":
        result = _retry_guard_admission_rejection(root, package, expected_codes=expected_codes,
            expected_details=expected_details, source_digest=source_digest)
    else:
        assert rejection_stage == "submission" and expected_codes and not source_digest
        source = "\n".join(("expected_codes = " + repr(tuple(expected_codes)),
            "expected_details = " + repr(tuple(expected_details)))) + "\n" + r"""
scheduler_calls, api_calls, initial_calls, recovery_calls = [], [], [], []
_retry_previous_profile = sys.getprofile()
_retry_api_code = submitter.submit_arrays.__code__
_retry_wrapper_code = submitter.build_array_submission_argv.__code__
_retry_validator_code = submitter.require_candidate_dispatch_manifest.__code__
def _retry_observe(frame, event, arg):
    if event == "call" and frame.f_code is _retry_api_code:
        api_calls.append(True)
    if event == "call" and frame.f_code is _retry_wrapper_code:
        if frame.f_locals["mode"] == "initial" and frame.f_back.f_code is _retry_validator_code:
            initial_calls.append(True)
        else:
            recovery_calls.append(frame.f_locals["mode"])
    if _retry_previous_profile is not None:
        _retry_previous_profile(frame, event, arg)
try:
    sys.setprofile(_retry_observe)
    with pytest.raises(protocol.OperationalTwinProtocolError) as error:
        submitter.submit_arrays(config, root, mode="recovery",
            command_capture=lambda argv, cwd: scheduler_calls.append(argv) or "0"CALL_ARGUMENTS)
finally:
    sys.setprofile(_retry_previous_profile)
assert sys.getprofile() is _retry_previous_profile
parts = str(error.value).split("; ")
assert tuple(part.split(":", 1)[0] for part in parts) == expected_codes
for part, detail in zip(parts, expected_details, strict=True):
    if detail is not None:
        assert part.partition(":")[2].strip() == detail
assert api_calls == [True] and scheduler_calls == recovery_calls == []
result = {"rejection_stage": "submission", "api_calls": len(api_calls), "scheduler_calls": len(scheduler_calls),
    "wrapper_calls": len(recovery_calls), "initial_validation_wrapper_calls": len(initial_calls),
    "rejection": str(error.value), "profile_restored": True}
""".replace("CALL_ARGUMENTS", call_arguments)
        result = _positive_retry_code(root, package, source)
    assert result["rejection_stage"] == rejection_stage
    assert result["scheduler_calls"] == result["wrapper_calls"] == 0
    assert _git_retry_fixture(root, "status", "--porcelain", "--untracked-files=all") == state_before
    assert {name: (root / name).exists() for name in targets_before} == targets_before



def test_fully_approved_positive_retry_prepares_exact_allowlist_without_submission(tmp_path, _positive_retry_state):
    config, root, package = _positive_retry_repository(tmp_path, _positive_retry_state)
    status_before = _git_retry_fixture(root, "status", "--porcelain", "--untracked-files=all")
    assert status_before == ""
    result = _positive_retry_code(root, package, r"""
source_before = runtime.require_clean_tracked_source(root)
prepared = submitter.prepare_submission(config, root, mode="recovery")
assert set(prepared) == {"resolved_protocol", "execution", "source_commit", "mode", "groups", "selected_indices",
    "execution_selection", "calibration_execution_authority", "binding"}
assert isinstance(prepared["resolved_protocol"], runtime.ResolvedCandidateExecutionProtocol)
assert prepared["resolved_protocol"].command_catalog == resolved.command_catalog
assert prepared["selected_indices"] == [15] and prepared["execution"] == resolved.execution
assert prepared["execution_selection"] and prepared["calibration_execution_authority"]
assert prepared["binding"] == submitter.worker_binding(root, resolved, prepared["source_commit"])
assert [index for group in prepared["groups"] for index in group["selected_indices"]] == [15]
assert runtime.require_clean_tracked_source(root) == source_before == prepared["source_commit"]
result = {"selected_indices": [15], "assignments": 15, "slots": 30, "typed_resolver": True, "submission_writes": 0}
""")
    assert _git_retry_fixture(root, "status", "--porcelain", "--untracked-files=all") == status_before
    assert result == {"selected_indices": [15], "assignments": 15, "slots": 30, "typed_resolver": True, "submission_writes": 0}
    assert all(not (root / path).exists() for path in _retry_target_paths(package))


def test_retry_prepare_does_not_consume_predecessor_execution_config_or_catalog(tmp_path, _positive_retry_state):
    """The compatible node name checks current catalog authority, not all file reads."""
    config, root, package = _positive_retry_repository(tmp_path, _positive_retry_state)
    result = _positive_retry_code(root, package, r"""
prepared = submitter.prepare_submission(config, root, mode="recovery")
current_catalog = prepared["execution"]["frozen_artifacts"]["assignment_commands"]
catalog = prepared["resolved_protocol"].command_catalog
assert catalog == resolved.command_catalog
hash_field = current_catalog["internal_hash_field"]
assert isinstance(hash_field, str) and hash_field
assert catalog[hash_field] == current_catalog["internal_hash"]
assert catalog == json.loads((root / current_catalog["path"]).read_text())
assert [index for group in prepared["groups"] for index in group["selected_indices"]] == [15]
result = {"selected_indices": [15], "current_catalog_authority": True}
""")
    assert result == {"selected_indices": [15], "current_catalog_authority": True}


def test_non_dry_run_retry_requires_future_handoff_before_scheduler_or_wrapper(tmp_path, _positive_retry_state):
    config, root, package = _positive_retry_repository(tmp_path, _positive_retry_state)
    path = root / package["execution"]["output_paths"]["calibration_execution_authority"]
    def mutate():
        payload = json.loads(path.read_text())
        payload["source_round_id"] = "unapproved-future-round"
        payload["authority_uid"] = compute_candidate_calibration_authority_uid(payload)
        path.write_text(json.dumps(payload))
        _commit_retry_fixture(root, "Disposable unselected calibration round")
    _assert_retry_rejected_without_scheduler_or_writes(config, root, package, mutation=mutate,
        expected_codes=("candidate_calibration_authority_runtime_invalid",),
        expected_details=("calibration_execution_authority_invalid: source_round_id does not match the approved execution contract.",))



def test_retry_submission_rejects_arbitrary_and_legacy_authority_inputs_before_scheduler(tmp_path, _positive_retry_state):
    config, root, package = _positive_retry_repository(tmp_path, _positive_retry_state)
    for name, payload in (("arbitrary-recovery-metadata.json", {}), ("legacy-recovery-metadata.json", {"assignments": []})):
        path = root / name
        def mutate(target=path, value=payload):
            target.write_text(json.dumps(value))
        _assert_retry_rejected_without_scheduler_or_writes(config, root, package, mutation=mutate,
            expected_codes=("candidate_recovery_path_not_canonical",),
            call_arguments=", recovery_eligibility_path=root / " + repr(name))



def test_retry_submission_rejects_wrong_recovery_attempt_kind_before_scheduler(tmp_path, _positive_retry_state):
    config, root, package = _positive_retry_repository(tmp_path, _positive_retry_state)
    relative = package["execution"]["frozen_artifacts"]["assignment_commands"]["path"]
    path = root / relative

    def mutate():
        control = _positive_retry_code(root, package, r"""
from copy import deepcopy
catalog = deepcopy(resolved.command_catalog)
assert protocol.validate_candidate_command_catalog(
    config, resolved.candidate_manifest, resolved.assignment_manifest, catalog) == []
recovery = catalog["commands"][15]
assert recovery["command_index"] == 15
assert recovery["attempt_kind"] == "infrastructure_recovery_01"
recovery["attempt_kind"] = "infrastructure_recovery_02"
catalog["command_catalog_hash"] = protocol.command_catalog_hash(catalog)
blockers = protocol.validate_candidate_command_catalog(
    config, resolved.candidate_manifest, resolved.assignment_manifest, catalog)
assert "candidate_command_catalog_hash_mismatch" not in {row["code"] for row in blockers}
kind_blockers = [row for row in blockers if row["code"] == "candidate_command_attempt_kind_invalid"]
assert kind_blockers == [{"code": "candidate_command_attempt_kind_invalid",
    "message": recovery["attempt_uid"] + " attempt kind is invalid."}]
assert resolved.command_catalog == json.loads((root / resolved.execution["frozen_artifacts"]["assignment_commands"]["path"]).read_text())
assert resolved.command_catalog["commands"][15]["attempt_kind"] == "infrastructure_recovery_01"
result = {"catalog": catalog, "kind_rejection_count": len(kind_blockers), "source_catalog_unchanged": True}
""")
        assert control["kind_rejection_count"] == 1
        assert control["source_catalog_unchanged"] is True
        path.write_text(json.dumps(control["catalog"]))
        _commit_retry_fixture(root, "Disposable invalid recovery attempt kind")

    _assert_retry_rejected_without_scheduler_or_writes(
        config, root, package, mutation=mutate, rejection_stage="guard_admission",
        expected_codes=("handoff_committed_bytes_drift",), expected_details=(relative,))


@pytest.mark.parametrize(
    "artifact_key,mutation",
    [
        ("recovery_eligibility", "minimal"),
        ("recovery_eligibility", "row_predicate"),
        ("recovery_eligibility", "denominator"),
        ("recovery_eligibility", "allowlist"),
        ("recovery_eligibility", "ownership"),
        ("recovery_eligibility", "authorized_command_index"),
        ("calibration_execution_authority", "calibration"),
        ("candidate_lineage_manifest", "candidate_hash"),
        ("assignment_manifest", "assignment_hash"),
        ("assignment_commands", "command_hash"),
        ("assignment_commands", "recovery_slot_count"),
        ("preflight_authority", "preflight_candidate_hash"),
        ("preflight_authority", "preflight_assignment_hash"),
        ("preflight_authority", "preflight_command_hash"),
        ("preflight_authority", "protocol_uid"),
        ("preflight_authority", "preflight_uid"),
        ("preflight_authority", "source_identity"),
        ("preflight_authority", "resource_gate"),
        ("preflight_authority", "scientific_contract_hash"),
    ],
)
def test_retry_submission_rejects_committed_audit_and_package_drift_before_scheduler(tmp_path, _positive_retry_state, artifact_key, mutation):
    config, root, package = _positive_retry_repository(tmp_path, _positive_retry_state)
    execution = package["execution"]
    relative = (execution["output_paths"][artifact_key] if artifact_key in execution["output_paths"] else execution["frozen_artifacts"][artifact_key]["path"])
    path = root / relative
    def mutate():
        payload = json.loads(path.read_text())
        if mutation == "minimal": payload = {"assignments": []}
        elif mutation == "row_predicate": payload["assignments"][1]["recovery_eligibility"] = "ELIGIBLE"
        elif mutation == "denominator": payload["assignment_count"] = 14
        elif mutation == "allowlist": payload["assignments"] = payload["assignments"][:1]
        elif mutation == "ownership": payload["assignments"][0]["initial_execution_authority"]["authority_uid"] = "unowned-authority"
        elif mutation == "calibration": payload["authority_uid"] = "invalid-calibration-authority-uid"
        elif mutation == "authorized_command_index":
            assert len(payload["assignments"]) == payload["assignment_count"] == 15
            eligible = [row for row in payload["assignments"] if row["recovery_eligibility"] == "ELIGIBLE"]
            assert [row["recovery_command_index"] for row in eligible] == [15]
            other = next(row for row in payload["assignments"] if row["recovery_command_index"] == 16)
            assert other["recovery_eligibility"] != "ELIGIBLE"
            eligible[0]["recovery_command_index"] = other["recovery_command_index"]
            assert len(payload["assignments"]) == 15
        elif mutation == "recovery_slot_count":
            assert sorted(row["command_index"] for row in payload["commands"]) == list(range(30))
            assert payload["commands"][-1]["command_index"] == 29
            payload["commands"].pop()
            assert len(payload["commands"]) == 29
        elif mutation in {"protocol_uid", "preflight_uid"}:
            assert isinstance(payload[mutation], str) and payload[mutation]
            payload[mutation] += "_tampered"
        elif mutation in {"source_identity", "scientific_contract_hash"}:
            field = "source_identity_hash" if mutation == "source_identity" else mutation
            assert len(payload[field]) == 64 and payload[field] != "0" * 64
            payload[field] = "0" * 64
        elif mutation == "resource_gate":
            placement = payload["resource_execution_gate"]["trainable_placement"]
            assert placement["nodelist"] != "cpu04"
            placement["nodelist"] = "cpu04"
        else:
            field = {"candidate_hash":"candidate_manifest_hash", "assignment_hash":"assignment_manifest_hash", "command_hash":"command_catalog_hash",
                "preflight_candidate_hash":"candidate_manifest_hash", "preflight_assignment_hash":"assignment_manifest_hash", "preflight_command_hash":"command_catalog_hash"}[mutation]
            payload[field] = "0" * 64
        path.write_text(json.dumps(payload))
        _commit_retry_fixture(root, "Disposable recovery metadata mutation")
    if artifact_key == "recovery_eligibility":
        stage, codes, details = "submission", ("candidate_recovery_keyset_mismatch" if mutation == "minimal" else "candidate_recovery_evidence_drift",), ()
    elif artifact_key == "calibration_execution_authority":
        stage = "submission"
        codes = ("candidate_calibration_authority_uid_mismatch", "candidate_calibration_authority_runtime_invalid")
        details = ("Candidate calibration authority UID does not recompute.", "calibration_execution_authority_invalid: authority UID does not recompute.")
    else:
        stage, codes, details = "guard_admission", ("handoff_committed_bytes_drift",), (relative,)
    _assert_retry_rejected_without_scheduler_or_writes(config, root, package, mutation=mutate,
        rejection_stage=stage, expected_codes=codes, expected_details=details)



@pytest.mark.parametrize("review_state", ["missing","rejected","uncommitted","mismatched"])
def test_retry_submission_rejects_invalid_review_before_scheduler(tmp_path, _positive_retry_state, review_state):
    config, root, package = _positive_retry_repository(tmp_path, _positive_retry_state)
    relative = config["operational_twin_successor_execution_handoff"]["review_contract"]["path"]
    path = root / relative
    def mutate():
        review = json.loads(path.read_text())
        if review_state == "missing": path.unlink()
        else:
            if review_state == "rejected": review["decision"] = "rejected"
            else: review["evidence_summary"] = "unbound review"
            path.write_text(json.dumps(review))
        if review_state != "uncommitted": _commit_retry_fixture(root, "Disposable review mutation")
    expected = {"missing": (("handoff_git_evidence_unavailable",), ("show HEAD:" + relative,)), "rejected": (("handoff_review_not_approved",), ()),
        "uncommitted": (("handoff_committed_bytes_drift",), (relative,)), "mismatched": (("handoff_review_summary_binding_mismatch",), ())}
    codes, details = expected[review_state]
    _assert_retry_rejected_without_scheduler_or_writes(config, root, package, mutation=mutate,
        rejection_stage="guard_admission", expected_codes=codes, expected_details=details)



@pytest.mark.parametrize("drift_kind", ["uncommitted_audit","source_file","committed_report"])
def test_retry_submission_rejects_byte_or_source_drift_before_scheduler(tmp_path, _positive_retry_state, drift_kind):
    config, root, package = _positive_retry_repository(tmp_path, _positive_retry_state)
    def mutate():
        if drift_kind == "uncommitted_audit":
            path = root / package["execution"]["output_paths"]["recovery_eligibility"]
            path.write_bytes(path.read_bytes() + b" ")
        elif drift_kind == "source_file":
            path = root / "Libs/utils/evaluation/operational_twin_candidate_execution.py"
            content = path.read_bytes(); position = content.index(b"# ") + 1
            changed = content[:position] + b"\t" + content[position + 1:]
            assert len(changed) == len(content) and changed != content
            compile(changed, str(path), "exec"); path.write_bytes(changed)
        else:
            path = root / config["operational_twin_successor_execution_handoff"]["output_paths"]["report"]
            path.write_bytes(path.read_bytes() + b" ")
        if drift_kind != "uncommitted_audit": _commit_retry_fixture(root, "Disposable source metadata mutation")
    if drift_kind == "uncommitted_audit":
        stage, codes, details, digest = "submission", ("candidate_execution_source_dirty",), ("tracked execution source must be committed before gate creation.",), False
    elif drift_kind == "source_file":
        stage, codes, details, digest = "guard_admission", (), (), True
    else:
        stage, codes, details, digest = "guard_admission", ("handoff_materialized_publication_drift",), ("report",), False
    _assert_retry_rejected_without_scheduler_or_writes(config, root, package, mutation=mutate,
        rejection_stage=stage, expected_codes=codes, expected_details=details, source_digest=digest)


@pytest.mark.parametrize("authority_state", ["negative", "forged_positive"])
def test_retry_rejects_retained_negative_or_forged_positive_eligibility_before_scheduler(
    tmp_path, _positive_retry_state, authority_state
):
    config, root, package = _positive_retry_repository(tmp_path, _positive_retry_state)

    def mutate():
        assignment = package["assignment_manifest"]["assignments"][0]
        initial = next(row for row in assignment["attempts"] if row["attempt_kind"] == "initial")
        terminal_path = root / initial["output_root"] / "terminal_status.json"
        terminal = json.loads(terminal_path.read_text())
        assert terminal["status"] == "FAILED"
        assert terminal["failure_class"] == "infrastructure_interruption"
        # Invalidate the retained predecessor identity in this disposable fixture.
        # The maintained producer must derive the complete negative authority.
        terminal_path.write_bytes(terminal_path.read_bytes() + b"\n")
        negative = _positive_retry_code(root, package, r"""
from Libs.utils.evaluation import operational_twin_candidate_evidence as evidence
paths = resolved.execution["output_paths"]
dispatch = json.loads((root / paths["dispatch_manifest"]).read_text())
scheduler = evidence.load_scheduler_receipts(
    root, resolved.execution, dispatch["initial_arrays"], phase="initial",
    source_commit=dispatch["source_commit"], resolved_context=resolved)
negative = evidence.build_recovery_eligibility(
    config, root, dispatch_manifest=dispatch, scheduler_by_index=scheduler,
    resolved_context=resolved)
assert negative["assignment_count"] == len(negative["assignments"]) == 15
assert len(resolved.command_catalog["commands"]) == 30
assert all(row["recovery_eligibility"] != "ELIGIBLE" for row in negative["assignments"])
assert negative["assignments"][0]["validation_blockers"]
result = {"negative": negative}
""")["negative"]
        path = root / package["execution"]["output_paths"]["recovery_eligibility"]
        path.write_text(json.dumps(negative))
        _commit_retry_fixture(root, "Retain a recomputed complete negative recovery authority")
        verified = _positive_retry_code(root, package, r"""
from Libs.utils.evaluation import operational_twin_candidate_evidence as evidence
retained = evidence.require_recovery_eligibility(
    config, root, root / resolved.execution["output_paths"]["recovery_eligibility"],
    resolved_context=resolved)
assert len(retained["assignments"]) == retained["assignment_count"] == 15
assert all(row["recovery_eligibility"] != "ELIGIBLE" for row in retained["assignments"])
state_before = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root)
paths = resolved.execution["output_paths"]
targets_before = {key: (root / paths[key]).exists() for key in ("recovery_dispatch_claim", "recovery_dispatch_manifest")}
assert not any(targets_before.values())
prepared = submitter.prepare_submission(config, root, mode="recovery")
assert prepared["selected_indices"] == prepared["groups"] == []
assert subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root) == state_before
assert {key: (root / paths[key]).exists() for key in targets_before} == targets_before
result = {"complete_negative_verified": True}
""")
        assert verified == {"complete_negative_verified": True}
        if authority_state == "forged_positive":
            forged = deepcopy(negative)
            # Native authority is per-assignment eligibility. A resolved header
            # does not turn an invalid predecessor into permission to recover.
            assert forged["status"] == "PASS"
            assert forged["terminal_decision"] == "OPERATIONAL_TWIN_CANDIDATE_RECOVERY_ELIGIBILITY_RESOLVED"
            forged["assignments"][0]["recovery_eligibility"] = "ELIGIBLE"
            forged["assignments"][0]["validation_blockers"] = []
            path.write_text(json.dumps(forged))
            _commit_retry_fixture(root, "Forge positive eligibility on a complete negative authority")

    expected_code = (
        "candidate_recovery_no_eligible_assignments"
        if authority_state == "negative"
        else "candidate_recovery_evidence_drift"
    )
    _assert_retry_rejected_without_scheduler_or_writes(
        config, root, package, mutation=mutate, expected_codes=(expected_code,)
    )
