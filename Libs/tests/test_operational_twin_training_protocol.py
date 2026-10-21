from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from Libs.tests import operational_twin_witness as training_protocol_witness

from Libs.tests.test_operational_twin_candidate_execution import (
    _CANDIDATE_RUNTIME_METADATA_FIXTURE,
    _run_candidate_gate_fixture,
    config as config,
    isolated_execution_ready as isolated_execution_ready,
    isolated_reviewed_handoff as isolated_reviewed_handoff,
)

from Libs.exp.training import train_operational_twin_devcal as operational_twin_runner
from Libs.exp.training import run_operational_twin_assignment_command as operational_twin_command_runner
from Libs.exp.training.train_operational_twin_devcal import _formal_assignment_args, resolve_operational_twin_run_contract
from Libs.exp.training.train_operational_twin_devcal import (
    _action_encoding_for_contract,
    build_method_model,
    count_trainable_parameters,
    enforce_capacity,
    fit_calibration,
    loss_for_method,
    method_protocol_config,
    model_config_for_method,
    require_calibration_execution_authority,
    require_assignment_runtime_resource,
)
from Libs.scripts.builders import build_operational_twin_training_protocol as operational_twin_protocol_builder
from Libs.scripts.builders.build_operational_twin_development_training_commands import build_candidate_commands
from Libs.models.networks.operational_twin import (
    LinearDeltaOperationalTwin,
    MeanOnlyOperationalTwin,
    OperationalTwin,
    PersistenceNoActionPredictor,
)
from Libs.utils.evaluation import operational_twin_candidate_execution as candidate_execution
from Libs.utils.evaluation import operational_twin_training_protocol as operational_twin_protocol
from Libs.utils.evaluation import operational_twin_successor_execution_handoff as handoff
from Libs.utils.evaluation import operational_twin_candidate_retry_authority as retry_authority
from Libs.utils.evaluation.operational_twin_training_protocol import (
    CANDIDATE_COMMAND_ROW_KEYS,
    CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS,
    CANDIDATE_EXECUTION_AUTHORITY_KEYS,
    CANDIDATE_FAILED_TERMINAL_KEYS,
    CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS,
    CANDIDATE_IDENTITY_FIELDS,
    CANDIDATE_RESOURCE_ENVIRONMENT_KEYS,
    CANDIDATE_RESOURCE_ALLOCATION_KEYS,
    CANDIDATE_RESOURCE_PROFILES,
    CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS,
    CANDIDATE_RESOURCE_RECEIPT_KEYS,
    CANDIDATE_RESOURCE_SLURM_KEYS,
    CANDIDATE_SCIENTIFIC_EQUALITY_FIELDS,
    CANDIDATE_SUPERSESSION_CONTRACT,
    METHOD_EXECUTION_CONTRACTS,
    REQUIRED_METHOD_IDS,
    OperationalTwinProtocolError,
    build_assignment_manifest,
    build_candidate_assignment_manifest,
    build_candidate_lineage_manifest,
    build_preflight_authority,
    candidate_attempt_path,
    candidate_execution_artifact_paths,
    candidate_fresh_initialization_receipt,
    candidate_identity_payload,
    candidate_runtime_slurm_identity,
    candidate_source_identity_hash,
    candidate_supersession_contract_hash,
    command_catalog_hash,
    compute_candidate_execution_authority_uid,
    compute_candidate_resource_receipt_uid,
    compute_candidate_assignment_uid,
    compute_candidate_attempt_uid,
    compute_candidate_lineage_uid,
    compute_candidate_protocol_uid,
    expected_candidate_command_argv,
    require_assignment_preflight,
    require_candidate_assignment_preflight,
    require_candidate_execution_authority,
    repo_file_identity,
    resolve_method_execution_contract,
    sha256_file,
    sha256_text,
    validate_candidate_command_catalog,
    validate_candidate_protocol_config,
    validate_assignment_runtime_evidence,
    validate_resource_receipt,
    validate_preflight_authority,
    validate_protocol_config,
)
from Libs.utils.exp_utils import load_config_stack


ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict[str, object]:
    return load_config_stack(ROOT / "Libs/configs/config_stack.yaml")


def _tiny_model_config() -> dict[str, object]:
    return {
        "hidden_dim": 8,
        "posterior_dim": 2,
        "depth": 1,
        "head_count": 2,
        "mechanism_token_count": 2,
        "dropout_rate": 0.0,
        "minimum_scale": 0.0001,
    }


def _tiny_batch(action_value: float = 1.0) -> dict[str, object]:
    observation = torch.tensor(
        [
            [[1.0, 0.5], [2.0, 0.25], [0.0, 0.0]],
            [[-1.0, 1.5], [-0.5, 1.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor(
        [
            [[True, True], [True, True], [False, False]],
            [[True, True], [True, True], [False, False]],
        ]
    )
    action = torch.full((2, 4), action_value)
    return {
        "input_observation": observation,
        "input_mask": mask,
        "target_observation": observation + 0.1,
        "target_mask": torch.ones_like(mask, dtype=torch.bool),
        "action_encoding": action,
        "no_action_encoding": torch.zeros_like(action),
        "family_index": torch.tensor([0, 1], dtype=torch.long),
    }


def _resource_receipt(config: dict[str, object]) -> dict[str, object]:
    protocol_text = yaml.safe_dump(config["operational_twin_training_protocol"], sort_keys=True)
    return {
        "schema_version": "neuronsifter-operational-twin-resource-receipt-v1",
        "config_hash": sha256_text(protocol_text),
        "probes": {
            "cpu_probe": {
                "job_name": "NrnSift_protocol_cpu_probe",
                "exit_status": 0,
                "sacct_evidence": "535011|NrnSift_protocol_cpu_probe|cpu|cpu04|COMPLETED|0:0|1|2G|00:00:02",
            },
            "gpu_probe": {
                "job_id": "535012",
                "job_name": "NrnSift_protocol_gpu_probe",
                "exit_status": 0,
                "environment": {"probe_stdout": "SLURM_JOB_ID=535012\nCUDA_VISIBLE_DEVICES=0\n{\"cuda_available\": true}"},
                "sacct_evidence": "535012|NrnSift_protocol_gpu_probe|gpu1|gpu03|COMPLETED|0:0|1|4G|00:00:13",
            },
        },
    }


def _candidate_artifacts(
    config: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    resolved_config = config or _config()
    candidate_manifest = build_candidate_lineage_manifest(resolved_config)
    assignment_manifest = build_candidate_assignment_manifest(
        resolved_config,
        candidate_manifest,
    )
    command_catalog = build_candidate_commands(
        resolved_config,
        candidate_manifest,
        assignment_manifest,
    )
    return resolved_config, candidate_manifest, assignment_manifest, command_catalog


def _candidate_runtime_artifacts(
    _tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    config = deepcopy(_config())
    return _candidate_artifacts(config)


def _candidate_runtime_kwargs(
    candidate_manifest: dict[str, object],
    assignment_manifest: dict[str, object],
    command_catalog: dict[str, object],
    assignment: dict[str, object],
    attempt: dict[str, object],
) -> dict[str, object]:
    return {
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest["assignment_manifest_hash"],
        "command_catalog_hash_value": command_catalog["command_catalog_hash"],
        "preflight_uid": candidate_manifest["preflight_uid"],
        "state_inputs": {
            field: None for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS
        },
    }


def _write_candidate_execution_gate(
    root: Path,
    config: dict[str, object],
    candidate_manifest: dict[str, object],
    assignment_manifest: dict[str, object],
    command_catalog: dict[str, object],
    preflight: dict[str, object],
    assignment: dict[str, object],
    attempt: dict[str, object],
    *,
    source_commit: str,
    slurm: dict[str, object],
    environment: dict[str, object],
    timestamp: datetime | None = None,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    protocol = config["operational_twin_candidate_lineage_protocol"]
    contract = protocol["execution_authority_contract"]
    execution_paths = candidate_execution_artifact_paths(
        protocol,
        assignment,
        attempt,
    )
    review_path = root / contract["approved_protocol_review_path"]
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
        json.dumps(
            {
                "schema_version": contract["required_review_schema_version"],
                "round_id": contract["required_review_round_id"],
                "direction_id": contract["required_review_direction_id"],
                "decision": "approved",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    review_identity = repo_file_identity(root, review_path)
    timestamp_text = (timestamp or datetime.now(timezone.utc)).isoformat()
    sinfo_stdout = (
        f"{assignment['resource_requirements']['nodelist']}|"
        f"{assignment['resource_requirements']['partition']}|alloc|"
        f"{assignment['resource_requirements']['gres']}\n"
    )
    resource_requirements = assignment["resource_requirements"]
    tres = (
        f"cpu={resource_requirements['cpus_per_task']},node=1,"
        f"billing={resource_requirements['cpus_per_task']}"
    )
    tres_per_node = (
        " TresPerNode=gres:gpu:1"
        if resource_requirements["gres"] == "gpu:1"
        else ""
    )
    resource_stdout = (
        f"JobId={slurm['slurm_job_id']} "
        f"ArrayJobId={slurm['slurm_array_job_id']} "
        f"ArrayTaskId={slurm['slurm_array_task_id']} "
        f"Partition={resource_requirements['partition']} "
        f"NodeList={resource_requirements['nodelist']} "
        f"NumCPUs={resource_requirements['cpus_per_task']} "
        f"CPUs/Task={resource_requirements['cpus_per_task']} "
        f"TRES={tres} MinMemoryNode={resource_requirements['memory']} "
        f"OverSubscribe=OK{tres_per_node}\n"
    )
    receipt: dict[str, object] = {
        "schema_version": contract["resource_receipt_schema_version"],
        "status": "PASS",
        "terminal_decision": contract["positive_resource_receipt_terminal"],
        "receipt_uid": "pending",
        "created_at": timestamp_text,
        "sinfo_captured_at": timestamp_text,
        "sinfo_command": [
            str(value).format(
                nodelist=assignment["resource_requirements"]["nodelist"]
            )
            for value in contract["fresh_sinfo_command"]
        ],
        "sinfo_stdout": sinfo_stdout,
        "sinfo_stdout_sha256": sha256_text(sinfo_stdout),
        "resource_captured_at": timestamp_text,
        "resource_command": [
            str(value).format(
                resource_query_job_id=slurm["resource_query_job_id"]
            )
            for value in contract["fresh_resource_command"]
        ],
        "resource_stdout": resource_stdout,
        "resource_stdout_sha256": sha256_text(resource_stdout),
        "resource_allocation": resource_requirements,
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "preflight_uid": candidate_manifest["preflight_uid"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest[
            "assignment_manifest_hash"
        ],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
        "source_commit": source_commit,
        "source_identity_hash": candidate_source_identity_hash(preflight),
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
        "resource_profile": assignment["resource_profile"],
        "resource_requirements": assignment["resource_requirements"],
        "slurm": slurm,
        "environment": environment,
        "fresh_sinfo": True,
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "access_counters": protocol["access_counters"],
    }
    assert set(receipt) == CANDIDATE_RESOURCE_RECEIPT_KEYS
    assert set(receipt["slurm"]) == CANDIDATE_RESOURCE_SLURM_KEYS
    assert set(receipt["environment"]) == CANDIDATE_RESOURCE_ENVIRONMENT_KEYS
    assert set(receipt["resource_allocation"]) == CANDIDATE_RESOURCE_ALLOCATION_KEYS
    receipt["receipt_uid"] = compute_candidate_resource_receipt_uid(receipt)
    receipt_path = root / execution_paths["resource_environment_receipt"]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_identity = repo_file_identity(root, receipt_path)
    authority: dict[str, object] = {
        "schema_version": contract["authority_schema_version"],
        "status": "PASS",
        "terminal_decision": contract["positive_authority_terminal"],
        "authority_uid": "pending",
        "issued_at": timestamp_text,
        "authority_role": contract["authority_role"],
        "approved_protocol_review_identity": review_identity,
        "approved_protocol_review_decision": "approved",
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "preflight_uid": candidate_manifest["preflight_uid"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest[
            "assignment_manifest_hash"
        ],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
        "source_commit": source_commit,
        "source_identity_hash": candidate_source_identity_hash(preflight),
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
        "resource_profile": assignment["resource_profile"],
        "resource_requirements": assignment["resource_requirements"],
        "resource_environment_receipt_identity": {
            **receipt_identity,
            "receipt_uid": receipt["receipt_uid"],
        },
        "fresh_initialization_receipt": candidate_fresh_initialization_receipt(
            protocol
        ),
        "state_resume_allowed": False,
        "training_execution_authorized": True,
        "checkpoint_creation_authorized": assignment[
            "checkpoint_creation_allowed"
        ],
        "calibration_fitting_authorized": assignment["calibration_eligible"],
        "access_counters": protocol["access_counters"],
    }
    assert set(authority) == CANDIDATE_EXECUTION_AUTHORITY_KEYS
    authority["authority_uid"] = compute_candidate_execution_authority_uid(
        authority
    )
    authority_path = root / execution_paths["execution_authority"]
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.write_text(
        json.dumps(authority, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return authority_path, receipt_path, authority, receipt


def _set_candidate_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
    resource_requirements: dict[str, object],
    *,
    torch_version: str = "test-torch",
) -> tuple[dict[str, object], dict[str, object]]:
    node = str(resource_requirements["nodelist"])
    profile_uses_gpu = resource_requirements["gres"] == "gpu:1"
    values: dict[str, str | None] = {
        "SLURM_JOB_ID": "900001",
        "SLURM_ARRAY_JOB_ID": "900000",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_NAME": "NrnSift_candidate_assignment",
        "SLURM_JOB_PARTITION": str(resource_requirements["partition"]),
        "SLURM_JOB_NODELIST": node,
        "SLURM_CPUS_PER_TASK": str(resource_requirements["cpus_per_task"]),
        "SLURM_MEM_PER_NODE": {
            "8G": "8192",
            "32G": "32768",
        }[str(resource_requirements["memory"])],
        "SLURM_MEM_PER_CPU": None,
        "SLURM_JOB_GRES": "gpu:1" if profile_uses_gpu else None,
        "SLURM_JOB_GPUS": "0" if profile_uses_gpu else None,
        "SLURM_GPUS_ON_NODE": "1" if profile_uses_gpu else None,
        "SLURM_STEP_GPUS": "0" if profile_uses_gpu else None,
        "CUDA_VISIBLE_DEVICES": "0" if profile_uses_gpu else None,
    }
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        operational_twin_protocol.socket,
        "gethostname",
        lambda: node,
    )
    slurm = candidate_runtime_slurm_identity(hostname=node)
    environment = operational_twin_protocol._runtime_environment_identity(
        hostname=node,
        cuda_available=profile_uses_gpu,
        cuda_device_count=1 if profile_uses_gpu else 0,
        torch_version=torch_version,
    )
    return slurm, environment


def _candidate_execution_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_id: str,
) -> dict[str, object]:
    config, candidate_manifest, assignment_manifest, command_catalog = (
        _candidate_runtime_artifacts(tmp_path)
    )
    assignment = next(
        row
        for row in assignment_manifest["assignments"]
        if row["method_id"] == method_id and row["seed"] == 360101
    )
    attempt = assignment["attempts"][0]
    preflight = {
        "preflight_uid": candidate_manifest["preflight_uid"],
        "config_identity": {"path": "candidate.yaml", "sha256": "config"},
        "schema_identity": {"path": "candidate.schema.yaml", "sha256": "schema"},
        "source_identities": [{"path": "trainer.py", "sha256": "source"}],
    }
    slurm, environment = _set_candidate_runtime_environment(
        monkeypatch,
        assignment["resource_requirements"],
    )
    timestamp = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
    authority_path, receipt_path, authority, receipt = (
        _write_candidate_execution_gate(
            tmp_path,
            config,
            candidate_manifest,
            assignment_manifest,
            command_catalog,
            preflight,
            assignment,
            attempt,
            source_commit="candidate-source-commit",
            slurm=slurm,
            environment=environment,
            timestamp=timestamp,
        )
    )
    return {
        "config": config,
        "candidate_manifest": candidate_manifest,
        "assignment_manifest": assignment_manifest,
        "command_catalog": command_catalog,
        "preflight": preflight,
        "assignment": assignment,
        "attempt": attempt,
        "authority_path": authority_path,
        "receipt_path": receipt_path,
        "authority": authority,
        "receipt": receipt,
        "source_commit": "candidate-source-commit",
        "now": datetime(2026, 9, 8, 12, 5, tzinfo=timezone.utc),
    }


def _require_candidate_execution_fixture(
    tmp_path: Path,
    fixture: dict[str, object],
) -> dict[str, object]:
    return require_candidate_execution_authority(
        fixture["config"],
        tmp_path,
        assignment=fixture["assignment"],
        attempt=fixture["attempt"],
        candidate_manifest=fixture["candidate_manifest"],
        assignment_manifest=fixture["assignment_manifest"],
        command_catalog=fixture["command_catalog"],
        preflight=fixture["preflight"],
        execution_authority_path=fixture["authority_path"],
        resource_environment_receipt_path=fixture["receipt_path"],
        cuda_available=fixture["assignment"]["resource_profile"] == "gpu_single",
        cuda_device_count=(
            1 if fixture["assignment"]["resource_profile"] == "gpu_single" else 0
        ),
        torch_version="test-torch",
        current_source_commit=fixture["source_commit"],
        now=fixture["now"],
    )


def _patch_candidate_runtime_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    candidate_manifest: dict[str, object],
    assignment_manifest: dict[str, object],
    command_catalog: dict[str, object],
    preflight: dict[str, object] | None = None,
) -> None:
    resolved_preflight = preflight if preflight is not None else {"status": "PASS"}
    monkeypatch.setattr(
        operational_twin_protocol,
        "load_candidate_protocol_artifacts",
        lambda config, root, *, resolved_context=None: (
            candidate_manifest,
            assignment_manifest,
            command_catalog,
            resolved_preflight,
        ),
    )
    monkeypatch.setattr(
        operational_twin_protocol,
        "validate_candidate_artifacts",
        lambda *args, **kwargs: [],
    )


def _candidate_recovery_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    execution_fixture = _candidate_execution_fixture(
        tmp_path,
        monkeypatch,
        "persistence_no_action",
    )
    config = execution_fixture["config"]
    protocol = config["operational_twin_candidate_lineage_protocol"]
    candidate_manifest = execution_fixture["candidate_manifest"]
    assignment_manifest = execution_fixture["assignment_manifest"]
    command_catalog = execution_fixture["command_catalog"]
    preflight = execution_fixture["preflight"]
    assignment = execution_fixture["assignment"]
    initial, recovery = assignment["attempts"]
    assert initial == execution_fixture["attempt"]
    _patch_candidate_runtime_artifacts(
        monkeypatch,
        candidate_manifest,
        assignment_manifest,
        command_catalog,
        preflight,
    )

    authority = execution_fixture["authority"]
    receipt = execution_fixture["receipt"]
    authority_path = execution_fixture["authority_path"]
    receipt_path = execution_fixture["receipt_path"]
    expected_paths = candidate_execution_artifact_paths(
        protocol,
        assignment,
        initial,
    )
    assert initial["execution_authority"] == expected_paths["execution_authority"]
    assert (
        initial["resource_environment_receipt"]
        == expected_paths["resource_environment_receipt"]
    )

    source_identity_hash = candidate_source_identity_hash(preflight)
    expected_evidence_identity = {
        "candidate_lineage_uid": candidate_manifest["candidate_lineage_uid"],
        "protocol_uid": candidate_manifest["protocol_uid"],
        "preflight_uid": candidate_manifest["preflight_uid"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest[
            "assignment_manifest_hash"
        ],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
        "source_commit": execution_fixture["source_commit"],
        "source_identity_hash": source_identity_hash,
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": initial["attempt_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": initial["output_root"],
        "resource_profile": assignment["resource_profile"],
        "resource_requirements": assignment["resource_requirements"],
    }
    for field, expected_value in expected_evidence_identity.items():
        assert authority[field] == expected_value
        assert receipt[field] == expected_value

    contract = protocol["execution_authority_contract"]
    review_path = tmp_path / contract["approved_protocol_review_path"]
    review_identity = repo_file_identity(tmp_path, review_path)
    assert authority["approved_protocol_review_identity"] == review_identity
    assert receipt["resource_allocation"] == CANDIDATE_RESOURCE_PROFILES[
        assignment["resource_profile"]
    ]

    authority_binding = {
        **repo_file_identity(tmp_path, authority_path),
        "authority_uid": authority["authority_uid"],
        "source_commit": execution_fixture["source_commit"],
        "approved_protocol_review_identity": review_identity,
    }
    receipt_binding = {
        **repo_file_identity(tmp_path, receipt_path),
        "receipt_uid": receipt["receipt_uid"],
        "source_commit": execution_fixture["source_commit"],
        "scheduler": dict(receipt["slurm"]),
        "environment": dict(receipt["environment"]),
        "resource_allocation": dict(receipt["resource_allocation"]),
    }
    assert set(authority_binding) == CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS
    assert set(receipt_binding) == CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS
    assert authority_binding["path"] == initial["execution_authority"]
    assert authority_binding["sha256"] == sha256_file(authority_path)
    assert receipt_binding["path"] == initial["resource_environment_receipt"]
    assert receipt_binding["sha256"] == sha256_file(receipt_path)

    predecessor_status = {
        "schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
        **candidate_identity_payload(assignment, initial),
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": initial["output_root"],
        "status": "FAILED",
        "failure_class": "infrastructure_interruption",
        "terminal_reason": "InterruptedError",
        "error": "retained infrastructure interruption fixture",
        "started_at": execution_fixture["now"].isoformat(),
        "finished_at": "2026-09-08T12:06:00+00:00",
        "scheduler_identity": dict(receipt["slurm"]),
        "access_flags": deepcopy(operational_twin_runner.EXPECTED_ACCESS_POLICY),
        "claim_flags": deepcopy(operational_twin_runner.EXPECTED_CLAIM_BOUNDARY),
        "fresh_initialization_receipt": candidate_fresh_initialization_receipt(
            protocol
        ),
        "state_resume_used": False,
        "candidate_execution_authority": authority_binding,
        "candidate_resource_environment_receipt": receipt_binding,
    }
    assert len(predecessor_status) == 21
    assert set(predecessor_status) == CANDIDATE_FAILED_TERMINAL_KEYS
    return {
        **execution_fixture,
        "protocol": protocol,
        "initial": initial,
        "recovery": recovery,
        "predecessor_status": predecessor_status,
        "status_path": tmp_path / initial["output_root"] / "terminal_status.json",
        "recovery_kwargs": _candidate_runtime_kwargs(
            candidate_manifest,
            assignment_manifest,
            command_catalog,
            assignment,
            recovery,
        ),
    }


def _write_candidate_recovery_status(
    status_path: Path,
    status: dict[str, object],
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(status, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_validator_authority_bytes(
    config: dict[str, object],
) -> dict[str, bytes]:
    protocol = config["operational_twin_candidate_lineage_protocol"]
    scientific = protocol["scientific_contract"]
    sealed = protocol["sealed_manifest_metadata"]
    source_bindings = protocol["source_bindings"]
    source_bytes: dict[str, bytes] = {}

    source_bytes[protocol["historical_boundary"]["audit_path"]] = (
        json.dumps(
            {
            "terminal": "OPERATIONAL_TWIN_TRAINING_EVIDENCE_INVALID",
            "boundary": {
                "sufficient_training": False,
                "twin_qualification_pass": False,
                "formal_qualification": False,
            },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    source_bytes[sealed["metadata_source"]] = yaml.safe_dump(
        {
            "controlled_world_open_action_identification": {
                "historical_benchmark_provenance": {
                    field: sealed[field]
                    for field in (
                        "complete_test_manifest_git_blob",
                        "freeze_uid",
                        "test_manifest_uid",
                        "test_content_identity",
                    )
                }
            }
        },
        sort_keys=True,
    ).encode("utf-8")
    for key, value in source_bindings.items():
        if key in {"snapshot_invariant", "snapshot_repair_prospective_only"}:
            continue
        source_bytes[str(value)] = f"# {key} fixture\n".encode("utf-8")
    source_bytes[source_bindings["trainer"]] = b"best_payload = deepcopy(payload)\n"
    source_bytes[source_bindings["protocol_regression_test"]] = (
        b"def test_selected_checkpoint_snapshot_survives_later_training_and_calibration():\n"
        b"    pass\n"
    )
    for fixture_key in (
        "cpu_receipt_fixture",
        "gpu_receipt_fixture",
        "array_element_resource_fixture",
        "array_terminal_accounting_fixture",
    ):
        fixture_path = str(source_bindings[fixture_key])
        source_bytes[fixture_path] = operational_twin_protocol.read_repo_bytes(
            ROOT, fixture_path
        )

    input_identities: dict[str, object] = {}
    input_content_identities: dict[str, object] = {}
    for role, path_value in scientific["manifests"].items():
        path = str(path_value)
        content_identity = f"{role}_content_identity"
        content = (
            json.dumps(
                {"content_identity": content_identity},
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        source_bytes[path] = content
        input_identities[role] = {
            "path": path,
            "sha256": sha256_text(content.decode("utf-8")),
        }
        input_content_identities[role] = content_identity
    source_bytes[scientific["open_data_authority"]] = (
        json.dumps(
            {
                "terminal_decision": scientific["required_open_data_terminal"],
                "training_authorized": True,
                "input_identities": input_identities,
                "input_content_identities": input_content_identities,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    source_bytes[
        "Libs/configs/model/operational_twin_candidate_lineage_protocol.yaml"
    ] = b"candidate protocol config fixture\n"
    source_bytes[
        "Libs/configs/schema/operational_twin_candidate_lineage_protocol.schema.yaml"
    ] = b"candidate protocol schema fixture\n"
    rejected_review = CANDIDATE_SUPERSESSION_CONTRACT["rejected_review_record"]
    source_bytes[str(rejected_review["path"])] = (
        operational_twin_protocol.read_repo_bytes(
            ROOT, str(rejected_review["path"])
        )
    )
    for identity in CANDIDATE_SUPERSESSION_CONTRACT[
        "predecessor_artifacts"
    ].values():
        source_bytes[str(identity["path"])] = operational_twin_protocol.read_repo_bytes(
            ROOT, str(identity["path"])
        )
    amendment = CANDIDATE_SUPERSESSION_CONTRACT[
        "resource_receipt_amendment_basis"
    ]
    for evidence_role in (
        "approved_round_43_review_record",
        "approved_round_44_review_record",
        "approved_rev_007_update",
        "approved_rev_007_update_review",
    ):
        identity = amendment[evidence_role]
        source_bytes[str(identity["path"])] = operational_twin_protocol.read_repo_bytes(
            ROOT, str(identity["path"])
        )
    for generation in amendment["predecessor_generations"].values():
        for identity in generation["artifacts"].values():
            source_bytes[str(identity["path"])] = (
                operational_twin_protocol.read_repo_bytes(
                    ROOT, str(identity["path"])
                )
            )
    repair = CANDIDATE_SUPERSESSION_CONTRACT["execution_interface_repair_basis"]
    for evidence_role in (
        "approved_round_46_review_record",
        "implementation_notes",
        "blocker_summary",
    ):
        identity = repair[evidence_role]
        source_bytes[str(identity["path"])] = operational_twin_protocol.read_repo_bytes(
            ROOT, str(identity["path"])
        )

    return source_bytes


def _patch_candidate_validator_authorities(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
) -> dict[str, bytes]:
    source_bytes = _candidate_validator_authority_bytes(config)

    def normalized_path(root: Path, value: str | Path) -> str:
        path = Path(value)
        if path.is_absolute():
            return path.resolve().relative_to(root.resolve()).as_posix()
        return path.as_posix()

    def read_bytes(root: Path, value: str | Path) -> bytes:
        path = normalized_path(root, value)
        if path not in source_bytes:
            raise AssertionError(f"Unexpected candidate validator read: {path}")
        return source_bytes[path]

    def read_json(root: Path, value: str | Path) -> dict[str, object]:
        payload = json.loads(read_bytes(root, value).decode("utf-8"))
        assert isinstance(payload, dict)
        return payload

    monkeypatch.setattr(operational_twin_protocol, "read_repo_json", read_json)
    monkeypatch.setattr(operational_twin_protocol, "read_repo_bytes", read_bytes)
    rejected_review = CANDIDATE_SUPERSESSION_CONTRACT["rejected_review_record"]
    amendment = CANDIDATE_SUPERSESSION_CONTRACT[
        "resource_receipt_amendment_basis"
    ]
    rejected_object_bytes = operational_twin_protocol._git_object_bytes(
        ROOT,
        str(rejected_review["git_object_id"]),
    )
    object_bytes = {
        str(rejected_review["git_object_id"]): rejected_object_bytes
    }
    for evidence_role in (
        "approved_round_43_review_record",
        "approved_round_44_review_record",
        "approved_rev_007_update",
        "approved_rev_007_update_review",
    ):
        identity = amendment[evidence_role]
        object_bytes[str(identity["git_object_id"])] = source_bytes[
            str(identity["path"])
        ]
    repair = CANDIDATE_SUPERSESSION_CONTRACT["execution_interface_repair_basis"]
    for evidence_role in (
        "approved_round_46_review_record",
        "implementation_notes",
        "blocker_summary",
    ):
        identity = repair[evidence_role]
        object_bytes[str(identity["git_object_id"])] = source_bytes[
            str(identity["path"])
        ]
    monkeypatch.setattr(
        operational_twin_protocol,
        "_git_object_bytes",
        lambda root, object_id: object_bytes[str(object_id)],
    )
    monkeypatch.setattr(
        operational_twin_protocol,
        "_git_commit_path_bytes",
        lambda root, commit, path: source_bytes[path],
    )
    return source_bytes


def _materialize_candidate_validator_authorities(
    root: Path,
    config: dict[str, object],
) -> None:
    for relative_path, content in _candidate_validator_authority_bytes(config).items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _rebind_candidate_catalog_hash(catalog: dict[str, object]) -> None:
    digest = command_catalog_hash(catalog)
    catalog["command_catalog_hash"] = digest
    for command in catalog["commands"]:
        argv = command["argv"]
        argv[argv.index("--command-catalog-hash") + 1] = digest


def _mutate_candidate_command_catalog(
    catalog: dict[str, object],
    mutation: str,
) -> None:
    row = catalog["commands"][0]
    argv = row["argv"]
    if mutation == "extra_row_field":
        row["unexpected_field"] = "not_allowed"
    elif mutation == "missing_row_field":
        row.pop("wall_time")
    elif mutation == "unknown_flag":
        argv.extend(["--unknown-candidate-flag", "not_allowed"])
    elif mutation == "duplicate_flag":
        argv.extend(["--seed", str(row["seed"])])
    elif mutation == "reordered_pair":
        config_index = argv.index("--config")
        training_index = argv.index("--training-manifest")
        assert training_index == config_index + 2
        argv[config_index : training_index + 2] = (
            argv[training_index : training_index + 2]
            + argv[config_index : config_index + 2]
        )
    elif mutation == "duplicate_trailing_output_root":
        argv.extend(["--output-root", str(row["output_root"])])
    elif mutation == "changed_authority_path":
        replacement = f"{row['execution_authority']}.tampered"
        row["execution_authority"] = replacement
        argv[argv.index("--candidate-execution-authority") + 1] = replacement
    elif mutation == "changed_receipt_path":
        replacement = f"{row['resource_environment_receipt']}.tampered"
        row["resource_environment_receipt"] = replacement
        argv[
            argv.index("--candidate-resource-environment-receipt") + 1
        ] = replacement
    elif mutation == "mutated_value":
        argv[argv.index("--config") + 1] = "Libs/configs/model/tampered.yaml"
    else:
        raise AssertionError(f"Unknown candidate command mutation: {mutation}")


def _assert_nested_checkpoint_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_checkpoint_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_checkpoint_equal(actual_value, expected_value)
        return
    assert actual == expected


def test_protocol_config_freezes_exact_15_assignments() -> None:
    config = _config()
    blockers = validate_protocol_config(config, ROOT)
    manifest = build_assignment_manifest(config, ROOT)
    assert blockers == []
    assert manifest["method_ids"] == [
        "persistence_no_action",
        "linear_or_delta",
        "same_backbone_no_action",
        "deterministic_action_conditioned",
        "full_uncertainty_aware_action_conditioned",
    ]
    assert len(manifest["seeds"]) == 3
    assert manifest["assignment_count"] == 15
    assert len({row["assignment_uid"] for row in manifest["assignments"]}) == 15
    assert len({row["output_root"] for row in manifest["assignments"]}) == 15
    assert 310031 not in manifest["seeds"]


def test_malformed_seed_count_and_hidden_seed_list_fail() -> None:
    config = _config()
    protocol = deepcopy(config["operational_twin_training_protocol"])
    protocol["seeds"] = [1, 2]
    protocol["replacement_seeds"] = [3]
    config["operational_twin_training_protocol"] = protocol
    codes = {row["code"] for row in validate_protocol_config(config, ROOT)}
    assert "seed_policy_mismatch" in codes
    assert "hidden_seed_list_present" in codes


def test_missing_method_and_duplicate_output_path_fail() -> None:
    config = _config()
    protocol = deepcopy(config["operational_twin_training_protocol"])
    protocol["methods"] = protocol["methods"][:-1]
    protocol["output_roots"]["assignment_root"] = protocol["output_roots"]["catalog_root"]
    config["operational_twin_training_protocol"] = protocol
    codes = {row["code"] for row in validate_protocol_config(config, ROOT)}
    assert "method_identity_mismatch" in codes
    assert "assignment_count_mismatch" in codes
    assert "catalog_writer_collision" in codes


def test_nonzero_access_counter_and_forbidden_manifest_path_fail() -> None:
    config = _config()
    protocol = deepcopy(config["operational_twin_training_protocol"])
    protocol["access_counters"]["protected_holdout_access"] = 1
    protocol["manifests"]["training"] = "Input/processed/sealed/training_manifest.json"
    config["operational_twin_training_protocol"] = protocol
    codes = {row["code"] for row in validate_protocol_config(config, ROOT)}
    assert "nonzero_access_counter" in codes
    assert "forbidden_manifest_path" in codes


def test_preflight_rejects_stale_resource_receipt() -> None:
    config = _config()
    manifest = build_assignment_manifest(config, ROOT)
    receipt = _resource_receipt(config)
    receipt["config_hash"] = "stale"
    preflight = {
        "status": "PASS",
        "terminal_decision": "OPERATIONAL_TWIN_TRAINING_PROTOCOL_PREFLIGHT_PASSED",
        "config_hash": "stale",
        "assignment_manifest_hash": manifest["assignment_manifest_hash"],
        "resource_receipt_hash": "stale",
        "input_content_identities": {},
    }
    codes = {row["code"] for row in validate_preflight_authority(config, ROOT, preflight, manifest, receipt)}
    assert "resource_receipt_config_hash_mismatch" in codes
    assert "preflight_config_hash_mismatch" in codes


def test_assignment_preflight_rejects_before_output_creation(tmp_path: Path) -> None:
    config = _config()
    protocol = deepcopy(config["operational_twin_training_protocol"])
    protocol["output_roots"]["assignment_manifest"] = str(tmp_path / "assignment_manifest.json")
    protocol["output_roots"]["resource_receipt"] = str(tmp_path / "resource_receipt.json")
    protocol["output_roots"]["preflight_authority"] = str(tmp_path / "preflight.json")
    config["operational_twin_training_protocol"] = protocol
    manifest = build_assignment_manifest(config, ROOT)
    receipt = _resource_receipt(config)
    open_authority = json.loads(
        operational_twin_protocol.read_repo_bytes(
            ROOT,
            "Output/audits/controlled_world_open_data_qualification.json",
        ).decode("utf-8")
    )
    preflight = build_preflight_authority(config, ROOT, manifest, receipt, open_authority)
    Path(protocol["output_roots"]["assignment_manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    Path(protocol["output_roots"]["resource_receipt"]).write_text(json.dumps(receipt), encoding="utf-8")
    Path(protocol["output_roots"]["preflight_authority"]).write_text(json.dumps(preflight), encoding="utf-8")
    output_root = tmp_path / "wrong_output"
    with pytest.raises(OperationalTwinProtocolError):
        require_assignment_preflight(
            config,
            ROOT,
            assignment_uid="operational_twin__persistence_no_action__seed_360101",
            method_id="persistence_no_action",
            seed=360101,
            output_root=output_root,
        )
    assert not output_root.exists()


def test_candidate_uid_hierarchy_is_deterministic_and_exactly_five_by_three() -> None:
    config, candidate_manifest, assignment_manifest, _ = _candidate_artifacts()
    protocol = config["operational_twin_candidate_lineage_protocol"]
    expected_seeds = [360101, 360211, 360307]

    assert protocol["candidate_lineage_uid"] == compute_candidate_lineage_uid(protocol)
    assert protocol["protocol_uid"] == compute_candidate_protocol_uid(protocol)
    assert protocol["candidate_lineage_uid"] != protocol["protocol_uid"]
    assert build_candidate_lineage_manifest(config) == candidate_manifest
    assert build_candidate_assignment_manifest(config, candidate_manifest) == assignment_manifest
    assert assignment_manifest["method_ids"] == REQUIRED_METHOD_IDS
    assert assignment_manifest["seeds"] == expected_seeds
    assert assignment_manifest["assignment_count"] == 15
    assert assignment_manifest["attempt_slot_count"] == 30

    assignments = assignment_manifest["assignments"]
    assert [
        (row["method_id"], row["seed"])
        for row in assignments
    ] == [
        (method_id, seed)
        for method_id in REQUIRED_METHOD_IDS
        for seed in expected_seeds
    ]
    assignment_uids: list[str] = []
    attempt_uids: list[str] = []
    for assignment in assignments:
        assignment_uid = compute_candidate_assignment_uid(
            protocol,
            assignment["method_id"],
            assignment["seed"],
        )
        assignment_uids.append(assignment_uid)
        assert assignment["candidate_lineage_uid"] == protocol["candidate_lineage_uid"]
        assert assignment["protocol_uid"] == protocol["protocol_uid"]
        assert assignment["assignment_uid"] == assignment_uid
        assert len(assignment["attempts"]) == 2
        initial, recovery = assignment["attempts"]
        assert initial["attempt_kind"] == "initial"
        assert initial["attempt_ordinal"] == 1
        assert initial["conditional"] is False
        assert initial["predecessor_attempt_uid"] is None
        assert recovery["attempt_kind"] == "infrastructure_recovery_01"
        assert recovery["attempt_ordinal"] == 2
        assert recovery["conditional"] is True
        assert recovery["eligible_after_failure_class"] == "infrastructure_interruption"
        assert recovery["predecessor_attempt_uid"] == initial["attempt_uid"]
        for attempt in (initial, recovery):
            expected_attempt_uid = compute_candidate_attempt_uid(
                protocol,
                assignment_uid,
                attempt["attempt_kind"],
            )
            attempt_uids.append(expected_attempt_uid)
            assert attempt["attempt_uid"] == expected_attempt_uid
            assert candidate_identity_payload(assignment, attempt) == {
                "candidate_lineage_uid": protocol["candidate_lineage_uid"],
                "protocol_uid": protocol["protocol_uid"],
                "assignment_uid": assignment_uid,
                "attempt_uid": expected_attempt_uid,
            }
            assert attempt["output_root"].endswith(f"/{expected_attempt_uid}")
            assert attempt["fresh_random_initialization"] is True
            assert attempt["state_resume_allowed"] is False

    assert len(set(assignment_uids)) == 15
    assert len(set(attempt_uids)) == 30
    assert candidate_manifest["assignment_uids"] == assignment_uids
    assert candidate_manifest["attempt_uids"] == attempt_uids
    assert candidate_manifest["required_identity_fields"] == list(CANDIDATE_IDENTITY_FIELDS)


def test_candidate_successor_freeze_binds_rejected_predecessor_and_disjoint_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_config())
    _patch_candidate_validator_authorities(monkeypatch, config)
    protocol = config["operational_twin_candidate_lineage_protocol"]
    supersession = protocol["supersession_contract"]

    assert validate_candidate_protocol_config(config, ROOT) == []
    assert supersession == CANDIDATE_SUPERSESSION_CONTRACT
    assert (
        protocol["candidate_lineage_uid"]
        == CANDIDATE_SUPERSESSION_CONTRACT["predecessor_candidate_lineage_uid"]
    )
    assert protocol["protocol_uid"] == compute_candidate_protocol_uid(protocol)
    assert (
        protocol["protocol_uid"]
        != CANDIDATE_SUPERSESSION_CONTRACT["predecessor_protocol_uid"]
    )

    predecessor_paths = {
        identity["path"]
        for generation in CANDIDATE_SUPERSESSION_CONTRACT[
            "resource_receipt_amendment_basis"
        ]["predecessor_generations"].values()
        for identity in generation["artifacts"].values()
    }
    successor_paths = {
        protocol["output_paths"][field]
        for field in (
            "candidate_lineage_manifest",
            "assignment_manifest",
            "assignment_commands",
            "preflight_authority",
            "preflight_report",
        )
    }
    assert predecessor_paths.isdisjoint(successor_paths)
    assert len(predecessor_paths) == 20
    assert protocol["output_paths"]["protocol_root"].endswith(
        "/candidate_lineage_protocol/execution_interface_guarded"
    )
    assert protocol["schema_version"].endswith("protocol-v3")
    assert protocol["execution_authority_contract"][
        "resource_receipt_schema_version"
    ].endswith("receipt-v3")
    assert protocol["execution_authority_contract"][
        "authority_schema_version"
    ].endswith("authority-v3")
    assert protocol["terminal_policy"]["positive_terminal"] == (
        "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_INTERFACES_REPAIRED"
    )
    assert protocol["terminal_policy"]["negative_terminal"] == (
        "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_INTERFACES_NOT_REPAIRED"
    )
    assert supersession["execution_interface_repair_basis"][
        "operational_boundary"
    ]["scientific_twin_result"] == "NOT_RUN"
    assert all(protocol["protocol_uid"] not in path for path in successor_paths)
    for generation in CANDIDATE_SUPERSESSION_CONTRACT[
        "resource_receipt_amendment_basis"
    ]["predecessor_generations"].values():
        for identity in generation["artifacts"].values():
            assert (
                hashlib.sha256(
                    operational_twin_protocol.read_repo_bytes(
                        ROOT, identity["path"]
                    )
                ).hexdigest()
                == identity["sha256"]
            )

    candidate_manifest = build_candidate_lineage_manifest(config)
    assignment_manifest = build_candidate_assignment_manifest(
        config,
        candidate_manifest,
    )
    command_catalog = build_candidate_commands(
        config,
        candidate_manifest,
        assignment_manifest,
    )
    preflight = operational_twin_protocol.build_candidate_preflight_authority(
        config,
        ROOT,
        candidate_manifest,
        assignment_manifest,
        command_catalog,
    )
    expected_hash = candidate_supersession_contract_hash(protocol)
    assert candidate_manifest["supersession_contract"] == supersession
    assert candidate_manifest["supersession_contract_hash"] == expected_hash
    assert preflight["supersession_contract"] == supersession
    assert preflight["supersession_contract_hash"] == expected_hash


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_contract",
        "extra_contract_field",
        "predecessor_commit_drift",
        "predecessor_protocol_drift",
        "review_decision_drift",
        "artifact_hash_drift",
    ),
)
def test_candidate_supersession_contract_rejects_missing_extra_or_drifted_fields(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = deepcopy(_config())
    protocol = config["operational_twin_candidate_lineage_protocol"]
    supersession = protocol.get("supersession_contract")
    assert isinstance(supersession, dict)
    if mutation == "missing_contract":
        protocol.pop("supersession_contract")
    elif mutation == "extra_contract_field":
        supersession["unexpected_field"] = False
    elif mutation == "predecessor_commit_drift":
        supersession["predecessor_implementation_commit"] = "0" * 40
    elif mutation == "predecessor_protocol_drift":
        supersession["predecessor_protocol_uid"] = "tampered_protocol"
    elif mutation == "review_decision_drift":
        supersession["rejected_review_record"]["decision"] = "approved"
    elif mutation == "artifact_hash_drift":
        supersession["predecessor_artifacts"]["preflight_report"][
            "sha256"
        ] = "0" * 64
    protocol["protocol_uid"] = compute_candidate_protocol_uid(protocol)
    _patch_candidate_validator_authorities(monkeypatch, config)

    codes = {
        row["code"] for row in validate_candidate_protocol_config(config, ROOT)
    }
    assert "candidate_supersession_contract_drift" in codes


def test_candidate_command_catalog_is_structured_hash_bound_and_manifest_complete() -> None:
    config, candidate_manifest, assignment_manifest, catalog = _candidate_artifacts()
    assert validate_candidate_command_catalog(
        config,
        candidate_manifest,
        assignment_manifest,
        catalog,
    ) == []
    assert catalog["command_catalog_hash"] == command_catalog_hash(catalog)
    assert catalog["assignment_count"] == 15
    assert catalog["command_count"] == 30
    assert catalog["initial_command_count"] == 15
    assert catalog["conditional_recovery_command_count"] == 15
    assert [row["attempt_kind"] for row in catalog["commands"][:15]] == ["initial"] * 15
    assert [row["attempt_kind"] for row in catalog["commands"][15:]] == [
        "infrastructure_recovery_01"
    ] * 15

    manifest_attempts = {
        attempt["attempt_uid"]: (assignment, attempt)
        for assignment in assignment_manifest["assignments"]
        for attempt in assignment["attempts"]
    }
    forbidden_flags = {
        "--resume",
        "--resume-input",
        "--history-input",
        "--checkpoint-input",
        "--optimizer-state-input",
        "--model-state-input",
        "--calibration-state-input",
    }
    for index, row in enumerate(catalog["commands"]):
        assignment, attempt = manifest_attempts[row["attempt_uid"]]
        argv = row["argv"]
        assert row["command_index"] == index
        assert set(row) == CANDIDATE_COMMAND_ROW_KEYS
        assert isinstance(argv, list)
        assert all(isinstance(value, str) for value in argv)
        assert argv == expected_candidate_command_argv(
            config,
            candidate_manifest,
            assignment_manifest,
            assignment,
            attempt,
            command_catalog_hash_value=catalog["command_catalog_hash"],
        )
        assert forbidden_flags.isdisjoint(argv)


def test_candidate_command_catalog_rejects_tampered_hashes_rows_indexes_and_roots() -> None:
    config, candidate_manifest, assignment_manifest, catalog = _candidate_artifacts()

    stale_hash = deepcopy(catalog)
    stale_hash["command_catalog_hash"] = "0" * 64
    codes = {
        row["code"]
        for row in validate_candidate_command_catalog(
            config,
            candidate_manifest,
            assignment_manifest,
            stale_hash,
        )
    }
    assert "candidate_command_catalog_hash_mismatch" in codes

    stale_manifest_hash = deepcopy(catalog)
    stale_manifest_hash["candidate_manifest_hash"] = "1" * 64
    _rebind_candidate_catalog_hash(stale_manifest_hash)
    codes = {
        row["code"]
        for row in validate_candidate_command_catalog(
            config,
            candidate_manifest,
            assignment_manifest,
            stale_manifest_hash,
        )
    }
    assert "candidate_command_manifest_hash_mismatch" in codes
    assert "candidate_command_catalog_hash_mismatch" not in codes

    tampered_row = deepcopy(catalog)
    tampered_row["commands"][0]["assignment_uid"] = "tampered_assignment_uid"
    assignment_index = tampered_row["commands"][0]["argv"].index("--assignment-uid") + 1
    tampered_row["commands"][0]["argv"][assignment_index] = "tampered_assignment_uid"
    _rebind_candidate_catalog_hash(tampered_row)
    codes = {
        row["code"]
        for row in validate_candidate_command_catalog(
            config,
            candidate_manifest,
            assignment_manifest,
            tampered_row,
        )
    }
    assert "candidate_command_row_identity_mismatch" in codes
    assert "candidate_command_argv_exact_mismatch" in codes
    assert "candidate_command_catalog_hash_mismatch" not in codes

    tampered_index = deepcopy(catalog)
    tampered_index["commands"][0]["command_index"] = 29
    _rebind_candidate_catalog_hash(tampered_index)
    codes = {
        row["code"]
        for row in validate_candidate_command_catalog(
            config,
            candidate_manifest,
            assignment_manifest,
            tampered_index,
        )
    }
    assert "candidate_command_index_mismatch" in codes

    tampered_root = deepcopy(catalog)
    tampered_root["commands"][0]["output_root"] += "_tampered"
    output_index = tampered_root["commands"][0]["argv"].index("--output-root") + 1
    tampered_root["commands"][0]["argv"][output_index] += "_tampered"
    _rebind_candidate_catalog_hash(tampered_root)
    codes = {
        row["code"]
        for row in validate_candidate_command_catalog(
            config,
            candidate_manifest,
            assignment_manifest,
            tampered_root,
        )
    }
    assert "candidate_command_output_mismatch" in codes
    assert "candidate_command_argv_exact_mismatch" in codes


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("extra_row_field", "candidate_command_row_keyset_mismatch"),
        ("missing_row_field", "candidate_command_row_keyset_mismatch"),
        ("unknown_flag", "candidate_command_argv_exact_mismatch"),
        ("duplicate_flag", "candidate_command_argv_exact_mismatch"),
        ("reordered_pair", "candidate_command_argv_exact_mismatch"),
        ("duplicate_trailing_output_root", "candidate_command_argv_exact_mismatch"),
        ("changed_authority_path", "candidate_command_argv_exact_mismatch"),
        ("changed_receipt_path", "candidate_command_argv_exact_mismatch"),
        ("mutated_value", "candidate_command_argv_exact_mismatch"),
    ],
)
def test_candidate_command_catalog_rejects_noncanonical_rows_after_hash_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_code: str,
) -> None:
    config, candidate_manifest, assignment_manifest, catalog = _candidate_artifacts()
    _mutate_candidate_command_catalog(catalog, mutation)
    _rebind_candidate_catalog_hash(catalog)
    _patch_candidate_validator_authorities(monkeypatch, config)
    preflight = operational_twin_protocol.build_candidate_preflight_authority(
        config,
        ROOT,
        candidate_manifest,
        assignment_manifest,
        catalog,
    )

    assert catalog["command_catalog_hash"] == command_catalog_hash(catalog)
    assert preflight["command_catalog_hash"] == catalog["command_catalog_hash"]
    codes = {
        row["code"]
        for row in validate_candidate_command_catalog(
            config,
            candidate_manifest,
            assignment_manifest,
            catalog,
        )
    }
    assert "candidate_command_catalog_hash_mismatch" not in codes
    assert expected_code in codes


_CANDIDATE_GATE_METADATA_MUTATIONS = r"""
from datetime import timedelta

gate = prepare_gate()
publish_gate(gate)
gate_baseline = validate_gate(gate)
gate_originals = {gate_path: gate_path.read_bytes() for gate_path in (gate.receipt_path, gate.authority_path)}
gate_receipt = json.loads(gate_originals[gate.receipt_path])
gate_authority = json.loads(gate_originals[gate.authority_path])
assert gate_baseline["resource_receipt_payload"] == gate_receipt
assert gate_baseline["execution_authority"]["authority_uid"] == gate_authority["authority_uid"]
gate_commands = gate.context["command_catalog"]["commands"]
assert len(gate_commands) == 30 and len({row["output_root"] for row in gate_commands}) == 30
assert all(not (root / row["output_root"]).exists() for row in gate_commands)
gate_baseline_count = 1
gate_observed_errors = []
for gate_mutation in gate_mutations:
    gate_kind = gate_mutation["kind"]
    gate_mutated_path = gate.authority_path if gate_kind.startswith("authority") else gate.receipt_path
    if gate_kind == "missing":
        gate_mutated_path = getattr(gate, gate_mutation["field"])
    try:
        if gate_kind == "missing":
            gate_mutated_path.unlink()
        else:
            gate_payload = json.loads(gate_originals[gate_mutated_path])
            if gate_kind == "authority_rejected":
                gate_payload["status"] = "BLOCKED"
                gate_payload["terminal_decision"] = "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_REJECTED"
                gate_payload["authority_uid"] = protocol.compute_candidate_execution_authority_uid(gate_payload)
            elif gate_kind == "authority_uid":
                gate_payload["authority_uid"] = "tampered-authority-uid"
            elif gate_kind == "receipt_uid":
                gate_payload["receipt_uid"] = "tampered-receipt-uid"
            elif gate_kind == "receipt_stale":
                gate_execution_contract = protocol._candidate_execution_contract(resolved.protocol, resolved)
                gate_maximum_age = gate_execution_contract["maximum_resource_receipt_age_seconds"]
                assert type(gate_maximum_age) is int and gate_maximum_age > 0
                gate_expired_capture = gate.timestamp - timedelta(seconds=gate_maximum_age, microseconds=1)
                for gate_time_field in ("created_at", "sinfo_captured_at", "resource_captured_at"):
                    gate_payload[gate_time_field] = gate_expired_capture.isoformat()
                gate_payload["receipt_uid"] = protocol.compute_candidate_resource_receipt_uid(gate_payload)
            else:
                assert gate_kind == "receipt_identity"
                gate_payload[gate_mutation["field"]] = "wrong-" + gate_mutation["field"]
            gate_mutated_path.write_text(json.dumps(gate_payload, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(protocol.OperationalTwinProtocolError) as gate_failure:
            validate_gate(gate)
        assert str(gate_failure.value).split(":", 1)[0] == gate_mutation["expected_code"]
        gate_observed_errors.append(str(gate_failure.value))
    finally:
        gate_mutated_path.write_bytes(gate_originals[gate_mutated_path])
        assert validate_gate(gate) == gate_baseline
        gate_baseline_count += 1
        assert all(gate_path.read_bytes() == gate_bytes for gate_path, gate_bytes in gate_originals.items())
        assert all(not (root / row["output_root"]).exists() for row in gate_commands)
result = {
    "mutations": gate_mutations,
    "observed_errors": gate_observed_errors,
    "positive_baselines": gate_baseline_count,
    "receipt_uid": gate_receipt["receipt_uid"],
    "authority_uid": gate_authority["authority_uid"],
    "source_commit": expected_source_commit,
    "gate_bytes_unchanged": True,
    "declared_attempt_roots": len(gate_commands),
    "attempt_roots": 0,
}
"""


@pytest.mark.parametrize(
    "method_id",
    ["persistence_no_action", "linear_or_delta"],
)
def test_candidate_execution_gate_accepts_only_exact_cpu_and_gpu_placements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_id: str,
) -> None:
    fixture = _candidate_execution_fixture(tmp_path, monkeypatch, method_id)
    context = _require_candidate_execution_fixture(tmp_path, fixture)

    assert context["execution_authority"]["authority_uid"] == fixture[
        "authority"
    ]["authority_uid"]
    assert context["resource_environment_receipt"]["receipt_uid"] == fixture[
        "receipt"
    ]["receipt_uid"]
    assert context["resource_environment_receipt"][
        "resource_allocation"
    ] == fixture["assignment"]["resource_requirements"]
    assert fixture["config"]["operational_twin_candidate_lineage_protocol"][
        "closed_flags"
    ]["training_execution_authorized"] is False
    assert fixture["config"]["operational_twin_candidate_lineage_protocol"][
        "execution_authority_contract"
    ]["authority_bound_in_this_round"] is False


@pytest.mark.parametrize(
    ("missing_path", "error_code"),
    [
        ("authority_path", "candidate_execution_authority_missing"),
        ("receipt_path", "candidate_resource_receipt_missing"),
    ],
)
def test_candidate_execution_gate_rejects_missing_authority_or_receipt(
    isolated_execution_ready, training_protocol_call_witness, tmp_path: Path, missing_path: str, error_code: str,
) -> None:
    mutations = [{"kind": "missing", "field": missing_path, "expected_code": error_code}]
    result = _run_candidate_gate_fixture(
        isolated_execution_ready, tmp_path,
        "gate_mutations = " + repr(mutations) + "\n" + _CANDIDATE_GATE_METADATA_MUTATIONS,
        stage="training protocol gate metadata",
        test_call_witness=training_protocol_call_witness,
    )
    assert result["mutations"] == mutations
    assert [error.split(":", 1)[0] for error in result["observed_errors"]] == [error_code]
    assert result["positive_baselines"] == 2


def test_candidate_execution_gate_rejects_rejected_and_tampered_authority(
    isolated_execution_ready, training_protocol_call_witness, tmp_path: Path,
) -> None:
    mutations = [
        {"kind": "authority_rejected", "expected_code": "candidate_execution_authority_not_approved"},
        {"kind": "authority_uid", "expected_code": "candidate_execution_authority_uid_mismatch"},
    ]
    result = _run_candidate_gate_fixture(
        isolated_execution_ready, tmp_path,
        "gate_mutations = " + repr(mutations) + "\n" + _CANDIDATE_GATE_METADATA_MUTATIONS,
        stage="training protocol gate metadata",
        test_call_witness=training_protocol_call_witness,
    )
    assert result["mutations"] == mutations
    assert [error.split(":", 1)[0] for error in result["observed_errors"]] == [row["expected_code"] for row in mutations]
    assert result["positive_baselines"] == 3


def test_candidate_execution_gate_rejects_tampered_and_stale_receipt(
    isolated_execution_ready, training_protocol_call_witness, tmp_path: Path,
) -> None:
    mutations = [
        {"kind": "receipt_uid", "expected_code": "candidate_resource_receipt_uid_mismatch"},
        {"kind": "receipt_stale", "expected_code": "candidate_resource_receipt_stale"},
    ]
    result = _run_candidate_gate_fixture(
        isolated_execution_ready, tmp_path,
        "gate_mutations = " + repr(mutations) + "\n" + _CANDIDATE_GATE_METADATA_MUTATIONS,
        stage="training protocol gate metadata",
        test_call_witness=training_protocol_call_witness,
    )
    assert result["mutations"] == mutations
    assert [error.split(":", 1)[0] for error in result["observed_errors"]] == [row["expected_code"] for row in mutations]
    assert result["positive_baselines"] == 3


def test_candidate_execution_gate_rejects_stale_review_and_source_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review_root = tmp_path / "review"
    review_fixture = _candidate_execution_fixture(
        review_root,
        monkeypatch,
        "persistence_no_action",
    )
    contract = review_fixture["config"][
        "operational_twin_candidate_lineage_protocol"
    ]["execution_authority_contract"]
    review_path = review_root / contract["approved_protocol_review_path"]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["reviewed_head"] = "stale-review-head"
    review_path.write_text(
        json.dumps(review, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_execution_review_identity_mismatch",
    ):
        _require_candidate_execution_fixture(review_root, review_fixture)

    source_root = tmp_path / "source"
    source_fixture = _candidate_execution_fixture(
        source_root,
        monkeypatch,
        "persistence_no_action",
    )
    source_fixture["source_commit"] = "different-current-source"
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_resource_receipt_source_commit_mismatch",
    ):
        _require_candidate_execution_fixture(source_root, source_fixture)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_lineage_uid",
        "protocol_uid",
        "candidate_manifest_hash",
        "assignment_manifest_hash",
        "command_catalog_hash",
        "assignment_uid",
        "attempt_uid",
        "output_root",
    ],
)
def test_candidate_execution_gate_rejects_every_runtime_identity_drift(
    isolated_execution_ready, training_protocol_call_witness, tmp_path: Path, field: str,
) -> None:
    mutations = [{"kind": "receipt_identity", "field": field, "expected_code": "candidate_resource_receipt_identity_mismatch"}]
    result = _run_candidate_gate_fixture(
        isolated_execution_ready, tmp_path,
        "gate_mutations = " + repr(mutations) + "\n" + _CANDIDATE_GATE_METADATA_MUTATIONS,
        stage="training protocol gate metadata",
        test_call_witness=training_protocol_call_witness,
    )
    assert result["mutations"] == mutations
    assert [error.split(":", 1)[0] for error in result["observed_errors"]] == ["candidate_resource_receipt_identity_mismatch"]
    assert result["positive_baselines"] == 2


@pytest.mark.parametrize(
    ("method_id", "old", "new", "error_code"),
    [
        (
            "persistence_no_action",
            "NodeList=cpu04",
            "NodeList=cpu03",
            "candidate_resource_receipt_resource_mismatch",
        ),
        (
            "persistence_no_action",
            "Partition=cpu",
            "Partition=gpu1",
            "candidate_resource_receipt_resource_mismatch",
        ),
        (
            "persistence_no_action",
            "CPUs/Task=2",
            "CPUs/Task=3",
            "candidate_resource_receipt_resource_mismatch",
        ),
        (
            "persistence_no_action",
            "MinMemoryNode=8G",
            "MinMemoryNode=7G",
            "candidate_resource_receipt_resource_mismatch",
        ),
        (
            "linear_or_delta",
            "TresPerNode=gres:gpu:1",
            "TresPerNode=gres:gpu:2",
            "candidate_resource_receipt_gres_mismatch",
        ),
        (
            "persistence_no_action",
            "OverSubscribe=OK",
            "OverSubscribe=NO",
            "candidate_resource_receipt_exclusivity_mismatch",
        ),
    ],
)
def test_candidate_execution_gate_rejects_wrong_frozen_resource_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_id: str,
    old: str,
    new: str,
    error_code: str,
) -> None:
    fixture = _candidate_execution_fixture(tmp_path, monkeypatch, method_id)
    receipt = deepcopy(fixture["receipt"])
    assert old in receipt["resource_stdout"]
    receipt["resource_stdout"] = receipt["resource_stdout"].replace(old, new)
    receipt["resource_stdout_sha256"] = sha256_text(
        receipt["resource_stdout"]
    )
    receipt["receipt_uid"] = compute_candidate_resource_receipt_uid(receipt)
    fixture["receipt_path"].write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(OperationalTwinProtocolError, match=error_code):
        _require_candidate_execution_fixture(tmp_path, fixture)


@pytest.mark.parametrize(
    ("environment_name", "wrong_value"),
    [
        ("SLURM_JOB_NODELIST", "cpu03"),
        ("SLURM_JOB_PARTITION", "gpu1"),
        ("SLURM_CPUS_PER_TASK", "3"),
        ("SLURM_MEM_PER_NODE", "4096"),
        ("SLURM_JOB_GRES", "gpu:1"),
    ],
)
def test_candidate_execution_gate_rejects_live_resource_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    wrong_value: str,
) -> None:
    fixture = _candidate_execution_fixture(
        tmp_path,
        monkeypatch,
        "persistence_no_action",
    )
    monkeypatch.setenv(environment_name, wrong_value)
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_resource_receipt_runtime_mismatch",
    ):
        _require_candidate_execution_fixture(tmp_path, fixture)


def test_candidate_config_rejects_nonfresh_history_method_seed_access_retry_and_path_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_config())
    _patch_candidate_validator_authorities(monkeypatch, config)
    assert validate_candidate_protocol_config(config, ROOT) == []

    protocol = config["operational_twin_candidate_lineage_protocol"]
    protocol["parent_candidate_lineage_uid"] = "historical_candidate"
    protocol["initialization"]["initialization_mode"] = "resume"
    protocol["initialization"]["resume_allowed"] = True
    protocol["initialization"]["history_input"] = "historical_metrics.jsonl"
    protocol["scientific_contract"]["methods"] = protocol["scientific_contract"]["methods"][:-1]
    protocol["scientific_contract"]["seeds"] = [360101, 360211, 999999]
    protocol["access_counters"]["protected_holdout_access"] = 1
    protocol["execution_contract"]["attempt_slots"].append(
        {
            "attempt_kind": "infrastructure_recovery_02",
            "attempt_ordinal": 3,
            "conditional": True,
            "eligible_after_failure_class": "infrastructure_interruption",
        }
    )
    protocol["output_paths"]["assignment_commands"] = (
        tmp_path / "protected" / "commands.json"
    ).as_posix()
    codes = {row["code"] for row in validate_candidate_protocol_config(config, ROOT)}
    assert {
        "candidate_parent_lineage_not_null",
        "candidate_initialization_not_fresh",
        "candidate_state_reuse_open",
        "candidate_forbidden_state_input",
        "candidate_scientific_contract_drift",
        "candidate_method_identity_mismatch",
        "candidate_seed_identity_mismatch",
        "candidate_nonzero_access_counter",
        "candidate_attempt_slots_invalid",
        "candidate_output_contract_drift",
        "candidate_output_path_invalid",
    } <= codes


def test_candidate_schema_rejects_missing_extra_and_tampered_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_config())
    _patch_candidate_validator_authorities(monkeypatch, config)
    assert validate_candidate_protocol_config(config, ROOT) == []

    schema_key = operational_twin_protocol.CANDIDATE_SCHEMA_KEY
    cases: list[tuple[dict[str, object], set[str]]] = []

    missing_schema = deepcopy(config)
    missing_schema.pop(schema_key)
    cases.append((missing_schema, {"missing_candidate_protocol_schema"}))

    missing_top_level = deepcopy(config)
    missing_top_level["operational_twin_candidate_lineage_protocol"].pop(
        "identity_policy"
    )
    cases.append(
        (
            missing_top_level,
            {"candidate_protocol_top_level_fields_mismatch"},
        )
    )

    extra_top_level = deepcopy(config)
    extra_top_level["operational_twin_candidate_lineage_protocol"][
        "unexpected_protocol_field"
    ] = None
    cases.append(
        (
            extra_top_level,
            {"candidate_protocol_top_level_fields_mismatch"},
        )
    )

    missing_nested_field = deepcopy(config)
    missing_nested_field["operational_twin_candidate_lineage_protocol"][
        "initialization"
    ].pop("checkpoint_input")
    cases.append((missing_nested_field, {"candidate_schema_fields_mismatch"}))

    extra_nested_field = deepcopy(config)
    extra_nested_field["operational_twin_candidate_lineage_protocol"][
        "initialization"
    ]["unexpected_state_input"] = None
    cases.append((extra_nested_field, {"candidate_schema_fields_mismatch"}))

    tampered_schema_field = deepcopy(config)
    tampered_schema_field[schema_key]["protocol_schema_version"] = (
        "tampered-candidate-protocol-version"
    )
    cases.append(
        (
            tampered_schema_field,
            {"candidate_protocol_schema_version_mismatch"},
        )
    )

    for candidate_config, expected_codes in cases:
        codes = {
            row["code"]
            for row in validate_candidate_protocol_config(candidate_config, ROOT)
        }
        assert expected_codes <= codes


def test_candidate_runtime_forbidden_state_inputs_fail_before_artifact_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, candidate_manifest, assignment_manifest, catalog = _candidate_runtime_artifacts(tmp_path)
    assignment = assignment_manifest["assignments"][0]
    attempt = assignment["attempts"][0]
    kwargs = _candidate_runtime_kwargs(
        candidate_manifest,
        assignment_manifest,
        catalog,
        assignment,
        attempt,
    )

    def fail_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("candidate artifacts must not be loaded after forbidden state input")

    monkeypatch.setattr(
        operational_twin_protocol,
        "load_candidate_protocol_artifacts",
        fail_load,
    )
    for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS:
        state_inputs = {
            name: None for name in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS
        }
        state_inputs[field] = True if field == "resume_input" else "forbidden_state.pt"
        kwargs["state_inputs"] = state_inputs
        with pytest.raises(
            OperationalTwinProtocolError,
            match=f"candidate_forbidden_state_input: {field}",
        ):
            require_candidate_assignment_preflight(config, tmp_path, **kwargs)
    assert not (tmp_path / attempt["output_root"]).exists()


@pytest.mark.parametrize(
    ("field", "bad_value", "error_code"),
    [
        ("candidate_lineage_uid", "tampered_lineage", "candidate_runtime_identity_mismatch"),
        ("protocol_uid", "tampered_protocol", "candidate_runtime_identity_mismatch"),
        ("assignment_uid", "tampered_assignment", "candidate_assignment_uid_invalid"),
        ("attempt_uid", "tampered_attempt", "candidate_attempt_uid_invalid"),
        ("method_id", "linear_or_delta", "candidate_assignment_identity_mismatch"),
        ("seed", 999999, "candidate_assignment_identity_mismatch"),
        ("candidate_manifest_hash", "0" * 64, "candidate_runtime_identity_mismatch"),
        ("assignment_manifest_hash", "1" * 64, "candidate_runtime_identity_mismatch"),
        ("command_catalog_hash_value", "2" * 64, "candidate_runtime_identity_mismatch"),
        ("preflight_uid", "tampered_preflight", "candidate_runtime_identity_mismatch"),
        ("output_root", "wrong_attempt_root", "candidate_attempt_output_path_invalid"),
    ],
)
def test_candidate_runtime_rejects_tampered_ids_hashes_method_seed_and_root(
    tmp_path: Path,
    isolated_execution_ready,
    training_protocol_call_witness,
    field: str,
    bad_value: object,
    error_code: str,
) -> None:
    case_kind = "mutation"
    pure_helper = training_protocol_call_witness.kwargs_helper_source(_candidate_runtime_kwargs)
    source = (
        "CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS = protocol.CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS\n"
        + pure_helper
        + "field = " + repr(field) + "\nbad_value = " + repr(bad_value)
        + "\nerror_code = " + repr(error_code) + "\ncase_kind = " + repr(case_kind) + "\n"
        + r"""
context = runtime.resolve_candidate_attempt_context(config, root, 0)
assert context["command"]["command_index"] == 0
assignment, attempt = context["assignment"], context["attempt"]
assert assignment["method_id"] == "persistence_no_action"
assert attempt["attempt_kind"] == "initial" and attempt["conditional"] is False
assert context["command"]["assignment_uid"] == assignment["assignment_uid"]
assert context["command"]["attempt_uid"] == attempt["attempt_uid"]
assert context["command"]["output_root"] == attempt["output_root"]
metadata_paths = sorted({row["path"] for row in context["execution"]["frozen_artifacts"].values()})
assert set(metadata_paths).issubset(_ce_metadata)
metadata_before = {path: (root / path).read_bytes() for path in metadata_paths}
declared_roots = [row["output_root"] for item in context["assignment_manifest"]["assignments"] for row in item["attempts"]]
assert all(not (root / path).exists() for path in declared_roots)
kwargs = _candidate_runtime_kwargs(context["candidate_manifest"], context["assignment_manifest"],
                                   context["command_catalog"], assignment, attempt)
baseline = protocol.require_candidate_assignment_preflight(
    config, root, **kwargs, resolved_context=context["resolved_protocol"])
assert baseline["assignment"] == assignment and baseline["attempt"] == attempt
assert baseline["candidate_manifest"] == context["candidate_manifest"]
assert baseline["assignment_manifest"] == context["assignment_manifest"]
assert baseline["command_catalog"] == context["command_catalog"]
assert baseline["preflight"] == context["preflight"]
if case_kind == "valid_nonmanifest":
    expected_errors = ["candidate_attempt_output_root_mismatch: Attempt output root differs from the manifest."]
elif field in {"candidate_lineage_uid", "protocol_uid"}:
    expected_errors = [
        f"candidate_runtime_identity_mismatch: Runtime {field} drifted.",
        "candidate_runtime_identity_mismatch: Manifest assignment and attempt identities differ from runtime arguments.",
    ]
elif field in {"candidate_manifest_hash", "assignment_manifest_hash", "command_catalog_hash_value", "preflight_uid"}:
    message_field = "command_catalog_hash" if field == "command_catalog_hash_value" else field
    expected_errors = [f"candidate_runtime_identity_mismatch: Runtime {message_field} drifted."]
elif field == "assignment_uid":
    expected_errors = ["candidate_assignment_uid_invalid: Assignment UID is not present exactly once."]
elif field == "attempt_uid":
    expected_errors = ["candidate_attempt_uid_invalid: Attempt UID is not manifest-provided exactly once."]
elif field in {"method_id", "seed"}:
    expected_errors = ["candidate_assignment_identity_mismatch: Method or seed differs from the frozen assignment."]
else:
    assert field == "output_root" and bad_value == "wrong_attempt_root"
    expected_errors = [
        "candidate_attempt_output_path_invalid: candidate_attempt_path_outside_root: "
        "candidate attempt roots must be lexically contained under "
        + protocol.CANDIDATE_LINEAGE_ROOT.as_posix() + "."
    ]
changed = deepcopy(kwargs)
changed[field] = bad_value
with pytest.raises(protocol.OperationalTwinProtocolError, match=error_code) as rejection:
    protocol.require_candidate_assignment_preflight(
        config, root, **changed, resolved_context=context["resolved_protocol"])
actual_errors = str(rejection.value).split("; ")
assert Counter(actual_errors) == Counter(expected_errors), (actual_errors, expected_errors)
restored_context = runtime.resolve_candidate_attempt_context(config, root, 0)
restored = protocol.require_candidate_assignment_preflight(
    config, root, **kwargs, resolved_context=restored_context["resolved_protocol"])
for key in ("assignment", "attempt", "candidate_manifest", "assignment_manifest", "command_catalog", "preflight"):
    assert restored[key] == baseline[key], key
assert {path: (root / path).read_bytes() for path in metadata_paths} == metadata_before
assert all(not (root / path).exists() for path in declared_roots)
assert not (root / attempt["output_root"]).exists()
if field == "output_root":
    assert not (root / bad_value).exists()
result = {"runtime_identity_case": {
    "field": field, "bad_value": bad_value, "errors": actual_errors,
    "assignment_uid": assignment["assignment_uid"], "attempt_uid": attempt["attempt_uid"],
    "initial_command_index": context["command"]["command_index"],
    "baseline_and_restored": True, "metadata_unchanged": True,
    "declared_roots_absent": len(declared_roots),
}}
"""
    )
    result = _run_candidate_gate_fixture(
        isolated_execution_ready, tmp_path, source,
        stage="training protocol runtime identity", test_call_witness=training_protocol_call_witness,
    )
    assert result["runtime_identity_case"]["field"] == field
    assert result["runtime_identity_case"]["bad_value"] == bad_value
    assert result["runtime_identity_case"]["baseline_and_restored"] is True



def test_candidate_runtime_rejects_valid_nonmanifest_attempt_root(
    tmp_path: Path,
    isolated_execution_ready,
    training_protocol_call_witness,
) -> None:
    wrong_output_root = (
        "Output/results/operational_twin/candidate_lineages/"
        "valid_nonmanifest_attempt_root"
    )
    field = "output_root"
    bad_value = wrong_output_root
    error_code = "candidate_attempt_output_root_mismatch"
    case_kind = "valid_nonmanifest"
    pure_helper = training_protocol_call_witness.kwargs_helper_source(_candidate_runtime_kwargs)
    source = (
        "CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS = protocol.CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS\n"
        + pure_helper
        + "field = " + repr(field) + "\nbad_value = " + repr(bad_value)
        + "\nerror_code = " + repr(error_code) + "\ncase_kind = " + repr(case_kind) + "\n"
        + r"""
context = runtime.resolve_candidate_attempt_context(config, root, 0)
assert context["command"]["command_index"] == 0
assignment, attempt = context["assignment"], context["attempt"]
assert assignment["method_id"] == "persistence_no_action"
assert attempt["attempt_kind"] == "initial" and attempt["conditional"] is False
assert context["command"]["assignment_uid"] == assignment["assignment_uid"]
assert context["command"]["attempt_uid"] == attempt["attempt_uid"]
assert context["command"]["output_root"] == attempt["output_root"]
metadata_paths = sorted({row["path"] for row in context["execution"]["frozen_artifacts"].values()})
assert set(metadata_paths).issubset(_ce_metadata)
metadata_before = {path: (root / path).read_bytes() for path in metadata_paths}
declared_roots = [row["output_root"] for item in context["assignment_manifest"]["assignments"] for row in item["attempts"]]
assert all(not (root / path).exists() for path in declared_roots)
kwargs = _candidate_runtime_kwargs(context["candidate_manifest"], context["assignment_manifest"],
                                   context["command_catalog"], assignment, attempt)
baseline = protocol.require_candidate_assignment_preflight(
    config, root, **kwargs, resolved_context=context["resolved_protocol"])
assert baseline["assignment"] == assignment and baseline["attempt"] == attempt
assert baseline["candidate_manifest"] == context["candidate_manifest"]
assert baseline["assignment_manifest"] == context["assignment_manifest"]
assert baseline["command_catalog"] == context["command_catalog"]
assert baseline["preflight"] == context["preflight"]
if case_kind == "valid_nonmanifest":
    expected_errors = ["candidate_attempt_output_root_mismatch: Attempt output root differs from the manifest."]
elif field in {"candidate_lineage_uid", "protocol_uid"}:
    expected_errors = [
        f"candidate_runtime_identity_mismatch: Runtime {field} drifted.",
        "candidate_runtime_identity_mismatch: Manifest assignment and attempt identities differ from runtime arguments.",
    ]
elif field in {"candidate_manifest_hash", "assignment_manifest_hash", "command_catalog_hash_value", "preflight_uid"}:
    message_field = "command_catalog_hash" if field == "command_catalog_hash_value" else field
    expected_errors = [f"candidate_runtime_identity_mismatch: Runtime {message_field} drifted."]
elif field == "assignment_uid":
    expected_errors = ["candidate_assignment_uid_invalid: Assignment UID is not present exactly once."]
elif field == "attempt_uid":
    expected_errors = ["candidate_attempt_uid_invalid: Attempt UID is not manifest-provided exactly once."]
elif field in {"method_id", "seed"}:
    expected_errors = ["candidate_assignment_identity_mismatch: Method or seed differs from the frozen assignment."]
else:
    assert field == "output_root" and bad_value == "wrong_attempt_root"
    expected_errors = [
        "candidate_attempt_output_path_invalid: candidate_attempt_path_outside_root: "
        "candidate attempt roots must be lexically contained under "
        + protocol.CANDIDATE_LINEAGE_ROOT.as_posix() + "."
    ]
changed = deepcopy(kwargs)
changed[field] = bad_value
with pytest.raises(protocol.OperationalTwinProtocolError, match=error_code) as rejection:
    protocol.require_candidate_assignment_preflight(
        config, root, **changed, resolved_context=context["resolved_protocol"])
actual_errors = str(rejection.value).split("; ")
assert Counter(actual_errors) == Counter(expected_errors), (actual_errors, expected_errors)
restored_context = runtime.resolve_candidate_attempt_context(config, root, 0)
restored = protocol.require_candidate_assignment_preflight(
    config, root, **kwargs, resolved_context=restored_context["resolved_protocol"])
for key in ("assignment", "attempt", "candidate_manifest", "assignment_manifest", "command_catalog", "preflight"):
    assert restored[key] == baseline[key], key
assert {path: (root / path).read_bytes() for path in metadata_paths} == metadata_before
assert all(not (root / path).exists() for path in declared_roots)
assert not (root / attempt["output_root"]).exists()
if field == "output_root":
    assert not (root / bad_value).exists()
result = {"runtime_identity_case": {
    "field": field, "bad_value": bad_value, "errors": actual_errors,
    "assignment_uid": assignment["assignment_uid"], "attempt_uid": attempt["attempt_uid"],
    "initial_command_index": context["command"]["command_index"],
    "baseline_and_restored": True, "metadata_unchanged": True,
    "declared_roots_absent": len(declared_roots),
}}
"""
    )
    result = _run_candidate_gate_fixture(
        isolated_execution_ready, tmp_path, source,
        stage="training protocol runtime identity", test_call_witness=training_protocol_call_witness,
    )
    assert result["runtime_identity_case"]["field"] == field
    assert result["runtime_identity_case"]["bad_value"] == bad_value
    assert result["runtime_identity_case"]["baseline_and_restored"] is True



def test_candidate_recovery_is_capped_manifest_bound_and_never_reuses_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _candidate_recovery_fixture(tmp_path, monkeypatch)
    config = fixture["config"]
    protocol = fixture["protocol"]
    candidate_manifest = fixture["candidate_manifest"]
    assignment_manifest = fixture["assignment_manifest"]
    catalog = fixture["command_catalog"]
    preflight = fixture["preflight"]
    assignment = fixture["assignment"]
    initial = fixture["initial"]
    recovery = fixture["recovery"]
    recovery_kwargs = fixture["recovery_kwargs"]
    assert protocol["execution_contract"]["maximum_attempts_per_assignment"] == 2
    assert protocol["execution_contract"]["maximum_infrastructure_recovery_attempts_per_assignment"] == 1
    assert len(assignment["attempts"]) == 2

    initial_context = require_candidate_assignment_preflight(
        config,
        tmp_path,
        **_candidate_runtime_kwargs(
            candidate_manifest,
            assignment_manifest,
            catalog,
            assignment,
            initial,
        ),
    )
    assert initial_context["attempt"]["attempt_uid"] == initial["attempt_uid"]
    assert initial_context["initialization"] == candidate_fresh_initialization_receipt(protocol)

    third_attempt_kwargs = _candidate_runtime_kwargs(
        candidate_manifest,
        assignment_manifest,
        catalog,
        assignment,
        recovery,
    )
    third_attempt_kwargs["attempt_uid"] = compute_candidate_attempt_uid(
        protocol,
        assignment["assignment_uid"],
        "infrastructure_recovery_02",
    )
    with pytest.raises(OperationalTwinProtocolError, match="candidate_attempt_uid_invalid"):
        require_candidate_assignment_preflight(
            config,
            tmp_path,
            **third_attempt_kwargs,
        )

    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_recovery_predecessor_status_missing",
    ):
        require_candidate_assignment_preflight(config, tmp_path, **recovery_kwargs)

    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_resource_receipt_stale",
    ):
        require_candidate_execution_authority(
            config,
            tmp_path,
            assignment=assignment,
            attempt=initial,
            candidate_manifest=candidate_manifest,
            assignment_manifest=assignment_manifest,
            command_catalog=catalog,
            preflight=preflight,
            execution_authority_path=fixture["authority_path"],
            resource_environment_receipt_path=fixture["receipt_path"],
            cuda_available=False,
            cuda_device_count=0,
            torch_version="test-torch",
            current_source_commit=fixture["source_commit"],
            now=datetime(2026, 9, 8, 13, 0, tzinfo=timezone.utc),
            validate_current_runtime=False,
        )

    predecessor_status = fixture["predecessor_status"]
    status_path = fixture["status_path"]
    _write_candidate_recovery_status(status_path, predecessor_status)
    recovery_context = require_candidate_assignment_preflight(
        config,
        tmp_path,
        **recovery_kwargs,
    )
    assert recovery_context["attempt"]["attempt_uid"] == recovery["attempt_uid"]
    assert recovery_context["attempt"]["predecessor_attempt_uid"] == initial["attempt_uid"]
    assert recovery_context["attempt"]["state_resume_allowed"] is False
    assert recovery_context["initialization"]["historical_state_use_allowed"] is False
    assert all(
        recovery_context["initialization"][field] is None
        for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS
    )

    ineligible_status = deepcopy(predecessor_status)
    ineligible_status["failure_class"] = "runtime_failure"
    _write_candidate_recovery_status(status_path, ineligible_status)
    with pytest.raises(OperationalTwinProtocolError, match="candidate_recovery_not_eligible"):
        require_candidate_assignment_preflight(config, tmp_path, **recovery_kwargs)
    assert not (tmp_path / recovery["output_root"]).exists()


@pytest.mark.parametrize(
    ("binding_field", "operation", "key", "expected_code"),
    [
        (
            "candidate_execution_authority",
            "remove",
            "authority_uid",
            "candidate_recovery_predecessor_authority_binding_invalid",
        ),
        (
            "candidate_execution_authority",
            "add",
            "unexpected_authority_field",
            "candidate_recovery_predecessor_authority_binding_invalid",
        ),
        (
            "candidate_resource_environment_receipt",
            "remove",
            "receipt_uid",
            "candidate_recovery_predecessor_resource_receipt_binding_invalid",
        ),
        (
            "candidate_resource_environment_receipt",
            "add",
            "unexpected_receipt_field",
            "candidate_recovery_predecessor_resource_receipt_binding_invalid",
        ),
    ],
    ids=(
        "authority-missing-key",
        "authority-extra-key",
        "receipt-missing-key",
        "receipt-extra-key",
    ),
)
def test_candidate_recovery_rejects_nonexact_retained_binding_keysets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding_field: str,
    operation: str,
    key: str,
    expected_code: str,
) -> None:
    fixture = _candidate_recovery_fixture(tmp_path, monkeypatch)
    predecessor_status = deepcopy(fixture["predecessor_status"])
    binding = predecessor_status[binding_field]
    assert isinstance(binding, dict)
    if operation == "remove":
        binding.pop(key)
    else:
        binding[key] = "unexpected"
    expected_keys = (
        CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS
        if binding_field == "candidate_execution_authority"
        else CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS
    )
    assert set(binding) != expected_keys
    assert set(predecessor_status) == CANDIDATE_FAILED_TERMINAL_KEYS
    _write_candidate_recovery_status(fixture["status_path"], predecessor_status)

    with pytest.raises(OperationalTwinProtocolError, match=expected_code):
        require_candidate_assignment_preflight(
            fixture["config"],
            tmp_path,
            **fixture["recovery_kwargs"],
        )
    assert not (tmp_path / fixture["recovery"]["output_root"]).exists()


@pytest.mark.parametrize(
    ("field_path", "bad_value", "expected_code"),
    [
        (
            ("candidate_execution_authority", "authority_uid"),
            "wrong-authority-uid",
            "candidate_recovery_predecessor_authority_binding_mismatch",
        ),
        (
            ("candidate_resource_environment_receipt", "receipt_uid"),
            "wrong-receipt-uid",
            "candidate_recovery_predecessor_resource_receipt_binding_mismatch",
        ),
        (
            ("candidate_execution_authority", "sha256"),
            "0" * 64,
            "candidate_recovery_predecessor_authority_binding_mismatch",
        ),
        (
            ("candidate_resource_environment_receipt", "sha256"),
            "1" * 64,
            "candidate_recovery_predecessor_resource_receipt_binding_mismatch",
        ),
        (
            ("candidate_execution_authority", "path"),
            "wrong/authority.json",
            "candidate_recovery_predecessor_artifact_path_mismatch",
        ),
        (
            ("candidate_resource_environment_receipt", "path"),
            "wrong/receipt.json",
            "candidate_recovery_predecessor_artifact_path_mismatch",
        ),
        (
            ("candidate_execution_authority", "source_commit"),
            "wrong-authority-source",
            "candidate_recovery_predecessor_authority_binding_mismatch",
        ),
        (
            ("candidate_resource_environment_receipt", "source_commit"),
            "wrong-receipt-source",
            "candidate_recovery_predecessor_binding_invalid",
        ),
        (
            (
                "candidate_execution_authority",
                "approved_protocol_review_identity",
                "sha256",
            ),
            "2" * 64,
            "candidate_recovery_predecessor_authority_binding_mismatch",
        ),
        (
            ("candidate_lineage_uid",),
            "wrong-candidate-lineage",
            "candidate_recovery_predecessor_identity_mismatch",
        ),
        (
            ("protocol_uid",),
            "wrong-protocol",
            "candidate_recovery_predecessor_identity_mismatch",
        ),
        (
            ("assignment_uid",),
            "wrong-assignment",
            "candidate_recovery_predecessor_identity_mismatch",
        ),
        (
            ("attempt_uid",),
            "wrong-attempt",
            "candidate_recovery_predecessor_identity_mismatch",
        ),
        (
            ("method_id",),
            "linear_or_delta",
            "candidate_recovery_predecessor_identity_mismatch",
        ),
        (
            ("seed",),
            999999,
            "candidate_recovery_predecessor_identity_mismatch",
        ),
        (
            (
                "candidate_resource_environment_receipt",
                "resource_allocation",
                "cpus_per_task",
            ),
            99,
            "candidate_recovery_predecessor_resource_receipt_binding_mismatch",
        ),
        (
            ("output_root",),
            "wrong/output-root",
            "candidate_recovery_predecessor_identity_mismatch",
        ),
        (
            ("fresh_initialization_receipt", "initialization_mode"),
            "resume",
            "candidate_recovery_predecessor_identity_mismatch",
        ),
        (
            ("state_resume_used",),
            True,
            "candidate_recovery_predecessor_identity_mismatch",
        ),
    ],
    ids=(
        "authority-uid",
        "receipt-uid",
        "authority-hash",
        "receipt-hash",
        "authority-path",
        "receipt-path",
        "authority-source",
        "receipt-source",
        "approved-review",
        "candidate-lineage",
        "protocol",
        "assignment",
        "attempt",
        "method",
        "seed",
        "resource-allocation",
        "output-root",
        "fresh-initialization",
        "state-resume",
    ),
)
def test_candidate_recovery_rejects_retained_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    bad_value: object,
    expected_code: str,
) -> None:
    fixture = _candidate_recovery_fixture(tmp_path, monkeypatch)
    predecessor_status = deepcopy(fixture["predecessor_status"])
    target = predecessor_status
    for field in field_path[:-1]:
        nested = target[field]
        assert isinstance(nested, dict)
        target = nested
    target[field_path[-1]] = deepcopy(bad_value)
    assert set(predecessor_status) == CANDIDATE_FAILED_TERMINAL_KEYS
    assert (
        set(predecessor_status["candidate_execution_authority"])
        == CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS
    )
    assert (
        set(predecessor_status["candidate_resource_environment_receipt"])
        == CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS
    )
    _write_candidate_recovery_status(fixture["status_path"], predecessor_status)

    with pytest.raises(OperationalTwinProtocolError, match=expected_code):
        require_candidate_assignment_preflight(
            fixture["config"],
            tmp_path,
            **fixture["recovery_kwargs"],
        )
    assert not (tmp_path / fixture["recovery"]["output_root"]).exists()


def test_candidate_preflight_source_hash_drift_returns_negative_terminal_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_config())
    source_bytes = _patch_candidate_validator_authorities(monkeypatch, config)
    protocol = config["operational_twin_candidate_lineage_protocol"]
    trainer_path = protocol["source_bindings"]["trainer"]
    real_build_preflight = operational_twin_protocol.build_candidate_preflight_authority
    observed: dict[str, object] = {}

    def build_then_tamper_source(
        loaded_config: dict[str, object],
        root: Path,
        candidate_manifest: dict[str, object],
        assignment_manifest: dict[str, object],
        command_catalog: dict[str, object],
    ) -> dict[str, object]:
        preflight = real_build_preflight(
            loaded_config,
            root,
            candidate_manifest,
            assignment_manifest,
            command_catalog,
        )
        frozen_trainer = next(
            row
            for row in preflight["source_identities"]
            if row["path"] == trainer_path
        )
        observed["frozen_trainer_sha256"] = frozen_trainer["sha256"]
        source_bytes[trainer_path] += b"# tampered after preflight construction\n"
        observed["tampered_trainer_sha256"] = sha256_text(
            source_bytes[trainer_path].decode("utf-8")
        )
        return preflight

    publication_calls: list[list[tuple[Path, bytes]]] = []

    def capture_publication(
        paths_and_content: list[tuple[Path, bytes]],
        *,
        protocol_root: Path,
    ) -> str:
        del protocol_root
        publication_calls.append(paths_and_content)
        return "created"

    monkeypatch.setattr(operational_twin_protocol_builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "build_candidate_preflight_authority",
        build_then_tamper_source,
    )
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "_publish_candidate_artifacts",
        capture_publication,
    )
    monkeypatch.setattr(
        operational_twin_protocol_builder.subprocess,
        "check_output",
        lambda *args, **kwargs: "interface-repair-source\n",
    )

    result = operational_twin_protocol_builder._build_candidate_lineage(
        config,
        "2026-09-08",
    )

    assert observed["frozen_trainer_sha256"] != observed["tampered_trainer_sha256"]
    assert result["status"] == "BLOCKED"
    assert result["terminal_decision"] == operational_twin_protocol.CANDIDATE_NEGATIVE_TERMINAL
    assert "publication" not in result
    assert {
        row["code"] for row in result["blockers"]
    } >= {"candidate_preflight_drift"}
    assert publication_calls == []
    output_paths = protocol["output_paths"]
    assert not (tmp_path / output_paths["protocol_root"]).exists()
    assert not (tmp_path / output_paths["preflight_authority"]).exists()
    assert not (tmp_path / output_paths["preflight_report"]).exists()


def test_execution_interface_repair_builder_publishes_complete_successor(
    isolated_execution_ready,
    tmp_path: Path,
) -> None:
    from Libs.tests.test_operational_twin_successor_execution_handoff import (
        _isolated_python,
        _PRECOMPUTE_IMPORTS_FIXTURE,
        _PRECOMPUTE_READ_BOUNDARY_FIXTURE,
        _PRECOMPUTE_STOP_BOUNDARY_FIXTURE,
    )

    config = _config()
    protocol = config["operational_twin_candidate_lineage_protocol"]
    supersession = protocol["supersession_contract"]
    amendment = supersession["resource_receipt_amendment_basis"]
    repair = supersession["execution_interface_repair_basis"]
    identities = [
        supersession["rejected_review_record"],
        *supersession["predecessor_artifacts"].values(),
        *(amendment[role] for role in (
            "approved_round_43_review_record", "approved_round_44_review_record",
            "approved_rev_007_update", "approved_rev_007_update_review",
        )),
        *(identity for generation in amendment["predecessor_generations"].values()
          for identity in generation["artifacts"].values()),
        *(repair[role] for role in (
            "approved_round_46_review_record", "implementation_notes", "blocker_summary",
        )),
    ]
    public_paths = {str(identity["path"]) for identity in identities}
    public_paths.update(protocol["source_bindings"][role] for role in (
        "cpu_receipt_fixture", "gpu_receipt_fixture",
        "array_element_resource_fixture", "array_terminal_accounting_fixture",
    ))
    public_paths.update(
        "orchestrator/rounds/round-47-repair-operational-twin-candidate-execution-interfaces/"
        f"array_element_probe_547789_{index}.log" for index in (0, 1)
    )
    fixture_root = tmp_path / "interface_repair_authority"
    shutil.copytree(isolated_execution_ready["root"], fixture_root)
    for relative in sorted(public_paths):
        content = operational_twin_protocol.read_repo_bytes(ROOT, relative)
        target = fixture_root / relative
        assert not target.is_symlink()
        if target.exists():
            assert target.is_file() and target.read_bytes() == content
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    # Reuse the existing legacy input fixtures inside the real authority context.
    # Their bytes stay in memory; current source and frozen files remain intact.
    module_source = Path(__file__).read_text(encoding="utf-8")
    helper_names = {"_candidate_validator_authority_bytes", "_patch_candidate_validator_authorities"}
    helper_nodes = [node for node in ast.parse(module_source).body
                    if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    assert {node.name for node in helper_nodes} == helper_names
    helper_source = "\n\n".join(ast.get_source_segment(module_source, node) for node in helper_nodes)
    result = _isolated_python(
        fixture_root,
        _PRECOMPUTE_IMPORTS_FIXTURE
        + _PRECOMPUTE_READ_BOUNDARY_FIXTURE
        + _PRECOMPUTE_STOP_BOUNDARY_FIXTURE
        + r"""
from copy import deepcopy
import subprocess
import pytest
import yaml
from Libs.scripts.builders import build_operational_twin_training_protocol as operational_twin_protocol_builder
from Libs.utils.evaluation import operational_twin_training_protocol as operational_twin_protocol
ROOT = root
CANDIDATE_SUPERSESSION_CONTRACT = operational_twin_protocol.CANDIDATE_SUPERSESSION_CONTRACT
sha256_text = operational_twin_protocol.sha256_text
""" + helper_source + r"""
def exercise_interface_repair(config, root):
    tmp_path = root
    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    with pytest.MonkeyPatch.context() as monkeypatch:
        config = deepcopy(config)
        for reader_name, reader in prepared_fixture_readers.items():
            monkeypatch.setattr(operational_twin_protocol, reader_name, reader)
        validation = tmp_path / "validation.json"
        repair = config["operational_twin_candidate_execution"][
            "interface_repair_contract"
        ]
        validation.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "terminal_decision": repair["positive_terminal"],
                    "side_effect_count": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        protocol = config["operational_twin_candidate_lineage_protocol"]
        repair_basis = protocol["supersession_contract"][
            "execution_interface_repair_basis"
        ]
        round_dir = (
            tmp_path
            / "orchestrator/rounds"
            / protocol["execution_authority_contract"]["required_review_round_id"]
        )
        round_dir.mkdir(parents=True, exist_ok=True)
        scheduler_probe_logs = []
        for index in (0, 1):
            target = round_dir / f"array_element_probe_547789_{index}.log"
            assert target.is_file()
            scheduler_probe_logs.append(
                operational_twin_protocol_builder._artifact_identity(tmp_path, target)
            )
        predecessor_protocol_artifacts = []
        for generation_name, generation in protocol["supersession_contract"][
            "resource_receipt_amendment_basis"
        ]["predecessor_generations"].items():
            for role, identity in generation["artifacts"].items():
                predecessor_protocol_artifacts.append(
                    {
                        "generation": generation_name,
                        "role": role,
                        **operational_twin_protocol_builder._artifact_identity(
                            tmp_path,
                            tmp_path / identity["path"],
                        ),
                    }
                )
        predecessor_protocol_artifacts.sort(
            key=lambda row: (row["generation"], row["role"])
        )
        round_46_evidence = {
            role: {
                **operational_twin_protocol_builder._artifact_identity(
                    tmp_path,
                    tmp_path / repair_basis[role]["path"],
                ),
                "git_object_id": repair_basis[role]["git_object_id"],
            }
            for role in (
                "approved_round_46_review_record",
                "implementation_notes",
                "blocker_summary",
            )
        }
        output_paths = protocol["output_paths"]
        pre_change_manifest = {
            "schema_version": "neuronsifter-operational-twin-candidate-interface-prechange-v1",
            "captured_at": "2026-09-12T16:50:00+00:00",
            "source_commit": source_head,
            "candidate_lineage_uid": protocol["candidate_lineage_uid"],
            "predecessor_protocol_artifacts": predecessor_protocol_artifacts,
            "round_46_evidence": round_46_evidence,
            "scheduler_probe_logs": scheduler_probe_logs,
            "successor_paths_absent": sorted(
                output_paths[key]
                for key in (
                    "candidate_lineage_manifest",
                    "assignment_manifest",
                    "assignment_commands",
                    "preflight_authority",
                    "preflight_report",
                    "interface_repair_audit",
                    "interface_repair_report",
                )
            ),
            "operational_boundary": repair_basis["operational_boundary"],
        }
        pre_change_manifest_path = round_dir / "pre_change_identity_manifest.json"
        pre_change_manifest_path.write_text(
            json.dumps(pre_change_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        published: list[tuple[Path, bytes]] = []
        preserved: list[dict[str, object]] = []

        def capture_publication(
            paths_and_content: list[tuple[Path, bytes]],
            *,
            protocol_root: Path,
            preserved_identities: object = (),
        ) -> str:
            del protocol_root
            published.extend(paths_and_content)
            preserved.extend(dict(row) for row in preserved_identities)
            return "created"

        monkeypatch.setattr(operational_twin_protocol_builder, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            operational_twin_protocol_builder,
            "_publish_candidate_artifacts",
            capture_publication,
        )
        source_bindings = protocol["source_bindings"]
        result = operational_twin_protocol_builder._build_candidate_execution_interface_repair(
            config,
            "2026-09-08",
            array_resource_fixture_path=(
                tmp_path / source_bindings["array_element_resource_fixture"]
            ),
            array_terminal_fixture_path=(
                tmp_path / source_bindings["array_terminal_accounting_fixture"]
            ),
            pre_change_identity_manifest_path=pre_change_manifest_path,
            validation_evidence_path=validation,
        )
        assert result["status"] == "PASS", result
        assert result["terminal_decision"] == (
            "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_INTERFACES_REPAIRED"
        )
        assert len(published) == 7
        assert len(preserved) == 26
        audit_path = tmp_path / output_paths["interface_repair_audit"]
        audit_bytes = next(content for path, content in published if path == audit_path)
        audit = json.loads(audit_bytes.decode("utf-8"))
        assert audit["predecessor_file_count"] == 20
        assert audit["interfaces"] == {
            "calibration_authority_evaluation": "PASS",
            "array_element_resource_identity": "PASS",
            "terminal_scheduler_mapping": "PASS",
        }
        assert audit["materialized_candidate_execution_artifact_count"] == 0
        assert audit["candidate_dispatch_authorized"] is False
        assert audit["predecessor_pre_post_match"] is True
        assert audit["side_effect_count"] == 0

        tampered_manifest = deepcopy(pre_change_manifest)
        tampered_manifest["predecessor_protocol_artifacts"][0]["sha256"] = "0" * 64
        pre_change_manifest_path.write_text(
            json.dumps(tampered_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publication_count = len(published)
        blocked = operational_twin_protocol_builder._build_candidate_execution_interface_repair(
            config,
            "2026-09-08",
            array_resource_fixture_path=(
                tmp_path / source_bindings["array_element_resource_fixture"]
            ),
            array_terminal_fixture_path=(
                tmp_path / source_bindings["array_terminal_accounting_fixture"]
            ),
            pre_change_identity_manifest_path=pre_change_manifest_path,
            validation_evidence_path=validation,
        )
        assert blocked["status"] == "BLOCKED"
        assert blocked["terminal_decision"] == (
            "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_INTERFACES_NOT_REPAIRED"
        )
        assert "candidate_interface_repair_prechange_manifest_drift" in {
            row["code"] for row in blocked["blockers"]
        }
        assert len(published) == publication_count
        return {"published_count": len(published), "preserved_count": len(preserved),
                "tamper_rejected": blocked["status"] == "BLOCKED",
                "source_head": source_head}

# This setup reads only the already-bound public metadata. Preserve its identity
# before installing the precompute guard, which forbids unscoped blob reads.
config = load_config_stack(root / "Libs/configs/config_stack.yaml")
reader_names = ("read_repo_json", "read_repo_bytes", "_git_object_bytes", "_git_commit_path_bytes")
with pytest.MonkeyPatch.context() as preparation_patch:
    _patch_candidate_validator_authorities(preparation_patch, config)
    rejected = CANDIDATE_SUPERSESSION_CONTRACT["rejected_review_record"]
    review_bytes = operational_twin_protocol._git_object_bytes(root, rejected["git_object_id"])
    assert hashlib.sha256(review_bytes).hexdigest() == rejected["sha256"]
    framed = b"blob " + str(len(review_bytes)).encode() + b"\0" + review_bytes
    assert hashlib.sha1(framed).hexdigest() == rejected["git_object_id"]
    assert subprocess.check_output(["git", "cat-file", "-t", rejected["git_object_id"]], cwd=root, text=True).strip() == "blob"
    assert int(subprocess.check_output(["git", "cat-file", "-s", rejected["git_object_id"]], cwd=root, text=True)) == len(review_bytes)
    assert json.loads(review_bytes)["decision"] == rejected["decision"] == "rejected"
    prepared_fixture_readers = {name: getattr(operational_twin_protocol, name) for name in reader_names}

phase = "identity_guard"
sys.addaudithook(audit)
prior_trace = sys.gettrace()
sys.settrace(trace)
try:
    resolved = runtime.resolve_candidate_execution_protocol(config, root)
    attempt_roots = {row["output_root"] for row in resolved.command_catalog["commands"]}
    assert len(attempt_roots) == 30
    assert all(not (root / path).exists() for path in attempt_roots)
    dispatch_path = root / resolved.execution["output_paths"]["dispatch_manifest"]
    assert not dispatch_path.exists()
    result = exercise_interface_repair(config, root)
    restored = runtime.resolve_candidate_execution_protocol(config, root)
    assert restored.protocol == resolved.protocol
    assert restored.preflight == resolved.preflight
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == result["source_head"]
    assert all(not (root / path).exists() for path in attempt_roots)
    assert not dispatch_path.exists()
    assert not observed_inputs and not observed_transport
    assert not stops and not guarded_entries
finally:
    sys.settrace(prior_trace)
assert sys.gettrace() is prior_trace
result["current_authority_preserved"] = True
result["guard"] = {"physical_input_reads": len(observed_inputs),
                   "transport_events": len(observed_transport),
                   "scientific_entries": len(stops) + len(guarded_entries),
                   "attempt_slots": len(attempt_roots), "attempt_roots": 0,
                   "dispatch_absent": not dispatch_path.exists(), "trace_restored": True}
print(json.dumps(result))
""", stage="execution interface repair publication")
    assert result["published_count"] == 7
    assert result["preserved_count"] == 26
    assert result["tamper_rejected"] is True
    assert result["current_authority_preserved"] is True
    assert result["guard"] == {
        "physical_input_reads": 0, "transport_events": 0, "scientific_entries": 0,
        "attempt_slots": 30, "attempt_roots": 0, "dispatch_absent": True,
        "trace_restored": True,
    }


def test_candidate_protocol_publication_collisions_are_hard_and_write_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operational_twin_protocol_builder, "REPO_ROOT", tmp_path)
    protocol_root = tmp_path / "Output/protocol"
    protocol_root.mkdir(parents=True)
    protocol_manifest = protocol_root / "candidate_lineage_manifest.json"
    external_audit = tmp_path / "Output/audits/candidate_preflight.json"
    with pytest.raises(FileExistsError, match="candidate_protocol_collision"):
        operational_twin_protocol_builder._publish_candidate_artifacts(
            [
                (protocol_manifest, b"manifest"),
                (external_audit, b"audit"),
            ],
            protocol_root=protocol_root,
        )
    assert not protocol_manifest.exists()
    assert not external_audit.exists()

    second_protocol_root = tmp_path / "Output/second_protocol"
    existing_report = tmp_path / "Output/reports/existing_report.md"
    existing_report.parent.mkdir(parents=True)
    existing_report.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="candidate_protocol_collision"):
        operational_twin_protocol_builder._publish_candidate_artifacts(
            [
                (second_protocol_root / "assignment_manifest.json", b"assignment"),
                (existing_report, b"replacement"),
            ],
            protocol_root=second_protocol_root,
        )
    assert not second_protocol_root.exists()
    assert existing_report.read_text(encoding="utf-8") == "preserve"


def test_candidate_protocol_publication_rolls_back_on_preserved_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operational_twin_protocol_builder, "REPO_ROOT", tmp_path)
    preserved_path = tmp_path / "preserved.txt"
    preserved_path.write_text("before\n", encoding="utf-8")
    preserved_identity = operational_twin_protocol_builder._artifact_identity(
        tmp_path,
        preserved_path,
    )
    protocol_root = tmp_path / "Output/protocol"
    output_path = protocol_root / "candidate_lineage_manifest.json"
    original_write = candidate_execution.write_bytes_exclusive
    published = []

    def write_then_drift(path: Path, content: bytes) -> None:
        assert path == output_path
        original_write(path, content)
        assert path.read_bytes() == content
        published.append(path)
        preserved_path.write_text("after\n", encoding="utf-8")

    monkeypatch.setattr(
        candidate_execution, "write_bytes_exclusive", write_then_drift
    )
    with pytest.raises(
        ValueError,
        match="candidate_protocol_preserved_identity_drift",
    ):
        operational_twin_protocol_builder._publish_candidate_artifacts(
            [(output_path, b"manifest")],
            protocol_root=protocol_root,
            preserved_identities=[preserved_identity],
        )
    assert published == [output_path]
    assert preserved_path.read_text(encoding="utf-8") == "after\n"
    assert not output_path.exists()
    assert not protocol_root.exists()


def test_candidate_attempt_root_collision_fails_before_seed_data_or_model_work(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path,
        _CANDIDATE_RUNTIME_METADATA_FIXTURE + r"""
gate = prepare_control_gate()
with control_runtime(gate):
    publish_control_gate(gate)
    validate_gate(gate)
    output_root = gate.attempt_root
    output_root.mkdir(parents=True)
    sentinel = output_root / "preexisting.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    args = control_args(gate)
    calls = []
    previous_profile = sys.getprofile()
    forbidden = {trainer._seed_everything.__code__, trainer._next_attempt_uid.__code__,
                 trainer.tensorize_manifest.__code__, trainer.build_method_model.__code__}
    def observe(frame, event, value):
        if previous_profile is not None:
            previous_profile(frame, event, value)
        if event == "call" and frame.f_code in forbidden:
            calls.append(frame.f_code.co_name)
            raise AssertionError("candidate work started after an attempt-root collision")
    sys.setprofile(observe)
    try:
        with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_attempt_output_collision"):
            trainer.run(args)
    finally:
        sys.setprofile(previous_profile)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert list(output_root.iterdir()) == [sentinel]
    assert calls == []
result = {"collision_rejected": True, "sentinel_preserved": True, "work_calls": calls}
""", stage="candidate original attempt collision")
    assert result == {"collision_rejected": True, "sentinel_preserved": True, "work_calls": []}


def _patch_candidate_run_through_attempt_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict[str, object],
    candidate_manifest: dict[str, object],
    assignment_manifest: dict[str, object],
    catalog: dict[str, object],
    assignment: dict[str, object],
    attempt: dict[str, object],
) -> tuple[SimpleNamespace, list[str]]:
    initialization = candidate_fresh_initialization_receipt(
        config["operational_twin_candidate_lineage_protocol"]
    )
    paths = {
        "output_root": tmp_path / attempt["output_root"],
        "calibration_execution_authority": tmp_path / "calibration_authority.json",
    }
    # This fixture supplies inert governance inputs. The real run and shared
    # no-follow claim still execute; no source/data/model work may start.
    resolved = operational_twin_protocol.ResolvedCandidateExecutionProtocol(
        protocol=config["operational_twin_candidate_lineage_protocol"],
        candidate_manifest=candidate_manifest, assignment_manifest=assignment_manifest,
        command_catalog=catalog, preflight={"status": "PASS"},
        execution=config["operational_twin_candidate_execution"], handoff={}, binding={},
    )
    def resolved_fixture(runtime_config, runtime_root):
        assert runtime_config is config and runtime_root == tmp_path
        return resolved
    monkeypatch.setattr(operational_twin_runner, "resolve_candidate_execution_protocol", resolved_fixture)

    def claim_fixture(runtime_config, runtime_root, runtime_resolved, attempt_uid,
                      execution_context, *, claimed_at):
        assert runtime_config is config and runtime_root == tmp_path
        assert runtime_resolved is resolved and attempt_uid == attempt["attempt_uid"]
        return operational_twin_protocol._claim_candidate_attempt_root(
            runtime_root, attempt["output_root"])
    monkeypatch.setattr(operational_twin_runner, "claim_candidate_attempt", claim_fixture)
    monkeypatch.setattr(operational_twin_runner, "resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        operational_twin_runner,
        "resolve_operational_twin_run_contract",
        lambda args, root, require_output=False: {
            "config": config,
            "cfg": config["operational_twin"],
            "paths": paths,
        },
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_operational_twin_training_authorized",
        lambda *args: None,
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_candidate_assignment_preflight",
        lambda *args, **kwargs: {
            "assignment": assignment,
            "attempt": attempt,
            "preflight": {"status": "PASS"},
            "initialization": initialization,
            "candidate_manifest": candidate_manifest,
            "assignment_manifest": assignment_manifest,
            "command_catalog": catalog,
        },
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_candidate_execution_authority",
        lambda *args, **kwargs: {
            "execution_authority": {"authority_uid": "fixture-authority"},
            "resource_environment_receipt": {"receipt_uid": "fixture-receipt"},
            "resource_receipt_payload": {"source_commit": "disposable-run-fixture"},
        },
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_calibration_execution_authority",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_assignment_runtime_resource",
        lambda assignment, **kwargs: None,
    )

    work_calls: list[str] = []

    def fail_if_work_starts(*args: object, **kwargs: object) -> None:
        work_calls.append("started")
        raise AssertionError("candidate data/model work started before path rejection")

    monkeypatch.setattr(operational_twin_runner, "_seed_everything", fail_if_work_starts)
    monkeypatch.setattr(operational_twin_runner, "_next_attempt_uid", fail_if_work_starts)
    monkeypatch.setattr(operational_twin_runner, "tensorize_manifest", fail_if_work_starts)
    monkeypatch.setattr(operational_twin_runner, "build_method_model", fail_if_work_starts)

    return (
        SimpleNamespace(
            protocol_scope="candidate-lineage",
            assignment_uid=assignment["assignment_uid"],
            method_id=assignment["method_id"],
            seed=assignment["seed"],
            candidate_lineage_uid=assignment["candidate_lineage_uid"],
            protocol_uid=assignment["protocol_uid"],
            attempt_uid=attempt["attempt_uid"],
            candidate_manifest_hash=candidate_manifest["candidate_manifest_hash"],
            assignment_manifest_hash=assignment_manifest[
                "assignment_manifest_hash"
            ],
            preflight_uid=candidate_manifest["preflight_uid"],
            command_catalog_hash=catalog["command_catalog_hash"],
            output_root=attempt["output_root"],
            candidate_execution_authority=attempt["execution_authority"],
            candidate_resource_environment_receipt=attempt[
                "resource_environment_receipt"
            ],
            resume=False,
            resume_input=None,
            history_input=None,
            checkpoint_input=None,
            optimizer_state_input=None,
            model_state_input=None,
            calibration_state_input=None,
            dry_run=False,
        ),
        work_calls,
    )


@pytest.mark.parametrize(
    "path_case",
    [
        "existing_symlink_leaf",
        "dangling_symlink_leaf",
        "symlinked_ancestor",
    ],
)
def test_candidate_attempt_symlinks_fail_preflight_and_atomic_claim_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_case: str,
) -> None:
    config, candidate_manifest, assignment_manifest, catalog = (
        _candidate_runtime_artifacts(tmp_path)
    )
    assignment = assignment_manifest["assignments"][0]
    attempt = assignment["attempts"][0]
    declared_path = Path(attempt["output_root"])
    output_root = tmp_path / declared_path
    outside_target = tmp_path / f"outside_{path_case}"
    outside_descendant: Path | None = None

    if path_case == "existing_symlink_leaf":
        outside_target.mkdir()
        (outside_target / "sentinel.txt").write_text("preserve", encoding="utf-8")
        output_root.parent.mkdir(parents=True)
        output_root.symlink_to(outside_target, target_is_directory=True)
        symlink_path = output_root
    elif path_case == "dangling_symlink_leaf":
        output_root.parent.mkdir(parents=True)
        output_root.symlink_to(outside_target, target_is_directory=True)
        symlink_path = output_root
    else:
        lineage_index = declared_path.parts.index("candidate_lineages") + 1
        ancestor_relative = Path(*declared_path.parts[: lineage_index + 1])
        ancestor_path = tmp_path / ancestor_relative
        ancestor_path.parent.mkdir(parents=True)
        outside_target.mkdir()
        (outside_target / "sentinel.txt").write_text("preserve", encoding="utf-8")
        ancestor_path.symlink_to(outside_target, target_is_directory=True)
        symlink_path = ancestor_path
        outside_descendant = outside_target / Path(
            *declared_path.parts[lineage_index + 1 :]
        )

    _patch_candidate_runtime_artifacts(
        monkeypatch,
        candidate_manifest,
        assignment_manifest,
        catalog,
    )
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_attempt_path_symlink",
    ):
        require_candidate_assignment_preflight(
            config,
            tmp_path,
            **_candidate_runtime_kwargs(
                candidate_manifest,
                assignment_manifest,
                catalog,
                assignment,
                attempt,
            ),
        )
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_attempt_path_symlink",
    ):
        candidate_attempt_path(
            tmp_path,
            attempt["output_root"],
            require_absent=True,
        )
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_attempt_path_symlink",
    ):
        operational_twin_protocol._claim_candidate_attempt_root(
            tmp_path,
            declared_output_root=attempt["output_root"],
        )

    run_args, work_calls = _patch_candidate_run_through_attempt_claim(
        tmp_path,
        monkeypatch,
        config=config,
        candidate_manifest=candidate_manifest,
        assignment_manifest=assignment_manifest,
        catalog=catalog,
        assignment=assignment,
        attempt=attempt,
    )
    with pytest.raises(
        OperationalTwinProtocolError,
        match="candidate_attempt_path_symlink",
    ):
        operational_twin_runner.run(run_args)

    assert work_calls == []
    assert symlink_path.is_symlink()
    if path_case == "dangling_symlink_leaf":
        assert not outside_target.exists()
    else:
        assert (outside_target / "sentinel.txt").read_text(encoding="utf-8") == (
            "preserve"
        )
        assert sorted(path.name for path in outside_target.iterdir()) == [
            "sentinel.txt"
        ]
    if outside_descendant is not None:
        assert not outside_descendant.exists()


def test_candidate_trainer_requires_execution_gate_before_attempt_claim(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path,
        _CANDIDATE_RUNTIME_METADATA_FIXTURE + r"""
gate = prepare_control_gate()
with control_runtime(gate):
    publish_control_gate(gate)
    baseline = validate_gate(gate)
    retained = {path: path.read_bytes() for path in (gate.authority_path, gate.receipt_path)}
    args = control_args(gate)
    calls = []
    previous_profile = sys.getprofile()
    forbidden = {trainer._seed_everything.__code__, trainer.tensorize_manifest.__code__,
                 trainer.build_method_model.__code__, runtime.claim_candidate_attempt.__code__}
    def observe(frame, event, value):
        if previous_profile is not None:
            previous_profile(frame, event, value)
        if event == "call" and frame.f_code in forbidden:
            calls.append(frame.f_code.co_name)
            raise AssertionError("candidate work started without execution authority")
    try:
        for path in retained:
            path.unlink()
        sys.setprofile(observe)
        with pytest.raises(protocol.OperationalTwinProtocolError, match="candidate_execution_authority_missing"):
            trainer.run(args)
        assert not gate.attempt_root.exists()
        assert calls == []
    finally:
        sys.setprofile(previous_profile)
        for path, content in retained.items():
            path.write_bytes(content)
        assert validate_gate(gate) == baseline
result = {"missing_authority_rejected": True, "attempt_created": False, "work_calls": calls}
""", stage="candidate original trainer authority before claim")
    assert result == {"missing_authority_rejected": True, "attempt_created": False, "work_calls": []}


def test_candidate_four_level_identity_propagates_to_runtime_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, candidate_manifest, assignment_manifest, catalog = _candidate_runtime_artifacts(tmp_path)
    protocol = config["operational_twin_candidate_lineage_protocol"]
    assignment = assignment_manifest["assignments"][0]
    attempt = assignment["attempts"][0]
    _patch_candidate_runtime_artifacts(
        monkeypatch,
        candidate_manifest,
        assignment_manifest,
        catalog,
    )
    context = require_candidate_assignment_preflight(
        config,
        tmp_path,
        **_candidate_runtime_kwargs(
            candidate_manifest,
            assignment_manifest,
            catalog,
            assignment,
            attempt,
        ),
    )
    runtime_identity = candidate_identity_payload(assignment, attempt)
    initialization = candidate_fresh_initialization_receipt(protocol)
    assert context["initialization"] == initialization
    assert initialization["parent_candidate_lineage_uid"] is None
    assert initialization["initialization_mode"] == "fresh_random"
    assert initialization["resume_allowed"] is False
    assert initialization["historical_state_use_allowed"] is False

    output_root = tmp_path / attempt["output_root"]
    operational_twin_protocol._claim_candidate_attempt_root(
        tmp_path,
        declared_output_root=attempt["output_root"],
    )
    method_contract = resolve_method_execution_contract(
        assignment["method_id"],
        assignment["execution_contract"],
    )
    result = operational_twin_runner._write_no_fit_result(
        output_root,
        assignment=assignment,
        method_contract=method_contract,
        seed=assignment["seed"],
        development_metrics={"reconstruction_rmse": 1.0},
        calibration_metrics={"reconstruction_rmse": 1.1},
        device_provenance={"resource_profile": "cpu_light"},
        scheduler_identity={"slurm_job_id": "fixture"},
        runtime_identity=runtime_identity,
        initialization=initialization,
    )
    checkpoint_payload = operational_twin_runner._bind_candidate_runtime_payload(
        {"schema_version": "checkpoint-fixture-v1", "model_state_dict": {}},
        runtime_identity,
        initialization,
    )
    terminal_status = operational_twin_runner._bind_candidate_runtime_payload(
        {
            "schema_version": "terminal-fixture-v1",
            "status": "COMPLETED",
            "assignment_uid": assignment["assignment_uid"],
            "attempt_uid": attempt["attempt_uid"],
        },
        runtime_identity,
        initialization,
    )
    operational_twin_runner._write_attempt_status(
        output_root,
        terminal_status,
        candidate_lineage=True,
    )

    metric_path = output_root / method_contract["metrics_path"]
    failure_path = output_root / method_contract["failure_retention_path"]
    payloads = [
        result,
        checkpoint_payload,
        json.loads(metric_path.read_text(encoding="utf-8").splitlines()[0]),
        json.loads(failure_path.read_text(encoding="utf-8").splitlines()[0]),
        json.loads((output_root / "provenance/runtime_provenance.json").read_text(encoding="utf-8")),
        json.loads((output_root / "attempt_receipt.json").read_text(encoding="utf-8")),
        json.loads((output_root / "terminal_status.json").read_text(encoding="utf-8")),
    ]
    for payload in payloads:
        assert {field: payload[field] for field in CANDIDATE_IDENTITY_FIELDS} == runtime_identity
        assert payload["fresh_initialization_receipt"] == initialization
    assert not (output_root / "attempts").exists()


def test_candidate_command_runner_main_defaults_to_configured_successor_before_authority_gate(
    isolated_execution_ready, training_protocol_call_witness, tmp_path: Path,
) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
import subprocess as shared_subprocess
import sys
from Libs.exp.training import run_operational_twin_assignment_command as command_runner

assert command_runner.REPO_ROOT == root
configured_catalog = root / resolved.protocol["output_paths"]["assignment_commands"]
legacy_catalog = root / config[protocol.CANDIDATE_PROTOCOL_KEY]["output_paths"]["assignment_commands"]
assert configured_catalog != legacy_catalog
actual_default = command_runner._candidate_catalog_path(None)
assert actual_default == configured_catalog
catalog_bytes = configured_catalog.read_bytes()
command_catalog = resolved.command_catalog
command_rows = command_catalog["commands"]
assert len(command_rows) == 30
selected_command = command_rows[0]
assert selected_command["command_index"] == 0
assert selected_command["attempt_kind"] == "initial"
assert selected_command["conditional"] is False
assert not (root / selected_command["execution_authority"]).exists()
assert not (root / selected_command["resource_environment_receipt"]).exists()
assert all(not (root / row["output_root"]).exists() for row in command_rows)
observed = {"authority_calls": 0, "dispatched": False}
real_authority = command_runner.require_candidate_execution_authority
shared_run = shared_subprocess.run

def observe_authority(*args, **kwargs):
    observed["authority_calls"] += 1
    assert kwargs["command_catalog"] == command_catalog
    assert kwargs["resolved_context"].protocol == resolved.protocol
    observed["authority_catalog_hash"] = kwargs["command_catalog"]["command_catalog_hash"]
    return real_authority(*args, **kwargs)

def fail_dispatch(*args, **kwargs):
    observed["dispatched"] = True
    raise AssertionError("training dispatched without execution authority")

# Replace only this runner's subprocess reference. Handoff Git calls retain
# the shared stdlib implementation and the existing read guard.
with patch.object(command_runner, "subprocess", SimpleNamespace(
    run=fail_dispatch, check_output=shared_subprocess.check_output)), patch.object(
    command_runner, "require_candidate_execution_authority", observe_authority), patch.object(
    sys, "argv", ["run_operational_twin_assignment_command.py",
                  "--protocol-scope", "candidate-lineage", "--task-index", "0"]):
    assert shared_subprocess.run is shared_run
    with pytest.raises(protocol.OperationalTwinProtocolError) as authority_failure:
        command_runner.main()
assert shared_subprocess.run is shared_run
assert str(authority_failure.value) == (
    "candidate_execution_authority_missing: independently approved execution authority is absent."
)
assert configured_catalog.read_bytes() == catalog_bytes
assert all(not (root / row["output_root"]).exists() for row in command_rows)
result = {
    "default_catalog_path": actual_default.relative_to(root).as_posix(),
    "resolved_catalog_path": configured_catalog.relative_to(root).as_posix(),
    "legacy_catalog_path": legacy_catalog.relative_to(root).as_posix(),
    "authority_calls": observed["authority_calls"],
    "authority_catalog_hash": observed["authority_catalog_hash"],
    "expected_catalog_hash": command_catalog["command_catalog_hash"],
    "missing_authority_error": str(authority_failure.value),
    "dispatched": observed["dispatched"],
    "catalog_bytes_unchanged": True,
    "declared_attempt_roots": len(command_rows),
    "attempt_roots": 0,
}
""", stage="configured successor command default",
        test_call_witness=training_protocol_call_witness)
    assert set(result) == {
        "default_catalog_path", "resolved_catalog_path", "legacy_catalog_path",
        "authority_calls", "authority_catalog_hash", "expected_catalog_hash",
        "missing_authority_error", "dispatched", "catalog_bytes_unchanged",
        "declared_attempt_roots", "attempt_roots", "test_call_witness",
    }
    assert result["default_catalog_path"] == result["resolved_catalog_path"]
    assert result["default_catalog_path"] != result["legacy_catalog_path"]
    assert result["authority_calls"] == 1
    assert result["authority_catalog_hash"] == result["expected_catalog_hash"]
    assert result["missing_authority_error"] == (
        "candidate_execution_authority_missing: independently approved execution authority is absent."
    )
    assert result["dispatched"] is False and result["catalog_bytes_unchanged"] is True
    assert result["declared_attempt_roots"] == 30 and result["attempt_roots"] == 0


def test_candidate_command_runner_rejects_catalog_tamper_and_bad_task_indexes_before_dispatch(
    isolated_execution_ready, training_protocol_call_witness, tmp_path: Path,
) -> None:
    # Preserve this nodeid's deep-hash obligation at its supported legacy
    # validator, separately from the current runner's committed-byte gate.
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
import yaml
from Libs.scripts.builders.build_operational_twin_development_training_commands import build_candidate_commands

legacy_ref = frozen_package["source_commit"]
legacy_config_path = "Libs/configs/model/operational_twin_candidate_lineage_protocol.yaml"
legacy_builder_path = "Libs/scripts/builders/build_operational_twin_development_training_commands.py"
legacy_source_authority = {}
for legacy_path in (legacy_config_path, legacy_builder_path):
    legacy_bytes = handoff._blob(root, legacy_ref, legacy_path)
    assert (root / legacy_path).read_bytes() == legacy_bytes
    legacy_source_authority[legacy_path] = hashlib.sha256(legacy_bytes).hexdigest()
legacy_protocol = yaml.safe_load((root / legacy_config_path).read_text())[protocol.CANDIDATE_PROTOCOL_KEY]
assert config[protocol.CANDIDATE_PROTOCOL_KEY] == legacy_protocol
assert legacy_protocol["candidate_lineage_uid"] != resolved.protocol["candidate_lineage_uid"]
legacy_candidate = protocol.build_candidate_lineage_manifest(config)
legacy_assignment = protocol.build_candidate_assignment_manifest(config, legacy_candidate)
legacy_catalog = build_candidate_commands(config, legacy_candidate, legacy_assignment)
assert protocol.validate_candidate_command_catalog(config, legacy_candidate, legacy_assignment, legacy_catalog) == []
legacy_tampered = deepcopy(legacy_catalog)
legacy_tampered["command_catalog_hash"] = "0" * 64
legacy_blockers = protocol.validate_candidate_command_catalog(
    config, legacy_candidate, legacy_assignment, legacy_tampered)
legacy_hash_blocker = {
    "code": "candidate_command_catalog_hash_mismatch",
    "message": "Command catalog hash does not recompute.",
}
assert legacy_hash_blocker in legacy_blockers
# This direct validator result does not claim execution of the old mocked
# runner path. Current-runner rejection is proved independently below.
assert runner.REPO_ROOT == root
command_catalog_relative = resolved.protocol["output_paths"]["assignment_commands"]
command_catalog_path = root / command_catalog_relative
command_catalog_bytes = command_catalog_path.read_bytes()
command_artifacts = protocol.load_candidate_protocol_artifacts(config, root, resolved_context=resolved)
assert protocol.validate_candidate_artifacts(
    config, root, *command_artifacts, require_attempt_roots_absent=True, resolved_context=resolved) == []
command_rows = command_artifacts[2]["commands"]
assert len(command_rows) == 30 and len({row["output_root"] for row in command_rows}) == 30
assert all(not (root / row["output_root"]).exists() for row in command_rows)
command_tampered = json.loads(command_catalog_bytes)
command_tampered["command_catalog_hash"] = "0" * 64
try:
    command_catalog_path.write_text(json.dumps(command_tampered), encoding="utf-8")
    with pytest.raises(protocol.OperationalTwinProtocolError) as command_failure:
        runner._run_candidate_command(command_catalog_path, 0)
    assert str(command_failure.value) == "handoff_committed_bytes_drift: " + command_catalog_relative
finally:
    command_catalog_path.write_bytes(command_catalog_bytes)
    assert command_catalog_path.read_bytes() == command_catalog_bytes
    restored = runtime.resolve_candidate_execution_protocol(config, root)
    restored_artifacts = protocol.load_candidate_protocol_artifacts(config, root, resolved_context=restored)
    assert restored_artifacts == command_artifacts
    assert protocol.validate_candidate_artifacts(
        config, root, *restored_artifacts, require_attempt_roots_absent=True, resolved_context=restored) == []
    assert all(not (root / row["output_root"]).exists() for row in command_rows)
index_errors = []
for task_index in (-1, 30):
    with pytest.raises(protocol.OperationalTwinProtocolError) as index_failure:
        runner._run_candidate_command(command_catalog_path, task_index)
    assert str(index_failure.value) == f"candidate_task_index_out_of_bounds: {task_index} is outside [0, 29]."
    index_errors.append(str(index_failure.value))
assert command_catalog_path.read_bytes() == command_catalog_bytes
assert all(not (root / row["output_root"]).exists() for row in command_rows)
result = {
    "legacy_source_ref": legacy_ref,
    "legacy_source_authority": legacy_source_authority,
    "legacy_hash_blocker": legacy_hash_blocker,
    "legacy_positive_baselines": 1,
    "catalog_path": command_catalog_relative,
    "catalog_tamper_error": str(command_failure.value),
    "index_errors": index_errors,
    "positive_baselines": 2,
    "catalog_bytes_unchanged": True,
    "declared_attempt_roots": len(command_rows),
    "attempt_roots": 0,
}
""", stage="training protocol command catalog preflight",
        test_call_witness=training_protocol_call_witness)
    assert set(result) == {
        "legacy_source_ref", "legacy_source_authority", "legacy_hash_blocker", "legacy_positive_baselines",
        "catalog_path", "catalog_tamper_error", "index_errors", "positive_baselines",
        "catalog_bytes_unchanged", "declared_attempt_roots", "attempt_roots", "test_call_witness",
    }
    assert result["legacy_source_ref"] == "a31757c9b2e40c05e9d2b273d38c36eaa6928d73"
    assert set(result["legacy_source_authority"]) == {
        "Libs/configs/model/operational_twin_candidate_lineage_protocol.yaml",
        "Libs/scripts/builders/build_operational_twin_development_training_commands.py",
    }
    assert result["legacy_hash_blocker"] == {
        "code": "candidate_command_catalog_hash_mismatch",
        "message": "Command catalog hash does not recompute.",
    }
    assert result["legacy_positive_baselines"] == 1 and result["positive_baselines"] == 2
    assert result["catalog_tamper_error"] == "handoff_committed_bytes_drift: " + result["catalog_path"]
    assert result["index_errors"] == [
        "candidate_task_index_out_of_bounds: -1 is outside [0, 29].",
        "candidate_task_index_out_of_bounds: 30 is outside [0, 29].",
    ]
    assert result["catalog_bytes_unchanged"] is True
    assert result["declared_attempt_roots"] == 30 and result["attempt_roots"] == 0



def test_candidate_command_runner_requires_execution_gate_before_dispatch(
    isolated_execution_ready, training_protocol_call_witness, tmp_path: Path,
) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path, r"""
from Libs.exp.training import run_operational_twin_assignment_command as command_runner

command_gate = prepare_gate()
command_published = publish_gate(command_gate)
command_baseline = validate_gate(command_gate)
command_gate_paths = (command_gate.authority_path, command_gate.receipt_path)
command_originals = {gate_path: gate_path.read_bytes() for gate_path in command_gate_paths}
assert command_baseline["resource_receipt_payload"] == json.loads(command_originals[command_gate.receipt_path])
assert command_baseline["execution_authority"]["authority_uid"] == json.loads(command_originals[command_gate.authority_path])["authority_uid"]
assert command_published["attempt_uid"] == command_gate.context["attempt"]["attempt_uid"]
assert command_runner.REPO_ROOT == root
command_catalog_path = root / resolved.protocol["output_paths"]["assignment_commands"]
command_catalog_bytes = command_catalog_path.read_bytes()
command_rows = command_gate.context["command_catalog"]["commands"]
assert len(command_rows) == 30 and len({row["output_root"] for row in command_rows}) == 30
assert all(not (root / row["output_root"]).exists() for row in command_rows)
try:
    # Ownership rejects a partial gate before checking authority. With both
    # owned files absent, the real command runner reaches its authority gate.
    for gate_path in command_gate_paths:
        gate_path.unlink()
    assert all(not gate_path.exists() for gate_path in command_gate_paths)
    with pytest.raises(protocol.OperationalTwinProtocolError) as command_failure:
        command_runner._run_candidate_command(command_catalog_path, 0)
    assert str(command_failure.value) == (
        "candidate_execution_authority_missing: independently approved execution authority is absent."
    )
finally:
    for gate_path, gate_bytes in command_originals.items():
        gate_path.write_bytes(gate_bytes)
    assert validate_gate(command_gate) == command_baseline
    assert all(gate_path.read_bytes() == gate_bytes for gate_path, gate_bytes in command_originals.items())
    assert command_catalog_path.read_bytes() == command_catalog_bytes
    assert all(not (root / row["output_root"]).exists() for row in command_rows)
result = {
    "missing_authority_error": str(command_failure.value),
    "positive_baselines": 2,
    "gate_bytes_unchanged": True,
    "catalog_bytes_unchanged": True,
    "declared_attempt_roots": len(command_rows),
    "attempt_roots": 0,
}
""", stage="training protocol command gate preflight",
        test_call_witness=training_protocol_call_witness)
    assert set(result) == {
        "missing_authority_error", "positive_baselines", "gate_bytes_unchanged",
        "catalog_bytes_unchanged", "declared_attempt_roots", "attempt_roots", "test_call_witness",
    }
    assert result["missing_authority_error"] == (
        "candidate_execution_authority_missing: independently approved execution authority is absent."
    )
    assert result["positive_baselines"] == 2
    assert result["gate_bytes_unchanged"] is result["catalog_bytes_unchanged"] is True
    assert result["declared_attempt_roots"] == 30 and result["attempt_roots"] == 0


def test_candidate_command_runner_executes_the_stored_argv_without_shell_reparse(
    isolated_execution_ready, tmp_path: Path,
) -> None:
    result = _run_candidate_gate_fixture(isolated_execution_ready, tmp_path,
        _CANDIDATE_RUNTIME_METADATA_FIXTURE + r"""
gate = prepare_control_gate()
with control_runtime(gate):
    publish_control_gate(gate)
    validate_gate(gate)
    catalog_path = root / resolved.protocol["output_paths"]["assignment_commands"]
    observed = {}
    previous_profile = sys.getprofile()
    def observe(frame, event, value):
        if previous_profile is not None:
            previous_profile(frame, event, value)
        parent = frame.f_back
        if parent is None or parent.f_code is not command_runner._run_candidate_command.__code__:
            return
        if event == "call" and frame.f_code in {
            protocol.load_candidate_protocol_artifacts.__code__, protocol.require_candidate_assignment_preflight.__code__,
            protocol.require_candidate_execution_authority.__code__}:
            context = frame.f_locals["resolved_context"]
            assert context is parent.f_locals["resolved"]
            assert dict(context.binding) == dict(resolved.binding)
        if event == "return" and frame.f_code is protocol.load_candidate_protocol_artifacts.__code__:
            observed["catalog"] = value[2]
        if event == "call" and frame.f_code is protocol.require_candidate_assignment_preflight.__code__:
            observed["preflight_kwargs"] = dict(frame.f_locals)
        if event == "call" and frame.f_code is protocol.require_candidate_execution_authority.__code__:
            observed["execution_authority_kwargs"] = dict(frame.f_locals)
    original_run = subprocess.run
    def capture_run(argv, **kwargs):
        if argv == gate.command["argv"]:
            observed.update(argv=argv, cwd=kwargs["cwd"], shell=kwargs["shell"])
            return SimpleNamespace(returncode=17)
        return original_run(argv, **kwargs)
    sys.setprofile(observe)
    try:
        with patch.object(subprocess, "run", capture_run):
            assert command_runner._run_candidate_command(catalog_path, 0) == 17
    finally:
        sys.setprofile(previous_profile)
    selected = observed["catalog"]["commands"][0]
    assert observed["argv"] == selected["argv"]
    assert observed["argv"] is not selected["argv"]
    assert observed["cwd"] == root
    assert observed["shell"] is False
    preflight_kwargs = observed["preflight_kwargs"]
    assert preflight_kwargs["assignment_uid"] == selected["assignment_uid"]
    assert preflight_kwargs["attempt_uid"] == selected["attempt_uid"]
    assert preflight_kwargs["output_root"] == selected["output_root"]
    assert preflight_kwargs["command_catalog_hash_value"] == observed["catalog"]["command_catalog_hash"]
    execution_kwargs = observed["execution_authority_kwargs"]
    assert execution_kwargs["execution_authority_path"] == selected["execution_authority"]
    assert execution_kwargs["resource_environment_receipt_path"] == selected["resource_environment_receipt"]
    assert not gate.attempt_root.exists()
result = {"stored_argv_preserved": True, "shell": False, "returned_status": 17,
          "real_preflight": True, "real_execution_authority": True, "resolved_context_retained": True}
""", stage="candidate original stored argv dispatch")
    assert result == {"stored_argv_preserved": True, "shell": False, "returned_status": 17,
                      "real_preflight": True, "real_execution_authority": True, "resolved_context_retained": True}


def test_candidate_builders_preserve_historical_v1_contract_and_assignment_ids() -> None:
    config = _config()
    historical_protocol = deepcopy(config["operational_twin_training_protocol"])
    historical_manifest = build_assignment_manifest(config, ROOT)

    _, candidate_manifest, assignment_manifest, catalog = _candidate_artifacts(config)
    candidate_protocol = config["operational_twin_candidate_lineage_protocol"]

    assert config["operational_twin_training_protocol"] == historical_protocol
    assert build_assignment_manifest(config, ROOT) == historical_manifest
    assert [row["assignment_uid"] for row in historical_manifest["assignments"]] == [
        f"operational_twin__{method_id}__seed_{seed}"
        for method_id in REQUIRED_METHOD_IDS
        for seed in (360101, 360211, 360307)
    ]
    for field in CANDIDATE_SCIENTIFIC_EQUALITY_FIELDS:
        assert candidate_protocol["scientific_contract"][field] == historical_protocol[field]
    assert candidate_manifest["candidate_lineage_uid"].startswith(
        "operational_twin_candidate_lineage_"
    )
    assert assignment_manifest["assignment_count"] == 15
    assert catalog["command_count"] == 30


def test_candidate_protocol_binds_the_round41_best_payload_snapshot_regression() -> None:
    protocol = _config()["operational_twin_candidate_lineage_protocol"]
    source_bindings = protocol["source_bindings"]
    assert source_bindings["protocol_regression_test"] == (
        "Libs/tests/test_operational_twin_training_protocol.py"
    )
    assert source_bindings["snapshot_repair_prospective_only"] is True
    assert source_bindings["snapshot_invariant"] == (
        "complete_best_development_payload_deepcopied_before_later_model_optimizer_or_nested_metadata_"
        "mutation_and_reused_for_save_restore_calibration_attachment_and_final_serialization"
    )
    trainer_source = (ROOT / source_bindings["trainer"]).read_text(encoding="utf-8")
    regression_source = (ROOT / source_bindings["protocol_regression_test"]).read_text(
        encoding="utf-8"
    )
    assert "best_payload = deepcopy(payload)" in trainer_source
    assert "def test_selected_checkpoint_snapshot_survives_later_training_and_calibration" in regression_source


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(assignment_uid="", method_id="", seed=None),
        SimpleNamespace(assignment_uid="operational_twin__persistence_no_action__seed_360101", method_id="", seed=360101),
        SimpleNamespace(assignment_uid="", method_id="persistence_no_action", seed=360101),
        SimpleNamespace(assignment_uid="operational_twin__persistence_no_action__seed_360101", method_id="persistence_no_action", seed=None),
    ],
)
def test_formal_training_requires_complete_assignment_args_before_output_creation(tmp_path: Path, args: SimpleNamespace) -> None:
    output_root = tmp_path / "formal_output"
    with pytest.raises(OperationalTwinProtocolError, match="formal_assignment_required"):
        _formal_assignment_args(args)
    assert not output_root.exists()


def test_all_five_methods_resolve_distinct_execution_contracts_and_unsupported_fails() -> None:
    config = _config()
    methods = config["operational_twin_training_protocol"]["methods"]
    resolved = {
        row["method_id"]: resolve_method_execution_contract(row["method_id"], row["execution_contract"])
        for row in methods
    }
    assert set(resolved) == set(METHOD_EXECUTION_CONTRACTS)
    assert len({row["runtime_family"] for row in resolved.values()}) == 5
    assert resolved["persistence_no_action"]["checkpoint_creation_allowed"] is False
    assert resolved["linear_or_delta"]["runtime_family"] == "linear_delta_baseline"
    assert resolved["same_backbone_no_action"]["action_conditioning"] == "no_action_ablation"
    assert resolved["deterministic_action_conditioned"]["action_conditioning"] == "structured_world_action_mean_only"
    assert resolved["full_uncertainty_aware_action_conditioned"]["action_conditioning"] == "structured_world_action_distributional"
    with pytest.raises(OperationalTwinProtocolError, match="unsupported_method_id"):
        resolve_method_execution_contract("unsupported_method")


def test_runtime_dispatch_uses_distinct_model_classes_without_unintended_operational_twin() -> None:
    contracts = {
        method_id: resolve_method_execution_contract(method_id, row["execution_contract"])
        for row in _config()["operational_twin_training_protocol"]["methods"]
        for method_id in [row["method_id"]]
    }
    persistence = build_method_model(2, 4, _tiny_model_config(), contracts["persistence_no_action"])
    linear = build_method_model(2, 4, _tiny_model_config(), contracts["linear_or_delta"])
    no_action = build_method_model(2, 4, _tiny_model_config(), contracts["same_backbone_no_action"])
    deterministic = build_method_model(2, 4, _tiny_model_config(), contracts["deterministic_action_conditioned"])
    full = build_method_model(2, 4, _tiny_model_config(), contracts["full_uncertainty_aware_action_conditioned"])
    assert isinstance(persistence, PersistenceNoActionPredictor)
    assert isinstance(linear, LinearDeltaOperationalTwin)
    assert not isinstance(persistence, OperationalTwin)
    assert not isinstance(linear, OperationalTwin)
    assert isinstance(no_action, OperationalTwin)
    assert isinstance(deterministic, MeanOnlyOperationalTwin)
    assert isinstance(full, OperationalTwin)
    assert not isinstance(full, MeanOnlyOperationalTwin)


def test_persistence_and_linear_execute_distinct_no_uncertainty_contracts() -> None:
    batch = _tiny_batch(action_value=1.0)
    persistence_contract = resolve_method_execution_contract("persistence_no_action", "no_fit_persistence")
    linear_contract = resolve_method_execution_contract("linear_or_delta", "trainable_linear_delta")
    persistence = build_method_model(2, 4, _tiny_model_config(), persistence_contract)
    linear = build_method_model(2, 4, _tiny_model_config(), linear_contract)
    persistence_output = persistence(
        batch["input_observation"],
        batch["input_mask"],
        _action_encoding_for_contract(batch, persistence_contract),
        batch["family_index"],
    )
    linear_output = linear(
        batch["input_observation"],
        batch["input_mask"],
        _action_encoding_for_contract(batch, linear_contract),
        batch["family_index"],
    )
    assert persistence_output["predictive_scale"] is None
    assert linear_output["predictive_scale"] is None
    assert persistence_contract["checkpoint_creation_allowed"] is False
    assert linear_contract["checkpoint_creation_allowed"] is True
    assert persistence_contract["loss_name"] != linear_contract["loss_name"]
    assert not torch.allclose(persistence_output["predictive_mean"], linear_output["predictive_mean"])
    assert torch.isfinite(loss_for_method(persistence_output, batch["target_observation"], batch["target_mask"], persistence_contract))
    assert torch.isfinite(loss_for_method(linear_output, batch["target_observation"], batch["target_mask"], linear_contract))


def test_same_backbone_no_action_masks_action_signal() -> None:
    contract = resolve_method_execution_contract("same_backbone_no_action", "trainable_operational_twin_backbone_no_action")
    left = _tiny_batch(action_value=1.0)
    right = _tiny_batch(action_value=100.0)
    assert torch.equal(_action_encoding_for_contract(left, contract), _action_encoding_for_contract(right, contract))
    model = build_method_model(2, 4, _tiny_model_config(), contract)
    with torch.no_grad():
        left_output = model(left["input_observation"], left["input_mask"], _action_encoding_for_contract(left, contract), left["family_index"])
        right_output = model(right["input_observation"], right["input_mask"], _action_encoding_for_contract(right, contract), right["family_index"])
    assert torch.allclose(left_output["predictive_mean"], right_output["predictive_mean"])


def test_deterministic_action_conditioned_is_action_mean_only_and_not_calibrated() -> None:
    contract = resolve_method_execution_contract("deterministic_action_conditioned", "trainable_action_conditioned_mean")
    left = _tiny_batch(action_value=0.0)
    right = _tiny_batch(action_value=1.0)
    assert not torch.equal(_action_encoding_for_contract(left, contract), _action_encoding_for_contract(right, contract))
    model = build_method_model(2, 4, _tiny_model_config(), contract)
    with torch.no_grad():
        left_output = model(left["input_observation"], left["input_mask"], _action_encoding_for_contract(left, contract), left["family_index"])
        output = model(right["input_observation"], right["input_mask"], _action_encoding_for_contract(right, contract), right["family_index"])
    assert output["predictive_scale"] is None
    assert output["calibration_eligible"] is False
    assert not torch.allclose(left_output["predictive_mean"], output["predictive_mean"])
    assert contract["loss_name"] == "masked_mse_deterministic"
    with pytest.raises(OperationalTwinProtocolError, match="calibration_not_allowed"):
        fit_calibration(model, [], torch.zeros(2), torch.ones(2), torch.device("cpu"), contract)


def test_full_uncertainty_aware_action_conditioned_supplies_scale_and_calibration_contract() -> None:
    contract = resolve_method_execution_contract("full_uncertainty_aware_action_conditioned", "trainable_action_conditioned_distributional")
    batch = _tiny_batch(action_value=1.0)
    no_action_batch = _tiny_batch(action_value=0.0)
    model = build_method_model(2, 4, _tiny_model_config(), contract)
    with torch.no_grad():
        output = model(batch["input_observation"], batch["input_mask"], _action_encoding_for_contract(batch, contract), batch["family_index"])
        no_action_output = model(no_action_batch["input_observation"], no_action_batch["input_mask"], _action_encoding_for_contract(no_action_batch, contract), no_action_batch["family_index"])
    assert isinstance(output["predictive_scale"], torch.Tensor)
    assert torch.all(output["predictive_scale"] > 0)
    assert not torch.allclose(no_action_output["predictive_mean"], output["predictive_mean"])
    assert contract["calibration_eligible"] is True
    assert contract["loss_name"] == "masked_gaussian_nll"
    assert torch.isfinite(loss_for_method(output, batch["target_observation"], batch["target_mask"], contract))


def test_selected_checkpoint_snapshot_survives_later_training_and_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_id = "full_uncertainty_aware_action_conditioned"
    assignment_uid = f"operational_twin__{method_id}__seed_360101"
    output_root = tmp_path / "training_run"
    method_protocol = {
        "method_id": method_id,
        "execution_contract": "trainable_action_conditioned_distributional",
        "budget": {
            "epochs": 2,
            "max_steps": 2,
            "batch_size": 1,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "gradient_clip_norm": 100.0,
        },
        "capacity": {},
    }
    cfg = {
        "training": {
            "deterministic_kernels": True,
            "batch_size": 1,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "gradient_clip_norm": 100.0,
        },
        "model": {"source_metadata": {"revision": "selected_epoch"}},
        "access_policy": deepcopy(operational_twin_runner.EXPECTED_ACCESS_POLICY),
    }
    config = {
        "operational_twin": cfg,
        "operational_twin_training_protocol": {
            "protocol_uid": "snapshot_regression_protocol",
            "methods": [method_protocol],
        },
    }
    paths = {
        "training": tmp_path / "training.json",
        "development": tmp_path / "development.json",
        "calibration": tmp_path / "calibration.json",
        "calibration_execution_authority": tmp_path / "calibration_authority.json",
        "open_action_package_receipt": tmp_path / "package_receipt.json",
        "future_open_data_qualification": tmp_path / "open_data_qualification.json",
        "output_root": output_root,
    }
    assignment = {
        "assignment_uid": assignment_uid,
        "method_id": method_id,
        "seed": 360101,
        "execution_contract": "trainable_action_conditioned_distributional",
        "resource_profile": "cpu_light",
    }

    class TinyControlledWorldDataset(torch.utils.data.Dataset):
        def __init__(self, split: str, action_dim: int) -> None:
            self.split = split
            self.action_dim = action_dim
            self.sample_count = 1 if split == "training" else len(operational_twin_runner.FAMILY_UIDS)
            self.records = [{"support_ood": False} for _ in range(self.sample_count)]

        def __len__(self) -> int:
            return self.sample_count

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            observation = torch.tensor([[1.0], [2.0 + 0.1 * index]], dtype=torch.float32)
            action = torch.zeros(self.action_dim, dtype=torch.float32)
            dose_index = operational_twin_runner.WORLD_ACTION_FIELDS.index("dose_or_intensity")
            action[dose_index] = float(index + 1)
            return {
                "input_observation": observation,
                "input_mask": torch.tensor([[True], [False]]),
                "target_observation": observation,
                "target_mask": torch.tensor([[False], [True]]),
                "action_encoding": action,
                "no_action_encoding": torch.zeros_like(action),
                "family_index": torch.tensor(index, dtype=torch.long),
            }

    class TinyDistributionalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.location = torch.nn.Parameter(torch.tensor(0.25))
            self.log_scale = torch.nn.Parameter(torch.tensor(-0.25))

        def forward(
            self,
            inputs: torch.Tensor,
            input_mask: torch.Tensor,
            action: torch.Tensor,
            family: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del input_mask, action, family
            return {
                "predictive_mean": self.location.expand_as(inputs),
                "predictive_scale": (torch.nn.functional.softplus(self.log_scale) + 0.1).expand_as(inputs),
            }

    def tensorized_fixture(split: str) -> dict[str, object]:
        sample_count = 1 if split == "training" else len(operational_twin_runner.FAMILY_UIDS)
        records = [
            {
                "observation": torch.tensor([[1.0], [2.0 + 0.1 * index]], dtype=torch.float32),
                "observation_mask": torch.ones((2, 1), dtype=torch.bool),
            }
            for index in range(sample_count)
        ]
        return {
            "split": split,
            "records": records,
            "feature_names": ["observable_state"],
            "manifest_hash": f"{split}_manifest_hash",
            "manifest": {"manifest_uid": f"{split}_manifest"},
        }

    tensorized = {split: tensorized_fixture(split) for split in ("training", "development", "calibration")}
    model = TinyDistributionalModel()
    observed: dict[str, object] = {}
    original_adamw = torch.optim.AdamW
    original_evaluate = operational_twin_runner.evaluate
    original_fit_calibration = operational_twin_runner.fit_calibration

    def build_model(
        feature_count: int,
        action_dim: int,
        model_config: dict[str, object],
        method_contract: dict[str, object],
    ) -> torch.nn.Module:
        assert feature_count == 1
        assert action_dim > 0
        assert method_contract["calibration_eligible"] is True
        observed["model_config"] = model_config
        return model

    def capture_adamw(parameters: object, **kwargs: object) -> torch.optim.AdamW:
        optimizer = original_adamw(parameters, **kwargs)
        observed["optimizer"] = optimizer
        return optimizer

    def controlled_evaluate(
        evaluated_model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
        calibration: torch.Tensor | None = None,
        method_contract: dict[str, object] | None = None,
    ) -> dict[str, object]:
        metrics = original_evaluate(
            evaluated_model,
            loader,
            norm_mean,
            norm_std,
            device,
            calibration,
            method_contract,
        )
        if loader.dataset.split == "development":
            development_call = int(observed.get("development_call", 0)) + 1
            observed["development_call"] = development_call
            metrics["selection_marker"] = "selected_epoch" if development_call == 1 else "later_epoch"
            metrics["reconstruction_rmse"] = 0.1 if development_call == 1 else 1.0
            if development_call == 2:
                selected_payload = torch.load(
                    output_root / "checkpoints/best_dev.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                observed["selected_payload"] = selected_payload
                later_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in evaluated_model.state_dict().items()
                }
                optimizer = observed["optimizer"]
                assert isinstance(optimizer, torch.optim.Optimizer)
                later_optimizer_state = deepcopy(optimizer.state_dict())
                observed["later_model_state"] = later_model_state
                observed["later_optimizer_state"] = later_optimizer_state
                assert any(
                    not torch.equal(value, selected_payload["model_state_dict"][name])
                    for name, value in later_model_state.items()
                )
                selected_slots = selected_payload["optimizer_state_dict"]["state"]
                later_slots = later_optimizer_state["state"]
                assert selected_slots.keys() == later_slots.keys()
                assert all(
                    float(later_slots[key]["step"]) > float(selected_slots[key]["step"])
                    for key in selected_slots
                )
                assert any(
                    not torch.equal(later_slots[key]["exp_avg"], selected_slots[key]["exp_avg"])
                    for key in selected_slots
                )
                assert any(
                    not torch.equal(later_slots[key]["exp_avg_sq"], selected_slots[key]["exp_avg_sq"])
                    for key in selected_slots
                )
                model_config = observed["model_config"]
                assert isinstance(model_config, dict)
                assert model_config["source_metadata"]["revision"] == "selected_epoch"
                model_config["source_metadata"]["revision"] = "later_epoch"
        elif loader.dataset.split == "calibration":
            observed["calibration_metrics"] = deepcopy(metrics)
        return metrics

    def observed_fit_calibration(
        calibrated_model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
        method_contract: dict[str, object] | None = None,
    ) -> torch.Tensor:
        selected_payload = observed["selected_payload"]
        assert isinstance(selected_payload, dict)
        restored_state = {
            name: value.detach().cpu().clone()
            for name, value in calibrated_model.state_dict().items()
        }
        _assert_nested_checkpoint_equal(restored_state, selected_payload["model_state_dict"])
        observed["calibration_model_state"] = restored_state
        scale = original_fit_calibration(
            calibrated_model,
            loader,
            norm_mean,
            norm_std,
            device,
            method_contract,
        )
        observed["calibration_scale"] = scale.detach().cpu().clone()
        return scale

    monkeypatch.setattr(operational_twin_runner, "resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        operational_twin_runner,
        "resolve_operational_twin_run_contract",
        lambda args, root, require_output=False: {"config": config, "cfg": cfg, "paths": paths},
    )
    monkeypatch.setattr(operational_twin_runner, "require_operational_twin_training_authorized", lambda *args: None)
    monkeypatch.setattr(
        operational_twin_runner,
        "require_assignment_preflight",
        lambda *args, **kwargs: assignment,
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_calibration_execution_authority",
        lambda *args, **kwargs: {"authority_uid": "snapshot_regression_authority"},
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_assignment_runtime_resource",
        lambda assignment, **kwargs: None,
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "tensorize_manifest",
        lambda path, split, loaded_config, root: tensorized[split],
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "ControlledWorldTensorDataset",
        lambda value, encoder: TinyControlledWorldDataset(value["split"], encoder.output_dim),
    )
    monkeypatch.setattr(operational_twin_runner, "build_method_model", build_model)
    monkeypatch.setattr(operational_twin_runner.torch.optim, "AdamW", capture_adamw)
    monkeypatch.setattr(operational_twin_runner, "evaluate", controlled_evaluate)
    monkeypatch.setattr(operational_twin_runner, "fit_calibration", observed_fit_calibration)
    monkeypatch.setattr(
        operational_twin_runner,
        "tracked_reference_identity",
        lambda root, path: {
            "path": path.relative_to(root).as_posix(),
            "authority": "test_fixture",
            "receipt_content_opened": False,
        },
    )
    monkeypatch.setattr(
        operational_twin_runner.subprocess,
        "check_output",
        lambda *args, **kwargs: "snapshot-regression-commit\n",
    )

    result = operational_twin_runner.run(
        SimpleNamespace(
            assignment_uid=assignment_uid,
            method_id=method_id,
            seed=360101,
            dry_run=False,
        )
    )

    selected_payload = observed["selected_payload"]
    assert isinstance(selected_payload, dict)
    best_payload = torch.load(output_root / "checkpoints/best_dev.pt", map_location="cpu", weights_only=False)
    last_payload = torch.load(output_root / "checkpoints/last.pt", map_location="cpu", weights_only=False)

    for field in ("model_state_dict", "optimizer_state_dict", "epoch", "step", "dev_metrics", "model_config"):
        _assert_nested_checkpoint_equal(best_payload[field], selected_payload[field])
    assert best_payload["epoch"] == 1
    assert best_payload["step"] == 1
    assert best_payload["dev_metrics"]["selection_marker"] == "selected_epoch"
    assert best_payload["model_config"]["source_metadata"]["revision"] == "selected_epoch"
    _assert_nested_checkpoint_equal(best_payload["calibration_scale"], observed["calibration_scale"])
    expected_calibration_metrics = deepcopy(observed["calibration_metrics"])
    expected_calibration_metrics["calibration_scale_by_family"] = {
        name: float(observed["calibration_scale"][index])
        for index, name in enumerate(operational_twin_runner.FAMILY_UIDS)
    }
    _assert_nested_checkpoint_equal(best_payload["calibration_metrics"], expected_calibration_metrics)
    _assert_nested_checkpoint_equal(observed["calibration_model_state"], selected_payload["model_state_dict"])

    assert last_payload["epoch"] == 2
    assert last_payload["step"] == 2
    assert last_payload["dev_metrics"]["selection_marker"] == "later_epoch"
    assert last_payload["model_config"]["source_metadata"]["revision"] == "later_epoch"
    _assert_nested_checkpoint_equal(last_payload["model_state_dict"], observed["later_model_state"])
    _assert_nested_checkpoint_equal(last_payload["optimizer_state_dict"], observed["later_optimizer_state"])
    assert any(
        not torch.equal(last_payload["model_state_dict"][name], best_payload["model_state_dict"][name])
        for name in best_payload["model_state_dict"]
    )
    assert result["selected_checkpoint_epoch"] == 1
    assert result["epochs_completed"] == 2
    assert result["steps_completed"] == 2


def test_candidate_trainable_run_persists_manifest_identity_and_snapshot_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_config())
    cfg = config["operational_twin"]
    cfg["model"]["source_metadata"] = {"revision": "selected_epoch"}
    _materialize_candidate_validator_authorities(tmp_path, config)

    candidate_manifest = build_candidate_lineage_manifest(config)
    assignment_manifest = build_candidate_assignment_manifest(
        config,
        candidate_manifest,
    )
    catalog = build_candidate_commands(
        config,
        candidate_manifest,
        assignment_manifest,
    )
    preflight = operational_twin_protocol.build_candidate_preflight_authority(
        config,
        tmp_path,
        candidate_manifest,
        assignment_manifest,
        catalog,
    )
    protocol = config["operational_twin_candidate_lineage_protocol"]
    protocol_paths = protocol["output_paths"]
    serialized_artifacts = {
        "candidate_lineage_manifest": candidate_manifest,
        "assignment_manifest": assignment_manifest,
        "assignment_commands": catalog,
        "preflight_authority": preflight,
    }
    for path_key, payload in serialized_artifacts.items():
        path = tmp_path / protocol_paths[path_key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    loaded_artifacts = operational_twin_protocol.load_candidate_protocol_artifacts(
        config,
        tmp_path,
    )
    assert loaded_artifacts == (
        candidate_manifest,
        assignment_manifest,
        catalog,
        preflight,
    )
    assert operational_twin_protocol.validate_candidate_artifacts(
        config,
        tmp_path,
        *loaded_artifacts,
        require_attempt_roots_absent=True,
    ) == []

    method_id = "full_uncertainty_aware_action_conditioned"
    assignment = next(
        row
        for row in assignment_manifest["assignments"]
        if row["method_id"] == method_id and row["seed"] == 360101
    )
    attempt = assignment["attempts"][0]
    output_root = tmp_path / attempt["output_root"]
    runtime_identity = candidate_identity_payload(assignment, attempt)
    initialization = candidate_fresh_initialization_receipt(protocol)
    scientific = protocol["scientific_contract"]
    paths = {
        "training": tmp_path / scientific["manifests"]["training"],
        "development": tmp_path / scientific["manifests"]["development"],
        "calibration": tmp_path / scientific["manifests"]["calibration"],
        "calibration_execution_authority": (
            tmp_path / protocol_paths["candidate_calibration_execution_authority"]
        ),
        "open_action_package_receipt": (
            tmp_path / cfg["references"]["open_action_package_receipt"]
        ),
        "future_open_data_qualification": (
            tmp_path / scientific["open_data_authority"]
        ),
        "output_root": output_root,
    }

    class TinyControlledWorldDataset(torch.utils.data.Dataset):
        def __init__(self, split: str, action_dim: int) -> None:
            self.split = split
            self.action_dim = action_dim
            self.records = [{"support_ood": False}]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            del index
            observation = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
            action = torch.zeros(self.action_dim, dtype=torch.float32)
            dose_index = operational_twin_runner.WORLD_ACTION_FIELDS.index(
                "dose_or_intensity"
            )
            action[dose_index] = 1.0
            return {
                "input_observation": observation,
                "input_mask": torch.tensor([[True], [False]]),
                "target_observation": observation,
                "target_mask": torch.tensor([[False], [True]]),
                "action_encoding": action,
                "no_action_encoding": torch.zeros_like(action),
                "family_index": torch.tensor(0, dtype=torch.long),
            }

    class TinyDistributionalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.location = torch.nn.Parameter(torch.tensor(0.25))
            self.log_scale = torch.nn.Parameter(torch.tensor(-0.25))

        def forward(
            self,
            inputs: torch.Tensor,
            input_mask: torch.Tensor,
            action: torch.Tensor,
            family: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del input_mask, action, family
            return {
                "predictive_mean": self.location.expand_as(inputs),
                "predictive_scale": (
                    torch.nn.functional.softplus(self.log_scale) + 0.1
                ).expand_as(inputs),
            }

    tensorized = {
        split: {
            "split": split,
            "records": [
                {
                    "observation": torch.tensor(
                        [[1.0], [2.0]],
                        dtype=torch.float32,
                    ),
                    "observation_mask": torch.ones((2, 1), dtype=torch.bool),
                }
            ],
            "feature_names": ["observable_state"],
            "manifest_hash": f"{split}_manifest_hash",
            "manifest": {"manifest_uid": f"{split}_manifest"},
        }
        for split in ("training", "development", "calibration")
    }
    model = TinyDistributionalModel()
    observed: dict[str, object] = {}
    original_adamw = torch.optim.AdamW
    original_method_protocol_config = operational_twin_runner.method_protocol_config
    real_candidate_preflight = (
        operational_twin_protocol.require_candidate_assignment_preflight
    )

    def bounded_method_protocol(
        loaded_config: dict[str, object],
        selected_method_id: str,
        *,
        candidate_lineage: bool = False,
    ) -> dict[str, object]:
        method_protocol = deepcopy(
            original_method_protocol_config(
                loaded_config,
                selected_method_id,
                candidate_lineage=candidate_lineage,
            )
        )
        if candidate_lineage and selected_method_id == method_id:
            method_protocol["budget"] = {
                "epochs": 2,
                "max_steps": 2,
                "batch_size": 1,
                "learning_rate": 0.05,
                "weight_decay": 0.0,
                "gradient_clip_norm": 100.0,
            }
            method_protocol["capacity"] = {}
        return method_protocol

    def candidate_preflight_on_cpu(
        loaded_config: dict[str, object],
        root: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        context = real_candidate_preflight(
            loaded_config,
            root,
            **kwargs,
        )
        context["assignment"] = {
            **context["assignment"],
            "resource_profile": "cpu_light",
        }
        return context

    def build_model(
        feature_count: int,
        action_dim: int,
        model_config: dict[str, object],
        method_contract: dict[str, object],
    ) -> torch.nn.Module:
        assert feature_count == 1
        assert action_dim > 0
        assert method_contract["calibration_eligible"] is True
        observed["model_config"] = model_config
        return model

    def capture_adamw(
        parameters: object,
        **kwargs: object,
    ) -> torch.optim.AdamW:
        optimizer = original_adamw(parameters, **kwargs)
        observed["optimizer"] = optimizer
        return optimizer

    def controlled_evaluate(
        evaluated_model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
        calibration: torch.Tensor | None = None,
        method_contract: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del norm_mean, norm_std, device, calibration, method_contract
        if loader.dataset.split == "development":
            development_call = int(observed.get("development_call", 0)) + 1
            observed["development_call"] = development_call
            metrics: dict[str, object] = {
                "reconstruction_rmse": 0.1 if development_call == 1 else 1.0,
                "selection_marker": (
                    "selected_epoch" if development_call == 1 else "later_epoch"
                ),
            }
            if development_call == 2:
                selected_payload = torch.load(
                    output_root / "checkpoints/best_dev.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                observed["selected_payload"] = selected_payload
                observed["later_model_state"] = {
                    name: value.detach().cpu().clone()
                    for name, value in evaluated_model.state_dict().items()
                }
                optimizer = observed["optimizer"]
                assert isinstance(optimizer, torch.optim.Optimizer)
                observed["later_optimizer_state"] = deepcopy(optimizer.state_dict())
                model_config = observed["model_config"]
                assert isinstance(model_config, dict)
                model_config["source_metadata"]["revision"] = "later_epoch"
            return metrics
        assert loader.dataset.split == "calibration"
        metrics = {
            "reconstruction_rmse": 0.25,
            "selection_marker": "calibration_on_selected_epoch",
        }
        observed["calibration_metrics"] = deepcopy(metrics)
        return metrics

    def observed_fit_calibration(
        calibrated_model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
        device: torch.device,
        method_contract: dict[str, object] | None = None,
    ) -> torch.Tensor:
        del loader, norm_mean, norm_std, device, method_contract
        selected_payload = observed["selected_payload"]
        assert isinstance(selected_payload, dict)
        restored_state = {
            name: value.detach().cpu().clone()
            for name, value in calibrated_model.state_dict().items()
        }
        _assert_nested_checkpoint_equal(
            restored_state,
            selected_payload["model_state_dict"],
        )
        observed["calibration_model_state"] = restored_state
        calibration_scale = torch.full(
            (len(operational_twin_runner.FAMILY_UIDS),),
            1.5,
        )
        observed["calibration_scale"] = calibration_scale.clone()
        return calibration_scale

    monkeypatch.setattr(operational_twin_runner, "resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        operational_twin_runner,
        "resolve_operational_twin_run_contract",
        lambda args, root, require_output=False: {
            "config": config,
            "cfg": cfg,
            "paths": paths,
        },
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_operational_twin_training_authorized",
        lambda *args: None,
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_candidate_assignment_preflight",
        candidate_preflight_on_cpu,
    )
    execution_context = {
        "execution_authority": {"authority_uid": "candidate-execution-fixture"},
        "resource_environment_receipt": {
            "receipt_uid": "candidate-resource-fixture"
        },
        "resource_receipt_payload": {},
    }
    monkeypatch.setattr(
        operational_twin_runner,
        "require_candidate_execution_authority",
        lambda *args, **kwargs: execution_context,
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_calibration_execution_authority",
        lambda *args, **kwargs: {"authority_uid": "candidate_snapshot_test"},
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "require_assignment_runtime_resource",
        lambda assignment, **kwargs: None,
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "method_protocol_config",
        bounded_method_protocol,
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "tensorize_manifest",
        lambda path, split, loaded_config, root: tensorized[split],
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "ControlledWorldTensorDataset",
        lambda value, encoder: TinyControlledWorldDataset(
            value["split"],
            encoder.output_dim,
        ),
    )
    monkeypatch.setattr(operational_twin_runner, "build_method_model", build_model)
    monkeypatch.setattr(operational_twin_runner.torch.optim, "AdamW", capture_adamw)
    monkeypatch.setattr(operational_twin_runner, "evaluate", controlled_evaluate)
    monkeypatch.setattr(
        operational_twin_runner,
        "fit_calibration",
        observed_fit_calibration,
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "tracked_reference_identity",
        lambda root, path: {
            "path": path.relative_to(root).as_posix(),
            "authority": "test_fixture",
            "receipt_content_opened": False,
        },
    )
    monkeypatch.setattr(
        operational_twin_runner,
        "subprocess",
        SimpleNamespace(
            check_output=lambda *args, **kwargs: "candidate-snapshot-test-commit\n"
        ),
    )

    result = operational_twin_runner.run(
        SimpleNamespace(
            protocol_scope="candidate-lineage",
            assignment_uid=assignment["assignment_uid"],
            method_id=method_id,
            seed=assignment["seed"],
            candidate_lineage_uid=assignment["candidate_lineage_uid"],
            protocol_uid=assignment["protocol_uid"],
            attempt_uid=attempt["attempt_uid"],
            candidate_manifest_hash=candidate_manifest["candidate_manifest_hash"],
            assignment_manifest_hash=assignment_manifest[
                "assignment_manifest_hash"
            ],
            preflight_uid=candidate_manifest["preflight_uid"],
            command_catalog_hash=catalog["command_catalog_hash"],
            output_root=attempt["output_root"],
            candidate_execution_authority=attempt["execution_authority"],
            candidate_resource_environment_receipt=attempt[
                "resource_environment_receipt"
            ],
            resume=False,
            resume_input=None,
            history_input=None,
            checkpoint_input=None,
            optimizer_state_input=None,
            model_state_input=None,
            calibration_state_input=None,
            dry_run=False,
        )
    )

    best_path = output_root / "checkpoints/best_dev.pt"
    last_path = output_root / "checkpoints/last.pt"
    assert best_path.is_file()
    assert last_path.is_file()
    assert best_path != last_path
    assert sha256_file(best_path) != sha256_file(last_path)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    last_payload = torch.load(last_path, map_location="cpu", weights_only=False)
    selected_payload = observed["selected_payload"]
    assert isinstance(selected_payload, dict)

    for payload in (best_payload, last_payload, result):
        assert {
            field: payload[field]
            for field in CANDIDATE_IDENTITY_FIELDS
        } == runtime_identity
        assert payload["attempt_uid"] == attempt["attempt_uid"]
        assert payload["fresh_initialization_receipt"] == initialization
        assert payload["state_resume_used"] is False
        assert payload["output_root"] == attempt["output_root"]
        assert payload["candidate_execution_authority"] == execution_context[
            "execution_authority"
        ]
        assert payload[
            "candidate_resource_environment_receipt"
        ] == execution_context["resource_environment_receipt"]
    assert initialization["parent_candidate_lineage_uid"] is None
    assert initialization["initialization_mode"] == "fresh_random"
    assert initialization["resume_allowed"] is False
    assert initialization["historical_state_use_allowed"] is False
    assert all(
        initialization[field] is None
        for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS
    )

    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "step",
        "dev_metrics",
        "model_config",
    ):
        _assert_nested_checkpoint_equal(best_payload[field], selected_payload[field])
    assert best_payload["epoch"] == 1
    assert best_payload["step"] == 1
    assert best_payload["dev_metrics"]["selection_marker"] == "selected_epoch"
    assert best_payload["model_config"]["source_metadata"]["revision"] == (
        "selected_epoch"
    )
    _assert_nested_checkpoint_equal(
        best_payload["calibration_scale"],
        observed["calibration_scale"],
    )
    expected_calibration_metrics = deepcopy(observed["calibration_metrics"])
    expected_calibration_metrics["calibration_scale_by_family"] = {
        family: float(observed["calibration_scale"][index])
        for index, family in enumerate(operational_twin_runner.FAMILY_UIDS)
    }
    calibration_metrics = best_payload["calibration_metrics"]
    assert isinstance(calibration_metrics, dict)
    assert set(calibration_metrics) == {
        *expected_calibration_metrics,
        *CANDIDATE_IDENTITY_FIELDS,
        "fresh_initialization_receipt",
        "state_resume_used",
        "output_root",
        "candidate_execution_authority",
        "candidate_resource_environment_receipt",
    }
    _assert_nested_checkpoint_equal(
        {
            field: calibration_metrics[field]
            for field in expected_calibration_metrics
        },
        expected_calibration_metrics,
    )
    assert {
        field: calibration_metrics[field]
        for field in CANDIDATE_IDENTITY_FIELDS
    } == runtime_identity
    assert calibration_metrics["fresh_initialization_receipt"] == initialization
    _assert_nested_checkpoint_equal(
        observed["calibration_model_state"],
        selected_payload["model_state_dict"],
    )

    assert last_payload["epoch"] == 2
    assert last_payload["step"] == 2
    assert last_payload["dev_metrics"]["selection_marker"] == "later_epoch"
    assert last_payload["model_config"]["source_metadata"]["revision"] == (
        "later_epoch"
    )
    _assert_nested_checkpoint_equal(
        last_payload["model_state_dict"],
        observed["later_model_state"],
    )
    _assert_nested_checkpoint_equal(
        last_payload["optimizer_state_dict"],
        observed["later_optimizer_state"],
    )
    assert any(
        not torch.equal(
            last_payload["model_state_dict"][name],
            best_payload["model_state_dict"][name],
        )
        for name in best_payload["model_state_dict"]
    )
    best_optimizer_slots = best_payload["optimizer_state_dict"]["state"]
    last_optimizer_slots = last_payload["optimizer_state_dict"]["state"]
    assert best_optimizer_slots.keys() == last_optimizer_slots.keys()
    assert all(
        float(last_optimizer_slots[key]["step"])
        > float(best_optimizer_slots[key]["step"])
        for key in best_optimizer_slots
    )
    assert any(
        not torch.equal(
            last_optimizer_slots[key]["exp_avg"],
            best_optimizer_slots[key]["exp_avg"],
        )
        for key in best_optimizer_slots
    )
    assert any(
        not torch.equal(
            last_optimizer_slots[key]["exp_avg_sq"],
            best_optimizer_slots[key]["exp_avg_sq"],
        )
        for key in best_optimizer_slots
    )
    assert result["selected_checkpoint_epoch"] == 1
    assert result["epochs_completed"] == 2
    assert result["steps_completed"] == 2

    preflight_path = tmp_path / protocol_paths["preflight_authority"]
    tampered_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    tampered_preflight["command_catalog_hash"] = "0" * 64
    preflight_path.write_text(
        json.dumps(tampered_preflight, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_artifacts = operational_twin_protocol.load_candidate_protocol_artifacts(
        config,
        tmp_path,
    )
    tamper_codes = {
        row["code"]
        for row in operational_twin_protocol.validate_candidate_artifacts(
            config,
            tmp_path,
            *tampered_artifacts,
            require_attempt_roots_absent=False,
        )
    }
    assert "candidate_preflight_drift" in tamper_codes


def test_gpu_probe_receipt_binds_direct_stdout_job_id_sacct_and_hash(tmp_path: Path) -> None:
    config = _config()
    stdout = tmp_path / "gpu_probe.txt"
    stdout.write_text(
        'SLURM_JOB_ID=999\nHOSTNAME=gpu03\nCUDA_VISIBLE_DEVICES=2\n{"cuda_available": true, "torch_version": "2.9.0"}\n',
        encoding="utf-8",
    )
    receipt = _resource_receipt(config)
    receipt["probes"]["gpu_probe"].update(
        {
            "job_id": "999",
            "stdout_path": stdout.as_posix(),
            "stdout_sha256": sha256_file(stdout),
            "environment": {"probe_stdout": stdout.read_text(encoding="utf-8")},
            "sacct_evidence": "999|NrnSift_protocol_gpu_probe_retry|gpu1|gpu03|COMPLETED|0:0|1|4G|00:00:11",
        }
    )
    assert validate_resource_receipt(receipt, config_hash=receipt["config_hash"]) == []
    receipt["probes"]["gpu_probe"]["stdout_sha256"] = "0" * 64
    codes = {row["code"] for row in validate_resource_receipt(receipt, config_hash=receipt["config_hash"])}
    assert "resource_probe_stdout_hash_mismatch" in codes


def test_assignment_mode_allows_frozen_assignment_output_root() -> None:
    args = SimpleNamespace(
        config="Libs/configs/model/operational_twin.yaml",
        training_manifest="Input/processed/controlled_neural_world/open_action_identification/training_manifest.json",
        development_manifest="Input/processed/controlled_neural_world/open_action_identification/development_manifest.json",
        calibration_manifest="Input/processed/controlled_neural_world/open_action_identification/calibration_manifest.json",
        package_receipt="Input/processed/controlled_neural_world/open_action_identification/package_receipt.json",
        open_data_qualification="Output/audits/controlled_world_open_data_qualification.json",
        output_root="Output/results/operational_twin/development_training/persistence_no_action/seed_360101",
        assignment_uid="operational_twin__persistence_no_action__seed_360101",
        method_id="persistence_no_action",
        seed=360101,
    )
    contract = resolve_operational_twin_run_contract(args, ROOT, require_output=False)
    assert contract["paths"]["output_root"].as_posix().endswith("persistence_no_action/seed_360101")


def test_method_protocol_budgets_are_full_round37_values() -> None:
    config = _config()
    expected = {
        "persistence_no_action": (0, 0),
        "linear_or_delta": (200, 20000),
        "same_backbone_no_action": (300, 50000),
        "deterministic_action_conditioned": (350, 65000),
        "full_uncertainty_aware_action_conditioned": (400, 80000),
    }
    for method_id, (epochs, max_steps) in expected.items():
        budget = method_protocol_config(config, method_id)["budget"]
        assert int(budget["epochs"]) == epochs
        assert int(budget["max_steps"]) == max_steps
    assert int(config["operational_twin"]["training"]["seed"]) == 310031
    assert 310031 not in config["operational_twin_training_protocol"]["seeds"]


def test_capacity_checks_use_method_specific_protocol_config() -> None:
    config = _config()
    protocol = method_protocol_config(config, "same_backbone_no_action")
    model_config = model_config_for_method(config["operational_twin"]["model"], protocol)
    assert model_config["hidden_dim"] == protocol["capacity"]["hidden_dim"]
    model = build_method_model(13, 59, model_config, resolve_method_execution_contract("same_backbone_no_action"))
    assert count_trainable_parameters(model) >= int(protocol["capacity"]["trainable_parameters_expected_min"])
    bad_protocol = deepcopy(protocol)
    bad_protocol["capacity"]["trainable_parameters_expected_min"] = 10**12
    with pytest.raises(OperationalTwinProtocolError, match="capacity_below_protocol_minimum"):
        enforce_capacity(bad_protocol, count_trainable_parameters(model))


def test_formal_assignment_resource_profile_forbids_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_GPUS", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    require_assignment_runtime_resource({"resource_profile": "cpu_light"})
    with pytest.raises(OperationalTwinProtocolError, match="gpu_single_assignment_requires_cuda"):
        require_assignment_runtime_resource({"resource_profile": "gpu_single"})


def test_formal_gpu_assignment_requires_explicit_slurm_gpu_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "cpu")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(OperationalTwinProtocolError, match="gpu_single_assignment_requires_slurm_gpu_allocation"):
        require_assignment_runtime_resource({"resource_profile": "gpu_single"})


@pytest.mark.parametrize(
    ("resource_profile", "method_id"),
    [
        ("cpu_light", "persistence_no_action"),
        ("gpu_single", "linear_or_delta"),
    ],
)
def test_candidate_assignment_accepts_only_exact_frozen_runtime_placement(
    monkeypatch: pytest.MonkeyPatch,
    resource_profile: str,
    method_id: str,
) -> None:
    requirements = deepcopy(CANDIDATE_RESOURCE_PROFILES[resource_profile])
    slurm, environment = _set_candidate_runtime_environment(
        monkeypatch,
        requirements,
        torch_version=str(torch.__version__),
    )
    monkeypatch.setattr(
        operational_twin_runner.os,
        "uname",
        lambda: SimpleNamespace(nodename=requirements["nodelist"]),
    )
    uses_gpu = resource_profile == "gpu_single"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: uses_gpu)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if uses_gpu else 0)
    assignment = {
        "method_id": method_id,
        "resource_profile": resource_profile,
        "resource_requirements": requirements,
    }
    receipt = {
        "resource_requirements": requirements,
        "resource_allocation": requirements,
        "slurm": slurm,
        "environment": environment,
    }
    require_assignment_runtime_resource(
        assignment,
        resource_receipt=receipt,
    )


def test_formal_assignment_resource_profile_requires_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(OperationalTwinProtocolError, match="formal_assignment_requires_slurm"):
        require_assignment_runtime_resource({"resource_profile": "cpu_light"})


def test_calibration_execution_authority_is_bound_to_current_round_sources(
    tmp_path: Path,
) -> None:
    relative_authority = (
        "Output/audits/operational_twin_calibration_execution_authority.json"
    )
    authority_content = operational_twin_protocol.read_repo_bytes(
        ROOT, relative_authority
    )
    authority = json.loads(authority_content.decode("utf-8"))
    authority_path = tmp_path / relative_authority
    authority_path.parent.mkdir(parents=True)
    authority_path.write_bytes(authority_content)
    for source in authority["source_files"]:
        target = tmp_path / source["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            operational_twin_protocol.read_repo_bytes(ROOT, source["path"])
        )
    authority = require_calibration_execution_authority(
        _config(),
        tmp_path,
        authority_path,
        method_id="full_uncertainty_aware_action_conditioned",
    )
    assert authority is not None
    assert authority["authority_uid"] == "operational_twin_development_calibration_execution"


def test_non_uq_method_does_not_require_calibration_execution_authority(tmp_path: Path) -> None:
    assert require_calibration_execution_authority(
        _config(),
        ROOT,
        tmp_path / "missing.json",
        method_id="deterministic_action_conditioned",
    ) is None


def test_runtime_evidence_requires_explicit_gpu_allocation() -> None:
    assignment = {
        "assignment_uid": "operational_twin__linear_or_delta__seed_360101",
        "method_id": "linear_or_delta",
        "seed": 360101,
        "resource_profile": "gpu_single",
    }
    status = {"status": "COMPLETED"}
    provenance = {
        "assignment_uid": assignment["assignment_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "scheduler_identity": {"slurm_job_id": "123", "hostname": "cpu04"},
        "device_provenance": {
            "resource_profile": "gpu_single",
            "device_type": "cuda",
            "cuda_available": True,
            "slurm_job_gpus": None,
            "slurm_gpus_on_node": None,
            "slurm_step_gpus": None,
        },
    }
    blockers = validate_assignment_runtime_evidence(assignment, status, provenance)
    assert [row["code"] for row in blockers] == ["runtime_gpu_allocation_missing"]


def test_runtime_evidence_accepts_gpu_allocation_on_authorized_compute_node() -> None:
    assignment = {
        "assignment_uid": "operational_twin__linear_or_delta__seed_360101",
        "method_id": "linear_or_delta",
        "seed": 360101,
        "resource_profile": "gpu_single",
    }
    status = {"status": "COMPLETED"}
    provenance = {
        "assignment_uid": assignment["assignment_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "scheduler_identity": {"slurm_job_id": "123", "hostname": "cpu04"},
        "device_provenance": {
            "resource_profile": "gpu_single",
            "device_type": "cuda",
            "cuda_available": True,
            "slurm_job_gpus": "1",
            "slurm_gpus_on_node": "1",
            "slurm_step_gpus": None,
        },
    }
    assert validate_assignment_runtime_evidence(assignment, status, provenance) == []

def _positive_retry_audit(config: dict[str, object]) -> dict[str, object]:
    audit = retry_authority.adjudicate_retry_authority(config, ROOT)
    audit = deepcopy(audit)
    for index, row in enumerate(audit["rows"]):
        row["predecessor_evidence_kind"] = "materialized_evaluation_failure"
        row["predecessor_failure_class"] = "infrastructure_interruption"
        row["historical_recovery_classification"] = "ELIGIBLE"
        row["predecessor_identity"] = {
            "terminal_status": f"{index + 1:040x}",
            "phase_receipt": f"{index + 101:040x}",
        }
        row["predicates"] = {key: True for key in row["predicates"]}
        row["result"] = retry_authority.AUTHORIZED
    contract = config["operational_twin_candidate_retry_authority"]
    audit.update(
        {
            "terminal_decision": retry_authority.POSITIVE,
            "next_direction": contract["next_directions"]["positive"],
            "authorized_row_count": 15,
            "successor_required_row_count": 0,
            "positive_artifacts_materialized": False,
        }
    )
    audit["decision_uid"] = retry_authority._decision_uid(audit)
    return audit


def test_retry_authority_positive_overlay_changes_only_protocol_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config_stack(ROOT / "Libs/configs/config_stack.yaml")
    audit = _positive_retry_audit(config)
    monkeypatch.setattr(
        retry_authority,
        "adjudicate_retry_authority",
        lambda loaded_config, root: audit,
    )
    successor = retry_authority.derive_positive_successor(config, ROOT, audit)
    assignments = successor["assignment_manifest"]
    assert assignments["assignment_count"] == 15
    assert assignments["attempt_slot_count"] == 30
    assert len(successor["assignment_commands"]["commands"]) == 30
    assert successor["candidate_lineage_manifest"]["candidate_lineage_uid"] == config["operational_twin_candidate_retry_authority"]["candidate_lineage_uid"]
    predecessor = operational_twin_protocol.read_repo_json(
        ROOT,
        config["operational_twin_candidate_retry_authority"]["repaired_artifacts"][
            "assignment_manifest"
        ],
    )
    assert [row["assignment_uid"] for row in assignments["assignments"]] == [
        row["assignment_uid"] for row in predecessor["assignments"]
    ]
    assert [
        [attempt["attempt_uid"] for attempt in row["attempts"]]
        for row in assignments["assignments"]
    ] == [
        [attempt["attempt_uid"] for attempt in row["attempts"]]
        for row in predecessor["assignments"]
    ]
    preflight = successor["preflight_authority"]
    assert preflight["schema_version"] == retry_authority.POSITIVE_PREFLIGHT_SCHEMA_VERSION
    assert preflight["authorized_command_indices"] == list(range(15, 30))
    assert preflight["historical_command_indices"] == list(range(15))
    assert preflight["retry_attempt_roots_absent_at_freeze"] is True
    assert successor["preflight_report"] == retry_authority.render_positive_successor_report(preflight)
    bound_paths = {row["path"] for row in preflight["source_identities"]}
    assert set(config["operational_twin_candidate_retry_authority"]["source_bindings"]).issubset(bound_paths)


def test_retry_authority_builder_blocks_failed_positive_prepublication_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_config())
    audit = _positive_retry_audit(config)
    package = {
        "candidate_lineage_manifest": {
            "protocol_uid": "protocol",
            "preflight_uid": "preflight",
            "candidate_manifest_hash": "candidate",
        },
        "assignment_manifest": {"assignment_manifest_hash": "assignment"},
        "assignment_commands": {"command_catalog_hash": "commands"},
        "preflight_authority": {},
        "preflight_report": "report\n",
    }
    publication_calls: list[object] = []
    monkeypatch.setattr(operational_twin_protocol_builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "adjudicate_retry_authority",
        lambda *args, **kwargs: audit,
    )
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "derive_positive_successor",
        lambda *args, **kwargs: package,
    )
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "validate_positive_successor_package",
        lambda *args, **kwargs: [
            {"code": "injected_positive_prepublication_failure", "message": "blocked"}
        ],
    )
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "_publish_candidate_artifacts",
        lambda *args, **kwargs: publication_calls.append(args),
    )
    with pytest.raises(ValueError, match="positive_prepublication_invalid"):
        operational_twin_protocol_builder._build_candidate_retry_authority(
            config,
            "2026-09-12",
        )
    assert publication_calls == []


def test_retry_authority_positive_publication_collision_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = deepcopy(_config())
    audit = _positive_retry_audit(config)
    package = {
        "candidate_lineage_manifest": {
            "protocol_uid": "protocol",
            "preflight_uid": "preflight",
            "candidate_manifest_hash": "candidate",
        },
        "assignment_manifest": {"assignment_manifest_hash": "assignment"},
        "assignment_commands": {"command_catalog_hash": "commands"},
        "preflight_authority": {},
        "preflight_report": "report\n",
    }
    monkeypatch.setattr(operational_twin_protocol_builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "adjudicate_retry_authority",
        lambda *args, **kwargs: audit,
    )
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "derive_positive_successor",
        lambda *args, **kwargs: package,
    )
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "validate_positive_successor_package",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "render_positive_successor_report",
        lambda preflight: "report\n",
    )
    decision_path = (
        tmp_path
        / config["operational_twin_candidate_retry_authority"]["output_paths"][
            "decision_audit"
        ]
    )
    decision_path.parent.mkdir(parents=True)
    decision_path.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="candidate_protocol_collision"):
        operational_twin_protocol_builder._build_candidate_retry_authority(
            config,
            "2026-09-12",
        )
    assert decision_path.read_text(encoding="utf-8") == "preserve"

    decision_path.unlink()
    original_write = candidate_execution.write_bytes_exclusive
    calls = 0

    def fail_after_first_write(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected positive publication failure")
        original_write(path, content)

    monkeypatch.setattr(
        candidate_execution,
        "write_bytes_exclusive",
        fail_after_first_write,
    )
    with pytest.raises(OSError, match="injected positive publication failure"):
        operational_twin_protocol_builder._build_candidate_retry_authority(
            config,
            "2026-09-12",
        )
    paths = config["operational_twin_candidate_retry_authority"]["output_paths"]
    assert all(
        not os.path.lexists(tmp_path / paths[key])
        for key in (
            "decision_audit",
            "decision_report",
            "candidate_lineage_manifest",
            "assignment_manifest",
            "assignment_commands",
            "preflight_authority",
            "preflight_report",
        )
    )


def test_successor_builder_recomputes_before_exact_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_paths = {
        "candidate_lineage_manifest": "Output/results/protocol/candidate.json",
        "assignment_manifest": "Output/results/protocol/assignments.json",
        "assignment_commands": "Output/results/protocol/commands.json",
        "preflight_authority": "Output/audits/preflight.json",
        "protocol_audit": "Output/audits/protocol.json",
        "preflight_report": "Output/reports/preflight.md",
        "protocol_report": "Output/reports/protocol.md",
    }
    package = {
        key: (path, f"{key}\n".encode("utf-8"))
        for key, path in output_paths.items()
    }
    audit = {
        "terminal_decision": (
            operational_twin_protocol_builder.SUCCESSOR_INVALID_TERMINAL.replace(
                "INVALID", "FROZEN"
            )
        ),
        "candidate_lineage_uid": "successor-lineage",
        "protocol_uid": "successor-protocol",
        "preflight_uid": "successor-preflight",
        "assignment_count": 15,
        "attempt_count": 30,
        "command_count": 30,
        "publication_file_count": 7,
    }
    package["protocol_audit"] = (
        output_paths["protocol_audit"],
        (json.dumps(audit, sort_keys=True) + "\n").encode("utf-8"),
    )
    calls: list[object] = []
    monkeypatch.setattr(operational_twin_protocol_builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "successor_publication_paths",
        lambda config: output_paths,
    )

    def build_package(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return deepcopy(package), [{"path": "preserved"}], []

    published: list[object] = []
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "build_successor_package",
        build_package,
    )
    monkeypatch.setattr(
        operational_twin_protocol_builder,
        "_publish_candidate_artifacts",
        lambda selected, **kwargs: published.append((selected, kwargs)) or "created",
    )
    config = {
        "operational_twin_successor_lineage_protocol": {
            "authoritative_date": "2026-09-13",
            "output_namespace": {
                "protocol_root": "Output/results/protocol",
            },
        }
    }
    result = (
        operational_twin_protocol_builder._build_candidate_lineage_successor_protocol(
            config,
            "2026-09-13",
            pre_change_identity_manifest_path=tmp_path / "pre-change.json",
            validation_evidence_path=tmp_path / "source-gate.json",
        )
    )
    assert len(calls) == 2
    assert len(published) == 1
    assert [path.relative_to(tmp_path).as_posix() for path, _ in published[0][0]] == list(
        output_paths.values()
    )
    assert result["status"] == "PASS"
    assert result["publication_file_count"] == 7
    assert result["candidate_dispatch_authorized"] is False


def test_candidate_publisher_rejects_symlink_and_rolls_back_created_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operational_twin_protocol_builder, "REPO_ROOT", tmp_path)
    (tmp_path / "Output").mkdir()
    os.symlink(tmp_path / "outside", tmp_path / "Output/audits")
    with pytest.raises(ValueError, match="candidate_protocol_symlink_path"):
        operational_twin_protocol_builder._publish_candidate_artifacts(
            [(tmp_path / "Output/audits/preflight.json", b"payload")],
            protocol_root=None,
        )

    (tmp_path / "Output/audits").unlink()
    original_write = candidate_execution.write_bytes_exclusive
    write_calls = []

    def fail_second_write(path: Path, content: bytes) -> None:
        position = len(write_calls)
        assert (path, content) == targets[position]
        write_calls.append(path)
        if len(write_calls) == 2:
            assert targets[0][0].read_bytes() == targets[0][1]
            raise OSError("injected successor publication failure")
        original_write(path, content)

    monkeypatch.setattr(
        candidate_execution, "write_bytes_exclusive", fail_second_write
    )
    protocol_root = tmp_path / "Output/results/successor_protocol"
    targets = [
        (protocol_root / "candidate.json", b"candidate"),
        (tmp_path / "Output/audits/preflight.json", b"preflight"),
    ]
    with pytest.raises(OSError, match="injected successor publication failure"):
        operational_twin_protocol_builder._publish_candidate_artifacts(
            targets,
            protocol_root=protocol_root,
        )
    assert write_calls == [path for path, _ in targets]
    assert all(not os.path.lexists(path) for path, _ in targets)
    assert not protocol_root.exists()
    assert not (tmp_path / "Output/results").exists()
    assert not (tmp_path / "Output/audits").exists()
    assert (tmp_path / "Output").is_dir()


@pytest.fixture
def committed_source_checkout(tmp_path):
    """Real source-only Git history; no candidate data or execution authority."""
    root = tmp_path / "committed_source"
    (root / "Libs/tree").mkdir(parents=True)
    contents = {
        "Libs/first.py": b"value = 1\n",
        "Libs/config.yaml": b"setting: exact\n",
        "Libs/equal.py": b"value = 1\n",
        "Libs/tree/member": b"source-only fixture\n",
    }
    for relative, content in contents.items():
        (root / relative).write_bytes(content)

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Non-authoritative committed source fixture")
    git("config", "user.email", "fixture@invalid.example")
    git("add", "--", *contents)
    git("commit", "-q", "-m", "Source-only fixture")
    source = git("rev-parse", "HEAD")
    (root / "review.txt").write_text("source-preserving descendant\n")
    git("add", "--", "review.txt")
    git("commit", "-q", "-m", "Source-preserving descendant")
    descendant = git("rev-parse", "HEAD")
    (root / "Libs/first.py").write_bytes(b"value = 2\n")
    git("add", "--", "Libs/first.py")
    git("commit", "-q", "-m", "Changed committed source")
    changed = git("rev-parse", "HEAD")
    unrelated = git("commit-tree", source + "^{tree}", "-m", "Unrelated source tree")
    rows = [{"path": path, "sha256": hashlib.sha256(contents[path]).hexdigest()}
            for path in ("Libs/config.yaml", "Libs/first.py")]
    return SimpleNamespace(root=root, source=source, descendant=descendant,
                           changed=changed, unrelated=unrelated, rows=rows, contents=contents)


def _committed_source_context(case, rows=None):
    return operational_twin_protocol.ResolvedCandidateExecutionProtocol(
        protocol={}, candidate_manifest={}, assignment_manifest={}, command_catalog={},
        preflight={}, execution={}, binding={},
        handoff={"manifest": {"executing_source": {
            "source_commit": case.source, "source_files": deepcopy(case.rows if rows is None else rows),
        }}},
    )


def test_executing_commit_batch_preserves_ancestry_and_source_validation(committed_source_checkout, monkeypatch):
    case = committed_source_checkout
    actual_run = subprocess.run
    calls = []

    def counted_run(argv, **kwargs):
        calls.append((list(argv), kwargs.get("input")))
        return actual_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", counted_run)
    equal_rows = [case.rows[1], {"path": "Libs/equal.py", "sha256": case.rows[1]["sha256"]}]
    for reference, rows in ((case.source, case.rows), (case.source, case.rows),
                            (case.descendant, list(reversed(case.rows))), (case.source, equal_rows)):
        calls.clear()
        operational_twin_protocol._require_executing_commit(
            case.root, reference, _committed_source_context(case, rows),
        )
        assert calls == [
            (["git", "merge-base", "--is-ancestor", case.source, reference], None),
            (["git", "cat-file", "--batch"],
             "".join(f"{reference}:{row['path']}\n" for row in rows).encode()),
        ]
    calls.clear()
    with pytest.raises(OperationalTwinProtocolError, match="^candidate_execution_commit_source_drift$"):
        operational_twin_protocol._require_executing_commit(case.root, case.changed, _committed_source_context(case))
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(OperationalTwinProtocolError, match="^candidate_unreviewed_execution_commit$"):
        operational_twin_protocol._require_executing_commit(case.root, case.unrelated, _committed_source_context(case))
    assert calls == [(["git", "merge-base", "--is-ancestor", case.source, case.unrelated], None)]
    calls.clear()
    operational_twin_protocol._require_executing_commit(case.root, case.source, None)
    assert calls == []
    operational_twin_protocol._require_executing_commit(case.root, case.source, _committed_source_context(case, []))
    assert calls == [(["git", "merge-base", "--is-ancestor", case.source, case.source], None)]


@pytest.mark.parametrize("kind,reason,batch_count", [
    ("missing", "handoff_git_evidence_unavailable", 1),
    ("tree", "handoff_git_evidence_unavailable", 1),
    ("duplicate", "handoff_identity_path_duplicate", 0),
    ("invalid", "handoff_identity_path_invalid", 0),
    ("input", "handoff_identity_path_invalid", 0),
])
def test_executing_commit_batch_rejects_invalid_sources(committed_source_checkout, monkeypatch, kind, reason, batch_count):
    case = committed_source_checkout
    rows = deepcopy(case.rows)
    if kind == "duplicate":
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]["path"] = {"missing": "Libs/missing.py", "tree": "Libs/tree",
                           "invalid": "../outside.py", "input": "Input/unopened.json"}[kind]
    actual_run = subprocess.run
    calls = []

    def counted_run(argv, **kwargs):
        calls.append(list(argv))
        return actual_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", counted_run)
    with pytest.raises(OperationalTwinProtocolError, match=reason):
        operational_twin_protocol._require_executing_commit(case.root, case.source, _committed_source_context(case, rows))
    assert calls.count(["git", "cat-file", "--batch"]) == batch_count
    assert len(calls) == 1 + batch_count


@pytest.mark.parametrize("kind", [
    "invalid_utf8", "unicode_digit", "invalid_numeric", "field_count", "object_type",
    "object_id", "empty", "header_without_newline", "truncated_payload", "stderr",
    "trailing_record", "process_failure", "os_failure", "swapped_payloads",
])
def test_executing_commit_batch_rejects_malformed_transport(committed_source_checkout, monkeypatch, kind):
    """Synthetic transport negatives exercise the real batch parser and caller."""
    case = committed_source_checkout
    actual_run = subprocess.run
    payloads = [case.contents[row["path"]] for row in case.rows]

    def frame(payload):
        return b"0" * 40 + b" blob " + str(len(payload)).encode() + b"\n" + payload + b"\n"

    valid = b"".join(frame(payload) for payload in payloads)
    malformed = {
        "invalid_utf8": b"\xff\n",
        "unicode_digit": ("0" * 40 + " blob ²\n").encode(),
        "invalid_numeric": b"0" * 40 + b" blob nonnumeric\n",
        "field_count": b"0" * 40 + b" blob\n",
        "object_type": b"0" * 40 + b" tree 0\n\n",
        "object_id": b"g" * 40 + b" blob 0\n\n",
        "empty": b"",
        "header_without_newline": b"0" * 40 + b" blob 1",
        "truncated_payload": b"0" * 40 + b" blob 9\nx\n",
        "stderr": valid,
        "trailing_record": valid + b"unrequested\n",
        "swapped_payloads": b"".join(frame(payload) for payload in reversed(payloads)),
    }
    batches = []

    def inject_transport(argv, **kwargs):
        if list(argv) != ["git", "cat-file", "--batch"]:
            return actual_run(argv, **kwargs)
        batches.append(list(argv))
        if kind == "process_failure":
            raise subprocess.CalledProcessError(1, argv, stderr=b"injected transport failure")
        if kind == "os_failure":
            raise OSError("injected process startup failure")
        return subprocess.CompletedProcess(argv, 0, malformed[kind], b"unexpected" if kind == "stderr" else b"")

    monkeypatch.setattr(subprocess, "run", inject_transport)
    reason = ("candidate_execution_commit_batch_malformed" if kind in {"invalid_utf8", "unicode_digit"}
              else "candidate_execution_commit_source_drift" if kind == "swapped_payloads"
              else "handoff_git_evidence_unavailable")
    with pytest.raises(OperationalTwinProtocolError, match=reason) as caught:
        operational_twin_protocol._require_executing_commit(case.root, case.source, _committed_source_context(case))
    assert batches == [["git", "cat-file", "--batch"]]
    if kind in {"invalid_utf8", "unicode_digit"}:
        assert isinstance(caught.value.__cause__, ValueError)
        assert not isinstance(caught.value.__cause__, OperationalTwinProtocolError)


@pytest.mark.parametrize("mutation", ["short", "extra", "path"])
def test_executing_commit_batch_rejects_returned_row_mismatch(committed_source_checkout, monkeypatch, mutation):
    """Explicit software tamper control after genuine batch parsing."""
    case = committed_source_checkout
    actual_identities = handoff._identities

    def altered_identities(*args, **kwargs):
        rows = actual_identities(*args, **kwargs)
        if mutation == "short":
            return rows[:-1]
        if mutation == "extra":
            return rows + [dict(rows[0])]
        rows[0]["path"] = rows[1]["path"]
        return rows

    monkeypatch.setattr(handoff, "_identities", altered_identities)
    with pytest.raises(OperationalTwinProtocolError, match="candidate_execution_commit_batch_malformed: row"):
        operational_twin_protocol._require_executing_commit(case.root, case.source, _committed_source_context(case))


def test_executing_commit_batch_preserves_existing_domain_error(committed_source_checkout, monkeypatch):
    case = committed_source_checkout
    original = OperationalTwinProtocolError("specific_committed_source_failure")

    def fail_reader(*args, **kwargs):
        raise original

    monkeypatch.setattr(handoff, "_identities", fail_reader)
    with pytest.raises(OperationalTwinProtocolError) as caught:
        operational_twin_protocol._require_executing_commit(case.root, case.source, _committed_source_context(case))
    assert caught.value is original


@pytest.mark.parametrize("first", ["operational_twin_training_protocol", "operational_twin_successor_execution_handoff"])
def test_executing_commit_batch_import_orders(first):
    second = ("operational_twin_successor_execution_handoff" if first == "operational_twin_training_protocol"
              else "operational_twin_training_protocol")
    script = """import importlib, sys
sys.path.insert(0, sys.argv[1])
modules = [importlib.import_module('Libs.utils.evaluation.' + name) for name in sys.argv[2:]]
assert modules[0].OperationalTwinProtocolError is modules[1].OperationalTwinProtocolError
print('IMPORTED_BOTH_ORDERS_WITH_SHARED_PROTOCOL_ERROR')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT), first, second],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "IMPORTED_BOTH_ORDERS_WITH_SHARED_PROTOCOL_ERROR"
    assert result.stderr == ""


def test_executing_commit_batch_uses_native_registered_guard(committed_source_checkout, monkeypatch):
    """Reuse exact guard bodies at the subprocess boundary, with genuine Git."""
    case = committed_source_checkout
    source = (ROOT / "Libs/tests/test_operational_twin_successor_execution_handoff.py").read_text()
    assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "_SCHEDULER_RECEIPT_READ_BOUNDARY_FIXTURE"
                              for target in node.targets))
    literal = ast.literal_eval(assignment.value)
    names = {"_ce_frame", "_ce_deny", "_ce_relative", "_ce_ref", "_ce_payload", "_ce_git", "_ce_checked_run"}
    nodes = [node for node in ast.parse(literal).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = dict(sys=sys, os=os, Path=Path, PurePosixPath=PurePosixPath, handoff=handoff,
                     protocol=operational_twin_protocol, runtime=candidate_execution,
                     trainer=operational_twin_runner, root=case.root, _ce_refs={case.source},
                     _ce_sources={row["path"] for row in case.rows}, _ce_metadata=set(),
                     _ce_active_batch=None, _ce_reads=Counter())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<maintained-committed-source-guard>", "exec"), namespace)
    actual_run = subprocess.run
    executed = []

    def native_after_guard(argv, **kwargs):
        namespace["_ce_git"](argv, kwargs.get("cwd"))
        executed.append(list(argv))
        return actual_run(argv, **kwargs)

    namespace["_ce_real_run"] = native_after_guard
    checked_run = namespace["_ce_checked_run"]
    monkeypatch.setattr(subprocess, "run", checked_run)
    operational_twin_protocol._require_executing_commit(case.root, case.source, _committed_source_context(case))
    assert executed == [["git", "merge-base", "--is-ancestor", case.source, case.source],
                        ["git", "cat-file", "--batch"]]
    assert namespace["_ce_active_batch"] is None
    assert all(namespace["_ce_reads"][("batch_payload", case.source, row["path"])] == 1 for row in case.rows)
    before = list(executed)
    with pytest.raises(PermissionError, match="batch_frame_or_root"):
        checked_run(["git", "cat-file", "--batch"], cwd=case.root, input=b"", capture_output=True, check=True)
    with pytest.raises(PermissionError, match="unregistered_batch"):
        namespace["_ce_git"](["git", "cat-file", "--batch"], case.root)
    with pytest.raises(PermissionError, match="payload_ref_or_path"):
        namespace["_ce_payload"](case.source, "Input/unopened.json")
    assert executed == before
    rejections = []

    def wrong_stdin(argv, **kwargs):
        if list(argv) == ["git", "cat-file", "--batch"]:
            kwargs["input"] += b"\n"
        try:
            return checked_run(argv, **kwargs)
        except PermissionError as error:
            rejections.append(str(error))
            raise

    monkeypatch.setattr(subprocess, "run", wrong_stdin)
    with pytest.raises(OperationalTwinProtocolError, match="handoff_git_evidence_unavailable"):
        operational_twin_protocol._require_executing_commit(case.root, case.source, _committed_source_context(case))
    assert rejections == ["scheduler_receipt_boundary_forbidden: batch_stdin"]
    assert executed == before + [["git", "merge-base", "--is-ancestor", case.source, case.source]]
    assert namespace["_ce_active_batch"] is None


@pytest.mark.parametrize("case", ["default_old", "gate_metadata", "unknown_group", "mixed_groups", "cross_group", "unknown_node", "command_gate_preflight", "command_with_old", "command_with_gate", "command_mixed"])
def test_training_protocol_witness_caller_group_membership(case: str) -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    old = training_protocol_witness._case_table(source)
    gate = training_protocol_witness._group_case_table(source, "gate_metadata")
    assert len(old) == len(gate) == 12 and not set(old).intersection(gate)
    if case == "default_old":
        assert training_protocol_witness._group_case_table(source) == old
        assert training_protocol_witness._group_case_table(source, selected_nodeids=list(old)) == old
    elif case == "gate_metadata":
        assert training_protocol_witness._group_case_table(source, "gate_metadata", list(gate)) == gate
    elif case == "unknown_group":
        with pytest.raises(ValueError, match="unknown caller group"):
            training_protocol_witness._group_case_table(source, "unknown")
    elif case.startswith("command_"):
        command = training_protocol_witness._group_case_table(source, "command_gate_preflight")
        assert len(command) == 1 and not set(command).intersection({*old, *gate})
        if case == "command_gate_preflight":
            assert training_protocol_witness._group_case_table(source, "command_gate_preflight", list(command)) == command
        else:
            if case == "command_with_old":
                selected = [next(iter(old))]
            elif case == "command_with_gate":
                selected = [next(iter(gate))]
            else:
                assert case == "command_mixed"
                selected = [*command, next(iter(gate))]
            with pytest.raises(ValueError, match="caller group selectors"):
                training_protocol_witness._group_case_table(source, "command_gate_preflight", selected)
    else:
        if case == "mixed_groups":
            selected = [next(iter(old)), *list(gate)[1:]]
        elif case == "cross_group":
            selected = list(old)
        else:
            assert case == "unknown_node"
            selected = ["unknown-nodeid", *list(gate)[1:]]
        with pytest.raises(ValueError, match="caller group selectors"):
            training_protocol_witness._group_case_table(source, "gate_metadata", selected)
