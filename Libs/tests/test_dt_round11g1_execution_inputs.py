from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from Libs.scripts.builders.build_dt_round11g1_dependency_closure import _audit_linkage_contract
from Libs.utils.data.round11g_execution_inputs import AUDIT_ROOT_ENV
from Libs.utils.data.round11g_execution_inputs import BUNDLE_ENV
from Libs.utils.data.round11g_execution_inputs import BUNDLE_SHA256_ENV
from Libs.utils.data.round11g_execution_inputs import OUTPUT_ROOT_ENV
from Libs.utils.data.round11g_execution_inputs import PROCESSED_ROOT_ENV
from Libs.utils.data.round11g_execution_inputs import SCHEMA_VERSION
from Libs.utils.data.round11g_execution_inputs import build_execution_bwrap_command
from Libs.utils.data.round11g_execution_inputs import canonical_sha256
from Libs.utils.data.round11g_execution_inputs import load_execution_input_bundle
from Libs.utils.data.round11g_execution_inputs import resolve_execution_input
from Libs.utils.data.round11g_execution_inputs import run_execution_namespace_smoke
from Libs.utils.data.round11g_execution_inputs import run_with_execution_inputs


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict[str, str], Path, str, str]:
    processed_root = tmp_path / "processed"
    processed_path = processed_root / "simulation_audit" / "simulation_qc_summary.json"
    processed_path.parent.mkdir(parents=True)
    processed_path.write_text('{"status":"available"}\n', encoding="utf-8")
    audit_root = tmp_path / "audit-root"
    audit_path = audit_root / "Output/audits/formal_experiment_readiness_audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text('{"readiness_status":"blocked"}\n', encoding="utf-8")
    config_path = tmp_path / "config_stack.yaml"
    config_path.write_text("config_stack: test\n", encoding="utf-8")
    code_commit = "a" * 40
    assets = [
        {
            "asset_uid": "processed-qc-summary",
            "logical_role": "simulation_audit",
            "artifact_type": "generated_json",
            "source_uid": "source-processed",
            "producer_uid": "build_simulation_readiness_audit",
            "version": "v1",
            "content_hash": _sha256(processed_path),
            "size": processed_path.stat().st_size,
            "canonical_uri": "repo://Input/processed/simulation_audit/simulation_qc_summary.json",
            "license_or_access_status": "approved_public_source_derived_internal_read_only",
            "required_by_testcases": ["test::processed"],
            "allowed_execution_role": "processed_test_input",
        },
        {
            "asset_uid": "formal-audit",
            "logical_role": "formal_readiness",
            "artifact_type": "generated_json",
            "source_uid": "source-audit",
            "producer_uid": "validate_formal_experiment_readiness",
            "version": "v1",
            "content_hash": _sha256(audit_path),
            "size": audit_path.stat().st_size,
            "canonical_uri": "repo://Output/audits/formal_experiment_readiness_audit.json",
            "license_or_access_status": "approved_internal_read_only",
            "required_by_testcases": ["test::audit"],
            "allowed_execution_role": "read_only_parent_audit",
        },
    ]
    environment_entries = [{"environment_uid": "test", "kind": "unit", "version": "1"}]
    bundle = {
        "bundle_uid": "round11g1-test-bundle",
        "schema_version": SCHEMA_VERSION,
        "code_commit": code_commit,
        "config_hash": _sha256(config_path),
        "processed_input_entries": assets,
        "external_asset_entries": [],
        "environment_entries": environment_entries,
        "producer_lineage": [{"producer_uid": "test", "status": "known"}],
        "license_or_access_decisions": [{"decision_uid": "test", "status": "approved"}],
        "content_hashes": {entry["asset_uid"]: entry["content_hash"] for entry in assets},
        "canonical_locations": {entry["asset_uid"]: entry["canonical_uri"] for entry in assets},
        "materialization_commands": ["test-only materialization"],
        "read_only_policy": {"mode": "strict"},
        "missing_entries": [],
        "validation_status": "passed",
        "environment_hash": canonical_sha256(environment_entries),
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    environment = {
        BUNDLE_ENV: str(bundle_path),
        BUNDLE_SHA256_ENV: _sha256(bundle_path),
        PROCESSED_ROOT_ENV: str(processed_root),
        AUDIT_ROOT_ENV: str(audit_root),
    }
    return environment, config_path, code_commit, str(audit_root)


def test_execution_input_bundle_schema_is_valid() -> None:
    schema = json.loads((REPO_ROOT / "Libs/configs/schema/round11g_execution_input_bundle.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_execution_audit_linkage_contract_covers_round11f_frozen_inputs() -> None:
    contract = _audit_linkage_contract(REPO_ROOT)
    expected = {
        Path("Output/audits/dt_round11d_full_suite_execution.json"),
        Path("Output/audits/dt_round11d_full_suite_results.xml"),
        Path("Output/audits/dt_round11d_full_suite_failure_triage.json"),
        Path("Output/audits/dt_round11_closure_decision.json"),
        Path("Output/audits/dt_round11e_targeted_gate.json"),
        Path("Output/audits/dt_round11e_repair_wave_record.json"),
        Path("Output/audits/dt_round11e_testcase_failure_ledger.tsv"),
        Path("Output/audits/dt_round11e_failure_dependency_graph.json"),
        Path("Output/audits/dt_round11e_full_suite_execution.json"),
    }
    assert expected.issubset(contract)
    assert len(contract) == 11
    assert all(entry["testcase_scope"] for entry in contract.values())


def test_execution_input_resolver_binds_config_hash_role_and_producer(tmp_path: Path) -> None:
    environment, config_path, code_commit, _ = _fixture(tmp_path)
    bundle = load_execution_input_bundle(environment, config_stack_path=config_path, expected_code_commit=code_commit)
    asset = resolve_execution_input(bundle, config_path, "processed-qc-summary", expected_role="processed_test_input")
    assert asset.producer_uid == "build_simulation_readiness_audit"
    assert asset.physical_path.read_text(encoding="utf-8") == '{"status":"available"}\n'
    with pytest.raises(ValueError, match="role mismatch"):
        resolve_execution_input(bundle, config_path, "processed-qc-summary", expected_role="read_only_parent_audit")


def test_execution_input_resolver_fails_closed_on_hash_access_and_duplicate_uid(tmp_path: Path) -> None:
    environment, config_path, code_commit, _ = _fixture(tmp_path)
    bundle_path = Path(environment[BUNDLE_ENV])
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["processed_input_entries"][0]["content_hash"] = "0" * 64
    payload["content_hashes"]["processed-qc-summary"] = "0" * 64
    bundle_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    environment[BUNDLE_SHA256_ENV] = _sha256(bundle_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_execution_input_bundle(environment, config_stack_path=config_path, expected_code_commit=code_commit)

    environment, config_path, code_commit, _ = _fixture(tmp_path / "access")
    bundle_path = Path(environment[BUNDLE_ENV])
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["processed_input_entries"][0]["license_or_access_status"] = "unresolved"
    bundle_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    environment[BUNDLE_SHA256_ENV] = _sha256(bundle_path)
    with pytest.raises(PermissionError, match="access is unresolved"):
        load_execution_input_bundle(environment, config_stack_path=config_path, expected_code_commit=code_commit)

    environment, config_path, code_commit, _ = _fixture(tmp_path / "duplicate")
    bundle_path = Path(environment[BUNDLE_ENV])
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["processed_input_entries"][1]["asset_uid"] = "processed-qc-summary"
    payload["content_hashes"] = {"processed-qc-summary": payload["processed_input_entries"][0]["content_hash"]}
    bundle_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    environment[BUNDLE_SHA256_ENV] = _sha256(bundle_path)
    with pytest.raises(ValueError, match="Duplicate logical asset UID"):
        load_execution_input_bundle(environment, config_stack_path=config_path, expected_code_commit=code_commit)


def test_bwrap_command_preserves_tracked_input_and_hides_unlisted_processed_files(tmp_path: Path) -> None:
    environment, config_path, code_commit, audit_root = _fixture(tmp_path)
    bundle = load_execution_input_bundle(environment, config_stack_path=config_path, expected_code_commit=code_commit)
    worktree = tmp_path / "worktree"
    tracked_structure = worktree / "Input/structures/predicted/model.pdb"
    tracked_structure.parent.mkdir(parents=True)
    tracked_structure.write_text("MODEL\n", encoding="utf-8")
    (worktree / "Output/audits").mkdir(parents=True)
    output_root = tmp_path / "execution-output"
    output_root.mkdir()
    (output_root / "repository-output").mkdir()
    (output_root / "runtime-output").mkdir()
    command = build_execution_bwrap_command(worktree, bundle, ["python", "-V"], output_root=output_root)
    joined = " ".join(command)
    assert str(worktree / "Input/processed") in joined
    assert str(tracked_structure) not in joined
    assert "Input/structures" not in joined
    assert "simulation_qc_summary.json" in joined
    assert "formal_experiment_readiness_audit.json" in joined
    assert audit_root not in command[-8:]
    assert "--setenv" in command
    assert "NRNSIFTER_AUDIT_ARTIFACT_ROOT" in command
    assert command.index("/run") < command.index("/run/neuronsifter-round11g1-inputs")
    assert command.count("--tmpfs") == 5
    assert str(Path(environment[PROCESSED_ROOT_ENV])) in command
    assert audit_root in command
    assert str(output_root) in command
    assert str(output_root / "repository-output") in command
    assert str(output_root / "runtime-output") in command
    physical_audit = Path(audit_root) / "Output/audits/formal_experiment_readiness_audit.json"
    assert command.count(str(physical_audit)) == 2
    assert OUTPUT_ROOT_ENV in command
    assert "/run/neuronsifter-round11g1-outputs" in command
    assert all(variable in command for variable in ("TMPDIR", "TMP", "TEMP"))
    root_bind = ["--ro-bind", str(worktree.resolve()), str(worktree.resolve())]
    assert any(command[index : index + 3] == root_bind for index in range(len(command) - 2))


def test_execution_namespace_smoke_enforces_manifested_visibility_boundary(tmp_path: Path) -> None:
    environment, config_path, code_commit, _ = _fixture(tmp_path)
    unlisted_source = Path(environment[PROCESSED_ROOT_ENV]) / "registry/source_asset_catalog.tsv"
    unlisted_source.parent.mkdir(parents=True)
    unlisted_source.write_text("source_uid\nnot-authorized\n", encoding="utf-8")
    bundle = load_execution_input_bundle(environment, config_stack_path=config_path, expected_code_commit=code_commit)
    worktree = tmp_path / "worktree"
    tracked_structure = worktree / "Input/structures/predicted/model.pdb"
    tracked_structure.parent.mkdir(parents=True)
    tracked_structure.write_text("MODEL\n", encoding="utf-8")
    (worktree / "Output/audits").mkdir(parents=True)

    result = run_execution_namespace_smoke(
        worktree,
        bundle,
        tracked_input_path="Input/structures/predicted/model.pdb",
        unlisted_processed_path="Input/processed/registry/source_asset_catalog.tsv",
    )

    assert result["status"] == "passed"
    assert result["returncode"] == 0
    assert result["checks"] == {
        "audit_source_hidden": True,
        "manifested_processed_visible": True,
        "materialized_audit_visible": True,
        "processed_source_hidden": True,
        "tracked_input_visible": True,
        "unlisted_processed_hidden": True,
    }
    assert not (worktree / "Input/processed").exists()

    output_root = tmp_path / "execution-output"
    output_root.mkdir()
    completed = run_with_execution_inputs(
        worktree,
        bundle,
        [
            "python",
            "-c",
            (
                "import os, pathlib, sys, tempfile; "
                "worktree=pathlib.Path(sys.argv[1]); "
                "blocked=worktree / 'Libs/forbidden-write.json'; "
                "exec(\"try:\\n blocked.write_text('forbidden')\\nexcept OSError:\\n pass\\nelse:\\n raise SystemExit(3)\"); "
                "(worktree / 'Output/audits/generated.json').write_text('external'); "
                "audit=worktree / 'Output/audits/formal_experiment_readiness_audit.json'; "
                "assert audit.read_text() == '{\"readiness_status\":\"blocked\"}\\n'; "
                "output=pathlib.Path(os.environ['NRNSIFTER_EXECUTION_OUTPUT_ROOT']); "
                "assert not worktree.is_relative_to(output) and not output.is_relative_to(worktree); "
                "temp=pathlib.Path(tempfile.mkdtemp()); "
                "assert temp.is_relative_to(output); "
                "(output / 'probe.txt').write_text('passed')"
            ),
            str(worktree),
        ],
        output_root=output_root,
        capture_output=True,
    )
    assert completed.returncode == 0
    assert (output_root / "runtime-output/probe.txt").read_text(encoding="utf-8") == "passed"
    assert (output_root / "repository-output/audits/generated.json").read_text(encoding="utf-8") == "external"
    assert (output_root / "repository-output/audits/formal_experiment_readiness_audit.json").read_text(
        encoding="utf-8"
    ) == '{"readiness_status":"blocked"}\n'
    assert not (worktree / "Libs/forbidden-write.json").exists()
    assert not (worktree / "Output/audits/generated.json").exists()
    assert not (worktree / "Input/processed").exists()
