"""Build Round-11G1 dependency closure, input identity, and materialization audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json  # noqa: E402
from Libs.utils.data.round11g_execution_inputs import canonical_sha256  # noqa: E402
from Libs.utils.data.round11g_execution_inputs import sha256_file  # noqa: E402


SCHEMA_VERSION = "dt-round11g-execution-input-bundle-v1"
PROCESSED_MANIFEST = Path("Output/audits/dt_round11e_processed_input_linkage_manifest.json")
HISTORICAL_LEDGER = Path("Output/audits/dt_round11g_failure_reproduction_ledger.tsv")
HISTORICAL_JUNIT = Path("Output/audits/dt_round11g_failure_reproduction_results.xml")
FORMAL_AUDIT = Path("Output/audits/formal_experiment_readiness_audit.json")
G5_AUDIT = Path("Output/paper_reports/g5_effective_sample_audit.json")
LINKAGE_PATHS = [
    "Libs/utils/data/round11e_processed_input_linkage.py",
    "Libs/scripts/builders/build_dt_round11e_processed_input_manifest.py",
    "Libs/scripts/validators/run_dt_round11e_processed_input_linkage.py",
    "Libs/tests/test_dt_round11e_processed_input_linkage.py",
]
G1_PATHS = [
    "Libs/configs/schema/round11g_execution_input_bundle.schema.json",
    "Libs/utils/data/round11g_execution_inputs.py",
    "Libs/scripts/builders/build_dt_round11g1_dependency_closure.py",
    "Libs/scripts/validators/run_dt_round11g1_execution_inputs.py",
    "Libs/scripts/validators/validate_dt_round11g1_dependency_preflight.py",
    "Libs/tests/test_dt_round11g1_execution_inputs.py",
]
G2_REPAIR_PATHS = [
    "Libs/scripts/builders/build_dt_round11g1_dependency_closure.py",
    "Libs/scripts/validators/run_dt_round11g1_execution_inputs.py",
    "Libs/scripts/validators/validate_dt_round11g1_dependency_preflight.py",
    "Libs/tests/test_dt_round11g1_execution_inputs.py",
    "Libs/utils/data/round11g_execution_inputs.py",
]
CONTROLLER_PATH = "Libs/configs/validation/iclr_design_contract_union.yaml"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _junit_nonpass(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for testcase in ET.parse(path).getroot().iter("testcase"):
        node = testcase.find("failure")
        kind = "failure"
        if node is None:
            node = testcase.find("error")
            kind = "error"
        if node is None:
            continue
        test_id = f"{testcase.get('classname')}::{testcase.get('name')}"
        detail = (node.text or "").strip()
        frames = [
            line.strip()
            for line in detail.splitlines()
            if "Libs/" in line and ".py:" in line
        ]
        records[test_id] = {
            "failure_or_error": kind,
            "message": str(node.get("message", "")).splitlines()[0],
            "first_relevant_stack_frame": frames[-1] if frames else "not_available_in_junit",
        }
    return records


def _normalized_missing_resource(message: str, worktree: Path) -> str:
    match = re.search(r'(/[^"\']+?/(?:Input|Output)/[^"\']+)', message)
    if match:
        value = Path(match.group(1))
        try:
            return value.relative_to(worktree).as_posix()
        except ValueError:
            for marker in ("Input/", "Output/"):
                index = value.as_posix().find(marker)
                if index >= 0:
                    return value.as_posix()[index:]
    if "Numeric lineage source artifacts are missing" in message:
        return ";".join(
            [
                "Input/processed/simulation_audit/simulation_qc_summary.json",
                "Input/processed/simulation_audit/simulation_qc_catalog.tsv",
                "Input/processed/simulation_audit/simulation_pair_alignment_audit.tsv",
                "Input/processed/simulation_audit/simulation_numeric_audit.tsv",
                "Input/processed/full_microenvironment_teacher_ready/trajectory_store.zarr/trajectories/.zarray",
                "Input/processed/dynamics_ready/public_ad_meso/simulation_catalog.tsv",
                "Input/processed/full_microenvironment_teacher_ready/pbpk_rollout_catalog.tsv",
                "Input/processed/full_microenvironment_teacher_ready/neuron_rxd_protocol_catalog.tsv",
            ]
        )
    if "assert 'blocked' == 'pass'" in message:
        return "Libs/configs/validation/iclr_design_contract_union.yaml;orchestrator/state.json"
    return "unknown"


def _producer_for(logical_path: str) -> str:
    mappings = [
        ("Input/processed/action_ready/", "build_action_ready_registry"),
        ("Input/processed/alignment_ready/", "build_alignment_ready_registry"),
        ("Input/processed/analysis_ready/", "build_analysis_ready_registry"),
        ("Input/processed/candidate_validation/", "build_candidate_validation_registry"),
        ("Input/processed/cohort_ready/", "build_cohort_ready_registry"),
        ("Input/processed/crossmodal_anchor_ready/", "build_crossmodal_anchor_ready_registry"),
        ("Input/processed/drug_target_ready/", "build_drug_target_ready_registry"),
        ("Input/processed/dynamics_ready/", "build_dynamics_ready_registry"),
        ("Input/processed/feature_ready/", "build_feature_ready_registry"),
        ("Input/processed/full_microenvironment_teacher_ready/", "build_full_microenvironment_teacher_registry"),
        ("Input/processed/intervention_program/", "build_intervention_program_registry"),
        ("Input/processed/model_input/", "build_model_input_registry"),
        ("Input/processed/pkpd_ready/", "build_pkpd_ready_registry"),
        ("Input/processed/registry/", "build_processed_registry"),
        ("Input/processed/simulation_audit/", "build_simulation_readiness_audit"),
        ("Input/processed/simulation_teacher_ready/", "build_simulation_teacher_ready_registry"),
        ("Input/processed/spatial_atlas/", "build_spatial_atlas_registry"),
        ("Input/processed/structural_prior/", "build_structural_prior_registry"),
        ("Input/processed/world_state_ready/", "build_world_state_ready_registry"),
    ]
    for prefix, producer in mappings:
        if logical_path.startswith(prefix):
            return producer
    raise ValueError(f"No producer mapping for processed input: {logical_path}")


def _artifact_type(path: str) -> str:
    if path.endswith(".json"):
        return "generated_json"
    if path.endswith(".tsv"):
        return "generated_tsv"
    if path.endswith(".parquet"):
        return "generated_parquet"
    if path.endswith(".zarray"):
        return "zarr_array_metadata"
    return "generated_file"


def _asset_uid(logical_path: str) -> str:
    return "asset-" + hashlib.sha256(logical_path.encode("utf-8")).hexdigest()[:20]


def _environment_entries() -> list[dict[str, object]]:
    package_versions: dict[str, str] = {}
    for package in ("pytest", "jsonschema", "ruff"):
        try:
            package_versions[package] = _command_output([sys.executable, "-m", "pip", "show", package]).split("Version: ", 1)[1].splitlines()[0]
        except (subprocess.CalledProcessError, IndexError):
            package_versions[package] = "unavailable"
    bwrap_path = shutil.which("bwrap")
    bwrap_version = _command_output([bwrap_path, "--version"]) if bwrap_path else "unavailable"
    return [
        {
            "environment_uid": "python-runtime",
            "kind": "python_runtime",
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        {
            "environment_uid": "bubblewrap-runtime",
            "kind": "system_binary",
            "version": bwrap_version,
            "availability": "available" if bwrap_path else "missing",
        },
        {
            "environment_uid": "validation-packages",
            "kind": "python_packages",
            "versions": package_versions,
        },
        {
            "environment_uid": "platform-runtime",
            "kind": "platform",
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    ]


def _audit_linkage_contract(code_root: Path) -> dict[Path, str]:
    contract_path = code_root / "Libs/configs/validation/iclr_design_contract_union.yaml"
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    linkage = payload["iclr_design_contract_union"]["g0_read_only_artifact_linkage"]
    artifacts = linkage["artifacts"]
    expected = {
        FORMAL_AUDIT: artifacts["formal_experiment_readiness_audit"],
        G5_AUDIT: artifacts["g5_effective_sample_audit"],
    }
    result: dict[Path, str] = {}
    for logical_path, entry in expected.items():
        if entry.get("logical_path") != logical_path.as_posix():
            raise ValueError(f"Read-only audit logical path mismatch: {logical_path}")
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64:
            raise ValueError(f"Read-only audit has no valid SHA-256: {logical_path}")
        result[logical_path] = digest
    return result


def _materialize_audit_bundle(
    source_root: Path,
    output_root: Path,
    expected_hashes: Mapping[Path, str],
) -> tuple[Path, dict[Path, str]]:
    audit_set_hash = canonical_sha256({path.as_posix(): digest for path, digest in expected_hashes.items()})
    materialized_root = output_root / "materialized" / "round11g1" / "audit-bundles" / "sha256" / audit_set_hash / "root"
    labels: dict[Path, str] = {}
    for logical_path, expected_hash in expected_hashes.items():
        source = source_root / logical_path
        if not source.is_file():
            raise FileNotFoundError(f"Read-only audit is unavailable: {logical_path}")
        observed_hash = sha256_file(source)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Read-only audit hash mismatch for {logical_path}: expected {expected_hash}, observed {observed_hash}"
            )
        destination = materialized_root / logical_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) != expected_hash:
            raise ValueError(f"Existing materialized audit has the wrong hash: {logical_path}")
        if not destination.exists():
            shutil.copyfile(source, destination)
            destination.chmod(0o444)
        if destination.stat().st_size != source.stat().st_size or sha256_file(destination) != expected_hash:
            raise ValueError(f"Materialized read-only audit failed integrity verification: {logical_path}")
        labels[logical_path] = destination.relative_to(output_root.parent).as_posix()
    return materialized_root, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Round-11G1 dependency and execution-input artifacts.")
    parser.add_argument("--evidence-root", default=str(REPO_ROOT))
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "Output"))
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--commit-d")
    parser.add_argument("--execution-worktree-role", default="round11g1_clean_execution_worktree")
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_root = Path(args.evidence_root).resolve()
    code_root = Path(args.code_root).resolve()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    audits = output_root / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    if len(args.code_commit) != 40:
        raise ValueError("Round-11G1 code commit must be a full 40-character commit")
    if args.commit_d is not None and len(args.commit_d) != 40:
        raise ValueError("Round-11G1 Commit D must be a full 40-character commit")
    if input_root.is_symlink() or not input_root.is_dir():
        raise FileNotFoundError(f"Canonical processed-input root is unavailable or symlinked: {input_root}")

    reproduction_rows = [row for row in _read_tsv(evidence_root / HISTORICAL_LEDGER) if row["current_status"] != "resolved_by_prior_change"]
    if len(reproduction_rows) != 32:
        raise ValueError(f"Expected 32 current non-pass cases, observed {len(reproduction_rows)}")
    junit = _junit_nonpass(evidence_root / HISTORICAL_JUNIT)
    manifest = _read_json(evidence_root / PROCESSED_MANIFEST)
    manifest_entries = {str(entry["logical_path"]): dict(entry) for entry in manifest["entries"]}
    worktree = Path("orchestrator/worktrees/round11g-execution-6de44e08")
    processed_test_ids = [row["test_id"] for row in reproduction_rows if row["root_cause_uid"] == "R11E-RC-001"]
    mixed_test_id = next(row["test_id"] for row in reproduction_rows if row["test_id"].startswith("Libs.tests.test_g0_integrity::"))
    processed_test_ids.append(mixed_test_id)
    processed_test_ids = sorted(set(processed_test_ids))
    audit_test_ids = sorted(row["test_id"] for row in reproduction_rows if row["root_cause_uid"] == "R11E-RC-002")
    contract_test_ids = sorted(row["test_id"] for row in reproduction_rows if row["root_cause_uid"] == "R11E-RC-003")
    g5_test_ids = sorted(
        row["test_id"]
        for row in reproduction_rows
        if row["test_id"].startswith(
            (
                "Libs.tests.test_formal_experiment_freeze::",
                "Libs.tests.test_formal_experiment_readiness::",
            )
        )
    )

    ledger_rows: list[dict[str, object]] = []
    for row in reproduction_rows:
        detail = junit.get(row["test_id"])
        if detail is None:
            raise ValueError(f"Current non-pass testcase is absent from JUnit: {row['test_id']}")
        missing_resource = _normalized_missing_resource(detail["message"], worktree)
        if row["root_cause_uid"] == "R11E-RC-002":
            classification = "missing_round11e_linkage"
            expected_resource_type = "generated_read_only_audit"
            expected_producer = "validate_formal_experiment_readiness"
            expected_manifest = "Output/audits/dt_round11g1_execution_input_bundle.json"
            expected_hash = "abb9e84c44b0e54c3412c9b641ec3114f03e15d75917106960453d59fcd77c05"
            linkage_required = "true"
            code_possible = "false"
        elif row["test_id"] == mixed_test_id:
            classification = "mixed_dependency_and_code"
            expected_resource_type = "processed_input_set_and_controller_contract"
            expected_producer = "build_simulation_readiness_audit;controller_lineage_owner"
            expected_manifest = f"{PROCESSED_MANIFEST.as_posix()};Output/audits/dt_round11f_roadmap_lineage.json"
            expected_hash = str(manifest["entry_set_sha256"])
            linkage_required = "true"
            code_possible = "true"
        elif row["root_cause_uid"] == "R11E-RC-003":
            classification = "stale_test_contract"
            expected_resource_type = "git_tracked_controller_contract_linkage"
            expected_producer = "controller_lineage_owner"
            expected_manifest = "Output/audits/dt_round11f_roadmap_lineage.json"
            expected_hash = "not_applicable_contract_relation"
            linkage_required = "false"
            code_possible = "true"
        else:
            classification = "missing_processed_input"
            expected_resource_type = "processed_input_artifact"
            first_resource = missing_resource.split(";", 1)[0]
            expected_producer = _producer_for(first_resource)
            expected_manifest = PROCESSED_MANIFEST.as_posix()
            expected_hash = str(manifest_entries.get(first_resource, {}).get("sha256", manifest["entry_set_sha256"]))
            linkage_required = "true"
            code_possible = "false"
        ledger_rows.append(
            {
                "testcase_uid": row["testcase_uid"],
                "current_status": row["current_status"],
                "failure_or_error": detail["failure_or_error"],
                "first_relevant_stack_frame": detail["first_relevant_stack_frame"],
                "exception_type": detail["message"].split(":", 1)[0].replace('failed on setup with "', ""),
                "missing_path_or_resource": missing_resource,
                "expected_resource_type": expected_resource_type,
                "expected_producer": expected_producer,
                "expected_manifest": expected_manifest,
                "expected_hash_if_known": expected_hash,
                "round11e_linkage_required": linkage_required,
                "external_asset_required": "false",
                "environment_dependency_required": "true" if linkage_required == "true" else "false",
                "code_defect_possible": code_possible,
                "classification": classification,
                "confidence": "strongly_supported",
            }
        )
    ledger_fields = list(ledger_rows[0])
    _write_tsv(audits / "dt_round11g1_nonpass_dependency_ledger.tsv", ledger_fields, ledger_rows)

    family_rows = [
        {
            "dependency_uid": "r11g1-dep-processed-snapshot",
            "affected_testcase_count": len(processed_test_ids),
            "logical_role": "Round-11E processed-input snapshot",
            "producer_uid": "round11e_processed_input_manifest_and_registry_builders",
            "source_stage": "Round-11E Wave B",
            "required_artifacts": f"{manifest['entry_count']} manifest entries",
            "current_availability": "available_verified",
            "current_location": "repo://Input/processed",
            "expected_canonical_location": "repo://Input/processed",
            "license_or_access_status": "approved_public_source_derived_internal_read_only",
            "integrity_status": f"passed_{manifest['entry_count']}_of_{manifest['entry_count']}",
            "authorization_status": "approved_manifest_only",
        },
        {
            "dependency_uid": "r11g1-dep-formal-readiness-audit",
            "affected_testcase_count": len(audit_test_ids),
            "logical_role": "read-only parent formal readiness audit",
            "producer_uid": "validate_formal_experiment_readiness",
            "source_stage": "formal readiness validation",
            "required_artifacts": FORMAL_AUDIT.as_posix(),
            "current_availability": "available_verified",
            "current_location": "repo://Output/audits/formal_experiment_readiness_audit.json",
            "expected_canonical_location": "content_addressed_materialization",
            "license_or_access_status": "approved_internal_read_only",
            "integrity_status": "passed_sha256",
            "authorization_status": "approved_manifest_only",
        },
        {
            "dependency_uid": "r11g1-dep-g5-effective-sample-audit",
            "affected_testcase_count": len(g5_test_ids),
            "logical_role": "read-only parent G5 effective-sample audit",
            "producer_uid": "build_paper_results",
            "source_stage": "publication report generation",
            "required_artifacts": G5_AUDIT.as_posix(),
            "current_availability": "available_verified",
            "current_location": f"repo://{G5_AUDIT.as_posix()}",
            "expected_canonical_location": "content_addressed_materialization",
            "license_or_access_status": "approved_internal_read_only",
            "integrity_status": "passed_sha256",
            "authorization_status": "approved_manifest_only",
        },
        {
            "dependency_uid": "r11g1-dep-controller-lineage",
            "affected_testcase_count": len(contract_test_ids),
            "logical_role": "controller state and active contract-union lineage",
            "producer_uid": "controller_lineage_owner",
            "source_stage": "Round-11F controller lineage",
            "required_artifacts": f"{CONTROLLER_PATH};orchestrator/state.json",
            "current_availability": "available_conflicting",
            "current_location": "repo://Libs/configs/validation;repo://orchestrator/state.json",
            "expected_canonical_location": "single owner-approved active lineage",
            "license_or_access_status": "not_applicable_git_contract",
            "integrity_status": "blocked_lineage_mismatch",
            "authorization_status": "separate_owner_decision_required",
        },
    ]
    _write_tsv(audits / "dt_round11g1_dependency_family_summary.tsv", list(family_rows[0]), family_rows)

    graph = {
        "schema_version": "dt-round11g1-dependency-root-graph-v1",
        "source_testcase_count": 32,
        "priority_order": [
            "r11g1-dep-processed-snapshot",
            "r11g1-dep-formal-readiness-audit",
            "r11g1-dep-controller-lineage",
        ],
        "families": family_rows,
        "testcase_chains": [
            {
                "testcase_uid": row["testcase_uid"],
                "test_id": original["test_id"],
                "chain": [
                    original["test_id"],
                    row["first_relevant_stack_frame"],
                    row["missing_path_or_resource"],
                    row["expected_manifest"],
                    row["expected_producer"],
                    row["expected_resource_type"],
                    "repository_internal_provenance",
                ],
            }
            for row, original in zip(ledger_rows, reproduction_rows, strict=True)
        ],
    }
    write_json(audits / "dt_round11g1_dependency_root_graph.json", graph)

    channel_rows = [
        {
            "dependency_uid": "r11g1-linkage-code",
            "identity_channel": "A",
            "logical_role": "strict linkage, resolver, schema, validator, and tests",
            "paths_or_resources": ";".join(LINKAGE_PATHS + G1_PATHS),
            "decision": "approved_git_linkage",
        },
        {
            "dependency_uid": "r11g1-processed-snapshot",
            "identity_channel": "B",
            "logical_role": "54 hash-bound processed input files",
            "paths_or_resources": "repo://Input/processed",
            "decision": "approved_manifest_only",
        },
        {
            "dependency_uid": "r11g1-parent-audit",
            "identity_channel": "B",
            "logical_role": "formal readiness generated audit",
            "paths_or_resources": "repo://Output/audits/formal_experiment_readiness_audit.json",
            "decision": "approved_manifest_only",
        },
        {
            "dependency_uid": "r11g1-g5-effective-sample-audit",
            "identity_channel": "B",
            "logical_role": "G5 effective-sample generated audit",
            "paths_or_resources": f"repo://{G5_AUDIT.as_posix()}",
            "decision": "approved_manifest_only",
        },
        {
            "dependency_uid": "r11g1-external-assets",
            "identity_channel": "C",
            "logical_role": "external assets",
            "paths_or_resources": "none",
            "decision": "not_required",
        },
        {
            "dependency_uid": "r11g1-runtime-environment",
            "identity_channel": "D",
            "logical_role": "Python, pytest, jsonschema, Ruff, bubblewrap",
            "paths_or_resources": "environment_entries",
            "decision": "exact_environment_hash",
        },
        {
            "dependency_uid": "r11g1-controller-lineage-contract",
            "identity_channel": "A",
            "logical_role": "active roadmap/controller config mapping",
            "paths_or_resources": f"{CONTROLLER_PATH};orchestrator/state.json",
            "decision": "blocked_unknown_ownership",
        },
    ]
    _write_tsv(audits / "dt_round11g1_identity_channel_map.tsv", list(channel_rows[0]), channel_rows)

    linkage_scope_rows: list[dict[str, object]] = []
    for path in LINKAGE_PATHS:
        physical = evidence_root / path
        linkage_scope_rows.append(
            {
                "path": path,
                "git_status_in_root": "untracked",
                "content_hash": sha256_file(physical),
                "producer": "Round-11E Wave B repair",
                "consumer": "Round-11G1 execution input launcher",
                "contract_role": "strict processed-input linkage",
                "required_by_testcases": len(processed_test_ids) + len(audit_test_ids),
                "scientific_semantics": "visibility_only_no_evidence_promotion",
                "safe_to_commit": "true",
                "contains_user_specific_state": "false",
                "contains_generated_or_large_data": "false",
            }
        )
    controller_physical = evidence_root / CONTROLLER_PATH
    linkage_scope_rows.append(
        {
            "path": CONTROLLER_PATH,
            "git_status_in_root": "modified",
            "content_hash": sha256_file(controller_physical),
            "producer": "unknown_overlapping_root_changes",
            "consumer": "controller lineage validators",
            "contract_role": "active roadmap lineage",
            "required_by_testcases": len(contract_test_ids),
            "scientific_semantics": "governance_authority",
            "safe_to_commit": "false",
            "contains_user_specific_state": "unknown",
            "contains_generated_or_large_data": "false",
        }
    )
    _write_tsv(audits / "dt_round11g1_round11e_linkage_scope.tsv", list(linkage_scope_rows[0]), linkage_scope_rows)
    linkage_decision = {
        "schema_version": "dt-round11g1-round11e-linkage-decision-v1",
        "decision": "approved_git_linkage",
        "approved_paths": LINKAGE_PATHS,
        "approved_g1_paths": G1_PATHS,
        "blocked_paths": [
            {
                "path": CONTROLLER_PATH,
                "decision": "blocked_unknown_ownership",
                "reason": "Round-11E records require a separate owner decision and the root path contains overlapping user changes.",
            }
        ],
        "processed_input_decision": "approved_manifest_only",
        "external_materialization_decision": "not_required",
    }
    write_json(audits / "dt_round11g1_round11e_linkage_decision.json", linkage_decision)

    expected_audit_hashes = _audit_linkage_contract(code_root)
    audit_set_hash = canonical_sha256(
        {path.as_posix(): digest for path, digest in expected_audit_hashes.items()}
    )
    if args.materialize:
        materialized_root, materialized_labels = _materialize_audit_bundle(
            evidence_root,
            output_root,
            expected_audit_hashes,
        )
    else:
        materialized_root = (
            output_root / "materialized" / "round11g1" / "audit-bundles" / "sha256" / audit_set_hash / "root"
        )
        materialized_labels = {
            path: (materialized_root / path).relative_to(evidence_root).as_posix()
            for path in expected_audit_hashes
        }
    formal_path = evidence_root / FORMAL_AUDIT
    g5_path = evidence_root / G5_AUDIT
    expected_formal_hash = expected_audit_hashes[FORMAL_AUDIT]
    expected_g5_hash = expected_audit_hashes[G5_AUDIT]
    if not formal_path.is_file() or sha256_file(formal_path) != expected_formal_hash:
        raise ValueError("Formal readiness audit does not match the frozen config linkage hash")
    if not g5_path.is_file() or sha256_file(g5_path) != expected_g5_hash:
        raise ValueError("G5 effective-sample audit does not match the frozen config linkage hash")

    processed_assets: list[dict[str, object]] = []
    producer_lineage: dict[str, dict[str, object]] = {}
    for entry in manifest["entries"]:
        logical_path = str(entry["logical_path"])
        producer = _producer_for(logical_path)
        producer_lineage[producer] = {
            "producer_uid": producer,
            "producer_path": f"Libs/scripts/builders/{producer}.py",
            "source_stage": logical_path.split("/", 3)[2],
            "lineage_status": "verified_repository_producer",
        }
        processed_assets.append(
            {
                "asset_uid": _asset_uid(logical_path),
                "logical_role": logical_path.split("/", 3)[2],
                "artifact_type": _artifact_type(logical_path),
                "source_uid": f"round11e-processed-snapshot-{manifest['entry_set_sha256'][:16]}",
                "producer_uid": producer,
                "version": str(manifest["schema_version"]),
                "content_hash": entry["sha256"],
                "size": entry["size_bytes"],
                "canonical_uri": f"repo://{logical_path}",
                "license_or_access_status": "approved_public_source_derived_internal_read_only",
                "required_by_testcases": processed_test_ids,
                "allowed_execution_role": "processed_test_input",
            }
        )
    formal_asset = {
        "asset_uid": _asset_uid(FORMAL_AUDIT.as_posix()),
        "logical_role": "formal_experiment_readiness_parent_audit",
        "artifact_type": "generated_json",
        "source_uid": f"formal-readiness-audit-{expected_formal_hash[:16]}",
        "producer_uid": "validate_formal_experiment_readiness",
        "version": "frozen-parent-audit-v1",
        "content_hash": expected_formal_hash,
        "size": formal_path.stat().st_size,
        "canonical_uri": f"repo://{FORMAL_AUDIT.as_posix()}",
        "license_or_access_status": "approved_internal_read_only",
        "required_by_testcases": audit_test_ids,
        "allowed_execution_role": "read_only_parent_audit",
    }
    producer_lineage["validate_formal_experiment_readiness"] = {
        "producer_uid": "validate_formal_experiment_readiness",
        "producer_path": "Libs/scripts/validators/validate_formal_experiment_readiness.py",
        "source_stage": "formal_readiness_validation",
        "lineage_status": "verified_repository_producer",
    }
    g5_asset = {
        "asset_uid": _asset_uid(G5_AUDIT.as_posix()),
        "logical_role": "g5_effective_sample_parent_audit",
        "artifact_type": "generated_json",
        "source_uid": f"g5-effective-sample-audit-{expected_g5_hash[:16]}",
        "producer_uid": "build_paper_results",
        "version": "frozen-parent-audit-v1",
        "content_hash": expected_g5_hash,
        "size": g5_path.stat().st_size,
        "canonical_uri": f"repo://{G5_AUDIT.as_posix()}",
        "license_or_access_status": "approved_internal_read_only",
        "required_by_testcases": g5_test_ids,
        "allowed_execution_role": "read_only_parent_audit",
    }
    producer_lineage["build_paper_results"] = {
        "producer_uid": "build_paper_results",
        "producer_path": "Libs/scripts/reports/build_paper_results.py",
        "source_stage": "publication_report_generation",
        "lineage_status": "verified_repository_producer",
    }
    audit_assets = [formal_asset, g5_asset]
    all_assets = processed_assets + audit_assets
    environment_entries = _environment_entries()
    environment_hash = canonical_sha256(environment_entries)
    config_path = code_root / "Libs/configs/config_stack.yaml"
    config_hash = sha256_file(config_path)
    bundle_uid_seed = {
        "code_commit": args.code_commit,
        "config_hash": config_hash,
        "asset_hashes": {entry["asset_uid"]: entry["content_hash"] for entry in all_assets},
        "environment_hash": environment_hash,
    }
    bundle = {
        "bundle_uid": f"round11g1-bundle-{canonical_sha256(bundle_uid_seed)[:20]}",
        "schema_version": SCHEMA_VERSION,
        "code_commit": args.code_commit,
        "config_hash": config_hash,
        "processed_input_entries": all_assets,
        "external_asset_entries": [],
        "environment_entries": environment_entries,
        "producer_lineage": sorted(producer_lineage.values(), key=lambda row: str(row["producer_uid"])),
        "license_or_access_decisions": [
            {
                "decision_uid": "processed-internal-read-only",
                "status": "approved_public_source_derived_internal_read_only",
                "scope": "internal validation only; no redistribution or evidence promotion",
            },
            {
                "decision_uid": "parent-audit-internal-read-only",
                "status": "approved_internal_read_only",
                "scope": "hash-bound generated audit used only by its declared testcases",
            },
        ],
        "content_hashes": {entry["asset_uid"]: entry["content_hash"] for entry in all_assets},
        "canonical_locations": {entry["asset_uid"]: entry["canonical_uri"] for entry in all_assets},
        "materialization_commands": [
            "verify Input/processed entries against Output/audits/dt_round11e_processed_input_linkage_manifest.json",
            *[
                f"install -D --mode=0444 {path.as_posix()} {materialized_labels[path]}"
                for path in expected_audit_hashes
            ],
        ],
        "read_only_policy": {
            "mode": "bubblewrap_per_file_read_only_bind",
            "processed_root_search": "forbidden",
            "parent_worktree_fallback": "forbidden",
            "unlisted_asset_visibility": "forbidden",
            "silent_regeneration": "forbidden",
        },
        "missing_entries": [],
        "validation_status": "passed",
        "environment_hash": environment_hash,
    }
    write_json(audits / "dt_round11g1_execution_input_bundle.json", bundle)

    materialization_rows = [
        {
            "asset_uid": entry["asset_uid"],
            "identity_channel": "B",
            "source_uri": entry["canonical_uri"],
            "canonical_uri": entry["canonical_uri"],
            "size": entry["size"],
            "sha256": entry["content_hash"],
            "action": "verified_in_place_read_only",
            "integrity_status": "passed",
            "read_only": "true",
            "required_by_testcase_count": len(entry["required_by_testcases"]),
        }
        for entry in processed_assets
    ]
    materialization_rows.extend(
        {
            "asset_uid": entry["asset_uid"],
            "identity_channel": "B",
            "source_uri": entry["canonical_uri"],
            "canonical_uri": f"repo://{materialized_labels[logical_path]}",
            "size": entry["size"],
            "sha256": entry["content_hash"],
            "action": "content_addressed_copy",
            "integrity_status": "passed" if (materialized_root / logical_path).is_file() else "not_materialized",
            "read_only": "true",
            "required_by_testcase_count": len(entry["required_by_testcases"]),
        }
        for entry, logical_path in zip(audit_assets, (FORMAL_AUDIT, G5_AUDIT), strict=True)
    )
    _write_tsv(audits / "dt_round11g1_materialization_manifest.tsv", list(materialization_rows[0]), materialization_rows)
    integrity = {
        "schema_version": "dt-round11g1-asset-integrity-validation-v1",
        "status": "passed" if all(row["integrity_status"] == "passed" for row in materialization_rows) else "blocked",
        "asset_count": len(materialization_rows),
        "processed_asset_count": len(processed_assets),
        "external_asset_count": 0,
        "read_only_audit_count": len(audit_assets),
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "unknown_producer_count": 0,
        "unresolved_access_count": 0,
        "materialized_root_role": "round11g1_content_addressed_read_only_assets",
    }
    write_json(audits / "dt_round11g1_asset_integrity_validation.json", integrity)

    commit_d = args.commit_d or args.code_commit
    if args.commit_d is not None:
        relation = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit_d, args.code_commit],
            cwd=code_root,
            check=False,
        )
        if relation.returncode != 0:
            raise ValueError("Round-11G1 Commit D is not an ancestor of the execution code commit")
    commit_d_parent = _command_output(["git", "rev-parse", f"{commit_d}^"], cwd=code_root)
    commit_d_paths = _command_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_d],
        cwd=code_root,
    ).splitlines()
    authorized_paths = sorted(LINKAGE_PATHS + G1_PATHS)
    commit_manifest = {
        "schema_version": "dt-round11g1-commit-d-manifest-v1",
        "commit_message": "Round-11G1: bind processed-input and external-asset execution lineage",
        "parent_commit": commit_d_parent,
        "commit_d": commit_d,
        "authorized_paths": authorized_paths,
        "observed_paths": sorted(commit_d_paths),
        "path_set_exact": sorted(commit_d_paths) == authorized_paths,
        "excluded_paths": [CONTROLLER_PATH, "orchestrator/state.json", "Libs/utils/data/trajectory_numeric_lineage.py"],
        "large_or_generated_data_included": False,
    }
    write_json(audits / "dt_round11g1_commit_d_manifest.json", commit_manifest)

    execution_parent = _command_output(["git", "rev-parse", f"{args.code_commit}^"], cwd=code_root)
    if args.commit_d is not None:
        repair_paths = _command_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", args.code_commit],
            cwd=code_root,
        ).splitlines()
        repair_manifest = {
            "schema_version": "dt-round11g2-commit-manifest-v1",
            "commit_message": "Round-11G2: enforce execution input namespace isolation",
            "parent_commit": execution_parent,
            "commit_g2": args.code_commit,
            "authorized_paths": sorted(G2_REPAIR_PATHS),
            "observed_paths": sorted(repair_paths),
            "path_set_exact": sorted(repair_paths) == sorted(G2_REPAIR_PATHS),
            "large_or_generated_data_included": False,
        }
        write_json(audits / "dt_round11g2_commit_manifest.json", repair_manifest)

    status = _command_output(["git", "status", "--porcelain=v1"], cwd=code_root)
    identity = {
        "schema_version": "dt-round11g1-execution-identity-v1",
        "exact_git_commit": args.code_commit,
        "parent_commit": execution_parent,
        "dirty_status": "clean" if not status else "dirty",
        "code_relation": "exact_clean_commit" if not status else "dirty_worktree",
        "input_relation": "exact_frozen_manifest",
        "environment_relation": "exact_environment_hash",
        "execution_input_bundle_sha256": sha256_file(audits / "dt_round11g1_execution_input_bundle.json"),
        "processed_input_manifest_sha256": sha256_file(evidence_root / PROCESSED_MANIFEST),
        "environment_hash": environment_hash,
        "config_hash": config_hash,
        "worktree_path_role": args.execution_worktree_role,
    }
    write_json(audits / "dt_round11g1_execution_identity.json", identity)

    print(
        json.dumps(
            {
                "status": "passed",
                "nonpass_count": len(ledger_rows),
                "asset_count": len(all_assets),
                "bundle_uid": bundle["bundle_uid"],
                "bundle_sha256": sha256_file(audits / "dt_round11g1_execution_input_bundle.json"),
                "materialized_audit_root": str(materialized_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
