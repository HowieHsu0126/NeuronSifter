#!/usr/bin/env python3
"""Freeze and validate the Round-20 shared fact-source contracts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_RELATIVE = Path("Output/audits/dt_round20_full_landing_program_contract.json")
AUDIT_RELATIVE = Path("Output/audits/dt_round20_shared_schema_freeze.json")
BASE_COMMIT = "cd964535839f76369a3b34ff60902053c082b0ba"
EXPECTED_BRANCH = "round20-full-landing-program-20260804"
EXPECTED_TEST_COUNT = 607
GATE_UID = "round20_shared_schema_gate"

SCHEMA_REGISTRY = {
    "cell_taxonomy_crosswalk": (
        "neuronsifter-round20-cell-taxonomy-crosswalk-v1",
        "Libs/configs/schema/round20_cell_taxonomy_crosswalk.schema.json",
    ),
    "biological_unit": (
        "neuronsifter-round20-biological-unit-v1",
        "Libs/configs/schema/round20_biological_unit.schema.json",
    ),
    "allen_rights_release_repair_contract": (
        "neuronsifter-round20-allen-rights-release-repair-contract-v1",
        "Libs/configs/schema/round20_allen_rights_release_repair_contract.schema.json",
    ),
}


def _block(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _resolve_relative(repository: Path, value: object, label: str, blockers: list[dict[str, object]]) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        blockers.append(_block("path_not_relative", "Shared contract paths must be non-empty relative paths.", label=label, value=value))
        return None
    return repository / value


def _contract_paths(repository: Path, contract: Mapping[str, Any], blockers: list[dict[str, object]]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    sources = contract.get("shared_fact_sources", {}).get("sources", {})
    if not isinstance(sources, Mapping):
        blockers.append(_block("shared_sources_invalid", "The shared fact-source map is missing or is not an object."))
        return paths
    for source_name, source_spec in sources.items():
        if not isinstance(source_spec, Mapping):
            blockers.append(_block("shared_source_invalid", "A shared fact-source entry is not an object.", source=source_name))
            continue
        for key in ("path", "schema", "implementation", "config", "receipt_helper", "contract", "baseline_artifact", "identity_baseline_artifact", "successful_execution_junit", "reconciliation_artifact", "execution_audit", "inventory_audit", "request_schema", "result_schema"):
            value = source_spec.get(key)
            if isinstance(value, str):
                path = _resolve_relative(repository, value, f"{source_name}.{key}", blockers)
                if path is not None:
                    paths[f"{source_name}.{key}"] = path
        schemas = source_spec.get("schemas")
        if isinstance(schemas, list):
            for index, value in enumerate(schemas):
                path = _resolve_relative(repository, value, f"{source_name}.schemas[{index}]", blockers)
                if path is not None:
                    paths[f"{source_name}.schemas[{index}]"] = path
    stage0_audit = contract.get("inherited_stage0", {}).get("stage0_audit")
    path = _resolve_relative(repository, stage0_audit, "inherited_stage0.stage0_audit", blockers)
    if path is not None:
        paths["inherited_stage0.stage0_audit"] = path
    return paths


def _junit_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    values = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in values:
            values[key] += int(suite.attrib.get(key, suite.attrib.get("skips" if key == "skipped" else key, "0")))
    if values["tests"] == 0:
        values["tests"] = len(root.findall(".//testcase"))
    return values


def _cell_fixture() -> dict[str, Any]:
    return {
        "schema_version": "neuronsifter-round20-cell-taxonomy-crosswalk-v1",
        "crosswalk_uid": "cw:allen:abc:0001",
        "source_taxonomy": {"namespace": "Allen_Cell_Types", "release_uid": "abc-release-1", "term_id": "allen:inhibitory", "term_label": "inhibitory neuron", "species": "mouse"},
        "target_ontology": {"namespace": "Cell_Ontology", "term_id": "CL:0000101", "term_label": "inhibitory interneuron"},
        "mapping_relation": "related_match",
        "cardinality": "many_to_many",
        "mapping_confidence": "low",
        "mapping_score": 0.62,
        "mapping_method": "curated_semantic",
        "mapping_group_uid": "cw-group:allen:inhibitory",
        "ambiguity": {"explicit": True, "candidate_target_term_ids": ["CL:0000101", "CL:0000099"], "reason": "Allen taxonomy spans two Cell Ontology descendants.", "review_status": "pending"},
        "provenance_refs": ["sha256:example-crosswalk-source"],
        "status": "active",
    }


def _biological_fixture(identity_status: str = "exact") -> dict[str, Any]:
    blocked = identity_status == "ambiguous_blocked"
    return {
        "schema_version": "neuronsifter-round20-biological-unit-v1",
        "biological_unit_uid": "bu:allen:sample:0001",
        "unit_type": "sample",
        "source": {"source_family_uid": "ABC_mouse", "release_uid": "abc-release-1", "source_record_uid": "sample-0001", "access_class": "public_open"},
        "native_identity": {"namespace": "Allen", "native_id": "sample-0001", "id_version": "release-1"},
        "lineage": [{"relation": "derived_from", "unit_uid": "bu:allen:donor:0001", "unit_type": "donor"}],
        "identity_axes": {"donor_uid": "bu:allen:donor:0001", "specimen_uid": None, "tissue_block_uid": None, "sample_uid": "sample-0001", "cell_uid": None, "session_uid": None, "sweep_uid": None, "morphology_reconstruction_uid": None, "projection_experiment_uid": None, "atlas_structure_uid": None, "voxel_object_uid": None},
        "identity_status": identity_status,
        "independence": {"independent_unit_uid": "bu:allen:donor:0001", "split_identity_uid": "split:donor:0001", "effective_sample_size_contribution": 1.0},
        "context": {"organism": "mouse", "disease_context": "ad_control", "tissue": "brain", "brain_region": "CCF:1", "modality": "transcriptomics"},
        "mapping": {"region_ref": "CCF:1", "cell_type_ref": "CL:0000101", "coordinate_frame": "CCFv3", "ambiguity": {"explicit": blocked, "candidate_refs": ["CCF:1", "CCF:2"] if blocked else [], "reason": "unresolved region assignment" if blocked else "none", "review_status": "blocked" if blocked else "not_needed"}},
        "qc": {"grade": "B", "missingness_fraction": 0.0, "reject_reasons": []},
        "provenance": {"provenance_ref": "sha256:example-sample-source", "content_hash": "a" * 64, "parser_version": "round20-fixture-parser-v1"},
        "train_eval_role": "excluded_ambiguous" if blocked else "public_state_observation",
        "status": "blocked" if blocked else "active",
        "flattening_prohibited": True,
    }


def _validate_schema_semantics(schemas: Mapping[str, dict[str, Any]], blockers: list[dict[str, object]]) -> None:
    cell_validator = Draft202012Validator(schemas["cell_taxonomy_crosswalk"])
    cell = _cell_fixture()
    if list(cell_validator.iter_errors(cell)):
        blockers.append(_block("cell_fixture_invalid", "The many-to-many low-confidence cell crosswalk fixture does not validate."))
    missing_group = copy.deepcopy(cell)
    missing_group["mapping_group_uid"] = None
    if not list(cell_validator.iter_errors(missing_group)):
        blockers.append(_block("cell_many_to_many_flattened", "The cell crosswalk schema permits a many-to-many edge without a mapping group."))
    hidden_ambiguity = copy.deepcopy(cell)
    hidden_ambiguity["ambiguity"]["explicit"] = False
    if not list(cell_validator.iter_errors(hidden_ambiguity)):
        blockers.append(_block("cell_ambiguity_hidden", "The cell crosswalk schema permits low-confidence ambiguity to be hidden."))

    unit_validator = Draft202012Validator(schemas["biological_unit"])
    exact = _biological_fixture()
    if list(unit_validator.iter_errors(exact)):
        blockers.append(_block("biological_unit_fixture_invalid", "The exact biological-unit fixture does not validate."))
    ambiguous = _biological_fixture("ambiguous_blocked")
    if list(unit_validator.iter_errors(ambiguous)):
        blockers.append(_block("biological_unit_ambiguity_fixture_invalid", "The explicit blocked biological-unit fixture does not validate."))
    flattened = copy.deepcopy(ambiguous)
    flattened["mapping"]["ambiguity"]["explicit"] = False
    flattened["status"] = "active"
    flattened["train_eval_role"] = "public_state_observation"
    if not list(unit_validator.iter_errors(flattened)):
        blockers.append(_block("biological_unit_flattened", "The biological-unit schema permits an ambiguous identity to enter active evidence."))


def validate(*, repository: Path, contract_path: Path, require_clean: bool) -> dict[str, Any]:
    blockers: list[dict[str, object]] = []
    contract = _load_json(contract_path)
    candidate_commit = _git(repository, "rev-parse", "HEAD")
    candidate_branch = _git(repository, "rev-parse", "--abbrev-ref", "HEAD")
    candidate_tree = _git(repository, "rev-parse", "HEAD^{tree}")

    if contract.get("contract_id") != "round20-full-landing-program" or contract.get("contract_version") != "round20-v1":
        blockers.append(_block("contract_identity", "Round-20 umbrella contract identity is not frozen."))
    if contract.get("base_commit") != BASE_COMMIT or subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=repository, check=False).returncode != 0:
        blockers.append(_block("base_lineage", "Candidate does not descend from the accepted Round-19I closure commit."))
    if contract.get("worktree_branch") != candidate_branch:
        blockers.append(_block("branch_identity", "Candidate branch does not match the frozen Round-20 worktree branch.", expected=contract.get("worktree_branch"), actual=candidate_branch))
    if contract.get("status") != "frozen_for_planning" or contract.get("execution_authorization") != "blocked_until_shared_schema_gate":
        blockers.append(_block("contract_mutation", "The umbrella contract was not frozen in its pre-gate state."))
    if contract.get("scope", {}).get("public_only") is not True or contract.get("scope", {}).get("remote_push_authorized") is not False:
        blockers.append(_block("scope_policy", "Public-only or no-remote-push scope is not frozen."))
    if contract.get("resource_contract", {}).get("d_state_stop_threshold") != 15 or contract.get("resource_contract", {}).get("io_wait_stop_threshold_percent") != 25:
        blockers.append(_block("resource_threshold_drift", "The current D-state or I/O-wait stop thresholds drifted from the accepted contract."))

    paths = _contract_paths(repository, contract, blockers)
    for label, path in paths.items():
        if not path.is_file():
            blockers.append(_block("shared_source_missing", "A frozen shared fact source is missing.", label=label, path=str(path.relative_to(repository))))

    registry_path = repository / "Libs/configs/schema/schemas.yaml"
    schemas_config = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry = schemas_config.get("schemas", {}).get("round20_shared_schema_registry", {})
    if set(registry) != set(SCHEMA_REGISTRY):
        blockers.append(_block("schema_registry_keys", "Round-20 schema registry keys are not exact.", expected=sorted(SCHEMA_REGISTRY), actual=sorted(registry)))
    schemas: dict[str, dict[str, Any]] = {}
    for name, (version, relative) in SCHEMA_REGISTRY.items():
        entry = registry.get(name, {})
        path = repository / relative
        if entry.get("schema_version") != version or entry.get("path") != relative or not path.is_file():
            blockers.append(_block("schema_registry_entry", "Round-20 schema registry entry is invalid.", schema=name))
            continue
        try:
            schema = _load_json(path)
            Draft202012Validator.check_schema(schema)
            schemas[name] = schema
        except Exception as exc:  # jsonschema emits multiple schema exception classes.
            blockers.append(_block("schema_invalid", str(exc), schema=name))
    if len(schemas) == len(SCHEMA_REGISTRY):
        _validate_schema_semantics(schemas, blockers)

    stage0_gate = paths.get("inherited_stage0.stage0_audit")
    stage0 = _load_json(stage0_gate) if stage0_gate and stage0_gate.is_file() else {}
    if stage0.get("status") != "passed" or stage0.get("gate") != "real_e2_e4_readiness_mainline_integrated" or stage0.get("candidate_commit") != BASE_COMMIT:
        blockers.append(_block("stage0_gate", "The accepted Round-19I Gate is missing or does not bind to the frozen base commit."))
    full_junit = paths.get("test_identity_registry.successful_execution_junit")
    baseline_junit = paths.get("test_identity_registry.identity_baseline_artifact")
    reconciliation_path = paths.get("test_identity_registry.reconciliation_artifact")
    execution_audit_path = paths.get("test_identity_registry.execution_audit")
    inventory_audit_path = paths.get("test_identity_registry.inventory_audit")
    baseline_summary = _junit_summary(baseline_junit) if baseline_junit and baseline_junit.is_file() else {}
    execution_summary = _junit_summary(full_junit) if full_junit and full_junit.is_file() else {}
    if baseline_summary.get("tests") != EXPECTED_TEST_COUNT:
        blockers.append(_block("test_identity_baseline", "The inherited identity baseline does not contain 607 test identities."))
    if execution_summary != {"tests": EXPECTED_TEST_COUNT, "failures": 0, "errors": 0, "skipped": 0}:
        blockers.append(_block("test_execution", "The successful Round-19I execution JUnit does not report 607/607 with zero failures, errors, and skips.", observed=execution_summary))
    reconciliation = _load_json(reconciliation_path) if reconciliation_path and reconciliation_path.is_file() else {}
    expected_reconciliation = {
        "status": "passed",
        "baseline_identity_count": EXPECTED_TEST_COUNT,
        "expected_identity_count": EXPECTED_TEST_COUNT,
        "executed_identity_count": EXPECTED_TEST_COUNT,
        "tests_collected_or_executed": EXPECTED_TEST_COUNT,
        "failures": 0,
        "errors": 0,
        "skips": 0,
        "missing_accepted_identities": [],
        "unexpected_unreconciled_identities": [],
    }
    if any(reconciliation.get(key) != value for key, value in expected_reconciliation.items()):
        blockers.append(_block("test_identity_reconciliation", "The inherited 607-test identity reconciliation is not exact."))
    execution_audit = _load_json(execution_audit_path) if execution_audit_path and execution_audit_path.is_file() else {}
    inventory_audit = _load_json(inventory_audit_path) if inventory_audit_path and inventory_audit_path.is_file() else {}
    if execution_audit.get("status") != "passed" or execution_audit.get("tests", {}).get("passed") != EXPECTED_TEST_COUNT or execution_audit.get("tests", {}).get("failures") != 0 or execution_audit.get("tests", {}).get("errors") != 0 or execution_audit.get("tests", {}).get("skips") != 0:
        blockers.append(_block("stage0_execution_audit", "The inherited Stage-0 full-suite audit is not a green 607-test execution."))
    if inventory_audit.get("status") != "passed" or inventory_audit.get("identity_diff_count") != 0 or inventory_audit.get("passed_identity_count") != EXPECTED_TEST_COUNT:
        blockers.append(_block("stage0_inventory_audit", "The inherited Stage-0 test inventory audit is not an exact 607-identity reconciliation."))
    if full_junit and full_junit.is_file() and execution_audit.get("artifact_hashes", {}).get("full-suite.junit.xml") != _sha256(full_junit):
        blockers.append(_block("execution_junit_hash", "The successful JUnit hash does not match the inherited Stage-0 execution audit."))
    if baseline_junit and baseline_junit.is_file() and inventory_audit.get("artifact_hashes", {}).get("accepted-test-baseline.junit.xml") != _sha256(baseline_junit):
        blockers.append(_block("baseline_junit_hash", "The identity baseline hash does not match the inherited Stage-0 inventory audit."))
    if reconciliation_path and reconciliation_path.is_file() and inventory_audit.get("artifact_hashes", {}).get("test-inventory-reconciliation.json") != _sha256(reconciliation_path):
        blockers.append(_block("reconciliation_hash", "The identity reconciliation hash does not match the inherited Stage-0 inventory audit."))

    dirty = _git(repository, "status", "--porcelain")
    if require_clean and dirty:
        blockers.append(_block("dirty_candidate", "Exact-clean validation was requested but the candidate has tracked changes.", status=dirty.splitlines()))

    source_hashes = {label: _sha256(path) for label, path in sorted(paths.items()) if path.is_file()}
    report: dict[str, Any] = {
        "schema_version": "neuronsifter-round20-shared-schema-freeze-v1",
        "gate_uid": GATE_UID,
        "status": "passed" if not blockers else "blocked",
        "assessment": "round20_shared_fact_sources_frozen" if not blockers else "blocked_shared_schema_freeze",
        "blocker_count": len(blockers),
        "blockers": sorted(blockers, key=lambda item: (str(item.get("code")), json.dumps(item, sort_keys=True))),
        "contract_hash": _sha256(contract_path),
        "candidate_commit": candidate_commit,
        "candidate_tree": candidate_tree,
        "candidate_branch": candidate_branch,
        "dirty_status": "clean" if not dirty else "dirty",
        "base_commit": BASE_COMMIT,
        "schema_registry_hash": _sha256(registry_path),
        "shared_fact_source_hashes": source_hashes,
        "test_identity_registry": {
            "identity_count": EXPECTED_TEST_COUNT,
            "identity_baseline": baseline_summary,
            "successful_execution": execution_summary,
            "reconciliation": reconciliation,
            "historical_baseline_failures_preserved": baseline_summary.get("failures", 0) > 0 or baseline_summary.get("errors", 0) > 0,
            "successful_execution_job_id": execution_audit.get("scheduler", {}).get("job_id"),
        },
        "scope": {"public_only": True, "controlled_sources": 0, "training_authorized": False, "training_started": False, "evidence_fabric_sealed": False, "dt_core": "not_evaluated", "frozen_gap_count": 20},
        "execution_authorization": "track_contracts_authorized" if not blockers else "blocked_until_shared_schema_gate",
        "track_execution_authorized": not blockers,
        "allen_payload_download_authorized": False,
        "e2_e4_training_authorized": False,
        "next_action": "begin_governed_round20_track_a_b_c_contract_work" if not blockers else "repair_shared_schema_freeze_blockers_and_rerun_with_new_run_identity",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--contract", type=Path, default=CONTRACT_RELATIVE)
    parser.add_argument("--output", type=Path, default=AUDIT_RELATIVE)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    contract = args.contract if args.contract.is_absolute() else repository / args.contract
    output = args.output if args.output.is_absolute() else repository / args.output
    report = validate(repository=repository, contract_path=contract.resolve(), require_clean=args.require_clean)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "assessment": report["assessment"], "blockers": report["blocker_count"]}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
