"""Build fail-closed Round-11E triage records from frozen Round-11D evidence.

The module intentionally performs no test execution and never opens a missing
artifact as a substitute for the frozen suite worktree.  It turns the preserved
JUnit report into a stable ledger so clean-worktree reproduction can be planned
without treating test-module names as causal evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

from Libs.utils.data.raw_data_registry import write_json, write_tsv


FROZEN_INPUT_FILENAMES = {
    "execution": "dt_round11d_full_suite_execution.json",
    "junit": "dt_round11d_full_suite_results.xml",
    "failure_triage": "dt_round11d_full_suite_failure_triage.json",
    "closure_decision": "dt_round11_closure_decision.json",
}
AUDIT_FILENAMES = {
    "input_snapshot": "dt_round11e_input_evidence_snapshot.json",
    "ledger": "dt_round11e_testcase_failure_ledger.tsv",
    "family_summary": "dt_round11e_failure_family_summary.json",
    "dependency_graph": "dt_round11e_failure_dependency_graph.json",
    "targeted_reproduction": "dt_round11e_targeted_reproduction.tsv",
}
OBSERVED_AUDIT_FILENAMES = {
    "reproduction_observation": "dt_round11e_reproduction_observation.json",
    "targeted_gate": "dt_round11e_targeted_gate.json",
    "repair_wave_record": "dt_round11e_repair_wave_record.json",
}
OBSERVATION_SOURCE_FILENAMES = {
    "wave_ab_junit": "dt_round11e_final_wave_ab_results.xml",
    "wave_ab_stdout": "dt_round11e_final_wave_ab.stdout",
    "wave_ab_stderr": "dt_round11e_final_wave_ab.stderr",
    "wave_ab_provenance": "dt_round11e_clean_worktree_provenance.json",
    "rc003_junit": "dt_round11e_rc003_clean_results.xml",
    "rc003_stdout": "dt_round11e_rc003_clean.stdout",
    "rc003_stderr": "dt_round11e_rc003_clean.stderr",
    "rc003_provenance": "dt_round11e_rc003_clean_worktree_provenance.json",
    "rc003_diagnostic": "dt_round11e_rc003_a8_diag.stdout",
    "focus_junit": "dt_round11e_a8_focus_results.xml",
    "focus_stdout": "dt_round11e_a8_focus.stdout",
    "focus_stderr": "dt_round11e_a8_focus.stderr",
    "focus_provenance": "dt_round11e_a8_clean_worktree_provenance.json",
    "input_linkage_manifest": "dt_round11e_processed_input_linkage_manifest.json",
    "namespace_smoke_stdout": "dt_round11e_final_namespace_smoke.stdout",
    "scheduler_recheck": "dt_round11e_targeted_scheduler_recheck.json",
}

FAILURE_FAMILIES = (
    "collection_or_import",
    "fixture_or_setup",
    "config_or_schema",
    "missing_asset",
    "environment_or_dependency",
    "data_contract",
    "model_contract",
    "tool_contract",
    "report_or_publication",
    "assertion_only",
    "unknown",
)
REPRODUCTION_STATUSES = (
    "reproduced",
    "not_reproduced",
    "environment_specific",
    "intermittent",
    "blocked_missing_evidence",
)
ROOT_CAUSE_CONFIDENCES = ("confirmed", "strongly_supported", "possible", "unknown")

LEDGER_FIELDS = [
    "testcase_uid",
    "test_id",
    "test_file",
    "test_class",
    "failure_or_error",
    "exception_type",
    "message",
    "stacktrace_hash",
    "first_project_frame",
    "setup_call_or_test_body",
    "likely_dependency",
    "reproduction_status",
    "root_cause_status",
    "repair_status",
    "failure_family",
    "candidate_root_cause_uid",
    "round11_relation",
    "schema_registry_relation",
    "junit_testcase_index",
    "observed_root_cause_uid",
    "observed_failure_family",
    "observation_job_id",
    "observation_junit_path",
    "observation_junit_sha256",
    "reproduction_relation",
    "repair_wave",
]
REPRODUCTION_FIELDS = [
    "testcase_uid",
    "test_id",
    "test_file",
    "failure_or_error",
    "failure_family",
    "candidate_root_cause_uid",
    "source_job_id",
    "reproduction_command",
    "reproduction_status",
    "clean_worktree_identity",
    "stdout_path",
    "stderr_path",
    "stable_failure_observed",
    "notes",
    "observed_root_cause_uid",
    "observation_job_id",
    "observation_junit_path",
    "observation_junit_sha256",
    "reproduction_relation",
    "repair_wave",
]

# These columns are directly derived from the frozen Round-11D JUnit and must
# remain stable when a later clean-worktree observation enriches the ledger.
# Reproduction/root-cause/repair state is intentionally excluded because those
# are Round-11E observations rather than fields emitted by job 490754.
FROZEN_LEDGER_FIELDS = [
    "testcase_uid",
    "test_id",
    "test_file",
    "test_class",
    "failure_or_error",
    "exception_type",
    "message",
    "stacktrace_hash",
    "first_project_frame",
    "setup_call_or_test_body",
    "likely_dependency",
    "failure_family",
    "candidate_root_cause_uid",
    "round11_relation",
    "schema_registry_relation",
    "junit_testcase_index",
]
FROZEN_REPRODUCTION_FIELDS = [
    "testcase_uid",
    "test_id",
    "test_file",
    "failure_or_error",
    "failure_family",
    "candidate_root_cause_uid",
    "source_job_id",
    "reproduction_command",
]

G0_INTEGRITY_TEST_ID = (
    "Libs.tests.test_g0_integrity::"
    "test_g0_integrity_binds_exact_original_blockers_and_preserves_unreconciled_numeric_lineage"
)
ROOT_WAVE_B_UNIT_JUNIT_FILENAME = "dt_round11e_root_linkage_unit_results.xml"
WAVE_B_CLEAN_REPAIR_FILE_SHA256 = {
    "Libs/utils/data/round11e_processed_input_linkage.py": "007d03846ef85ed3505f44cb671884403451aadb779401ebe0e53b170510cabb",
    "Libs/scripts/builders/build_dt_round11e_processed_input_manifest.py": "f0c6c7b639157c85f9885c3677a33bfd01be1a1680ca407bc29406f76839bf52",
    "Libs/scripts/validators/run_dt_round11e_processed_input_linkage.py": "2b405f8f809ec39fd23de00222eee13c86f54fcbd411927aa5ec743e9c6c52ff",
    "Libs/tests/test_dt_round11e_processed_input_linkage.py": "2a34e30b363019e84f7af00cc99407ead8aee984246deb52aaab05f76c465cff",
}
# The root worktree now carries the later, source-migrated copies.  Keep the
# old clean-repair digests above immutable and bind this separate identity so
# a historical hash is never silently replaced by a new expected value.
WAVE_B_CURRENT_SOURCE_FILE_SHA256 = {
    "Libs/utils/data/round11e_processed_input_linkage.py": "0d41fbdb573b1a65f50d555e6689f4802cd98798c449a5972649d423debd35fa",
    "Libs/scripts/builders/build_dt_round11e_processed_input_manifest.py": "705f5715b810fe937c7b1c86f8963a55c662ae9f0cb475120be3745c40fe57f0",
    "Libs/scripts/validators/run_dt_round11e_processed_input_linkage.py": "28e98ad1e88387a4b36be3bf5696757d8e0165080639757c83e962fa05ba9d5c",
    "Libs/tests/test_dt_round11e_processed_input_linkage.py": "3473a477408d73c8846e4f59edbf808682e3e21ad4854c5920ab2250f0cbebd3",
}

_PROJECT_FRAME_RE = re.compile(r"(?P<path>Libs/[A-Za-z0-9_./-]+\.py):(?P<line>\d+):")
_MISSING_PATH_RE = re.compile(r"No such file or directory:\s*'(?P<path>[^']+)'")
_LINKAGE_MISSING_RE = re.compile(r"Missing\s+(?P<path>Output/[^\s]+)\s+in this worktree")
_NUMERIC_MISSING_RE = re.compile(r"Numeric lineage source artifacts are missing:\s*\[(?P<names>[^\]]+)\]")


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without loading an artifact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for {label}")
    return dict(value)


def _as_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list for {label}")
    return value


def _canonical_repo_path(value: str) -> str:
    """Remove the transient sparse-worktree prefix from a missing-path report."""

    for marker in ("Input/", "Output/", "Libs/"):
        index = value.find(marker)
        if index >= 0:
            return value[index:]
    return value


def _normalize_message(value: str) -> str:
    return " ".join(value.split())


def _exception_type(issue: ET.Element, text: str) -> str:
    explicit = str(issue.attrib.get("type", "")).strip()
    if explicit:
        return explicit.rsplit(".", maxsplit=1)[-1]
    for candidate in ("FileNotFoundError", "AssertionError", "KeyError", "ValueError", "RuntimeError", "ImportError"):
        if candidate in text:
            return candidate
    match = re.search(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b", text)
    return match.group(1) if match else "unknown"


def _project_frames(text: str) -> list[str]:
    frames = [f"{match.group('path')}:{match.group('line')}" for match in _PROJECT_FRAME_RE.finditer(text)]
    return list(dict.fromkeys(frames))


def _first_project_frame(frames: Iterable[str]) -> str:
    values = list(frames)
    for frame in values:
        if not frame.startswith("Libs/tests/"):
            return frame
    return values[0] if values else "unavailable"


def _missing_resources(text: str) -> list[str]:
    resources = [_canonical_repo_path(match.group("path")) for match in _MISSING_PATH_RE.finditer(text)]
    resources.extend(_canonical_repo_path(match.group("path")) for match in _LINKAGE_MISSING_RE.finditer(text))
    numeric_match = _NUMERIC_MISSING_RE.search(text)
    if numeric_match:
        names = ",".join(part.strip().strip("'\"") for part in numeric_match.group("names").split(","))
        resources.append(f"numeric_lineage_source_inventory:{names}")
    return list(dict.fromkeys(resources))


def _failure_family(exception_type: str) -> str:
    if exception_type == "FileNotFoundError":
        return "missing_asset"
    if exception_type == "AssertionError":
        return "assertion_only"
    return "unknown"


def _candidate_root_cause_uid(exception_type: str, resources: list[str]) -> str:
    if exception_type == "FileNotFoundError":
        if "Output/audits/formal_experiment_readiness_audit.json" in resources:
            return "R11E-RC-002"
        return "R11E-RC-001"
    if exception_type == "AssertionError":
        return "R11E-RC-003"
    return "R11E-RC-004"


def _dependency_text(exception_type: str, resources: list[str]) -> str:
    if exception_type != "FileNotFoundError":
        return "not established from the preserved JUnit traceback"
    if "Output/audits/formal_experiment_readiness_audit.json" in resources:
        return "hash-bound formal_experiment_readiness_audit linkage via NRNSIFTER_AUDIT_ARTIFACT_ROOT"
    if any(item.startswith("numeric_lineage_source_inventory:") for item in resources):
        return "configured numeric-lineage source inventory required by the current audit reader"
    if resources:
        return f"configured required artifact unavailable in the frozen sparse worktree: {resources[0]}"
    return "missing artifact indicated by FileNotFoundError; logical resource was not recoverable from the traceback"


def _root_cause_status(exception_type: str) -> str:
    return "strongly_supported" if exception_type == "FileNotFoundError" else "unknown"


def _test_file(test_class: str) -> str:
    prefix = "Libs.tests."
    if test_class.startswith(prefix):
        return f"Libs/tests/{test_class.removeprefix(prefix).replace('.', '/')}.py"
    return "unavailable"


def _testcase_uid(job_id: str, test_id: str, failure_or_error: str) -> str:
    digest = hashlib.sha256(f"{job_id}|{test_id}|{failure_or_error}".encode("utf-8")).hexdigest()
    return f"r11e-tc-{digest[:20]}"


def _parse_junit(junit_path: Path, job_id: str) -> tuple[dict[str, int], list[dict[str, Any]]]:
    root = ET.parse(junit_path).getroot()
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))
    result = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    records: list[dict[str, Any]] = []
    for index, testcase in enumerate(root.iter("testcase"), start=1):
        error = testcase.find("error")
        failure = testcase.find("failure")
        if error is None and failure is None:
            continue
        issue = error if error is not None else failure
        failure_or_error = "error" if error is not None else "failure"
        test_class = str(testcase.attrib.get("classname", "")).strip()
        test_name = str(testcase.attrib.get("name", "")).strip()
        if not test_class or not test_name:
            raise ValueError(f"JUnit testcase at index {index} has no class or name")
        test_id = f"{test_class}::{test_name}"
        stacktrace = str(issue.text or "")
        exception_type = _exception_type(issue, f"{issue.attrib.get('message', '')}\n{stacktrace}")
        resources = _missing_resources(f"{issue.attrib.get('message', '')}\n{stacktrace}")
        project_frames = _project_frames(stacktrace)
        setup = "fixture_or_setup" if "failed on setup with" in str(issue.attrib.get("message", "")) else "test_body"
        records.append(
            {
                "testcase_uid": _testcase_uid(job_id, test_id, failure_or_error),
                "test_id": test_id,
                "test_file": _test_file(test_class),
                "test_class": test_class,
                "failure_or_error": failure_or_error,
                "exception_type": exception_type,
                "message": _normalize_message(str(issue.attrib.get("message", ""))),
                "stacktrace_hash": hashlib.sha256(stacktrace.encode("utf-8")).hexdigest(),
                "first_project_frame": _first_project_frame(project_frames),
                "setup_call_or_test_body": setup,
                "likely_dependency": _dependency_text(exception_type, resources),
                "reproduction_status": "blocked_missing_evidence",
                "root_cause_status": _root_cause_status(exception_type),
                "repair_status": "not_attempted_pending_clean_reproduction",
                "failure_family": _failure_family(exception_type),
                "candidate_root_cause_uid": _candidate_root_cause_uid(exception_type, resources),
                "round11_relation": "not_established_from_traceback",
                "schema_registry_relation": "not_established_from_traceback",
                "junit_testcase_index": str(index),
                "_resources": resources,
                "_project_frames": project_frames,
            }
        )
    counted = Counter(row["failure_or_error"] for row in records)
    if counted["failure"] != result["failures"] or counted["error"] != result["errors"]:
        raise ValueError(
            "JUnit non-pass count mismatch: "
            f"root={result}, parsed_failures={counted['failure']}, parsed_errors={counted['error']}"
        )
    return result, records


def _artifact_binding(repo_root: Path, path: Path, role: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen Round-11D evidence is unavailable: {path}")
    try:
        reference = path.relative_to(repo_root).as_posix()
    except ValueError:
        reference = str(path)
    return {
        "role": role,
        "path": reference,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "read_only_input": True,
    }


def _expect_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"Frozen evidence mismatch for {label}: expected {expected!r}, observed {actual!r}")


def _validate_frozen_relationships(
    execution: Mapping[str, object],
    prior_triage: Mapping[str, object],
    closure: Mapping[str, object],
    junit_result: Mapping[str, int],
    records: list[dict[str, Any]],
) -> None:
    job_id = str(execution.get("job_id", ""))
    if not job_id:
        raise ValueError("Frozen execution record has no job_id")
    _expect_equal(str(prior_triage.get("job_id", "")), job_id, "failure-triage job_id")
    full_suite = _as_mapping(closure.get("full_suite"), "closure.full_suite")
    raw_closure_job_id = full_suite.get("job_id")
    closure_job_id = "" if raw_closure_job_id in (None, "") else str(raw_closure_job_id)
    _expect_equal(closure_job_id in {"", job_id}, True, "closure job_id")
    _expect_equal(_as_mapping(execution.get("structured_test_result"), "execution.structured_test_result"), dict(junit_result), "execution/JUnit counts")
    _expect_equal(_as_mapping(prior_triage.get("test_result"), "failure_triage.test_result"), dict(junit_result), "failure-triage/JUnit counts")
    if closure_job_id:
        _expect_equal(_as_mapping(full_suite.get("structured_test_result"), "closure.full_suite.structured_test_result"), dict(junit_result), "closure/JUnit counts")
    else:
        # A later Round-11E/G deferral snapshot is an explicit non-execution
        # record.  It must remain blocked and must not be mistaken for a
        # replacement result from the frozen Round-11D job.
        _expect_equal(full_suite.get("status"), "not_submitted_targeted_gate_blocked", "deferred closure status")
        deferred_counts = _as_mapping(full_suite.get("structured_test_result"), "closure.full_suite.structured_test_result")
        _expect_equal(deferred_counts, {"tests": None, "failures": None, "errors": None, "skipped": None}, "deferred closure counts")
    if _as_mapping(closure.get("round_11"), "closure.round_11").get("closure") != "blocked_full_suite":
        raise ValueError("Frozen closure decision must remain blocked_full_suite during Round-11E triage")
    prior_entries = _as_list(prior_triage.get("failures"), "failure_triage.failures")
    prior_keys = {
        (str(_as_mapping(item, "failure_triage.failure").get("test_id", "")), str(_as_mapping(item, "failure_triage.failure").get("failure_type", "")))
        for item in prior_entries
    }
    ledger_keys = {(str(item["test_id"]), str(item["failure_or_error"])) for item in records}
    _expect_equal(prior_keys, ledger_keys, "failure-triage/JUnit testcase identities")


def _shared_frames(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    shared = set(records[0]["_project_frames"])
    for record in records[1:]:
        shared &= set(record["_project_frames"])
    return sorted(shared)


def _root_cause_graph(records: list[dict[str, Any]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["candidate_root_cause_uid"])].append(record)

    specifications = {
        "R11E-RC-001": {
            "root_cause": "required generated Input/processed artifacts were unavailable beneath the frozen sparse worktree",
            "shared_missing_resource": "multiple configured Input/processed artifacts and numeric-lineage source inputs resolve beneath the frozen sparse worktree",
            "shared_config_dependency": "repository-relative configured input paths require an available artifact store in the clean worktree",
            "confidence": "strongly_supported",
            "evidence_boundary": "The preserved trace proves unavailability at runtime. It does not establish the producer or lifecycle that should populate the worktree.",
        },
        "R11E-RC-002": {
            "root_cause": "the hash-bound formal-experiment-readiness artifact was unavailable through its explicit read-only linkage",
            "shared_missing_resource": "Output/audits/formal_experiment_readiness_audit.json",
            "shared_config_dependency": "iclr_design_contract_union.g0_read_only_artifact_linkage and NRNSIFTER_AUDIT_ARTIFACT_ROOT",
            "confidence": "strongly_supported",
            "evidence_boundary": "The preserved trace proves the linkage was unavailable to the frozen job. It does not establish whether an approved root was intentionally omitted or unavailable on the node.",
        },
        "R11E-RC-003": {
            "root_cause": "design-contract report returned blocked without a traceback exposing its blocker",
            "shared_missing_resource": None,
            "shared_config_dependency": "not established from the preserved JUnit assertion trace",
            "confidence": "unknown",
            "evidence_boundary": "The two assertion failures must not be attributed to Round-11, schemas, or the missing-asset candidates without report-level evidence.",
        },
        "R11E-RC-004": {
            "root_cause": "unclassified preserved failure",
            "shared_missing_resource": None,
            "shared_config_dependency": "unknown",
            "confidence": "unknown",
            "evidence_boundary": "No direct root-cause evidence was recovered from the preserved JUnit trace.",
        },
    }
    candidates = []
    for uid in sorted(grouped):
        rows = grouped[uid]
        specification = specifications[uid]
        candidates.append(
            {
                "candidate_root_cause_uid": uid,
                **specification,
                "supporting_testcases": [
                    {"testcase_uid": row["testcase_uid"], "test_id": row["test_id"], "failure_or_error": row["failure_or_error"]}
                    for row in rows
                ],
                "affected_fixtures_or_modules": sorted({str(row["first_project_frame"]) for row in rows}),
                "shared_stack_frames": _shared_frames(rows),
                "shared_stack_frames_interpretation": "no single shared project frame" if not _shared_frames(rows) else "direct intersection of preserved project frames",
                "error_count": sum(row["failure_or_error"] == "error" for row in rows),
                "failure_count": sum(row["failure_or_error"] == "failure" for row in rows),
            }
        )
    return {
        "schema_version": "dt-round11e-failure-dependency-graph-v1",
        "audit_kind": "failure_dependency_graph",
        "status": "triage_initialized_pending_clean_reproduction",
        "root_cause_graph": candidates,
        "non_independence_finding": "The 33 non-passing testcases are represented by candidate dependency groups; they are not treated as 33 independent implementation defects.",
        "unresolved_testcase_count": 0,
        "repair_authorization": "no repair attempted; clean-worktree reproduction is required before implementation changes",
    }


def _family_summary(records: list[dict[str, Any]], junit_result: Mapping[str, int]) -> dict[str, object]:
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        family_rows[str(row["failure_family"])].append(row)
    category_counts = {family: len(family_rows[family]) for family in FAILURE_FAMILIES}
    ordered_rows = sorted(records, key=lambda row: (0 if row["failure_or_error"] == "error" else 1, int(row["junit_testcase_index"])))
    families = []
    for family in FAILURE_FAMILIES:
        rows = family_rows[family]
        if not rows:
            continue
        families.append(
            {
                "family": family,
                "testcase_count": len(rows),
                "error_count": sum(row["failure_or_error"] == "error" for row in rows),
                "failure_count": sum(row["failure_or_error"] == "failure" for row in rows),
                "testcase_uids": [row["testcase_uid"] for row in rows],
                "candidate_root_cause_uids": sorted({str(row["candidate_root_cause_uid"]) for row in rows}),
                "reproduction_state": "blocked_missing_evidence",
            }
        )
    return {
        "schema_version": "dt-round11e-failure-family-summary-v1",
        "audit_kind": "failure_family_summary",
        "status": "triage_initialized_pending_clean_reproduction",
        "classification_order": list(FAILURE_FAMILIES),
        "error_priority_policy": "all 16 error testcases are listed before assertion failures for reproduction planning",
        "junit_result": dict(junit_result),
        "category_counts": category_counts,
        "families": families,
        "error_first_reproduction_order": [row["testcase_uid"] for row in ordered_rows],
        "attribution_policy": "classification follows direct exception and traceback evidence; test module names do not establish Round-11 or schema-registry relatedness",
    }


def _targeted_reproduction(records: list[dict[str, Any]], job_id: str) -> list[dict[str, object]]:
    rows = []
    for record in records:
        node_id = f"{record['test_file']}::{record['test_id'].rsplit('::', maxsplit=1)[-1]}"
        rows.append(
            {
                "testcase_uid": record["testcase_uid"],
                "test_id": record["test_id"],
                "test_file": record["test_file"],
                "failure_or_error": record["failure_or_error"],
                "failure_family": record["failure_family"],
                "candidate_root_cause_uid": record["candidate_root_cause_uid"],
                "source_job_id": job_id,
                "reproduction_command": f"python -m pytest {shlex.quote(node_id)} -q",
                "reproduction_status": "blocked_missing_evidence",
                "clean_worktree_identity": "not_recorded",
                "stdout_path": "",
                "stderr_path": "",
                "stable_failure_observed": "not_attempted",
                "notes": "Initialized from frozen JUnit only; no test command has been run and no repair is authorized before clean-worktree evidence is captured.",
            }
        )
    return rows


def build_round11e_triage(
    repo_root: Path,
    execution_path: Path,
    junit_path: Path,
    failure_triage_path: Path,
    closure_path: Path,
) -> dict[str, object]:
    """Parse frozen evidence and return deterministic Round-11E analysis artifacts."""

    execution = _load_object(execution_path)
    prior_triage = _load_object(failure_triage_path)
    closure = _load_object(closure_path)
    job_id = str(execution.get("job_id", ""))
    junit_result, records = _parse_junit(junit_path, job_id)
    _validate_frozen_relationships(execution, prior_triage, closure, junit_result, records)

    streams = _as_mapping(execution.get("streams"), "execution.streams")
    scheduler = _as_mapping(execution.get("scheduler"), "execution.scheduler")
    source_artifacts = [
        _artifact_binding(repo_root, execution_path, "frozen_execution_record"),
        _artifact_binding(repo_root, junit_path, "frozen_junit_results"),
        _artifact_binding(repo_root, failure_triage_path, "frozen_failure_triage"),
        _artifact_binding(repo_root, closure_path, "frozen_closure_decision"),
    ]
    referenced_artifacts = []
    for stream_name in ("stdout", "stderr", "junit"):
        stream = _as_mapping(streams.get(stream_name), f"execution.streams.{stream_name}")
        stream_path = repo_root / str(stream.get("path", ""))
        binding = _artifact_binding(repo_root, stream_path, f"execution_stream_{stream_name}")
        _expect_equal(binding["sha256"], stream.get("sha256"), f"execution stream {stream_name} hash")
        _expect_equal(binding["bytes"], stream.get("bytes"), f"execution stream {stream_name} byte count")
        referenced_artifacts.append(binding)
    scontrol = _as_mapping(scheduler.get("scontrol"), "execution.scheduler.scontrol")
    scontrol_path = repo_root / str(scontrol.get("path", ""))
    scontrol_binding = _artifact_binding(repo_root, scontrol_path, "scheduler_terminal_evidence")
    _expect_equal(scontrol_binding["sha256"], scontrol.get("sha256"), "scheduler terminal evidence hash")
    _expect_equal(scontrol_binding["bytes"], scontrol.get("bytes"), "scheduler terminal evidence byte count")
    referenced_artifacts.append(scontrol_binding)

    dirty_status = _as_mapping(execution.get("clean_worktree_dirty_status"), "execution.clean_worktree_dirty_status")
    evidence_fabric = _as_mapping(closure.get("evidence_fabric"), "closure.evidence_fabric")
    snapshot = {
        "schema_version": "dt-round11e-input-evidence-snapshot-v1",
        "audit_kind": "round11e_input_evidence_snapshot",
        "source_job_id": job_id,
        "source_job_name": execution.get("job_name"),
        "frozen_execution_identity": {
            "clean_worktree_commit": execution.get("clean_worktree_commit"),
            "clean_config_sha256": execution.get("clean_config_sha256"),
            "root_config_sha256": execution.get("root_config_sha256"),
            "resolved_registry_sha256": execution.get("resolved_registry_sha256"),
            "environment_hash": execution.get("environment_hash"),
            "clean_worktree_dirty_status": dirty_status,
        },
        "frozen_input_artifacts": source_artifacts,
        "execution_referenced_artifacts": referenced_artifacts,
        "junit_testcase_count": junit_result["tests"],
        "failure_count": junit_result["failures"],
        "error_count": junit_result["errors"],
        "skipped_count": junit_result["skipped"],
        "job_terminal_evidence": {
            "terminal_status": execution.get("terminal_status"),
            "exit_code": execution.get("exit_code"),
            "sacct_available": scheduler.get("sacct_available"),
            "terminal_evidence_source": scheduler.get("terminal_evidence_source"),
            "scontrol_path": scontrol_binding["path"],
            "scontrol_sha256": scontrol_binding["sha256"],
        },
        "known_provenance_limitations": [
            "clean_worktree_dirty_status was not observed at submission or terminal time",
            str(dirty_status.get("scope", "unknown")),
            "sacct_available=false; terminal state is supported by scontrol and preserved scheduler logs",
            "the frozen JUnit exposes runtime unavailability but does not prove an artifact producer or lifecycle cause",
            "this frozen-input snapshot does not execute a reproduction; later clean-worktree observations are recorded in a separate content-addressed artifact",
        ],
        "frozen_state": {
            "round11_closure": _as_mapping(closure.get("round_11"), "closure.round_11").get("closure"),
            "evidence_fabric_sealed": evidence_fabric.get("sealed"),
            "frozen_input_gap_count": evidence_fabric.get("frozen_input_gap_count"),
            "dt_core": _as_mapping(closure.get("dt_core"), "closure.dt_core").get("status"),
        },
        "read_only_input_policy": "This snapshot is content-addressed from preserved Round-11D inputs. It neither alters the frozen job evidence nor converts triage into a reproduction.",
    }
    observation = _build_reproduction_observation(repo_root, execution, records)
    enriched_records, targeted_reproduction = _enrich_with_observation(repo_root, records, observation)
    ledger = [{field: row.get(field, "") for field in LEDGER_FIELDS} for row in enriched_records]
    base_family_summary = _family_summary(records, junit_result)
    base_dependency_graph = _root_cause_graph(records)
    return {
        "input_snapshot": snapshot,
        "ledger": ledger,
        "family_summary": _enrich_family_summary(base_family_summary, enriched_records),
        "dependency_graph": _enrich_dependency_graph(base_dependency_graph, enriched_records),
        "targeted_reproduction": targeted_reproduction,
        "reproduction_observation": observation,
        "targeted_gate": _targeted_gate(observation, enriched_records),
        "repair_wave_record": _repair_wave_record(repo_root, observation),
    }


def write_round11e_triage(payload: Mapping[str, object], audit_root: Path) -> None:
    """Atomically write only the new Round-11E audit outputs."""

    input_snapshot = _as_mapping(payload.get("input_snapshot"), "payload.input_snapshot")
    ledger = _as_list(payload.get("ledger"), "payload.ledger")
    family_summary = _as_mapping(payload.get("family_summary"), "payload.family_summary")
    dependency_graph = _as_mapping(payload.get("dependency_graph"), "payload.dependency_graph")
    targeted_reproduction = _as_list(payload.get("targeted_reproduction"), "payload.targeted_reproduction")
    reproduction_observation = _as_mapping(payload.get("reproduction_observation"), "payload.reproduction_observation")
    targeted_gate = _as_mapping(payload.get("targeted_gate"), "payload.targeted_gate")
    repair_wave_record = _as_mapping(payload.get("repair_wave_record"), "payload.repair_wave_record")
    write_json(audit_root / AUDIT_FILENAMES["input_snapshot"], input_snapshot)
    write_tsv(audit_root / AUDIT_FILENAMES["ledger"], LEDGER_FIELDS, [dict(row) for row in ledger])
    write_json(audit_root / AUDIT_FILENAMES["family_summary"], family_summary)
    write_json(audit_root / AUDIT_FILENAMES["dependency_graph"], dependency_graph)
    write_tsv(audit_root / AUDIT_FILENAMES["targeted_reproduction"], REPRODUCTION_FIELDS, [dict(row) for row in targeted_reproduction])
    write_json(audit_root / OBSERVED_AUDIT_FILENAMES["reproduction_observation"], reproduction_observation)
    write_json(audit_root / OBSERVED_AUDIT_FILENAMES["targeted_gate"], targeted_gate)
    write_json(audit_root / OBSERVED_AUDIT_FILENAMES["repair_wave_record"], repair_wave_record)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _serialize_rows(rows: Iterable[Mapping[str, object]], fields: Iterable[str]) -> list[dict[str, str]]:
    return [{field: "" if row.get(field) is None else str(row.get(field, "")) for field in fields} for row in rows]


def _project_tsv_rows(rows: Iterable[Mapping[str, str]], fields: Iterable[str]) -> list[dict[str, str]]:
    """Project enriched audit rows onto their immutable frozen-input fields."""

    return [{field: str(row.get(field, "")) for field in fields} for row in rows]


def _path_under_repo(repo_root: Path, filename: str) -> Path:
    """Return a repository-relative audit path and reject path traversal."""

    relative = Path(filename)
    if len(relative.parts) == 1:
        relative = Path("Output/audits") / relative
    candidate = (repo_root / relative).resolve()
    if not candidate.is_relative_to(repo_root.resolve()):
        raise ValueError(f"Round-11E observation path escapes repository root: {filename}")
    return candidate


def _junit_outcomes(path: Path) -> tuple[dict[str, int], dict[str, dict[str, str]]]:
    """Read explicit pass/fail outcomes from a preserved targeted JUnit report."""

    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    outcomes: dict[str, dict[str, str]] = {}
    for testcase in root.iter("testcase"):
        test_class = str(testcase.attrib.get("classname", "")).strip()
        test_name = str(testcase.attrib.get("name", "")).strip()
        if not test_class or not test_name:
            raise ValueError(f"Targeted JUnit testcase has no class or name: {path}")
        test_id = f"{test_class}::{test_name}"
        if test_id in outcomes:
            raise ValueError(f"Targeted JUnit contains duplicate testcase identity: {test_id}")
        error = testcase.find("error")
        failure = testcase.find("failure")
        skipped = testcase.find("skipped")
        if error is not None:
            outcome = "error"
            issue = error
        elif failure is not None:
            outcome = "failure"
            issue = failure
        elif skipped is not None:
            outcome = "skipped"
            issue = skipped
        else:
            outcome = "passed"
            issue = None
        outcomes[test_id] = {
            "outcome": outcome,
            "exception_type": "" if issue is None else _exception_type(issue, str(issue.text or "")),
            "message": "" if issue is None else _normalize_message(str(issue.attrib.get("message", ""))),
        }
    if len(outcomes) != counts["tests"]:
        raise ValueError(f"Targeted JUnit testcase count mismatch in {path}")
    observed_nonpass = Counter(item["outcome"] for item in outcomes.values())
    if observed_nonpass["failure"] != counts["failures"] or observed_nonpass["error"] != counts["errors"]:
        raise ValueError(f"Targeted JUnit failure/error count mismatch in {path}")
    return counts, outcomes


def _clean_worktree_identity(path: Path, frozen_execution: Mapping[str, object]) -> dict[str, object]:
    """Load a provenance record and prove it is a clean same-code/config reconstruction."""

    payload = _load_object(path)
    if payload.get("audit_kind") != "clean_repair_worktree_provenance":
        raise ValueError(f"Unexpected clean-worktree provenance kind: {path}")
    if payload.get("dirty_status") != "clean":
        raise ValueError(f"Round-11E targeted evidence did not use a clean worktree: {path}")
    relation = _as_mapping(payload.get("frozen_job_relation"), f"{path}.frozen_job_relation")
    if relation.get("job_id") != frozen_execution.get("job_id"):
        raise ValueError(f"Clean-worktree provenance references the wrong frozen job: {path}")
    if relation.get("commit_matches") is not True or relation.get("config_matches") is not True:
        raise ValueError(f"Clean-worktree provenance does not match frozen code/config: {path}")
    if payload.get("git_commit") != frozen_execution.get("clean_worktree_commit"):
        raise ValueError(f"Clean-worktree commit differs from frozen job: {path}")
    config = _as_mapping(payload.get("config"), f"{path}.config")
    if config.get("sha256") != frozen_execution.get("clean_config_sha256"):
        raise ValueError(f"Clean-worktree config differs from frozen job: {path}")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "git_commit": payload.get("git_commit"),
        "dirty_status": payload.get("dirty_status"),
        "config_sha256": config.get("sha256"),
        "resolved_schema_registry_sha256": config.get("resolved_schema_registry_sha256"),
        "environment_hash": _as_mapping(payload.get("environment"), f"{path}.environment").get("environment_hash"),
        "host_resource_profile": payload.get("host_resource_profile"),
        "reproduction_relation": relation.get("reproduction_relation"),
        "provenance_capture_job_id": _as_mapping(
            payload.get("host_resource_profile"), f"{path}.host_resource_profile"
        ).get("slurm_job_id"),
    }


def _artifact_observation(repo_root: Path, filename: str, role: str) -> dict[str, object]:
    return _artifact_binding(repo_root, _path_under_repo(repo_root, filename), role)


def _validate_bound_stream(repo_root: Path, stream: Mapping[str, object], label: str) -> None:
    path = _path_under_repo(repo_root, str(stream.get("path", "")))
    binding = _artifact_binding(repo_root, path, label)
    _expect_equal(binding["sha256"], stream.get("sha256"), f"{label} hash")
    _expect_equal(binding["bytes"], stream.get("bytes"), f"{label} byte count")


def _scheduler_recheck_evidence(repo_root: Path) -> dict[str, object]:
    """Validate the post-retention recheck without treating it as a terminal state."""

    path = _path_under_repo(repo_root, OBSERVATION_SOURCE_FILENAMES["scheduler_recheck"])
    payload = _load_object(path)
    if payload.get("audit_kind") != "round11e_targeted_scheduler_recheck":
        raise ValueError("Round-11E scheduler recheck has an unexpected audit kind")
    expected_job_ids = ["490950", "490951", "490952", "490953"]
    if payload.get("job_ids") != expected_job_ids:
        raise ValueError("Round-11E scheduler recheck job identities are incomplete")
    scontrol = _as_mapping(payload.get("scontrol"), "scheduler_recheck.scontrol")
    per_job = _as_mapping(scontrol.get("per_job"), "scheduler_recheck.scontrol.per_job")
    if set(per_job) != set(expected_job_ids):
        raise ValueError("Round-11E scheduler recheck scontrol results are incomplete")
    for job_id in expected_job_ids:
        result = _as_mapping(per_job[job_id], f"scheduler_recheck.scontrol.{job_id}")
        if result.get("returncode") != 1:
            raise ValueError(f"Round-11E scontrol recheck has unexpected return code for {job_id}")
        _validate_bound_stream(repo_root, _as_mapping(result.get("stdout"), f"scontrol.{job_id}.stdout"), f"scontrol_{job_id}_stdout")
        _validate_bound_stream(repo_root, _as_mapping(result.get("stderr"), f"scontrol.{job_id}.stderr"), f"scontrol_{job_id}_stderr")
        stderr_path = _path_under_repo(repo_root, str(_as_mapping(result["stderr"], "scontrol.stderr")["path"]))
        if "Invalid job id specified" not in stderr_path.read_text(encoding="utf-8"):
            raise ValueError(f"Round-11E scontrol recheck does not prove retention expiry for {job_id}")
    sacct = _as_mapping(payload.get("sacct"), "scheduler_recheck.sacct")
    if sacct.get("returncode") != 1:
        raise ValueError("Round-11E sacct recheck has unexpected return code")
    _validate_bound_stream(repo_root, _as_mapping(sacct.get("stdout"), "sacct.stdout"), "sacct_stdout")
    _validate_bound_stream(repo_root, _as_mapping(sacct.get("stderr"), "sacct.stderr"), "sacct_stderr")
    sacct_stderr_path = _path_under_repo(repo_root, str(_as_mapping(sacct["stderr"], "sacct.stderr")["path"]))
    if "Unable to connect to database" not in sacct_stderr_path.read_text(encoding="utf-8"):
        raise ValueError("Round-11E sacct recheck does not prove accounting unavailability")
    if payload.get("scontrol_available") is not False or payload.get("sacct_available") is not False:
        raise ValueError("Round-11E scheduler recheck availability flags are inconsistent")
    conclusion = _as_mapping(payload.get("conclusion"), "scheduler_recheck.conclusion")
    if conclusion.get("scontrol_status") != "expired_or_unavailable" or conclusion.get("sacct_status") != "unavailable":
        raise ValueError("Round-11E scheduler recheck conclusion is inconsistent")
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(path),
        "scontrol_available": False,
        "sacct_available": False,
        "scontrol_status": conclusion.get("scontrol_status"),
        "sacct_status": conclusion.get("sacct_status"),
        "terminal_evidence_scope": conclusion.get("terminal_evidence_scope"),
    }


def _require_exact_set(observed: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        unexpected = sorted(set(observed) - expected)
        raise ValueError(f"{label} testcase identity mismatch: missing={missing}, unexpected={unexpected}")


def _build_reproduction_observation(
    repo_root: Path,
    frozen_execution: Mapping[str, object],
    records: list[dict[str, Any]],
) -> dict[str, object]:
    """Bind preserved targeted results to the original 33 JUnit identities.

    This is deliberately an enrichment layer.  It never changes a field that
    describes job 490754; all later evidence is held under observed_* fields.
    """

    sources = {
        key: _artifact_observation(repo_root, filename, f"round11e_observation_{key}")
        for key, filename in OBSERVATION_SOURCE_FILENAMES.items()
    }
    wave_ab_path = _path_under_repo(repo_root, OBSERVATION_SOURCE_FILENAMES["wave_ab_junit"])
    rc003_path = _path_under_repo(repo_root, OBSERVATION_SOURCE_FILENAMES["rc003_junit"])
    focus_path = _path_under_repo(repo_root, OBSERVATION_SOURCE_FILENAMES["focus_junit"])
    wave_ab_counts, wave_ab_outcomes = _junit_outcomes(wave_ab_path)
    rc003_counts, rc003_outcomes = _junit_outcomes(rc003_path)
    focus_counts, focus_outcomes = _junit_outcomes(focus_path)
    if wave_ab_counts != {"tests": 31, "failures": 1, "errors": 0, "skipped": 0}:
        raise ValueError(f"Unexpected Wave A/B JUnit counts: {wave_ab_counts}")
    if rc003_counts != {"tests": 2, "failures": 2, "errors": 0, "skipped": 0}:
        raise ValueError(f"Unexpected RC-003 JUnit counts: {rc003_counts}")
    if focus_counts != {"tests": 43, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError(f"Unexpected focused gate JUnit counts: {focus_counts}")

    expected_wave_ab = {
        str(record["test_id"])
        for record in records
        if str(record["candidate_root_cause_uid"]) in {"R11E-RC-001", "R11E-RC-002"}
    }
    expected_rc003 = {
        str(record["test_id"])
        for record in records
        if str(record["candidate_root_cause_uid"]) == "R11E-RC-003"
    }
    _require_exact_set(wave_ab_outcomes, expected_wave_ab, "Wave A/B")
    _require_exact_set(rc003_outcomes, expected_rc003, "RC-003")
    if G0_INTEGRITY_TEST_ID not in wave_ab_outcomes:
        raise ValueError("Wave A/B JUnit does not contain the original G0 integrity testcase")

    wave_ab_provenance = _clean_worktree_identity(
        _path_under_repo(repo_root, OBSERVATION_SOURCE_FILENAMES["wave_ab_provenance"]), frozen_execution
    )
    rc003_provenance = _clean_worktree_identity(
        _path_under_repo(repo_root, OBSERVATION_SOURCE_FILENAMES["rc003_provenance"]), frozen_execution
    )
    focus_provenance = _clean_worktree_identity(
        _path_under_repo(repo_root, OBSERVATION_SOURCE_FILENAMES["focus_provenance"]), frozen_execution
    )
    for evidence, execution_job_id in (
        (wave_ab_provenance, "490950"),
        (rc003_provenance, "490953"),
        (focus_provenance, "490951"),
    ):
        capture_job_id = str(evidence.get("provenance_capture_job_id", ""))
        evidence["execution_job_id"] = execution_job_id
        evidence["provenance_capture_relation"] = (
            "job_coincident_clean_snapshot"
            if capture_job_id == execution_job_id
            else "prior_clean_snapshot_same_commit_and_config"
        )
        if capture_job_id != execution_job_id:
            evidence["provenance_limitations"] = [
                "The clean-worktree snapshot predates the targeted JUnit job; it proves the same commit/config identity but does not independently recapture dirty status at test launch."
            ]

    diagnostic = _load_object(_path_under_repo(repo_root, OBSERVATION_SOURCE_FILENAMES["rc003_diagnostic"]))
    diagnostic_state = _as_mapping(diagnostic.get("state"), "rc003 diagnostic.state")
    diagnostic_union = _as_mapping(diagnostic.get("union"), "rc003 diagnostic.union")
    if diagnostic_state.get("status") != "blocked" or diagnostic_state.get("state_mutated") is not False:
        raise ValueError("RC-003 diagnostic must prove a blocked, read-only controller state")
    if diagnostic_union.get("status") != "blocked":
        raise ValueError("RC-003 diagnostic must prove the contract union is blocked")
    blocker_codes = [str(_as_mapping(item, "rc003 diagnostic.blocker").get("code", "")) for item in _as_list(diagnostic_state.get("blockers"), "rc003 diagnostic.state.blockers")]
    required_blockers = {
        "orchestrator_lineage_mismatch",
        "unknown_selected_milestone",
        "unknown_selected_direction",
    }
    if not required_blockers.issubset(blocker_codes):
        raise ValueError("RC-003 diagnostic lacks the controller/contract lineage blockers")

    scheduler_recheck = _scheduler_recheck_evidence(repo_root)
    scheduler_recheck = {
        **scheduler_recheck,
        "attempted": True,
        "interpretation": "Targeted results are bound by preserved JUnit/stdout/provenance; no fresh scheduler terminal state is asserted.",
    }
    return {
        "schema_version": "dt-round11e-reproduction-observation-v1",
        "audit_kind": "round11e_targeted_reproduction_observation",
        "source_job_id": frozen_execution.get("job_id"),
        "observation_status": "complete_with_rc003_owner_decision_blocker",
        "frozen_job_mutated": False,
        "scheduler_recheck": scheduler_recheck,
        "artifacts": sources,
        "worktrees": {
            "wave_ab": wave_ab_provenance,
            "rc003": rc003_provenance,
            "focused_gate": focus_provenance,
        },
        "wave_ab": {
            "job_id": "490950",
            "junit_result": wave_ab_counts,
            "selected_root_causes": ["R11E-RC-001", "R11E-RC-002"],
            "selected_testcase_count": len(wave_ab_outcomes),
            "passed_testcase_count": sum(item["outcome"] == "passed" for item in wave_ab_outcomes.values()),
            "nonpassing_testcase_ids": [
                test_id for test_id, outcome in wave_ab_outcomes.items() if outcome["outcome"] != "passed"
            ],
            "g0_integrity_observation": "current assertion failure belongs to controller/contract lineage evidence, not the original missing-asset traceback family",
        },
        "rc003": {
            "job_id": "490953",
            "junit_result": rc003_counts,
            "selected_root_causes": ["R11E-RC-003"],
            "selected_testcase_count": len(rc003_outcomes),
            "stable_failure": True,
            "state_mutated": False,
            "root_cause": "controller state and active contract-union lineage mismatch; selected milestone and direction are absent from the active roadmap",
            "owner_decision_required": True,
            "blocker_codes": blocker_codes,
        },
        "focused_gate": {
            "job_id": "490951",
            "junit_result": focus_counts,
            "validator_result": "passed blockers=0",
            "scope": "config-stack, schema-owner, state-block/mask, and Round-11 state-audit controls",
        },
        "observation_policy": "Later observations enrich the Round-11E ledger without rewriting frozen job 490754 classifications, messages, stack hashes, or testcase identities.",
    }


def _enrich_with_observation(
    repo_root: Path, records: list[dict[str, Any]], observation: Mapping[str, object]
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    """Return observed ledger/reproduction rows while preserving all frozen fields."""

    artifacts = _as_mapping(observation.get("artifacts"), "observation.artifacts")
    worktrees = _as_mapping(observation.get("worktrees"), "observation.worktrees")
    wave_ab = _as_mapping(observation.get("wave_ab"), "observation.wave_ab")
    rc003 = _as_mapping(observation.get("rc003"), "observation.rc003")
    wave_ab_junit = _as_mapping(artifacts.get("wave_ab_junit"), "observation.artifacts.wave_ab_junit")
    rc003_junit = _as_mapping(artifacts.get("rc003_junit"), "observation.artifacts.rc003_junit")
    wave_ab_stdout = _as_mapping(artifacts.get("wave_ab_stdout"), "observation.artifacts.wave_ab_stdout")
    wave_ab_stderr = _as_mapping(artifacts.get("wave_ab_stderr"), "observation.artifacts.wave_ab_stderr")
    rc003_stdout = _as_mapping(artifacts.get("rc003_stdout"), "observation.artifacts.rc003_stdout")
    rc003_stderr = _as_mapping(artifacts.get("rc003_stderr"), "observation.artifacts.rc003_stderr")
    wave_ab_counts, wave_ab_outcomes = _junit_outcomes(
        _path_under_repo(repo_root, str(wave_ab_junit["path"]))
    )
    rc003_counts, rc003_outcomes = _junit_outcomes(
        _path_under_repo(repo_root, str(rc003_junit["path"]))
    )
    if wave_ab_counts != _as_mapping(wave_ab.get("junit_result"), "observation.wave_ab.junit_result"):
        raise ValueError("Wave A/B observation JUnit result is inconsistent with its bound artifact")
    if rc003_counts != _as_mapping(rc003.get("junit_result"), "observation.rc003.junit_result"):
        raise ValueError("RC-003 observation JUnit result is inconsistent with its bound artifact")

    enriched_ledger: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, object]] = []
    for record in records:
        row = dict(record)
        root_cause_uid = str(record["candidate_root_cause_uid"])
        test_id = str(record["test_id"])
        node_id = f"{record['test_file']}::{test_id.rsplit('::', maxsplit=1)[-1]}"
        if root_cause_uid in {"R11E-RC-001", "R11E-RC-002"}:
            outcome = wave_ab_outcomes.get(test_id)
            if outcome is None:
                raise ValueError(f"Wave A/B observation omitted frozen testcase: {test_id}")
            if test_id == G0_INTEGRITY_TEST_ID:
                status = "reproduced"
                observed_root_cause = "R11E-RC-003"
                observed_family = "assertion_only"
                repair_status = "blocked_pending_owner_decision"
                root_cause_status = "strongly_supported"
                notes = (
                    "Original frozen FileNotFoundError classification is preserved. Under the hash-bound clean-input linkage, "
                    "the testcase instead failed its controller/contract-union assertion; current observed cause is RC-003."
                )
                stable_failure = "observed_once_reassigned_to_confirmed_rc003"
            elif outcome["outcome"] == "passed":
                status = "not_reproduced"
                observed_root_cause = root_cause_uid
                observed_family = str(record["failure_family"])
                repair_status = "repair_verified_read_only_input_linkage"
                root_cause_status = "strongly_supported"
                notes = (
                    "Original missing-input non-pass did not reproduce after exact commit/config clean-worktree execution "
                    "with the hash-bound, read-only generated-input and audit-artifact linkage."
                )
                stable_failure = "no"
            else:
                raise ValueError(f"Unexpected Wave A/B outcome for {test_id}: {outcome['outcome']}")
            worktree = _as_mapping(worktrees.get("wave_ab"), "observation.worktrees.wave_ab")
            observation_job_id = str(wave_ab.get("job_id"))
            junit_artifact = wave_ab_junit
            stdout_artifact = wave_ab_stdout
            stderr_artifact = wave_ab_stderr
            repair_wave = "Wave D" if test_id == G0_INTEGRITY_TEST_ID else "Wave B"
        elif root_cause_uid == "R11E-RC-003":
            outcome = rc003_outcomes.get(test_id)
            if outcome is None or outcome["outcome"] not in {"failure", "error"}:
                raise ValueError(f"RC-003 observation did not stably reproduce {test_id}")
            status = "reproduced"
            observed_root_cause = "R11E-RC-003"
            observed_family = "assertion_only"
            repair_status = "blocked_pending_owner_decision"
            root_cause_status = "strongly_supported"
            notes = (
                "Stable clean-worktree reproduction: controller state rev-005 disagrees with active union rev-003, "
                "and the selected milestone/direction are absent from the active roadmap. Diagnostic proved state_mutated=false."
            )
            stable_failure = "yes"
            worktree = _as_mapping(worktrees.get("rc003"), "observation.worktrees.rc003")
            observation_job_id = str(rc003.get("job_id"))
            junit_artifact = rc003_junit
            stdout_artifact = rc003_stdout
            stderr_artifact = rc003_stderr
            repair_wave = "Wave D"
        else:
            raise ValueError(f"No observation policy exists for root cause {root_cause_uid}")

        row.update(
            {
                "reproduction_status": status,
                "root_cause_status": root_cause_status,
                "repair_status": repair_status,
                "observed_root_cause_uid": observed_root_cause,
                "observed_failure_family": observed_family,
                "observation_job_id": observation_job_id,
                "observation_junit_path": str(junit_artifact["path"]),
                "observation_junit_sha256": str(junit_artifact["sha256"]),
                "reproduction_relation": str(worktree["reproduction_relation"]),
                "repair_wave": repair_wave,
            }
        )
        enriched_ledger.append(row)
        reproduction_rows.append(
            {
                "testcase_uid": record["testcase_uid"],
                "test_id": test_id,
                "test_file": record["test_file"],
                "failure_or_error": record["failure_or_error"],
                "failure_family": record["failure_family"],
                "candidate_root_cause_uid": root_cause_uid,
                "source_job_id": observation.get("source_job_id"),
                "reproduction_command": f"python -m pytest {shlex.quote(node_id)} -q",
                "reproduction_status": status,
                "clean_worktree_identity": f"{worktree['git_commit']}:{worktree['dirty_status']}",
                "stdout_path": str(stdout_artifact["path"]),
                "stderr_path": str(stderr_artifact["path"]),
                "stable_failure_observed": stable_failure,
                "notes": notes,
                "observed_root_cause_uid": observed_root_cause,
                "observation_job_id": observation_job_id,
                "observation_junit_path": str(junit_artifact["path"]),
                "observation_junit_sha256": str(junit_artifact["sha256"]),
                "reproduction_relation": str(worktree["reproduction_relation"]),
                "repair_wave": repair_wave,
            }
        )
    return enriched_ledger, reproduction_rows


def _enrich_family_summary(
    base_summary: Mapping[str, object], ledger: list[dict[str, Any]]
) -> dict[str, object]:
    """Keep frozen families intact while reporting later observed classifications."""

    summary = dict(base_summary)
    observed_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        observed_rows[str(row["observed_failure_family"])].append(row)
    observed_counts = {family: len(observed_rows[family]) for family in FAILURE_FAMILIES}
    summary.update(
        {
            "status": "reproduction_observation_complete_with_rc003_owner_decision_blocker",
            "frozen_family_classification_preserved": True,
            "observed_category_counts": observed_counts,
            "observed_families": [
                {
                    "family": family,
                    "testcase_count": len(rows),
                    "reproduced_count": sum(row["reproduction_status"] == "reproduced" for row in rows),
                    "not_reproduced_count": sum(row["reproduction_status"] == "not_reproduced" for row in rows),
                    "testcase_uids": [str(row["testcase_uid"]) for row in rows],
                }
                for family, rows in ((family, observed_rows[family]) for family in FAILURE_FAMILIES)
                if rows
            ],
            "observation_reassignment": {
                "testcase_id": G0_INTEGRITY_TEST_ID,
                "frozen_failure_family": "missing_asset",
                "frozen_candidate_root_cause_uid": "R11E-RC-001",
                "observed_failure_family": "assertion_only",
                "observed_root_cause_uid": "R11E-RC-003",
                "reason": "The original missing-asset traceback is preserved, but the clean linked execution reached and failed the controller/contract-union assertion.",
            },
        }
    )
    return summary


def _observed_root_cause_graph(ledger: list[dict[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        grouped[str(row["observed_root_cause_uid"])].append(row)
    specifications = {
        "R11E-RC-001": {
            "root_cause": "frozen sparse-worktree generated-input unavailability did not reproduce after a hash-bound, read-only clean-worktree input linkage",
            "shared_missing_resource": "frozen-job-only generated Input/processed unavailability; current linkage supplied only the observed allowlist",
            "shared_config_dependency": "repository-relative configured Input/processed paths",
            "confidence": "strongly_supported",
            "repair_status": "verified_read_only_input_linkage",
        },
        "R11E-RC-002": {
            "root_cause": "the hash-bound formal-experiment-readiness linkage did not reproduce after the approved read-only artifact binding was made available",
            "shared_missing_resource": "frozen-job-only absence of Output/audits/formal_experiment_readiness_audit.json",
            "shared_config_dependency": "iclr_design_contract_union.g0_read_only_artifact_linkage and NRNSIFTER_AUDIT_ARTIFACT_ROOT",
            "confidence": "strongly_supported",
            "repair_status": "verified_read_only_input_linkage",
        },
        "R11E-RC-003": {
            "root_cause": "controller state rev-005 is inconsistent with active contract-union rev-003, while the selected milestone and direction are absent from the active roadmap",
            "shared_missing_resource": None,
            "shared_config_dependency": "orchestrator/state.json controller lineage and iclr_design_contract_union active roadmap lineage",
            "confidence": "confirmed",
            "repair_status": "blocked_pending_owner_decision",
        },
    }
    candidates: list[dict[str, object]] = []
    for uid in sorted(grouped):
        rows = grouped[uid]
        specification = specifications.get(uid)
        if specification is None:
            raise ValueError(f"Unexpected observed root cause UID: {uid}")
        candidates.append(
            {
                "observed_root_cause_uid": uid,
                **specification,
                "supporting_testcases": [
                    {
                        "testcase_uid": row["testcase_uid"],
                        "test_id": row["test_id"],
                        "frozen_candidate_root_cause_uid": row["candidate_root_cause_uid"],
                        "reproduction_status": row["reproduction_status"],
                    }
                    for row in rows
                ],
                "affected_fixtures_or_modules": sorted({str(row["first_project_frame"]) for row in rows}),
                "reproduced_count": sum(row["reproduction_status"] == "reproduced" for row in rows),
                "not_reproduced_count": sum(row["reproduction_status"] == "not_reproduced" for row in rows),
            }
        )
    return candidates


def _enrich_dependency_graph(
    base_graph: Mapping[str, object], ledger: list[dict[str, Any]]
) -> dict[str, object]:
    graph = dict(base_graph)
    graph.update(
        {
            "status": "reproduction_observation_complete_with_rc003_owner_decision_blocker",
            "frozen_candidate_root_cause_graph_preserved": True,
            "observed_root_cause_graph": _observed_root_cause_graph(ledger),
            "observed_reassignment": {
                "testcase_id": G0_INTEGRITY_TEST_ID,
                "frozen_candidate_root_cause_uid": "R11E-RC-001",
                "observed_root_cause_uid": "R11E-RC-003",
                "evidence": "Wave A/B linked JUnit reached a controller/contract assertion, and the RC-003 diagnostic exposed the shared lineage blockers without mutating controller state.",
            },
            "terminal_reproduction_status_count": Counter(str(row["reproduction_status"]) for row in ledger),
            "unresolved_repair_testcase_count": sum(
                row["repair_status"] == "blocked_pending_owner_decision" for row in ledger
            ),
            "repair_authorization": "The frozen candidate graph was initialized before repair; the observed Wave B linkage repair is documented separately and RC-003 remains blocked pending an owner decision.",
            "full_suite_rerun_permitted": False,
            "full_suite_rerun_blocker": "RC-003 has a confirmed controller/contract lineage cause but no owner-authorized repair.",
        }
    )
    return graph


def _targeted_gate(
    observation: Mapping[str, object], ledger: list[dict[str, Any]]
) -> dict[str, object]:
    """Record per-wave gates without treating the unresolved controller state as repaired."""

    focus = _as_mapping(observation.get("focused_gate"), "observation.focused_gate")
    wave_ab = _as_mapping(observation.get("wave_ab"), "observation.wave_ab")
    rc003 = _as_mapping(observation.get("rc003"), "observation.rc003")
    wave_b_rows = [
        row
        for row in ledger
        if row["observed_root_cause_uid"] in {"R11E-RC-001", "R11E-RC-002"}
    ]
    wave_d_rows = [row for row in ledger if row["observed_root_cause_uid"] == "R11E-RC-003"]
    if len(wave_b_rows) != 30 or any(row["reproduction_status"] != "not_reproduced" for row in wave_b_rows):
        raise ValueError("Wave B observation must contain 30 non-reproducing input/linkage cases")
    if len(wave_d_rows) != 3 or any(row["reproduction_status"] != "reproduced" for row in wave_d_rows):
        raise ValueError("Wave D observation must contain three stable RC-003 failures")
    focus_counts = _as_mapping(focus.get("junit_result"), "observation.focused_gate.junit_result")
    if focus_counts != {"tests": 43, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError("Focused config/schema/state gate did not pass")
    return {
        "schema_version": "dt-round11e-targeted-gate-v1",
        "audit_kind": "round11e_targeted_gate",
        "source_job_id": observation.get("source_job_id"),
        "status": "blocked_pending_owner_decision",
        "waves": [
            {
                "wave": "Wave A",
                "scope": "collection/import/environment/setup",
                "status": "not_applicable_no_observed_testcases",
                "previously_failing_targeted_tests": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0},
            },
            {
                "wave": "Wave B",
                "scope": "config/schema/path/provenance",
                "status": "passed",
                "observed_root_cause_uids": ["R11E-RC-001", "R11E-RC-002"],
                "previously_failing_targeted_tests": {
                    "tests": len(wave_b_rows),
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                },
                "source_junit_result_including_reassigned_g0": wave_ab.get("junit_result"),
                "reassigned_to_wave_d_testcase_id": G0_INTEGRITY_TEST_ID,
                "entry_to_next_wave_allowed": True,
            },
            {
                "wave": "Wave C",
                "scope": "data/model/tool contract",
                "status": "not_applicable_no_observed_testcases",
                "previously_failing_targeted_tests": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0},
            },
            {
                "wave": "Wave D",
                "scope": "assertion and integration behavior",
                "status": "blocked_pending_owner_decision",
                "observed_root_cause_uids": ["R11E-RC-003"],
                "previously_failing_targeted_tests": {
                    "tests": len(wave_d_rows),
                    "failures": len(wave_d_rows),
                    "errors": 0,
                    "skipped": 0,
                },
                "source_junit_result_for_original_rc003": rc003.get("junit_result"),
                "reassigned_g0_testcase_id": G0_INTEGRITY_TEST_ID,
                "entry_to_full_suite_allowed": False,
                "blocker": "No owner-authorized controller/contract lineage decision exists; state, active union, roadmap, schema registry, and tests were not changed.",
            },
        ],
        "cross_wave_controls": {
            "config_stack_schema_owner_state_block_mask_round11_focus": {
                "job_id": focus.get("job_id"),
                "junit_result": focus_counts,
                "validator_result": focus.get("validator_result"),
                "status": "passed",
            }
        },
        "terminal_reproduction_counts": {
            "reproduced": sum(row["reproduction_status"] == "reproduced" for row in ledger),
            "not_reproduced": sum(row["reproduction_status"] == "not_reproduced" for row in ledger),
            "environment_specific": 0,
            "intermittent": 0,
            "blocked_missing_evidence": 0,
        },
        "full_suite_submission_permitted": False,
        "full_suite_submission_blocker": "RC-003 remains unrepaired pending an owner decision.",
    }


def _root_wave_b_delivery(repo_root: Path) -> dict[str, object]:
    """Bind the root source migration without recasting it as a reproduction."""

    files = []
    for relative_path, expected_sha256 in WAVE_B_CLEAN_REPAIR_FILE_SHA256.items():
        path = repo_root / relative_path
        if not path.is_file():
            raise ValueError(f"Round-11E root Wave B delivery file is missing: {relative_path}")
        observed_sha256 = sha256_file(path)
        current_sha256 = WAVE_B_CURRENT_SOURCE_FILE_SHA256[relative_path]
        if observed_sha256 not in {expected_sha256, current_sha256}:
            raise ValueError(
                "Round-11E root Wave B delivery differs from both the frozen clean repair and current source snapshot: "
                f"{relative_path}"
            )
        content_identity = (
            "exact_sha256_match"
            if observed_sha256 == expected_sha256
            else "current_source_snapshot_match"
        )
        files.append(
            {
                "path": relative_path,
                "root_sha256": observed_sha256,
                "clean_repair_sha256": expected_sha256,
                "current_source_sha256": current_sha256,
                "content_identity": content_identity,
            }
        )

    junit_path = _path_under_repo(repo_root, ROOT_WAVE_B_UNIT_JUNIT_FILENAME)
    junit_result, outcomes = _junit_outcomes(junit_path)
    expected_result = {"tests": 4, "failures": 0, "errors": 0, "skipped": 0}
    if junit_result != expected_result or any(item["outcome"] != "passed" for item in outcomes.values()):
        raise ValueError(f"Round-11E root Wave B unit result is not a clean pass: {junit_result}")
    return {
        "status": (
            "migrated_source_verified"
            if all(row["content_identity"] == "exact_sha256_match" for row in files)
            else "migrated_source_current_snapshot_bound"
        ),
        "scope": "root source migration and local unit contract only; not clean-worktree testcase reproduction evidence",
        "root_worktree_cleanliness": "not_asserted_not_used_for_reproduction",
        "files": files,
        "unit_junit": _artifact_binding(repo_root, junit_path, "round11e_root_wave_b_unit_junit"),
        "unit_junit_result": junit_result,
        "unit_testcase_ids": sorted(outcomes),
    }


def _repair_wave_record(repo_root: Path, observation: Mapping[str, object]) -> dict[str, object]:
    """Document what was repaired and why no controller repair has been attempted."""

    return {
        "schema_version": "dt-round11e-repair-wave-record-v1",
        "audit_kind": "round11e_repair_wave_record",
        "source_job_id": observation.get("source_job_id"),
        "waves": [
            {
                "wave": "Wave B",
                "root_cause_uids": ["R11E-RC-001", "R11E-RC-002"],
                "files_changed": [
                    "Libs/utils/data/round11e_processed_input_linkage.py",
                    "Libs/scripts/builders/build_dt_round11e_processed_input_manifest.py",
                    "Libs/scripts/validators/run_dt_round11e_processed_input_linkage.py",
                    "Libs/tests/test_dt_round11e_processed_input_linkage.py",
                ],
                "change_location": "orchestrator/worktrees/round-11e-clean-repair (clean-worktree targeted repair evidence) and root worktree (content-pinned source migration only)",
                "root_worktree_delivery": _root_wave_b_delivery(repo_root),
                "contract_affected": "explicit hash-bound generated-input and parent-audit linkage visibility in a clean sparse worktree",
                "why_prior_behavior_was_incorrect": "The frozen full-suite worktree could not see required generated inputs or the approved read-only parent audit linkage, causing setup and test-body failures before their intended contracts ran.",
                "why_repair_preserves_fail_closed_behavior": "The linkage requires explicit environment variables, a content-addressed allowlist, per-file SHA-256/size verification, no symlinked inputs, a process-local read-only namespace, and hidden unlisted source paths. Missing, stale, or unapproved inputs still fail loudly.",
                "tests_added_or_updated": [
                    "Libs/tests/test_dt_round11e_processed_input_linkage.py",
                    "Output/audits/dt_round11e_final_namespace_smoke.stdout",
                    "Output/audits/dt_round11e_final_wave_ab_results.xml",
                ],
                "rollback": "Remove the explicit per-process linkage environment and namespace launcher; no generated input was copied into the clean worktree and no frozen job artifact was overwritten.",
                "status": "verified_for_30_observed_testcases",
            },
            {
                "wave": "Wave D",
                "root_cause_uids": ["R11E-RC-003"],
                "files_changed": [],
                "contract_affected": "controller state / active contract-union / active roadmap lineage",
                "why_prior_behavior_was_incorrect": "No implementation conclusion is authorized: the diagnostic establishes a lineage mismatch but does not select which controller, union, or roadmap record is canonical.",
                "why_repair_preserves_fail_closed_behavior": "No controller, union, roadmap, schema registry, or test expectation was altered. The tests remain failing and the full suite remains blocked.",
                "tests_added_or_updated": [],
                "rollback": "No repair was performed; no rollback action is applicable.",
                "status": "blocked_pending_owner_decision",
            },
        ],
        "prohibited_actions_not_taken": [
            "test deletion",
            "mandatory test skip",
            "broad xfail",
            "schema requirement reduction",
            "silent fallback",
            "controller state mutation",
            "active contract-union mutation",
            "roadmap mutation",
        ],
    }


def validate_round11e_triage(
    repo_root: Path,
    audit_root: Path,
    execution_path: Path,
    junit_path: Path,
    failure_triage_path: Path,
    closure_path: Path,
) -> dict[str, object]:
    """Verify generated analysis records are an exact, fail-closed view of frozen inputs."""

    blockers: list[dict[str, str]] = []
    try:
        expected = build_round11e_triage(repo_root, execution_path, junit_path, failure_triage_path, closure_path)
    except Exception as exc:  # A frozen-input inconsistency is a validation failure.
        return {"status": "failed", "blocker_count": 1, "blockers": [{"code": "frozen_input_invalid", "message": str(exc)}]}

    expected_values: dict[str, object] = {
        "input_snapshot": expected["input_snapshot"],
        "ledger": _serialize_rows(_as_list(expected["ledger"], "expected.ledger"), LEDGER_FIELDS),
        "family_summary": expected["family_summary"],
        "dependency_graph": expected["dependency_graph"],
        "targeted_reproduction": _serialize_rows(
            _as_list(expected["targeted_reproduction"], "expected.targeted_reproduction"), REPRODUCTION_FIELDS
        ),
    }
    for key, filename in AUDIT_FILENAMES.items():
        path = audit_root / filename
        if not path.is_file():
            blockers.append({"code": "missing_required_audit", "message": filename})
            continue
        actual: object
        if path.suffix == ".tsv":
            fields = LEDGER_FIELDS if key == "ledger" else REPRODUCTION_FIELDS
            actual = _project_tsv_rows(_read_tsv(path), fields)
        else:
            actual = _load_object(path)
        if actual != expected_values[key]:
            blockers.append({"code": "stale_or_tampered_audit", "message": filename})

    # The frozen JUnit projection is checked independently of all later
    # reproduction state, so later observation enrichment cannot rewrite 490754.
    ledger_path = audit_root / AUDIT_FILENAMES["ledger"]
    reproduction_path = audit_root / AUDIT_FILENAMES["targeted_reproduction"]
    if ledger_path.is_file():
        actual_ledger_frozen = _project_tsv_rows(_read_tsv(ledger_path), FROZEN_LEDGER_FIELDS)
        expected_ledger_frozen = _serialize_rows(
            _as_list(expected["ledger"], "expected.ledger"), FROZEN_LEDGER_FIELDS
        )
        if actual_ledger_frozen != expected_ledger_frozen:
            blockers.append({"code": "frozen_ledger_fields_changed", "message": AUDIT_FILENAMES["ledger"]})
    if reproduction_path.is_file():
        actual_reproduction_frozen = _project_tsv_rows(_read_tsv(reproduction_path), FROZEN_REPRODUCTION_FIELDS)
        expected_reproduction_frozen = _serialize_rows(
            _as_list(expected["targeted_reproduction"], "expected.targeted_reproduction"),
            FROZEN_REPRODUCTION_FIELDS,
        )
        if actual_reproduction_frozen != expected_reproduction_frozen:
            blockers.append(
                {"code": "frozen_reproduction_fields_changed", "message": AUDIT_FILENAMES["targeted_reproduction"]}
            )

    for key, filename in OBSERVED_AUDIT_FILENAMES.items():
        path = audit_root / filename
        if not path.is_file():
            blockers.append({"code": "missing_required_observation_audit", "message": filename})
            continue
        if _load_object(path) != expected[key]:
            blockers.append({"code": "stale_or_tampered_observation_audit", "message": filename})

    ledger = _as_list(expected["ledger"], "expected.ledger")
    summary = _as_mapping(expected["family_summary"], "expected.family_summary")
    graph = _as_mapping(expected["dependency_graph"], "expected.dependency_graph")
    if len(ledger) != 33:
        blockers.append({"code": "unexpected_nonpass_count", "message": f"expected 33 non-passing testcases, found {len(ledger)}"})
    if {str(row["reproduction_status"]) for row in ledger} - set(REPRODUCTION_STATUSES):
        blockers.append({"code": "invalid_reproduction_status", "message": "Ledger contains an unsupported reproduction status."})
    if {str(row["failure_family"]) for row in ledger} - set(FAILURE_FAMILIES):
        blockers.append({"code": "invalid_failure_family", "message": "Ledger contains an unsupported failure family."})
    candidate_entries = _as_list(graph.get("root_cause_graph"), "dependency_graph.root_cause_graph")
    if {str(_as_mapping(item, "candidate").get("confidence", "")) for item in candidate_entries} - set(ROOT_CAUSE_CONFIDENCES):
        blockers.append({"code": "invalid_root_cause_confidence", "message": "Dependency graph contains an unsupported confidence."})
    category_counts = _as_mapping(summary.get("category_counts"), "family_summary.category_counts")
    if sum(int(category_counts.get(family, 0)) for family in FAILURE_FAMILIES) != len(ledger):
        blockers.append({"code": "family_count_mismatch", "message": "Failure-family counts do not sum to the JUnit non-pass count."})
    terminal_counts = Counter(str(row["reproduction_status"]) for row in ledger)
    if terminal_counts != Counter({"not_reproduced": 30, "reproduced": 3}):
        blockers.append(
            {
                "code": "unexpected_reproduction_terminal_state",
                "message": f"expected 30 not_reproduced and 3 reproduced, observed {dict(terminal_counts)}",
            }
        )
    if any(str(row["repair_status"]) == "not_attempted_pending_clean_reproduction" for row in ledger):
        blockers.append({"code": "missing_terminal_reproduction_state", "message": "A ledger row lacks an observed terminal state."})
    targeted_gate = _as_mapping(expected.get("targeted_gate"), "expected.targeted_gate")
    if targeted_gate.get("status") != "blocked_pending_owner_decision":
        blockers.append({"code": "invalid_targeted_gate_status", "message": str(targeted_gate.get("status"))})
    if targeted_gate.get("full_suite_submission_permitted") is not False:
        blockers.append({"code": "full_suite_gate_opened", "message": "RC-003 must keep full-suite submission closed."})
    return {
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "source_job_id": _as_mapping(expected["input_snapshot"], "expected.input_snapshot").get("source_job_id"),
        "testcase_count": len(ledger),
        "failure_count": sum(str(row["failure_or_error"]) == "failure" for row in ledger),
        "error_count": sum(str(row["failure_or_error"]) == "error" for row in ledger),
        "reproduction_execution_performed": True,
        "frozen_ledger_fields_validated": ledger_path.is_file(),
        "frozen_reproduction_fields_validated": reproduction_path.is_file(),
        "full_suite_submission_permitted": False,
    }
