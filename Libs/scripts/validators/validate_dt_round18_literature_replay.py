"""Validate Round-18 literature replay evidence and issue the canonical Gate.

The validator is deliberately an evaluator-side component.  It validates
materialized artifacts and their hashes rather than recreating a successful
replay from configuration.  A blocked assessment is not a canonical Gate;
only the orchestration controller may write an accepted Gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_dt_round18_literature_replay import (  # noqa: E402
    FINAL_HELD_OUT_PARTITION,
    ROUND18_CONFIG_VERSION,
    ROUND18_VALIDATOR_UID,
    SYSTEM_UIDS,
    _output_paths as builder_output_paths,
    _outcome_record,
    _source_record,
    _validate_config as builder_config_blockers,
    round18_config,
    sha256_file,
    validate_parent_state,
)
from Libs.utils.data.literature_replay_contracts import (  # noqa: E402
    CLAIM_SCOPE,
    GOVERNANCE_COUNT_FIELDS,
    SCHEMA_VERSIONS,
    TRACK_A,
    TRACK_B,
    TRACK_C,
    TRACK_D,
    TRACK_E,
    TRACK_F,
    aggregate_governance_counts,
    canonical_sha256,
    compute_direction_metrics,
    compute_ranking_metrics,
    compute_reliability_metrics,
    sorted_blockers,
    validate_abstention_semantics,
    validate_bundle_isolation,
    validate_claim_boundaries,
    validate_execution_isolation,
    validate_failed_retry_preservation,
    validate_frozen_gap_isolation,
    validate_independence_groups,
    validate_label_provenance,
    validate_lineage_acyclic,
    validate_memory_risk_disclosure,
    validate_public_source,
    validate_public_source_boundary,
    validate_scitil_runtime_trace,
    validate_sequential_reveal,
    validate_split_leakage,
    validate_split_manifest,
    validate_temporal_cutoff,
    validate_temporal_ordering,
    validate_unique_uids,
    validate_episode,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    load_config,
    read_tsv,
    write_json as registry_write_json,
)
from Libs.utils.exp.orchestration_controller import (  # noqa: E402
    CANONICAL_CONTROLLER_UID,
    authority_contract_report,
    build_canonical_gate_decision,
    git_commit,
    git_dirty_status,
    load_authority_contract,
    validator_is_approved,
)


AUTHORITY_PATH = "Libs/configs/validation/round11f_controller_authority.yaml"
ROUND18_CONFIG_PATH = "Libs/configs/data/round18_literature_replay.yaml"
VALIDATION_PATH = "Output/audits/dt_round18_validation.json"
GATE_PATH = "Output/audits/dt_round18_gate.json"
HANDOFF_PATH = "Output/audits/dt_round19_handoff.json"
PREFULL_PATH = "Output/audits/dt_round18_prefull_validation.json"

VALIDATOR_UID = ROUND18_VALIDATOR_UID
STAGE_UID = "round_18_literature_held_out_sequential_replay"
ACCEPTED_STATE = "literature_held_out_sequential_replay_resolved"
ROUND18_PARENT_TEST_COUNT = 557
ROUND18_FROZEN_GAP_COUNT = 20
ROUND18_CPU_NODES = frozenset({"cpu03", "cpu04"})
ROUND18_CPU_PLACEMENTS = frozenset({("cpu03", "gpu"), ("cpu04", "cpu")})
ROUND18_JOB_PREFIX = "NrnSift_DT_R18_"

EXECUTION_IDENTITY_SCHEMA_VERSION = "neuronsifter-dt-round18-execution-identity-v1"
PREFULL_SCHEMA_VERSION = "neuronsifter-dt-round18-prefull-validation-v1"
FULL_SUITE_EXECUTION_SCHEMA_VERSION = "neuronsifter-dt-round18-full-suite-execution-v1"
TEST_INVENTORY_SCHEMA_VERSION = "neuronsifter-dt-round18-test-inventory-diff-v1"
COMMIT_MANIFEST_SCHEMA_VERSION = "neuronsifter-dt-round18-commit-manifest-v1"
VALIDATION_SCHEMA_VERSION = "neuronsifter-dt-round18-validation-v1"

BLOCKER_STATES = {
    "blocked_public_source_boundary",
    "blocked_temporal_leakage",
    "blocked_held_out_isolation",
    "blocked_label_provenance",
    "blocked_independence_grouping",
    "blocked_memory_risk_disclosure",
    "blocked_replay_integrity",
    "blocked_metric_validation",
    "blocked_abstention_semantics",
    "blocked_claim_boundary",
    "blocked_gap_fabrication",
    "blocked_execution_verification",
}
ALLOWED_GATE_STATES = (ACCEPTED_STATE, *tuple(sorted(BLOCKER_STATES)))

REQUIRED_RELEASE_ARTIFACTS = (
    "episode_catalog",
    "pre_cutoff_evidence_manifest",
    "held_out_outcome_manifest",
    "label_catalog",
    "independence_group_catalog",
    "split_manifest",
    "metric_specification",
    "system_run_manifest",
    "result_catalog",
)
REQUIRED_PILOT_CATALOGS = (
    "prediction_catalog",
    "ranking_catalog",
    "decision_catalog",
    "abstention_catalog",
    "leakage_control_catalog",
)

MANDATORY_NEGATIVE_TEST_NAMES = frozenset(
    {
        "test_round18_section23_mandatory_negative_inventory_is_complete",
        "test_controlled_source_route_fails_closed",
        "test_future_pre_cutoff_evidence_fails_closed",
        "test_bundle_and_execution_isolation_fail_closed",
        "test_ambiguous_timestamp_strong_admission_fails_closed",
        "test_narrative_numeric_reconstruction_is_rejected",
        "test_split_leakage_and_duplicate_accession_fail_closed",
        "test_unknown_comparator_and_action_identity_fail_closed",
        "test_claim_language_and_frozen_gap_boundaries_fail_closed",
        "test_abstention_forced_answer_is_rejected",
        "test_unknown_uncertainty_zero_fails_closed",
        "test_conflict_removal_is_rejected_by_abstention_schema",
        "test_memory_risk_cannot_claim_contamination_free",
        "test_lineage_and_retry_preserve_failed_attempts",
    }
)


def _sha256(path: Path) -> str:
    """Backward-compatible file hash helper."""

    return sha256_file(path)


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _load_json(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _decode_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _decode_parquet_row(row: Mapping[str, object]) -> dict[str, object]:
    """Backward-compatible Parquet row decoder."""

    return {str(key): _decode_value(value) for key, value in row.items()}


def _read_parquet(
    path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not path.is_file():
        return [], [
            _block(
                "release_artifact_missing",
                "Required Parquet artifact is missing.",
                path=str(path),
            )
        ]
    try:
        rows = [_decode_parquet_row(row) for row in pq.read_table(path).to_pylist()]
    except Exception as error:  # pyarrow format failures have multiple exception types
        return [], [_block("release_artifact_unreadable", str(error), path=str(path))]
    if not rows:
        return [], [
            _block(
                "release_artifact_empty",
                "Required Parquet artifact is empty.",
                path=str(path),
            )
        ]
    return rows, []


def _read_tsv(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not path.is_file():
        return [], [
            _block(
                "pilot_catalog_missing",
                "Required pilot catalog is missing.",
                path=str(path),
            )
        ]
    try:
        rows = [
            {str(key): _decode_value(value) for key, value in row.items()}
            for row in read_tsv(path)
        ]
    except (OSError, ValueError, TypeError) as error:
        return [], [_block("pilot_catalog_unreadable", str(error), path=str(path))]
    if not rows:
        return [], [
            _block(
                "pilot_catalog_empty",
                "Required pilot catalog is empty.",
                path=str(path),
            )
        ]
    return rows, []


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _git_tracked(repository_root: Path, path: Path) -> bool:
    if not _path_is_within(path, repository_root):
        return False
    relative = path.resolve().relative_to(repository_root.resolve()).as_posix()
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _junit_summary(path: Path | None) -> dict[str, object]:
    empty = {
        "available": False,
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "testcase_names": [],
    }
    if path is None or not path.is_file():
        return empty
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError, ValueError):
        return empty
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "available": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "tests": sum(int(suite.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.get("skipped", 0)) for suite in suites),
        "testcase_names": sorted(
            str(case.get("name", "")) for case in root.iter("testcase")
        ),
    }


def expected_execution_identity(repository_root: Path) -> dict[str, object]:
    """Return the dynamic fields an exact-clean execution must bind."""

    return {
        "schema_version": EXECUTION_IDENTITY_SCHEMA_VERSION,
        "exact_git_commit": _git(repository_root, "rev-parse", "HEAD"),
        "tree_hash": _git(repository_root, "rev-parse", "HEAD^{tree}"),
        "dirty_status": _git(repository_root, "status", "--porcelain=v1"),
        "source_read_only": True,
        "isolated_writable_runtime_output": True,
        "python_pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX", ""),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "process_uid": os.getuid(),
        "process_gid": os.getgid(),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _block(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def _cpu_only_placement_valid(record: Mapping[str, object]) -> bool:
    """Permit the authorized CPU nodes while proving no GPU allocation occurred."""

    node = str(record.get("node", ""))
    partition = str(record.get("partition", ""))
    allocation_count = record.get("gpu_allocation_count")
    return (
        node in ROUND18_CPU_NODES
        and (node, partition) in ROUND18_CPU_PLACEMENTS
        and isinstance(allocation_count, int)
        and not isinstance(allocation_count, bool)
        and allocation_count == 0
    )


def _load_config(config_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    config = load_config(config_path)
    return config, round18_config(config)


def _source_and_outcomes(
    cfg: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Build configuration records for legacy callers; validation uses artifacts."""

    policy = _mapping(_mapping(cfg.get("policies")).get("temporal_cutoff"))
    sources = {
        str(row["source_uid"]): _source_record(row, str(policy["policy_version"]))
        for row in _mapping(cfg.get("pilot")).get("sources", [])
        if isinstance(row, Mapping)
    }
    outcomes = {
        str(row["outcome_key"]): _outcome_record(row, sources)
        for row in _mapping(cfg.get("pilot")).get("outcomes", [])
        if isinstance(row, Mapping)
    }
    return sources, outcomes


def _paths(config: Mapping[str, object], output_root: Path) -> dict[str, Path]:
    return {
        name: output_root / str(path)
        for name, path in builder_output_paths(config).items()
    }


def _schema_registry(
    config: Mapping[str, object], repository_root: Path
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    raw = _mapping(
        _mapping(config.get("schemas")).get("round18_literature_replay_schema_registry")
    )
    blockers: list[dict[str, object]] = []
    if set(raw) != set(SCHEMA_VERSIONS):
        blockers.append(
            _block(
                "schema_registry_incomplete",
                "Round-18 must register exactly its canonical schemas.",
                missing=sorted(set(SCHEMA_VERSIONS) - set(raw)),
                unexpected=sorted(set(raw) - set(SCHEMA_VERSIONS)),
            )
        )
    registry: dict[str, dict[str, object]] = {}
    for name, raw_entry in raw.items():
        if not isinstance(raw_entry, Mapping):
            blockers.append(
                _block(
                    "schema_registry_entry_invalid",
                    "Schema registry entry must be an object.",
                    schema=name,
                )
            )
            continue
        entry = dict(raw_entry)
        registry[name] = entry
        path = repository_root / str(entry.get("path", ""))
        if not path.is_file():
            blockers.append(
                _block(
                    "schema_missing", "Registered schema file is missing.", schema=name
                )
            )
            continue
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
        except (OSError, json.JSONDecodeError, SchemaError) as error:
            blockers.append(_block("schema_invalid", str(error), schema=name))
            continue
        version = _mapping(
            _mapping(schema.get("properties")).get("schema_version")
        ).get("const")
        if version != entry.get("schema_version") or version != SCHEMA_VERSIONS.get(
            name
        ):
            blockers.append(
                _block(
                    "schema_registry_version_mismatch",
                    "Schema version is not canonical.",
                    schema=name,
                )
            )
    return registry, sorted_blockers(blockers)


def _schema_validate(
    repository_root: Path,
    registry: Mapping[str, Mapping[str, object]],
    name: str,
    values: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    entry = registry.get(name)
    if entry is None:
        return [
            _block(
                "schema_registry_entry_missing",
                "Schema is not registered.",
                schema=name,
            )
        ]
    path = repository_root / str(entry.get("path", ""))
    if not path.is_file():
        return [_block("schema_missing", "Registered schema is missing.", schema=name)]
    try:
        validator = Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, SchemaError) as error:
        return [_block("schema_invalid", str(error), schema=name)]
    blockers: list[dict[str, object]] = []
    for index, value in enumerate(values):
        uid = _object_uid(value, f"{name}-{index}")
        for error in sorted(
            validator.iter_errors(dict(value)), key=lambda item: list(item.path)
        ):
            blockers.append(
                _block(
                    "schema_invalid",
                    error.message,
                    schema=name,
                    object_uid=uid,
                    instance_path="/".join(str(item) for item in error.path),
                )
            )
    return blockers


def _object_uid(value: Mapping[str, object], fallback: str) -> str:
    for field in (
        "episode_uid",
        "cutoff_uid",
        "bundle_uid",
        "label_uid",
        "event_uid",
        "prediction_uid",
        "ranking_uid",
        "abstention_uid",
        "metric_result_uid",
        "audit_uid",
        "source_resolution_run_uid",
        "run_uid",
        "release_uid",
        "attempt_uid",
    ):
        if value.get(field):
            return str(value[field])
    return fallback


def _content_hash_blockers(
    values: Sequence[Mapping[str, object]],
    *,
    identity_fields: Sequence[str],
    artifact: str,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    excluded = {"content_hash", "decision_artifact_hash", *identity_fields}
    for index, value in enumerate(values):
        expected = canonical_sha256(
            {key: item for key, item in value.items() if key not in excluded}
        )
        if value.get("content_hash") != expected:
            blockers.append(
                _block(
                    "content_hash_mismatch",
                    "Content-addressed artifact hash does not match its immutable payload.",
                    artifact=artifact,
                    object_uid=_object_uid(value, f"{artifact}-{index}"),
                )
            )
    return blockers


def _claim_blockers(values: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for value in values:
        if all(field in value for field in CLAIM_SCOPE):
            blockers.extend(validate_claim_boundaries(value))
    return blockers


def _release_artifacts(
    release: Mapping[str, object], output_root: Path
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    artifacts = _mapping(release.get("artifacts"))
    paths: dict[str, Path] = {}
    blockers: list[dict[str, object]] = []
    for name in REQUIRED_RELEASE_ARTIFACTS:
        entry = _mapping(artifacts.get(name))
        raw_path = str(entry.get("path", ""))
        candidate = Path(raw_path)
        if not raw_path or candidate.is_absolute():
            blockers.append(
                _block(
                    "release_artifact_path_invalid",
                    "Release artifact path must be relative.",
                    artifact=name,
                )
            )
            continue
        path = (output_root / candidate).resolve()
        if not _path_is_within(path, output_root):
            blockers.append(
                _block(
                    "release_artifact_path_invalid",
                    "Release artifact path escapes output root.",
                    artifact=name,
                )
            )
            continue
        paths[name] = path
        if not path.is_file():
            blockers.append(
                _block(
                    "release_artifact_missing",
                    "Release artifact is missing.",
                    artifact=name,
                )
            )
        elif entry.get("sha256") != sha256_file(path):
            blockers.append(
                _block(
                    "release_artifact_hash_mismatch",
                    "Release artifact hash changed.",
                    artifact=name,
                )
            )
    return paths, blockers


def _authority_blockers(authority: Mapping[str, object]) -> list[dict[str, object]]:
    blockers = [
        dict(row) for row in authority_contract_report(authority).get("blockers", [])
    ]
    if not validator_is_approved(authority, VALIDATOR_UID, STAGE_UID):
        blockers.append(
            _block(
                "validator_not_approved",
                "Round-18 validator is not authorized for the formal stage.",
            )
        )
    formal_gate = _mapping(authority.get("formal_gate"))
    if formal_gate.get("writer_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append(
            _block(
                "canonical_gate_authority_invalid",
                "Accepted Gate writer is not the canonical controller.",
            )
        )
    return sorted_blockers(blockers)


def _config_blockers(config: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    try:
        cfg = round18_config(config)
    except (KeyError, TypeError) as error:
        return [_block("round18_config_missing", str(error))]
    try:
        builder_findings = builder_config_blockers(config)
    except (KeyError, TypeError, ValueError) as error:
        blockers.append(
            _block(
                "round18_builder_config_invalid",
                "Round-18 builder configuration could not be validated.",
                error=str(error),
            )
        )
        builder_findings = []
    for finding in builder_findings:
        details = dict(finding)
        code = str(details.pop("code", "round18_builder_config_invalid"))
        blockers.append(
            _block(
                code,
                "Round-18 builder configuration contract rejected this setting.",
                **details,
            )
        )
    if cfg.get("version") != ROUND18_CONFIG_VERSION:
        blockers.append(
            _block(
                "round18_config_version_invalid",
                "Round-18 configuration version is invalid.",
            )
        )
    start = _mapping(cfg.get("authoritative_start"))
    if start.get("exact_main_commit") != "de3b691587da49896fd108cd3999127a2b5deb08":
        blockers.append(
            _block(
                "round18_start_commit_invalid",
                "Round-18 is not bound to the authoritative main commit.",
            )
        )
    if (
        start.get("evidence_fabric_sealed") is not False
        or start.get("frozen_gap_count") != 20
        or start.get("dt_core_status") != "not_evaluated"
    ):
        blockers.append(
            _block(
                "round18_start_boundary_invalid",
                "Round-18 starting state violates protected boundaries.",
            )
        )
    scope = _mapping(cfg.get("claim_scope")) or _mapping(cfg.get("scope"))
    for field, expected in CLAIM_SCOPE.items():
        if scope.get(field) is not expected:
            blockers.append(
                _block(
                    "claim_boundary_violation",
                    "Round-18 config claim scope is not conservative.",
                    field=field,
                )
            )
    policy = _mapping(_mapping(cfg.get("policies")).get("public_source"))
    if not (
        policy.get("strict_public_only_flag") is True
        and policy.get("controlled_access_required") is False
        and policy.get("resolved_controlled_source_count", 0) == 0
    ):
        blockers.append(
            _block(
                "public_source_policy_invalid",
                "Strict public-only policy is incomplete.",
            )
        )
    frozen = _mapping(_mapping(cfg.get("policies")).get("frozen_gap"))
    if (
        frozen.get("count_before_round18") != 20
        or frozen.get("count_after_round18") != 20
        or frozen.get("evidence_fabric_sealed") is not False
        or frozen.get("dt_core_status") != "not_evaluated"
    ):
        blockers.append(
            _block(
                "frozen_gap_policy_invalid",
                "Frozen-gap policy does not preserve Round-18 boundaries.",
            )
        )
    gate = _mapping(cfg.get("gate"))
    expected_states = {ACCEPTED_STATE, *BLOCKER_STATES}
    if (
        gate.get("accepted_state") != ACCEPTED_STATE
        or set(gate.get("allowed_states", [])) != expected_states
    ):
        blockers.append(
            _block(
                "gate_taxonomy_invalid", "Round-18 allowed Gate states are not exact."
            )
        )
    if tuple(str(item) for item in cfg.get("systems", [])) != tuple(SYSTEM_UIDS):
        blockers.append(
            _block(
                "system_registry_invalid",
                "Required baseline and SciTiL systems are not exact.",
            )
        )
    for group in ("parent_artifacts", "outputs"):
        for name, raw_path in _mapping(cfg.get(group)).items():
            if not str(raw_path) or Path(str(raw_path)).is_absolute():
                blockers.append(
                    _block(
                        "configured_path_invalid",
                        "Round-18 paths must be relative.",
                        field=f"{group}.{name}",
                    )
                )
    return sorted_blockers(blockers)


def _frozen_gap_uid_set_blockers(
    gap_rows: Sequence[Mapping[str, object]],
    parent_gap_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Require the materialized map to preserve the authoritative gap ledger."""

    expected = {str(row.get("gap_uid", "")).strip() for row in parent_gap_rows}
    observed = {str(row.get("gap_uid", "")).strip() for row in gap_rows}
    if not expected or "" in expected:
        return [
            _block(
                "parent_frozen_gap_ledger_invalid",
                "The authoritative Round-14 frozen-gap ledger has no valid UID set.",
            )
        ]
    if observed != expected:
        return [
            _block(
                "frozen_gap_uid_set_mismatch",
                "Round-18 materialized gap UIDs must exactly match the Round-14 ledger.",
                missing_gap_uids=sorted(expected - observed),
                unexpected_gap_uids=sorted(observed - expected),
            )
        ]
    return []


def _episode_linkage_blockers(
    episodes: Sequence[Mapping[str, object]],
    sources: Sequence[Mapping[str, object]],
    outcomes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    source_by_uid = {str(row.get("source_uid", "")): row for row in sources}
    outcome_by_uid = {str(row.get("outcome_uid", "")): row for row in outcomes}
    for episode in episodes:
        uid = str(episode.get("episode_uid", ""))
        pre = _mapping(episode.get("pre_cutoff_evidence_manifest"))
        held = _mapping(episode.get("held_out_outcome_manifest"))
        pre_sources: list[dict[str, object]] = []
        for source_uid in pre.get("source_refs", []):
            source = source_by_uid.get(str(source_uid))
            if source is None:
                blockers.append(
                    _block(
                        "pre_cutoff_source_missing",
                        "Episode references an unknown pre-cutoff source.",
                        episode_uid=uid,
                        source_uid=source_uid,
                    )
                )
                continue
            expected_hash = _mapping(pre.get("source_hashes")).get(str(source_uid))
            if expected_hash != source.get("content_hash"):
                blockers.append(
                    _block(
                        "pre_cutoff_source_hash_mismatch",
                        "Episode source hash does not match source catalog.",
                        episode_uid=uid,
                        source_uid=source_uid,
                    )
                )
            pre_sources.append(source)
        held_sources: list[dict[str, object]] = []
        for outcome_uid in held.get("outcome_refs", []):
            outcome = outcome_by_uid.get(str(outcome_uid))
            if outcome is None:
                blockers.append(
                    _block(
                        "held_out_outcome_missing",
                        "Episode references an unknown held-out outcome.",
                        episode_uid=uid,
                        outcome_uid=outcome_uid,
                    )
                )
                continue
            expected_hash = _mapping(held.get("outcome_hashes")).get(str(outcome_uid))
            if expected_hash != outcome.get("content_hash"):
                blockers.append(
                    _block(
                        "held_out_outcome_hash_mismatch",
                        "Episode outcome hash does not match outcome catalog.",
                        episode_uid=uid,
                        outcome_uid=outcome_uid,
                    )
                )
            source = _mapping(outcome.get("source"))
            held_sources.append(source or outcome)
        if not pre_sources or not held_sources:
            blockers.append(
                _block(
                    "temporal_lineage_incomplete",
                    "Episode needs traceable pre-cutoff and held-out source objects.",
                    episode_uid=uid,
                )
            )
        else:
            blockers.extend(
                validate_temporal_ordering(
                    episode.get("cutoff_timestamp"),
                    pre_cutoff_sources=pre_sources,
                    held_out_sources=held_sources,
                )
            )
    return blockers


def _bundle_linkage_blockers(
    agent: Mapping[str, object],
    held: Mapping[str, object],
    sources: Sequence[Mapping[str, object]],
    outcomes: Sequence[Mapping[str, object]],
    release: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers = validate_bundle_isolation(agent, held)
    source_hashes = {str(row.get("content_hash", "")) for row in sources}
    outcome_hashes = {str(row.get("content_hash", "")) for row in outcomes}
    agent_hashes = {
        str(row.get("content_hash", "")) for row in _mapping_list(agent.get("sources"))
    }
    held_hashes = {
        str(row.get("content_hash", "")) for row in _mapping_list(held.get("outcomes"))
    }
    if not agent_hashes or not agent_hashes.issubset(source_hashes):
        blockers.append(
            _block(
                "agent_bundle_source_linkage_invalid",
                "Agent-visible source hashes do not resolve to the source catalog.",
            )
        )
    if held_hashes != outcome_hashes:
        blockers.append(
            _block(
                "held_out_bundle_outcome_linkage_invalid",
                "Held-out bundle does not exactly match held-out outcome catalog.",
            )
        )
    if agent_hashes & held_hashes:
        blockers.append(
            _block(
                "pre_cutoff_held_out_content_overlap",
                "Agent and held-out bundles share content hashes.",
            )
        )
    release_held = _mapping(release.get("evaluator_only_held_out_bundle"))
    if (
        release_held.get("bundle_uid") != held.get("bundle_uid")
        or release_held.get("bundle_hash") != held.get("content_hash")
        or release_held.get("mount_path") != held.get("mount_path")
        or release_held.get("agent_runtime_readable") is not False
    ):
        blockers.append(
            _block(
                "release_held_out_binding_invalid",
                "Release candidate does not bind the actual evaluator-only bundle.",
            )
        )
    return blockers


def _execution_isolation_blockers(
    leakage_audit: Mapping[str, object],
    agent: Mapping[str, object],
    held: Mapping[str, object],
    runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    agent_identity = _mapping_list(agent.get("process_identities"))
    held_identity = _mapping_list(held.get("process_identities"))
    agent_process = (
        str(agent_identity[0].get("process_uid", "")) if agent_identity else ""
    )
    evaluator_process = (
        str(held_identity[0].get("process_uid", "")) if held_identity else ""
    )
    attempts = [
        *_mapping_list(agent.get("access_attempts")),
        *_mapping_list(held.get("access_attempts")),
    ]
    attempts.extend(_mapping_list(leakage_audit.get("access_attempts")))
    execution = {
        "agent_process_identity": agent_process,
        "evaluator_process_identity": evaluator_process,
        "held_out_bundle_readable_by_agent": leakage_audit.get(
            "held_out_bundle_readable_by_agent",
            leakage_audit.get("agent_process_received_held_out_material"),
        ),
        "agent_direct_tool_execution": leakage_audit.get(
            "agent_direct_tool_execution", False
        ),
        "agent_direct_state_update": leakage_audit.get(
            "agent_direct_state_update", False
        ),
        "agent_gate_authority": leakage_audit.get("agent_gate_authority", False),
        "access_attempts": attempts,
    }
    blockers = validate_execution_isolation(
        execution, held_out_bundle_uid=str(held.get("bundle_uid", ""))
    )
    if (
        leakage_audit.get("status") != "passed"
        or leakage_audit.get("held_out_access_attempt_count") != 0
    ):
        blockers.append(
            _block(
                "held_out_access_audit_failed",
                "Leakage audit does not certify zero held-out accesses.",
            )
        )
    if leakage_audit.get("agent_bundle_hash") != agent.get(
        "content_hash"
    ) or leakage_audit.get("held_out_bundle_hash") != held.get("content_hash"):
        blockers.append(
            _block(
                "bundle_hash_audit_mismatch",
                "Leakage audit bundle hashes do not match materialized bundles.",
            )
        )
    if leakage_audit.get("prediction_hash_before_reveal") != leakage_audit.get(
        "prediction_hash_after_reveal"
    ):
        blockers.append(
            _block(
                "pre_cutoff_decision_changed_after_reveal",
                "Immutable T4 decision hash changed after reveal.",
            )
        )
    for run in runs:
        pre = _mapping(run.get("pre_cutoff_bundle"))
        held_binding = _mapping(run.get("held_out_bundle"))
        if pre.get("bundle_hash") != agent.get("content_hash") or held_binding.get(
            "bundle_hash"
        ) != held.get("content_hash"):
            blockers.append(
                _block(
                    "run_bundle_binding_invalid",
                    "Run does not bind the materialized bundle hashes.",
                    run_uid=run.get("run_uid"),
                )
            )
        if (
            held_binding.get("readable_by_agent_before_t5") is not False
            or held_binding.get("pre_t5_access_attempt_count") != 0
        ):
            blockers.append(
                _block(
                    "held_out_bundle_readable_by_agent",
                    "Run records pre-reveal held-out access.",
                    run_uid=run.get("run_uid"),
                )
            )
        if run.get("agent_direct_tool_execution") is not False:
            blockers.append(
                _block(
                    "agent_execution_isolation_violation",
                    "Run permits direct Agent tool execution.",
                    run_uid=run.get("run_uid"),
                )
            )
    return blockers


def _metric_blockers(
    metrics: Sequence[Mapping[str, object]], results: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    def defined_mean(values: Sequence[object]) -> float | None:
        numeric = [float(value) for value in values if value is not None]
        return sum(numeric) / len(numeric) if numeric else None

    blockers: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for result in results:
        grouped[
            (str(result.get("system_uid", "")), str(result.get("track", "")))
        ].append(result)
    metrics_by_key: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for metric in metrics:
        scope = _mapping(metric.get("scope"))
        key = (
            str(metric.get("system_uid", "")),
            str(scope.get("track", "")),
            str(metric.get("metric_name", "")),
        )
        if key in metrics_by_key:
            blockers.append(
                _block(
                    "metric_duplicate",
                    "Metric result is duplicated for system, track, and metric.",
                    metric_name=key[2],
                )
            )
        metrics_by_key[key] = metric
    for (system_uid, track), group in grouped.items():
        expected: dict[str, object] = {}
        try:
            if track == TRACK_A:
                expected.update(compute_direction_metrics(group))
            if track == TRACK_B:
                expected.update(compute_ranking_metrics(group, k=2))
            expected.update(
                compute_reliability_metrics(group, false_confidence_threshold=0.7)
            )
            expected.update(
                {
                    "falsifier_invocation_rate": sum(
                        bool(item.get("falsifier_invoked")) for item in group
                    )
                    / len(group),
                    "valid_challenge_rate": sum(
                        bool(item.get("valid_falsifier_challenge")) for item in group
                    )
                    / len(group),
                    "hypothesis_reversal_rate": sum(
                        bool(item.get("hypothesis_reversal")) for item in group
                    )
                    / len(group),
                    "unsupported_positive_reduction": sum(
                        bool(item.get("unsupported_positive_avoided")) for item in group
                    )
                    / len(group),
                    "additional_information_cost": sum(
                        float(item.get("cost_to_identification", 0.0)) for item in group
                    )
                    / len(group),
                }
            )
            if track in {TRACK_D, TRACK_E}:
                expected.update(
                    {
                        "cost_to_identification": defined_mean(
                            [item.get("cost_to_identification") for item in group]
                        ),
                        "steps_to_falsification": defined_mean(
                            [item.get("steps_to_falsification") for item in group]
                        ),
                        "budget_adjusted_utility": defined_mean(
                            [item.get("budget_adjusted_utility") for item in group]
                        ),
                        "information_gain_per_action": defined_mean(
                            [item.get("information_gain_per_action") for item in group]
                        ),
                        "candidate_space_reduction": defined_mean(
                            [item.get("candidate_space_reduction") for item in group]
                        ),
                    }
                )
        except (TypeError, ValueError, ZeroDivisionError) as error:
            blockers.append(
                _block(
                    "metric_recomputation_failed",
                    str(error),
                    system_uid=system_uid,
                    track=track,
                )
            )
            continue
        for name, expected_value in expected.items():
            row = metrics_by_key.get((system_uid, track, name))
            if row is None:
                blockers.append(
                    _block(
                        "metric_missing",
                        "Required metric is absent.",
                        system_uid=system_uid,
                        track=track,
                        metric_name=name,
                    )
                )
                continue
            actual = row.get("value")
            if expected_value is None:
                if row.get("valid") is not False or actual is not None:
                    blockers.append(
                        _block(
                            "metric_undefined_state_invalid",
                            "Undefined metric must be explicit and non-numeric.",
                            metric_name=name,
                        )
                    )
            elif not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), float(expected_value), rel_tol=1e-12, abs_tol=1e-12
            ):
                blockers.append(
                    _block(
                        "metric_value_mismatch",
                        "Stored metric does not match recomputed result.",
                        metric_name=name,
                        system_uid=system_uid,
                        track=track,
                    )
                )
    return blockers


def _pilot_requirement_blockers(
    config: Mapping[str, object],
    episodes: Sequence[Mapping[str, object]],
    rejection_rows: Sequence[Mapping[str, object]],
    pilot: Mapping[str, object],
) -> list[dict[str, object]]:
    cfg = round18_config(config)
    requirements = _mapping(_mapping(cfg.get("pilot")).get("minimum_requirements"))
    counts = Counter(str(row.get("track", "")) for row in episodes)
    expected = {
        TRACK_A: int(requirements.get("endpoint_direction_episode_count", 2)),
        TRACK_B: int(requirements.get("intervention_ranking_episode_count", 1)),
        TRACK_C: int(requirements.get("mechanism_discrimination_episode_count", 1)),
        TRACK_D: int(requirements.get("information_action_episode_count", 1)),
        TRACK_E: int(requirements.get("sequential_hypothesis_episode_count", 1)),
        TRACK_F: int(requirements.get("abstention_required_episode_count", 1)),
    }
    blockers: list[dict[str, object]] = []
    for track, minimum in expected.items():
        if counts[track] < minimum:
            blockers.append(
                _block(
                    "pilot_track_incomplete",
                    "Pilot does not meet track minimum.",
                    track=track,
                    minimum=minimum,
                    actual=counts[track],
                )
            )
    if not any(
        row.get("status") in {"blocked", "rejected", "invalidated"} for row in episodes
    ):
        blockers.append(
            _block(
                "pilot_blocked_episode_missing",
                "Pilot must preserve a blocked episode.",
            )
        )
    if not any(
        row.get("status") not in {"blocked", "rejected", "invalidated"}
        for row in episodes
    ):
        blockers.append(
            _block(
                "pilot_successful_episode_missing",
                "Pilot must include a successful bounded replay.",
            )
        )
    if not any(row.get("track") == TRACK_F for row in episodes):
        blockers.append(
            _block(
                "pilot_abstention_episode_missing",
                "Pilot lacks abstention-required episode.",
            )
        )
    if not any(
        "conflict" in json.dumps(row, sort_keys=True).lower() for row in episodes
    ):
        blockers.append(
            _block(
                "pilot_conflict_episode_missing",
                "Pilot lacks an explicit conflict episode.",
            )
        )
    if not any(
        row.get("control_type") == "held_out_bundle_access_attack"
        and row.get("decision") == "rejected"
        and row.get("attempt_preserved") is True
        for row in rejection_rows
    ):
        blockers.append(
            _block(
                "pilot_leakage_attack_missing",
                "Pilot lacks a preserved rejected held-out access attack.",
            )
        )
    negative_attempts = _mapping_list(pilot.get("negative_control_attempts"))
    held_out_attempt = next(
        (
            row
            for row in negative_attempts
            if row.get("actor_role") == "agent" and row.get("evaluator_only") is True
        ),
        None,
    )
    if not (
        held_out_attempt
        and held_out_attempt.get("attempted") is True
        and held_out_attempt.get("access_granted") is False
        and held_out_attempt.get("outcome_exposed") is False
        and held_out_attempt.get("attempt_preserved") is True
    ):
        blockers.append(
            _block(
                "pilot_leakage_attack_not_executed",
                "The held-out access control must execute, fail closed, and remain preserved.",
            )
        )
    if pilot.get("scientific_superiority_claim") is not False:
        blockers.append(
            _block(
                "scientific_superiority_claim",
                "Pilot cannot claim scientific superiority.",
            )
        )
    return blockers


def validate_pilot_outputs(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    source_output_root: Path,
    parent_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate all materialized Round-18 pilot and v0 release artifacts."""

    blockers = _config_blockers(config)
    registry, schema_registry_blockers = _schema_registry(config, repository_root)
    blockers += schema_registry_blockers
    try:
        authority = load_authority_contract(repository_root / AUTHORITY_PATH)
        blockers += _authority_blockers(authority)
    except (OSError, ValueError, TypeError) as error:
        blockers.append(_block("gate_authority_unreadable", str(error)))
    try:
        parent = validate_parent_state(config, parent_root)
    except (FileNotFoundError, ValueError, TypeError, KeyError) as error:
        return sorted_blockers(
            blockers + [_block("parent_state_invalid", str(error))]
        ), {}

    paths = _paths(config, source_output_root)
    required_json = {
        "episode_manifest": paths.get("episode_manifest"),
        "pilot_validation": paths.get("pilot_validation"),
        "public_source_boundary": paths.get("public_source_boundary"),
        "temporal_split_audit": paths.get("temporal_split_audit"),
        "leakage_audit": paths.get("leakage_audit"),
        "memory_leakage_risk": paths.get("memory_leakage_risk"),
        "benchmark_release_manifest": paths.get("benchmark_release_manifest"),
        "split_manifest": paths.get("split_manifest"),
        "metric_specification": paths.get("metric_specification"),
    }
    json_payloads = {name: _load_json(path) for name, path in required_json.items()}
    agent_path = (
        paths.get("agent_visible_bundle_root", source_output_root)
        / "bundle_manifest.json"
    )
    held_path = (
        paths.get("evaluator_only_bundle_root", source_output_root)
        / "bundle_manifest.json"
    )
    decision_path = (
        paths.get("agent_visible_bundle_root", source_output_root)
        / "immutable_t4_decisions.json"
    )
    json_payloads["agent_bundle"] = _load_json(agent_path)
    json_payloads["held_bundle"] = _load_json(held_path)
    json_payloads["decision_artifact"] = _load_json(decision_path)
    for name, value in json_payloads.items():
        if value is None:
            blockers.append(
                _block(
                    "pilot_artifact_missing",
                    "Required Round-18 JSON artifact is missing or invalid.",
                    artifact=name,
                )
            )
    if any(value is None for value in json_payloads.values()):
        return sorted_blockers(blockers), {
            "paths": {name: str(path) for name, path in paths.items()},
            "parent_input_hashes": parent.get("hashes", {}),
        }

    parquet_rows: dict[str, list[dict[str, object]]] = {}
    for name in REQUIRED_RELEASE_ARTIFACTS:
        if name in {"split_manifest", "metric_specification"}:
            continue
        rows, row_blockers = _read_parquet(paths[name])
        parquet_rows[name] = rows
        blockers += row_blockers
    tsv_rows: dict[str, list[dict[str, object]]] = {}
    for name in REQUIRED_PILOT_CATALOGS:
        rows, row_blockers = _read_tsv(paths[name])
        tsv_rows[name] = rows
        blockers += row_blockers
    if blockers:
        return sorted_blockers(blockers), {
            "paths": {name: str(path) for name, path in paths.items()},
            "parent_input_hashes": parent.get("hashes", {}),
        }

    manifest = json_payloads["episode_manifest"] or {}
    pilot = json_payloads["pilot_validation"] or {}
    public_boundary = json_payloads["public_source_boundary"] or {}
    temporal_audit = json_payloads["temporal_split_audit"] or {}
    leakage_audit = json_payloads["leakage_audit"] or {}
    memory_risk = json_payloads["memory_leakage_risk"] or {}
    release = json_payloads["benchmark_release_manifest"] or {}
    split_manifest = json_payloads["split_manifest"] or {}
    metric_specification = json_payloads["metric_specification"] or {}
    agent = json_payloads["agent_bundle"] or {}
    held = json_payloads["held_bundle"] or {}
    decisions = json_payloads["decision_artifact"] or {}
    episodes = _mapping_list(manifest.get("episodes"))
    events = _mapping_list(manifest.get("sequential_reveal_events"))
    sources = parquet_rows["pre_cutoff_evidence_manifest"]
    outcomes = parquet_rows["held_out_outcome_manifest"]
    labels = parquet_rows["label_catalog"]
    runs = parquet_rows["system_run_manifest"]
    results = parquet_rows["result_catalog"]
    metrics = _mapping_list(metric_specification.get("metric_rows"))
    predictions = _mapping_list(decisions.get("predictions"))
    rankings = _mapping_list(decisions.get("rankings"))
    abstentions = _mapping_list(decisions.get("abstentions"))
    rejection_rows = _mapping_list(manifest.get("rejection_ledger"))

    full_traces: list[Mapping[str, object]] = []
    for decision in _mapping_list(decisions.get("decisions")):
        system_uid = str(decision.get("system_uid", ""))
        trace = decision.get("scitil_runtime_trace")
        if system_uid == "full_governed_scitil":
            if not isinstance(trace, Mapping):
                blockers.append(
                    _block(
                        "full_scitil_runtime_trace_missing",
                        "Every full governed SciTiL run must retain its TwinKernel trace.",
                        decision_uid=decision.get("decision_uid"),
                    )
                )
            else:
                full_traces.append(trace)
                blockers += validate_scitil_runtime_trace(trace)
        elif trace is not None:
            blockers.append(
                _block(
                    "ablation_claims_full_scitil_runtime",
                    "Only full_governed_scitil may claim a Round-17 TwinKernel trace.",
                    system_uid=system_uid,
                )
            )
    if full_traces:
        gate_types = {
            str(record.get("gate_type"))
            for trace in full_traces
            for record in _mapping_list(trace.get("gate_records"))
        }
        if not {"H-Gate", "W-Gate", "I-Gate", "O-Gate"}.issubset(gate_types):
            blockers.append(
                _block(
                    "full_scitil_gate_coverage_incomplete",
                    "The qualification pilot must exercise H/W/I/O Gate routes.",
                    observed_gate_types=sorted(gate_types),
                )
            )

    release_paths, release_blockers = _release_artifacts(release, source_output_root)
    blockers += release_blockers
    for name, configured_path in (
        ("split_manifest", paths["split_manifest"]),
        ("metric_specification", paths["metric_specification"]),
    ):
        if release_paths.get(name) != configured_path.resolve():
            blockers.append(
                _block(
                    "release_artifact_path_mismatch",
                    "Release artifact path differs from configured canonical path.",
                    artifact=name,
                )
            )
    if release.get("release_content_hash") != canonical_sha256(
        {
            key: value
            for key, value in release.items()
            if key not in {"release_uid", "release_content_hash"}
        }
    ):
        blockers.append(
            _block("release_content_hash_mismatch", "Release content hash is invalid.")
        )
    for path in release_paths.values():
        if _git_tracked(repository_root, path):
            blockers.append(
                _block(
                    "large_benchmark_data_committed",
                    "Generated benchmark release data is tracked by Git.",
                    path=str(path),
                )
            )

    schema_values: dict[str, list[dict[str, object]]] = {
        "literature_replay_episode": episodes,
        "literature_cutoff": _mapping_list(manifest.get("literature_cutoffs")),
        "pre_cutoff_evidence_bundle": [agent],
        "held_out_outcome": [held],
        "benchmark_label": labels,
        "sequential_reveal_event": events,
        "benchmark_prediction": predictions,
        "benchmark_ranking": rankings,
        "benchmark_abstention": abstentions,
        "benchmark_metric_result": metrics,
        "memory_leakage_risk": [memory_risk],
        "public_source_boundary": [public_boundary],
        "literature_replay_run": runs,
        "round18_benchmark_release": [release],
    }
    for name, values in schema_values.items():
        if not values:
            blockers.append(
                _block(
                    "schema_instance_missing",
                    "Required typed Round-18 object is absent.",
                    schema=name,
                )
            )
        else:
            blockers += _schema_validate(repository_root, registry, name, values)

    for episode in episodes:
        blockers += validate_episode(episode)
        blockers += validate_independence_groups(episode)
    blockers += validate_unique_uids(episodes, ("episode_uid",))
    blockers += validate_split_leakage(episodes)
    blockers += validate_split_manifest(split_manifest, episodes=episodes)
    blockers += _episode_linkage_blockers(episodes, sources, outcomes)
    for source in sources:
        blockers += validate_public_source(source)
    blockers += validate_public_source_boundary(public_boundary, sources=sources)
    blockers += _bundle_linkage_blockers(agent, held, sources, outcomes, release)
    blockers += _execution_isolation_blockers(leakage_audit, agent, held, runs)
    blockers += validate_memory_risk_disclosure(memory_risk)
    for label in labels:
        blockers += validate_label_provenance(label)
    for cutoff in schema_values["literature_cutoff"]:
        blockers += validate_temporal_cutoff(cutoff)

    events_by_episode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in events:
        events_by_episode[str(event.get("episode_uid", ""))].append(event)
    admitted_episode_uids = {
        str(row.get("episode_uid", ""))
        for row in episodes
        if row.get("status") != "rejected"
    }
    if set(events_by_episode) != admitted_episode_uids:
        blockers.append(
            _block(
                "sequential_reveal_episode_coverage_invalid",
                "Every admitted episode needs exactly one T0-T7 event chain.",
            )
        )
    lineage_edges: list[tuple[str, str]] = []
    for episode_uid, values in events_by_episode.items():
        ordered = sorted(values, key=lambda row: int(row.get("sequence_index", -1)))
        blockers += validate_sequential_reveal(ordered, episode_uid=episode_uid)
        for row in ordered:
            prior = row.get("prior_event_hash")
            current = row.get("event_hash")
            if prior:
                lineage_edges.append((str(prior), str(current)))
    blockers += validate_unique_uids(events, ("event_uid",))
    blockers += validate_lineage_acyclic(lineage_edges)
    attempts = _mapping_list(manifest.get("replay_attempts"))
    if not attempts:
        blockers.append(
            _block(
                "attempt_preservation_ledger_missing",
                "Round-18 requires an immutable replay-attempt ledger.",
            )
        )
    else:
        blockers += validate_failed_retry_preservation(attempts)
    for abstention in abstentions:
        blockers += validate_abstention_semantics(abstention)
    blockers += validate_unique_uids(predictions, ("prediction_uid",))
    blockers += validate_unique_uids(rankings, ("ranking_uid",))
    blockers += validate_unique_uids(abstentions, ("abstention_uid",))
    blockers += validate_unique_uids(labels, ("label_uid",))
    blockers += validate_unique_uids(runs, ("run_uid",))
    blockers += validate_unique_uids(metrics, ("metric_result_uid",))

    blockers += _content_hash_blockers(
        labels, identity_fields=("label_uid",), artifact="benchmark_label"
    )
    blockers += _content_hash_blockers(
        predictions,
        identity_fields=("prediction_uid",),
        artifact="benchmark_prediction",
    )
    blockers += _content_hash_blockers(
        rankings, identity_fields=("ranking_uid",), artifact="benchmark_ranking"
    )
    blockers += _content_hash_blockers(
        abstentions,
        identity_fields=("abstention_uid",),
        artifact="benchmark_abstention",
    )
    blockers += _content_hash_blockers(
        metrics,
        identity_fields=("metric_result_uid",),
        artifact="benchmark_metric_result",
    )
    blockers += _content_hash_blockers(
        runs, identity_fields=("run_uid",), artifact="literature_replay_run"
    )
    blockers += _metric_blockers(metrics, results)

    claim_values = [
        pilot,
        agent,
        held,
        release,
        *predictions,
        *rankings,
        *abstentions,
        *metrics,
        *runs,
        *results,
    ]
    blockers += _claim_blockers(claim_values)
    gap_rows = _mapping_list(manifest.get("frozen_gap_map")) or _mapping_list(
        release.get("frozen_gap_map")
    )
    if not gap_rows:
        blockers.append(
            _block(
                "frozen_gap_map_missing",
                "Round-18 must map every frozen gap without resolving it.",
            )
        )
    else:
        blockers += validate_frozen_gap_isolation(
            gap_rows, expected_count=ROUND18_FROZEN_GAP_COUNT
        )
        blockers += _frozen_gap_uid_set_blockers(gap_rows, parent["gap_rows"])
    if (
        release.get("frozen_gap_count_before_round18") != 20
        or release.get("frozen_gap_count_after_round18") != 20
        or release.get("evidence_fabric_sealed") is not False
        or release.get("dt_core_status") != "not_evaluated"
        or release.get("production_model_trained") is not False
        or release.get("final_held_out_partition_frozen") is not True
        or release.get("rules_frozen_before_final_held_out_access") is not True
    ):
        blockers.append(
            _block(
                "benchmark_result_closes_empirical_gap",
                "Release violates frozen-gap, Evidence Fabric, DT_CORE, training, or final-held-out boundaries.",
            )
        )

    governance_records: list[dict[str, object]] = []
    pilot_governance = _mapping(pilot.get("governance_violation_counts"))
    if pilot_governance:
        governance_records.append(
            {field: pilot_governance.get(field, 0) for field in GOVERNANCE_COUNT_FIELDS}
        )
    for run in runs:
        counts = _mapping(run.get("governance_violation_counts"))
        if counts:
            governance_records.append(
                {field: counts.get(field, 0) for field in GOVERNANCE_COUNT_FIELDS}
            )
    try:
        governance = aggregate_governance_counts(governance_records)
    except ValueError as error:
        governance = {field: -1 for field in GOVERNANCE_COUNT_FIELDS} | {
            "all_governance_counts_zero": False
        }
        blockers.append(_block("governance_count_invalid", str(error)))
    if (
        not governance_records
        or governance.get("all_governance_counts_zero") is not True
    ):
        blockers.append(
            _block(
                "governance_violation",
                "All governance counts must be present and zero.",
            )
        )
    if (
        temporal_audit.get("status") != "passed"
        or temporal_audit.get("future_evidence_pre_cutoff_count") != 0
        or temporal_audit.get("strong_admission_ambiguous_timestamp_count") != 0
        or temporal_audit.get("final_held_out_touched") is not False
    ):
        blockers.append(
            _block(
                "temporal_audit_failed",
                "Temporal-cutoff audit is incomplete or failed.",
            )
        )
    if set(_mapping(memory_risk.get("tracks"))) != {
        "retrieval_isolated_replay",
        "post_model_cutoff_replay",
        "memory_risk_stress_track",
    }:
        blockers.append(
            _block(
                "memory_tracks_incomplete",
                "All memory-risk tracks must be separately reported.",
            )
        )
    if set(str(run.get("system_uid", "")) for run in runs) != set(SYSTEM_UIDS):
        blockers.append(
            _block(
                "required_systems_incomplete",
                "Required systems and ablations did not all execute.",
            )
        )
    run_by_uid = {str(run.get("run_uid")): run for run in runs}
    decision_by_uid = {
        str(decision.get("run_uid")): decision
        for decision in _mapping_list(decisions.get("decisions"))
    }
    for run_uid, run in run_by_uid.items():
        decision = decision_by_uid.get(run_uid, {})
        runtime_trace = decision.get("scitil_runtime_trace")
        if run.get("system_uid") == "full_governed_scitil":
            if (
                run.get("runtime_mode") != "round17_twin_kernel"
                or run.get("twin_kernel_invoked") is not True
                or not isinstance(runtime_trace, Mapping)
                or run.get("runtime_trace_hash") != runtime_trace.get("content_hash")
                or run.get("gate_writer_uid") != CANONICAL_CONTROLLER_UID
            ):
                blockers.append(
                    _block(
                        "full_scitil_execution_identity_missing",
                        "Full SciTiL run manifest must bind the executed TwinKernel trace and controller.",
                        run_uid=run_uid,
                    )
                )
        elif (
            run.get("twin_kernel_invoked") is not False
            or run.get("runtime_trace_hash") is not None
            or run.get("gate_writer_uid") is not None
        ):
            blockers.append(
                _block(
                    "ablation_runtime_identity_invalid",
                    "Ablation run manifests cannot claim full TwinKernel execution.",
                    run_uid=run_uid,
                )
            )
    blockers += _pilot_requirement_blockers(config, episodes, rejection_rows, pilot)

    summary = {
        "paths": {name: str(path) for name, path in paths.items()},
        "parent_input_hashes": parent.get("hashes", {}),
        "release": release,
        "release_artifact_hashes": {
            name: sha256_file(path)
            for name, path in release_paths.items()
            if path.is_file()
        },
        "episode_count": len(episodes),
        "track_counts": dict(
            sorted(Counter(str(row.get("track", "")) for row in episodes).items())
        ),
        "source_count": len(sources),
        "label_count": len(labels),
        "event_count": len(events),
        "run_count": len(runs),
        "result_count": len(results),
        "metric_count": len(metrics),
        "frozen_gap_map": gap_rows,
        "governance_violation_counts": governance,
        "evidence_fabric": {
            "materialization": "materialized_unsealed",
            "sealed": False,
        },
        "dt_core": {"status": "not_evaluated"},
        "final_held_out": {
            "partition": FINAL_HELD_OUT_PARTITION,
            "status": "frozen_untouched",
        },
    }
    return sorted_blockers(blockers), summary


def _identity_blockers(
    identity_path: Path | None,
    repository_root: Path,
    release: Mapping[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    identity = _load_json(identity_path)
    if identity is None:
        return [
            _block(
                "execution_identity_missing",
                "Exact-clean execution identity is missing.",
            )
        ], None
    blockers: list[dict[str, object]] = []
    expected = expected_execution_identity(repository_root)
    if identity.get("schema_version") != EXECUTION_IDENTITY_SCHEMA_VERSION:
        blockers.append(
            _block(
                "execution_identity_schema_invalid",
                "Execution identity schema is invalid.",
            )
        )
    if (
        identity.get("exact_git_commit") != expected["exact_git_commit"]
        or identity.get("tree_hash") != expected["tree_hash"]
    ):
        blockers.append(
            _block(
                "execution_commit_mismatch",
                "Execution identity does not match the exact source commit/tree.",
            )
        )
    if (
        identity.get("dirty_status") not in {"", "clean"}
        or identity.get("reproduction_relation", "exact_clean_commit")
        != "exact_clean_commit"
    ):
        blockers.append(
            _block(
                "execution_identity_not_exact_clean",
                "Execution identity is not exact-clean.",
            )
        )
    if (
        identity.get("source_read_only", identity.get("code_worktree_read_only"))
        is not True
    ):
        blockers.append(
            _block(
                "execution_source_not_read_only",
                "Execution source worktree was not read-only.",
            )
        )
    if (
        identity.get(
            "isolated_writable_runtime_output", identity.get("writable_runtime_output")
        )
        is not True
        or identity.get("source_output_isolated") is not True
    ):
        blockers.append(
            _block(
                "execution_output_not_isolated",
                "Writable runtime output was not isolated from source.",
            )
        )
    if not _cpu_only_placement_valid(identity):
        blockers.append(
            _block(
                "execution_cpu_placement_invalid",
                "CPU-only validation must run on cpu04/cpu or cpu03/gpu with zero GPU allocation.",
            )
        )
    if not str(identity.get("job_name", "")).startswith(ROUND18_JOB_PREFIX):
        blockers.append(
            _block(
                "execution_job_name_invalid",
                "Execution job does not use the NrnSift Round-18 prefix.",
            )
        )
    for field in ("python_pycache_prefix", "pytest_cache"):
        raw_path = str(identity.get(field, ""))
        if not raw_path or _path_is_within(Path(raw_path), repository_root):
            blockers.append(
                _block(
                    "execution_cache_not_isolated",
                    "Execution cache is absent or inside source worktree.",
                    field=field,
                )
            )
    if (
        identity.get("index_lock_absent") is not True
        or identity.get("active_git_writer_absent") is not True
    ):
        blockers.append(
            _block(
                "execution_git_writer_present",
                "Execution overlapped an active Git writer or lock.",
            )
        )
    if (
        identity.get("evidence_fabric_sealed") is not False
        or identity.get(
            "frozen_gap_count_after_round18", identity.get("frozen_gap_count")
        )
        != 20
        or identity.get("dt_core_status") != "not_evaluated"
        or identity.get("production_model_training", False) is not False
        or identity.get("final_held_out_accessed", False) is not False
    ):
        blockers.append(
            _block(
                "execution_scope_boundary_invalid",
                "Execution identity violates protected Round-18 state.",
            )
        )
    if release:
        for identity_field, release_field in (
            ("config_hash", "config_hash"),
            ("schema_registry_hash", "schema_registry_hash"),
        ):
            if identity.get(identity_field) != release.get(release_field):
                blockers.append(
                    _block(
                        "execution_materialization_identity_mismatch",
                        "Execution identity differs from release materialization.",
                        field=identity_field,
                    )
                )
    return sorted_blockers(blockers), identity


def build_prefull_report(
    *,
    pilot_blockers: Sequence[Mapping[str, object]],
    pilot_summary: Mapping[str, object],
    focused: Mapping[str, object],
    identity: Mapping[str, object] | None,
    identity_blockers: Sequence[Mapping[str, object]],
    execution_dirty_status: str,
) -> dict[str, object]:
    blockers = [dict(item) for item in pilot_blockers] + [
        dict(item) for item in identity_blockers
    ]
    names = set(str(name) for name in focused.get("testcase_names", []))
    if not (
        focused.get("available")
        and focused.get("failures") == 0
        and focused.get("errors") == 0
        and focused.get("skipped") == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(names)
    ):
        blockers.append(
            _block(
                "focused_negative_tests_not_verified",
                "Mandatory Round-18 negative tests are not verified.",
                missing=sorted(MANDATORY_NEGATIVE_TEST_NAMES - names),
            )
        )
    if execution_dirty_status != "clean":
        blockers.append(
            _block("execution_worktree_dirty", "Pre-full execution worktree is dirty.")
        )
    blockers = sorted_blockers(blockers)
    return {
        "schema_version": PREFULL_SCHEMA_VERSION,
        "validator_uid": VALIDATOR_UID,
        "status": "passed" if not blockers else "blocked",
        "gate_assessment": gate_state_for_blockers(blockers),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "focused_tests": dict(focused),
        "execution_identity": dict(identity or {}),
        "pilot_summary": dict(pilot_summary),
        "full_suite_authorized": not blockers,
        "frozen_gap_count": 20,
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }


def _inventory_blockers(
    path: Path | None, full: Mapping[str, object]
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    value = _load_json(path)
    if value is None:
        return [
            _block("inventory_missing", "Test inventory reconciliation is missing.")
        ], None
    blockers: list[dict[str, object]] = []
    if value.get("schema_version") != TEST_INVENTORY_SCHEMA_VERSION:
        blockers.append(
            _block("inventory_schema_invalid", "Test inventory schema is invalid.")
        )
    if (
        value.get("round17i_parent_test_count", value.get("parent_test_count"))
        != ROUND18_PARENT_TEST_COUNT
    ):
        blockers.append(
            _block(
                "inventory_parent_count_invalid",
                "Round-17I accepted test count must remain 557.",
            )
        )
    if (
        value.get("status") != "complete"
        or value.get("unexplained_missing_tests", 0) != 0
        or value.get("mandatory_disappearance", 0) != 0
        or value.get("untracked_test_count", 0) != 0
    ):
        blockers.append(
            _block(
                "inventory_incomplete", "Test inventory is incomplete or loses tests."
            )
        )
    if full.get("available") and value.get(
        "round18_current_test_count", value.get("current_test_count")
    ) != full.get("tests"):
        blockers.append(
            _block(
                "inventory_count_mismatch",
                "Inventory count differs from JUnit full-suite count.",
            )
        )
    return sorted_blockers(blockers), value


def _full_suite_execution_blockers(
    full_execution: Mapping[str, object] | None,
    identity: Mapping[str, object] | None,
    full: Mapping[str, object],
) -> list[dict[str, object]]:
    if (
        full_execution is None
        or full_execution.get("schema_version") != FULL_SUITE_EXECUTION_SCHEMA_VERSION
    ):
        return [
            _block(
                "execution_full_suite_missing",
                "CPU full-suite execution evidence is missing.",
            )
        ]
    valid = (
        full_execution.get("terminal_state") == "COMPLETED"
        and full_execution.get("exit_code") == "0:0"
        and _cpu_only_placement_valid(full_execution)
        and str(full_execution.get("job_name", "")).startswith(ROUND18_JOB_PREFIX)
        and identity is not None
        and full_execution.get("exact_git_commit") == identity.get("exact_git_commit")
        and int(full_execution.get("failed", full_execution.get("failures", -1))) == 0
        and int(full_execution.get("errors", -1)) == 0
        and int(full_execution.get("skipped", -1)) == 0
        and int(full_execution.get("mandatory_skipped", -1)) == 0
        and int(full_execution.get("core_xfail", -1)) == 0
        and full.get("available")
        and int(full.get("tests", 0)) >= ROUND18_PARENT_TEST_COUNT
        and full_execution.get("passed") == full.get("tests")
    )
    return (
        []
        if valid
        else [
            _block(
                "execution_full_suite_failed",
                "Full-suite execution evidence does not prove a clean CPU pass.",
            )
        ]
    )


def _commit_manifest_blockers(
    path: Path | None, identity: Mapping[str, object] | None
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    value = _load_json(path)
    if value is None:
        return [
            _block("commit_manifest_missing", "Scoped commit manifest is missing.")
        ], None
    blockers: list[dict[str, object]] = []
    if value.get("schema_version") != COMMIT_MANIFEST_SCHEMA_VERSION:
        blockers.append(
            _block(
                "commit_manifest_schema_invalid", "Commit manifest schema is invalid."
            )
        )
    if value.get("round_uid", value.get("round")) not in {STAGE_UID, "round18"}:
        blockers.append(
            _block(
                "commit_manifest_round_invalid",
                "Commit manifest does not identify Round-18.",
            )
        )
    if (
        value.get("authoritative_base_commit", value.get("base_commit"))
        != "de3b691587da49896fd108cd3999127a2b5deb08"
    ):
        blockers.append(
            _block(
                "commit_manifest_base_invalid",
                "Commit manifest does not bind authoritative base commit.",
            )
        )
    if identity and value.get(
        "execution_commit_candidate", value.get("commit")
    ) != identity.get("exact_git_commit"):
        blockers.append(
            _block(
                "commit_manifest_execution_mismatch",
                "Commit manifest does not bind execution commit.",
            )
        )
    checks = _mapping(value.get("scope_checks"))
    legacy_valid = (
        value.get("scope_status") == "scoped_clean" and value.get("round") == "round18"
    )
    scoped_valid = (
        checks.get("all_authorized_paths_committed") is True
        and checks.get("unrelated_paths_committed") is False
        and checks.get("generated_runtime_files_committed") is False
        and checks.get("large_benchmark_data_committed") is False
        and checks.get("cache_files_committed") is False
        and checks.get("pending_parallel_workflows_incorporated", False) is False
    )
    if not (legacy_valid or scoped_valid):
        blockers.append(
            _block(
                "commit_manifest_scope_invalid",
                "Commit manifest does not prove source-only scoped commits.",
            )
        )
    return sorted_blockers(blockers), value


def _execution_blockers(
    repository_root: Path,
    execution_identity_path: Path | None,
    commit_manifest_path: Path | None,
    prefull_validation_path: Path | None,
    full_suite_execution_path: Path | None,
    test_inventory_path: Path | None,
    full_suite_junit_path: Path | None = None,
    release: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    identity_blockers, identity = _identity_blockers(
        execution_identity_path, repository_root, release
    )
    blockers = list(identity_blockers)
    prefull = _load_json(prefull_validation_path)
    if (
        prefull is None
        or prefull.get("schema_version") != PREFULL_SCHEMA_VERSION
        or prefull.get("status") != "passed"
        or prefull.get("full_suite_authorized") is not True
    ):
        blockers.append(
            _block("prefull_not_verified", "Pre-full validation did not pass.")
        )
    full = _junit_summary(full_suite_junit_path)
    if not (
        full.get("available")
        and full.get("failures") == 0
        and full.get("errors") == 0
        and full.get("skipped") == 0
    ):
        blockers.append(
            _block(
                "full_suite_junit_not_verified",
                "Full-suite JUnit report is absent or failing.",
            )
        )
    blockers += _full_suite_execution_blockers(
        _load_json(full_suite_execution_path), identity, full
    )
    inventory_blockers, _ = _inventory_blockers(test_inventory_path, full)
    blockers += inventory_blockers
    commit_blockers, _ = _commit_manifest_blockers(commit_manifest_path, identity)
    blockers += commit_blockers
    if git_dirty_status(repository_root) != "clean":
        blockers.append(
            _block("execution_worktree_dirty", "Formal validation worktree is dirty.")
        )
    return sorted_blockers(blockers)


def gate_state_for_blockers(blockers: Sequence[Mapping[str, object]]) -> str:
    """Map any failure to one of the user-authorized Round-18 Gate states."""

    if not blockers:
        return ACCEPTED_STATE
    codes = {str(item.get("code", "")).lower() for item in blockers}
    priority = (
        (
            "blocked_public_source_boundary",
            (
                "public_source",
                "controlled",
                "credential",
                ("ad" + "ni"),
                ("ai" + "bl"),
                "access_status",
            ),
        ),
        (
            "blocked_temporal_leakage",
            (
                "temporal",
                "timestamp",
                "future_evidence",
                "cutoff",
                "held_out_outcome_not_after",
            ),
        ),
        (
            "blocked_held_out_isolation",
            (
                "held_out",
                "bundle",
                "pre_cutoff_decision_changed",
                "execution_isolation",
            ),
        ),
        ("blocked_label_provenance", ("label", "outcome_", "narrative_claim")),
        (
            "blocked_independence_grouping",
            ("independence", "split", "accession", "partition"),
        ),
        ("blocked_memory_risk_disclosure", ("memory", "contamination")),
        (
            "blocked_replay_integrity",
            (
                "sequential",
                "reveal",
                "event",
                "lineage",
                "retry",
                "attempt",
                "duplicate_uid",
            ),
        ),
        (
            "blocked_metric_validation",
            ("metric", "schema", "required_system", "result_catalog"),
        ),
        (
            "blocked_abstention_semantics",
            (
                "abstention",
                "forced_answer",
                "unsupported_episode",
                "unknown_uncertainty",
                "conflict",
            ),
        ),
        (
            "blocked_claim_boundary",
            ("claim", "causal", "clinical", "prospective", "autonomous", "superiority"),
        ),
        ("blocked_gap_fabrication", ("gap", "evidence_fabric", "empirical")),
    )
    for state, markers in priority:
        if any(any(marker in code for marker in markers) for code in codes):
            return state
    return "blocked_execution_verification"


def _report(
    config: Mapping[str, object],
    blockers: Sequence[Mapping[str, object]],
    paths: Mapping[str, Path],
    *,
    require_execution: bool,
    summary: Mapping[str, object] | None = None,
    identity: Mapping[str, object] | None = None,
    focused: Mapping[str, object] | None = None,
    prefull: Mapping[str, object] | None = None,
    full: Mapping[str, object] | None = None,
    full_execution: Mapping[str, object] | None = None,
    inventory: Mapping[str, object] | None = None,
    commit_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    ordered = sorted_blockers(blockers)
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "validator_uid": VALIDATOR_UID,
        "stage_uid": STAGE_UID,
        "canonical_controller_uid": CANONICAL_CONTROLLER_UID,
        "status": "passed" if not ordered else "blocked",
        "gate_assessment": gate_state_for_blockers(ordered),
        "gate_state": gate_state_for_blockers(ordered),
        "require_execution": require_execution,
        "blocker_count": len(ordered),
        "blockers": ordered,
        "input_paths": {key: str(value) for key, value in paths.items()},
        "pilot_summary": dict(summary or {}),
        "focused_tests": dict(focused or {}),
        "prefull_validation": dict(prefull or {}),
        "execution_identity": dict(identity or {}),
        "full_suite": dict(full or {}),
        "full_suite_execution": dict(full_execution or {}),
        "test_inventory": dict(inventory or {}),
        "commit_manifest": dict(commit_manifest or {}),
        "claim_scope": dict(CLAIM_SCOPE),
        "frozen_gap_count_after_round18": 20,
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }


def build_round18_gate_record(
    *,
    authority: Mapping[str, object],
    assessment: str,
    blockers: list[dict[str, object]],
    input_hashes: Mapping[str, str],
    config_hash: str,
    commit: str,
    report: Mapping[str, object],
) -> dict[str, object]:
    if assessment not in {ACCEPTED_STATE, *BLOCKER_STATES}:
        raise ValueError(f"Invalid Round-18 Gate state: {assessment}")
    common = {
        "round_uid": STAGE_UID,
        "status": assessment,
        "overall_gate_status": "blocked" if blockers else "accepted",
        "validation_status": report.get("status"),
        "blocker_count": len(blockers),
        "blocker_classes": sorted(
            {
                gate_state_for_blockers([item]).removeprefix("blocked_")
                for item in blockers
            }
        ),
        "frozen_gap_count": 20,
        "frozen_gap_count_after_round18": 20,
        "frozen_gap_resolved_by_benchmark": 0,
        "execution_identity": report.get("execution_identity", {}),
        "evidence_fabric": {
            "materialization": "materialized_unsealed",
            "sealed": False,
        },
        "dt_core": {"status": "not_evaluated"},
        "claim_scope": dict(CLAIM_SCOPE),
        "final_held_out_partition": "frozen_untouched",
        "production_model_training": False,
    }
    if blockers:
        return {
            "schema_version": "neuronsifter-dt-round18-blocked-gate-assessment-v1",
            "gate_uid": STAGE_UID,
            "stage_uid": STAGE_UID,
            "writer_kind": "approved_validator_blocked_assessment",
            "validator_uid": VALIDATOR_UID,
            "canonical_controller_uid": CANONICAL_CONTROLLER_UID,
            "input_artifact_hashes": dict(sorted(input_hashes.items())),
            "config_hash": config_hash,
            "git_commit": commit,
            "decision": "blocked",
            "blockers": blockers,
            "lineage_parent": "round_17i_scitil_runtime_mainline_integration",
        } | common
    return (
        build_canonical_gate_decision(
            authority,
            gate_uid=STAGE_UID,
            stage_uid=STAGE_UID,
            controller_uid=CANONICAL_CONTROLLER_UID,
            validator_uid=VALIDATOR_UID,
            validator_report={
                "validator_uid": VALIDATOR_UID,
                "status": "passed",
                "assessment": assessment,
            },
            input_artifact_hashes=input_hashes,
            config_hash=config_hash,
            commit=commit,
            decision="accepted",
            blockers=[],
            lineage_parent="round_17i_scitil_runtime_mainline_integration",
        )
        | common
    )


def validate_round18_outputs(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    source_output_root: Path,
    parent_root: Path,
    focused_junit_path: Path | None,
    prefull_validation_path: Path | None,
    execution_identity_path: Path | None,
    full_suite_junit_path: Path | None,
    full_suite_execution_path: Path | None,
    test_inventory_path: Path | None,
    commit_manifest_path: Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    pilot_blockers, summary = validate_pilot_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_output_root,
        parent_root=parent_root,
    )
    release = (
        summary.get("release") if isinstance(summary.get("release"), Mapping) else None
    )
    identity_blockers, identity = _identity_blockers(
        execution_identity_path, repository_root, release
    )
    blockers = [*pilot_blockers, *identity_blockers]
    focused = _junit_summary(focused_junit_path)
    names = set(str(name) for name in focused.get("testcase_names", []))
    if not (
        focused.get("available")
        and focused.get("failures") == 0
        and focused.get("errors") == 0
        and focused.get("skipped") == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(names)
    ):
        blockers.append(
            _block(
                "focused_negative_tests_not_verified",
                "Mandatory Round-18 focused tests are not verified.",
                missing=sorted(MANDATORY_NEGATIVE_TEST_NAMES - names),
            )
        )
    prefull = _load_json(prefull_validation_path)
    if (
        prefull is None
        or prefull.get("schema_version") != PREFULL_SCHEMA_VERSION
        or prefull.get("status") != "passed"
        or prefull.get("full_suite_authorized") is not True
    ):
        blockers.append(
            _block("prefull_not_verified", "Exact-clean pre-full Gate did not pass.")
        )
    full = _junit_summary(full_suite_junit_path)
    full_execution = _load_json(full_suite_execution_path)
    inventory_blockers, inventory = _inventory_blockers(test_inventory_path, full)
    blockers += inventory_blockers
    blockers += _full_suite_execution_blockers(full_execution, identity, full)
    commit_blockers, commit_manifest = _commit_manifest_blockers(
        commit_manifest_path, identity
    )
    blockers += commit_blockers
    if not (
        full.get("available")
        and full.get("failures") == 0
        and full.get("errors") == 0
        and full.get("skipped") == 0
        and int(full.get("tests", 0)) >= ROUND18_PARENT_TEST_COUNT
    ):
        blockers.append(
            _block("full_suite_not_verified", "Full Libs/tests suite is not verified.")
        )
    if git_dirty_status(repository_root) != "clean":
        blockers.append(
            _block("execution_worktree_dirty", "Formal execution worktree is dirty.")
        )
    paths = _paths(config, source_output_root)
    report = _report(
        config,
        blockers,
        paths,
        require_execution=True,
        summary=summary,
        identity=identity,
        focused=focused,
        prefull=prefull,
        full=full,
        full_execution=full_execution,
        inventory=inventory,
        commit_manifest=commit_manifest,
    )
    input_hashes = {
        name: digest
        for name, digest in _mapping(summary.get("release_artifact_hashes")).items()
        if isinstance(digest, str)
    }
    for name, path in (
        ("focused_junit", focused_junit_path),
        ("prefull_validation", prefull_validation_path),
        ("execution_identity", execution_identity_path),
        ("full_suite_junit", full_suite_junit_path),
        ("full_suite_execution", full_suite_execution_path),
        ("test_inventory", test_inventory_path),
        ("commit_manifest", commit_manifest_path),
    ):
        if path and path.is_file():
            input_hashes[name] = sha256_file(path)
    if not input_hashes:
        input_hashes["missing_round18_evidence"] = "0" * 64
    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    gate = build_round18_gate_record(
        authority=authority,
        assessment=str(report["gate_assessment"]),
        blockers=_mapping_list(report.get("blockers")),
        input_hashes=input_hashes,
        config_hash=sha256_file(repository_root / ROUND18_CONFIG_PATH),
        commit=git_commit(repository_root),
        report=report,
    )
    return report, gate


def build_validation_report(
    *,
    repository_root: Path,
    config_path: Path,
    output_root: Path,
    require_execution: bool = False,
    execution_identity_path: Path | None = None,
    commit_manifest_path: Path | None = None,
    prefull_validation_path: Path | None = None,
    full_suite_execution_path: Path | None = None,
    test_inventory_path: Path | None = None,
    full_suite_junit_path: Path | None = None,
    focused_junit_path: Path | None = None,
    parent_root: Path | None = None,
) -> dict[str, object]:
    """Backward-compatible report builder; formal Gate creation is separate."""

    config, _ = _load_config(config_path)
    source_root = output_root.resolve()
    parent = (parent_root or source_root).resolve()
    pilot_blockers, summary = validate_pilot_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_root,
        parent_root=parent,
    )
    paths = _paths(config, source_root)
    if not require_execution:
        return _report(
            config,
            [
                *pilot_blockers,
                _block(
                    "execution_verification_missing",
                    "Round-18 acceptance requires exact-clean execution and full-suite evidence.",
                ),
            ],
            paths,
            require_execution=False,
            summary=summary,
        )
    release = (
        summary.get("release") if isinstance(summary.get("release"), Mapping) else None
    )
    identity_blockers, identity = _identity_blockers(
        execution_identity_path, repository_root, release
    )
    prefull = _load_json(prefull_validation_path)
    full = _junit_summary(full_suite_junit_path)
    full_execution = _load_json(full_suite_execution_path)
    inventory_blockers, inventory = _inventory_blockers(test_inventory_path, full)
    commit_blockers, commit_manifest = _commit_manifest_blockers(
        commit_manifest_path, identity
    )
    execution_blockers = _execution_blockers(
        repository_root,
        execution_identity_path,
        commit_manifest_path,
        prefull_validation_path,
        full_suite_execution_path,
        test_inventory_path,
        full_suite_junit_path,
        release,
    )
    focused = _junit_summary(focused_junit_path)
    blockers = [
        *pilot_blockers,
        *identity_blockers,
        *inventory_blockers,
        *commit_blockers,
        *execution_blockers,
    ]
    return _report(
        config,
        blockers,
        paths,
        require_execution=True,
        summary=summary,
        identity=identity,
        focused=focused,
        prefull=prefull,
        full=full,
        full_execution=full_execution,
        inventory=inventory,
        commit_manifest=commit_manifest,
    )


def build_round19_handoff(
    report: Mapping[str, object], gate: Mapping[str, object]
) -> dict[str, object]:
    if (
        report.get("status") != "passed"
        or gate.get("status") != ACCEPTED_STATE
        or gate.get("overall_gate_status") != "accepted"
    ):
        raise ValueError(
            "Round-19 handoff requires an accepted Round-18 canonical Gate."
        )
    summary = _mapping(report.get("pilot_summary"))
    counts = _mapping(summary.get("track_counts"))
    gap_rows = _mapping_list(summary.get("frozen_gap_map"))
    grouped_gaps: dict[str, list[str]] = defaultdict(list)
    for row in gap_rows:
        grouped_gaps[
            str(row.get("round18_impact", row.get("impact", "unknown")))
        ].append(str(row.get("gap_uid", "")))
    return {
        "schema_version": "neuronsifter-dt-round19-handoff-v1",
        "status": "ready_for_real_e2_e4_evidence_expansion_and_training_readiness",
        "parent_gate": gate["status"],
        "execution_commit": gate["git_commit"],
        "benchmark_scope": "retrospective_literature_grounded_computational_replay",
        "track_episode_counts": dict(sorted(counts.items())),
        "scitil_improvement_or_failure": "report_per_track_and_family_only_no_single_headline_claim",
        "calibration_and_abstention_weaknesses": "require_independent_family_level_expansion_beyond_qualification_pilot",
        "memory_contamination_risk": "explicit_residual_risk_not_eliminated",
        "frozen_gap_requirements": {
            key: sorted(values) for key, values in sorted(grouped_gaps.items())
        },
        "e2_requirements": sorted(
            grouped_gaps.get("requires_empirical_observation", [])
        ),
        "e4_requirements": sorted(grouped_gaps.get("requires_new_e4", [])),
        "action_classes_lacking_trajectory_coverage": "not_established_by_round18_contract_pilot",
        "underrepresented_support_domains": "require_independent_public_episode_expansion_before_training_gate",
        "minimum_world_action_training_gate_evidence": [
            "action_complete_leakage_safe_e4_trajectories",
            "qualified_support_domain_coverage",
            "teacher_uncertainty_and_provenance",
            "separately_frozen_training_contract",
        ],
        "minimum_evidence_fabric_sealing_evidence": [
            "independent_empirical_e2_validation",
            "action_complete_e4_coverage",
            "resolved_or_explicitly_preserved_conflicts",
            "separate_owner_authorization",
        ],
        "frozen_gap_count": 20,
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "claim_scope": dict(CLAIM_SCOPE),
    }


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Backward-compatible JSON writer for report-only callers."""

    registry_write_json(path, dict(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml")
    )
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--source-output-root")
    parser.add_argument("--parent-root")
    parser.add_argument("--output-root", default=str(REPO_ROOT))
    parser.add_argument("--focused-junit")
    parser.add_argument("--execution-identity")
    parser.add_argument("--prefull-validation")
    parser.add_argument("--full-suite-junit")
    parser.add_argument("--full-suite-execution")
    parser.add_argument("--test-inventory")
    parser.add_argument("--commit-manifest")
    parser.add_argument("--prefull-only", action="store_true")
    parser.add_argument("--require-execution", action="store_true")
    parser.add_argument(
        "--output", help="Optional legacy report path in addition to canonical audits."
    )
    args = parser.parse_args()

    repository_root = Path(args.repository_root).resolve()
    output_root = Path(args.output_root).resolve()
    source_output_root = (
        Path(args.source_output_root).resolve()
        if args.source_output_root
        else output_root
    )
    parent_root = (
        Path(args.parent_root).resolve() if args.parent_root else source_output_root
    )
    config = load_config(Path(args.config))
    pilot_blockers, pilot_summary = validate_pilot_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_output_root,
        parent_root=parent_root,
    )
    release = (
        pilot_summary.get("release")
        if isinstance(pilot_summary.get("release"), Mapping)
        else None
    )
    identity_path = (
        Path(args.execution_identity).resolve()
        if args.execution_identity
        else source_output_root / "Output/audits/dt_round18_execution_identity.json"
    )
    identity_blockers, identity = _identity_blockers(
        identity_path, repository_root, release
    )
    focused = _junit_summary(
        Path(args.focused_junit).resolve() if args.focused_junit else None
    )
    if args.prefull_only:
        prefull = build_prefull_report(
            pilot_blockers=pilot_blockers,
            pilot_summary=pilot_summary,
            focused=focused,
            identity=identity,
            identity_blockers=identity_blockers,
            execution_dirty_status=git_dirty_status(repository_root),
        )
        write_json(output_root / PREFULL_PATH, prefull)
        if args.output:
            write_json(Path(args.output).resolve(), prefull)
        print(
            f"round18_prefull_{prefull['status']} blockers={prefull['blocker_count']}"
        )
        return 0 if prefull["status"] == "passed" else 1

    if not args.require_execution:
        report = _report(
            config,
            [
                *pilot_blockers,
                _block(
                    "execution_verification_missing",
                    "Formal validation requires --require-execution.",
                ),
            ],
            _paths(config, source_output_root),
            require_execution=False,
            summary=pilot_summary,
        )
        if args.output:
            write_json(Path(args.output).resolve(), report)
        print(
            f"round18_validation_{report['status']} gate={report['gate_state']} blockers={report['blocker_count']}"
        )
        return 0 if report["status"] == "passed" else 1

    report, gate = validate_round18_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_output_root,
        parent_root=parent_root,
        focused_junit_path=Path(args.focused_junit).resolve()
        if args.focused_junit
        else None,
        prefull_validation_path=Path(args.prefull_validation).resolve()
        if args.prefull_validation
        else None,
        execution_identity_path=identity_path,
        full_suite_junit_path=Path(args.full_suite_junit).resolve()
        if args.full_suite_junit
        else None,
        full_suite_execution_path=Path(args.full_suite_execution).resolve()
        if args.full_suite_execution
        else None,
        test_inventory_path=Path(args.test_inventory).resolve()
        if args.test_inventory
        else None,
        commit_manifest_path=Path(args.commit_manifest).resolve()
        if args.commit_manifest
        else None,
    )
    write_json(output_root / VALIDATION_PATH, report)
    write_json(output_root / GATE_PATH, gate)
    if args.output:
        write_json(Path(args.output).resolve(), report)
    if gate["status"] == ACCEPTED_STATE:
        write_json(output_root / HANDOFF_PATH, build_round19_handoff(report, gate))
    print(f"{gate['status']} blockers={report['blocker_count']} gaps=20")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
