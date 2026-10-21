"""Validate Round-17 SciTiL pilot outputs and issue the canonical Gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_dt_round17_scitil_runtime import (  # noqa: E402
    sha256_file,
    validate_parent_state,
)
from Libs.utils.data.action_observation_contracts import (  # noqa: E402
    conflicts_preserved,
    validate_canonical_observation,
    validate_information_action,
    validate_observation_qc,
    validate_observation_request,
    validate_raw_tool_result,
    validate_world_action,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json  # noqa: E402
from Libs.utils.data.scitil_contracts import (  # noqa: E402
    AGENT_ROLES,
    CANONICAL_CONTROLLER_UID,
    ROUND17_VALIDATOR_UID,
    SCHEMA_VERSIONS,
    canonical_sha256,
    release_content_hash,
    round17_config,
    sorted_blockers,
    validate_agent_identity,
    validate_agent_proposal,
    validate_attempt_preservation,
    validate_belief_state,
    validate_budget_contracts,
    validate_dual_action_decision,
    validate_episode,
    validate_explanation,
    validate_frozen_gap_map,
    validate_h_gate,
    validate_hypothesis,
    validate_hypothesis_ledger_entry,
    validate_i_gate,
    validate_lineage_acyclic,
    validate_loop_event,
    validate_o_gate,
    validate_release_candidate,
    validate_round17_config,
    validate_tool_sandbox,
    validate_twin_debt,
    validate_unique_uids,
    validate_w_gate,
)
from Libs.utils.data.what_if_contracts import validate_what_if_request  # noqa: E402
from Libs.utils.exp.orchestration_controller import (  # noqa: E402
    authority_contract_report,
    build_canonical_gate_decision,
    git_commit,
    git_dirty_status,
    load_authority_contract,
    validator_is_approved,
)


AUTHORITY_PATH = "Libs/configs/validation/round11f_controller_authority.yaml"
ROUND17_CONFIG_PATH = "Libs/configs/data/round17_scitil_runtime.yaml"
VALIDATION_PATH = "Output/audits/dt_round17_validation.json"
GATE_PATH = "Output/audits/dt_round17_gate.json"
HANDOFF_PATH = "Output/audits/dt_round18_handoff.json"
PREFULL_PATH = "Output/audits/dt_round17_prefull_validation.json"

EXECUTION_IDENTITY_SCHEMA_VERSION = "neuronsifter-dt-round17-execution-identity-v1"
PREFULL_SCHEMA_VERSION = "neuronsifter-dt-round17-prefull-validation-v1"
FULL_SUITE_EXECUTION_SCHEMA_VERSION = "neuronsifter-dt-round17-full-suite-execution-v1"
TEST_INVENTORY_SCHEMA_VERSION = "neuronsifter-dt-round17-test-inventory-diff-v1"
COMMIT_MANIFEST_SCHEMA_VERSION = "neuronsifter-dt-round17-commit-manifest-v1"
PILOT_VALIDATION_SCHEMA_VERSION = "neuronsifter-dt-round17-pilot-validation-v1"
ROUND17_PARENT_TEST_COUNT = 517
ROUND17_CPU_FULL_SUITE_NODES = frozenset({"cpu03", "cpu04"})
ROUND17_STAGE_UID = "round_17_scitil_twin_gated_agent_loop"

ROUND17_SCHEMA_KEYS = frozenset(
    {
        "twin_belief_state",
        "agent_identity",
        "agent_proposal",
        "scientific_hypothesis",
        "hypothesis_gate_decision",
        "world_action_gate_decision",
        "information_action_gate_decision",
        "observation_gate_decision",
        "dual_action_decision",
        "twin_debt",
        "scientific_budget",
        "risk_budget",
        "agent_loop_event",
        "agent_loop_episode",
        "hypothesis_ledger_entry",
        "agent_explanation",
        "round17_scitil_release_candidate",
    }
)

MANDATORY_NEGATIVE_TEST_NAMES = frozenset(
    {
        "test_agent_direct_state_update_rejected_round17",
        "test_agent_gate_write_rejected_round17",
        "test_agent_arbitrary_shell_invocation_rejected_round17",
        "test_agent_h_gate_bypass_rejected_round17",
        "test_agent_w_gate_bypass_rejected_round17",
        "test_agent_i_gate_bypass_rejected_round17",
        "test_agent_o_gate_bypass_rejected_round17",
        "test_tool_result_direct_belief_state_rejected_round17",
        "test_model_rollout_empirical_observation_rejected_round17",
        "test_agent_evidence_qc_upgrade_rejected_round17",
        "test_agent_biological_pairing_creation_rejected_round17",
        "test_agent_conflict_removal_rejected_round17",
        "test_agent_frozen_gap_reasoning_closure_rejected_round17",
        "test_unknown_twin_debt_zero_rejected_round17",
        "test_high_twin_debt_world_action_rejected_round17",
        "test_budget_exceeded_silently_rejected_round17",
        "test_unsupported_domain_ignored_rejected_round17",
        "test_causal_effect_statement_rejected_round17",
        "test_clinical_recommendation_rejected_round17",
        "test_failed_tool_attempt_overwrite_rejected_round17",
        "test_blocked_loop_belief_update_rejected_round17",
        "test_noncanonical_assimilation_authority_rejected_round17",
    }
)

SCENARIO_TERMINAL_STATES = {
    "scenario_a_valid_restricted_hypothesis_loop": "hypothesis_retained",
    "scenario_b_information_action_loop": "uncertainty_only_updated",
    "scenario_c_direct_state_update_attack": "blocked_by_gate",
    "scenario_d_unsupported_world_action": "unsupported_domain",
    "scenario_e_conflict_preservation": "conflict_registered",
    "scenario_f_budget_exhaustion": "budget_exhausted",
    "scenario_g_unsafe_claim_request": "blocked_by_gate",
}

REQUIRED_EVENT_GATE_TYPES = {
    ("hypothesis_proposed", "hypothesis_gated"): frozenset({"H-Gate"}),
    ("action_candidates_generated", "action_gated"): frozenset({"W-Gate", "I-Gate"}),
    ("action_gated", "execution_authorized"): frozenset({"W-Gate", "I-Gate"}),
    ("assimilation_proposed", "assimilation_gated"): frozenset({"O-Gate"}),
    ("assimilation_gated", "belief_updated"): frozenset({"O-Gate"}),
    ("assimilation_gated", "explanation_generated"): frozenset({"O-Gate"}),
    ("assimilation_gated", "blocked"): frozenset({"O-Gate"}),
}


def validate_event_authority_binding(
    *,
    event: Mapping[str, object],
    gate_by_uid: Mapping[str, Mapping[str, object]],
    agent_by_uid: Mapping[str, Mapping[str, object]],
    proposal_by_uid: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    event_uid = event.get("event_uid")
    agent_uid = event.get("agent_uid")
    gate_uid = event.get("gate_uid")
    if agent_uid is not None and agent_uid not in agent_by_uid:
        blockers.append(
            {
                "code": "round17_event_authority_lineage_missing",
                "message": "A transition Agent binding must reference a registered Agent.",
                "event_uid": event_uid,
            }
        )
    if gate_uid is not None and gate_uid not in gate_by_uid:
        blockers.append(
            {
                "code": "round17_event_authority_lineage_missing",
                "message": "A transition Gate binding must reference a canonical Gate.",
                "event_uid": event_uid,
            }
        )
    transition = (str(event.get("prior_state", "")), str(event.get("next_state", "")))
    required_gate_types = REQUIRED_EVENT_GATE_TYPES.get(transition, frozenset())
    gate = gate_by_uid.get(str(gate_uid), {}) if gate_uid is not None else {}
    if required_gate_types and gate.get("gate_type") not in required_gate_types:
        blockers.append(
            {
                "code": "round17_event_gate_type_mismatch",
                "message": "The transition is not bound to the required Gate type.",
                "event_uid": event_uid,
                "required_gate_types": sorted(required_gate_types),
                "actual_gate_type": gate.get("gate_type"),
            }
        )
    if agent_uid is not None and gate.get("gate_type") in {"H-Gate", "W-Gate", "I-Gate"}:
        proposal = proposal_by_uid.get(str(gate.get("proposal_uid", "")), {})
        if proposal.get("agent_uid") != agent_uid:
            blockers.append(
                {
                    "code": "round17_event_agent_proposal_mismatch",
                    "message": "H/W/I transition Agent must own the proposal authorized by its Gate.",
                    "event_uid": event_uid,
                }
            )
    return blockers


def validate_positive_assessment_lineage(
    *,
    ledger: Mapping[str, object],
    falsifier_uids: set[str],
    proposals: Sequence[Mapping[str, object]],
    gates: Sequence[Mapping[str, object]],
    episodes: Sequence[Mapping[str, object]],
    events_by_episode: Mapping[str, Sequence[Mapping[str, object]]],
    explanations: Sequence[Mapping[str, object]],
    lineage_edges: set[tuple[str, str]],
) -> list[dict[str, object]]:
    if ledger.get("current_status") != "supported_model_conditionally":
        return []
    hypothesis_uid = str(ledger.get("hypothesis_uid", ""))
    matching_falsifier_proposals = [
        row
        for row in proposals
        if row.get("agent_role") == "FalsifierAgent"
        and row.get("proposal_type") == "falsification_test"
        and row.get("hypothesis_ref") == hypothesis_uid
    ]
    linked_falsifier_gate_uids = {
        str(gate.get("gate_uid", ""))
        for proposal in matching_falsifier_proposals
        for gate in gates
        if gate.get("proposal_uid") == proposal.get("proposal_uid")
        and (str(proposal.get("proposal_uid", "")), str(gate.get("gate_uid", ""))) in lineage_edges
    }
    gate_by_uid = {str(row.get("gate_uid", "")): row for row in gates}
    explanation_by_episode = {str(row.get("episode_uid", "")): row for row in explanations}
    test_scenarios = {str(item) for item in ledger.get("test_history", [])}
    positive_episodes = [row for row in episodes if str(row.get("scenario_uid", "")) in test_scenarios]
    linked_episode = False
    o_gate_episode = False
    explanation_bound = False
    for episode in positive_episodes:
        episode_uid = str(episode.get("episode_uid", ""))
        episode_events = list(events_by_episode.get(episode_uid, []))
        explanation_index = next(
            (index for index, event in enumerate(episode_events) if event.get("next_state") == "explanation_generated"),
            len(episode_events),
        )
        linked_episode_gate_uids = {
            str(event.get("gate_uid", ""))
            for index, event in enumerate(episode_events)
            if index < explanation_index and event.get("agent_uid") in falsifier_uids
        } & linked_falsifier_gate_uids
        episode_o_gate_uids = {
            str(event.get("gate_uid", ""))
            for event in episode_events
            if gate_by_uid.get(str(event.get("gate_uid", "")), {}).get("gate_type") == "O-Gate"
            and gate_by_uid.get(str(event.get("gate_uid", "")), {}).get("decision") == "no_update"
        }
        linked_episode = linked_episode or bool(linked_episode_gate_uids)
        o_gate_episode = o_gate_episode or bool(episode_o_gate_uids)
        explanation_refs = set(explanation_by_episode.get(episode_uid, {}).get("gate_decision_refs", []))
        explanation_bound = explanation_bound or (
            bool(linked_episode_gate_uids & explanation_refs) and bool(episode_o_gate_uids & explanation_refs)
        )
    if (
        ledger.get("falsifier_agent_ref") not in falsifier_uids
        or not matching_falsifier_proposals
        or not ledger.get("conflicting_evidence")
        or int(ledger.get("unresolved_conflict_count", 0)) < 1
        or "composite_score" in ledger
        or not linked_episode
        or not o_gate_episode
        or not explanation_bound
    ):
        return [
            {
                "code": "falsifier_or_conflict_dimension_missing",
                "message": "Positive model-conditional assessment requires episode-linked Falsifier participation, a no-update O-Gate, and separate conflict dimensions before explanation.",
                "hypothesis_uid": hypothesis_uid,
            }
        ]
    return []

REQUIRED_CATALOGS = (
    "agent_identity_catalog",
    "hypothesis_ledger",
    "agent_proposal_catalog",
    "gate_decision_catalog",
    "agent_loop_event_catalog",
    "agent_loop_episode_catalog",
    "twin_debt_catalog",
    "budget_decision_catalog",
    "explanation_catalog",
)


def _load_json(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _junit_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "testcase_names": [],
        }
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return {
            "available": False,
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "testcase_names": [],
        }
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    names = [case.get("name", "") for case in root.iter("testcase")]
    return {
        "available": True,
        "tests": sum(int(suite.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.get("skipped", 0)) for suite in suites),
        "testcase_names": sorted(names),
    }


def _paths(config: Mapping[str, object], root: Path) -> dict[str, Path]:
    return {
        name: root / str(path)
        for name, path in _mapping(round17_config(config).get("output_paths")).items()
    }


def _object_uid(value: Mapping[str, object], fallback: str) -> str:
    for field in (
        "belief_state_uid",
        "agent_uid",
        "proposal_uid",
        "hypothesis_uid",
        "gate_uid",
        "decision_uid",
        "debt_uid",
        "budget_uid",
        "risk_budget_uid",
        "event_uid",
        "episode_uid",
        "explanation_uid",
        "release_uid",
    ):
        if value.get(field):
            return str(value[field])
    return fallback


def _schema_registry(
    config: Mapping[str, object], repository_root: Path
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    schemas = _mapping(config.get("schemas"))
    raw_registry = _mapping(schemas.get("round17_scitil_schema_registry"))
    blockers: list[dict[str, object]] = []
    if set(raw_registry) != set(ROUND17_SCHEMA_KEYS):
        blockers.append(
            {
                "code": "round17_schema_registry_incomplete",
                "message": "Round-17 must use exactly the 17 canonical registered schemas.",
                "missing": sorted(ROUND17_SCHEMA_KEYS - set(raw_registry)),
                "unexpected": sorted(set(raw_registry) - ROUND17_SCHEMA_KEYS),
            }
        )
    registry: dict[str, dict[str, object]] = {}
    for name, raw_entry in raw_registry.items():
        if not isinstance(raw_entry, Mapping):
            blockers.append(
                {
                    "code": "round17_schema_registry_entry_invalid",
                    "message": "Round-17 schema registry entries must be mappings.",
                    "schema": name,
                }
            )
            continue
        entry = dict(raw_entry)
        registry[name] = entry
        schema_path = repository_root / str(entry.get("path", ""))
        if not schema_path.is_file():
            blockers.append(
                {
                    "code": "round17_schema_missing",
                    "message": "A registered Round-17 schema file is missing.",
                    "schema": name,
                    "path": str(entry.get("path", "")),
                }
            )
            continue
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
        except (OSError, json.JSONDecodeError, SchemaError) as error:
            blockers.append(
                {
                    "code": "round17_schema_invalid",
                    "message": str(error),
                    "schema": name,
                }
            )
            continue
        properties = _mapping(schema.get("properties"))
        version = _mapping(properties.get("schema_version")).get("const")
        if version != entry.get("schema_version"):
            blockers.append(
                {
                    "code": "round17_schema_version_registry_mismatch",
                    "message": "Schema version constant differs from the canonical registry.",
                    "schema": name,
                }
            )
    return registry, sorted_blockers(blockers)


def _schema_validate(
    *,
    repository_root: Path,
    schema_entry: Mapping[str, object],
    instance: Mapping[str, object],
    object_uid: str,
) -> list[dict[str, object]]:
    path = repository_root / str(schema_entry.get("path", ""))
    if not path.is_file():
        return [
            {
                "code": "round17_schema_missing",
                "message": "Registered schema file is missing.",
                "object_uid": object_uid,
                "schema_path": str(schema_entry.get("path", "")),
            }
        ]
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
    except (OSError, json.JSONDecodeError, SchemaError) as error:
        return [
            {
                "code": "round17_schema_invalid",
                "message": str(error),
                "object_uid": object_uid,
                "schema_path": str(schema_entry.get("path", "")),
            }
        ]
    return [
        {
            "code": "round17_schema_validation_failed",
            "message": error.message,
            "object_uid": object_uid,
            "schema_path": str(path.relative_to(repository_root)),
            "instance_path": "/".join(str(item) for item in error.path),
        }
        for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.path))
    ]


def _catalog_objects(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not path.is_file():
        return [], [
            {
                "code": "round17_pilot_catalog_missing",
                "message": "Required Round-17 pilot catalog is missing.",
                "catalog": path.name,
            }
        ]
    rows = read_tsv(path)
    if not rows:
        return [], [
            {
                "code": "round17_pilot_catalog_empty",
                "message": "Required Round-17 pilot catalog is empty.",
                "catalog": path.name,
            }
        ]
    blockers: list[dict[str, object]] = []
    objects: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        try:
            value = json.loads(str(row.get("object_json", "")))
        except json.JSONDecodeError as error:
            blockers.append(
                {
                    "code": "round17_catalog_object_invalid",
                    "message": str(error),
                    "catalog": path.name,
                    "row": index,
                }
            )
            continue
        if not isinstance(value, Mapping):
            blockers.append(
                {
                    "code": "round17_catalog_object_invalid",
                    "message": "Catalog object_json must contain an object.",
                    "catalog": path.name,
                    "row": index,
                }
            )
            continue
        objects.append(dict(value))
    return objects, blockers


def _authority_blockers(authority: Mapping[str, object]) -> list[dict[str, object]]:
    blockers = [dict(row) for row in authority_contract_report(authority).get("blockers", [])]
    if not validator_is_approved(authority, ROUND17_VALIDATOR_UID, ROUND17_STAGE_UID):
        blockers.append(
            {
                "code": "round17_validator_not_approved",
                "message": "Round-17 validator is not approved for the canonical stage.",
            }
        )
    runtime = _mapping(authority.get("scitil_runtime_authority"))
    expected_writers = {
        "llm_agent",
        "scientist_agent",
        "falsifier_agent",
        "experiment_agent",
        "evidence_agent",
    }
    if (
        runtime.get("controller_uid") != CANONICAL_CONTROLLER_UID
        or runtime.get("approved_validator_uid") != ROUND17_VALIDATOR_UID
        or set(runtime.get("gate_types", [])) != {"H-Gate", "W-Gate", "I-Gate", "O-Gate"}
        or set(runtime.get("agent_roles", [])) != set(AGENT_ROLES)
        or not expected_writers.issubset(set(runtime.get("disallowed_gate_writers", [])))
    ):
        blockers.append(
            {
                "code": "round17_gate_authority_invalid",
                "message": "SciTiL controller, validator, Gate, role, or forbidden-writer authority is incomplete.",
            }
        )
    assimilation = _mapping(authority.get("assimilation_authority"))
    if (
        assimilation.get("controller_uid") != CANONICAL_CONTROLLER_UID
        or ROUND17_VALIDATOR_UID not in assimilation.get("approved_validators", [])
        or "llm_agent" not in assimilation.get("disallowed_writer_kinds", [])
    ):
        blockers.append(
            {
                "code": "round17_assimilation_authority_invalid",
                "message": "Round-17 O-Gate assimilation authority is not canonical.",
            }
        )
    return sorted_blockers(blockers)


def _identity_blockers(
    identity_path: Path | None,
    repository_root: Path,
    release: Mapping[str, object] | None,
    graph: Mapping[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    identity = _load_json(identity_path)
    if identity is None:
        return (
            [
                {
                    "code": "execution_identity_missing",
                    "message": "Exact-clean Round-17 execution identity is missing.",
                }
            ],
            None,
        )
    blockers: list[dict[str, object]] = []
    if identity.get("schema_version") != EXECUTION_IDENTITY_SCHEMA_VERSION:
        blockers.append(
            {
                "code": "execution_identity_schema_invalid",
                "message": "Round-17 execution identity schema is invalid.",
            }
        )
    commit = git_commit(repository_root)
    if identity.get("exact_git_commit") != commit:
        blockers.append(
            {
                "code": "execution_commit_mismatch",
                "message": "Execution commit differs from the exact code worktree.",
            }
        )
    if identity.get("dirty_status") != "clean" or identity.get("reproduction_relation") != "exact_clean_commit":
        blockers.append(
            {
                "code": "execution_identity_not_exact_clean",
                "message": "Round-17 execution identity is not exact-clean.",
            }
        )
    if identity.get("code_worktree_read_only") is not True:
        blockers.append(
            {
                "code": "execution_code_not_read_only",
                "message": "Round-17 execution source must be read-only.",
            }
        )
    if identity.get("writable_runtime_output") is not True or identity.get("source_output_isolated") is not True:
        blockers.append(
            {
                "code": "execution_output_not_isolated",
                "message": "Round-17 runtime output must be isolated and writable.",
            }
        )
    if identity.get("node") not in ROUND17_CPU_FULL_SUITE_NODES or identity.get("partition") != "cpu":
        blockers.append(
            {
                "code": "execution_cpu_placement_invalid",
                "message": "Round-17 CPU validation must run on cpu03 or cpu04 in the cpu partition.",
            }
        )
    if not str(identity.get("job_name", "")).startswith("NrnSift_DT_R17_"):
        blockers.append(
            {
                "code": "execution_job_name_invalid",
                "message": "Round-17 job name must use the NrnSift_DT_R17_ prefix.",
            }
        )
    for field in ("python_pycache_prefix", "pytest_cache"):
        raw_path = str(identity.get(field, ""))
        if not raw_path or _path_is_within(Path(raw_path), repository_root):
            blockers.append(
                {
                    "code": "execution_cache_not_isolated",
                    "message": "Python and pytest caches must remain outside the source worktree.",
                    "field": field,
                }
            )
    if identity.get("index_lock_absent") is not True or identity.get("active_git_writer_absent") is not True:
        blockers.append(
            {
                "code": "execution_git_writer_present",
                "message": "Exact-clean execution cannot overlap an active Git writer or index lock.",
            }
        )
    if identity.get("evidence_fabric_sealed") is not False or identity.get("dt_core_status") != "not_evaluated":
        blockers.append(
            {
                "code": "execution_scope_boundary_invalid",
                "message": "Execution identity changed Evidence Fabric or DT_CORE boundaries.",
            }
        )
    if release:
        for identity_field, release_field in (
            ("config_hash", "config_hash"),
            ("schema_registry_hash", "schema_registry_hash"),
            ("input_bundle_hash", "input_bundle_hash"),
            ("environment_hash", "environment_hash"),
        ):
            if identity.get(identity_field) != release.get(release_field):
                blockers.append(
                    {
                        "code": "execution_materialization_identity_mismatch",
                        "message": "Execution identity differs from Round-17 materialization.",
                        "field": identity_field,
                    }
                )
    if graph and identity.get("exact_git_commit") != graph.get("code_commit"):
        blockers.append(
            {
                "code": "execution_graph_commit_mismatch",
                "message": "Round-17 graph was not produced by the exact execution commit.",
            }
        )
    return sorted_blockers(blockers), identity


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_release_path(source_output_root: Path, pilot: Mapping[str, object]) -> Path | None:
    raw_path = str(pilot.get("release_path", ""))
    candidate = Path(raw_path)
    if not raw_path or candidate.is_absolute():
        return None
    resolved = (source_output_root / candidate).resolve()
    return resolved if _path_is_within(resolved, source_output_root) else None


def _objects_equal_member(value: Mapping[str, object], candidates: Sequence[Mapping[str, object]]) -> bool:
    target = canonical_sha256(value)
    return any(canonical_sha256(candidate) == target for candidate in candidates)


def validate_pilot_outputs(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    source_output_root: Path,
    parent_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    blockers = validate_round17_config(config)
    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    blockers += _authority_blockers(authority)
    try:
        parent = validate_parent_state(config, parent_root)
    except (FileNotFoundError, ValueError, TypeError, KeyError) as error:
        return (
            sorted_blockers(
                blockers
                + [
                    {
                        "code": "round16_parent_invalid",
                        "message": str(error),
                    }
                ]
            ),
            {},
        )

    paths = _paths(config, source_output_root)
    pilot = _load_json(paths.get("pilot_validation"))
    graph = _load_json(paths.get("graph"))
    if pilot is None or graph is None:
        return (
            sorted_blockers(
                blockers
                + [
                    {
                        "code": "round17_pilot_missing",
                        "message": "Round-17 pilot validation or graph is missing.",
                    }
                ]
            ),
            {},
        )

    release_path = _resolve_release_path(source_output_root, pilot)
    release = _load_json(release_path)
    if release_path is None or release is None:
        blockers.append(
            {
                "code": "round17_release_missing",
                "message": "Round-17 content-addressed successor release is missing or escaped its output root.",
            }
        )

    objects = _mapping(graph.get("objects"))
    belief_states = _mapping_list(objects.get("belief_states"))
    agents = _mapping_list(objects.get("agents"))
    hypotheses = _mapping_list(objects.get("hypotheses"))
    proposals = _mapping_list(objects.get("proposals"))
    gates = _mapping_list(objects.get("gates"))
    world_actions = _mapping_list(objects.get("world_actions"))
    what_if_requests = _mapping_list(objects.get("what_if_requests"))
    information_actions = _mapping_list(objects.get("information_actions"))
    observation_requests = _mapping_list(objects.get("observation_requests"))
    observation_chain = _mapping(objects.get("observation_chain"))
    dual_actions = _mapping_list(objects.get("dual_action_decisions"))
    twin_debts = _mapping_list(objects.get("twin_debts"))
    scientific_budget = _mapping(objects.get("scientific_budget"))
    risk_budget = _mapping(objects.get("risk_budget"))
    budget_decisions = _mapping_list(objects.get("budget_decisions"))
    sandboxes = _mapping_list(objects.get("sandbox_contracts"))
    execution_attempts = _mapping_list(objects.get("execution_attempts"))
    ledgers = _mapping_list(objects.get("hypothesis_ledger_entries"))
    events = _mapping_list(objects.get("events"))
    episodes = _mapping_list(objects.get("episodes"))
    explanations = _mapping_list(objects.get("explanations"))
    negative_cases = _mapping(objects.get("negative_cases"))

    catalog_objects: dict[str, list[dict[str, object]]] = {}
    catalog_rows: dict[str, list[dict[str, str]]] = {}
    for name in REQUIRED_CATALOGS:
        path = paths.get(name)
        if path is None:
            blockers.append(
                {
                    "code": "round17_pilot_catalog_unconfigured",
                    "message": "Required Round-17 catalog path is not configured.",
                    "catalog": name,
                }
            )
            continue
        values, catalog_blockers = _catalog_objects(path)
        blockers += catalog_blockers
        catalog_objects[name] = values
        catalog_rows[name] = read_tsv(path) if path.is_file() else []

    proposal_catalog = catalog_rows.get("agent_proposal_catalog", [])
    proposal_catalog_pairs = list(
        zip(
            proposal_catalog,
            catalog_objects.get("agent_proposal_catalog", []),
            strict=False,
        )
    )
    catalog_proposals = {
        str(value.get("proposal_uid", "")): value
        for _, value in proposal_catalog_pairs
    }
    proposal_decisions = {
        str(value.get("proposal_uid", "")): str(row.get("decision", ""))
        for row, value in proposal_catalog_pairs
    }
    for row, value in proposal_catalog_pairs:
        if row.get("record_uid") != value.get("proposal_uid"):
            blockers.append(
                {
                    "code": "round17_proposal_catalog_identity_mismatch",
                    "message": "AgentProposal catalog rows must use proposal_uid as record_uid.",
                    "proposal_uid": value.get("proposal_uid"),
                }
            )
    rejected_proposal_uids = {
        uid for uid, decision in proposal_decisions.items() if decision == "rejected_contract"
    }

    if len(belief_states) != 2:
        blockers.append(
            {
                "code": "round17_belief_state_count_invalid",
                "message": "The deterministic pilot requires one initial and one controller-authorized successor belief state.",
            }
        )
    initial_states = [row for row in belief_states if row.get("parent_belief_state_uid") is None]
    if len(initial_states) != 1:
        blockers.append(
            {
                "code": "round17_initial_belief_identity_invalid",
                "message": "Exactly one qualified initial TwinBeliefState is required.",
            }
        )
    for state in belief_states:
        blockers += validate_belief_state(
            state,
            allow_initial=state.get("parent_belief_state_uid") is None,
        )

    agent_by_uid = {str(row.get("agent_uid", "")): row for row in agents}
    if len(agents) != 4 or {str(row.get("agent_role", "")) for row in agents} != set(AGENT_ROLES):
        blockers.append(
            {
                "code": "round17_agent_roles_incomplete",
                "message": "Round-17 requires exactly the four privilege-separated Agent roles.",
            }
        )
    for agent in agents:
        blockers += validate_agent_identity(agent)

    known_beliefs = {str(row.get("belief_state_uid", "")) for row in belief_states}
    for proposal in proposals:
        uid = str(proposal.get("proposal_uid", ""))
        agent = agent_by_uid.get(str(proposal.get("agent_uid", "")), {})
        proposal_blockers = validate_agent_proposal(
            proposal,
            agent=agent,
            known_belief_states=known_beliefs,
        )
        if uid in rejected_proposal_uids:
            if not proposal_blockers and _mapping(proposal.get("provenance")).get("execution_authorized") is not False:
                blockers.append(
                    {
                        "code": "round17_rejected_proposal_did_not_fail",
                        "message": "Rejected adversarial proposal did not fail closed.",
                        "proposal_uid": uid,
                    }
                )
        else:
            blockers += proposal_blockers
        if uid not in catalog_proposals or canonical_sha256(catalog_proposals[uid]) != canonical_sha256(proposal):
            blockers.append(
                {
                    "code": "round17_proposal_catalog_mismatch",
                    "message": "AgentProposal graph and catalog representations differ.",
                    "proposal_uid": uid,
                }
            )
        world_ref = proposal.get("world_action_ref")
        information_ref = proposal.get("information_action_ref")
        if world_ref and information_ref:
            blockers.append(
                {
                    "code": "world_information_action_boundary_violation",
                    "message": "An AgentProposal cannot collapse WorldAction and InformationAction identities.",
                    "proposal_uid": uid,
                }
            )

    for hypothesis in hypotheses:
        blockers += validate_hypothesis(hypothesis)
    for action in world_actions:
        blockers += validate_world_action(action)
    for request in what_if_requests:
        blockers += validate_what_if_request(request)
    for action in information_actions:
        blockers += validate_information_action(action)
    for request in observation_requests:
        blockers += validate_observation_request(request)

    parent_round16_objects = _mapping(_mapping(parent["round16_graph"]).get("objects"))
    parent_round13_graph = _mapping(parent["round13_graph"])
    if any(
        not _objects_equal_member(row, [_mapping(parent_round16_objects.get("action_binding")) .get("world_action")])
        for row in world_actions
    ):
        blockers.append(
            {
                "code": "round17_world_action_not_parent_governed",
                "message": "Pilot WorldActions must be exact accepted Round-16 governed artifacts.",
            }
        )
    if any(
        not _objects_equal_member(row, [_mapping(parent_round16_objects.get("what_if_request"))])
        for row in what_if_requests
    ):
        blockers.append(
            {
                "code": "round17_what_if_not_parent_governed",
                "message": "Pilot WhatIfRequests must be exact accepted Round-16 governed artifacts.",
            }
        )
    if any(
        not _objects_equal_member(row, _mapping_list(parent_round13_graph.get("information_actions")))
        for row in information_actions
    ):
        blockers.append(
            {
                "code": "round17_information_action_not_parent_governed",
                "message": "Pilot InformationActions must be exact accepted Round-13 governed artifacts.",
            }
        )
    if any(
        not _objects_equal_member(row, _mapping_list(parent_round13_graph.get("observation_requests")))
        for row in observation_requests
    ):
        blockers.append(
            {
                "code": "round17_observation_request_not_parent_governed",
                "message": "Pilot ObservationRequests must be exact accepted Round-13 governed artifacts.",
            }
        )

    proposal_by_uid = {str(row.get("proposal_uid", "")): row for row in proposals}
    hypothesis_by_uid = {str(row.get("hypothesis_uid", "")): row for row in hypotheses}
    debt_by_uid = {str(row.get("debt_uid", "")): row for row in twin_debts}
    budget_by_proposal = {
        str(row.get("proposal_uid", "")): row for row in budget_decisions
    }
    belief_by_uid = {str(row.get("belief_state_uid", "")): row for row in belief_states}
    information_by_uid = {
        str(row.get("information_action_uid", "")): row for row in information_actions
    }
    world_by_uid = {str(row.get("action_uid", "")): row for row in world_actions}
    what_if_by_uid = {str(row.get("request_uid", "")): row for row in what_if_requests}
    assimilation_proposal = _mapping(observation_chain.get("assimilation_proposal"))
    assimilation_decision = _mapping(observation_chain.get("assimilation_decision"))
    tool_request = _mapping(observation_chain.get("tool_request"))
    tool_capability_match = _mapping(observation_chain.get("tool_capability_match"))
    tool_availability = _mapping(observation_chain.get("tool_availability"))

    for gate in gates:
        proposal = proposal_by_uid.get(str(gate.get("proposal_uid", "")), {})
        gate_type = str(gate.get("gate_type", ""))
        if gate_type == "H-Gate":
            hypothesis = hypothesis_by_uid.get(str(proposal.get("hypothesis_ref", "")), {})
            blockers += validate_h_gate(gate, proposal=proposal, hypothesis=hypothesis)
        elif gate_type == "W-Gate":
            belief = belief_by_uid.get(str(proposal.get("belief_state_ref", "")), {})
            debt = debt_by_uid.get(str(belief.get("twin_debt_ref", "")), {})
            world_action = world_by_uid.get(str(proposal.get("world_action_ref", "")))
            what_if_request = what_if_by_uid.get(str(proposal.get("what_if_request_ref", "")))
            if world_action is None and str(gate.get("decision", "")).startswith("authorized_"):
                world_action = world_actions[0] if world_actions else None
            blockers += validate_w_gate(
                gate,
                proposal=proposal,
                world_action=world_action,
                what_if_request=what_if_request,
                twin_debt=debt,
                budget_decision=budget_by_proposal.get(str(proposal.get("proposal_uid", "")), {}),
            )
        elif gate_type == "I-Gate":
            information_action = information_by_uid.get(str(proposal.get("information_action_ref", "")))
            if information_action is None:
                information_action = information_actions[0] if information_actions else {}
            blockers += validate_i_gate(
                gate,
                proposal=proposal,
                information_action=information_action,
                tool_request=tool_request,
                allowlisted_adapters=set(_mapping(round17_config(config).get("adapter_allowlist"))),
                budget_decision=budget_by_proposal.get(str(proposal.get("proposal_uid", "")), {}),
                tool_capability_match=tool_capability_match,
                tool_availability_decision=tool_availability,
            )
        elif gate_type == "O-Gate":
            approved = str(gate.get("decision", "")).startswith("approved_")
            blockers += validate_o_gate(
                gate,
                proposal=proposal,
                assimilation_proposal=assimilation_proposal if approved else {},
                assimilation_decision=assimilation_decision if approved else {},
            )
        else:
            blockers.append(
                {
                    "code": "unknown_gate_type",
                    "message": "Round-17 Gate type is not canonical.",
                    "gate_uid": gate.get("gate_uid"),
                }
            )

    gate_types = {str(row.get("gate_type", "")) for row in gates}
    required_gate_decisions = {
        "admitted_for_testing",
        "authorized_restricted_rollout",
        "authorized_restricted_information_action",
        "approved_uncertainty_only",
        "no_update",
        "blocked_incomplete_action",
        "authorized_evidence_query",
        "blocked_budget",
        "revision_required",
    }
    if gate_types != {"H-Gate", "W-Gate", "I-Gate", "O-Gate"} or not required_gate_decisions.issubset(
        {str(row.get("decision", "")) for row in gates}
    ):
        blockers.append(
            {
                "code": "round17_four_gate_paths_incomplete",
                "message": "Pilot must exercise distinct H/W/I/O Gates and all required authorization or rejection paths.",
            }
        )

    approved_o_gates = {
        str(row.get("gate_uid", "")): row
        for row in gates
        if row.get("gate_type") == "O-Gate" and str(row.get("decision", "")).startswith("approved_")
    }
    for state in belief_states:
        if state.get("parent_belief_state_uid") is None:
            continue
        gate = approved_o_gates.get(str(state.get("creation_decision_ref", "")))
        parent_state = belief_by_uid.get(str(state.get("parent_belief_state_uid", "")))
        if gate is None or parent_state is None:
            blockers.append(
                {
                    "code": "belief_state_bypasses_o_gate",
                    "message": "Every successor belief state requires a valid parent and approved O-Gate.",
                    "belief_state_uid": state.get("belief_state_uid"),
                }
            )
            continue
        if assimilation_decision.get("resulting_state_ref") != state.get("belief_state_uid"):
            blockers.append(
                {
                    "code": "assimilation_resulting_state_identity_mismatch",
                    "message": "AssimilationDecision must bind the exact successor TwinBeliefState UID.",
                    "belief_state_uid": state.get("belief_state_uid"),
                }
            )
        if gate.get("decision") == "approved_uncertainty_only" and any(
            state.get(field) != parent_state.get(field)
            for field in (
                "state_posterior_ref",
                "parameter_posterior_ref",
                "model_family_posterior_ref",
                "evidence_snapshot_ref",
                "constraint_state_ref",
                "support_domain_ref",
            )
        ):
            blockers.append(
                {
                    "code": "uncertainty_only_scope_overwritten",
                    "message": "An uncertainty-only O-Gate cannot overwrite state or parameter posterior identities.",
                    "belief_state_uid": state.get("belief_state_uid"),
                }
            )

    for debt in twin_debts:
        blockers += validate_twin_debt(debt)
    if len(dual_actions) != len(twin_debts):
        blockers.append(
            {
                "code": "dual_action_debt_pairing_incomplete",
                "message": "Every qualification DualActionDecision must bind an explicit Twin Debt vector.",
            }
        )
    for decision, debt in zip(dual_actions, twin_debts, strict=False):
        blockers += validate_dual_action_decision(decision, twin_debt=debt)
    for decision in budget_decisions:
        blockers += validate_budget_contracts(scientific_budget, risk_budget, decision)
    allowlisted_adapters = set(_mapping(round17_config(config).get("adapter_allowlist")))
    for sandbox in sandboxes:
        blockers += validate_tool_sandbox(sandbox, allowlisted_adapters=allowlisted_adapters)
    blockers += validate_attempt_preservation(execution_attempts)
    if len(execution_attempts) != 1:
        blockers.append(
            {
                "code": "round17_execution_attempt_ledger_invalid",
                "message": "The read-only pilot must preserve exactly one governed adapter attempt.",
            }
        )
    if any(
        attempt.get("adapter_uid") != tool_request.get("tool_uid")
        for attempt in execution_attempts
    ):
        blockers.append(
            {
                "code": "tool_attempt_adapter_identity_mismatch",
                "message": "Every preserved tool attempt must bind the I-Gate authorized adapter.",
            }
        )

    raw_directional = _mapping(observation_chain.get("raw_directional"))
    raw_conflict = _mapping(observation_chain.get("raw_conflict"))
    observation_directional = _mapping(observation_chain.get("observation_directional"))
    observation_conflict = _mapping(observation_chain.get("observation_conflict"))
    observation_qc = _mapping(observation_chain.get("qc_directional"))
    blockers += validate_raw_tool_result(raw_directional)
    blockers += validate_raw_tool_result(raw_conflict)
    blockers += validate_canonical_observation(observation_directional)
    blockers += validate_canonical_observation(observation_conflict)
    blockers += validate_observation_qc(observation_qc)
    blockers += conflicts_preserved(raw_conflict, observation_conflict)
    parent_raw_results = _mapping_list(parent_round13_graph.get("raw_tool_results"))
    parent_observations = _mapping_list(parent_round13_graph.get("canonical_observations"))
    parent_qc = _mapping_list(parent_round13_graph.get("observation_qc"))
    parent_tool_requests = _mapping_list(parent_round13_graph.get("tool_requests"))
    parent_tool_matches = _mapping_list(parent_round13_graph.get("tool_capability_matches"))
    parent_tool_availability = _mapping_list(parent_round13_graph.get("tool_availability_decisions"))
    for row, candidates, label in (
        (raw_directional, parent_raw_results, "raw_directional"),
        (raw_conflict, parent_raw_results, "raw_conflict"),
        (observation_directional, parent_observations, "observation_directional"),
        (observation_conflict, parent_observations, "observation_conflict"),
        (observation_qc, parent_qc, "qc_directional"),
    ):
        if not _objects_equal_member(row, candidates):
            blockers.append(
                {
                    "code": "round17_observation_chain_not_parent_governed",
                    "message": "Round-17 observation chain must reuse exact accepted Round-13 artifacts.",
                    "artifact": label,
                }
            )
    for row, candidates, label in (
        (tool_request, parent_tool_requests, "tool_request"),
        (tool_capability_match, parent_tool_matches, "tool_capability_match"),
        (tool_availability, parent_tool_availability, "tool_availability"),
    ):
        if not _objects_equal_member(row, candidates):
            blockers.append(
                {
                    "code": "round17_tool_authorization_not_parent_governed",
                    "message": "I-Gate tool authorization must reuse exact accepted Round-13 evidence.",
                    "artifact": label,
                }
            )
    if any(
        sandbox.get("adapter_uid") != tool_request.get("tool_uid")
        or sandbox.get("output_hash") not in set(_mapping(raw_directional.get("raw_artifact_hashes")).values())
        for sandbox in sandboxes
    ):
        blockers.append(
            {
                "code": "round17_tool_sandbox_lineage_mismatch",
                "message": "Sandbox adapter and output hash must bind the authorized request and governed result.",
            }
        )

    events_by_episode: dict[str, list[dict[str, object]]] = {}
    gate_by_uid = {str(row.get("gate_uid", "")): row for row in gates}
    agent_by_uid = {str(row.get("agent_uid", "")): row for row in agents}
    for event in events:
        blockers += validate_loop_event(event)
        blockers += validate_event_authority_binding(
            event=event,
            gate_by_uid=gate_by_uid,
            agent_by_uid=agent_by_uid,
            proposal_by_uid=proposal_by_uid,
        )
        events_by_episode.setdefault(str(event.get("episode_uid", "")), []).append(event)
    for episode in episodes:
        blockers += validate_episode(
            episode,
            events=events_by_episode.get(str(episode.get("episode_uid", "")), []),
        )
    scenario_states = {
        str(row.get("scenario_uid", "")): str(row.get("terminal_state", ""))
        for row in episodes
    }
    if scenario_states != SCENARIO_TERMINAL_STATES:
        blockers.append(
            {
                "code": "round17_pilot_scenarios_incomplete",
                "message": "Deterministic pilot scenarios A-G or their scientific terminal states changed.",
            }
        )

    falsifier_uids = {
        str(row.get("agent_uid", ""))
        for row in agents
        if row.get("agent_role") == "FalsifierAgent"
    }
    falsifier_proposals = [
        row
        for row in proposals
        if row.get("agent_role") == "FalsifierAgent" and row.get("proposal_type") == "falsification_test"
    ]
    raw_lineage_edges = {
        (str(edge[0]), str(edge[1]))
        for edge in graph.get("lineage_edges", [])
        if isinstance(edge, list) and len(edge) == 2
    }
    for ledger in ledgers:
        blockers += validate_hypothesis_ledger_entry(ledger)
        blockers += validate_positive_assessment_lineage(
            ledger=ledger,
            falsifier_uids=falsifier_uids,
            proposals=proposals,
            gates=gates,
            episodes=episodes,
            events_by_episode=events_by_episode,
            explanations=explanations,
            lineage_edges=raw_lineage_edges,
        )
    for explanation in explanations:
        blockers += validate_explanation(explanation)

    gap_path = paths.get("frozen_gap_map")
    gap_rows = read_tsv(gap_path) if gap_path and gap_path.is_file() else []
    blockers += validate_frozen_gap_map(gap_rows)
    parent_gap_uids = {str(row.get("gap_uid", "")) for row in parent["gap_rows"]}
    if {str(row.get("gap_uid", "")) for row in gap_rows} != parent_gap_uids:
        blockers.append(
            {
                "code": "frozen_gap_identity_drift",
                "message": "Round-17 frozen-gap identities differ from the accepted ledger.",
            }
        )
    if any(
        str(row.get("resolved_by_agent_reasoning", "")).lower() not in {"false", "0"}
        or str(row.get("independent_admitted_evidence_ref", ""))
        or str(row.get("frozen_gap_count_after_round17", "")) != "20"
        for row in gap_rows
    ):
        blockers.append(
            {
                "code": "agent_closes_frozen_gap_through_reasoning",
                "message": "Round-17 must preserve all frozen gaps without narrative closure.",
            }
        )

    for rows, fields in (
        (belief_states, ("belief_state_uid",)),
        (agents, ("agent_uid",)),
        (proposals, ("proposal_uid",)),
        (gates, ("gate_uid",)),
        (dual_actions, ("decision_uid",)),
        (twin_debts, ("debt_uid",)),
        (budget_decisions, ("decision_uid",)),
        (events, ("event_uid",)),
        (episodes, ("episode_uid",)),
        (explanations, ("explanation_uid",)),
    ):
        blockers += validate_unique_uids(rows, fields)
    lineage_edges: list[tuple[str, str]] = []
    for edge in graph.get("lineage_edges", []):
        if not isinstance(edge, list) or len(edge) != 2 or not all(str(item) for item in edge):
            blockers.append(
                {
                    "code": "lineage_edge_invalid",
                    "message": "Round-17 lineage edges must contain exactly two nonempty UIDs.",
                }
            )
            continue
        lineage_edges.append((str(edge[0]), str(edge[1])))
    blockers += validate_lineage_acyclic(lineage_edges)

    required_negative_cases = {
        "direct_state_update_attack",
        "incomplete_world_action",
        "budget_exhaustion",
        "unsafe_claim_request",
        "failed_tool_attempt_overwrite",
        "failed_tool_attempt_overwrite",
    }
    if set(negative_cases) != required_negative_cases:
        blockers.append(
            {
                "code": "round17_negative_pilot_cases_incomplete",
                "message": "Round-17 pilot adversarial fixtures are incomplete.",
            }
        )
    direct_case = _mapping(negative_cases.get("direct_state_update_attack"))
    incomplete_case = _mapping(negative_cases.get("incomplete_world_action"))
    budget_case = _mapping(negative_cases.get("budget_exhaustion"))
    unsafe_case = _mapping(negative_cases.get("unsafe_claim_request"))
    failed_attempt_case = _mapping(negative_cases.get("failed_tool_attempt_overwrite"))
    attempt_case = _mapping(negative_cases.get("failed_tool_attempt_overwrite"))
    if (
        not direct_case.get("blockers")
        or direct_case.get("execution_occurred") is not False
        or not incomplete_case.get("blockers")
        or incomplete_case.get("execution_occurred") is not False
        or _mapping(budget_case.get("decision")).get("decision") != "blocked_budget"
        or budget_case.get("execution_occurred") is not False
        or not unsafe_case.get("proposal_blockers")
        or not unsafe_case.get("hypothesis_blockers")
        or unsafe_case.get("execution_occurred") is not False
        or not failed_attempt_case.get("blockers")
        or failed_attempt_case.get("execution_occurred") is not False
        or not attempt_case.get("blockers")
        or attempt_case.get("execution_occurred") is not False
    ):
        blockers.append(
            {
                "code": "round17_negative_pilot_did_not_fail_closed",
                "message": "One or more deterministic adversarial scenarios did not fail closed.",
            }
        )

    registry, registry_blockers = _schema_registry(config, repository_root)
    blockers += registry_blockers
    gates_by_type = {
        "hypothesis_gate_decision": [row for row in gates if row.get("gate_type") == "H-Gate"],
        "world_action_gate_decision": [row for row in gates if row.get("gate_type") == "W-Gate"],
        "information_action_gate_decision": [row for row in gates if row.get("gate_type") == "I-Gate"],
        "observation_gate_decision": [row for row in gates if row.get("gate_type") == "O-Gate"],
    }
    schema_objects: dict[str, list[dict[str, object]]] = {
        "twin_belief_state": belief_states,
        "agent_identity": agents,
        "agent_proposal": proposals,
        "scientific_hypothesis": hypotheses,
        **gates_by_type,
        "dual_action_decision": dual_actions,
        "twin_debt": twin_debts,
        "scientific_budget": [scientific_budget],
        "risk_budget": [risk_budget],
        "agent_loop_event": events,
        "agent_loop_episode": episodes,
        "hypothesis_ledger_entry": ledgers,
        "agent_explanation": explanations,
        "round17_scitil_release_candidate": [release or {}],
    }
    for name in ROUND17_SCHEMA_KEYS:
        entry = registry.get(name)
        if entry is None:
            continue
        for index, value in enumerate(schema_objects.get(name, [])):
            blockers += _schema_validate(
                repository_root=repository_root,
                schema_entry=entry,
                instance=value,
                object_uid=_object_uid(value, f"{name}-{index}"),
            )

    if release:
        blockers += validate_release_candidate(release)
        expected_artifacts = {
            *REQUIRED_CATALOGS,
            "graph",
            "frozen_gap_map",
        }
        release_hashes = _mapping(release.get("artifact_hashes"))
        if set(release_hashes) != expected_artifacts:
            blockers.append(
                {
                    "code": "round17_release_artifact_set_invalid",
                    "message": "Round-17 release must bind every pilot catalog, graph, and gap map.",
                }
            )
        for name, expected_hash in release_hashes.items():
            path = paths.get(name)
            if path is None or not path.is_file() or sha256_file(path) != expected_hash:
                blockers.append(
                    {
                        "code": "round17_release_artifact_hash_mismatch",
                        "message": "Round-17 release artifact hash changed.",
                        "artifact": name,
                    }
                )
        if release.get("release_content_hash") != release_content_hash(release):
            blockers.append(
                {
                    "code": "round17_release_content_hash_mismatch",
                    "message": "Round-17 release content hash changed.",
                }
            )
        expected_parent_hash = canonical_sha256(parent["hashes"])
        expected_environment_hash = canonical_sha256(_mapping(graph.get("environment")))
        if (
            release.get("code_commit") != graph.get("code_commit")
            or release.get("config_hash") != sha256_file(repository_root / ROUND17_CONFIG_PATH)
            or release.get("schema_registry_hash")
            != sha256_file(repository_root / "Libs/configs/schema/schemas.yaml")
            or release.get("input_bundle_hash") != expected_parent_hash
            or release.get("environment_hash") != expected_environment_hash
        ):
            blockers.append(
                {
                    "code": "round17_release_identity_mismatch",
                    "message": "Round-17 release code, config, schema, input, or environment identity differs from materialization.",
                }
            )

    if pilot.get("schema_version") != PILOT_VALIDATION_SCHEMA_VERSION or pilot.get("status") != "passed":
        blockers.append(
            {
                "code": "round17_pilot_validation_invalid",
                "message": "Round-17 pilot validation record is not passing.",
            }
        )
    if (
        pilot.get("scenario_count") != 7
        or pilot.get("scenario_terminal_states") != SCENARIO_TERMINAL_STATES
        or pilot.get("agent_count") != 4
        or pilot.get("belief_state_count") != 2
        or pilot.get("falsifier_participation") is not True
        or pilot.get("direct_state_attack_rejected") is not True
        or pilot.get("arbitrary_shell_execution") is not False
        or pilot.get("autonomous_production_execution") is not False
    ):
        blockers.append(
            {
                "code": "round17_pilot_contract_incomplete",
                "message": "Round-17 deterministic pilot does not prove all A-G contract paths.",
            }
        )
    if (
        graph.get("frozen_gap_count") != 20
        or graph.get("frozen_gap_resolved_by_agent_reasoning") != 0
        or graph.get("frozen_gap_resolved_by_model_rollout") != 0
        or pilot.get("frozen_gap_count") != 20
        or pilot.get("frozen_gap_resolved_by_agent_reasoning") != 0
    ):
        blockers.append(
            {
                "code": "agent_closes_frozen_gap_through_reasoning",
                "message": "Model output or Agent reasoning cannot resolve empirical frozen gaps.",
            }
        )
    if (
        graph.get("evidence_fabric_sealed") is not False
        or pilot.get("evidence_fabric_sealed") is not False
        or graph.get("dt_core_status") != "not_evaluated"
        or pilot.get("dt_core_status") != "not_evaluated"
        or graph.get("autonomous_production_experiment") is not False
        or (release and release.get("empirical_observation_count") != 0)
    ):
        blockers.append(
            {
                "code": "round17_scope_boundary_violation",
                "message": "Round-17 changed an Evidence Fabric, empirical, production, or DT_CORE boundary.",
            }
        )

    summary = {
        "pilot_status": pilot.get("status"),
        "scenario_count": len(episodes),
        "scenario_terminal_states": scenario_states,
        "belief_state_count": len(belief_states),
        "agent_count": len(agents),
        "agent_roles": sorted(str(row.get("agent_role", "")) for row in agents),
        "proposal_count": len(proposals),
        "gate_count": len(gates),
        "gate_types": sorted({str(row.get("gate_type", "")) for row in gates}),
        "episode_count": len(episodes),
        "falsifier_participation": bool(falsifier_proposals),
        "conflict_count": sum(int(row.get("unresolved_conflict_count", 0)) for row in ledgers),
        "gap_count": len(gap_rows),
        "resolved_by_agent_reasoning": 0,
        "release_uid": release.get("release_uid") if release else None,
        "release_path": str(release_path) if release_path else None,
        "release": release,
        "artifact_hashes": _mapping(release.get("artifact_hashes")) if release else {},
        "parent_input_hashes": parent["hashes"],
        "graph_code_commit": graph.get("code_commit"),
        "evidence_fabric": {"materialization": "materialized_unsealed", "sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }
    return sorted_blockers(blockers), summary


def build_prefull_report(
    *,
    pilot_blockers: Sequence[Mapping[str, object]],
    pilot_summary: Mapping[str, object],
    focused: Mapping[str, object],
    identity: Mapping[str, object] | None,
    identity_blockers: Sequence[Mapping[str, object]],
    execution_dirty_status: str,
) -> dict[str, object]:
    blockers = [dict(row) for row in pilot_blockers] + [dict(row) for row in identity_blockers]
    names = set(focused.get("testcase_names", []))
    if not (
        focused.get("available")
        and focused.get("failures") == 0
        and focused.get("errors") == 0
        and focused.get("skipped") == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(names)
    ):
        blockers.append(
            {
                "code": "focused_negative_tests_not_verified",
                "message": "Mandatory Round-17 focused negative tests are not verified.",
                "missing": sorted(MANDATORY_NEGATIVE_TEST_NAMES - names),
            }
        )
    if execution_dirty_status != "clean":
        blockers.append(
            {
                "code": "execution_worktree_dirty",
                "message": "Pre-full Round-17 execution worktree is dirty.",
            }
        )
    blockers = sorted_blockers(blockers)
    return {
        "schema_version": PREFULL_SCHEMA_VERSION,
        "validator_uid": ROUND17_VALIDATOR_UID,
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "focused_tests": dict(focused),
        "execution_identity": dict(identity or {}),
        "pilot_summary": dict(pilot_summary),
        "full_suite_authorized": not blockers,
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }


def _inventory_blockers(
    path: Path | None, full: Mapping[str, object]
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    value = _load_json(path)
    if value is None:
        return (
            [
                {
                    "code": "inventory_missing",
                    "message": "Round-17 test inventory reconciliation is missing.",
                }
            ],
            None,
        )
    blockers: list[dict[str, object]] = []
    if value.get("schema_version") != TEST_INVENTORY_SCHEMA_VERSION:
        blockers.append(
            {
                "code": "inventory_schema_invalid",
                "message": "Round-17 test inventory schema is invalid.",
            }
        )
    if value.get("round16_parent_test_count") != ROUND17_PARENT_TEST_COUNT:
        blockers.append(
            {
                "code": "inventory_parent_count_invalid",
                "message": "Accepted Round-16 parent inventory must remain 517 tests.",
            }
        )
    if value.get("status") != "complete":
        blockers.append(
            {
                "code": "inventory_status_incomplete",
                "message": "Round-17 test inventory reconciliation is incomplete.",
            }
        )
    if value.get("unexplained_missing_tests") != 0 or value.get("mandatory_disappearance") != 0:
        blockers.append(
            {
                "code": "inventory_mandatory_disappearance",
                "message": "Mandatory parent tests disappeared from Round-17 inventory.",
            }
        )
    if full.get("available") and value.get("round17_current_test_count") != full.get("tests"):
        blockers.append(
            {
                "code": "inventory_count_mismatch",
                "message": "Round-17 inventory and full JUnit counts differ.",
            }
        )
    return sorted_blockers(blockers), value


def _full_suite_execution_blockers(
    full_execution: Mapping[str, object] | None,
    identity: Mapping[str, object] | None,
    full: Mapping[str, object],
) -> list[dict[str, object]]:
    if full_execution is None or full_execution.get("schema_version") != FULL_SUITE_EXECUTION_SCHEMA_VERSION:
        return [
            {
                "code": "execution_full_suite_missing",
                "message": "Round-17 CPU full-suite execution evidence is missing.",
            }
        ]
    passed = full_execution.get("passed")
    valid = (
        full_execution.get("terminal_state") == "COMPLETED"
        and full_execution.get("exit_code") == "0:0"
        and full_execution.get("node") in ROUND17_CPU_FULL_SUITE_NODES
        and full_execution.get("partition") == "cpu"
        and str(full_execution.get("job_name", "")).startswith("NrnSift_DT_R17_")
        and identity
        and full_execution.get("exact_git_commit") == identity.get("exact_git_commit")
        and int(full_execution.get("failures", -1)) == 0
        and int(full_execution.get("errors", -1)) == 0
        and int(full_execution.get("skipped", -1)) == 0
        and int(full_execution.get("mandatory_skipped", -1)) == 0
        and int(full_execution.get("core_xfail", -1)) == 0
        and full.get("available")
        and passed == full.get("tests")
    )
    return [] if valid else [
        {
            "code": "execution_full_suite_failed",
            "message": "Round-17 CPU full-suite terminal evidence did not pass.",
        }
    ]


def _commit_manifest_blockers(
    path: Path | None,
    identity: Mapping[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    value = _load_json(path)
    if value is None:
        return (
            [
                {
                    "code": "commit_manifest_missing",
                    "message": "Round-17 scoped commit manifest is missing.",
                }
            ],
            None,
        )
    blockers: list[dict[str, object]] = []
    if value.get("schema_version") != COMMIT_MANIFEST_SCHEMA_VERSION:
        blockers.append(
            {
                "code": "commit_manifest_schema_invalid",
                "message": "Round-17 commit manifest schema is invalid.",
            }
        )
    if value.get("round_uid") != ROUND17_STAGE_UID:
        blockers.append(
            {
                "code": "commit_manifest_round_invalid",
                "message": "Round-17 commit manifest round identity is invalid.",
            }
        )
    if identity and value.get("execution_commit_candidate") != identity.get("exact_git_commit"):
        blockers.append(
            {
                "code": "commit_manifest_execution_mismatch",
                "message": "Round-17 commit manifest does not bind the execution commit.",
            }
        )
    checks = _mapping(value.get("scope_checks"))
    if (
        checks.get("all_authorized_paths_committed") is not True
        or checks.get("unrelated_paths_committed") is not False
        or checks.get("generated_runtime_files_committed") is not False
        or checks.get("cache_files_committed") is not False
        or checks.get("round18_implementation_committed") is not False
    ):
        blockers.append(
            {
                "code": "commit_manifest_scope_invalid",
                "message": "Round-17 commit manifest does not preserve the authorized source-only scope.",
            }
        )
    return sorted_blockers(blockers), value


def _gate_status(blockers: Sequence[Mapping[str, object]]) -> str:
    if not blockers:
        return "scitil_twin_gated_agent_loop_resolved"
    codes = {str(row.get("code", "")) for row in blockers}
    mappings = (
        (
            "blocked_execution_verification",
            (
                "execution",
                "full_suite",
                "inventory",
                "prefull",
                "focused",
                "commit_manifest",
            ),
        ),
        ("blocked_gap_fabrication", ("gap", "frozen")),
        ("blocked_claim_boundary", ("claim", "causal", "clinical", "empirical")),
        ("blocked_conflict_preservation", ("conflict", "falsifier")),
        ("blocked_budget_enforcement", ("budget", "risk")),
        ("blocked_twin_debt", ("debt",)),
        ("blocked_assimilation_authority", ("assimilation", "belief_state_bypasses_o_gate", "uncertainty_only_scope")),
        ("blocked_tool_sandbox", ("tool", "sandbox", "adapter", "shell", "attempt")),
        ("blocked_world_information_action_boundary", ("world_information", "world_action", "information_action", "what_if", "observation_request")),
        ("blocked_gate_authority", ("gate", "validator_not_approved", "authority")),
        ("blocked_hypothesis_semantics", ("hypothesis",)),
        ("blocked_agent_privilege", ("agent", "proposal")),
    )
    for state, markers in mappings:
        if any(any(marker in code for marker in markers) for code in codes):
            return state
    return "blocked_execution_verification"


def build_round17_gate_record(
    *,
    authority: Mapping[str, object],
    assessment: str,
    blockers: list[dict[str, object]],
    input_hashes: Mapping[str, str],
    config_hash: str,
    commit: str,
    report: Mapping[str, object],
) -> dict[str, object]:
    common = {
        "round_uid": ROUND17_STAGE_UID,
        "status": assessment,
        "overall_gate_status": "blocked" if blockers else "accepted",
        "validation_status": report["status"],
        "blocker_count": len(blockers),
        "blocker_classes": sorted(
            {_gate_status([row]).removeprefix("blocked_") for row in blockers}
        ),
        "frozen_gap_count": report.get("frozen_gap_count", 0),
        "frozen_gap_resolved_by_agent_reasoning": 0,
        "execution_identity": report["execution_identity"],
        "evidence_fabric": report["evidence_fabric"],
        "dt_core": {"status": "not_evaluated"},
        "round_11_closure": "accepted_unchanged",
        "round_12_gate": "typed_evidence_relationships_resolved_unchanged",
        "round_13_gate": "action_observation_contracts_resolved_unchanged",
        "round_14_gate": "evidence_fabric_v0_materialized_unchanged",
        "round_15_gate": "clean_action_complete_trajectory_domain_resolved_unchanged",
        "round_16_gate": "governed_what_if_engine_resolved_unchanged",
        "claim_boundary": {
            "causal_effect_claim": False,
            "clinical_recommendation": False,
            "agent_output_empirical_observation": False,
            "autonomous_discovery_claim": False,
        },
    }
    if blockers:
        return {
            "schema_version": "neuronsifter-dt-round17-blocked-gate-assessment-v1",
            "gate_uid": ROUND17_STAGE_UID,
            "stage_uid": ROUND17_STAGE_UID,
            "writer_kind": "approved_validator_blocked_assessment",
            "validator_uid": ROUND17_VALIDATOR_UID,
            "canonical_controller_uid": CANONICAL_CONTROLLER_UID,
            "input_artifact_hashes": dict(sorted(input_hashes.items())),
            "config_hash": config_hash,
            "git_commit": commit,
            "decision": "blocked",
            "blockers": blockers,
            "lineage_parent": "round_16_governed_what_if_engine",
        } | common
    return build_canonical_gate_decision(
        authority,
        gate_uid=ROUND17_STAGE_UID,
        stage_uid=ROUND17_STAGE_UID,
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid=ROUND17_VALIDATOR_UID,
        validator_report={
            "validator_uid": ROUND17_VALIDATOR_UID,
            "status": "passed",
            "assessment": assessment,
        },
        input_artifact_hashes=input_hashes,
        config_hash=config_hash,
        commit=commit,
        decision="accepted",
        blockers=[],
        lineage_parent="round_16_governed_what_if_engine",
    ) | common


def validate_round17_outputs(
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
    blockers, summary = validate_pilot_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_output_root,
        parent_root=parent_root,
    )
    release = summary.get("release") if isinstance(summary.get("release"), Mapping) else None
    graph = _load_json(_paths(config, source_output_root).get("graph"))
    identity_blockers, identity = _identity_blockers(
        execution_identity_path,
        repository_root,
        release,
        graph,
    )
    blockers += identity_blockers
    focused = _junit_summary(focused_junit_path)
    focused_names = set(focused.get("testcase_names", []))
    if not (
        focused.get("available")
        and focused.get("failures") == 0
        and focused.get("errors") == 0
        and focused.get("skipped") == 0
        and MANDATORY_NEGATIVE_TEST_NAMES.issubset(focused_names)
    ):
        blockers.append(
            {
                "code": "focused_negative_tests_not_verified",
                "message": "Mandatory Round-17 focused negative tests are not verified.",
                "missing": sorted(MANDATORY_NEGATIVE_TEST_NAMES - focused_names),
            }
        )
    prefull = _load_json(prefull_validation_path)
    if (
        prefull is None
        or prefull.get("schema_version") != PREFULL_SCHEMA_VERSION
        or prefull.get("status") != "passed"
        or prefull.get("full_suite_authorized") is not True
    ):
        blockers.append(
            {
                "code": "prefull_not_verified",
                "message": "Exact-clean Round-17 pre-full Gate did not pass.",
            }
        )
    full = _junit_summary(full_suite_junit_path)
    if not (
        full.get("available")
        and int(full.get("tests", 0)) >= ROUND17_PARENT_TEST_COUNT
        and full.get("failures") == 0
        and full.get("errors") == 0
        and full.get("skipped") == 0
    ):
        blockers.append(
            {
                "code": "full_suite_not_verified",
                "message": "Full Round-17 Libs/tests suite is not verified.",
            }
        )
    inventory_blockers, inventory = _inventory_blockers(test_inventory_path, full)
    blockers += inventory_blockers
    full_execution = _load_json(full_suite_execution_path)
    blockers += _full_suite_execution_blockers(full_execution, identity, full)
    commit_blockers, commit_manifest = _commit_manifest_blockers(commit_manifest_path, identity)
    blockers += commit_blockers
    if git_dirty_status(repository_root) != "clean":
        blockers.append(
            {
                "code": "execution_worktree_dirty",
                "message": "Exact Round-17 execution worktree is dirty.",
            }
        )
    blockers = sorted_blockers(blockers)
    assessment = _gate_status(blockers)
    report = {
        "schema_version": "neuronsifter-dt-round17-validation-v1",
        "validator_uid": ROUND17_VALIDATOR_UID,
        "status": "passed" if not blockers else "blocked",
        "gate_assessment": assessment,
        "pilot_summary": summary,
        "focused_tests": focused,
        "prefull_validation": prefull,
        "execution_identity": identity
        or {
            "exact_git_commit": git_commit(repository_root),
            "dirty_status": git_dirty_status(repository_root),
            "reproduction_relation": "not_exact_clean_commit",
        },
        "full_suite": full,
        "full_suite_execution": full_execution,
        "test_inventory": inventory,
        "commit_manifest": commit_manifest,
        "frozen_gap_count": summary.get("gap_count", 0),
        "frozen_gap_resolved_by_agent_reasoning": 0,
        "evidence_fabric": {"materialization": "materialized_unsealed", "sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "blockers": blockers,
        "blocker_count": len(blockers),
    }
    input_hashes = {
        name: str(value)
        for name, value in _mapping(summary.get("artifact_hashes")).items()
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
    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    gate = build_round17_gate_record(
        authority=authority,
        assessment=assessment,
        blockers=blockers,
        input_hashes=input_hashes,
        config_hash=sha256_file(repository_root / ROUND17_CONFIG_PATH),
        commit=git_commit(repository_root),
        report=report,
    )
    return report, gate


def build_round18_handoff(
    report: Mapping[str, object], gate: Mapping[str, object]
) -> dict[str, object]:
    summary = _mapping(report.get("pilot_summary"))
    release = _mapping(summary.get("release"))
    return {
        "schema_version": "neuronsifter-dt-round18-handoff-v1",
        "status": "ready_for_publication_time_and_literature_held_out_sequential_replay",
        "parent_gate": gate["status"],
        "execution_commit": gate["git_commit"],
        "twin_belief_state_contract": {
            "schema_version": SCHEMA_VERSIONS["twin_belief_state"],
            "content_addressed": True,
            "creation_requires_approved_o_gate": True,
            "agent_text_as_posterior": False,
        },
        "agent_roles_and_privileges": {
            role: {
                "proposal_only": True,
                "gate_writer": False,
                "tool_executor": False,
                "state_update_authority": False,
            }
            for role in sorted(AGENT_ROLES)
        },
        "gate_contracts": {
            "H-Gate": SCHEMA_VERSIONS["hypothesis_gate_decision"],
            "W-Gate": SCHEMA_VERSIONS["world_action_gate_decision"],
            "I-Gate": SCHEMA_VERSIONS["information_action_gate_decision"],
            "O-Gate": SCHEMA_VERSIONS["observation_gate_decision"],
            "sole_writer": CANONICAL_CONTROLLER_UID,
        },
        "twin_kernel_interface": {
            "llm_backed": False,
            "mandatory": True,
            "registered_adapters_only": True,
            "approved_assimilation_only": True,
        },
        "dual_action_decision_policy": {
            "schema_version": SCHEMA_VERSIONS["dual_action_decision"],
            "world_information_actions_distinct": True,
            "high_or_unknown_debt_policy": "information_action_or_stop",
        },
        "twin_debt_components": [
            "calibration_debt",
            "coverage_debt",
            "disagreement_debt",
            "provenance_debt",
            "constraint_debt",
        ],
        "budget_and_risk_policies": {
            "scientific_budget_schema": SCHEMA_VERSIONS["scientific_budget"],
            "risk_budget_schema": SCHEMA_VERSIONS["risk_budget"],
            "agent_may_increase_budget_or_risk": False,
            "production_execution_allowed": False,
        },
        "hypothesis_and_falsification_ledger": {
            "schema_version": SCHEMA_VERSIONS["hypothesis_ledger_entry"],
            "falsifier_required_before_positive_summary": True,
            "conflicts_preserved": True,
            "proven_or_true_status_allowed": False,
        },
        "tool_and_rollout_sandbox": {
            "allowlisted_adapters_only": True,
            "arbitrary_shell_or_python": False,
            "read_only_code": True,
            "isolated_output": True,
            "failed_attempts_preserved": True,
        },
        "claim_language_boundary": {
            "model_output_is_empirical_observation": False,
            "causal_effect_claim": False,
            "clinical_recommendation": False,
            "autonomous_discovery_claim": False,
        },
        "frozen_gap_isolation": {
            "count": report["frozen_gap_count"],
            "resolved_by_agent_reasoning": 0,
            "independent_admitted_evidence_required": True,
        },
        "episode_replay_identity": {
            "deterministic": True,
            "content_addressed_release_uid": release.get("release_uid"),
            "release_content_hash": release.get("release_content_hash"),
            "contract_evidence_only": True,
        },
        "allowed_literature_held_out_benchmark_interfaces": [
            "immutable_publication_time_snapshot",
            "typed_sequential_evidence_replay",
            "controller_gated_information_action",
            "controller_gated_restricted_rollout",
            "falsifier_first_hypothesis_assessment",
            "non_empirical_model_conditional_explanation",
        ],
        "forbidden_round18_paths": [
            "agent_direct_state_update",
            "agent_gate_write",
            "arbitrary_agent_tool_execution",
            "causal_or_clinical_claim",
            "empirical_gap_closure_by_reasoning",
            "evidence_fabric_sealing",
            "dt_core_evaluation",
        ],
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Round-17 SciTiL execution.")
    parser.add_argument("--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--source-output-root", required=True)
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--focused-junit")
    parser.add_argument("--execution-identity")
    parser.add_argument("--prefull-validation")
    parser.add_argument("--full-suite-junit")
    parser.add_argument("--full-suite-execution")
    parser.add_argument("--test-inventory")
    parser.add_argument("--commit-manifest")
    parser.add_argument("--prefull-only", action="store_true")
    args = parser.parse_args()

    repository_root = Path(args.repository_root).resolve()
    source_output_root = Path(args.source_output_root).resolve()
    parent_root = Path(args.parent_root).resolve()
    output_root = Path(args.output_root).resolve()
    config = load_config(Path(args.config))
    pilot_blockers, pilot_summary = validate_pilot_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_output_root,
        parent_root=parent_root,
    )
    release = pilot_summary.get("release") if isinstance(pilot_summary.get("release"), Mapping) else None
    graph = _load_json(_paths(config, source_output_root).get("graph"))
    identity_blockers, identity = _identity_blockers(
        Path(args.execution_identity) if args.execution_identity else None,
        repository_root,
        release,
        graph,
    )
    focused = _junit_summary(Path(args.focused_junit) if args.focused_junit else None)
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
        print(f"round17_prefull_{prefull['status']} blockers={prefull['blocker_count']}")
        return 0 if prefull["status"] == "passed" else 1

    report, gate = validate_round17_outputs(
        config=config,
        repository_root=repository_root,
        source_output_root=source_output_root,
        parent_root=parent_root,
        focused_junit_path=Path(args.focused_junit) if args.focused_junit else None,
        prefull_validation_path=Path(args.prefull_validation) if args.prefull_validation else None,
        execution_identity_path=Path(args.execution_identity) if args.execution_identity else None,
        full_suite_junit_path=Path(args.full_suite_junit) if args.full_suite_junit else None,
        full_suite_execution_path=Path(args.full_suite_execution) if args.full_suite_execution else None,
        test_inventory_path=Path(args.test_inventory) if args.test_inventory else None,
        commit_manifest_path=Path(args.commit_manifest) if args.commit_manifest else None,
    )
    write_json(output_root / VALIDATION_PATH, report)
    write_json(output_root / GATE_PATH, gate)
    if gate["status"] == "scitil_twin_gated_agent_loop_resolved":
        write_json(output_root / HANDOFF_PATH, build_round18_handoff(report, gate))
    print(f"{gate['status']} blockers={report['blocker_count']} gaps={report['frozen_gap_count']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
