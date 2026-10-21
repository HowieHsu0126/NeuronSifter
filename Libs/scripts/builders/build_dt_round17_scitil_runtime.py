"""Materialize the deterministic, contract-only Round-17 SciTiL pilot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.action_observation_contracts import (  # noqa: E402
    validate_world_action,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv  # noqa: E402
from Libs.utils.data.scitil_contracts import (  # noqa: E402
    AgentExplanation,
    AgentIdentity,
    AgentLoopBudgetDecision,
    AgentLoopEpisode,
    AgentLoopEvent,
    AgentProposal,
    CANONICAL_CONTROLLER_UID,
    DualActionDecision,
    HypothesisGateDecision,
    HypothesisLedgerEntry,
    InformationActionGateDecision,
    ObservationGateDecision,
    RiskBudget,
    ROUND17_VALIDATOR_UID,
    SCHEMA_VERSIONS,
    ScientificBudget,
    ScientificHypothesis,
    TwinBeliefState,
    TwinDebt,
    TwinKernel,
    WorldActionGateDecision,
    belief_state_content_hash,
    canonical_sha256,
    create_belief_state_from_assimilation,
    release_content_hash,
    round17_config,
    stable_uid,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def environment_identity() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_slurm"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", "not_slurm"),
        "slurm_node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "not_slurm"),
        "pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX", ""),
    }


def _resolve(root: Path, relative: object) -> Path:
    path = Path(str(relative))
    if path.is_absolute():
        raise ValueError(f"Round-17 configured paths must be relative: {path}")
    return root / path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _parent_paths(config: Mapping[str, object], parent_root: Path) -> dict[str, Path]:
    return {name: _resolve(parent_root, path) for name, path in dict(round17_config(config)["parent_artifacts"]).items()}


def validate_parent_state(config: Mapping[str, object], parent_root: Path) -> dict[str, object]:
    paths = _parent_paths(config, parent_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Round-17 parent artifacts are missing: {missing}")
    handoff = _load_json(paths["round17_handoff"])
    gate = _load_json(paths["round16_gate"])
    validation = _load_json(paths["round16_validation"])
    identity = _load_json(paths["round16_execution_identity"])
    full_suite = _load_json(paths["round16_full_suite_execution"])
    inventory = _load_json(paths["round16_test_inventory"])
    round16_graph = _load_json(paths["round16_graph"])
    round13_graph = _load_json(paths["round13_graph"])
    release = _load_json(paths["round16_release_candidate"])
    gap_rows = read_tsv(paths["round14_gap_ledger"])
    blockers: list[dict[str, object]] = []
    if handoff.get("status") != "ready_for_scitil_twin_gated_agent_contracts":
        blockers.append({"code": "round17_entry_not_ready", "message": "Round-17 handoff is not ready."})
    if handoff.get("execution_commit") != "856bf9e0c9f6f4fb97866780851d3dfc2afc73e0":
        blockers.append({"code": "round16_handoff_identity_drift", "message": "Round-16 handoff commit changed."})
    if gate.get("status") != "governed_what_if_engine_resolved" or gate.get("overall_gate_status") != "accepted":
        blockers.append({"code": "round16_gate_not_accepted", "message": "Round-16 Gate is not accepted."})
    if validation.get("status") != "passed" or validation.get("gate_assessment") != "governed_what_if_engine_resolved":
        blockers.append({"code": "round16_validation_not_passed", "message": "Round-16 validation is not passing."})
    if identity.get("exact_git_commit") != handoff.get("execution_commit") or identity.get("dirty_status") != "clean":
        blockers.append({"code": "round16_execution_identity_invalid", "message": "Round-16 exact-clean identity changed."})
    if not (
        full_suite.get("job_id") == "496843"
        and full_suite.get("terminal_state") == "COMPLETED"
        and full_suite.get("exit_code") == "0:0"
        and int(full_suite.get("passed", -1)) == 517
        and int(full_suite.get("failures", -1)) == 0
        and int(full_suite.get("errors", -1)) == 0
        and int(full_suite.get("skipped", -1)) == 0
    ):
        blockers.append({"code": "round16_full_suite_invalid", "message": "Accepted Round-16 full-suite evidence changed."})
    if not (
        inventory.get("status") == "complete"
        and int(inventory.get("round16_current_test_count", -1)) == 517
        and int(inventory.get("mandatory_disappearance", -1)) == 0
        and int(inventory.get("unexplained_missing_tests", -1)) == 0
    ):
        blockers.append({"code": "round16_test_inventory_invalid", "message": "Round-16 inventory is incomplete."})
    if release.get("sealed") is not False or release.get("status") != "materialized_unsealed":
        blockers.append({"code": "parent_release_not_unsealed", "message": "Parent Evidence Fabric must remain unsealed."})
    if round16_graph.get("frozen_gap_count") != 20 or round16_graph.get("frozen_gap_resolved_by_model_rollout") != 0:
        blockers.append({"code": "round16_gap_isolation_drift", "message": "Round-16 model rollout gap isolation changed."})
    if len(gap_rows) != 20:
        blockers.append({"code": "frozen_gap_count_drift", "message": "Exactly 20 frozen gaps are required."})
    if round13_graph.get("evidence_fabric", {}).get("sealed") is not False:
        blockers.append({"code": "round13_evidence_fabric_state_drift", "message": "Round-13 Evidence Fabric state changed."})
    if blockers:
        raise ValueError(blockers)
    return {
        "paths": paths,
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "handoff": handoff,
        "gate": gate,
        "validation": validation,
        "identity": identity,
        "full_suite": full_suite,
        "inventory": inventory,
        "round16_graph": round16_graph,
        "round13_graph": round13_graph,
        "release": release,
        "gap_rows": gap_rows,
    }


def _catalog_row(record_type: str, decision: str, value: Mapping[str, object]) -> dict[str, object]:
    uid_fields_by_type = {
        "AgentIdentity": ("agent_uid",),
        "HypothesisLedgerEntry": ("hypothesis_uid",),
        "AgentProposal": ("proposal_uid",),
        "H-Gate": ("gate_uid",),
        "W-Gate": ("gate_uid",),
        "I-Gate": ("gate_uid",),
        "O-Gate": ("gate_uid",),
        "AgentLoopEvent": ("event_uid",),
        "AgentLoopEpisode": ("episode_uid",),
        "TwinDebt": ("debt_uid",),
        "AgentLoopBudgetDecision": ("decision_uid",),
        "AgentExplanation": ("explanation_uid",),
    }
    uid_fields = uid_fields_by_type.get(record_type, ())
    uid = next((str(value[field]) for field in uid_fields if value.get(field)), stable_uid("catalog", value))
    return {
        "record_uid": uid,
        "record_type": record_type,
        "decision": decision,
        "schema_version": value.get("schema_version", "neuronsifter-agent-loop-budget-decision-v1"),
        "object_json": json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    }


def _write_catalog(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    write_tsv(
        path,
        ["record_uid", "record_type", "decision", "schema_version", "object_json"],
        [dict(row) for row in rows],
    )


def _agent(role: str, config: Mapping[str, object]) -> dict[str, object]:
    row = dict(dict(round17_config(config)["agent_roles"])[role])
    return AgentIdentity(
        schema_version=SCHEMA_VERSIONS["agent_identity"],
        agent_uid=str(row["agent_uid"]),
        agent_role=role,
        capabilities=list(row["capabilities"]),
        prohibited_capabilities=[
            "write_gate",
            "execute_tool_or_rollout",
            "update_twin_state",
            "upgrade_evidence",
            "close_frozen_gap_by_reasoning",
        ],
        gate_writer=False,
        tool_executor=False,
        state_update_authority=False,
        canonical_controller=False,
        provenance={"source": "Libs/configs/data/round17_scitil_runtime.yaml", "status": "registered"},
    ).to_dict()


def _proposal(
    *,
    uid: str,
    agent: Mapping[str, object],
    belief_state_ref: str,
    proposal_type: str,
    support_domain_ref: str,
    hypothesis_ref: str | None = None,
    world_action_ref: str | None = None,
    information_action_ref: str | None = None,
    what_if_request_ref: str | None = None,
    observation_request_ref: str | None = None,
    risk_class: str = "contract_only",
    provenance: Mapping[str, object] | None = None,
    claim_flags: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    return AgentProposal(
        schema_version=SCHEMA_VERSIONS["agent_proposal"],
        proposal_uid=uid,
        agent_uid=str(agent["agent_uid"]),
        agent_role=str(agent["agent_role"]),
        belief_state_ref=belief_state_ref,
        proposal_type=proposal_type,
        hypothesis_ref=hypothesis_ref,
        world_action_ref=world_action_ref,
        information_action_ref=information_action_ref,
        what_if_request_ref=what_if_request_ref,
        observation_request_ref=observation_request_ref,
        expected_information_gain={"status": "represented", "relative": 0.25},
        expected_twin_debt_change={"status": "represented", "direction": "nonincreasing_or_unknown"},
        expected_cost={"compute_seconds": 1.0, "tool_calls": 0.0},
        risk_class=risk_class,
        support_domain_ref=support_domain_ref,
        evidence_refs=["round16_governed_what_if_engine_resolved"],
        assumptions=["contract_pilot_only"],
        uncertainties=["biological_validity_not_established"],
        prohibited_claim_flags=dict(claim_flags or {"causal_effect_claim": False, "clinical_recommendation": False}),
        provenance=dict(provenance or {"source": "round17_deterministic_pilot", "compiled": True}),
    ).to_dict()


def _gate(
    *, gate_class: type[HypothesisGateDecision], schema_key: str, uid: str, gate_type: str,
    proposal_uid: str, decision: str, input_values: Mapping[str, object], reason: str,
    approved_scope: Mapping[str, object] | None = None,
) -> dict[str, object]:
    scope = {
        "execution_scope": "contract_pilot_only"
        if decision.startswith(("authorized", "admitted", "approved"))
        else "none"
    }
    scope.update(dict(approved_scope or {}))
    return gate_class(
        schema_version=SCHEMA_VERSIONS[schema_key],
        gate_uid=uid,
        gate_type=gate_type,
        proposal_uid=proposal_uid,
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid=ROUND17_VALIDATOR_UID,
        input_hashes={name: canonical_sha256(value) for name, value in input_values.items()},
        decision=decision,
        reason_codes=[reason],
        approved_scope=scope,
        budget_effect={"increase_allowed": False, "status": "bounded"},
        twin_debt_effect={"unknown_to_zero": False, "status": "preserved"},
        support_domain_effect={"expansion_allowed": False, "status": "exact_or_restricted"},
        provenance={"source": "round17_twin_kernel", "writer": CANONICAL_CONTROLLER_UID},
    ).to_dict()


def _debt(uid: str, belief_state_ref: str, statuses: Mapping[str, str], values: Mapping[str, float | None]) -> dict[str, object]:
    unknown = [name for name, status in statuses.items() if status == "unknown"]
    allowed_decisions = ["information_action", "stop"]
    if "high" not in statuses.values() and "unknown" not in statuses.values():
        allowed_decisions.insert(0, "world_action_restricted")
    return TwinDebt(
        schema_version=SCHEMA_VERSIONS["twin_debt"],
        debt_uid=uid,
        belief_state_ref=belief_state_ref,
        component_values=dict(values),
        component_status=dict(statuses),
        measurement_basis={"source": "round17_contract_diagnostics", "componentwise": True},
        unknown_components=unknown,
        versioned_thresholds={"version": "neuronsifter-round17-twin-debt-thresholds-v1", "low_max": 0.25, "moderate_max": 0.60},
        allowed_decisions=allowed_decisions,
        aggregation=None,
        provenance={"source": "round17_deterministic_pilot", "unknowns_preserved": True},
    ).to_dict()


def _budget_decision(
    uid: str, proposal_uid: str, requested: Mapping[str, float], risk_class: str, decision: str,
    remaining: Mapping[str, float], reason: str
) -> dict[str, object]:
    return AgentLoopBudgetDecision(
        decision_uid=uid,
        proposal_uid=proposal_uid,
        budget_uid="round17_scientific_budget",
        risk_budget_uid="round17_risk_budget",
        controller_uid=CANONICAL_CONTROLLER_UID,
        requested_cost=dict(requested),
        requested_risk_class=risk_class,
        decision=decision,
        remaining_after=dict(remaining),
        reason_codes=[reason],
    ).to_dict()


def _events(
    *, scenario_uid: str, transitions: Sequence[tuple[str, str]], created_at: str,
    authority_bindings: Sequence[tuple[str | None, str | None]],
) -> list[dict[str, object]]:
    if len(transitions) != len(authority_bindings):
        raise ValueError(f"Every {scenario_uid} transition requires an explicit authority binding.")
    rows: list[dict[str, object]] = []
    episode_uid = f"episode_{scenario_uid}"
    for index, ((prior, next_state), (agent_uid, gate_uid)) in enumerate(zip(transitions, authority_bindings, strict=True)):
        event_uid = f"event_{scenario_uid}_{index:02d}"
        rows.append(
            AgentLoopEvent(
                schema_version=SCHEMA_VERSIONS["agent_loop_event"],
                event_uid=event_uid,
                episode_uid=episode_uid,
                prior_state=prior,
                next_state=next_state,
                agent_uid=agent_uid,
                gate_uid=gate_uid,
                controller_uid=CANONICAL_CONTROLLER_UID,
                input_hashes={"prior": canonical_sha256({"scenario": scenario_uid, "state": prior, "index": index})},
                output_hashes={"next": canonical_sha256({"scenario": scenario_uid, "state": next_state, "index": index})},
                timestamp=created_at,
                decision_reason=f"round17_{scenario_uid}_{prior}_to_{next_state}",
            ).to_dict()
        )
    return rows


def _episode(
    *, scenario_uid: str, events: Sequence[Mapping[str, object]], initial_ref: str, final_ref: str | None,
    terminal_state: str, budget_ref: str, gap_uids: Sequence[str], conflict_refs: Sequence[str] = ()
) -> dict[str, object]:
    blocked = terminal_state in {"budget_exhausted", "unsupported_domain", "blocked_by_gate", "no_safe_action"}
    return AgentLoopEpisode(
        schema_version=SCHEMA_VERSIONS["agent_loop_episode"],
        episode_uid=f"episode_{scenario_uid}",
        initial_belief_state_ref=initial_ref,
        final_belief_state_ref=final_ref,
        event_uids=[str(row["event_uid"]) for row in events],
        scenario_uid=scenario_uid,
        terminal_state=terminal_state,
        controller_uid=CANONICAL_CONTROLLER_UID,
        budget_decision_refs=[budget_ref],
        conflict_refs=list(conflict_refs),
        frozen_gap_impacts={gap_uid: "no_effect" for gap_uid in gap_uids},
        replay_identity={"deterministic": True, "qualification_only": True, "autonomous_execution": False},
        status="blocked" if blocked else "completed",
        provenance={"source": "round17_deterministic_pilot", "contract_evidence_only": True},
    ).to_dict()


def _gap_rows(parent_rows: Sequence[Mapping[str, object]], proposals: Sequence[str], episodes: Sequence[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in parent_rows:
        remaining = str(row.get("remaining_blocker", ""))
        if remaining == "hypothesis_generation":
            impact = "hypothesis_generated"
        elif remaining == "trajectory_training":
            impact = "requires_new_e4"
        else:
            impact = "requires_empirical_observation"
        rows.append(
            {
                "gap_uid": row["gap_uid"],
                "parent_status": row["current_status"],
                "proposal_uids": "|".join(proposals),
                "episode_uids": "|".join(episodes),
                "round17_impact": impact,
                "resolved_by_agent_reasoning": False,
                "independent_admitted_evidence_ref": "",
                "remaining_blocker": remaining,
                "frozen_gap_count_after_round17": 20,
            }
        )
    return rows


def build_round17_pilot(
    *, config: Mapping[str, object], repository_root: Path, parent_root: Path, output_root: Path, created_at: str
) -> dict[str, object]:
    config_blockers = validate_round17_config(config)
    if config_blockers:
        raise ValueError(config_blockers)
    parent = validate_parent_state(config, parent_root)
    runtime_config = round17_config(config)
    paths = dict(runtime_config["output_paths"])
    commit = git_commit(repository_root)
    env = environment_identity()
    environment_hash = canonical_sha256(env)
    input_bundle_hash = canonical_sha256(parent["hashes"])
    round16_objects = dict(parent["round16_graph"]["objects"])
    round13_graph = dict(parent["round13_graph"])
    round16_request = dict(round16_objects["what_if_request"])
    no_action = dict(dict(round16_objects["action_binding"])["world_action"])
    support_domain_ref = str(no_action["support_domain_ref"])

    agents = {role: _agent(role, config) for role in ("ScientistAgent", "FalsifierAgent", "ExperimentAgent", "EvidenceAgent")}
    initial_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["twin_belief_state"],
        "belief_state_uid": "pending",
        "parent_belief_state_uid": None,
        "state_posterior_ref": str(round16_objects["baseline"]["state_ref"]),
        "parameter_posterior_ref": "round16_restricted_parameter_posterior",
        "model_family_posterior_ref": "round16_restricted_replay_model_family",
        "evidence_snapshot_ref": str(parent["release"]["release_uid"]),
        "active_hypothesis_refs": [],
        "constraint_state_ref": str(round16_objects["constraint_report"]["constraint_report_uid"]),
        "support_domain_ref": support_domain_ref,
        "twin_debt_ref": "debt_round17_low",
        "content_hash": "pending",
        "controller_uid": CANONICAL_CONTROLLER_UID,
        "creation_decision_ref": str(parent["gate"]["gate_uid"]),
        "provenance": {"source": "round16_restricted_contract_state", "initial_state": True},
    }
    initial_payload["content_hash"] = belief_state_content_hash(initial_payload)
    initial_payload["belief_state_uid"] = stable_uid("belief_state", initial_payload["content_hash"])
    initial_belief = TwinBeliefState(**initial_payload).to_dict()

    low_debt = _debt(
        "debt_round17_low", initial_belief["belief_state_uid"],
        {name: "low" for name in ("calibration_debt", "coverage_debt", "disagreement_debt", "provenance_debt", "constraint_debt")},
        {name: 0.10 for name in ("calibration_debt", "coverage_debt", "disagreement_debt", "provenance_debt", "constraint_debt")},
    )
    high_debt = _debt(
        "debt_round17_high_information_required", initial_belief["belief_state_uid"],
        {"calibration_debt": "moderate", "coverage_debt": "moderate", "disagreement_debt": "high", "provenance_debt": "high", "constraint_debt": "moderate"},
        {"calibration_debt": 0.50, "coverage_debt": 0.55, "disagreement_debt": 0.90, "provenance_debt": 0.90, "constraint_debt": 0.50},
    )
    conflict_debt = _debt(
        "debt_round17_conflict", initial_belief["belief_state_uid"],
        {"calibration_debt": "moderate", "coverage_debt": "moderate", "disagreement_debt": "high", "provenance_debt": "moderate", "constraint_debt": "low"},
        {"calibration_debt": 0.45, "coverage_debt": 0.45, "disagreement_debt": 0.80, "provenance_debt": 0.45, "constraint_debt": 0.10},
    )
    debts = [low_debt, high_debt, conflict_debt]

    budget_dimensions = {key: float(value) for key, value in dict(runtime_config["scientific_budget"]["dimensions"]).items()}
    scientific_budget = ScientificBudget(
        schema_version=SCHEMA_VERSIONS["scientific_budget"], budget_uid="round17_scientific_budget",
        dimensions=budget_dimensions, consumed={key: 0.0 for key in budget_dimensions},
        remaining=budget_dimensions, iteration_limit=int(budget_dimensions["loop_iterations"]),
        controller_uid=CANONICAL_CONTROLLER_UID, version=str(runtime_config["scientific_budget"]["version"]),
        provenance={"source": "Libs/configs/data/round17_scitil_runtime.yaml"},
    ).to_dict()
    risk_config = dict(runtime_config["risk_budget"])
    risk_budget = RiskBudget(
        schema_version=SCHEMA_VERSIONS["risk_budget"], risk_budget_uid="round17_risk_budget",
        allowed_risk_classes=list(risk_config["allowed_risk_classes"]), maximum_risk_class=str(risk_config["maximum_risk_class"]),
        production_execution_allowed=False, agent_may_increase_risk=False, controller_uid=CANONICAL_CONTROLLER_UID,
        version=str(risk_config["version"]), provenance={"source": "Libs/configs/data/round17_scitil_runtime.yaml"},
    ).to_dict()

    hypothesis = ScientificHypothesis(
        schema_version=SCHEMA_VERSIONS["scientific_hypothesis"], hypothesis_uid="hypothesis_round17_restricted_replay",
        mechanism_statement="The restricted replay may retain no-action numerical stability within its qualification domain.",
        target_variables=["constraint_violation_rate", "repeatability"],
        expected_outcome={"direction": "stable", "scope": "model_replay_only"},
        required_context={"baseline_class": "qualification_synthetic", "engine": "neuronsifter_round15_restricted_replay"},
        time_horizon=dict(round16_request["requested_time_horizon"]),
        potential_falsifier={"disconfirming_prediction": "A replay constraint violation or non-repeatable output refutes model-conditional stability.", "negative_control": "matched_no_action_control"},
        supporting_evidence=[str(round16_objects["result"]["result_uid"])], conflicting_evidence=["conflict_directional_pilot"],
        support_domain_ref=support_domain_ref, uncertainty={"biological_validity": "unknown", "model_scope": "restricted"},
        causal_effect_claim=False, clinical_efficacy_claim=False, hypothetical_as_empirical_support=False,
    ).to_dict()
    unsafe_hypothesis = dict(hypothesis)
    unsafe_hypothesis.update({"hypothesis_uid": "hypothesis_round17_unsafe_claim", "mechanism_statement": "The treatment is effective and clinically recommended.", "clinical_efficacy_claim": True})

    information_action = dict(round13_graph["information_actions"][0])
    observation_request = dict(round13_graph["observation_requests"][0])
    tool_request = next(
        row for row in round13_graph["tool_requests"] if row["tool_request_uid"] == "tool_request_supported_measurement"
    )
    tool_capability_match = next(
        row
        for row in round13_graph["tool_capability_matches"]
        if row["tool_request_uid"] == tool_request["tool_request_uid"] and row["state"] == "supported"
    )
    tool_availability = next(
        row
        for row in round13_graph["tool_availability_decisions"]
        if row["match_uid"] == tool_capability_match["match_uid"] and row["available"] is True
    )
    raw_directional = next(row for row in round13_graph["raw_tool_results"] if row["tool_result_uid"] == "raw_result_directional")
    raw_conflict = next(row for row in round13_graph["raw_tool_results"] if row["tool_result_uid"] == "raw_result_conflict")
    observation_directional = next(row for row in round13_graph["canonical_observations"] if row["source_result_uid"] == "raw_result_directional")
    observation_conflict = next(row for row in round13_graph["canonical_observations"] if row["source_result_uid"] == "raw_result_conflict")
    observation_qc_directional = next(
        row
        for row in round13_graph["observation_qc"]
        if row["source_result_uid"] == observation_directional["source_result_uid"]
    )
    assimilation_proposal = next(row for row in round13_graph["assimilation_proposals"] if row["proposal_uid"] == "assimilation_proposal_uncertainty_only")

    incomplete_action = dict(no_action)
    incomplete_action.update({"action_uid": "action_round17_incomplete_drug", "intervention_type": "pharmacological", "agent_set": ["compound_only"], "dose": None, "dose_unit": None, "route": None})

    p_hypothesis = _proposal(uid="proposal_a_hypothesis", agent=agents["ScientistAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="new_hypothesis", hypothesis_ref=str(hypothesis["hypothesis_uid"]), support_domain_ref=support_domain_ref)
    p_what_if = _proposal(uid="proposal_a_what_if", agent=agents["ScientistAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="what_if_candidate", what_if_request_ref=str(round16_request["request_uid"]), support_domain_ref=support_domain_ref, risk_class="restricted_computational")
    p_falsifier = _proposal(uid="proposal_a_falsifier", agent=agents["FalsifierAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="falsification_test", hypothesis_ref=str(hypothesis["hypothesis_uid"]), support_domain_ref=support_domain_ref)
    p_information = _proposal(uid="proposal_b_information", agent=agents["ExperimentAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="information_action_candidate", information_action_ref=str(information_action["information_action_uid"]), observation_request_ref=str(observation_request["request_uid"]), support_domain_ref=str(information_action["support_domain_ref"]), risk_class="external_tool_read_only")
    p_attack = _proposal(uid="proposal_c_direct_state_attack", agent=agents["ScientistAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="world_action_candidate", world_action_ref=str(no_action["action_uid"]), support_domain_ref=support_domain_ref, provenance={"source": "adversarial_fixture", "direct_state_update": {"posterior_write": True}})
    p_unsupported = _proposal(uid="proposal_d_unsupported_world_action", agent=agents["ScientistAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="world_action_candidate", world_action_ref=str(incomplete_action["action_uid"]), support_domain_ref=support_domain_ref)
    p_conflict = _proposal(uid="proposal_e_conflict_query", agent=agents["EvidenceAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="evidence_query", observation_request_ref=str(observation_request["request_uid"]), support_domain_ref=str(information_action["support_domain_ref"]), risk_class="external_tool_read_only")
    p_budget = _proposal(uid="proposal_f_budget_stop", agent=agents["ScientistAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="world_action_candidate", world_action_ref=str(no_action["action_uid"]), support_domain_ref=support_domain_ref)
    p_unsafe_raw = _proposal(uid="proposal_g_unsafe_claim_raw", agent=agents["ScientistAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="new_hypothesis", hypothesis_ref=str(unsafe_hypothesis["hypothesis_uid"]), support_domain_ref=support_domain_ref, claim_flags={"causal_effect_claim": True, "clinical_recommendation": True})
    p_unsafe = _proposal(uid="proposal_g_unsafe_claim_rejection", agent=agents["ScientistAgent"], belief_state_ref=str(initial_belief["belief_state_uid"]), proposal_type="new_hypothesis", hypothesis_ref=str(unsafe_hypothesis["hypothesis_uid"]), support_domain_ref=support_domain_ref, provenance={"source": "round17_claim_boundary_compiler", "compiled": True, "raw_request_ref": str(p_unsafe_raw["proposal_uid"]), "execution_authorized": False})
    proposals = [p_hypothesis, p_what_if, p_falsifier, p_information, p_attack, p_unsupported, p_conflict, p_budget, p_unsafe]

    remaining = dict(scientific_budget["remaining"])
    budget_decisions = [
        _budget_decision("budget_a", p_what_if["proposal_uid"], {"compute_seconds": 1.0}, "restricted_computational", "authorized", remaining, "within_budget"),
        _budget_decision("budget_b", p_information["proposal_uid"], {"tool_calls": 1.0}, "external_tool_read_only", "authorized", remaining, "within_budget"),
        _budget_decision("budget_c", p_attack["proposal_uid"], {"compute_seconds": 0.0}, "contract_only", "rejected_privilege", remaining, "direct_state_write_forbidden"),
        _budget_decision("budget_d", p_unsupported["proposal_uid"], {"compute_seconds": 0.0}, "contract_only", "authorized", remaining, "contract_validation_only"),
        _budget_decision("budget_e", p_conflict["proposal_uid"], {"evidence_retrievals": 1.0}, "external_tool_read_only", "authorized", remaining, "within_budget"),
        _budget_decision("budget_f", p_budget["proposal_uid"], {"compute_seconds": 999999.0}, "restricted_computational", "blocked_budget", remaining, "budget_exhausted"),
        _budget_decision("budget_g", p_unsafe["proposal_uid"], {"compute_seconds": 0.0}, "contract_only", "rejected_claim_boundary", remaining, "unsafe_claim_request"),
    ]
    budget_by_uid = {str(row["decision_uid"]): row for row in budget_decisions}

    h_gate_a = _gate(gate_class=HypothesisGateDecision, schema_key="hypothesis_gate_decision", uid="h_gate_a", gate_type="H-Gate", proposal_uid=str(p_hypothesis["proposal_uid"]), decision="admitted_for_testing", input_values={"proposal": p_hypothesis, "hypothesis": hypothesis}, reason="falsifiable_restricted_hypothesis")
    h_gate_a_falsifier = _gate(gate_class=HypothesisGateDecision, schema_key="hypothesis_gate_decision", uid="h_gate_a_falsifier", gate_type="H-Gate", proposal_uid=str(p_falsifier["proposal_uid"]), decision="admitted_for_testing", input_values={"proposal": p_falsifier, "hypothesis": hypothesis}, reason="disconfirming_prediction_registered_before_positive_summary")
    w_gate_a = _gate(gate_class=WorldActionGateDecision, schema_key="world_action_gate_decision", uid="w_gate_a", gate_type="W-Gate", proposal_uid=str(p_what_if["proposal_uid"]), decision="authorized_restricted_rollout", input_values={"proposal": p_what_if, "action": no_action, "request": round16_request}, reason="accepted_round16_restricted_replay_only")
    o_gate_a = _gate(gate_class=ObservationGateDecision, schema_key="observation_gate_decision", uid="o_gate_a", gate_type="O-Gate", proposal_uid=str(p_what_if["proposal_uid"]), decision="no_update", input_values={"rollout_result": round16_objects["result"], "output_class": round16_objects["rollouts"][0]["output_class"]}, reason="model_rollout_is_non_empirical_and_cannot_update_belief")
    i_gate_b = _gate(gate_class=InformationActionGateDecision, schema_key="information_action_gate_decision", uid="i_gate_b", gate_type="I-Gate", proposal_uid=str(p_information["proposal_uid"]), decision="authorized_restricted_information_action", input_values={"proposal": p_information, "information_action": information_action, "tool_request": tool_request, "tool_capability_match": tool_capability_match, "tool_availability": tool_availability}, reason="governed_read_only_existing_result", approved_scope={"adapter_uid": str(tool_request["tool_uid"])})
    assimilation_decision = {
        "schema_version": "neuronsifter-assimilation-decision-v1",
        "decision_uid": "assimilation_decision_round17_uncertainty_only",
        "proposal_uid": assimilation_proposal["proposal_uid"],
        "controller_uid": CANONICAL_CONTROLLER_UID,
        "validator_uid": ROUND17_VALIDATOR_UID,
        "prior_state_hash": initial_belief["content_hash"],
        "input_observation_hashes": {str(observation_directional["observation_uid"]): canonical_sha256(observation_directional)},
        "decision": "approved_uncertainty_only",
        "reason_codes": ["qc_c_directional_uncertainty_only"],
        "approved_update_scope": ["neuroinflammation_uncertainty"],
        "conflict_handling": {},
        "resulting_state_ref": "belief_state_round17_uncertainty_update",
        "provenance_ref": "round17_o_gate_b",
    }
    o_gate_b = _gate(gate_class=ObservationGateDecision, schema_key="observation_gate_decision", uid="o_gate_b", gate_type="O-Gate", proposal_uid=str(p_information["proposal_uid"]), decision="approved_uncertainty_only", input_values={"raw_result": raw_directional, "observation": observation_directional, "qc": observation_qc_directional, "assimilation_proposal": assimilation_proposal}, reason="governed_qc_c_uncertainty_only")
    successor_belief = create_belief_state_from_assimilation(
        parent=initial_belief, o_gate=o_gate_b, assimilation_decision=assimilation_decision,
        posterior_refs={"state_posterior_ref": str(initial_belief["state_posterior_ref"]), "parameter_posterior_ref": str(initial_belief["parameter_posterior_ref"]), "model_family_posterior_ref": str(initial_belief["model_family_posterior_ref"])},
        active_hypothesis_refs=[str(hypothesis["hypothesis_uid"])], constraint_state_ref=str(initial_belief["constraint_state_ref"]),
        support_domain_ref=str(initial_belief["support_domain_ref"]), twin_debt_ref="debt_round17_conflict",
        evidence_snapshot_ref=str(initial_belief["evidence_snapshot_ref"]), provenance={"source": "round17_o_gate_b", "update_type": "uncertainty_only", "agent_text_stored": False},
    )
    assimilation_decision["resulting_state_ref"] = successor_belief["belief_state_uid"]
    w_gate_d = _gate(gate_class=WorldActionGateDecision, schema_key="world_action_gate_decision", uid="w_gate_d", gate_type="W-Gate", proposal_uid=str(p_unsupported["proposal_uid"]), decision="blocked_incomplete_action", input_values={"proposal": p_unsupported, "action": incomplete_action}, reason="missing_action_fields")
    o_gate_c = _gate(gate_class=ObservationGateDecision, schema_key="observation_gate_decision", uid="o_gate_c", gate_type="O-Gate", proposal_uid=str(p_attack["proposal_uid"]), decision="no_update", input_values={"proposal": p_attack}, reason="agent_state_update_authority_forbidden")
    i_gate_e = _gate(gate_class=InformationActionGateDecision, schema_key="information_action_gate_decision", uid="i_gate_e", gate_type="I-Gate", proposal_uid=str(p_conflict["proposal_uid"]), decision="authorized_evidence_query", input_values={"proposal": p_conflict, "information_action": information_action, "tool_request": tool_request, "tool_capability_match": tool_capability_match, "tool_availability": tool_availability}, reason="typed_conflict_investigation", approved_scope={"adapter_uid": str(tool_request["tool_uid"])})
    o_gate_e = _gate(gate_class=ObservationGateDecision, schema_key="observation_gate_decision", uid="o_gate_e", gate_type="O-Gate", proposal_uid=str(p_conflict["proposal_uid"]), decision="no_update", input_values={"raw_result": raw_conflict, "observation": observation_conflict}, reason="conflict_retained_no_majority_resolution")
    w_gate_f = _gate(gate_class=WorldActionGateDecision, schema_key="world_action_gate_decision", uid="w_gate_f", gate_type="W-Gate", proposal_uid=str(p_budget["proposal_uid"]), decision="blocked_budget", input_values={"proposal": p_budget, "action": no_action, "budget_decision": budget_by_uid["budget_f"]}, reason="budget_exhausted_before_execution")
    h_gate_g = _gate(gate_class=HypothesisGateDecision, schema_key="hypothesis_gate_decision", uid="h_gate_g", gate_type="H-Gate", proposal_uid=str(p_unsafe["proposal_uid"]), decision="revision_required", input_values={"proposal": p_unsafe, "hypothesis": unsafe_hypothesis}, reason="unsafe_claim_boundary")
    gates = [h_gate_a, h_gate_a_falsifier, w_gate_a, o_gate_a, i_gate_b, o_gate_b, o_gate_c, w_gate_d, i_gate_e, o_gate_e, w_gate_f, h_gate_g]

    dual_actions = [
        DualActionDecision(schema_version=SCHEMA_VERSIONS["dual_action_decision"], decision_uid="dual_a", belief_state_ref=str(initial_belief["belief_state_uid"]), candidate_world_actions=[str(no_action["action_uid"])], candidate_information_actions=[str(information_action["information_action_uid"])], expected_utility={"world_action": 0.1, "information_action": 0.05}, expected_information_gain={"world_action": 0.0, "information_action": 0.1}, expected_twin_debt_change={"world_action": "unchanged", "information_action": "nonincreasing"}, cost={"bounded": True}, risk={"class": "restricted_computational"}, support_domain_ref=support_domain_ref, selected_action_type="world_action", selected_action_ref=str(no_action["action_uid"]), selection_reason=["low_debt_restricted_replay"], controller_uid=CANONICAL_CONTROLLER_UID).to_dict(),
        DualActionDecision(schema_version=SCHEMA_VERSIONS["dual_action_decision"], decision_uid="dual_b", belief_state_ref=str(initial_belief["belief_state_uid"]), candidate_world_actions=[str(no_action["action_uid"])], candidate_information_actions=[str(information_action["information_action_uid"])], expected_utility={"world_action": 0.1, "information_action": 0.05}, expected_information_gain={"world_action": 0.0, "information_action": 0.8}, expected_twin_debt_change={"world_action": "unresolved", "information_action": "decrease_possible"}, cost={"bounded": True}, risk={"class": "external_tool_read_only"}, support_domain_ref=str(information_action["support_domain_ref"]), selected_action_type="information_action", selected_action_ref=str(information_action["information_action_uid"]), selection_reason=["high_provenance_and_disagreement_debt"], controller_uid=CANONICAL_CONTROLLER_UID).to_dict(),
        DualActionDecision(schema_version=SCHEMA_VERSIONS["dual_action_decision"], decision_uid="dual_e", belief_state_ref=str(initial_belief["belief_state_uid"]), candidate_world_actions=[str(no_action["action_uid"])], candidate_information_actions=[str(information_action["information_action_uid"])], expected_utility={"world_action": 0.1, "information_action": 0.1}, expected_information_gain={"world_action": 0.0, "information_action": 0.8}, expected_twin_debt_change={"world_action": "unresolved", "information_action": "conflict_characterization"}, cost={"bounded": True}, risk={"class": "external_tool_read_only"}, support_domain_ref=str(information_action["support_domain_ref"]), selected_action_type="information_action", selected_action_ref=str(information_action["information_action_uid"]), selection_reason=["retained_disagreement_requires_information"], controller_uid=CANONICAL_CONTROLLER_UID).to_dict(),
    ]

    sandbox = {
        "adapter_uid": str(tool_request["tool_uid"]),
        "typed_request_schema": str(tool_request["input_schema_version"]),
        "read_only_code": True,
        "isolated_writable_runtime": True,
        "explicit_environment": {"network": "disabled", "source": "existing_governed_artifact"},
        "resource_limit": {"cpu_seconds": 1, "memory_mib": 64},
        "timeout_seconds": 30,
        "network_access_policy": "none",
        "output_hash": raw_directional["raw_artifact_hashes"]["result"],
        "stdout_ref": raw_directional["stdout_ref"],
        "stderr_ref": raw_directional["stderr_ref"],
        "terminal_state": "completed_existing_artifact_read",
        "failure_class": "none",
    }
    execution_attempt = {
        "attempt_uid": "round17_existing_governed_result_read_attempt",
        "adapter_uid": str(tool_request["tool_uid"]),
        "terminal_state": "COMPLETED",
        "exit_code": "0:0",
        "output_hashes": {"result": raw_directional["raw_artifact_hashes"]["result"]},
        "failure_class": "none",
        "preserved": True,
        "supersedes": None,
    }
    kernel = TwinKernel(controller_uid=CANONICAL_CONTROLLER_UID, allowlisted_adapters=dict(runtime_config["adapter_allowlist"]))

    transitions = {
        "scenario_a_valid_restricted_hypothesis_loop": [("initialized", "hypothesis_proposed"), ("hypothesis_proposed", "hypothesis_gated"), ("hypothesis_gated", "action_candidates_generated"), ("action_candidates_generated", "action_gated"), ("action_gated", "execution_authorized"), ("execution_authorized", "execution_completed"), ("execution_completed", "observation_normalized"), ("observation_normalized", "assimilation_proposed"), ("assimilation_proposed", "assimilation_gated"), ("assimilation_gated", "explanation_generated"), ("explanation_generated", "terminated")],
        "scenario_b_information_action_loop": [("initialized", "action_candidates_generated"), ("action_candidates_generated", "action_gated"), ("action_gated", "execution_authorized"), ("execution_authorized", "execution_completed"), ("execution_completed", "observation_normalized"), ("observation_normalized", "assimilation_proposed"), ("assimilation_proposed", "assimilation_gated"), ("assimilation_gated", "belief_updated"), ("belief_updated", "explanation_generated"), ("explanation_generated", "terminated")],
        "scenario_c_direct_state_update_attack": [("initialized", "blocked"), ("blocked", "terminated")],
        "scenario_d_unsupported_world_action": [("initialized", "action_candidates_generated"), ("action_candidates_generated", "action_gated"), ("action_gated", "blocked"), ("blocked", "terminated")],
        "scenario_e_conflict_preservation": [("initialized", "action_candidates_generated"), ("action_candidates_generated", "action_gated"), ("action_gated", "execution_authorized"), ("execution_authorized", "execution_completed"), ("execution_completed", "observation_normalized"), ("observation_normalized", "assimilation_proposed"), ("assimilation_proposed", "assimilation_gated"), ("assimilation_gated", "explanation_generated"), ("explanation_generated", "terminated")],
        "scenario_f_budget_exhaustion": [("initialized", "action_candidates_generated"), ("action_candidates_generated", "action_gated"), ("action_gated", "blocked"), ("blocked", "terminated")],
        "scenario_g_unsafe_claim_request": [("initialized", "hypothesis_proposed"), ("hypothesis_proposed", "hypothesis_gated"), ("hypothesis_gated", "blocked"), ("blocked", "terminated")],
    }
    scientist_uid = str(agents["ScientistAgent"]["agent_uid"])
    falsifier_uid = str(agents["FalsifierAgent"]["agent_uid"])
    experiment_uid = str(agents["ExperimentAgent"]["agent_uid"])
    evidence_uid = str(agents["EvidenceAgent"]["agent_uid"])
    scenario_bindings: dict[str, list[tuple[str | None, str | None]]] = {
        "scenario_a_valid_restricted_hypothesis_loop": [
            (scientist_uid, None),
            (scientist_uid, str(h_gate_a["gate_uid"])),
            (falsifier_uid, str(h_gate_a_falsifier["gate_uid"])),
            (scientist_uid, str(w_gate_a["gate_uid"])),
            (scientist_uid, str(w_gate_a["gate_uid"])),
            (None, str(w_gate_a["gate_uid"])),
            (None, str(w_gate_a["gate_uid"])),
            (falsifier_uid, str(h_gate_a_falsifier["gate_uid"])),
            (None, str(o_gate_a["gate_uid"])),
            (falsifier_uid, str(o_gate_a["gate_uid"])),
            (scientist_uid, None),
        ],
        "scenario_b_information_action_loop": [
            (experiment_uid, None),
            (experiment_uid, str(i_gate_b["gate_uid"])),
            (experiment_uid, str(i_gate_b["gate_uid"])),
            (None, str(i_gate_b["gate_uid"])),
            (None, str(i_gate_b["gate_uid"])),
            (None, None),
            (None, str(o_gate_b["gate_uid"])),
            (None, str(o_gate_b["gate_uid"])),
            (experiment_uid, None),
            (experiment_uid, None),
        ],
        "scenario_c_direct_state_update_attack": [(scientist_uid, str(o_gate_c["gate_uid"])), (None, str(o_gate_c["gate_uid"]))],
        "scenario_d_unsupported_world_action": [(scientist_uid, None), (scientist_uid, str(w_gate_d["gate_uid"])), (scientist_uid, str(w_gate_d["gate_uid"])), (None, str(w_gate_d["gate_uid"]))],
        "scenario_e_conflict_preservation": [(evidence_uid, None), (evidence_uid, str(i_gate_e["gate_uid"])), (evidence_uid, str(i_gate_e["gate_uid"])), (None, str(i_gate_e["gate_uid"])), (None, str(i_gate_e["gate_uid"])), (None, None), (None, str(o_gate_e["gate_uid"])), (None, str(o_gate_e["gate_uid"])), (evidence_uid, None)],
        "scenario_f_budget_exhaustion": [(scientist_uid, None), (scientist_uid, str(w_gate_f["gate_uid"])), (scientist_uid, str(w_gate_f["gate_uid"])), (None, str(w_gate_f["gate_uid"]))],
        "scenario_g_unsafe_claim_request": [(scientist_uid, None), (scientist_uid, str(h_gate_g["gate_uid"])), (scientist_uid, str(h_gate_g["gate_uid"])), (None, str(h_gate_g["gate_uid"]))],
    }
    events_by_scenario = {
        scenario: _events(
            scenario_uid=scenario,
            transitions=rows,
            created_at=created_at,
            authority_bindings=scenario_bindings[scenario],
        )
        for scenario, rows in transitions.items()
    }
    gap_uids = [str(row["gap_uid"]) for row in parent["gap_rows"]]
    terminal_states = {
        "scenario_a_valid_restricted_hypothesis_loop": "hypothesis_retained",
        "scenario_b_information_action_loop": "uncertainty_only_updated",
        "scenario_c_direct_state_update_attack": "blocked_by_gate",
        "scenario_d_unsupported_world_action": "unsupported_domain",
        "scenario_e_conflict_preservation": "conflict_registered",
        "scenario_f_budget_exhaustion": "budget_exhausted",
        "scenario_g_unsafe_claim_request": "blocked_by_gate",
    }
    budget_refs = {scenario: f"budget_{letter}" for scenario, letter in zip(transitions, "abcdefg", strict=True)}
    episodes = []
    for scenario in transitions:
        final_ref = str(successor_belief["belief_state_uid"]) if scenario == "scenario_b_information_action_loop" else str(initial_belief["belief_state_uid"])
        conflicts = ["conflict_directional_pilot"] if scenario in {"scenario_a_valid_restricted_hypothesis_loop", "scenario_e_conflict_preservation"} else []
        episodes.append(_episode(scenario_uid=scenario, events=events_by_scenario[scenario], initial_ref=str(initial_belief["belief_state_uid"]), final_ref=final_ref, terminal_state=terminal_states[scenario], budget_ref=budget_refs[scenario], gap_uids=gap_uids, conflict_refs=conflicts))
    all_events = [row for scenario in transitions for row in events_by_scenario[scenario]]

    ledger = HypothesisLedgerEntry(
        schema_version=SCHEMA_VERSIONS["hypothesis_ledger_entry"], hypothesis_uid=str(hypothesis["hypothesis_uid"]), version=1,
        parent_hypothesis=None, mechanism=str(hypothesis["mechanism_statement"]), expected_observations=["constraint_stability", "deterministic_repeatability"],
        falsification_criteria=[str(hypothesis["potential_falsifier"]["disconfirming_prediction"])], supporting_evidence=[str(round16_objects["result"]["result_uid"])],
        conflicting_evidence=["conflict_directional_pilot"], test_history=["scenario_a_valid_restricted_hypothesis_loop"],
        current_status="supported_model_conditionally", status_decision_ref=str(h_gate_a["gate_uid"]), support_score=0.5,
        falsification_score=0.5, unresolved_conflict_count=1, reliability_status="qualified_with_restrictions",
        support_domain_status="restricted_in_domain", falsifier_agent_ref=str(agents["FalsifierAgent"]["agent_uid"]),
    ).to_dict()
    explanations = [
        AgentExplanation(schema_version=SCHEMA_VERSIONS["agent_explanation"], explanation_uid="explanation_a", episode_uid="episode_scenario_a_valid_restricted_hypothesis_loop", belief_state_ref=str(initial_belief["belief_state_uid"]), gate_decision_refs=[str(h_gate_a["gate_uid"]), str(h_gate_a_falsifier["gate_uid"]), str(w_gate_a["gate_uid"]), str(o_gate_a["gate_uid"])], rollout_or_tool_artifact_refs=[str(round16_objects["result"]["result_uid"])], uncertainty_report_ref=str(round16_objects["uncertainty"]["uncertainty_uid"]), reliability_report_ref=str(round16_objects["reliability"]["reliability_uid"]), support_domain_report_ref="round16_support_domain_comparison", conflicts=["conflict_directional_pilot"], claim_eligibility_ref=str(round16_objects["eligibility"]["decision_uid"]), summary="The restricted replay retained model-conditional stability after a linked disconfirming check; the typed disagreement remains unresolved and O-Gate authorized no empirical assimilation.", limitations=["Qualification-only synthetic baseline", "No empirical, causal, or real-world action conclusion"], model_based_counterfactual=True, empirical_observation=False, causal_effect_claim=False, clinical_recommendation=False, provenance={"accepted_artifacts_only": True}).to_dict(),
        AgentExplanation(schema_version=SCHEMA_VERSIONS["agent_explanation"], explanation_uid="explanation_b", episode_uid="episode_scenario_b_information_action_loop", belief_state_ref=str(successor_belief["belief_state_uid"]), gate_decision_refs=[str(i_gate_b["gate_uid"]), str(o_gate_b["gate_uid"])], rollout_or_tool_artifact_refs=[str(raw_directional["tool_result_uid"]), str(observation_directional["observation_uid"])], uncertainty_report_ref="round17_uncertainty_only_update", reliability_report_ref="round13_observation_qc_directional", support_domain_report_ref=str(information_action["support_domain_ref"]), conflicts=[], claim_eligibility_ref="round13_directional_only", summary="The governed existing result authorized an uncertainty-only update.", limitations=["No direct state-value overwrite", "No autonomous tool execution"], model_based_counterfactual=False, empirical_observation=False, causal_effect_claim=False, clinical_recommendation=False, provenance={"accepted_artifacts_only": True}).to_dict(),
        AgentExplanation(schema_version=SCHEMA_VERSIONS["agent_explanation"], explanation_uid="explanation_e", episode_uid="episode_scenario_e_conflict_preservation", belief_state_ref=str(initial_belief["belief_state_uid"]), gate_decision_refs=[str(i_gate_e["gate_uid"]), str(o_gate_e["gate_uid"])], rollout_or_tool_artifact_refs=[str(raw_conflict["tool_result_uid"]), str(observation_conflict["observation_uid"])], uncertainty_report_ref="round13_conflict_uncertainty", reliability_report_ref="round13_observation_qc_conflict", support_domain_report_ref=str(information_action["support_domain_ref"]), conflicts=["conflict_directional_pilot"], claim_eligibility_ref="stress_only", summary="The governed evidence disagreement remains typed and unresolved; no majority resolution was applied.", limitations=["Conflict-register information only", "No posterior value update"], model_based_counterfactual=False, empirical_observation=False, causal_effect_claim=False, clinical_recommendation=False, provenance={"accepted_artifacts_only": True}).to_dict(),
        AgentExplanation(schema_version=SCHEMA_VERSIONS["agent_explanation"], explanation_uid="explanation_g", episode_uid="episode_scenario_g_unsafe_claim_request", belief_state_ref=str(initial_belief["belief_state_uid"]), gate_decision_refs=[str(h_gate_g["gate_uid"])], rollout_or_tool_artifact_refs=["none_execution_blocked"], uncertainty_report_ref="unknown_not_zero", reliability_report_ref="not_executed", support_domain_report_ref=support_domain_ref, conflicts=[], claim_eligibility_ref="blocked_claim_boundary", summary="The unsafe claim scope was blocked before execution.", limitations=["No scientific result was produced"], model_based_counterfactual=False, empirical_observation=False, causal_effect_claim=False, clinical_recommendation=False, provenance={"accepted_artifacts_only": True}).to_dict(),
    ]

    gap_rows = _gap_rows(parent["gap_rows"], [str(row["proposal_uid"]) for row in proposals], [str(row["episode_uid"]) for row in episodes])
    negative_cases = {
        "direct_state_update_attack": {"blockers": validate_agent_proposal(p_attack, agent=agents["ScientistAgent"], known_belief_states={str(initial_belief["belief_state_uid"])}), "execution_occurred": False},
        "incomplete_world_action": {"blockers": validate_world_action(incomplete_action), "execution_occurred": False},
        "budget_exhaustion": {"decision": budget_by_uid["budget_f"], "execution_occurred": False},
        "unsafe_claim_request": {"raw_proposal": p_unsafe_raw, "proposal_blockers": validate_agent_proposal(p_unsafe_raw, agent=agents["ScientistAgent"], known_belief_states={str(initial_belief["belief_state_uid"])}), "hypothesis": unsafe_hypothesis, "hypothesis_blockers": validate_hypothesis(unsafe_hypothesis), "execution_occurred": False},
        "failed_tool_attempt_overwrite": {
            "blockers": validate_attempt_preservation(
                [
                    {**execution_attempt, "terminal_state": "FAILED", "failure_class": "intentional_negative_fixture"},
                    {**execution_attempt, "terminal_state": "COMPLETED"},
                ]
            ),
            "execution_occurred": False,
        },
    }

    blockers: list[dict[str, object]] = []
    for agent in agents.values():
        blockers += validate_agent_identity(agent)
    blockers += validate_belief_state(initial_belief, allow_initial=True)
    blockers += validate_belief_state(successor_belief)
    for proposal in (p_hypothesis, p_what_if, p_falsifier, p_information, p_unsupported, p_conflict, p_budget):
        blockers += validate_agent_proposal(proposal, agent=agents[str(proposal["agent_role"])], known_belief_states={str(initial_belief["belief_state_uid"]), str(successor_belief["belief_state_uid"])})
    blockers += validate_hypothesis(hypothesis)
    blockers += validate_h_gate(h_gate_a, proposal=p_hypothesis, hypothesis=hypothesis)
    blockers += validate_w_gate(w_gate_a, proposal=p_what_if, world_action=no_action, what_if_request=round16_request, twin_debt=low_debt, budget_decision=budget_by_uid["budget_a"])
    blockers += validate_i_gate(i_gate_b, proposal=p_information, information_action=information_action, tool_request=tool_request, allowlisted_adapters=kernel.adapter_uids, budget_decision=budget_by_uid["budget_b"], tool_capability_match=tool_capability_match, tool_availability_decision=tool_availability)
    blockers += validate_o_gate(o_gate_b, proposal=p_information, assimilation_proposal=assimilation_proposal, assimilation_decision=assimilation_decision)
    blockers += validate_o_gate(o_gate_c, proposal=p_attack, assimilation_proposal={}, assimilation_decision={})
    blockers += validate_w_gate(w_gate_d, proposal=p_unsupported, world_action=incomplete_action, what_if_request=None, twin_debt=low_debt, budget_decision=budget_by_uid["budget_d"])
    blockers += validate_i_gate(i_gate_e, proposal=p_conflict, information_action=information_action, tool_request=tool_request, allowlisted_adapters=kernel.adapter_uids, budget_decision=budget_by_uid["budget_e"], tool_capability_match=tool_capability_match, tool_availability_decision=tool_availability)
    blockers += validate_o_gate(o_gate_e, proposal=p_conflict, assimilation_proposal={}, assimilation_decision={})
    blockers += validate_w_gate(w_gate_f, proposal=p_budget, world_action=no_action, what_if_request=None, twin_debt=low_debt, budget_decision=budget_by_uid["budget_f"])
    blockers += validate_h_gate(h_gate_g, proposal=p_unsafe, hypothesis=unsafe_hypothesis)
    for debt in debts:
        blockers += validate_twin_debt(debt)
    for dual, debt in zip(dual_actions, (low_debt, high_debt, conflict_debt), strict=True):
        blockers += validate_dual_action_decision(dual, twin_debt=debt)
    for decision in budget_decisions:
        blockers += validate_budget_contracts(scientific_budget, risk_budget, decision)
    blockers += validate_tool_sandbox(sandbox, allowlisted_adapters=set(kernel.adapter_uids))
    blockers += validate_attempt_preservation([execution_attempt])
    for event in all_events:
        blockers += validate_loop_event(event)
    for episode in episodes:
        blockers += validate_episode(episode, events=events_by_scenario[str(episode["scenario_uid"])])
    blockers += validate_hypothesis_ledger_entry(ledger)
    for explanation in explanations:
        blockers += validate_explanation(explanation)
    blockers += validate_frozen_gap_map(gap_rows)
    blockers += validate_unique_uids(proposals, ("proposal_uid",))
    blockers += validate_unique_uids(gates, ("gate_uid",))
    blockers += validate_unique_uids(all_events, ("event_uid",))
    blockers += validate_unique_uids(episodes, ("episode_uid",))
    lineage_edges = [(str(initial_belief["belief_state_uid"]), str(successor_belief["belief_state_uid"]))]
    lineage_edges += [(str(row["proposal_uid"]), str(gate["gate_uid"])) for row, gate in ((p_hypothesis, h_gate_a), (p_falsifier, h_gate_a_falsifier), (p_what_if, w_gate_a), (p_what_if, o_gate_a), (p_information, i_gate_b), (p_information, o_gate_b), (p_attack, o_gate_c), (p_unsupported, w_gate_d), (p_conflict, i_gate_e), (p_conflict, o_gate_e), (p_budget, w_gate_f), (p_unsafe, h_gate_g))]
    blockers += validate_lineage_acyclic(lineage_edges)
    if blockers:
        raise ValueError(sorted(blockers, key=lambda row: (str(row.get("code", "")), json.dumps(row, sort_keys=True))))
    if (
        not negative_cases["direct_state_update_attack"]["blockers"]
        or not negative_cases["incomplete_world_action"]["blockers"]
        or not negative_cases["unsafe_claim_request"]["proposal_blockers"]
        or not negative_cases["unsafe_claim_request"]["hypothesis_blockers"]
        or not negative_cases["failed_tool_attempt_overwrite"]["blockers"]
    ):
        raise ValueError("Round-17 negative pilot fixtures did not fail closed.")

    catalog_map = {
        "agent_identity_catalog": [_catalog_row("AgentIdentity", "registered_restricted", row) for row in agents.values()],
        "hypothesis_ledger": [_catalog_row("HypothesisLedgerEntry", str(ledger["current_status"]), ledger)],
        "agent_proposal_catalog": [
            _catalog_row(
                "AgentProposal",
                "rejected_contract"
                if row["proposal_uid"] == p_attack["proposal_uid"]
                else "compiled_claim_boundary_rejection"
                if row["proposal_uid"] == p_unsafe["proposal_uid"]
                else "compiled",
                row,
            )
            for row in proposals
        ],
        "gate_decision_catalog": [_catalog_row(str(row["gate_type"]), str(row["decision"]), row) for row in gates],
        "agent_loop_event_catalog": [_catalog_row("AgentLoopEvent", str(row["next_state"]), row) for row in all_events],
        "agent_loop_episode_catalog": [_catalog_row("AgentLoopEpisode", str(row["terminal_state"]), row) for row in episodes],
        "twin_debt_catalog": [_catalog_row("TwinDebt", "componentwise", row) for row in debts],
        "budget_decision_catalog": [_catalog_row("AgentLoopBudgetDecision", str(row["decision"]), row) for row in budget_decisions],
        "explanation_catalog": [_catalog_row("AgentExplanation", "claim_boundary_passed", row) for row in explanations],
    }
    for name, rows in catalog_map.items():
        _write_catalog(_resolve(output_root, paths[name]), rows)
    write_tsv(
        _resolve(output_root, paths["frozen_gap_map"]),
        ["gap_uid", "parent_status", "proposal_uids", "episode_uids", "round17_impact", "resolved_by_agent_reasoning", "independent_admitted_evidence_ref", "remaining_blocker", "frozen_gap_count_after_round17"],
        gap_rows,
    )
    graph = {
        "schema_version": "neuronsifter-round17-scitil-graph-v1",
        "created_at": created_at,
        "code_commit": commit,
        "input_artifact_hashes": parent["hashes"],
        "environment": env,
        "objects": {
            "belief_states": [initial_belief, successor_belief], "agents": list(agents.values()), "hypotheses": [hypothesis],
            "proposals": proposals, "gates": gates, "world_actions": [no_action], "what_if_requests": [round16_request],
            "information_actions": [information_action], "observation_requests": [observation_request],
            "observation_chain": {"tool_request": tool_request, "tool_capability_match": tool_capability_match, "tool_availability": tool_availability, "raw_directional": raw_directional, "observation_directional": observation_directional, "qc_directional": observation_qc_directional, "assimilation_proposal": assimilation_proposal, "assimilation_decision": assimilation_decision, "raw_conflict": raw_conflict, "observation_conflict": observation_conflict},
            "dual_action_decisions": dual_actions, "twin_debts": debts, "scientific_budget": scientific_budget,
            "risk_budget": risk_budget, "budget_decisions": budget_decisions, "sandbox_contracts": [sandbox],
            "execution_attempts": [execution_attempt],
            "hypothesis_ledger_entries": [ledger], "events": all_events, "episodes": episodes, "explanations": explanations,
            "negative_cases": negative_cases,
        },
        "lineage_edges": lineage_edges,
        "frozen_gap_count": 20,
        "frozen_gap_resolved_by_agent_reasoning": 0,
        "frozen_gap_resolved_by_model_rollout": 0,
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
        "autonomous_production_experiment": False,
    }
    write_json(_resolve(output_root, paths["graph"]), graph)
    artifact_paths = {name: _resolve(output_root, paths[name]) for name in (*catalog_map, "graph", "frozen_gap_map")}
    artifact_hashes = {name: sha256_file(path) for name, path in artifact_paths.items()}
    release = {
        "schema_version": SCHEMA_VERSIONS["release"],
        "parent_release_uid": parent["release"]["release_uid"],
        "created_at": created_at,
        "code_commit": commit,
        "config_hash": sha256_file(repository_root / "Libs/configs/data/round17_scitil_runtime.yaml"),
        "schema_registry_hash": sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        "input_bundle_hash": input_bundle_hash,
        "environment_hash": environment_hash,
        "artifact_hashes": artifact_hashes,
        "belief_state_count": 2,
        "proposal_count": len(proposals),
        "gate_decision_count": len(gates),
        "episode_count": len(episodes),
        "empirical_observation_count": 0,
        "frozen_gap_count": 20,
        "resolved_by_agent_reasoning": 0,
        "autonomous_discovery_claim": False,
        "status": "materialized_unsealed",
        "sealed": False,
        "dt_core_status": "not_evaluated",
    }
    release_hash = release_content_hash(release)
    release["release_uid"] = f"evidence_fabric_round17_release_{release_hash[:24]}"
    release["release_content_hash"] = release_hash
    release_path = _resolve(output_root, paths["release_root"]) / str(release["release_uid"]) / "round17_scitil_release_candidate.json"
    write_json(release_path, release)
    release_blockers = validate_release_candidate(release)
    if release_blockers:
        raise ValueError(release_blockers)
    pilot_validation = {
        "schema_version": "neuronsifter-dt-round17-pilot-validation-v1",
        "status": "passed",
        "blocker_count": 0,
        "blockers": [],
        "scenario_count": 7,
        "scenario_terminal_states": terminal_states,
        "belief_state_count": 2,
        "agent_count": 4,
        "proposal_count": len(proposals),
        "gate_count": len(gates),
        "event_count": len(all_events),
        "episode_count": len(episodes),
        "falsifier_participation": True,
        "direct_state_attack_rejected": True,
        "arbitrary_shell_execution": False,
        "autonomous_production_execution": False,
        "frozen_gap_count": 20,
        "frozen_gap_resolved_by_agent_reasoning": 0,
        "release_uid": release["release_uid"],
        "release_path": str(release_path.relative_to(output_root)),
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
    }
    write_json(_resolve(output_root, paths["pilot_validation"]), pilot_validation)
    return {
        "parent": parent,
        "graph": graph,
        "release": release,
        "release_path": release_path,
        "pilot_validation": pilot_validation,
        "artifact_paths": artifact_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic Round-17 SciTiL contract pilot.")
    parser.add_argument("--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    repository_root = Path(args.repository_root).resolve()
    result = build_round17_pilot(
        config=load_config(Path(args.config)),
        repository_root=repository_root,
        parent_root=Path(args.parent_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
    )
    print(
        f"round17_pilot_{result['pilot_validation']['status']} "
        f"scenarios={result['pilot_validation']['scenario_count']} gaps={result['pilot_validation']['frozen_gap_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
