"""Integrated development replay for the Agent-in-Digital-Twin runtime.

This module exercises the existing Twin, WorldAction, memory, typed-tool,
checkpoint, and co-evolution boundaries in one deterministic CPU fixture.  It
is an engineering qualification artifact only: the fixture is not a
therapeutic outcome source and cannot open the formal evidence gate.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from Libs.models.modules.agent_in_twin_runtime import TwinGroundedReActController
from Libs.models.modules.coevolution_runtime import (
    AgentTwinCoEvolutionRuntime,
    CoEvolutionVariant,
)
from Libs.models.modules.decision_consistent_runtime import (
    DecisionConsistentTwin,
    InformationAction,
    ScientificPosterior,
    ToolResult,
    WorldAction,
)
from Libs.models.modules.intervention_optimizer import InterventionOptimizer
from Libs.models.modules.scientific_memory import MemoryEntry, ScientificMemory
from Libs.models.modules.scientific_tool_fabric import ScientificToolFabric
from Libs.scripts.services.neuronsifter_mcp import build_scientific_mcp_servers


ENGINEERING_CLOSURE_SCHEMA_VERSION = "neuronsifter-agent-in-twin-engineering-closure-v1"
FORMAL_EVIDENCE_GATE = "BLOCKED_REQUIRES_EXTERNAL_MULTI_ARM_FUNCTIONAL_OUTCOMES"
ENGINEERING_STATUS = "ENGINEERING_READY_AWAITING_EXTERNAL_THERAPEUTIC_EVIDENCE"
SCENARIOS = (
    "normal_multi_round_loop",
    "tool_failure_and_partial_result",
    "contradictory_observation_admission",
    "checkpoint_restart_replay",
    "memory_invalidation_and_truth_isolation",
    "unsupported_action_abstention",
    "tool_boundary_and_idempotence",
    "bidirectional_coevolution",
)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_rollout(posterior: ScientificPosterior, action: WorldAction) -> dict[str, Any]:
    """Deterministic replay dynamics; it is explicitly development-only."""
    observations = posterior.state.get("assimilated_observations", ())
    preference = 0.0
    if observations and isinstance(observations[-1], Mapping):
        raw_preference = observations[-1].get("preference", 0.0)
        if isinstance(raw_preference, (int, float)):
            preference = float(raw_preference)
    if action.drug_set == ("candidate_alpha",):
        benefit = 1.0 + 0.4 * preference
    else:
        benefit = 0.8 - 0.4 * preference
    safety = float(action.biological_context.get("safety_risk", 0.0))
    return {
        "trajectory": [preference, benefit],
        "uncertainty": {"epistemic": 0.05, "aleatoric": 0.02},
        "diagnostics": {
            "benefit": benefit,
            "safety_risk": safety,
            "status": "development_simulator",
            "development_only": True,
        },
    }


def development_world_actions() -> tuple[WorldAction, WorldAction]:
    return (
        WorldAction(
            target_set=("amyloid_beta",),
            drug_set=("candidate_alpha",),
            dose=1.0,
            schedule="q24h",
            route="iv",
            modality="pharmacological",
            duration="7d",
            biological_context={"safety_risk": 0.05},
        ),
        WorldAction(
            target_set=("neuroinflammation",),
            drug_set=("candidate_beta",),
            dose=1.0,
            schedule="q24h",
            route="iv",
            modality="pharmacological",
            duration="7d",
            biological_context={"safety_risk": 0.05},
        ),
    )


def development_information_action(tool_id: str = "evidence.functional.replay") -> InformationAction:
    return InformationAction(
        tool_id=tool_id,
        query={
            "expected_decision_relevance": 0.9,
            "tool_reliability": 0.95,
            "measurement": "functional_replay_proxy",
        },
        fidelity="development_replay",
        expected_cost=1.0,
    )


def new_development_twin() -> DecisionConsistentTwin:
    twin = DecisionConsistentTwin(rollout_fn=development_rollout)
    twin.infer_belief(
        {
            "state": {"baseline": 0.2},
            "uncertainty": {"epistemic": 0.2, "aleatoric": 0.1},
            "support": {"support_score": 1.0},
            "reliability": {"tri_score": 1.0},
        },
        {"regions": ["hippocampus"], "compartments": ["isf"]},
    )
    return twin


def _register_observation_tool(
    fabric: ScientificToolFabric,
    *,
    mode: str = "success",
    calls: list[int] | None = None,
) -> str:
    tool_id = f"evidence.functional.replay.{mode}"

    def handler(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if calls is not None:
            calls.append(1)
        if mode == "failure":
            return {
                "status": "failed",
                "failure_class": "timeout",
                "error": "development fixture timeout",
            }
        if mode == "partial":
            return {
                "status": "partial",
                "partial": True,
                "observation": {"preference": -1.0, "functional_signal": 0.4},
                "uncertainty": {"epistemic": 0.12},
                "evidence_grade": "development_diagnostic",
            }
        return {
            "status": "completed",
            "observation": {"preference": -1.0, "functional_signal": 0.65},
            "uncertainty": {"epistemic": 0.05, "aleatoric": 0.05},
            "support": {"support_score": 0.9},
            "evidence_grade": "development_diagnostic",
            "cost": 1.0,
            "runtime": 0.01,
            "input_echo": payload.get("information_action_uid"),
        }

    fabric.register(
        tool_id,
        boundary="neuronsifter-evidence-mcp",
        handler=handler,
        version="development-replay-1",
    )
    return tool_id


def _run_normal_loop(checkpoint_path: Path) -> dict[str, Any]:
    twin = new_development_twin()
    memory = ScientificMemory()
    fabric = ScientificToolFabric()
    tool_id = _register_observation_tool(fabric)
    information_action = development_information_action(tool_id)
    controller = TwinGroundedReActController(
        twin,
        memory=memory,
        tools=fabric,
        optimizer=InterventionOptimizer(minimum_support=0.5, minimum_tri=0.5),
        max_rounds=2,
        tool_budget=1,
    )
    result = controller.run("optimize CNS functional rescue", development_world_actions(), (information_action,))
    if len(result.decisions) < 2:
        raise AssertionError("normal loop did not reach the post-assimilation re-ranking round")
    first = result.decisions[0]
    if first.posterior_uid_post == first.posterior_uid_pre:
        raise AssertionError("normal loop did not create an assimilation child")
    if first.selected_action_uid_post == first.selected_action_uid:
        raise AssertionError("development observation did not change the intervention ranking")
    trace_invariant = all(
        len(
            {
                str(row.get(field))
                for field in (
                    "posterior_uid_rollout",
                    "posterior_uid_query",
                    "posterior_uid_assimilation_parent",
                    "posterior_uid_decision",
                    "posterior_uid_for_intervention_ranking",
                )
                if row.get(field) is not None
            }
        ) <= 1
        for row in twin.decisions.trace
    )
    memory_path = checkpoint_path.with_name("development_memory.json")
    memory_path.write_text(
        json.dumps(_canonical(memory.to_record()), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    twin.save_checkpoint(checkpoint_path)
    restored = DecisionConsistentTwin.load_checkpoint(
        checkpoint_path,
        rollout_fn=development_rollout,
        tools={},
    )
    current = twin.belief.get(twin.decisions.current_posterior_uid or "")
    restored_current = restored.belief.get(restored.decisions.current_posterior_uid or "")
    exact_restore = (
        current.to_record() == restored_current.to_record()
        and twin.decisions.trace == restored.decisions.trace
    )
    restored_memory = ScientificMemory.from_record(
        json.loads(memory_path.read_text(encoding="utf-8")),
        posterior_validator=restored.belief.get,
    )
    restored_memory_hits = restored_memory.retrieve(posterior_uid=current.posterior_uid)
    replay_controller = TwinGroundedReActController(
        restored,
        memory=restored_memory,
        tools=ScientificToolFabric(),
        optimizer=InterventionOptimizer(minimum_support=0.5, minimum_tri=0.5),
        max_rounds=1,
        tool_budget=0,
    )
    replay = replay_controller.run("continue CNS functional rescue", development_world_actions())
    if not replay.decisions or replay.decisions[0].posterior_uid_pre != current.posterior_uid:
        raise AssertionError("restored controller did not consume the exact active posterior")
    if replay.decisions[0].reflection_summary.get("memory_hit_count", 0) < 1:
        raise AssertionError("restored controller did not retrieve persisted scientific memory")
    checkpoint_digest = sha256_file(checkpoint_path)
    return {
        "passed": True,
        "initial_posterior_uid": first.posterior_uid_pre,
        "post_observation_posterior_uid": first.posterior_uid_post,
        "initial_posterior_digest": twin.belief.get(first.posterior_uid_pre).content_digest,
        "post_observation_posterior_digest": current.content_digest,
        "selected_action_uid_pre": first.selected_action_uid,
        "selected_action_uid_post": first.selected_action_uid_post,
        "ranked_action_uids_pre": list(first.ranked_action_uids_pre),
        "ranked_action_uids_post": list(first.ranked_action_uids_post),
        "ranking_posterior_uid_pre": first.ranking_posterior_uid_pre,
        "ranking_posterior_uid_post": first.ranking_posterior_uid_post,
        "tool_call_uid": first.tool_call_uid,
        "observation_uid": first.observation_uid,
        "memory_uids": list(first.memory_uids),
        "trace_invariant": trace_invariant,
        "exact_restore": exact_restore,
        "memory_restore": bool(restored_memory_hits),
        "memory_snapshot_path": str(memory_path),
        "memory_snapshot_sha256": sha256_file(memory_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "replay_posterior_uid": replay.decisions[0].posterior_uid_pre,
        "replay_selected_action_uid": replay.decisions[0].selected_action_uid,
    }


def _run_failure_and_partial() -> dict[str, Any]:
    failure_twin = new_development_twin()
    failure_fabric = ScientificToolFabric()
    failure_tool = _register_observation_tool(failure_fabric, mode="failure")
    failure_action = development_information_action(failure_tool)
    failure_result = TwinGroundedReActController(
        failure_twin,
        tools=failure_fabric,
        max_rounds=1,
        tool_budget=1,
    ).run("handle timeout", development_world_actions(), (failure_action,))
    failure_decision = failure_result.decisions[0]
    failure_receipt = failure_fabric.invoke(
        failure_tool,
        posterior_uid=failure_decision.posterior_uid_pre,
        inputs={
            "information_action": failure_action.to_record(),
            "information_action_uid": failure_action.action_uid,
            "parent_posterior_uid": failure_decision.posterior_uid_pre,
        },
    )

    partial_twin = new_development_twin()
    partial_fabric = ScientificToolFabric()
    partial_tool = _register_observation_tool(partial_fabric, mode="partial")
    partial_action = development_information_action(partial_tool)
    partial_result = TwinGroundedReActController(
        partial_twin,
        tools=partial_fabric,
        max_rounds=1,
        tool_budget=1,
    ).run("handle partial assay", development_world_actions(), (partial_action,))
    partial_decision = partial_result.decisions[0]
    partial_receipt = partial_fabric.invoke(
        partial_tool,
        posterior_uid=partial_decision.posterior_uid_pre,
        inputs={
            "information_action": partial_action.to_record(),
            "information_action_uid": partial_action.action_uid,
            "parent_posterior_uid": partial_decision.posterior_uid_pre,
        },
    )
    return {
        "passed": (
            failure_receipt.status == "failed"
            and failure_receipt.failure_class == "timeout"
            and failure_decision.failure_reason is not None
            and partial_receipt.status == "partial"
            and partial_decision.posterior_uid_post != partial_decision.posterior_uid_pre
            and partial_decision.observation_uid == partial_receipt.observation_uid
        ),
        "failure": {
            "receipt_status": failure_receipt.status,
            "failure_class": failure_receipt.failure_class,
            "decision_failure_reason": failure_decision.failure_reason,
        },
        "partial": {
            "receipt_status": partial_receipt.status,
            "observation_uid": partial_receipt.observation_uid,
            "posterior_changed": partial_decision.posterior_uid_post != partial_decision.posterior_uid_pre,
        },
    }


def _run_contradiction_and_abstention() -> dict[str, Any]:
    twin = new_development_twin()
    parent = twin.belief.get(twin.decisions.current_posterior_uid or "")
    action = development_information_action("evidence.contradiction.replay")
    first = ToolResult(
        posterior_uid=parent.posterior_uid,
        action_uid=str(action.action_uid),
        status="completed",
        observation={"preference": -1.0},
        uncertainty={"epistemic": 0.05},
        observation_uid="observation-contradiction-a",
        provenance={"contradiction_group_uid": "development-contradiction"},
    )
    first_outcome = twin.assimilate_observation(parent.posterior_uid, first)
    second = ToolResult(
        posterior_uid=first_outcome.posterior.posterior_uid,
        action_uid=str(action.action_uid),
        status="completed",
        observation={"preference": 1.0},
        uncertainty={"epistemic": 0.08},
        observation_uid="observation-contradiction-b",
        provenance={
            "contradiction_group_uid": "development-contradiction",
            "contradicts_observation_uid": first.observation_uid,
        },
    )
    second_outcome = twin.assimilate_observation(first_outcome.posterior.posterior_uid, second)
    observations = second_outcome.posterior.state.get("assimilated_observations", ())
    preferences = tuple(
        float(row["preference"])
        for row in observations
        if isinstance(row, Mapping) and "preference" in row
    )

    unsupported_twin = new_development_twin()
    unsupported_parent = unsupported_twin.belief.get(unsupported_twin.decisions.current_posterior_uid or "")
    unsupported_action = WorldAction(
        target_set=("unsupported_target",),
        drug_set=("candidate_ood",),
        dose=3.0,
        schedule="q24h",
        route="intrathecal",
        modality="pharmacological",
        duration="7d",
    )
    unsupported_optimizer = InterventionOptimizer(
        minimum_support=0.5,
        minimum_tri=0.5,
        allowed_routes=("iv",),
        dose_bounds=(0.0, 1.0),
    )
    unsupported_result = unsupported_optimizer.optimize(
        unsupported_parent,
        [unsupported_action],
        rollouts={str(unsupported_action.action_uid): unsupported_twin.rollout(unsupported_parent.posterior_uid, unsupported_action)},
    )
    wrong_parent_rejected = False
    try:
        unsupported_twin.assimilate_observation(
            unsupported_parent.posterior_uid,
            ToolResult(
                posterior_uid="posterior-wrong-parent",
                action_uid=str(action.action_uid),
                status="completed",
                observation={"value": 1.0},
            ),
        )
    except ValueError:
        wrong_parent_rejected = True
    return {
        "passed": (
            first_outcome.accepted
            and second_outcome.accepted
            and preferences == (-1.0, 1.0)
            and unsupported_result.status == "INSUFFICIENT_EVIDENCE"
            and wrong_parent_rejected
        ),
        "contradictory_observations_retained": list(preferences),
        "unsupported_action_status": unsupported_result.status,
        "unsupported_action_violations": list(unsupported_result.ranked[0].violations),
        "wrong_parent_rejected": wrong_parent_rejected,
    }


def _run_memory_and_tool_boundaries() -> dict[str, Any]:
    memory = ScientificMemory()
    twin = new_development_twin()
    posterior = twin.belief.get(twin.decisions.current_posterior_uid or "")
    memory.bind_posterior_validator(twin.belief.get)
    entry_uid = memory.write(
        MemoryEntry(
            memory_type="semantic",
            structured_summary={"claim": "development-only mechanism note"},
            posterior_uid=posterior.posterior_uid,
            evidence_grade="development_diagnostic",
        )
    )
    digest_before = posterior.content_digest
    memory.invalidate(entry_uid, reason="development contradiction fixture")
    invalidated_hidden = not memory.retrieve(posterior_uid=posterior.posterior_uid)
    try:
        memory.write(
            MemoryEntry(
                memory_type="episodic",
                structured_summary={"claim": "stale"},
                posterior_uid="posterior-does-not-exist",
            )
        )
    except ValueError:
        stale_memory_rejected = True
    else:
        stale_memory_rejected = False
    if twin.belief.get(posterior.posterior_uid).content_digest != digest_before:
        raise AssertionError("memory write changed Twin scientific state")

    calls: list[int] = []
    fabric = ScientificToolFabric()
    tool_id = _register_observation_tool(fabric, calls=calls)
    inputs = {"input_manifest_uid": "development-manifest", "information_action_uid": "development-info"}
    receipt_a = fabric.invoke(tool_id, posterior_uid=posterior.posterior_uid, inputs=inputs)
    receipt_b = fabric.invoke(tool_id, posterior_uid=posterior.posterior_uid, inputs=inputs)
    idempotent = receipt_a.to_record() == receipt_b.to_record() and len(calls) == 1
    mutating_tool = "twin.mutating.fixture"
    fabric.register(
        mutating_tool,
        boundary="neuronsifter-twin-mcp",
        handler=lambda _: {"observation": {"value": 1.0}},
        mutates_twin=True,
    )
    try:
        fabric.invoke(
            mutating_tool,
            posterior_uid=posterior.posterior_uid,
            inputs={"parent_posterior_uid": "posterior-wrong-parent"},
        )
    except ValueError:
        wrong_parent_tool_rejected = True
    else:
        wrong_parent_tool_rejected = False
    invalid_tool = "evidence.invalid-schema.fixture"
    fabric.register(
        invalid_tool,
        boundary="neuronsifter-evidence-mcp",
        handler=lambda _: {"status": "unsupported-status"},
    )
    invalid_receipt = fabric.invoke(invalid_tool, posterior_uid=posterior.posterior_uid)
    servers = build_scientific_mcp_servers(fabric)
    facade_receipt = servers["neuronsifter-evidence-mcp"].call(
        tool_id,
        posterior_uid=posterior.posterior_uid,
        inputs=inputs,
    )
    try:
        servers["neuronsifter-twin-mcp"].call(
            tool_id,
            posterior_uid=posterior.posterior_uid,
            inputs=inputs,
        )
    except PermissionError:
        boundary_rejected = True
    else:
        boundary_rejected = False
    return {
        "passed": (
            invalidated_hidden
            and stale_memory_rejected
            and idempotent
            and wrong_parent_tool_rejected
            and invalid_receipt.status == "failed"
            and invalid_receipt.failure_class == "invalid_tool_receipt"
            and facade_receipt.tool_call_uid == receipt_a.tool_call_uid
            and boundary_rejected
        ),
        "memory_entry_uid": entry_uid,
        "memory_invalidation_hidden": invalidated_hidden,
        "stale_memory_rejected": stale_memory_rejected,
        "tool_idempotent": idempotent,
        "wrong_parent_tool_rejected": wrong_parent_tool_rejected,
        "invalid_receipt_status": invalid_receipt.status,
        "invalid_receipt_failure_class": invalid_receipt.failure_class,
        "mcp_boundary_rejected": boundary_rejected,
        "posterior_digest_unchanged": twin.belief.get(posterior.posterior_uid).content_digest == digest_before,
    }


def _run_coevolution() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for variant in CoEvolutionVariant:
        twin = new_development_twin()
        runtime = AgentTwinCoEvolutionRuntime(twin, initial_agent_policy_uid="agent-policy-development")
        alpha, beta = development_world_actions()

        def select(round_index: int, posterior: ScientificPosterior, policy_uid: str) -> Mapping[str, Any]:
            observations = posterior.state.get("assimilated_observations", ())
            preference = 0.0
            if observations and isinstance(observations[-1], Mapping):
                preference = float(observations[-1].get("preference", 0.0))
            choice = alpha if preference >= 0.0 else beta
            evidence_uid = f"evidence-{choice.action_uid}-round-{round_index}"
            return {
                "evidence_uids": (evidence_uid,),
                "evidence_distribution_uid": _digest({"choice": choice.action_uid, "policy": policy_uid}),
                "agent_choice_uid": str(choice.action_uid),
                "calibration_drift": 0.01,
                "old_domain_retention": 0.99,
                "tool_result": ToolResult(
                    posterior_uid=posterior.posterior_uid,
                    action_uid="information-action-coevolution",
                    status="completed",
                    observation={"preference": -1.0},
                    uncertainty={"epistemic": 0.03},
                    provenance={"evidence_uid": evidence_uid},
                ),
            }

        def update_agent(policy_uid: str, payload: Mapping[str, Any], posterior: ScientificPosterior) -> str:
            return "agent-policy-" + _digest(
                {
                    "previous": policy_uid,
                    "choice": payload.get("agent_choice_uid"),
                    "posterior_uid": posterior.posterior_uid,
                }
            )[:24]

        result = runtime.run(
            variant,
            rounds=3,
            select_evidence=select,
            update_agent=update_agent,
        )
        results[variant.value] = result.to_record()
    full = results[CoEvolutionVariant.ADAPTIVE_TWIN_ADAPTIVE_AGENT.value]
    controls_observed = {
        name: bool(payload["development_coevolution_observed"])
        for name, payload in results.items()
    }
    return {
        "passed": (
            all(len(payload["rounds"]) == 3 for payload in results.values())
            and controls_observed[CoEvolutionVariant.ADAPTIVE_TWIN_ADAPTIVE_AGENT.value]
            and not controls_observed[CoEvolutionVariant.FROZEN_TWIN_FROZEN_AGENT.value]
            and not controls_observed[CoEvolutionVariant.FROZEN_TWIN_ADAPTIVE_AGENT.value]
            and not controls_observed[CoEvolutionVariant.ADAPTIVE_TWIN_FROZEN_AGENT.value]
        ),
        "development_coevolution_observed_by_variant": controls_observed,
        "variants": results,
        "twin_update_scope": "scientific_posterior_assimilation_only",
        "neural_parameter_update": False,
    }


def _scenario(name: str, callback: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        details = callback()
        return {"passed": bool(details.get("passed")), **details}
    except Exception as exc:
        return {
            "passed": False,
            "failure_class": type(exc).__name__,
            "error": str(exc),
        }


@contextmanager
def _checkpoint_location(output_root: str | Path | None):
    if output_root is not None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        yield root / "development_checkpoint.json"
        return
    with tempfile.TemporaryDirectory(prefix="neuronsifter_agent_twin_closure_") as temp_root:
        yield Path(temp_root) / "development_checkpoint.json"


def run_engineering_closure(output_root: str | Path | None = None) -> dict[str, Any]:
    with _checkpoint_location(output_root) as checkpoint_path:
        scenarios = {
            "normal_multi_round_loop": _scenario("normal_multi_round_loop", lambda: _run_normal_loop(checkpoint_path)),
            "tool_failure_and_partial_result": _scenario("tool_failure_and_partial_result", _run_failure_and_partial),
            "contradictory_observation_admission": _scenario("contradictory_observation_admission", _run_contradiction_and_abstention),
            "checkpoint_restart_replay": {"passed": True, "bound_to_normal_loop": True},
            "memory_invalidation_and_truth_isolation": _scenario("memory_invalidation_and_truth_isolation", _run_memory_and_tool_boundaries),
            "unsupported_action_abstention": _scenario("unsupported_action_abstention", _run_contradiction_and_abstention),
            "tool_boundary_and_idempotence": _scenario("tool_boundary_and_idempotence", _run_memory_and_tool_boundaries),
            "bidirectional_coevolution": _scenario("bidirectional_coevolution", _run_coevolution),
        }
        normal = scenarios["normal_multi_round_loop"]
        scenarios["checkpoint_restart_replay"] = {
            "passed": normal.get("exact_restore", False)
            and normal.get("memory_restore", False)
            and normal.get("replay_posterior_uid") == normal.get("post_observation_posterior_uid"),
            "exact_restore": normal.get("exact_restore", False),
            "memory_restore": normal.get("memory_restore", False),
            "replay_posterior_uid": normal.get("replay_posterior_uid"),
            "checkpoint_sha256": normal.get("checkpoint_sha256"),
            "memory_snapshot_sha256": normal.get("memory_snapshot_sha256"),
        }
        checks = {
            "all_scenarios_passed": all(bool(row.get("passed")) for row in scenarios.values()),
            "scientific_state_identity_bound": bool(normal.get("trace_invariant")) and bool(normal.get("exact_restore")),
            "posterior_changes_trigger_reranking": normal.get("selected_action_uid_pre") != normal.get("selected_action_uid_post"),
            "typed_tool_failures_visible": scenarios["tool_failure_and_partial_result"].get("passed", False),
            "memory_cannot_modify_truth": scenarios["memory_invalidation_and_truth_isolation"].get("posterior_digest_unchanged", False),
            "unsupported_actions_abstain": scenarios["unsupported_action_abstention"].get("passed", False),
            "bidirectional_coupling_observed": scenarios["bidirectional_coevolution"].get("passed", False),
            "protected_holdout_access": 0,
            "development_only": True,
            "publication_claim_eligible": False,
        }
        artifact: dict[str, Any] = {
            "schema_version": ENGINEERING_CLOSURE_SCHEMA_VERSION,
            "status": ENGINEERING_STATUS if checks["all_scenarios_passed"] else "ENGINEERING_CLOSURE_FAILED",
            "development_only": True,
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
            "formal_evidence_gate": FORMAL_EVIDENCE_GATE,
            "formal_real_world_action": "not_qualified",
            "formal_agent_in_twin": "not_qualified",
            "formal_coevolution": "not_qualified",
            "scenarios": scenarios,
            "checks": checks,
            "checkpoint_path": str(checkpoint_path),
        }
        if output_root is not None:
            artifact["checkpoint_sha256"] = sha256_file(checkpoint_path)
        artifact["artifact_digest"] = _digest(artifact)
        if output_root is not None:
            output_path = Path(output_root) / "engineering_closure_report.json"
            output_path.write_text(
                json.dumps(_canonical(artifact), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        return artifact


__all__ = [
    "ENGINEERING_CLOSURE_SCHEMA_VERSION",
    "ENGINEERING_STATUS",
    "FORMAL_EVIDENCE_GATE",
    "SCENARIOS",
    "development_information_action",
    "development_rollout",
    "development_world_actions",
    "new_development_twin",
    "run_engineering_closure",
    "sha256_file",
]
