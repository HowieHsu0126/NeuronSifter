"""Twin-grounded ReAct controller for development-time CNS optimization.

The controller implements the scientific loop as a typed state machine.  It
records plans, actions, receipts, assimilation lineage, and reflection
summaries, but never persists an unstructured language-model reasoning trace.
The current implementation is deliberately dependency-light so it can drive
the existing NeuronSifter Twin, registered simulator adapters, or replay
fixtures through the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from Libs.models.modules.decision_consistent_runtime import (
    DecisionConsistentTwin,
    InformationAction,
    ScientificPosterior,
    ToolResult,
    WorldAction,
)
from Libs.models.modules.intervention_optimizer import (
    InterventionOptimizer,
    OptimizationResult,
)
from Libs.models.modules.scientific_memory import MemoryEntry, ScientificMemory
from Libs.models.modules.scientific_tool_fabric import (
    ScientificToolFabric,
    ScientificToolReceipt,
)


class ControllerPhase(str, Enum):
    OBSERVE = "OBSERVE"
    RETRIEVE_MEMORY = "RETRIEVE_MEMORY"
    PLAN = "PLAN"
    PROPOSE_WORLD_ACTIONS = "PROPOSE_WORLD_ACTIONS"
    TWIN_ROLLOUT = "TWIN_ROLLOUT"
    IDENTIFY_INFORMATION_NEED = "IDENTIFY_INFORMATION_NEED"
    SELECT_TOOL = "SELECT_TOOL"
    TOOL_CALL = "TOOL_CALL"
    OBSERVE_RESULT = "OBSERVE_RESULT"
    ASSIMILATE = "ASSIMILATE"
    OPTIMIZE_INTERVENTION = "OPTIMIZE_INTERVENTION"
    REFLECT = "REFLECT"
    WRITE_MEMORY = "WRITE_MEMORY"
    NEXT_ROUND = "NEXT_ROUND"
    STOP = "STOP"


@dataclass(frozen=True)
class PlannerState:
    goal: str
    posterior_uid: str
    candidate_action_uids: tuple[str, ...]
    uncertainty_keys: tuple[str, ...]
    selected_information_action_uid: str | None
    expected_decision_relevance: float
    stop_criterion: str

    def __post_init__(self) -> None:
        if not self.goal.strip() or not self.posterior_uid.startswith("posterior-"):
            raise ValueError("PlannerState requires a goal and versioned posterior UID")
        if not self.candidate_action_uids:
            raise ValueError("PlannerState requires candidate WorldAction identities")
        if not math.isfinite(float(self.expected_decision_relevance)):
            raise ValueError("PlannerState decision relevance must be finite")

    def to_record(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "posterior_uid": self.posterior_uid,
            "candidate_action_uids": list(self.candidate_action_uids),
            "uncertainty_keys": list(self.uncertainty_keys),
            "selected_information_action_uid": self.selected_information_action_uid,
            "expected_decision_relevance": self.expected_decision_relevance,
            "stop_criterion": self.stop_criterion,
        }


@dataclass(frozen=True)
class AgentDecision:
    round_index: int
    phase_history: tuple[str, ...]
    posterior_uid_pre: str
    posterior_uid_post: str
    planner: PlannerState
    selected_action_uid: str | None
    rejected_action_uids: tuple[str, ...]
    tool_call_uid: str | None
    observation_uid: str | None
    decision_change: str
    failure_reason: str | None
    reflection_summary: Mapping[str, Any]
    memory_uids: tuple[str, ...] = ()
    selected_action_uid_post: str | None = None
    ranking_posterior_uid_pre: str | None = None
    ranking_posterior_uid_post: str | None = None
    ranked_action_uids_pre: tuple[str, ...] = ()
    ranked_action_uids_post: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "phase_history": list(self.phase_history),
            "posterior_uid_pre": self.posterior_uid_pre,
            "posterior_uid_post": self.posterior_uid_post,
            "planner": self.planner.to_record(),
            "selected_action_uid": self.selected_action_uid,
            "rejected_action_uids": list(self.rejected_action_uids),
            "tool_call_uid": self.tool_call_uid,
            "observation_uid": self.observation_uid,
            "decision_change": self.decision_change,
            "failure_reason": self.failure_reason,
            "reflection_summary": dict(self.reflection_summary),
            "memory_uids": list(self.memory_uids),
            "selected_action_uid_post": self.selected_action_uid_post,
            "ranking_posterior_uid_pre": self.ranking_posterior_uid_pre,
            "ranking_posterior_uid_post": self.ranking_posterior_uid_post,
            "ranked_action_uids_pre": list(self.ranked_action_uids_pre),
            "ranked_action_uids_post": list(self.ranked_action_uids_post),
        }


@dataclass(frozen=True)
class AgentRunResult:
    status: str
    goal: str
    decisions: tuple[AgentDecision, ...]
    final_posterior_uid: str
    final_optimization: OptimizationResult | None
    development_only: bool = True
    publication_claim_eligible: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.goal,
            "decisions": [decision.to_record() for decision in self.decisions],
            "final_posterior_uid": self.final_posterior_uid,
            "final_optimization": self.final_optimization.to_record() if self.final_optimization else None,
            "development_only": self.development_only,
            "publication_claim_eligible": self.publication_claim_eligible,
        }


class TwinGroundedReActController:
    """Bounded scientific Plan/Act/Observe/Assimilate controller."""

    def __init__(
        self,
        twin: DecisionConsistentTwin,
        *,
        memory: ScientificMemory | None = None,
        tools: ScientificToolFabric | None = None,
        optimizer: InterventionOptimizer | None = None,
        max_rounds: int | None = None,
        tool_budget: int | None = None,
        controller_config: Mapping[str, Any] | None = None,
    ) -> None:
        if controller_config is not None:
            required = {
                "max_rounds",
                "tool_budget",
                "stop_on_insufficient_evidence",
            }
            missing = sorted(required.difference(controller_config))
            if missing:
                raise ValueError(f"controller_config is missing fields: {missing}")
            configured_rounds = int(controller_config["max_rounds"])
            configured_budget = int(controller_config["tool_budget"])
            if max_rounds is not None and int(max_rounds) != configured_rounds:
                raise ValueError("max_rounds conflicts with controller_config")
            if tool_budget is not None and int(tool_budget) != configured_budget:
                raise ValueError("tool_budget conflicts with controller_config")
            max_rounds = configured_rounds
            tool_budget = configured_budget
            stop_on_insufficient_evidence = bool(
                controller_config["stop_on_insufficient_evidence"]
            )
        else:
            stop_on_insufficient_evidence = True
            max_rounds = 3 if max_rounds is None else int(max_rounds)
            tool_budget = 3 if tool_budget is None else int(tool_budget)
        if max_rounds < 1 or tool_budget < 0:
            raise ValueError("max_rounds must be positive and tool_budget non-negative")
        self.twin = twin
        self.memory = memory or ScientificMemory()
        self.tools = tools or ScientificToolFabric()
        self.optimizer = optimizer or InterventionOptimizer()
        self.max_rounds = int(max_rounds)
        self.tool_budget = int(tool_budget)
        self.stop_on_insufficient_evidence = stop_on_insufficient_evidence
        self.phase_history: list[str] = []
        self.memory.bind_posterior_validator(self.twin.belief.get)

    @property
    def posterior_uid(self) -> str:
        uid = self.twin.decisions.current_posterior_uid
        if uid is None:
            raise RuntimeError("Twin-grounded controller has no active posterior")
        return uid

    def _phase(self, phase: ControllerPhase) -> None:
        self.phase_history.append(phase.value)

    def _planner(
        self,
        goal: str,
        posterior: ScientificPosterior,
        actions: Sequence[WorldAction],
        information_actions: Sequence[InformationAction],
    ) -> PlannerState:
        relevance = 0.0
        selected_info_uid = None
        if information_actions:
            values = [self.optimizer.value_information_action(posterior, action) for action in information_actions]
            values.sort(key=lambda row: (-float(row["utility"]), str(row["information_action_uid"])))
            selected_info_uid = str(values[0]["information_action_uid"])
            relevance = float(values[0]["expected_decision_relevance"])
        uncertainty_keys = tuple(sorted(str(key) for key in posterior.uncertainty))
        return PlannerState(
            goal=goal,
            posterior_uid=posterior.posterior_uid,
            candidate_action_uids=tuple(str(action.action_uid) for action in actions),
            uncertainty_keys=uncertainty_keys,
            selected_information_action_uid=selected_info_uid,
            expected_decision_relevance=relevance,
            stop_criterion="stop_when_ranked_or_information_budget_exhausted",
        )

    @staticmethod
    def _tool_result_from_receipt(
        receipt: ScientificToolReceipt,
        information_action: InformationAction,
    ) -> ToolResult:
        succeeded = receipt.status in {"succeeded", "completed", "partial"}
        summary = dict(receipt.result_summary)
        observation = summary.get("observation", summary if succeeded else {})
        if not isinstance(observation, Mapping):
            observation = {"value": observation}
        status = "partial" if receipt.status == "partial" else ("completed" if succeeded else "failed")
        return ToolResult(
            posterior_uid=receipt.posterior_uid,
            action_uid=str(information_action.action_uid),
            status=status,
            observation=dict(observation),
            uncertainty=dict(receipt.uncertainty),
            error=(str(summary.get("error")) if not succeeded and summary.get("error") else receipt.failure_class),
            provenance={
                "tool_call_uid": receipt.tool_call_uid,
                "tool_name": receipt.tool_name,
                "provenance_hash": receipt.provenance_hash,
                "evidence_grade": receipt.evidence_grade,
            },
            observation_uid=receipt.observation_uid,
            partial=receipt.status == "partial",
        )

    def _write_episode(
        self,
        *,
        posterior_uid: str,
        selected_action_uid: str | None,
        tool_call_uid: str | None,
        observation_uid: str | None,
        reflection: Mapping[str, Any],
        round_index: int,
    ) -> tuple[str, ...]:
        entry = MemoryEntry(
            memory_type="episodic",
            structured_summary={
                "round_index": round_index,
                "decision_change": reflection.get("decision_change", "none"),
                "selected_action_uid": selected_action_uid,
                "tool_call_uid": tool_call_uid,
                "observation_uid": observation_uid,
                "failure_reason": reflection.get("failure_reason"),
                "reflection_summary": dict(reflection),
            },
            source_uids=tuple(uid for uid in (observation_uid,) if uid),
            posterior_uid=posterior_uid,
            world_action_uid=selected_action_uid,
            tool_call_uids=tuple(uid for uid in (tool_call_uid,) if uid),
            confidence=1.0 if not reflection.get("failure_reason") else 0.0,
            evidence_grade="development_diagnostic",
        )
        return (self.memory.write(entry),)

    def run(
        self,
        goal: str,
        world_actions: Sequence[WorldAction],
        information_actions: Sequence[InformationAction] = (),
    ) -> AgentRunResult:
        if not goal.strip():
            raise ValueError("Agent goal must be non-empty")
        if not world_actions:
            raise ValueError("Agent requires at least one candidate WorldAction")
        if self.twin.decisions.current_posterior_uid is None:
            raise RuntimeError("Infer and bind a Twin posterior before running the controller")
        if len({str(action.action_uid) for action in world_actions}) != len(world_actions):
            raise ValueError("WorldAction candidates must have unique identities")
        decisions: list[AgentDecision] = []
        previous_selected: str | None = None
        used_tools: set[str] = set()
        final_optimization: OptimizationResult | None = None

        for round_index in range(1, self.max_rounds + 1):
            self.phase_history = []
            self._phase(ControllerPhase.OBSERVE)
            posterior_pre = self.twin.belief.get(self.posterior_uid)
            self._phase(ControllerPhase.RETRIEVE_MEMORY)
            relevant_memory = self.memory.retrieve(posterior_uid=posterior_pre.posterior_uid, limit=20)
            self._phase(ControllerPhase.PLAN)
            planner = self._planner(goal, posterior_pre, world_actions, information_actions)
            self._phase(ControllerPhase.PROPOSE_WORLD_ACTIONS)
            self._phase(ControllerPhase.TWIN_ROLLOUT)
            rollouts = {
                str(action.action_uid): self.twin.rollout(posterior_pre.posterior_uid, action)
                for action in world_actions
            }
            if information_actions:
                self.twin.decisions.value_information_actions(
                    posterior_pre.posterior_uid,
                    [action for action in information_actions if str(action.action_uid) not in used_tools],
                )
            self._phase(ControllerPhase.IDENTIFY_INFORMATION_NEED)
            final_optimization = self.optimizer.optimize(
                posterior_pre,
                world_actions,
                rollouts=rollouts,
                information_actions=[action for action in information_actions if str(action.action_uid) not in used_tools],
            )
            self.twin.decisions.rank_interventions(posterior_pre.posterior_uid, world_actions)
            optimization_pre = final_optimization
            self._phase(ControllerPhase.OPTIMIZE_INTERVENTION)
            selected_uid = final_optimization.selected_action_uid
            rejected = tuple(
                row.action_uid for row in final_optimization.ranked if row.action_uid != selected_uid
            )
            if selected_uid is None:
                decision_change = "no_eligible_action"
            elif previous_selected is None:
                decision_change = "initial_selection"
            else:
                decision_change = "unchanged" if previous_selected == selected_uid else "switched"
            previous_selected = selected_uid
            tool_call_uid = None
            observation_uid = None
            failure_reason: str | None = None
            posterior_post = posterior_pre.posterior_uid
            selected_action_uid_post: str | None = selected_uid
            ranking_posterior_uid_post: str | None = posterior_pre.posterior_uid
            ranked_action_uids_post = tuple(row.action_uid for row in optimization_pre.ranked)
            selected_info = None
            if final_optimization.next_information_action_uid is not None:
                selected_info = next(
                    (action for action in information_actions if str(action.action_uid) == final_optimization.next_information_action_uid and str(action.action_uid) not in used_tools),
                    None,
                )
            self._phase(ControllerPhase.SELECT_TOOL)
            if selected_info is not None and len(used_tools) < self.tool_budget:
                self._phase(ControllerPhase.TOOL_CALL)
                used_tools.add(str(selected_info.action_uid))
                try:
                    receipt = self.tools.invoke(
                        selected_info.tool_id,
                        posterior_uid=posterior_pre.posterior_uid,
                        inputs={
                            "information_action": selected_info.to_record(),
                            "information_action_uid": selected_info.action_uid,
                            "parent_posterior_uid": posterior_pre.posterior_uid,
                        },
                    )
                    tool_call_uid = receipt.tool_call_uid
                    self._phase(ControllerPhase.OBSERVE_RESULT)
                    tool_result = self._tool_result_from_receipt(receipt, selected_info)
                    self._phase(ControllerPhase.ASSIMILATE)
                    assimilation = self.twin.assimilate_observation(posterior_pre.posterior_uid, tool_result)
                    posterior_post = assimilation.posterior.posterior_uid
                    if not assimilation.accepted:
                        failure_reason = assimilation.reason
                    else:
                        observation_uid = receipt.observation_uid
                        posterior_post = assimilation.posterior.posterior_uid
                        post_rollouts = {
                            str(action.action_uid): self.twin.rollout(posterior_post, action)
                            for action in world_actions
                        }
                        final_optimization = self.optimizer.optimize(
                            assimilation.posterior,
                            world_actions,
                            rollouts=post_rollouts,
                            information_actions=(),
                        )
                        self.twin.decisions.rank_interventions(posterior_post, world_actions)
                        selected_action_uid_post = final_optimization.selected_action_uid
                        ranking_posterior_uid_post = posterior_post
                        ranked_action_uids_post = tuple(row.action_uid for row in final_optimization.ranked)
                except Exception as exc:
                    failure_reason = f"tool_or_assimilation_error:{type(exc).__name__}"
            elif selected_info is not None and len(used_tools) >= self.tool_budget:
                failure_reason = "information_budget_exhausted"
            self._phase(ControllerPhase.REFLECT)
            reflection = {
                "decision_change": decision_change,
                "selected_action_uid": selected_uid,
                "rejected_action_count": len(rejected),
                "tool_selected": selected_info.action_uid if selected_info else None,
                "tool_call_uid": tool_call_uid,
                "observation_uid": observation_uid,
                "posterior_uid_pre": posterior_pre.posterior_uid,
                "posterior_uid_post": posterior_post,
                "memory_hit_count": len(relevant_memory),
                "failure_reason": failure_reason,
                "development_only": True,
            }
            self._phase(ControllerPhase.WRITE_MEMORY)
            memory_uids = self._write_episode(
                posterior_uid=posterior_post,
                selected_action_uid=selected_uid,
                tool_call_uid=tool_call_uid,
                observation_uid=observation_uid,
                reflection=reflection,
                round_index=round_index,
            )
            decisions.append(
                AgentDecision(
                    round_index=round_index,
                    phase_history=tuple(self.phase_history),
                    posterior_uid_pre=posterior_pre.posterior_uid,
                    posterior_uid_post=posterior_post,
                    planner=planner,
                    selected_action_uid=selected_uid,
                    rejected_action_uids=rejected,
                    tool_call_uid=tool_call_uid,
                    observation_uid=observation_uid,
                    decision_change=decision_change,
                    failure_reason=failure_reason,
                    reflection_summary=reflection,
                    memory_uids=memory_uids,
                    selected_action_uid_post=selected_action_uid_post,
                    ranking_posterior_uid_pre=posterior_pre.posterior_uid,
                    ranking_posterior_uid_post=ranking_posterior_uid_post,
                    ranked_action_uids_pre=tuple(row.action_uid for row in optimization_pre.ranked),
                    ranked_action_uids_post=ranked_action_uids_post,
                )
            )
            self._phase(ControllerPhase.NEXT_ROUND)
            if (
                final_optimization.status == "INSUFFICIENT_EVIDENCE"
                and self.stop_on_insufficient_evidence
                and selected_info is None
            ):
                self._phase(ControllerPhase.STOP)
                break
            if posterior_post != posterior_pre.posterior_uid:
                # The next loop must consume the assimilation child.  This is
                # the explicit re-ranking boundary required by the Twin.
                continue
            if selected_info is None or round_index >= self.max_rounds:
                self._phase(ControllerPhase.STOP)
                break
        status = (
            "completed"
            if decisions and final_optimization is not None and final_optimization.status == "ranked"
            else "insufficient_evidence"
        )
        return AgentRunResult(
            status=status,
            goal=goal,
            decisions=tuple(decisions),
            final_posterior_uid=self.posterior_uid,
            final_optimization=final_optimization,
        )


__all__ = [
    "AgentDecision",
    "AgentRunResult",
    "ControllerPhase",
    "PlannerState",
    "TwinGroundedReActController",
]
