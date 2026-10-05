"""Bounded Agent--Twin co-evolution runtime.

This module provides the four experimental control surfaces while keeping the
formal therapeutic gate closed.  Updates are explicit, checkpointed, and
rollback-capable; no callback can silently replace a posterior without an
assimilation parent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Callable, Mapping

from Libs.models.modules.decision_consistent_runtime import (
    DecisionConsistentTwin,
    ScientificPosterior,
    ToolResult,
)


class CoEvolutionVariant(str, Enum):
    FROZEN_TWIN_FROZEN_AGENT = "frozen_twin_frozen_agent"
    FROZEN_TWIN_ADAPTIVE_AGENT = "frozen_twin_adaptive_agent"
    ADAPTIVE_TWIN_FROZEN_AGENT = "adaptive_twin_frozen_agent"
    ADAPTIVE_TWIN_ADAPTIVE_AGENT = "adaptive_twin_adaptive_agent"

    @property
    def adaptive_twin(self) -> bool:
        return self in {self.ADAPTIVE_TWIN_FROZEN_AGENT, self.ADAPTIVE_TWIN_ADAPTIVE_AGENT}

    @property
    def adaptive_agent(self) -> bool:
        return self in {self.FROZEN_TWIN_ADAPTIVE_AGENT, self.ADAPTIVE_TWIN_ADAPTIVE_AGENT}


@dataclass(frozen=True)
class CoEvolutionRound:
    round_index: int
    variant: str
    posterior_uid_pre: str
    posterior_uid_post: str
    agent_policy_uid_pre: str
    agent_policy_uid_post: str
    evidence_uids: tuple[str, ...]
    twin_updated: bool
    agent_updated: bool
    calibration_drift: float
    old_domain_retention: float
    twin_debt: float
    rollback: bool
    failure_reason: str | None = None
    twin_content_digest_pre: str = ""
    twin_content_digest_post: str = ""
    agent_choice_uid: str | None = None
    evidence_distribution_uid: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "variant": self.variant,
            "posterior_uid_pre": self.posterior_uid_pre,
            "posterior_uid_post": self.posterior_uid_post,
            "agent_policy_uid_pre": self.agent_policy_uid_pre,
            "agent_policy_uid_post": self.agent_policy_uid_post,
            "evidence_uids": list(self.evidence_uids),
            "twin_updated": self.twin_updated,
            "agent_updated": self.agent_updated,
            "calibration_drift": self.calibration_drift,
            "old_domain_retention": self.old_domain_retention,
            "twin_debt": self.twin_debt,
            "rollback": self.rollback,
            "failure_reason": self.failure_reason,
            "twin_content_digest_pre": self.twin_content_digest_pre,
            "twin_content_digest_post": self.twin_content_digest_post,
            "agent_choice_uid": self.agent_choice_uid,
            "evidence_distribution_uid": self.evidence_distribution_uid,
        }


@dataclass(frozen=True)
class CoEvolutionResult:
    status: str
    variant: str
    rounds: tuple[CoEvolutionRound, ...]
    final_posterior_uid: str
    final_agent_policy_uid: str
    development_coevolution_observed: bool = False
    development_only: bool = True
    publication_claim_eligible: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "variant": self.variant,
            "rounds": [row.to_record() for row in self.rounds],
            "final_posterior_uid": self.final_posterior_uid,
            "final_agent_policy_uid": self.final_agent_policy_uid,
            "development_coevolution_observed": self.development_coevolution_observed,
            "development_only": self.development_only,
            "publication_claim_eligible": self.publication_claim_eligible,
        }


def _policy_update(policy_uid: str, round_index: int, evidence_uids: tuple[str, ...]) -> str:
    payload = {"policy_uid": policy_uid, "round_index": round_index, "evidence_uids": evidence_uids}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    return f"agent-policy-{digest}"


class AgentTwinCoEvolutionRuntime:
    """Run controlled sequential updates against one immutable Twin lineage."""

    def __init__(
        self,
        twin: DecisionConsistentTwin,
        *,
        initial_agent_policy_uid: str = "agent-policy-initial",
        minimum_old_domain_retention: float = 0.95,
        maximum_calibration_drift: float = 0.10,
    ) -> None:
        if twin.decisions.current_posterior_uid is None:
            raise RuntimeError("Bind an initial scientific posterior before co-evolution")
        if not 0.0 < minimum_old_domain_retention <= 1.0:
            raise ValueError("minimum_old_domain_retention must be in (0, 1]")
        if maximum_calibration_drift < 0.0 or not math.isfinite(maximum_calibration_drift):
            raise ValueError("maximum_calibration_drift must be finite and non-negative")
        self.twin = twin
        self.initial_agent_policy_uid = str(initial_agent_policy_uid)
        self.minimum_old_domain_retention = float(minimum_old_domain_retention)
        self.maximum_calibration_drift = float(maximum_calibration_drift)
        self._checkpoints: dict[int, tuple[str, str]] = {}

    @staticmethod
    def variants() -> tuple[str, ...]:
        return tuple(variant.value for variant in CoEvolutionVariant)

    def _default_twin_update(self, parent: ScientificPosterior, evidence: Mapping[str, Any]) -> ScientificPosterior:
        result = evidence.get("tool_result")
        if not isinstance(result, ToolResult):
            raise ValueError("Adaptive Twin updates require a typed ToolResult or update_twin callback")
        outcome = self.twin.assimilate_observation(parent.posterior_uid, result)
        if not outcome.accepted:
            raise ValueError(f"Twin update was not admitted: {outcome.reason}")
        return outcome.posterior

    def run(
        self,
        variant: CoEvolutionVariant | str,
        *,
        rounds: int,
        select_evidence: Callable[[int, ScientificPosterior, str], Mapping[str, Any]],
        update_twin: Callable[[ScientificPosterior, Mapping[str, Any]], ScientificPosterior] | None = None,
        update_agent: Callable[[str, Mapping[str, Any], ScientificPosterior], str] | None = None,
    ) -> CoEvolutionResult:
        selected_variant = variant if isinstance(variant, CoEvolutionVariant) else CoEvolutionVariant(str(variant))
        if rounds < 1:
            raise ValueError("Co-evolution requires at least one sequential round")
        if not callable(select_evidence):
            raise TypeError("select_evidence must be callable")
        posterior = self.twin.belief.get(self.twin.decisions.current_posterior_uid or "")
        policy_uid = self.initial_agent_policy_uid
        records: list[CoEvolutionRound] = []
        twin_debt = 0.0
        for round_index in range(1, int(rounds) + 1):
            posterior_pre = posterior
            policy_pre = policy_uid
            self._checkpoints[round_index] = (posterior_pre.posterior_uid, policy_pre)
            payload = select_evidence(round_index, posterior_pre, policy_pre)
            if not isinstance(payload, Mapping):
                raise TypeError("select_evidence must return a mapping")
            evidence_uids = tuple(sorted(str(uid) for uid in payload.get("evidence_uids", ())))
            agent_choice_uid = (
                str(payload["agent_choice_uid"])
                if payload.get("agent_choice_uid") is not None
                else None
            )
            evidence_distribution_uid = (
                str(payload["evidence_distribution_uid"])
                if payload.get("evidence_distribution_uid") is not None
                else None
            )
            calibration_drift = float(payload.get("calibration_drift", 0.0))
            retention = float(payload.get("old_domain_retention", 1.0))
            if not math.isfinite(calibration_drift) or not math.isfinite(retention):
                raise ValueError("Co-evolution metrics must be finite")
            twin_updated = False
            agent_updated = False
            rollback = False
            failure_reason: str | None = None
            try:
                if selected_variant.adaptive_twin:
                    updater = update_twin or self._default_twin_update
                    candidate = updater(posterior_pre, payload)
                    if not isinstance(candidate, ScientificPosterior):
                        raise TypeError("update_twin must return ScientificPosterior")
                    if candidate.parent_posterior_uid != posterior_pre.posterior_uid:
                        raise ValueError("Twin update must bind the active posterior as parent")
                    if candidate.posterior_uid not in self.twin.belief.posteriors:
                        self.twin.register_posterior(
                            candidate,
                            activate=True,
                            expected_parent_posterior_uid=posterior_pre.posterior_uid,
                            assimilation_status="completed",
                            reason="coevolution_update",
                        )
                    elif self.twin.decisions.current_posterior_uid != candidate.posterior_uid:
                        self.twin.register_posterior(
                            candidate,
                            activate=True,
                            expected_parent_posterior_uid=posterior_pre.posterior_uid,
                            assimilation_status="completed",
                            reason="coevolution_update",
                        )
                    posterior = candidate
                    twin_updated = True
                if selected_variant.adaptive_agent:
                    if update_agent is not None:
                        policy_uid = str(update_agent(policy_pre, payload, posterior))
                    else:
                        policy_uid = _policy_update(policy_pre, round_index, evidence_uids)
                    if not policy_uid:
                        raise ValueError("Adaptive agent update returned an empty policy identity")
                    agent_updated = policy_uid != policy_pre
                if retention < self.minimum_old_domain_retention:
                    raise ValueError("old_domain_retention_below_threshold")
                if calibration_drift > self.maximum_calibration_drift:
                    raise ValueError("calibration_drift_above_threshold")
                twin_debt += float(payload.get("twin_debt", 0.0))
            except Exception as exc:
                failure_reason = str(exc)
                rollback = True
                twin_debt += 1.0
                previous_uid, previous_policy = self._checkpoints[round_index]
                # Rollback is explicit.  The controller can only restore a
                # previously registered posterior; it never fabricates a child.
                posterior = self.twin.decisions.restore_posterior(previous_uid, reason="coevolution_rollback")
                policy_uid = previous_policy
            records.append(
                CoEvolutionRound(
                    round_index=round_index,
                    variant=selected_variant.value,
                    posterior_uid_pre=posterior_pre.posterior_uid,
                    posterior_uid_post=posterior.posterior_uid,
                    agent_policy_uid_pre=policy_pre,
                    agent_policy_uid_post=policy_uid,
                    evidence_uids=evidence_uids,
                    twin_updated=twin_updated,
                    agent_updated=agent_updated,
                    calibration_drift=calibration_drift,
                    old_domain_retention=retention,
                    twin_debt=twin_debt,
                    rollback=rollback,
                    failure_reason=failure_reason,
                    twin_content_digest_pre=posterior_pre.content_digest,
                    twin_content_digest_post=posterior.content_digest,
                    agent_choice_uid=agent_choice_uid,
                    evidence_distribution_uid=evidence_distribution_uid,
                )
            )
        status = "completed" if records and not any(row.rollback for row in records) else "rolled_back"
        development_coevolution_observed = False
        for previous, current in zip(records, records[1:]):
            if (
                previous.twin_updated
                and current.agent_updated
                and previous.twin_content_digest_post != previous.twin_content_digest_pre
                and current.agent_choice_uid is not None
                and previous.agent_choice_uid is not None
                and current.agent_choice_uid != previous.agent_choice_uid
                and current.evidence_uids
            ):
                development_coevolution_observed = True
                break
        return CoEvolutionResult(
            status=status,
            variant=selected_variant.value,
            rounds=tuple(records),
            final_posterior_uid=posterior.posterior_uid,
            final_agent_policy_uid=policy_uid,
            development_coevolution_observed=development_coevolution_observed,
        )


__all__ = [
    "AgentTwinCoEvolutionRuntime",
    "CoEvolutionResult",
    "CoEvolutionRound",
    "CoEvolutionVariant",
]
