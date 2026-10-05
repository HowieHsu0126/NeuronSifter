"""Decision-aware CNS intervention optimization over the scientific Twin.

The optimizer is intentionally model-agnostic.  It consumes typed
``WorldAction`` objects and rollout summaries emitted by the existing Twin
runtime, then returns an auditable ranking with an explicit abstention path.
It never promotes an agent-memory claim to scientific evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from Libs.models.modules.decision_consistent_runtime import (
    InformationAction,
    ScientificPosterior,
    TrajectoryResult,
    WorldAction,
)


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    return True


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        values: list[float] = []
        for item in value.values():
            values.extend(_numeric_values(item))
        return values
    if isinstance(value, (list, tuple, np.ndarray)):
        values: list[float] = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
        return [float(value)]
    return []


def _mean(value: Any, default: float = 0.0) -> float:
    values = _numeric_values(value)
    return float(np.mean(values)) if values else float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class OptimizerWeights:
    """Weights for the CNS intervention utility contract."""

    uncertainty: float = 1.0
    safety: float = 1.0
    cost: float = 0.1
    violation: float = 5.0
    support: float = 1.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"Optimizer weight {name} must be finite and non-negative")


@dataclass(frozen=True)
class InterventionScore:
    action_uid: str
    posterior_uid: str
    benefit: float
    uncertainty: float
    safety_risk: float
    support_penalty: float
    cost: float
    violations: tuple[str, ...]
    utility: float
    rank: int = 0
    rollout_status: str = "unconfigured"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "action_uid": self.action_uid,
            "posterior_uid": self.posterior_uid,
            "benefit": self.benefit,
            "uncertainty": self.uncertainty,
            "safety_risk": self.safety_risk,
            "support_penalty": self.support_penalty,
            "cost": self.cost,
            "violations": list(self.violations),
            "utility": self.utility,
            "rank": self.rank,
            "rollout_status": self.rollout_status,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    posterior_uid: str
    ranked: tuple[InterventionScore, ...]
    pareto_action_uids: tuple[str, ...]
    selected_action_uid: str | None
    next_information_action_uid: str | None
    reason: str
    development_only: bool = True
    publication_claim_eligible: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "posterior_uid": self.posterior_uid,
            "ranked": [row.to_record() for row in self.ranked],
            "pareto_action_uids": list(self.pareto_action_uids),
            "selected_action_uid": self.selected_action_uid,
            "next_information_action_uid": self.next_information_action_uid,
            "reason": self.reason,
            "development_only": self.development_only,
            "publication_claim_eligible": self.publication_claim_eligible,
        }


class InterventionOptimizer:
    """Rank structured interventions and value information actions.

    The implementation uses explicit proxies when a model does not expose a
    named metric.  Proxy values are retained in ``diagnostics`` and are never
    presented as clinical outcomes.
    """

    def __init__(
        self,
        weights: OptimizerWeights | None = None,
        *,
        minimum_support: float = 0.0,
        minimum_tri: float = 0.0,
        allowed_modalities: Iterable[str] | None = None,
        allowed_routes: Iterable[str] | None = None,
        dose_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.weights = weights or OptimizerWeights()
        if not math.isfinite(float(minimum_support)) or not 0.0 <= minimum_support <= 1.0:
            raise ValueError("minimum_support must be in [0, 1]")
        if not math.isfinite(float(minimum_tri)) or not 0.0 <= minimum_tri <= 1.0:
            raise ValueError("minimum_tri must be in [0, 1]")
        self.minimum_support = float(minimum_support)
        self.minimum_tri = float(minimum_tri)
        self.allowed_modalities = frozenset(str(value) for value in (allowed_modalities or ()))
        self.allowed_routes = frozenset(str(value) for value in (allowed_routes or ()))
        if dose_bounds is not None:
            if len(dose_bounds) != 2 or not all(math.isfinite(float(v)) for v in dose_bounds):
                raise ValueError("dose_bounds must contain two finite values")
            if float(dose_bounds[0]) > float(dose_bounds[1]):
                raise ValueError("dose_bounds lower bound exceeds upper bound")
            self.dose_bounds = (float(dose_bounds[0]), float(dose_bounds[1]))
        else:
            self.dose_bounds = None

    def validate_action(self, action: WorldAction) -> tuple[str, ...]:
        if not isinstance(action, WorldAction):
            raise TypeError("InterventionOptimizer requires a WorldAction")
        violations: list[str] = []
        modality = str(getattr(action, "modality", "pharmacological"))
        if self.allowed_modalities and modality not in self.allowed_modalities:
            violations.append("modality_not_allowed")
        if self.allowed_routes and str(action.route) not in self.allowed_routes:
            violations.append("route_not_allowed")
        if self.dose_bounds is not None:
            values = _numeric_values(getattr(action, "dose_or_intensity", action.dose))
            if not values or any(value < self.dose_bounds[0] or value > self.dose_bounds[1] for value in values):
                violations.append("dose_out_of_bounds")
        if not action.target_set or not action.drug_set:
            violations.append("empty_action_entities")
        return tuple(violations)

    def _rollout_features(
        self,
        rollout: TrajectoryResult | Mapping[str, Any] | None,
        action: WorldAction,
    ) -> tuple[float, float, float, float, str, Mapping[str, Any]]:
        if rollout is None:
            return 0.0, 0.0, 0.0, 0.0, "unconfigured", {"proxy": "missing_rollout"}
        if isinstance(rollout, TrajectoryResult):
            trajectory = rollout.trajectory
            uncertainty = rollout.uncertainty
            diagnostics = _mapping(rollout.diagnostics)
            status = str(diagnostics.get("status", "completed"))
        elif isinstance(rollout, Mapping):
            trajectory = rollout.get("trajectory", [])
            uncertainty = rollout.get("uncertainty", {})
            diagnostics = _mapping(rollout.get("diagnostics", {}))
            status = str(rollout.get("status", diagnostics.get("status", "completed")))
        else:
            raise TypeError("rollout must be TrajectoryResult, mapping, or None")
        if not _finite(trajectory) or not _finite(uncertainty):
            raise ValueError("Rollout and uncertainty must be finite")
        if not _numeric_values(trajectory):
            status = "missing_rollout"
        benefit = float(diagnostics.get("benefit", _mean(trajectory)))
        uncertainty_value = float(diagnostics.get("uncertainty", _mean(uncertainty)))
        safety = float(diagnostics.get("safety_risk", _mapping(getattr(action, "biological_context", {})).get("safety_risk", 0.0)))
        cost = float(diagnostics.get("cost", 0.0))
        if not all(math.isfinite(value) for value in (benefit, uncertainty_value, safety, cost)):
            raise ValueError("Rollout features must be finite")
        return benefit, max(uncertainty_value, 0.0), max(safety, 0.0), max(cost, 0.0), status, diagnostics

    def score_action(
        self,
        posterior: ScientificPosterior,
        action: WorldAction,
        rollout: TrajectoryResult | Mapping[str, Any] | None = None,
    ) -> InterventionScore:
        violations = list(self.validate_action(action))
        benefit, uncertainty, safety, rollout_cost, status, diagnostics = self._rollout_features(rollout, action)
        if status in {
            "unconfigured",
            "unconfigured_world_executor",
            "missing",
            "missing_rollout",
            "failed",
            "invalid",
        }:
            # A finite diagnostic placeholder is not a scientific rollout.
            # Keep the score auditable, but make the candidate ineligible for
            # intervention selection until a real executor result is admitted.
            violations.append("rollout_unavailable")
        support_score = float(_mapping(posterior.support).get("support_score", 0.0))
        tri_score = float(_mapping(posterior.reliability).get("tri_score", 0.0))
        if not math.isfinite(support_score) or not math.isfinite(tri_score):
            raise ValueError("Posterior support and TRI values must be finite")
        support_penalty = max(0.0, 1.0 - support_score)
        if support_score < self.minimum_support:
            violations = (*violations, "support_below_threshold")
        if tri_score < self.minimum_tri:
            violations = (*violations, "tri_below_threshold")
        utility = (
            benefit
            - self.weights.uncertainty * uncertainty
            - self.weights.safety * safety
            - self.weights.cost * rollout_cost
            - self.weights.support * support_penalty
            - self.weights.violation * len(violations)
        )
        return InterventionScore(
            action_uid=str(action.action_uid),
            posterior_uid=posterior.posterior_uid,
            benefit=benefit,
            uncertainty=uncertainty,
            safety_risk=safety,
            support_penalty=support_penalty,
            cost=rollout_cost,
            violations=tuple(sorted(set(violations))),
            utility=float(utility),
            rollout_status=status,
            diagnostics={"support_score": support_score, "tri_score": tri_score, **dict(diagnostics)},
        )

    @staticmethod
    def _pareto(rows: Sequence[InterventionScore]) -> tuple[str, ...]:
        selected: list[str] = []
        for candidate in rows:
            dominated = False
            for other in rows:
                if other.action_uid == candidate.action_uid:
                    continue
                no_worse = (
                    other.benefit >= candidate.benefit
                    and other.uncertainty <= candidate.uncertainty
                    and other.safety_risk <= candidate.safety_risk
                    and other.cost <= candidate.cost
                )
                strictly_better = (
                    other.benefit > candidate.benefit
                    or other.uncertainty < candidate.uncertainty
                    or other.safety_risk < candidate.safety_risk
                    or other.cost < candidate.cost
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                selected.append(candidate.action_uid)
        return tuple(selected)

    def value_information_action(
        self,
        posterior: ScientificPosterior,
        action: InformationAction,
    ) -> dict[str, Any]:
        if not isinstance(action, InformationAction):
            raise TypeError("Information valuation requires an InformationAction")
        uncertainty = _mean(posterior.uncertainty, 0.0)
        query = _mapping(action.query)
        declared_relevance = query.get("expected_decision_relevance", uncertainty)
        relevance = float(declared_relevance) if isinstance(declared_relevance, (int, float)) else uncertainty
        reliability = query.get("tool_reliability", 1.0)
        reliability_value = float(reliability) if isinstance(reliability, (int, float)) else 1.0
        cost = max(float(action.expected_cost), 1.0e-12)
        utility = max(relevance, 0.0) * max(reliability_value, 0.0) / cost
        return {
            "information_action_uid": action.action_uid,
            "tool_id": action.tool_id,
            "posterior_uid": posterior.posterior_uid,
            "expected_decision_relevance": relevance,
            "tool_reliability": reliability_value,
            "expected_cost": float(action.expected_cost),
            "utility": utility,
            "valuation_kind": "decision_relevance_proxy",
        }

    def optimize(
        self,
        posterior: ScientificPosterior,
        actions: Sequence[WorldAction],
        rollouts: Mapping[str, TrajectoryResult | Mapping[str, Any]] | None = None,
        information_actions: Sequence[InformationAction] = (),
    ) -> OptimizationResult:
        if not actions:
            raise ValueError("Intervention optimization requires at least one WorldAction")
        rollout_map = dict(rollouts or {})
        scored = [self.score_action(posterior, action, rollout_map.get(str(action.action_uid))) for action in actions]
        scored.sort(key=lambda row: (-row.utility, row.uncertainty, row.action_uid))
        ranked = tuple(
            InterventionScore(**{**row.__dict__, "rank": index})
            for index, row in enumerate(scored, start=1)
        )
        pareto = self._pareto(ranked)
        usable = [row for row in ranked if not row.violations]
        info_uid = None
        if information_actions:
            info_uid = max(
                (self.value_information_action(posterior, action) for action in information_actions),
                key=lambda row: (float(row["utility"]), str(row["information_action_uid"])),
            )["information_action_uid"]
        if not usable:
            return OptimizationResult(
                status="INSUFFICIENT_EVIDENCE",
                posterior_uid=posterior.posterior_uid,
                ranked=ranked,
                pareto_action_uids=pareto,
                selected_action_uid=None,
                next_information_action_uid=info_uid,
                reason="all_candidates_failed_support_or_constraint_gate",
            )
        return OptimizationResult(
            status="ranked",
            posterior_uid=posterior.posterior_uid,
            ranked=ranked,
            pareto_action_uids=pareto,
            selected_action_uid=usable[0].action_uid,
            next_information_action_uid=info_uid,
            reason="ranked_by_twin_rollout_utility",
        )


__all__ = [
    "InterventionOptimizer",
    "InterventionScore",
    "OptimizationResult",
    "OptimizerWeights",
]
