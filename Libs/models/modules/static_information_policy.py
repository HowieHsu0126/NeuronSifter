"""Frozen information-action policies for a shared scientific posterior."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from Libs.models.modules.decision_consistent_runtime import (
    InformationAction,
    WorldAction,
)
from Libs.models.modules.round24b_posterior_adapter import PosteriorBoundState


POLICY_UIDS = (
    "R_random_information_action",
    "F_fixed_cascade",
    "U_uncertainty_only",
    "S_shared_belief_decision_risk",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class InformationCandidate:
    """One typed information query linked to an explicit world-action pair."""

    candidate_uid: str
    pair_uid: str
    state_uid: str
    world_action: WorldAction
    control_world_action: WorldAction
    information_action: InformationAction
    design: tuple[float, ...]
    order_index: int

    def __post_init__(self) -> None:
        if not self.candidate_uid or not self.pair_uid or not self.state_uid:
            raise ValueError("Static information candidates require stable identities")
        if not self.design or not all(math.isfinite(value) for value in self.design):
            raise ValueError("Static information candidate design must be finite")
        protocol_ref = self.information_action.query.get("world_action_uid")
        control_ref = self.information_action.query.get("control_world_action_uid")
        if protocol_ref != self.world_action.action_uid:
            raise ValueError("Information query does not reference its typed world action")
        if control_ref != self.control_world_action.action_uid:
            raise ValueError("Information query does not reference its typed control action")

    @property
    def expected_cost(self) -> float:
        return float(self.information_action.expected_cost)

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_uid": self.candidate_uid,
            "pair_uid": self.pair_uid,
            "state_uid": self.state_uid,
            "world_action_uid": self.world_action.action_uid,
            "control_world_action_uid": self.control_world_action.action_uid,
            "information_action_uid": self.information_action.action_uid,
            "design": list(self.design),
            "expected_cost": self.expected_cost,
            "order_index": self.order_index,
        }


def rank_interventions(
    posterior: PosteriorBoundState,
    candidates: Sequence[InformationCandidate],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        mean, std = posterior.action_response(np.asarray(candidate.design))
        rows.append(
            {
                "candidate_uid": candidate.candidate_uid,
                "world_action_uid": candidate.world_action.action_uid,
                "predicted_response": mean,
                "predictive_std": std,
                "posterior_uid": posterior.posterior_uid,
            }
        )
    rows.sort(
        key=lambda row: (-float(row["predicted_response"]), str(row["candidate_uid"]))
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def normalized_rank_disagreement(
    left: Iterable[str], right: Iterable[str]
) -> float:
    left_values = list(left)
    right_values = list(right)
    if set(left_values) != set(right_values):
        raise ValueError("Rank disagreement requires identical candidate identities")
    if len(left_values) < 2:
        return 0.0
    right_rank = {uid: index for index, uid in enumerate(right_values)}
    disagreements = 0
    pairs = 0
    for index, first in enumerate(left_values):
        for second in left_values[index + 1 :]:
            pairs += 1
            if right_rank[first] > right_rank[second]:
                disagreements += 1
    return float(disagreements / pairs)


def posterior_discrepancy_proxy(
    before: PosteriorBoundState, after: PosteriorBoundState
) -> float:
    """Stable Gaussian parameter proxy; this is not theoretical posterior TV."""
    if before.mean.shape != after.mean.shape:
        raise ValueError("Posterior discrepancy requires matched dimensions")
    mean_scale = max(float(np.linalg.norm(before.mean)), 1.0e-12)
    covariance_scale = max(float(np.linalg.norm(before.covariance)), 1.0e-12)
    mean_delta = float(np.linalg.norm(after.mean - before.mean) / mean_scale)
    covariance_delta = float(
        np.linalg.norm(after.covariance - before.covariance) / covariance_scale
    )
    value = math.sqrt(mean_delta * mean_delta + covariance_delta * covariance_delta)
    if not math.isfinite(value):
        raise ValueError("Posterior discrepancy proxy is non-finite")
    return value


class StaticInformationPolicy:
    """Immutable R/F/U/S query policy over one posterior identity."""

    def __init__(
        self,
        policy_uid: str,
        *,
        seed: int,
        fixed_order: Sequence[str],
        expected_cost_floor: float,
    ) -> None:
        if policy_uid not in POLICY_UIDS:
            raise ValueError(f"Unknown static information policy: {policy_uid}")
        if expected_cost_floor <= 0.0 or not math.isfinite(expected_cost_floor):
            raise ValueError("Expected-cost floor must be finite and positive")
        self.policy_uid = policy_uid
        self.seed = int(seed)
        self.fixed_order = tuple(str(value) for value in fixed_order)
        self.expected_cost_floor = float(expected_cost_floor)

    def to_record(self) -> dict[str, Any]:
        record = {
            "schema_version": "neuronsifter-static-information-policy-v1",
            "policy_uid": self.policy_uid,
            "seed": self.seed,
            "fixed_order": list(self.fixed_order),
            "expected_cost_floor": self.expected_cost_floor,
            "parameter_update_allowed": False,
            "outcome_visibility_during_selection": False,
        }
        record["policy_sha256"] = _digest(record)
        return record

    def _random_score(self, candidate: InformationCandidate) -> float:
        value = _digest(
            {
                "seed": self.seed,
                "policy_uid": self.policy_uid,
                "information_action_uid": candidate.information_action.action_uid,
            }
        )
        return int(value[:16], 16) / float(16**16 - 1)

    def _decision_risk_score(
        self,
        posterior: PosteriorBoundState,
        query: InformationCandidate,
        decision_candidates: Sequence[InformationCandidate],
    ) -> float:
        ranking = rank_interventions(posterior, decision_candidates)
        top_uid = str(ranking[0]["candidate_uid"])
        by_uid = {candidate.candidate_uid: candidate for candidate in decision_candidates}
        top_design = np.asarray(by_uid[top_uid].design, dtype=np.float64)
        query_design = np.asarray(query.design, dtype=np.float64)
        covariance = posterior.covariance
        innovation = float(
            query_design @ covariance @ query_design + posterior.noise_variance
        )
        if innovation <= 0.0 or not math.isfinite(innovation):
            raise ValueError("Decision-risk utility received invalid innovation variance")
        reduction = 0.0
        for candidate in decision_candidates:
            if candidate.candidate_uid == top_uid:
                continue
            contrast = top_design - np.asarray(candidate.design, dtype=np.float64)
            covariance_with_query = float(contrast @ covariance @ query_design)
            reduction += covariance_with_query * covariance_with_query / innovation
        return reduction / max(query.expected_cost, self.expected_cost_floor)

    def score(
        self,
        posterior: PosteriorBoundState,
        candidate: InformationCandidate,
        decision_candidates: Sequence[InformationCandidate],
    ) -> float:
        cost = max(candidate.expected_cost, self.expected_cost_floor)
        if self.policy_uid == "R_random_information_action":
            return self._random_score(candidate)
        if self.policy_uid == "F_fixed_cascade":
            try:
                position = self.fixed_order.index(candidate.candidate_uid)
            except ValueError as exc:
                raise ValueError(
                    f"Fixed cascade lacks candidate {candidate.candidate_uid}"
                ) from exc
            return float(len(self.fixed_order) - position)
        if self.policy_uid == "U_uncertainty_only":
            _, std = posterior.action_response(np.asarray(candidate.design))
            return float(std * std / cost)
        return self._decision_risk_score(posterior, candidate, decision_candidates)

    def rank(
        self,
        posterior: PosteriorBoundState,
        candidates: Sequence[InformationCandidate],
        *,
        used_information_action_uids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        used = {str(value) for value in used_information_action_uids}
        available = [
            candidate
            for candidate in candidates
            if candidate.information_action.action_uid not in used
        ]
        if not available:
            return []
        rows = [
            {
                "candidate_uid": candidate.candidate_uid,
                "information_action_uid": candidate.information_action.action_uid,
                "world_action_uid": candidate.world_action.action_uid,
                "utility": self.score(posterior, candidate, candidates),
                "posterior_uid": posterior.posterior_uid,
            }
            for candidate in available
        ]
        rows.sort(key=lambda row: (-float(row["utility"]), str(row["candidate_uid"])))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def select(
        self,
        posterior: PosteriorBoundState,
        candidates: Sequence[InformationCandidate],
        *,
        used_information_action_uids: Iterable[str] = (),
    ) -> tuple[InformationCandidate, list[dict[str, Any]]]:
        ranking = self.rank(
            posterior,
            candidates,
            used_information_action_uids=used_information_action_uids,
        )
        if not ranking:
            raise ValueError("No useful query is available under the frozen policy")
        selected_uid = str(ranking[0]["information_action_uid"])
        selected = next(
            candidate
            for candidate in candidates
            if candidate.information_action.action_uid == selected_uid
        )
        return selected, ranking


__all__ = [
    "InformationCandidate",
    "POLICY_UIDS",
    "StaticInformationPolicy",
    "normalized_rank_disagreement",
    "posterior_discrepancy_proxy",
    "rank_interventions",
]
