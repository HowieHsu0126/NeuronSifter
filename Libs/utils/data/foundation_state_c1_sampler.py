"""Shared objective-requirement planning for the frozen C1 sampler.

The capability audit and the runtime loader must answer the same feasibility
question.  This module owns that pure planning logic; callers provide only the
coordinate-specific ranking policy used to choose among valid plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


SAMPLER_CONTRACT_VERSION = "neuronsifter-c1-objective-aware-unit-sampler-v1"
C1_SAMPLER_BATCH_SIZE = 6
C1_MAX_OBSERVATIONS_PER_BIOLOGICAL_UNIT = 8
OBJECTIVE_CONSTRAINED_ROW_FILL_POLICY = "objective_constrained_row_proportional_fill"
C1_REPLAY_POLICY = "fixed_multi_coordinate_exact_loader_double_replay_v1"
C1_REPLAY_COORDINATES = ((0, 0), (0, 1), (1, 0))


@dataclass(frozen=True)
class C1ObjectiveRequirementRef:
    canonical_uid: str
    biological_unit_uid: str
    family_uid: str
    source_id: str
    ordinal: int
    semantic_key: str = ""
    action_available: bool = False

    @property
    def locator(self) -> tuple[str, int]:
        return self.canonical_uid, self.ordinal


RequirementRanker = Callable[[str, tuple[object, ...]], str]


def deduplicate_objective_requirement_refs(
    refs: Sequence[C1ObjectiveRequirementRef],
) -> list[C1ObjectiveRequirementRef]:
    """Deduplicate exact observation locators and reject semantic conflicts."""

    unique: dict[tuple[str, int], C1ObjectiveRequirementRef] = {}
    for ref in refs:
        existing = unique.get(ref.locator)
        if existing is not None and existing != ref:
            raise ValueError(
                "C1 objective observation locator has conflicting semantics: "
                f"{ref.locator}"
            )
        unique[ref.locator] = ref
    return list(unique.values())


def objective_requirement_refs_fit(
    refs: Sequence[C1ObjectiveRequirementRef],
    *,
    batch_size: int,
    max_observations_per_unit: int,
    family_quotas: Mapping[str, int],
) -> bool:
    """Return whether requirements fit the exact unit, cap, and family quotas."""

    if batch_size <= 0 or max_observations_per_unit <= 0:
        raise ValueError("C1 sampler batch size and observation cap must be positive")
    unique = deduplicate_objective_requirement_refs(refs)
    group_families: dict[str, str] = {}
    canonical_counts: dict[str, int] = {}
    for ref in unique:
        existing = group_families.setdefault(ref.biological_unit_uid, ref.family_uid)
        if existing != ref.family_uid:
            raise ValueError(
                "C1 objective requirement crosses biological-unit families"
            )
        canonical_counts[ref.canonical_uid] = (
            canonical_counts.get(ref.canonical_uid, 0) + 1
        )
    if len(group_families) > batch_size:
        return False
    if any(count > max_observations_per_unit for count in canonical_counts.values()):
        return False
    if family_quotas:
        required_by_family: dict[str, int] = {}
        for family in group_families.values():
            required_by_family[family] = required_by_family.get(family, 0) + 1
        if any(
            count > family_quotas.get(family, 0)
            for family, count in required_by_family.items()
        ):
            return False
    return True


def plan_objective_requirements(
    semantic_refs: Sequence[C1ObjectiveRequirementRef],
    action_refs: Sequence[C1ObjectiveRequirementRef],
    *,
    alignment_required: bool,
    action_required: bool,
    batch_size: int,
    max_observations_per_unit: int,
    family_quotas: Mapping[str, int],
    ranker: RequirementRanker,
) -> list[C1ObjectiveRequirementRef] | None:
    """Find one valid alignment/action requirement set for an exact batch scope."""

    if not alignment_required and not action_required:
        return []
    if alignment_required and batch_size < 3:
        return None
    if action_required and not action_refs:
        return None

    ranked_actions: list[C1ObjectiveRequirementRef] | None = None

    def action_candidates() -> list[C1ObjectiveRequirementRef]:
        nonlocal ranked_actions
        if ranked_actions is None:
            ranked_actions = sorted(
                action_refs,
                key=lambda ref: (
                    ranker(
                        "action",
                        (
                            ref.biological_unit_uid,
                            ref.canonical_uid,
                            ref.ordinal,
                        ),
                    ),
                    ref.biological_unit_uid,
                    ref.canonical_uid,
                    ref.ordinal,
                ),
            )
        return ranked_actions

    def with_action(
        base: Sequence[C1ObjectiveRequirementRef],
    ) -> list[C1ObjectiveRequirementRef] | None:
        base_unique = deduplicate_objective_requirement_refs(base)
        # Adding an action cannot repair an exceeded unit, observation, or
        # family quota.  Reject monotonically before scanning action rows.
        if not objective_requirement_refs_fit(
            base_unique,
            batch_size=batch_size,
            max_observations_per_unit=max_observations_per_unit,
            family_quotas=family_quotas,
        ):
            return None
        if not action_required:
            return base_unique
        base_groups = {ref.biological_unit_uid for ref in base_unique}
        actions = action_candidates()
        candidates = [
            ref for ref in actions if ref.biological_unit_uid in base_groups
        ] + [ref for ref in actions if ref.biological_unit_uid not in base_groups]
        for action_ref in candidates:
            candidate = deduplicate_objective_requirement_refs(
                [*base_unique, action_ref]
            )
            if objective_requirement_refs_fit(
                candidate,
                batch_size=batch_size,
                max_observations_per_unit=max_observations_per_unit,
                family_quotas=family_quotas,
            ):
                return candidate
        return None

    if not alignment_required:
        return with_action([])

    by_key_source: dict[str, dict[str, list[C1ObjectiveRequirementRef]]] = {}
    for ref in semantic_refs:
        by_key_source.setdefault(ref.semantic_key, {}).setdefault(
            ref.source_id, []
        ).append(ref)
    positive_pairs: list[
        tuple[C1ObjectiveRequirementRef, C1ObjectiveRequirementRef]
    ] = []
    for semantic_key in sorted(by_key_source):
        by_source = by_key_source[semantic_key]
        sources = sorted(by_source)
        for source_index, source_a in enumerate(sources):
            for source_b in sources[source_index + 1 :]:
                for ref_a in by_source[source_a]:
                    for ref_b in by_source[source_b]:
                        if ref_a.biological_unit_uid != ref_b.biological_unit_uid:
                            positive_pairs.append((ref_a, ref_b))
    positive_pairs.sort(
        key=lambda pair: (
            ranker(
                "alignment-positive",
                (
                    pair[0].semantic_key,
                    pair[0].source_id,
                    pair[0].biological_unit_uid,
                    pair[0].canonical_uid,
                    pair[0].ordinal,
                    pair[1].source_id,
                    pair[1].biological_unit_uid,
                    pair[1].canonical_uid,
                    pair[1].ordinal,
                ),
            ),
            pair[0].locator,
            pair[1].locator,
        )
    )
    for positive_a, positive_b in positive_pairs:
        negatives = [
            ref for ref in semantic_refs if ref.semantic_key != positive_a.semantic_key
        ]
        negatives.sort(
            key=lambda ref: (
                ranker(
                    "alignment-negative",
                    (
                        positive_a.semantic_key,
                        ref.semantic_key,
                        ref.biological_unit_uid,
                        ref.canonical_uid,
                        ref.ordinal,
                    ),
                ),
                ref.semantic_key,
                ref.locator,
            )
        )
        for negative in negatives:
            candidate = with_action([positive_a, positive_b, negative])
            if candidate is not None:
                return candidate
    return None
