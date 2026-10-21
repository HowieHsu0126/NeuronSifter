"""Fail-closed contracts and metrics for Round-18 literature replay.

This module is intentionally pure: it performs no retrieval, file access,
tool execution, model training, or state mutation. Validators return sorted
blockers and metric functions return ``None`` when a quantity is undefined.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import math
from pathlib import PurePosixPath
import re
from typing import Any

from Libs.utils.data.trajectory_contracts import (
    blocker,
    canonical_sha256,
    is_sha256,
    sorted_blockers,
    stable_uid,
)


SCHEMA_VERSIONS = {
    "literature_replay_episode": "neuronsifter-literature-replay-episode-v1",
    "literature_cutoff": "neuronsifter-literature-cutoff-v1",
    "pre_cutoff_evidence_bundle": "neuronsifter-pre-cutoff-evidence-bundle-v1",
    "held_out_outcome": "neuronsifter-held-out-outcome-v1",
    "benchmark_label": "neuronsifter-benchmark-label-v1",
    "sequential_reveal_event": "neuronsifter-sequential-reveal-event-v1",
    "benchmark_prediction": "neuronsifter-benchmark-prediction-v1",
    "benchmark_ranking": "neuronsifter-benchmark-ranking-v1",
    "benchmark_abstention": "neuronsifter-benchmark-abstention-v1",
    "benchmark_metric_result": "neuronsifter-benchmark-metric-result-v1",
    "memory_leakage_risk": "neuronsifter-memory-leakage-risk-v1",
    "public_source_boundary": "neuronsifter-public-source-boundary-v1",
    "literature_replay_run": "neuronsifter-literature-replay-run-v1",
    "round18_benchmark_release": "neuronsifter-round18-benchmark-release-v1",
}

TRACK_A_ENDPOINT_DIRECTION = "endpoint_direction_prediction"
TRACK_B_INTERVENTION_RANKING = "intervention_ranking"
TRACK_C_MECHANISM_DISCRIMINATION = "mechanism_discrimination"
TRACK_D_INFORMATION_ACTION = "information_action_selection"
TRACK_E_SEQUENTIAL_REPLAY = "sequential_hypothesis_replay"
TRACK_F_ABSTENTION_BOUNDARY = "abstention_boundary_compliance"

TRACK_A = TRACK_A_ENDPOINT_DIRECTION
TRACK_B = TRACK_B_INTERVENTION_RANKING
TRACK_C = TRACK_C_MECHANISM_DISCRIMINATION
TRACK_D = TRACK_D_INFORMATION_ACTION
TRACK_E = TRACK_E_SEQUENTIAL_REPLAY
TRACK_F = TRACK_F_ABSTENTION_BOUNDARY

TRACKS_BY_CODE = {
    "A": TRACK_A,
    "B": TRACK_B,
    "C": TRACK_C,
    "D": TRACK_D,
    "E": TRACK_E,
    "F": TRACK_F,
}
BENCHMARK_TRACKS = frozenset(TRACKS_BY_CODE.values())

MEMORY_TRACK_RETRIEVAL_ISOLATED = "retrieval_isolated_replay"
MEMORY_TRACK_POST_MODEL_CUTOFF = "post_model_cutoff_replay"
MEMORY_TRACK_STRESS = "memory_risk_stress_track"
MEMORY_RISK_TRACKS = frozenset(
    {
        MEMORY_TRACK_RETRIEVAL_ISOLATED,
        MEMORY_TRACK_POST_MODEL_CUTOFF,
        MEMORY_TRACK_STRESS,
    }
)

DIRECTION_LABELS = (
    "increase",
    "decrease",
    "no_clear_effect",
    "mixed_or_context_dependent",
    "insufficient_information",
)
LABEL_CLASSES = frozenset(
    {
        "data_derived_label",
        "author_claim_label",
        "cross_study_consensus_label",
    }
)
TARGET_TYPES = frozenset(
    {
        "directional_endpoint",
        "numeric_endpoint",
        "candidate_ranking",
        "mechanism_status",
        "information_action_value",
        "hypothesis_status",
    }
)
PUBLIC_SOURCE_CLASSES = frozenset(
    {
        "open_access_article",
        "public_abstract",
        "public_supplement",
        "public_accession_data",
        "public_model_repository",
        "public_code_repository",
        "public_benchmark_database",
        "public_institutional_repository",
    }
)
PUBLIC_ACCESS_STATUSES = frozenset(
    {
        "public",
        "open_access",
        "publicly_available",
        "public_abstract_only",
    }
)

TEMPORAL_POLICY_VERSION = "neuronsifter-public-availability-timestamp-policy-v1"
TIMESTAMP_TYPE_PRIORITY = (
    "publisher_publication_date",
    "preprint_public_date",
    "repository_release_date",
    "accession_public_release_date",
    "supplement_release_date",
    "model_code_public_release_date",
)
ALLOWED_TIMESTAMP_TYPES = frozenset(TIMESTAMP_TYPE_PRIORITY)

PARTITIONS = frozenset({"development", "qualification", "final_held_out", "stress"})
EPISODE_STATUSES = frozenset(
    {
        "draft",
        "development",
        "qualified",
        "blocked",
        "rejected",
        "invalidated",
        "completed",
        "final_held_out_frozen",
        "stress",
    }
)
INDEPENDENCE_GROUP_FIELDS = (
    "publication_family",
    "preprint_to_journal_family",
    "accession_family",
    "study_cohort",
    "biological_unit_family",
    "compound_scaffold",
    "target_program",
    "regimen",
    "disease_context",
    "mechanism_family",
    "simulator_model_family",
    "tool_family",
)

SEQUENTIAL_REVEAL_STEPS = tuple(f"T{index}" for index in range(8))
PRE_REVEAL_STEPS = frozenset(SEQUENTIAL_REVEAL_STEPS[:5])
SEQUENTIAL_REQUIRED_HASH_KEYS = {
    "T0": ({"agent_visible_bundle"}, {"episode"}),
    "T1": ({"episode"}, {"hypothesis_proposal", "falsifier_challenge"}),
    "T2": ({"hypothesis_proposal", "falsifier_challenge"}, {"action_selection"}),
    "T3": ({"action_selection"}, {"information_action_result"}),
    "T4": ({"information_action_result"}, {"prediction"}),
    "T5": ({"held_out_bundle"}, {"held_out_outcome"}),
    "T6": ({"prediction", "held_out_outcome"}, {"score"}),
    "T7": ({"score"}, {"retrospective_assimilation"}),
}

CLAIM_SCOPE = {
    "retrospective_replay": True,
    "prospective_real_world_validation": False,
    "causal_effect_claim": False,
    "clinical_recommendation": False,
    "autonomous_discovery_claim": False,
}

FROZEN_GAP_IMPACTS = frozenset(
    {
        "benchmark_evidence_only",
        "hypothesis_prioritized",
        "information_action_prioritized",
        "gap_relevance_confirmed",
        "requires_empirical_observation",
        "requires_new_e4",
        "requires_support_domain",
        "no_effect",
    }
)
FORBIDDEN_FROZEN_GAP_IMPACTS = frozenset(
    {
        "resolved_by_correct_prediction",
        "resolved_by_agent_reasoning",
        "resolved_by_literature_narrative",
        "resolved_by_model_rollout",
    }
)

GOVERNANCE_COUNT_FIELDS = (
    "gate_violation_count",
    "held_out_access_attempt_count",
    "evidence_upgrade_violation_count",
    "conflict_loss_count",
    "empirical_gap_fabrication_count",
    "causal_language_violation_count",
    "clinical_language_violation_count",
)

_UNKNOWN_VALUES = frozenset(
    {"", "unknown", "unclear", "unresolved", "not_known", "missing"}
)
_CONTROLLED_SOURCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bAD" + r"NI\b",
        r"\bAI" + r"BL\b",
        r"\bcontrolled[-_ ]access\b",
        r"\bcredentialed[-_ ]source\b",
        r"\bcontrolled clinical repository\b",
    )
)
_POLICY_EXCLUDED_SOURCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bpredatory\b",
        r"\buntraceable\b",
    )
)
_PROHIBITED_CLAIM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcausal treatment effect\b",
        r"\bcausal effect\b",
        r"\b(?:treatment|intervention) (?:causes|caused|will cause)\b",
        r"\bcausal(?:ly)? (?:established|proven|validated)\b",
        r"\bclinical(?:ly)? (?:recommend|recommended|validated)\b",
        r"\bclinical recommendations?\b",
        r"\bpatient[- ]specific prediction\b",
        r"\bprospective clinical prediction\b",
        r"\bautonomous discovery\b",
    )
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?cookie|credential)",
    re.IGNORECASE,
)
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class LiteratureReplayEpisode:
    """Content-addressed Round-18 benchmark episode."""

    schema_version: str
    episode_uid: str
    track: str
    scientific_question: str
    cutoff_timestamp: str
    cutoff_basis: str | Mapping[str, object]
    pre_cutoff_evidence_manifest: Mapping[str, object]
    candidate_hypotheses: Sequence[object]
    candidate_actions: Sequence[object]
    candidate_information_actions: Sequence[object]
    held_out_outcome_manifest: Mapping[str, object]
    evaluation_target: Mapping[str, object]
    support_domain: object
    independence_groups: Mapping[str, object]
    leakage_controls: Mapping[str, object]
    allowed_tools: Sequence[str]
    budget: Mapping[str, object]
    expected_outputs: Sequence[str]
    label_provenance: Mapping[str, object]
    status: str
    content_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _present(value: object) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _unknown(value: object) -> bool:
    return _text(value).lower().replace(" ", "_") in _UNKNOWN_VALUES


def _required(
    value: Mapping[str, object], fields: Sequence[str], uid: str
) -> list[dict[str, object]]:
    return [
        blocker(
            "missing_required_field",
            f"{field} must be explicit.",
            object_uid=uid,
            field=field,
        )
        for field in fields
        if field not in value or value.get(field) is None
    ]


def _leaf_texts(value: object) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for item in value.values():
            result.extend(_leaf_texts(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            result.extend(_leaf_texts(item))
        return result
    return [value] if isinstance(value, str) else []


def _sensitive_paths(value: object, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if (
                _SENSITIVE_KEY_RE.search(str(key))
                and _present(item)
                and item is not False
            ):
                paths.append(path)
            paths.extend(_sensitive_paths(item, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            paths.extend(_sensitive_paths(item, f"{prefix}[{index}]"))
    return paths


def _as_mapping(
    value: LiteratureReplayEpisode | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(value, LiteratureReplayEpisode):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def episode_content_payload(
    value: LiteratureReplayEpisode | Mapping[str, object],
) -> dict[str, object]:
    """Return the canonical episode payload, excluding circular identity fields."""

    return {
        key: item
        for key, item in _as_mapping(value).items()
        if key not in {"episode_uid", "content_hash"}
    }


def episode_content_hash(value: LiteratureReplayEpisode | Mapping[str, object]) -> str:
    return canonical_sha256(episode_content_payload(value))


def canonical_episode_uid(value: LiteratureReplayEpisode | Mapping[str, object]) -> str:
    return stable_uid("literature_replay_episode", episode_content_payload(value))


def validate_episode_content_address(
    value: LiteratureReplayEpisode | Mapping[str, object],
) -> list[dict[str, object]]:
    episode = _as_mapping(value)
    expected_hash = episode_content_hash(episode)
    expected_uid = canonical_episode_uid(episode)
    blockers: list[dict[str, object]] = []
    if episode.get("content_hash") != expected_hash:
        blockers.append(
            blocker(
                "episode_content_hash_mismatch",
                "Literature replay episode content hash is invalid.",
                episode_uid=episode.get("episode_uid"),
            )
        )
    if episode.get("episode_uid") != expected_uid:
        blockers.append(
            blocker(
                "episode_uid_not_content_addressed",
                "Literature replay episode UID is not derived from canonical content.",
                episode_uid=episode.get("episode_uid"),
            )
        )
    return sorted_blockers(blockers)


def _parse_timestamp(value: object) -> tuple[datetime, str] | None:
    text = _text(value)
    if not text:
        return None
    precision = "date" if _DATE_ONLY_RE.fullmatch(text) else "datetime"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if precision == "date":
        parsed = parsed.replace(tzinfo=timezone.utc)
    elif parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc), precision


def _normalize_timestamp(value: datetime, precision: str) -> str:
    if precision == "date":
        return value.date().isoformat()
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _timestamp_records(value: object) -> list[dict[str, object]]:
    if isinstance(value, Mapping):
        rows: list[dict[str, object]] = []
        for timestamp_type, raw in value.items():
            if timestamp_type not in ALLOWED_TIMESTAMP_TYPES:
                rows.append(
                    {
                        "timestamp_type": timestamp_type,
                        "timestamp": raw,
                        "invalid_type": True,
                    }
                )
            elif isinstance(raw, Mapping):
                rows.append({"timestamp_type": timestamp_type, **dict(raw)})
            else:
                rows.append({"timestamp_type": timestamp_type, "timestamp": raw})
        return rows
    return [_mapping(row) for row in _sequence(value)]


def select_earliest_public_timestamp(
    timestamps: object,
    *,
    policy_version: str = TEMPORAL_POLICY_VERSION,
) -> dict[str, object]:
    """Select the earliest admissible timestamp under the versioned policy.

    Invalid, non-public, or unknown timestamp types are not candidates. A
    ``ValueError`` is raised when no reliable public timestamp is available.
    """

    if policy_version != TEMPORAL_POLICY_VERSION:
        raise ValueError("Unsupported literature public-availability timestamp policy.")
    candidates: list[tuple[datetime, int, str, str, str, object]] = []
    for row in _timestamp_records(timestamps):
        timestamp_type = _text(
            row.get("timestamp_type") or row.get("type") or row.get("basis")
        )
        if timestamp_type not in ALLOWED_TIMESTAMP_TYPES:
            continue
        if (
            row.get("publicly_available", True) is not True
            or row.get("verified", True) is not True
            or row.get("admissible", True) is not True
        ):
            continue
        parsed = _parse_timestamp(row.get("timestamp") or row.get("public_timestamp"))
        if parsed is None:
            continue
        value, precision = parsed
        priority = TIMESTAMP_TYPE_PRIORITY.index(timestamp_type)
        candidates.append(
            (
                value,
                priority,
                timestamp_type,
                precision,
                _text(
                    row.get("version")
                    or row.get("source_version")
                    or row.get("version_note")
                ),
                row.get("timestamp") or row.get("public_timestamp"),
            )
        )
    if not candidates:
        raise ValueError("No admissible public-availability timestamp is available.")
    value, _, timestamp_type, precision, source_version, original = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return {
        "policy_version": policy_version,
        "timestamp_type": timestamp_type,
        "timestamp": _normalize_timestamp(value, precision),
        "timestamp_precision": precision,
        "source_version": source_version or None,
        "original_timestamp": original,
    }


def validate_temporal_cutoff(value: Mapping[str, object]) -> list[dict[str, object]]:
    cutoff = dict(value)
    uid = _text(cutoff.get("cutoff_uid"))
    blockers = _required(
        cutoff,
        (
            "policy_version",
            "available_timestamps",
            "selected_timestamp",
            "temporal_ordering_status",
        ),
        uid,
    )
    if cutoff.get("schema_version") not in {None, SCHEMA_VERSIONS["literature_cutoff"]}:
        blockers.append(
            blocker(
                "literature_cutoff_schema_invalid",
                "Literature cutoff schema is not canonical.",
                cutoff_uid=uid,
            )
        )
    if cutoff.get("policy_version") != TEMPORAL_POLICY_VERSION:
        blockers.append(
            blocker(
                "temporal_policy_version_invalid",
                "Temporal cutoff policy version is not canonical.",
                cutoff_uid=uid,
            )
        )
    try:
        selected = select_earliest_public_timestamp(
            cutoff.get("available_timestamps"),
            policy_version=_text(cutoff.get("policy_version")),
        )
    except ValueError:
        selected = None
        blockers.append(
            blocker(
                "public_timestamp_unavailable",
                "No reliable public timestamp supports the cutoff.",
                cutoff_uid=uid,
            )
        )
    selected_basis = cutoff.get("selected_timestamp_type") or cutoff.get(
        "selected_basis"
    )
    if selected is not None and (
        selected_basis != selected["timestamp_type"]
        or cutoff.get("selected_timestamp") != selected["timestamp"]
    ):
        blockers.append(
            blocker(
                "earliest_public_timestamp_not_selected",
                "Cutoff does not use the earliest admissible public timestamp.",
                cutoff_uid=uid,
            )
        )
    if cutoff.get("temporal_ordering_status") not in {
        "ordered",
        "unambiguous",
        "ambiguous",
        "blocked",
        "conflicting",
    }:
        blockers.append(
            blocker(
                "temporal_ordering_status_invalid",
                "Temporal ordering status is not canonical.",
                cutoff_uid=uid,
            )
        )
    strong_admission = cutoff.get(
        "strong_admission", cutoff.get("strong_admission_eligible")
    )
    if (
        cutoff.get("temporal_ordering_status") in {"ambiguous", "conflicting"}
        and strong_admission is not False
    ):
        blockers.append(
            blocker(
                "ambiguous_timestamp_strong_admission",
                "Ambiguous temporal ordering blocks strong admission.",
                cutoff_uid=uid,
            )
        )
    return sorted_blockers(blockers)


def _selected_source_timestamp(
    source: Mapping[str, object],
) -> tuple[datetime, str] | None:
    for field in (
        "selected_public_timestamp",
        "public_timestamp",
        "availability_timestamp",
        "timestamp",
    ):
        parsed = _parse_timestamp(source.get(field))
        if parsed is not None:
            return parsed
    basis = _mapping(source.get("public_availability"))
    for field in ("selected_timestamp", "timestamp"):
        parsed = _parse_timestamp(basis.get(field))
        if parsed is not None:
            return parsed
    records = source.get("available_timestamps") or basis.get("available_timestamps")
    if records is not None:
        try:
            selected = select_earliest_public_timestamp(records)
        except ValueError:
            return None
        return _parse_timestamp(selected["timestamp"])
    return None


def validate_temporal_ordering(
    cutoff_timestamp: object,
    *,
    pre_cutoff_sources: Sequence[Mapping[str, object]],
    held_out_sources: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Require every input at or before cutoff and every outcome after it."""

    cutoff = _parse_timestamp(cutoff_timestamp)
    if cutoff is None:
        return [
            blocker(
                "cutoff_timestamp_invalid",
                "Cutoff timestamp must be an ISO-8601 date or timezone-aware datetime.",
            )
        ]
    cutoff_value, cutoff_precision = cutoff
    blockers: list[dict[str, object]] = []
    for role, sources in (
        ("pre_cutoff", pre_cutoff_sources),
        ("held_out", held_out_sources),
    ):
        for source in sources:
            uid = _text(source.get("source_uid") or source.get("outcome_uid"))
            timestamp = _selected_source_timestamp(source)
            if timestamp is None:
                blockers.append(
                    blocker(
                        "public_timestamp_unavailable",
                        "Source lacks a reliable public timestamp.",
                        source_uid=uid,
                        role=role,
                    )
                )
                continue
            if (
                source.get("temporal_ordering_status") == "ambiguous"
                and source.get("strong_admission", True) is not False
            ):
                blockers.append(
                    blocker(
                        "ambiguous_timestamp_strong_admission",
                        "Ambiguous temporal ordering blocks strong admission.",
                        source_uid=uid,
                    )
                )
            same_date_mixed_precision = (
                timestamp[0].date() == cutoff_value.date()
                and timestamp[1] != cutoff_precision
            )
            if (
                same_date_mixed_precision
                and source.get("strong_admission", True) is not False
            ):
                blockers.append(
                    blocker(
                        "ambiguous_timestamp_strong_admission",
                        "Date-only and datetime evidence on the cutoff date is temporally ambiguous.",
                        source_uid=uid,
                    )
                )
            after_cutoff = (
                timestamp[0].date() > cutoff_value.date()
                if cutoff_precision == "date"
                else timestamp[0] > cutoff_value
            )
            if role == "pre_cutoff" and after_cutoff:
                blockers.append(
                    blocker(
                        "future_evidence_in_pre_cutoff_bundle",
                        "Pre-cutoff evidence became public after the cutoff.",
                        source_uid=uid,
                    )
                )
            if role == "held_out" and not after_cutoff:
                blockers.append(
                    blocker(
                        "held_out_outcome_not_after_cutoff",
                        "Held-out outcome must become public strictly after cutoff.",
                        source_uid=uid,
                    )
                )
    return sorted_blockers(blockers)


def validate_public_source(value: Mapping[str, object]) -> list[dict[str, object]]:
    source = dict(value)
    uid = _text(source.get("source_uid"))
    blockers = _required(
        source,
        (
            "source_uid",
            "source_class",
            "source_identity",
            "source_version",
            "access_status",
            "license_or_permitted_use",
            "content_hash",
            "retrieval_date",
            "publication_or_accession_identity",
            "parser_version",
        ),
        uid,
    )
    if source.get("source_class") not in PUBLIC_SOURCE_CLASSES:
        blockers.append(
            blocker(
                "public_source_class_invalid",
                "Source class is not allowed for Round-18.",
                source_uid=uid,
            )
        )
    if (
        source.get("access_status") not in PUBLIC_ACCESS_STATUSES
        or source.get("publicly_available", True) is not True
    ):
        blockers.append(
            blocker(
                "nonpublic_or_unknown_source",
                "Source access status must be explicitly public.",
                source_uid=uid,
            )
        )
    if source.get("controlled_access_required", False) is not False:
        blockers.append(
            blocker(
                "controlled_source_resolution",
                "Controlled-access sources are prohibited in Round-18.",
                source_uid=uid,
            )
        )
    if source.get("credential_required", False) is not False:
        blockers.append(
            blocker(
                "credentialed_source_rejected",
                "Credentialed sources are prohibited in Round-18.",
                source_uid=uid,
            )
        )
    if source.get("institutional_session_dependency", False) is not False:
        blockers.append(
            blocker(
                "institutional_session_source_rejected",
                "Institutional session dependencies are prohibited.",
                source_uid=uid,
            )
        )
    if not is_sha256(source.get("content_hash")):
        blockers.append(
            blocker(
                "public_source_hash_invalid",
                "Public source requires a SHA-256 content hash.",
                source_uid=uid,
            )
        )
    for field in (
        "source_identity",
        "source_version",
        "license_or_permitted_use",
        "publication_or_accession_identity",
        "parser_version",
    ):
        if _unknown(source.get(field)):
            blockers.append(
                blocker(
                    "public_source_provenance_unknown",
                    "Public source provenance must be known.",
                    source_uid=uid,
                    field=field,
                )
            )
    if _parse_timestamp(source.get("retrieval_date")) is None:
        blockers.append(
            blocker(
                "retrieval_date_invalid",
                "Retrieval date must be an ISO-8601 date or timezone-aware datetime.",
                source_uid=uid,
            )
        )
    if _selected_source_timestamp(source) is None:
        blockers.append(
            blocker(
                "public_timestamp_unavailable",
                "Public source requires a reliable availability timestamp.",
                source_uid=uid,
            )
        )
    timestamp_rows = _timestamp_records(source.get("public_timestamps"))
    if not timestamp_rows:
        blockers.append(
            blocker(
                "public_timestamp_record_missing",
                "Every public source must retain its versioned public timestamp records.",
                source_uid=uid,
            )
        )
    for index, row in enumerate(timestamp_rows):
        timestamp_type = _text(row.get("timestamp_type"))
        if timestamp_type not in ALLOWED_TIMESTAMP_TYPES:
            blockers.append(
                blocker(
                    "public_timestamp_type_invalid",
                    "Public timestamp type is not admitted by the versioned policy.",
                    source_uid=uid,
                    timestamp_index=index,
                    timestamp_type=timestamp_type,
                )
            )
        if _parse_timestamp(row.get("timestamp") or row.get("public_timestamp")) is None:
            blockers.append(
                blocker(
                    "public_timestamp_value_invalid",
                    "Every retained public timestamp must be ISO-8601 and timezone-aware.",
                    source_uid=uid,
                    timestamp_index=index,
                )
            )
        if _unknown(row.get("version") or row.get("source_version")):
            blockers.append(
                blocker(
                    "public_timestamp_version_unknown",
                    "Every retained public timestamp must identify its version or basis.",
                    source_uid=uid,
                    timestamp_index=index,
                )
            )
        verified = row.get("verified")
        if not isinstance(verified, bool):
            blockers.append(
                blocker(
                    "public_timestamp_verification_missing",
                    "Every retained public timestamp must declare verification status.",
                    source_uid=uid,
                    timestamp_index=index,
                )
            )
        elif verified is False and row.get("admissible", False) is True:
            blockers.append(
                blocker(
                    "public_timestamp_unverified",
                    "An unverified public timestamp cannot be used as an admissible cutoff basis.",
                    source_uid=uid,
                    timestamp_index=index,
                )
            )
    text = " ".join(_leaf_texts(source))
    if any(pattern.search(text) for pattern in _CONTROLLED_SOURCE_PATTERNS):
        blockers.append(
            blocker(
                "controlled_source_identifier",
                "Source contains a prohibited controlled-source route or identity.",
                source_uid=uid,
            )
        )
    source_policy_text = " ".join(
        _text(source.get(field))
        for field in ("source_identity", "source_quality", "access_status")
    )
    if any(
        pattern.search(source_policy_text)
        for pattern in _POLICY_EXCLUDED_SOURCE_PATTERNS
    ):
        blockers.append(
            blocker(
                "policy_excluded_source",
                "Policy-excluded or untraceable source material cannot enter Round-18.",
                source_uid=uid,
            )
        )
    sensitive_paths = _sensitive_paths(source)
    if sensitive_paths:
        blockers.append(
            blocker(
                "credential_material_present",
                "Source record contains credential-bearing fields.",
                source_uid=uid,
                paths=sorted(set(sensitive_paths)),
            )
        )
    return sorted_blockers(blockers)


def validate_public_source_boundary(
    value: Mapping[str, object],
    *,
    sources: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    boundary = dict(value)
    blockers: list[dict[str, object]] = []
    if boundary.get("schema_version") not in {
        None,
        SCHEMA_VERSIONS["public_source_boundary"],
    }:
        blockers.append(
            blocker(
                "public_source_boundary_schema_invalid",
                "Public-source boundary schema is not canonical.",
            )
        )
    if boundary.get("strict_public_only_flag") is not True:
        blockers.append(
            blocker(
                "strict_public_only_disabled",
                "Round-18 requires strict public-only resolution.",
            )
        )
    if boundary.get("controlled_access_required") is not False:
        blockers.append(
            blocker(
                "controlled_source_resolution",
                "Controlled-access resolution must remain disabled.",
            )
        )
    if boundary.get("resolved_controlled_source_count") != 0:
        blockers.append(
            blocker(
                "controlled_source_resolution",
                "Resolved controlled-source count must remain zero.",
            )
        )
    if boundary.get("unknown_access_status_count", 0) != 0:
        blockers.append(
            blocker(
                "unknown_access_status",
                "Unknown source access status is not admissible.",
            )
        )
    for source in sources:
        blockers.extend(validate_public_source(source))
    boundary_for_sensitive_scan = dict(boundary)
    boundary_for_sensitive_scan.pop("negative_controls", None)
    sensitive_paths = _sensitive_paths(boundary_for_sensitive_scan)
    if sensitive_paths:
        blockers.append(
            blocker(
                "credential_material_present",
                "Boundary audit contains credential-bearing fields.",
                paths=sorted(set(sensitive_paths)),
            )
        )
    return sorted_blockers(blockers)


def _bundle_hash(value: Mapping[str, object]) -> str:
    return _text(value.get("bundle_hash") or value.get("content_hash"))


def _bundle_role(value: Mapping[str, object]) -> str:
    return _text(value.get("bundle_role") or value.get("bundle_type"))


def _permission(value: Mapping[str, object], role: str) -> object:
    permissions = _mapping(value.get("read_permissions") or value.get("permissions"))
    for key in (role, f"{role}_process", f"{role}_read", f"{role}_readable"):
        if key in permissions:
            return permissions[key]
        if key in value:
            return value[key]
    return None


def _artifact_hashes(value: Mapping[str, object]) -> set[str]:
    hashes: set[str] = set()
    for field in ("artifact_hashes", "content_hashes", "member_hashes"):
        raw = value.get(field)
        if isinstance(raw, Mapping):
            hashes.update(_text(item) for item in raw.values() if is_sha256(item))
        else:
            hashes.update(_text(item) for item in _sequence(raw) if is_sha256(item))
    return hashes


def validate_bundle_isolation(
    agent_visible_pre_cutoff_bundle: Mapping[str, object],
    evaluator_only_held_out_bundle: Mapping[str, object],
) -> list[dict[str, object]]:
    visible = dict(agent_visible_pre_cutoff_bundle)
    held_out = dict(evaluator_only_held_out_bundle)
    blockers: list[dict[str, object]] = []
    if _bundle_role(visible) not in {
        "agent_visible_pre_cutoff_bundle",
        "pre_cutoff_evidence_bundle",
    }:
        blockers.append(
            blocker(
                "pre_cutoff_bundle_role_invalid",
                "Agent-visible bundle role is not canonical.",
            )
        )
    if _bundle_role(held_out) not in {
        "evaluator_only_held_out_bundle",
        "held_out_outcome_bundle",
    }:
        blockers.append(
            blocker(
                "held_out_bundle_role_invalid",
                "Evaluator-only bundle role is not canonical.",
            )
        )
    visible_uid = _text(visible.get("bundle_uid"))
    held_out_uid = _text(held_out.get("bundle_uid"))
    if not visible_uid or not held_out_uid or visible_uid == held_out_uid:
        blockers.append(
            blocker(
                "bundle_identity_not_isolated",
                "Visible and held-out bundles require distinct identities.",
            )
        )
    if not is_sha256(_bundle_hash(visible)) or not is_sha256(_bundle_hash(held_out)):
        blockers.append(
            blocker("bundle_hash_invalid", "Both bundles require SHA-256 hashes.")
        )
    visible_path = _text(visible.get("mount_path"))
    held_out_path = _text(held_out.get("mount_path"))
    if not visible_path or not held_out_path:
        blockers.append(
            blocker(
                "bundle_mount_path_missing", "Both bundle mount paths must be explicit."
            )
        )
    else:
        normalized_visible = PurePosixPath(visible_path)
        normalized_held_out = PurePosixPath(held_out_path)
        if (
            normalized_visible == normalized_held_out
            or normalized_visible in normalized_held_out.parents
            or normalized_held_out in normalized_visible.parents
        ):
            blockers.append(
                blocker(
                    "bundle_mount_paths_overlap",
                    "Pre-cutoff and held-out bundle mounts must be physically separate.",
                )
            )
    if _permission(visible, "agent") is not True:
        blockers.append(
            blocker(
                "agent_visible_bundle_unreadable",
                "Agent process must read only its pre-cutoff bundle.",
            )
        )
    if _permission(held_out, "agent") is not False:
        blockers.append(
            blocker(
                "held_out_bundle_readable_by_agent",
                "Held-out bundle must be unreadable by the Agent process.",
            )
        )
    if _permission(held_out, "evaluator") is not True:
        blockers.append(
            blocker(
                "held_out_bundle_unreadable_by_evaluator",
                "Evaluator process must be the held-out bundle reader.",
            )
        )
    overlap = _artifact_hashes(visible) & _artifact_hashes(held_out)
    if overlap:
        blockers.append(
            blocker(
                "pre_cutoff_held_out_content_overlap",
                "Bundle members overlap across the reveal boundary.",
                overlap_count=len(overlap),
            )
        )
    return sorted_blockers(blockers)


def validate_execution_isolation(
    value: Mapping[str, object],
    *,
    held_out_bundle_uid: str | None = None,
) -> list[dict[str, object]]:
    execution = dict(value)
    blockers: list[dict[str, object]] = []
    agent_identity = _text(execution.get("agent_process_identity"))
    evaluator_identity = _text(execution.get("evaluator_process_identity"))
    if (
        not agent_identity
        or not evaluator_identity
        or agent_identity == evaluator_identity
    ):
        blockers.append(
            blocker(
                "execution_identity_not_isolated",
                "Agent and evaluator require distinct process identities.",
            )
        )
    if execution.get("held_out_bundle_readable_by_agent") is not False:
        blockers.append(
            blocker(
                "held_out_bundle_readable_by_agent",
                "Execution contract must deny Agent reads of evaluator-only files.",
            )
        )
    if execution.get("agent_direct_tool_execution", False) is not False:
        blockers.append(
            blocker(
                "agent_execution_isolation_violation",
                "Agent-only execution cannot invoke tools directly.",
            )
        )
    if (
        execution.get("agent_direct_state_update", False) is not False
        or execution.get("agent_gate_authority", False) is not False
    ):
        blockers.append(
            blocker(
                "agent_execution_isolation_violation",
                "Agent cannot update state or own Gates.",
            )
        )
    target_uid = held_out_bundle_uid or _text(execution.get("held_out_bundle_uid"))
    for attempt in _sequence(execution.get("access_attempts")):
        row = _mapping(attempt)
        actor_role = _text(row.get("actor_role") or row.get("process_role"))
        actor_identity = _text(row.get("process_identity") or row.get("actor_identity"))
        target = _text(row.get("bundle_uid") or row.get("target_bundle_uid"))
        step = _text(row.get("reveal_step") or row.get("step"))
        targets_held_out = (
            bool(target_uid and target == target_uid)
            or row.get("evaluator_only") is True
        )
        agent_actor = (
            actor_role.lower() in {"agent", "twin", "agent_runtime"}
            or actor_identity == agent_identity
        )
        if targets_held_out and agent_actor:
            blockers.append(
                blocker(
                    "held_out_bundle_access_attempt",
                    "Any Agent/Twin attempt to access evaluator-only files invalidates the episode.",
                    attempt_uid=row.get("attempt_uid"),
                    step=step,
                )
            )
        if (
            targets_held_out
            and step in PRE_REVEAL_STEPS
            and row.get("outcome_exposed", False) is not False
        ):
            blockers.append(
                blocker(
                    "held_out_outcome_exposed_before_reveal",
                    "Held-out outcome was exposed before T5.",
                    attempt_uid=row.get("attempt_uid"),
                )
            )
        if (
            targets_held_out
            and row.get("access_granted") is True
            and actor_identity != evaluator_identity
        ):
            blockers.append(
                blocker(
                    "held_out_access_granted_to_non_evaluator",
                    "Only the evaluator identity may read held-out files.",
                    attempt_uid=row.get("attempt_uid"),
                )
            )
    return sorted_blockers(blockers)


def validate_scitil_runtime_trace(value: Mapping[str, object]) -> list[dict[str, object]]:
    """Re-validate the restricted Round-17 TwinKernel trace used by Round-18."""

    trace = dict(value)
    blockers: list[dict[str, object]] = []
    required = (
        "runtime_uid", "runtime_mode", "kernel_invoked", "kernel_llm_backed",
        "controller_uid", "validator_uid", "belief_state_ref", "agents",
        "proposals", "hypotheses", "gate_records", "falsifier_required",
        "falsifier_invoked", "external_tool_executed", "model_rollout_executed",
        "agent_direct_state_update", "agent_direct_gate_write",
        "assimilation_applied", "status", "content_hash",
    )
    for field in required:
        if field not in trace:
            blockers.append(blocker(
                "scitil_runtime_trace_field_missing",
                "Full SciTiL trace is missing a required field.",
                field=field,
            ))
    if blockers:
        return sorted_blockers(blockers)
    if trace.get("runtime_mode") != "round17_twin_kernel":
        blockers.append(blocker(
            "scitil_runtime_mode_invalid",
            "Full governed SciTiL must identify the accepted Round-17 TwinKernel runtime.",
        ))
    if trace.get("kernel_invoked") is not True or trace.get("kernel_llm_backed") is not False:
        blockers.append(blocker(
            "scitil_twin_kernel_not_mandatory",
            "The full replay must execute the non-LLM TwinKernel boundary.",
        ))
    if trace.get("controller_uid") != "neuronsifter_orchestration_controller":
        blockers.append(blocker(
            "scitil_noncanonical_controller",
            "Only the canonical orchestration controller may authorize the trace.",
        ))
    if trace.get("validator_uid") != "round17_scitil_twin_gated_agent_validator":
        blockers.append(blocker(
            "scitil_validator_identity_invalid",
            "Trace must bind the accepted Round-17 validator.",
        ))
    for field in (
        "external_tool_executed", "model_rollout_executed",
        "agent_direct_state_update", "agent_direct_gate_write", "assimilation_applied",
    ):
        if trace.get(field) is not False:
            blockers.append(blocker(
                "scitil_runtime_boundary_violation",
                "Round-18 full replay must remain proposal/Gate validation only.",
                field=field,
            ))
    if trace.get("status") != "validated_no_external_execution":
        blockers.append(blocker(
            "scitil_runtime_status_invalid",
            "Full replay trace has an invalid runtime status.",
        ))
    expected_hash = canonical_sha256({
        key: item for key, item in trace.items() if key != "content_hash"
    })
    if trace.get("content_hash") != expected_hash:
        blockers.append(blocker(
            "scitil_runtime_trace_hash_invalid",
            "Full replay trace content hash is invalid.",
        ))
    try:
        from Libs.utils.data.scitil_contracts import (
            CANONICAL_CONTROLLER_UID,
            ROUND17_VALIDATOR_UID,
            TwinKernel,
            validate_agent_identity,
            validate_h_gate,
            validate_i_gate,
            validate_o_gate,
            validate_w_gate,
        )
    except ImportError as exc:  # pragma: no cover - accepted runtime is required
        blockers.append(blocker(
            "scitil_contract_import_failed",
            f"Accepted Round-17 SciTiL contracts are unavailable: {exc}",
        ))
        return sorted_blockers(blockers)
    agents = [dict(item) for item in trace["agents"] if isinstance(item, Mapping)]
    proposals = [dict(item) for item in trace["proposals"] if isinstance(item, Mapping)]
    hypotheses = [dict(item) for item in trace["hypotheses"] if isinstance(item, Mapping)]
    agent_by_uid = {str(item.get("agent_uid")): item for item in agents}
    proposal_by_uid = {str(item.get("proposal_uid")): item for item in proposals}
    hypothesis_by_uid = {str(item.get("hypothesis_uid")): item for item in hypotheses}
    if len(agent_by_uid) != len(agents) or len(proposal_by_uid) != len(proposals):
        blockers.append(blocker(
            "scitil_runtime_duplicate_uid",
            "TwinKernel traces must preserve unique Agent and proposal UIDs.",
        ))
    if trace.get("controller_uid") != CANONICAL_CONTROLLER_UID:
        blockers.append(blocker(
            "scitil_controller_contract_mismatch",
            "Trace controller differs from the canonical SciTiL contract.",
        ))
    if trace.get("validator_uid") != ROUND17_VALIDATOR_UID:
        blockers.append(blocker(
            "scitil_validator_contract_mismatch",
            "Trace validator differs from the canonical SciTiL contract.",
        ))
    kernel = TwinKernel(
        controller_uid=CANONICAL_CONTROLLER_UID,
        allowlisted_adapters={"retrieval_limited_public_evidence": {"access_policy": "read_only"}},
    )
    known_beliefs = {_text(trace["belief_state_ref"])}
    for agent in agents:
        blockers.extend(validate_agent_identity(agent))
    for proposal in proposals:
        blockers.extend(kernel.validate_proposal(
            proposal,
            agent=agent_by_uid.get(str(proposal.get("agent_uid")), {}),
            belief_states=known_beliefs,
        ))
    gate_uids: set[str] = set()
    gate_types: set[str] = set()
    for record_value in trace["gate_records"]:
        record = dict(record_value) if isinstance(record_value, Mapping) else {}
        gate = dict(record.get("gate", {}))
        gate_type = _text(record.get("gate_type") or gate.get("gate_type"))
        gate_uid = _text(gate.get("gate_uid"))
        gate_types.add(gate_type)
        if not gate_uid or gate_uid in gate_uids:
            blockers.append(blocker(
                "scitil_runtime_duplicate_gate_uid",
                "TwinKernel traces must preserve unique Gate UIDs.",
                gate_uid=gate_uid,
            ))
        gate_uids.add(gate_uid)
        proposal = proposal_by_uid.get(str(record.get("proposal_uid")), {})
        if gate_type == "H-Gate":
            blockers.extend(validate_h_gate(
                gate,
                proposal=proposal,
                hypothesis=hypothesis_by_uid.get(str(record.get("hypothesis_uid")), {}),
            ))
        elif gate_type == "W-Gate":
            blockers.extend(validate_w_gate(
                gate,
                proposal=proposal,
                world_action=dict(record.get("world_action", {})),
                what_if_request=record.get("what_if_request"),
                twin_debt=dict(record.get("twin_debt", {})),
                budget_decision=dict(record.get("budget_decision", {})),
            ))
        elif gate_type == "I-Gate":
            blockers.extend(validate_i_gate(
                gate,
                proposal=proposal,
                information_action=dict(record.get("information_action", {})),
                tool_request=None,
                allowlisted_adapters=set(kernel.adapter_uids),
                budget_decision=dict(record.get("budget_decision", {})),
            ))
        elif gate_type == "O-Gate":
            blockers.extend(validate_o_gate(
                gate,
                proposal=proposal,
                assimilation_proposal={},
                assimilation_decision={},
            ))
        else:
            blockers.append(blocker(
                "scitil_runtime_gate_type_invalid",
                "Trace contains a noncanonical Gate type.",
                gate_type=gate_type,
            ))
    if not gate_types:
        blockers.append(blocker(
            "scitil_runtime_gate_trace_missing",
            "Full replay must retain its controller-owned Gate records.",
        ))
    if trace.get("falsifier_required") is True and trace.get("falsifier_invoked") is not True:
        blockers.append(blocker(
            "scitil_falsifier_missing",
            "Every positive hypothesis assessment requires the FalsifierAgent.",
        ))
    if trace.get("falsifier_invoked") is True and not any(
        str(item.get("agent_role")) == "FalsifierAgent" for item in agents
    ):
        blockers.append(blocker(
            "scitil_falsifier_identity_missing",
            "Falsifier invocation must bind the canonical FalsifierAgent role.",
        ))
    return sorted_blockers(blockers)


def validate_label_provenance(value: Mapping[str, object]) -> list[dict[str, object]]:
    label = dict(value)
    uid = _text(label.get("label_uid"))
    blockers = _required(label, ("label_class", "adjudication_status"), uid)
    label_class = label.get("label_class")
    if label_class not in LABEL_CLASSES:
        blockers.append(
            blocker(
                "label_class_invalid",
                "Benchmark label provenance class is not canonical.",
                label_uid=uid,
            )
        )
    raw_sources = label.get("source_refs", label.get("source_ref"))
    sources = (
        _sequence(raw_sources) if not isinstance(raw_sources, str) else [raw_sources]
    )
    if not sources:
        blockers.append(
            blocker(
                "label_source_missing",
                "Benchmark label requires traceable source references.",
                label_uid=uid,
            )
        )
    extraction = _text(
        label.get("label_extraction_method") or label.get("extraction_method")
    ).lower()
    if not extraction:
        blockers.append(
            blocker(
                "label_extraction_method_missing",
                "Benchmark label extraction method must be explicit.",
                label_uid=uid,
            )
        )
    target_type = _text(label.get("target_type"))
    if "narrative" in extraction and (
        target_type == "numeric_endpoint" or label.get("numeric_value") is not None
    ):
        blockers.append(
            blocker(
                "narrative_claim_as_numeric_truth",
                "Narrative claims cannot be reconstructed as numeric endpoints.",
                label_uid=uid,
            )
        )
    if (
        label_class == "data_derived_label"
        and target_type == "numeric_endpoint"
        and not _present(label.get("numerical_support_ref"))
    ):
        blockers.append(
            blocker(
                "data_derived_label_support_missing",
                "Data-derived labels require exact numerical support.",
                label_uid=uid,
            )
        )
    if label_class == "author_claim_label" and label.get("strength") not in {
        "weak",
        "directional",
        None,
    }:
        blockers.append(
            blocker(
                "author_claim_label_overstated",
                "Author-claim labels may only be weak or directional.",
                label_uid=uid,
            )
        )
    if label_class == "cross_study_consensus_label":
        groups = {
            _text(item)
            for item in _sequence(label.get("independence_group_refs"))
            if _text(item)
        }
        if len(sources) < 2 or len(groups) < 2:
            blockers.append(
                blocker(
                    "consensus_label_not_independent",
                    "Cross-study consensus requires at least two independent sources.",
                    label_uid=uid,
                )
            )
    if "content_hash" in label and not is_sha256(label.get("content_hash")):
        blockers.append(
            blocker(
                "label_content_hash_invalid",
                "Label content hash must be SHA-256.",
                label_uid=uid,
            )
        )
    return sorted_blockers(blockers)


def validate_unknown_uncertainty(
    value: object, *, object_uid: str = ""
) -> list[dict[str, object]]:
    uncertainty = _mapping(value)
    if not uncertainty:
        return [
            blocker(
                "uncertainty_missing",
                "Uncertainty must be explicit and typed.",
                object_uid=object_uid,
            )
        ]
    status = _text(uncertainty.get("status")).lower()
    if not status:
        return [
            blocker(
                "uncertainty_status_missing",
                "Uncertainty status must be explicit.",
                object_uid=object_uid,
            )
        ]
    numeric_fields = ("value", "variance", "standard_error", "width", "score")
    if status == "unknown" and any(
        uncertainty.get(field) == 0 for field in numeric_fields
    ):
        return [
            blocker(
                "unknown_uncertainty_replaced_by_zero",
                "Unknown uncertainty cannot be represented as zero.",
                object_uid=object_uid,
            )
        ]
    if status == "unknown" and uncertainty.get("unknown_components") == []:
        return [
            blocker(
                "unknown_uncertainty_components_lost",
                "Unknown uncertainty requires explicit unknown components.",
                object_uid=object_uid,
            )
        ]
    return []


def validate_outcome_target(value: Mapping[str, object]) -> list[dict[str, object]]:
    target = dict(value)
    uid = _text(target.get("target_uid"))
    blockers = _required(
        target,
        (
            "target_uid",
            "target_type",
            "measurement",
            "unit",
            "direction_semantics",
            "timepoint_or_horizon",
            "comparator",
            "uncertainty",
            "source",
            "label_extraction_method",
            "adjudication_status",
        ),
        uid,
    )
    target_type = _text(target.get("target_type"))
    if target_type not in TARGET_TYPES:
        blockers.append(
            blocker(
                "outcome_target_type_invalid",
                "Outcome target type is not canonical.",
                target_uid=uid,
            )
        )
    comparator = target.get("comparator")
    comparator_unknown = _unknown(comparator) or (
        isinstance(comparator, Mapping) and _unknown(comparator.get("status"))
    )
    if comparator_unknown:
        blockers.append(
            blocker(
                "outcome_comparator_unknown",
                "Outcome comparator must be explicit.",
                target_uid=uid,
            )
        )
    timepoint = target.get("timepoint_or_horizon")
    timepoint_unknown = _unknown(timepoint) or (
        isinstance(timepoint, Mapping) and _unknown(timepoint.get("status"))
    )
    if timepoint_unknown:
        blockers.append(
            blocker(
                "outcome_timepoint_unknown",
                "Outcome timepoint or horizon must be explicit.",
                target_uid=uid,
            )
        )
    if target.get("unit_conflict") is True or _text(
        target.get("unit_status")
    ).lower() in {"conflict", "incompatible"}:
        blockers.append(
            blocker(
                "outcome_unit_conflict",
                "Outcome target has an unresolved unit conflict.",
                target_uid=uid,
            )
        )
    extraction = _text(target.get("label_extraction_method")).lower()
    measurement = target.get("measurement")
    if target_type == "numeric_endpoint" and (
        "narrative" in extraction
        or isinstance(measurement, str)
        and _unknown(measurement)
    ):
        blockers.append(
            blocker(
                "narrative_claim_as_numeric_truth",
                "Numeric endpoints require traceable numerical extraction.",
                target_uid=uid,
            )
        )
    action_identity = target.get("action_identity")
    action_unresolved = _unknown(action_identity) or (
        isinstance(action_identity, Mapping) and _unknown(action_identity.get("status"))
    )
    if target_type in {"candidate_ranking", "information_action_value"} and (
        action_identity is None or action_unresolved
    ):
        blockers.append(
            blocker(
                "outcome_action_identity_unresolved",
                "Action identity must be resolved before replay.",
                target_uid=uid,
            )
        )
    if target.get("biological_context_compatible") is False:
        blockers.append(
            blocker(
                "outcome_biological_context_incompatible",
                "Outcome target biological context is incompatible.",
                target_uid=uid,
            )
        )
    blockers.extend(
        validate_unknown_uncertainty(target.get("uncertainty"), object_uid=uid)
    )
    return sorted_blockers(blockers)


def _group_values(value: object) -> set[str]:
    raw = _sequence(value) if not isinstance(value, str) else [value]
    if not raw and value is not None and not isinstance(value, Mapping):
        raw = [value]
    return {
        _text(item) for item in raw if _text(item) and _text(item) != "not_applicable"
    }


def validate_independence_groups(
    value: Mapping[str, object],
) -> list[dict[str, object]]:
    groups = (
        _mapping(value.get("independence_groups"))
        if "independence_groups" in value
        else dict(value)
    )
    uid = _text(value.get("episode_uid"))
    blockers: list[dict[str, object]] = []
    for field in INDEPENDENCE_GROUP_FIELDS:
        if field not in groups or not _present(groups.get(field)):
            blockers.append(
                blocker(
                    "independence_group_missing",
                    "Every independence dimension must be explicit.",
                    episode_uid=uid,
                    dimension=field,
                )
            )
    return sorted_blockers(blockers)


def _episode_partition(value: Mapping[str, object]) -> str:
    controls = _mapping(value.get("leakage_controls"))
    groups = _mapping(value.get("independence_groups"))
    return _text(
        value.get("partition")
        or value.get("split")
        or controls.get("partition")
        or controls.get("split")
        or groups.get("partition")
    )


def validate_split_leakage(
    episodes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    content_memberships: dict[str, set[str]] = defaultdict(set)
    accession_memberships: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        uid = _text(episode.get("episode_uid"))
        partition = _episode_partition(episode)
        if partition not in PARTITIONS:
            blockers.append(
                blocker(
                    "partition_invalid",
                    "Episode partition is not canonical.",
                    episode_uid=uid,
                    partition=partition,
                )
            )
            continue
        groups = _mapping(episode.get("independence_groups"))
        for dimension in INDEPENDENCE_GROUP_FIELDS:
            for group_uid in _group_values(groups.get(dimension)):
                memberships[(dimension, group_uid)].add(partition)
        content_hash = _text(episode.get("content_hash"))
        if content_hash:
            content_memberships[content_hash].add(partition)
        for accession in _group_values(groups.get("accession_family")):
            accession_memberships[accession].add(partition)
        controls = _mapping(episode.get("leakage_controls"))
        if partition == "final_held_out" and (
            controls.get("accessed_before_rules_frozen") is True
            or controls.get("result_inspected") is True
            or controls.get("frozen") is not True
        ):
            blockers.append(
                blocker(
                    "final_held_out_touched_before_freeze",
                    "Final held-out episode must remain frozen and untouched.",
                    episode_uid=uid,
                )
            )
    for (dimension, group_uid), partitions in memberships.items():
        if len(partitions) > 1:
            blockers.append(
                blocker(
                    "independence_group_split_leakage",
                    "An independence family crosses benchmark partitions.",
                    dimension=dimension,
                    group_uid=group_uid,
                    partitions=sorted(partitions),
                )
            )
    for content_hash, partitions in content_memberships.items():
        if len(partitions) > 1:
            blockers.append(
                blocker(
                    "content_hash_split_leakage",
                    "Identical episode content crosses benchmark partitions.",
                    content_hash=content_hash,
                    partitions=sorted(partitions),
                )
            )
    for accession, partitions in accession_memberships.items():
        if len(partitions) > 1:
            blockers.append(
                blocker(
                    "duplicate_accession_across_partitions",
                    "An accession family crosses benchmark partitions.",
                    accession_family=accession,
                    partitions=sorted(partitions),
                )
            )
    return sorted_blockers(blockers)


def validate_split_manifest(
    value: Mapping[str, object],
    *,
    episodes: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    manifest = dict(value)
    blockers: list[dict[str, object]] = []
    declared = set(_mapping(manifest.get("partitions"))) or {
        _text(item) for item in _sequence(manifest.get("partitions"))
    }
    if declared != PARTITIONS:
        blockers.append(
            blocker(
                "partition_set_incomplete",
                "Split manifest must declare development, qualification, final_held_out, and stress.",
            )
        )
    if (
        manifest.get("decision_rules_frozen") is not True
        or manifest.get("schemas_frozen") is not True
        or manifest.get("metrics_frozen") is not True
    ):
        blockers.append(
            blocker(
                "final_held_out_rules_not_frozen",
                "Schemas, metrics, and decision rules must freeze before final evaluation.",
            )
        )
    if manifest.get("final_held_out_touched") is not False:
        blockers.append(
            blocker(
                "final_held_out_touched_before_freeze",
                "Final held-out partition must remain untouched.",
            )
        )
    blockers.extend(validate_split_leakage(episodes))
    return sorted_blockers(blockers)


def validate_memory_risk_disclosure(
    value: Mapping[str, object],
) -> list[dict[str, object]]:
    risk = dict(value)
    uid = _text(risk.get("risk_uid"))
    blockers: list[dict[str, object]] = []
    if risk.get("schema_version") not in {None, SCHEMA_VERSIONS["memory_leakage_risk"]}:
        blockers.append(
            blocker(
                "memory_risk_schema_invalid",
                "Memory-risk disclosure schema is not canonical.",
                risk_uid=uid,
            )
        )
    track = risk.get("memory_risk_track") or risk.get("track")
    raw_tracks = risk.get("tracks")
    if isinstance(raw_tracks, Mapping):
        tracks = {str(item) for item in raw_tracks}
    else:
        tracks = {str(item) for item in _sequence(raw_tracks)}
    if track is not None:
        tracks.add(str(track))
    if not tracks or not tracks.issubset(MEMORY_RISK_TRACKS):
        blockers.append(
            blocker(
                "memory_risk_track_invalid",
                "Memory-risk track is not canonical.",
                risk_uid=uid,
            )
        )
    risk_explicit = (
        risk.get("pretrained_memory_contamination_possible") is True
        or risk.get("risk_cannot_be_fully_eliminated") is True
        or risk.get("residual_risk_disclosed") is True
    )
    if not risk_explicit:
        blockers.append(
            blocker(
                "memory_contamination_risk_hidden",
                "Pretrained-memory contamination risk must remain explicit.",
                risk_uid=uid,
            )
        )
    contamination_claims = (
        risk.get("contamination_free_claim"),
        risk.get("retrieval_isolation_contamination_free"),
        risk.get("retrieval_isolation_is_contamination_free"),
        risk.get("retrieval_isolation_only_contamination_free_claim"),
    )
    if any(
        item is not None and item is not False for item in contamination_claims
    ) or all(item is None for item in contamination_claims):
        blockers.append(
            blocker(
                "memory_track_claimed_contamination_free",
                "Retrieval isolation cannot be described as contamination-free.",
                risk_uid=uid,
            )
        )
    residual_disclosed = (
        _present(risk.get("residual_risk"))
        or risk.get("residual_risk_disclosed") is True
    )
    if not _present(risk.get("mitigations")) or not residual_disclosed:
        blockers.append(
            blocker(
                "memory_risk_disclosure_incomplete",
                "Memory-risk mitigations and residual risk must be explicit.",
                risk_uid=uid,
            )
        )
    reported_separately = (
        risk.get("reported_separately") is True or tracks == MEMORY_RISK_TRACKS
    )
    if not reported_separately:
        blockers.append(
            blocker(
                "memory_risk_tracks_aggregated",
                "Memory-risk tracks must be reported separately.",
                risk_uid=uid,
            )
        )
    if track == MEMORY_TRACK_POST_MODEL_CUTOFF:
        model_cutoff = _parse_timestamp(risk.get("declared_model_cutoff"))
        outcome_timestamp = _parse_timestamp(risk.get("outcome_public_timestamp"))
        if (
            model_cutoff is None
            or outcome_timestamp is None
            or outcome_timestamp[0] <= model_cutoff[0]
        ):
            blockers.append(
                blocker(
                    "post_model_cutoff_not_demonstrated",
                    "Post-model-cutoff track requires an outcome public after the declared model cutoff.",
                    risk_uid=uid,
                )
            )
    for text in _leaf_texts(risk):
        normalized = " ".join(text.lower().split())
        mentions_clean = re.search(r"\bcontamination[- ]free\b", normalized) is not None
        explicitly_negated = any(
            phrase in normalized
            for phrase in (
                "not contamination-free",
                "not contamination free",
                "cannot be contamination-free",
                "cannot be contamination free",
                "never contamination-free",
            )
        )
        if mentions_clean and not explicitly_negated:
            blockers.append(
                blocker(
                    "memory_track_claimed_contamination_free",
                    "Memory-risk narrative cannot claim contamination-free replay.",
                    risk_uid=uid,
                )
            )
            break
    return sorted_blockers(blockers)


def sequential_event_content_payload(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"event_uid", "event_hash", "content_hash"}
    }


def sequential_event_content_hash(value: Mapping[str, object]) -> str:
    return canonical_sha256(sequential_event_content_payload(value))


def canonical_sequential_event_uid(value: Mapping[str, object]) -> str:
    return stable_uid(
        "literature_reveal_event", sequential_event_content_payload(value)
    )


def _event_step(value: Mapping[str, object]) -> str:
    return _text(value.get("reveal_step") or value.get("step"))


def _event_hash(value: Mapping[str, object]) -> str:
    return _text(value.get("event_hash") or value.get("content_hash"))


def _hash_mapping_valid(value: object, *, require_nonempty: bool = False) -> bool:
    hashes = _mapping(value)
    return (bool(hashes) or not require_nonempty) and all(
        is_sha256(item) for item in hashes.values()
    )


def validate_sequential_reveal_integrity(
    events: Sequence[Mapping[str, object]],
    *,
    episode_uid: str | None = None,
) -> list[dict[str, object]]:
    rows = [dict(event) for event in events]
    blockers: list[dict[str, object]] = []
    actual_steps = [_event_step(event) for event in rows]
    if actual_steps != list(SEQUENTIAL_REVEAL_STEPS):
        blockers.append(
            blocker(
                "sequential_reveal_order_invalid",
                "Replay must contain immutable T0-T7 events in exact order.",
                observed_steps=actual_steps,
            )
        )
    previous_hash: str | None = None
    previous_time: datetime | None = None
    observed_episode_uid = episode_uid
    t4_decision_hash = ""
    for index, event in enumerate(rows):
        step = _event_step(event)
        uid = _text(event.get("event_uid"))
        if event.get("schema_version") not in {
            None,
            SCHEMA_VERSIONS["sequential_reveal_event"],
        }:
            blockers.append(
                blocker(
                    "sequential_event_schema_invalid",
                    "Sequential reveal event schema is not canonical.",
                    event_uid=uid,
                )
            )
        if event.get("sequence_index") != index:
            blockers.append(
                blocker(
                    "sequential_reveal_index_invalid",
                    "Reveal event index must match T0-T7 order.",
                    event_uid=uid,
                    step=step,
                )
            )
        event_episode_uid = _text(event.get("episode_uid"))
        observed_episode_uid = observed_episode_uid or event_episode_uid
        if not event_episode_uid or event_episode_uid != observed_episode_uid:
            blockers.append(
                blocker(
                    "sequential_reveal_episode_mismatch",
                    "All reveal events must belong to one episode.",
                    event_uid=uid,
                )
            )
        if event.get("immutable") is not True:
            blockers.append(
                blocker(
                    "sequential_event_mutable",
                    "Every reveal event must be immutable.",
                    event_uid=uid,
                    step=step,
                )
            )
        expected_hash = sequential_event_content_hash(event)
        if _event_hash(event) != expected_hash:
            blockers.append(
                blocker(
                    "sequential_event_hash_mismatch",
                    "Reveal event content hash is invalid.",
                    event_uid=uid,
                    step=step,
                )
            )
        if uid != canonical_sequential_event_uid(event):
            blockers.append(
                blocker(
                    "sequential_event_uid_not_content_addressed",
                    "Reveal event UID is not content-addressed.",
                    event_uid=uid,
                    step=step,
                )
            )
        parent_hash = event.get("prior_event_hash")
        if index == 0:
            if parent_hash not in {None, ""}:
                blockers.append(
                    blocker(
                        "sequential_event_parent_invalid",
                        "T0 cannot have a prior event hash.",
                        event_uid=uid,
                    )
                )
        elif parent_hash != previous_hash:
            blockers.append(
                blocker(
                    "sequential_event_parent_invalid",
                    "Reveal event hash chain is broken.",
                    event_uid=uid,
                    step=step,
                )
            )
        input_hashes = _mapping(event.get("input_hashes"))
        output_hashes = _mapping(event.get("output_hashes"))
        if not _hash_mapping_valid(
            input_hashes, require_nonempty=True
        ) or not _hash_mapping_valid(output_hashes, require_nonempty=True):
            blockers.append(
                blocker(
                    "sequential_event_artifact_hash_invalid",
                    "Reveal event artifact hashes must be SHA-256.",
                    event_uid=uid,
                    step=step,
                )
            )
        required_inputs, required_outputs = SEQUENTIAL_REQUIRED_HASH_KEYS.get(
            step, (set(), set())
        )
        if not required_inputs.issubset(input_hashes) or not required_outputs.issubset(
            output_hashes
        ):
            blockers.append(
                blocker(
                    "sequential_reveal_step_artifact_missing",
                    "Each replay step must record its required input and output artifacts.",
                    event_uid=uid,
                    step=step,
                    missing_inputs=sorted(required_inputs - set(input_hashes)),
                    missing_outputs=sorted(required_outputs - set(output_hashes)),
                )
            )
        timestamp = _parse_timestamp(event.get("timestamp") or event.get("recorded_at"))
        if timestamp is None:
            blockers.append(
                blocker(
                    "sequential_event_timestamp_invalid",
                    "Reveal event timestamp must be timezone-aware.",
                    event_uid=uid,
                    step=step,
                )
            )
        elif previous_time is not None and timestamp[0] < previous_time:
            blockers.append(
                blocker(
                    "sequential_event_timestamp_regression",
                    "Reveal event timestamps cannot regress.",
                    event_uid=uid,
                    step=step,
                )
            )
        if timestamp is not None:
            previous_time = timestamp[0]
        revealed = event.get(
            "held_out_revealed",
            event.get(
                "outcome_revealed", event.get("held_out_outcome_revealed", False)
            ),
        )
        accessed = event.get(
            "held_out_accessed", event.get("held_out_bundle_accessed", False)
        )
        if event.get("preserved") is not True:
            blockers.append(
                blocker(
                    "sequential_event_not_preserved",
                    "Every reveal event must remain preserved.",
                    event_uid=uid,
                    step=step,
                )
            )
        if step in PRE_REVEAL_STEPS and (
            revealed is not False or accessed is not False
        ):
            blockers.append(
                blocker(
                    "future_outcome_access_before_t5",
                    "Held-out outcome cannot be revealed or accessed before T5.",
                    event_uid=uid,
                    step=step,
                )
            )
        if step == "T5" and revealed is not True:
            blockers.append(
                blocker(
                    "held_out_outcome_not_revealed_at_t5",
                    "Evaluator reveal must occur explicitly at T5.",
                    event_uid=uid,
                )
            )
        if step == "T4":
            t4_decision_hash = _text(
                event.get("decision_hash")
                or output_hashes.get("prediction")
                or output_hashes.get("ranking")
                or output_hashes.get("abstention")
            )
            if not is_sha256(t4_decision_hash):
                blockers.append(
                    blocker(
                        "pre_cutoff_decision_hash_missing",
                        "T4 must freeze a content-addressed decision.",
                        event_uid=uid,
                    )
                )
        if (
            step in {"T5", "T6", "T7"}
            and t4_decision_hash
            and event.get("pre_cutoff_decision_hash") != t4_decision_hash
        ):
            blockers.append(
                blocker(
                    "pre_cutoff_decision_retroactively_changed",
                    "Post-reveal events must preserve the T4 decision hash.",
                    event_uid=uid,
                    step=step,
                )
            )
        previous_hash = _event_hash(event)
    return sorted_blockers(blockers)


validate_sequential_reveal = validate_sequential_reveal_integrity


def validate_abstention_semantics(
    value: Mapping[str, object],
) -> list[dict[str, object]]:
    result = dict(value)
    uid = _text(result.get("prediction_uid") or result.get("abstention_uid"))
    blockers: list[dict[str, object]] = []
    evidence_status = _text(
        result.get("evidence_status") or result.get("support_status")
    ).lower()
    abstained = result.get("abstained") is True or result.get("decision") == "abstain"
    answered = result.get("answered") is True or (
        result.get("answered") is None and not abstained
    )
    requires_abstention = (
        evidence_status
        in {
            "insufficient_information",
            "unsupported",
            "outside_support_domain",
            "unknown",
            "conflicting_unresolved",
            "missing_action_information",
        }
        or result.get("abstention_required") is True
    )
    if result.get("forced_answer") is True:
        blockers.append(
            blocker(
                "forced_answer_prompt_not_rejected",
                "Forced-answer requests must fail closed.",
                object_uid=uid,
            )
        )
    if requires_abstention and answered:
        blockers.append(
            blocker(
                "unsupported_episode_answered",
                "Unsupported or insufficient evidence requires abstention.",
                object_uid=uid,
            )
        )
    prediction = _text(result.get("prediction") or result.get("predicted_label"))
    if (
        abstained
        and prediction
        and prediction not in {"insufficient_information", "abstain"}
    ):
        blockers.append(
            blocker(
                "abstention_contains_substantive_prediction",
                "An abstention cannot contain a substantive prediction.",
                object_uid=uid,
            )
        )
    if abstained and not _present(
        result.get("abstention_reason")
        or result.get("reason")
        or result.get("reason_codes")
    ):
        blockers.append(
            blocker(
                "abstention_reason_missing",
                "Abstention requires an explicit reason.",
                object_uid=uid,
            )
        )
    if "uncertainty" in result:
        blockers.extend(
            validate_unknown_uncertainty(result.get("uncertainty"), object_uid=uid)
        )
    return sorted_blockers(blockers)


def validate_claim_boundaries(value: Mapping[str, object]) -> list[dict[str, object]]:
    artifact = dict(value)
    uid = _text(
        artifact.get("result_uid")
        or artifact.get("prediction_uid")
        or artifact.get("episode_uid")
    )
    blockers: list[dict[str, object]] = []
    for field, expected in CLAIM_SCOPE.items():
        if artifact.get(field) is not expected:
            code = {
                "prospective_real_world_validation": "prospective_validation_claim",
                "causal_effect_claim": "causal_language_violation",
                "clinical_recommendation": "clinical_language_violation",
                "autonomous_discovery_claim": "autonomous_discovery_claim",
                "retrospective_replay": "retrospective_replay_scope_missing",
            }[field]
            blockers.append(
                blocker(
                    code, f"{field} must be {expected!r} for Round-18.", object_uid=uid
                )
            )
    text = " ".join(_leaf_texts(artifact))
    if any(pattern.search(text) for pattern in _PROHIBITED_CLAIM_PATTERNS):
        blockers.append(
            blocker(
                "prohibited_claim_language",
                "Artifact contains causal, clinical, prospective, or autonomous-discovery language.",
                object_uid=uid,
            )
        )
    return sorted_blockers(blockers)


validate_claim_language = validate_claim_boundaries


def validate_frozen_gap_isolation(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_count: int = 20,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    uids = [_text(row.get("gap_uid")) for row in rows]
    if (
        len(rows) != expected_count
        or len(set(uids)) != expected_count
        or any(not uid for uid in uids)
    ):
        blockers.append(
            blocker(
                "frozen_gap_count_drift",
                f"Exactly {expected_count} unique frozen gaps must remain represented.",
            )
        )
    for row in rows:
        uid = _text(row.get("gap_uid"))
        impact = row.get("round18_impact") or row.get("impact")
        if impact not in FROZEN_GAP_IMPACTS:
            blockers.append(
                blocker(
                    "frozen_gap_impact_invalid",
                    "Round-18 frozen-gap impact is not allowed.",
                    gap_uid=uid,
                    impact=impact,
                )
            )
        if impact in FORBIDDEN_FROZEN_GAP_IMPACTS:
            blockers.append(
                blocker(
                    "benchmark_result_closes_empirical_gap",
                    "Benchmark outputs cannot resolve empirical Evidence Fabric gaps.",
                    gap_uid=uid,
                )
            )
        for forbidden in FORBIDDEN_FROZEN_GAP_IMPACTS:
            if row.get(forbidden) not in {None, False, 0, "false", "0"}:
                blockers.append(
                    blocker(
                        "benchmark_result_closes_empirical_gap",
                        "Benchmark outputs cannot resolve empirical Evidence Fabric gaps.",
                        gap_uid=uid,
                        field=forbidden,
                    )
                )
        if row.get("resolved") not in {None, False, 0, "false", "0"}:
            blockers.append(
                blocker(
                    "benchmark_result_closes_empirical_gap",
                    "Round-18 cannot mark a frozen empirical gap resolved.",
                    gap_uid=uid,
                )
            )
        if row.get("frozen_gap_count_after_round18", expected_count) != expected_count:
            blockers.append(
                blocker(
                    "frozen_gap_count_drift",
                    "Round-18 must preserve the frozen gap count.",
                    gap_uid=uid,
                )
            )
    return sorted_blockers(blockers)


def validate_unique_uids(
    objects: Sequence[Mapping[str, object]],
    uid_fields: Sequence[str] = (
        "episode_uid",
        "event_uid",
        "target_uid",
        "label_uid",
        "run_uid",
        "attempt_uid",
    ),
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in objects:
        uid = next(
            (
                _text(value.get(field))
                for field in uid_fields
                if _text(value.get(field))
            ),
            "",
        )
        if not uid or uid in seen:
            blockers.append(
                blocker(
                    "duplicate_uid",
                    "Object UIDs must be present and globally unique.",
                    object_uid=uid,
                )
            )
        seen.add(uid)
    return sorted_blockers(blockers)


def validate_lineage_acyclic(
    edges: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    graph: dict[str, set[str]] = {}
    for parent, child in edges:
        if not parent or not child:
            return [
                blocker(
                    "lineage_endpoint_missing",
                    "Lineage edges require explicit parent and child UIDs.",
                )
            ]
        graph.setdefault(parent, set()).add(child)
        graph.setdefault(child, set())
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        if not all(visit(child) for child in graph[node]):
            return False
        visiting.remove(node)
        visited.add(node)
        return True

    return (
        []
        if all(visit(node) for node in graph)
        else [blocker("lineage_cycle", "Round-18 lineage must be acyclic.")]
    )


def validate_failed_retry_preservation(
    attempts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    by_uid: dict[str, Mapping[str, object]] = {}
    for index, attempt in enumerate(attempts):
        uid = _text(attempt.get("attempt_uid"))
        if not uid or uid in by_uid:
            blockers.append(
                blocker(
                    "failed_episode_overwritten_by_retry",
                    "Attempt UIDs must be immutable and unique.",
                    attempt_uid=uid,
                )
            )
        preserved = attempt.get("preserved", attempt.get("attempt_preserved"))
        if preserved is not True:
            blockers.append(
                blocker(
                    "failed_episode_overwritten_by_retry",
                    "Every replay attempt, including failures, must be preserved.",
                    attempt_uid=uid,
                )
            )
        status = _text(attempt.get("terminal_state") or attempt.get("status")).upper()
        if status not in {
            "COMPLETED",
            "FAILED",
            "BLOCKED",
            "REJECTED",
            "INVALIDATED",
            "CANCELLED",
        }:
            blockers.append(
                blocker(
                    "replay_attempt_terminal_state_invalid",
                    "Replay attempt terminal state is not canonical.",
                    attempt_uid=uid,
                )
            )
        if status != "COMPLETED" and not _present(
            attempt.get("failure_class") or attempt.get("reason_codes")
        ):
            blockers.append(
                blocker(
                    "replay_attempt_failure_unclassified",
                    "Non-completed replay attempts require a failure class.",
                    attempt_uid=uid,
                )
            )
        retry_of = _text(attempt.get("retry_of") or attempt.get("parent_attempt_uid"))
        if retry_of:
            parent = by_uid.get(retry_of)
            parent_status = _text(
                (parent or {}).get("terminal_state") or (parent or {}).get("status")
            ).upper()
            if parent is None or parent_status == "COMPLETED":
                blockers.append(
                    blocker(
                        "replay_retry_parent_invalid",
                        "Retry must reference an earlier preserved non-completed attempt.",
                        attempt_uid=uid,
                        retry_of=retry_of,
                    )
                )
        if attempt.get("attempt_index", index) != index:
            blockers.append(
                blocker(
                    "replay_attempt_order_invalid",
                    "Attempt indices must be append-only.",
                    attempt_uid=uid,
                )
            )
        if uid:
            by_uid[uid] = attempt
    return sorted_blockers(blockers)


validate_attempt_preservation = validate_failed_retry_preservation


def _candidate_uid(value: object) -> str:
    if isinstance(value, Mapping):
        return _text(
            value.get("candidate_uid")
            or value.get("uid")
            or value.get("action_uid")
            or value.get("hypothesis_uid")
            or value.get("information_action_uid")
        )
    return _text(value)


def validate_literature_replay_episode(
    value: LiteratureReplayEpisode | Mapping[str, object],
) -> list[dict[str, object]]:
    episode = _as_mapping(value)
    uid = _text(episode.get("episode_uid"))
    required_fields = (
        "episode_uid",
        "track",
        "scientific_question",
        "cutoff_timestamp",
        "cutoff_basis",
        "pre_cutoff_evidence_manifest",
        "candidate_hypotheses",
        "candidate_actions",
        "candidate_information_actions",
        "held_out_outcome_manifest",
        "evaluation_target",
        "support_domain",
        "independence_groups",
        "leakage_controls",
        "allowed_tools",
        "budget",
        "expected_outputs",
        "label_provenance",
        "status",
        "content_hash",
    )
    blockers = _required(episode, required_fields, uid)
    if episode.get("schema_version") != SCHEMA_VERSIONS["literature_replay_episode"]:
        blockers.append(
            blocker(
                "literature_replay_episode_schema_invalid",
                "LiteratureReplayEpisode schema is not canonical.",
                episode_uid=uid,
            )
        )
    track = episode.get("track")
    if track not in BENCHMARK_TRACKS:
        blockers.append(
            blocker(
                "benchmark_track_invalid",
                "Episode benchmark track is not canonical.",
                episode_uid=uid,
            )
        )
    if not _text(episode.get("scientific_question")):
        blockers.append(
            blocker(
                "scientific_question_missing",
                "Episode scientific question must be explicit.",
                episode_uid=uid,
            )
        )
    if _parse_timestamp(episode.get("cutoff_timestamp")) is None:
        blockers.append(
            blocker(
                "cutoff_timestamp_invalid",
                "Episode cutoff timestamp is invalid.",
                episode_uid=uid,
            )
        )
    cutoff_basis_raw = episode.get("cutoff_basis")
    cutoff_basis = _mapping(cutoff_basis_raw)
    cutoff_basis_valid = (
        isinstance(cutoff_basis_raw, str)
        and cutoff_basis_raw in ALLOWED_TIMESTAMP_TYPES
    ) or cutoff_basis.get("policy_version") == TEMPORAL_POLICY_VERSION
    if not cutoff_basis_valid:
        blockers.append(
            blocker(
                "temporal_policy_version_invalid",
                "Episode cutoff basis must use the canonical policy.",
                episode_uid=uid,
            )
        )
    hypotheses = _sequence(episode.get("candidate_hypotheses"))
    actions = _sequence(episode.get("candidate_actions"))
    information_actions = _sequence(episode.get("candidate_information_actions"))
    if track == TRACK_B and (
        len(actions) < 2 or any(not _candidate_uid(item) for item in actions)
    ):
        blockers.append(
            blocker(
                "ranking_candidate_set_invalid",
                "Ranking episodes require at least two predeclared action identities.",
                episode_uid=uid,
            )
        )
    if track == TRACK_C and (
        len(hypotheses) < 2 or any(not _candidate_uid(item) for item in hypotheses)
    ):
        blockers.append(
            blocker(
                "mechanism_candidate_set_invalid",
                "Mechanism discrimination requires competing predeclared hypotheses.",
                episode_uid=uid,
            )
        )
    if track == TRACK_D and (
        not information_actions
        or any(not _candidate_uid(item) for item in information_actions)
    ):
        blockers.append(
            blocker(
                "information_action_set_invalid",
                "Information-action episodes require predeclared information actions.",
                episode_uid=uid,
            )
        )
    if track == TRACK_E and not hypotheses:
        blockers.append(
            blocker(
                "sequential_hypothesis_missing",
                "Sequential replay requires an initial hypothesis set.",
                episode_uid=uid,
            )
        )
    if not isinstance(episode.get("allowed_tools"), Sequence) or isinstance(
        episode.get("allowed_tools"), (str, bytes)
    ):
        blockers.append(
            blocker(
                "allowed_tools_invalid",
                "Allowed tools must be an explicit sequence.",
                episode_uid=uid,
            )
        )
    if not _mapping(episode.get("budget")):
        blockers.append(
            blocker(
                "benchmark_budget_missing",
                "Episode budget must be explicit.",
                episode_uid=uid,
            )
        )
    if episode.get("status") not in EPISODE_STATUSES:
        blockers.append(
            blocker(
                "episode_status_invalid",
                "Episode status is not canonical.",
                episode_uid=uid,
            )
        )
    blockers.extend(validate_episode_content_address(episode))
    blockers.extend(validate_independence_groups(episode))
    blockers.extend(validate_outcome_target(_mapping(episode.get("evaluation_target"))))
    blockers.extend(
        validate_label_provenance(_mapping(episode.get("label_provenance")))
    )
    return sorted_blockers(blockers)


validate_episode = validate_literature_replay_episode


def _validate_parallel_lengths(*values: Sequence[object]) -> int:
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError("Metric inputs must have equal lengths.")
    return next(iter(lengths), 0)


def _labels(
    y_true: Sequence[object], y_pred: Sequence[object], labels: Sequence[object] | None
) -> list[object]:
    if labels is not None:
        result = list(labels)
        if len(set(result)) != len(result):
            raise ValueError("Metric labels must be unique.")
        return result
    return sorted(set(y_true) | set(y_pred), key=lambda item: str(item))


def direction_accuracy(
    y_true: Sequence[object], y_pred: Sequence[object]
) -> float | None:
    count = _validate_parallel_lengths(y_true, y_pred)
    if count == 0:
        return None
    return (
        sum(observed == predicted for observed, predicted in zip(y_true, y_pred))
        / count
    )


def publication_macro_direction_accuracy(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    publication_groups: Sequence[object],
) -> float | None:
    count = _validate_parallel_lengths(y_true, y_pred, publication_groups)
    if count == 0:
        return None
    grouped: dict[object, list[bool]] = defaultdict(list)
    for observed, predicted, group in zip(y_true, y_pred, publication_groups):
        grouped[group].append(observed == predicted)
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def balanced_accuracy(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    *,
    labels: Sequence[object] | None = None,
) -> float | None:
    count = _validate_parallel_lengths(y_true, y_pred)
    if count == 0:
        return None
    classes = _labels(y_true, y_pred, labels)
    recalls = []
    for label in classes:
        positives = sum(observed == label for observed in y_true)
        if positives:
            recalls.append(
                sum(
                    observed == label and predicted == label
                    for observed, predicted in zip(y_true, y_pred)
                )
                / positives
            )
    return sum(recalls) / len(recalls) if recalls else None


def macro_f1(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    *,
    labels: Sequence[object] | None = None,
) -> float | None:
    count = _validate_parallel_lengths(y_true, y_pred)
    if count == 0:
        return None
    classes = _labels(y_true, y_pred, labels)
    scores: list[float] = []
    for label in classes:
        true_positive = sum(
            observed == label and predicted == label
            for observed, predicted in zip(y_true, y_pred)
        )
        false_positive = sum(
            observed != label and predicted == label
            for observed, predicted in zip(y_true, y_pred)
        )
        false_negative = sum(
            observed == label and predicted != label
            for observed, predicted in zip(y_true, y_pred)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else None


def _probability_rows(
    probabilities: Sequence[Mapping[object, float] | Sequence[float]],
    labels: Sequence[object],
) -> list[list[float]]:
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Probability labels must be non-empty and unique.")
    rows: list[list[float]] = []
    for row in probabilities:
        if isinstance(row, Mapping):
            if set(row) != set(labels):
                raise ValueError("Probability mapping keys must exactly match labels.")
            values = [float(row[label]) for label in labels]
        else:
            values = [float(value) for value in row]
            if len(values) != len(labels):
                raise ValueError("Probability row width must match labels.")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values
        ):
            raise ValueError("Probabilities must be finite values in [0, 1].")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Each probability row must sum to one.")
        rows.append(values)
    return rows


def multiclass_brier_score(
    y_true: Sequence[object],
    probabilities: Sequence[Mapping[object, float] | Sequence[float]],
    *,
    labels: Sequence[object],
) -> float | None:
    count = _validate_parallel_lengths(y_true, probabilities)
    if count == 0:
        return None
    label_list = list(labels)
    rows = _probability_rows(probabilities, label_list)
    if any(observed not in label_list for observed in y_true):
        raise ValueError("Observed label is absent from probability labels.")
    total = 0.0
    for observed, row in zip(y_true, rows):
        total += sum(
            (probability - float(label == observed)) ** 2
            for label, probability in zip(label_list, row)
        )
    return total / count


def multiclass_log_loss(
    y_true: Sequence[object],
    probabilities: Sequence[Mapping[object, float] | Sequence[float]],
    *,
    labels: Sequence[object],
    epsilon: float = 1e-15,
) -> float | None:
    count = _validate_parallel_lengths(y_true, probabilities)
    if count == 0:
        return None
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5).")
    label_list = list(labels)
    rows = _probability_rows(probabilities, label_list)
    index = {label: position for position, label in enumerate(label_list)}
    if any(observed not in index for observed in y_true):
        raise ValueError("Observed label is absent from probability labels.")
    return (
        -sum(
            math.log(min(1.0 - epsilon, max(epsilon, row[index[observed]])))
            for observed, row in zip(y_true, rows)
        )
        / count
    )


log_loss = multiclass_log_loss


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def calibration_intercept_slope(
    outcomes: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    epsilon: float = 1e-12,
    max_iterations: int = 100,
) -> dict[str, float | str | None]:
    """Fit logistic calibration on prediction logits with explicit undefined states."""

    count = _validate_parallel_lengths(outcomes, probabilities)
    undefined: dict[str, float | str | None] = {"intercept": None, "slope": None}
    if count == 0:
        return {**undefined, "status": "undefined_no_observations"}
    y = [int(value) for value in outcomes]
    if any(value not in {0, 1} for value in y):
        raise ValueError("Calibration outcomes must be binary.")
    if len(set(y)) < 2:
        return {**undefined, "status": "undefined_single_outcome_class"}
    p = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in p):
        raise ValueError("Calibration probabilities must be finite values in [0, 1].")
    logits = [
        math.log(
            min(1.0 - epsilon, max(epsilon, value))
            / (1.0 - min(1.0 - epsilon, max(epsilon, value)))
        )
        for value in p
    ]
    if max(logits) - min(logits) <= 1e-12:
        return {**undefined, "status": "undefined_constant_prediction"}
    intercept = 0.0
    slope = 1.0
    converged = False
    for _ in range(max_iterations):
        fitted = [_sigmoid(intercept + slope * logit) for logit in logits]
        weights = [max(value * (1.0 - value), 1e-15) for value in fitted]
        score_0 = sum(observed - predicted for observed, predicted in zip(y, fitted))
        score_1 = sum(
            (observed - predicted) * logit
            for observed, predicted, logit in zip(y, fitted, logits)
        )
        h00 = sum(weights)
        h01 = sum(weight * logit for weight, logit in zip(weights, logits))
        h11 = sum(weight * logit * logit for weight, logit in zip(weights, logits))
        determinant = h00 * h11 - h01 * h01
        if determinant <= 1e-18 or not math.isfinite(determinant):
            return {**undefined, "status": "undefined_singular_fit"}
        delta_intercept = (score_0 * h11 - score_1 * h01) / determinant
        delta_slope = (score_1 * h00 - score_0 * h01) / determinant
        if not math.isfinite(delta_intercept) or not math.isfinite(delta_slope):
            return {**undefined, "status": "undefined_nonfinite_fit"}
        intercept += max(-10.0, min(10.0, delta_intercept))
        slope += max(-10.0, min(10.0, delta_slope))
        if max(abs(delta_intercept), abs(delta_slope)) < 1e-10:
            converged = True
            break
    if not converged:
        return {**undefined, "status": "undefined_nonconvergent_fit"}
    return {"intercept": intercept, "slope": slope, "status": "defined"}


calibration_intercept_and_slope = calibration_intercept_slope


def ndcg_at_k(relevances: Sequence[float], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive.")
    values = [float(value) for value in relevances]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Relevance values must be finite and non-negative.")
    if not values:
        return None
    limit = min(k, len(values))

    def dcg(items: Sequence[float]) -> float:
        return sum(
            (2.0**value - 1.0) / math.log2(index + 2.0)
            for index, value in enumerate(items[:limit])
        )

    ideal = dcg(sorted(values, reverse=True))
    return dcg(values) / ideal if ideal > 0.0 else None


def reciprocal_rank(relevances: Sequence[object]) -> float | None:
    if not relevances:
        return None
    for index, value in enumerate(relevances, start=1):
        if float(value) > 0.0:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(relevance_groups: Sequence[Sequence[object]]) -> float | None:
    if not relevance_groups:
        return None
    values = [reciprocal_rank(group) for group in relevance_groups]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None) / len(values)


def mrr(relevances: Sequence[object] | Sequence[Sequence[object]]) -> float | None:
    """Return reciprocal rank for one query or mean reciprocal rank for many."""

    values = list(relevances)
    if not values:
        return None
    if isinstance(values[0], Sequence) and not isinstance(
        values[0], (str, bytes, bytearray)
    ):
        return mean_reciprocal_rank(values)  # type: ignore[arg-type]
    return reciprocal_rank(values)


def top_k_recall(relevances: Sequence[object], k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive.")
    relevant = [float(value) > 0.0 for value in relevances]
    total = sum(relevant)
    if total == 0:
        return None
    return sum(relevant[:k]) / total


def pairwise_ranking_accuracy(
    scores: Sequence[float], relevances: Sequence[float]
) -> float | None:
    _validate_parallel_lengths(scores, relevances)
    predicted = [float(value) for value in scores]
    observed = [float(value) for value in relevances]
    if any(not math.isfinite(value) for value in predicted + observed):
        raise ValueError("Pairwise ranking inputs must be finite.")
    correct = 0.0
    pairs = 0
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            target_delta = observed[left] - observed[right]
            if target_delta == 0:
                continue
            prediction_delta = predicted[left] - predicted[right]
            pairs += 1
            if prediction_delta == 0:
                correct += 0.5
            elif prediction_delta * target_delta > 0:
                correct += 1.0
    return correct / pairs if pairs else None


def coverage(answered: Sequence[bool]) -> float | None:
    if not answered:
        return None
    return sum(bool(value) for value in answered) / len(answered)


def selective_risk(correct: Sequence[bool], answered: Sequence[bool]) -> float | None:
    _validate_parallel_lengths(correct, answered)
    selected = [
        bool(is_correct)
        for is_correct, is_answered in zip(correct, answered)
        if is_answered
    ]
    return 1.0 - sum(selected) / len(selected) if selected else None


def aurc(correct: Sequence[bool], confidence: Sequence[float]) -> float | None:
    count = _validate_parallel_lengths(correct, confidence)
    if count == 0:
        return None
    if any(not math.isfinite(float(value)) for value in confidence):
        raise ValueError("Confidence values must be finite.")
    order = sorted(range(count), key=lambda index: (-float(confidence[index]), index))
    errors = 0
    risks: list[float] = []
    for rank, index in enumerate(order, start=1):
        errors += int(not bool(correct[index]))
        risks.append(errors / rank)
    return sum(risks) / count


area_under_risk_coverage = aurc


def false_confidence_rate(
    correct: Sequence[bool],
    confidence: Sequence[float],
    *,
    threshold: float,
    answered: Sequence[bool] | None = None,
) -> float | None:
    count = _validate_parallel_lengths(correct, confidence)
    if answered is not None:
        _validate_parallel_lengths(correct, answered)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Confidence threshold must lie in [0, 1].")
    if any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in confidence
    ):
        raise ValueError("Confidence values must be finite values in [0, 1].")
    selected = [
        index
        for index in range(count)
        if float(confidence[index]) >= threshold
        and (answered is None or bool(answered[index]))
    ]
    if not selected:
        return None
    return sum(not bool(correct[index]) for index in selected) / len(selected)


def unsupported_answer_rate(
    answered: Sequence[bool], supported: Sequence[bool]
) -> float | None:
    _validate_parallel_lengths(answered, supported)
    unsupported = [index for index, value in enumerate(supported) if not value]
    if not unsupported:
        return None
    return sum(bool(answered[index]) for index in unsupported) / len(unsupported)


def abstention_accuracy(
    answered: Sequence[bool], supported: Sequence[bool]
) -> float | None:
    count = _validate_parallel_lengths(answered, supported)
    if count == 0:
        return None
    return (
        sum(
            bool(is_answered) == bool(is_supported)
            for is_answered, is_supported in zip(answered, supported)
        )
        / count
    )


def aggregate_governance_counts(
    records: Sequence[Mapping[str, object]],
) -> dict[str, int | bool]:
    totals = {field: 0 for field in GOVERNANCE_COUNT_FIELDS}
    for record in records:
        for field in GOVERNANCE_COUNT_FIELDS:
            raw = record.get(field, 0)
            if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
                raise ValueError(f"{field} must be a non-negative integer.")
            totals[field] += raw
    violation_count = sum(totals.values())
    return {
        **totals,
        "governance_violation_count": violation_count,
        "all_governance_counts_zero": violation_count == 0,
    }


def compute_direction_metrics(
    y_true: Sequence[object] | Sequence[Mapping[str, object]],
    y_pred: Sequence[object] | None = None,
    publication_groups: Sequence[object] | None = None,
    *,
    probabilities: Sequence[Mapping[object, float] | Sequence[float]] | None = None,
    labels: Sequence[object] = DIRECTION_LABELS,
) -> dict[str, object]:
    values = list(y_true)
    if (
        y_pred is None
        and publication_groups is None
        and all(isinstance(item, Mapping) for item in values)
    ):
        catalog = [dict(item) for item in values]
        y_true = [item.get("true_label") for item in catalog]
        y_pred = [item.get("predicted_label") for item in catalog]
        publication_groups = [item.get("publication_family") for item in catalog]
        if probabilities is None and all(
            isinstance(item.get("probabilities"), Mapping) for item in catalog
        ):
            probabilities = [item["probabilities"] for item in catalog]  # type: ignore[list-item]
    if y_pred is None or publication_groups is None:
        raise ValueError(
            "Direction metrics require predictions and publication groups."
        )
    metrics: dict[str, object] = {
        "publication_macro_direction_accuracy": publication_macro_direction_accuracy(
            y_true, y_pred, publication_groups
        ),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, labels=labels),
        "macro_f1": macro_f1(y_true, y_pred, labels=labels),
        "brier_score": None,
        "log_loss": None,
        "calibration_intercept": None,
        "calibration_slope": None,
    }
    if probabilities is None:
        return metrics
    rows = _probability_rows(probabilities, labels)
    _validate_parallel_lengths(y_true, rows)
    index = {label: position for position, label in enumerate(labels)}
    if any(predicted not in index for predicted in y_pred):
        raise ValueError("Predicted label is absent from probability labels.")
    confidence = [row[index[predicted]] for predicted, row in zip(y_pred, rows)]
    correctness = [
        int(observed == predicted) for observed, predicted in zip(y_true, y_pred)
    ]
    metrics["brier_score"] = multiclass_brier_score(y_true, rows, labels=labels)
    metrics["log_loss"] = multiclass_log_loss(y_true, rows, labels=labels)
    calibration = calibration_intercept_slope(correctness, confidence)
    metrics["calibration_intercept"] = calibration["intercept"]
    metrics["calibration_slope"] = calibration["slope"]
    return metrics


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def compute_ranking_metrics(
    relevances: Sequence[float] | Sequence[Mapping[str, object]],
    scores: Sequence[float] | None = None,
    *,
    k: int | None = None,
) -> dict[str, float | None]:
    values = list(relevances)
    if scores is None and all(isinstance(item, Mapping) for item in values):
        if k is None:
            raise ValueError("Ranking catalog aggregation requires a frozen k value.")
        per_episode: dict[str, list[float | None]] = defaultdict(list)
        for raw in values:
            row = dict(raw)  # type: ignore[arg-type]
            predicted = [_text(item) for item in _sequence(row.get("ranking"))]
            reference = [
                _text(item) for item in _sequence(row.get("reference_ranking"))
            ]
            if (
                not predicted
                or not reference
                or len(set(predicted)) != len(predicted)
                or len(set(reference)) != len(reference)
            ):
                raise ValueError(
                    "Ranking rows require unique predicted and reference rankings."
                )
            candidates = list(dict.fromkeys(reference + predicted))
            observed = {
                uid: float(len(reference) - index)
                for index, uid in enumerate(reference)
            }
            predicted_scores = {
                uid: float(len(predicted) - index)
                for index, uid in enumerate(predicted)
            }
            ranked_gains = [observed.get(uid, 0.0) for uid in predicted]
            top_reference = set(reference[:k])
            per_episode["ndcg_at_k"].append(ndcg_at_k(ranked_gains, k))
            per_episode["mrr"].append(
                reciprocal_rank([uid == reference[0] for uid in predicted])
            )
            per_episode["top_k_recall"].append(
                len(set(predicted[:k]) & top_reference) / len(top_reference)
                if top_reference
                else None
            )
            per_episode["pairwise_ranking_accuracy"].append(
                pairwise_ranking_accuracy(
                    [predicted_scores.get(uid, 0.0) for uid in candidates],
                    [observed.get(uid, 0.0) for uid in candidates],
                )
            )
        return {name: _mean_defined(items) for name, items in per_episode.items()}
    if scores is None or k is None:
        raise ValueError("Ranking metrics require scores and a frozen k value.")
    _validate_parallel_lengths(relevances, scores)
    ranked_indices = sorted(
        range(len(scores)), key=lambda index: (-float(scores[index]), index)
    )
    ranked_relevances = [relevances[index] for index in ranked_indices]
    return {
        "ndcg_at_k": ndcg_at_k(ranked_relevances, k),
        "mrr": reciprocal_rank(ranked_relevances),
        "top_k_recall": top_k_recall(ranked_relevances, k),
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(scores, relevances),
    }


def compute_reliability_metrics(
    correct: Sequence[bool] | Sequence[Mapping[str, object]],
    answered: Sequence[bool] | None = None,
    supported: Sequence[bool] | None = None,
    confidence: Sequence[float] | None = None,
    *,
    false_confidence_threshold: float | None = None,
) -> dict[str, float | None]:
    values = list(correct)
    if (
        answered is None
        and supported is None
        and confidence is None
        and all(isinstance(item, Mapping) for item in values)
    ):
        catalog = [dict(item) for item in values]
        correct = [bool(item.get("correct")) for item in catalog]
        answered = [item.get("abstention_status") != "abstained" for item in catalog]
        supported = [
            item.get("true_label") != "insufficient_information" for item in catalog
        ]
        confidence = [float(item.get("confidence")) for item in catalog]
    if (
        answered is None
        or supported is None
        or confidence is None
        or false_confidence_threshold is None
    ):
        raise ValueError(
            "Reliability metrics require answered, supported, confidence, and a frozen threshold."
        )
    _validate_parallel_lengths(correct, answered, supported, confidence)
    return {
        "coverage": coverage(answered),
        "selective_risk": selective_risk(correct, answered),
        "aurc": aurc(correct, confidence),
        "false_confidence_rate": false_confidence_rate(
            correct,
            confidence,
            threshold=false_confidence_threshold,
            answered=answered,
        ),
        "unsupported_answer_rate": unsupported_answer_rate(answered, supported),
        "abstention_accuracy": abstention_accuracy(answered, supported),
    }


brier_score = multiclass_brier_score
calibration_slope_intercept = calibration_intercept_slope
aggregate_governance_violations = aggregate_governance_counts
validate_agent_twin_execution_isolation = validate_execution_isolation
validate_duplicate_uids = validate_unique_uids
validate_frozen_gap_map = validate_frozen_gap_isolation
validate_memory_leakage_risk = validate_memory_risk_disclosure
validate_outcome_target_contract = validate_outcome_target
validate_pre_cutoff_held_out_isolation = validate_bundle_isolation
validate_public_source_admission = validate_public_source


__all__ = [
    "ALLOWED_TIMESTAMP_TYPES",
    "BENCHMARK_TRACKS",
    "CLAIM_SCOPE",
    "DIRECTION_LABELS",
    "EPISODE_STATUSES",
    "FROZEN_GAP_IMPACTS",
    "GOVERNANCE_COUNT_FIELDS",
    "INDEPENDENCE_GROUP_FIELDS",
    "LABEL_CLASSES",
    "LiteratureReplayEpisode",
    "MEMORY_RISK_TRACKS",
    "MEMORY_TRACK_POST_MODEL_CUTOFF",
    "MEMORY_TRACK_RETRIEVAL_ISOLATED",
    "MEMORY_TRACK_STRESS",
    "PARTITIONS",
    "PUBLIC_SOURCE_CLASSES",
    "SCHEMA_VERSIONS",
    "SEQUENTIAL_REVEAL_STEPS",
    "TARGET_TYPES",
    "TEMPORAL_POLICY_VERSION",
    "TRACKS_BY_CODE",
    "TRACK_A",
    "TRACK_A_ENDPOINT_DIRECTION",
    "TRACK_B",
    "TRACK_B_INTERVENTION_RANKING",
    "TRACK_C",
    "TRACK_C_MECHANISM_DISCRIMINATION",
    "TRACK_D",
    "TRACK_D_INFORMATION_ACTION",
    "TRACK_E",
    "TRACK_E_SEQUENTIAL_REPLAY",
    "TRACK_F",
    "TRACK_F_ABSTENTION_BOUNDARY",
    "abstention_accuracy",
    "aggregate_governance_counts",
    "aggregate_governance_violations",
    "area_under_risk_coverage",
    "aurc",
    "balanced_accuracy",
    "blocker",
    "brier_score",
    "calibration_intercept_and_slope",
    "calibration_intercept_slope",
    "calibration_slope_intercept",
    "canonical_episode_uid",
    "canonical_sequential_event_uid",
    "compute_direction_metrics",
    "compute_ranking_metrics",
    "compute_reliability_metrics",
    "canonical_sha256",
    "coverage",
    "direction_accuracy",
    "episode_content_hash",
    "episode_content_payload",
    "false_confidence_rate",
    "log_loss",
    "macro_f1",
    "mean_reciprocal_rank",
    "mrr",
    "multiclass_brier_score",
    "multiclass_log_loss",
    "ndcg_at_k",
    "pairwise_ranking_accuracy",
    "publication_macro_direction_accuracy",
    "reciprocal_rank",
    "select_earliest_public_timestamp",
    "selective_risk",
    "sequential_event_content_hash",
    "sequential_event_content_payload",
    "top_k_recall",
    "unsupported_answer_rate",
    "validate_abstention_semantics",
    "validate_agent_twin_execution_isolation",
    "validate_attempt_preservation",
    "validate_bundle_isolation",
    "validate_claim_boundaries",
    "validate_claim_language",
    "validate_episode",
    "validate_episode_content_address",
    "validate_execution_isolation",
    "validate_duplicate_uids",
    "validate_failed_retry_preservation",
    "validate_frozen_gap_isolation",
    "validate_frozen_gap_map",
    "validate_independence_groups",
    "validate_label_provenance",
    "validate_lineage_acyclic",
    "validate_literature_replay_episode",
    "validate_memory_risk_disclosure",
    "validate_memory_leakage_risk",
    "validate_outcome_target",
    "validate_outcome_target_contract",
    "validate_pre_cutoff_held_out_isolation",
    "validate_public_source",
    "validate_public_source_admission",
    "validate_public_source_boundary",
    "validate_sequential_reveal",
    "validate_sequential_reveal_integrity",
    "validate_scitil_runtime_trace",
    "validate_split_leakage",
    "validate_split_manifest",
    "validate_temporal_cutoff",
    "validate_temporal_ordering",
    "validate_unique_uids",
    "validate_unknown_uncertainty",
    "sorted_blockers",
    "stable_uid",
]
