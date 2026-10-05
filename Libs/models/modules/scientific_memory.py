"""Provenance-bound, structured memory for scientific agent runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Literal, Mapping

MemoryLayer = Literal["working", "episodic", "semantic", "procedural"]
MEMORY_SCHEMA_VERSION = "neuronsifter-scientific-memory-v1"
_LAYERS = frozenset(("working", "episodic", "semantic", "procedural"))
_FORBIDDEN_KEYS = frozenset(
    (
        "chain_of_thought",
        "hidden_reasoning",
        "raw_reasoning",
        "thought",
        "thoughts",
        "scratchpad",
        "private_reasoning",
    )
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _uid(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _validate_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise TypeError("memory summaries must be structured mappings")
    def walk(value: Any, path: str = "summary") -> None:
        if isinstance(value, Mapping):
            forbidden = {str(key).lower() for key in value} & _FORBIDDEN_KEYS
            if forbidden:
                raise ValueError(
                    f"raw hidden reasoning fields are not allowed at {path}: {sorted(forbidden)}"
                )
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(summary)
    result = dict(summary)
    return result


@dataclass(frozen=True)
class MemoryEntry:
    """Immutable structured memory record bound to scientific provenance."""

    memory_type: MemoryLayer
    structured_summary: Mapping[str, Any]
    source_uids: tuple[str, ...] = ()
    posterior_uid: str | None = None
    world_action_uid: str | None = None
    tool_call_uids: tuple[str, ...] = ()
    confidence: float | None = None
    evidence_grade: str | None = None
    supersedes: tuple[str, ...] = ()
    created_at: str = ""
    invalidated_by: str | None = None
    memory_uid: str = field(init=False)

    def __post_init__(self) -> None:
        if self.memory_type not in _LAYERS:
            raise ValueError(f"unsupported memory layer: {self.memory_type}")
        summary = _validate_summary(self.structured_summary)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "structured_summary", summary)
        object.__setattr__(self, "source_uids", tuple(self.source_uids))
        object.__setattr__(self, "tool_call_uids", tuple(self.tool_call_uids))
        object.__setattr__(self, "supersedes", tuple(self.supersedes))
        created_at = self.created_at or datetime.now(timezone.utc).isoformat()
        object.__setattr__(self, "created_at", str(created_at))
        if not self.created_at.strip():
            raise ValueError("created_at must be a non-empty timestamp")
        object.__setattr__(
            self,
            "memory_uid",
            _uid(
                "memory",
                {
                    "memory_type": self.memory_type,
                    "structured_summary": summary,
                    "source_uids": self.source_uids,
                    "posterior_uid": self.posterior_uid,
                    "world_action_uid": self.world_action_uid,
                    "tool_call_uids": self.tool_call_uids,
                    "confidence": self.confidence,
                    "evidence_grade": self.evidence_grade,
                    "supersedes": self.supersedes,
                    # Timestamps and invalidation metadata are provenance
                    # annotations, not scientific content identity.
                },
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "memory_uid": self.memory_uid,
            "memory_type": self.memory_type,
            "structured_summary": dict(self.structured_summary),
            "source_uids": list(self.source_uids),
            "posterior_uid": self.posterior_uid,
            "world_action_uid": self.world_action_uid,
            "tool_call_uids": list(self.tool_call_uids),
            "confidence": self.confidence,
            "evidence_grade": self.evidence_grade,
            "supersedes": list(self.supersedes),
            "created_at": self.created_at,
            "invalidated_by": self.invalidated_by,
        }


class ScientificMemory:
    """Small deterministic store implementing the four agent-memory layers."""

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._links: dict[str, set[str]] = {}
        self._invalidated: set[str] = set()
        self._invalidation_reasons: dict[str, str] = {}
        self._posterior_validator: Callable[[str], Any] | None = None

    @property
    def entries(self) -> Mapping[str, MemoryEntry]:
        return dict(self._entries)

    def to_record(self) -> dict[str, Any]:
        """Serialize only structured, provenance-bound memory state."""
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "entries": [
                self._entries[memory_uid].to_record()
                for memory_uid in sorted(self._entries)
            ],
            "links": {
                memory_uid: sorted(related_uids)
                for memory_uid, related_uids in sorted(self._links.items())
            },
            "invalidated": sorted(self._invalidated),
            "invalidation_reasons": dict(sorted(self._invalidation_reasons.items())),
        }

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        posterior_validator: Callable[[str], Any] | None = None,
    ) -> "ScientificMemory":
        """Restore a memory snapshot while checking every immutable identity."""
        if not isinstance(record, Mapping) or record.get("schema_version") != MEMORY_SCHEMA_VERSION:
            raise ValueError("Unsupported scientific memory snapshot schema")
        entries = record.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Scientific memory snapshot entries must be a list")
        memory = cls()
        if posterior_validator is not None:
            memory.bind_posterior_validator(posterior_validator)
        for row in entries:
            if not isinstance(row, Mapping):
                raise ValueError("Scientific memory snapshot entry must be a mapping")
            required = {"memory_uid", "memory_type", "structured_summary"}
            missing = sorted(required.difference(row))
            if missing:
                raise ValueError(f"Scientific memory snapshot entry is missing fields: {missing}")
            entry = MemoryEntry(
                memory_type=row["memory_type"],
                structured_summary=row["structured_summary"],
                source_uids=tuple(row.get("source_uids", ())),
                posterior_uid=row.get("posterior_uid"),
                world_action_uid=row.get("world_action_uid"),
                tool_call_uids=tuple(row.get("tool_call_uids", ())),
                confidence=row.get("confidence"),
                evidence_grade=row.get("evidence_grade"),
                supersedes=tuple(row.get("supersedes", ())),
                created_at=str(row.get("created_at", "")),
                invalidated_by=row.get("invalidated_by"),
            )
            if str(row["memory_uid"]) != entry.memory_uid:
                raise ValueError(f"Scientific memory UID does not match its content: {row['memory_uid']}")
            memory.write(entry)
        links = record.get("links", {})
        if not isinstance(links, Mapping):
            raise ValueError("Scientific memory snapshot links must be a mapping")
        for memory_uid, related_uids in links.items():
            if not isinstance(related_uids, (list, tuple)):
                raise ValueError("Scientific memory links must contain UID lists")
            memory._require(str(memory_uid))
            for related_uid in related_uids:
                memory.link(str(memory_uid), str(related_uid))
        invalidated = record.get("invalidated", [])
        reasons = record.get("invalidation_reasons", {})
        if not isinstance(invalidated, list) or not isinstance(reasons, Mapping):
            raise ValueError("Scientific memory invalidation metadata is malformed")
        for memory_uid in invalidated:
            uid = str(memory_uid)
            reason = reasons.get(uid)
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"Missing invalidation reason for memory UID: {uid}")
            memory.invalidate(uid, reason=reason)
        return memory

    def write(self, entry: MemoryEntry) -> str:
        if self._posterior_validator is not None and entry.posterior_uid is not None:
            try:
                self._posterior_validator(str(entry.posterior_uid))
            except Exception as exc:
                raise ValueError(
                    f"memory entry references an unknown posterior: {entry.posterior_uid}"
                ) from exc
        self._entries.setdefault(entry.memory_uid, entry)
        return entry.memory_uid

    def bind_posterior_validator(self, validator: Callable[[str], Any]) -> None:
        if not callable(validator):
            raise TypeError("posterior validator must be callable")
        self._posterior_validator = validator

    def write_episode(self, summary: Mapping[str, Any], **provenance: Any) -> str:
        return self.write(MemoryEntry(memory_type="episodic", structured_summary=summary, **provenance))

    def write_semantic(self, summary: Mapping[str, Any], **provenance: Any) -> str:
        return self.write(MemoryEntry(memory_type="semantic", structured_summary=summary, **provenance))

    def write_procedural(self, summary: Mapping[str, Any], **provenance: Any) -> str:
        return self.write(MemoryEntry(memory_type="procedural", structured_summary=summary, **provenance))

    def write_working(self, summary: Mapping[str, Any], **provenance: Any) -> str:
        return self.write(MemoryEntry(memory_type="working", structured_summary=summary, **provenance))

    def get_history(self, *, posterior_uid: str | None = None, limit: int | None = None) -> list[MemoryEntry]:
        return self.retrieve(layer="episodic", posterior_uid=posterior_uid, limit=limit)

    def retrieve(
        self,
        *,
        layer: MemoryLayer | None = None,
        query: Mapping[str, Any] | None = None,
        posterior_uid: str | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        if layer is not None and layer not in _LAYERS:
            raise ValueError(f"unsupported memory layer: {layer}")
        query = dict(query or {})
        found = []
        for entry in self._entries.values():
            if entry.memory_uid in self._invalidated or (layer and entry.memory_type != layer):
                continue
            if posterior_uid is not None and entry.posterior_uid != posterior_uid:
                continue
            if query and any(entry.structured_summary.get(k) != v for k, v in query.items()):
                continue
            found.append(entry)
        found.sort(key=lambda item: item.memory_uid)
        return found if limit is None else found[: max(0, limit)]

    def link(self, memory_uid: str, related_uid: str) -> None:
        self._require(memory_uid)
        self._require(related_uid)
        self._links.setdefault(memory_uid, set()).add(related_uid)

    def links(self, memory_uid: str) -> tuple[str, ...]:
        self._require(memory_uid)
        return tuple(sorted(self._links.get(memory_uid, ())))

    def supersede(self, old_uid: str, replacement: MemoryEntry) -> str:
        self._require(old_uid)
        if old_uid not in replacement.supersedes:
            raise ValueError("replacement must explicitly declare supersedes")
        return self.write(replacement)

    def invalidate(self, memory_uid: str, *, reason: str) -> None:
        self._require(memory_uid)
        if not reason.strip():
            raise ValueError("invalidation requires a reason")
        self._invalidated.add(memory_uid)
        self._invalidation_reasons[memory_uid] = reason

    def is_invalidated(self, memory_uid: str) -> bool:
        self._require(memory_uid)
        return memory_uid in self._invalidated

    def invalidation_reason(self, memory_uid: str) -> str | None:
        self._require(memory_uid)
        return self._invalidation_reasons.get(memory_uid)

    def compress(self, memory_uids: Iterable[str], *, layer: MemoryLayer, summary: Mapping[str, Any]) -> str:
        ids = tuple(sorted(set(memory_uids)))
        for memory_uid in ids:
            self._require(memory_uid)
        return self.write(MemoryEntry(memory_type=layer, structured_summary=summary, supersedes=ids))

    def _require(self, memory_uid: str) -> MemoryEntry:
        try:
            return self._entries[memory_uid]
        except KeyError as exc:
            raise KeyError(f"unknown memory UID: {memory_uid}") from exc
