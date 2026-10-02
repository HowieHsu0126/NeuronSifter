"""Decision-consistent three-kernel runtime for NeuronSifter vNext.

This module is an auditable interface layer around the existing NeuronSifter
state encoder, world-action operator, tool adapters, and TRI components.  It
does not replace those model internals.  The layer owns posterior identity,
typed world/information actions, assimilation lineage, and decision tracing so
that a hidden scientific state cannot silently diverge from the state used by
rollout or reliability decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - the project runtime requires torch.
    torch = None  # type: ignore[assignment]


POSTERIOR_SCHEMA_VERSION = "neuronsifter-posterior-v1"
WORLD_ACTION_SCHEMA_VERSION = "neuronsifter-world-action-v1"
INFORMATION_ACTION_SCHEMA_VERSION = "neuronsifter-information-action-v1"
DECISION_TRACE_SCHEMA_VERSION = "neuronsifter-decision-trace-v1"
_INVARIANT_FIELDS = (
    "posterior_uid_rollout",
    "posterior_uid_query",
    "posterior_uid_assimilation_parent",
    "posterior_uid_decision",
    "posterior_uid_for_intervention_ranking",
)


def _is_finite(value: Any) -> bool:
    if torch is not None and isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all().item())
    if isinstance(value, Mapping):
        return all(_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _canonical(value: Any) -> Any:
    """Convert supported values to deterministic, JSON-safe identity input."""
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "values": tensor.tolist(),
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Scientific posterior identity cannot contain NaN or Inf.")
        return value
    return repr(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze(value: Any) -> Any:
    """Freeze mappings/sequences while retaining tensor values for model use."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    if torch is not None and isinstance(value, torch.Tensor):
        return _canonical(value)
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


@dataclass(frozen=True)
class ScientificPosterior:
    """Immutable registered scientific belief state."""

    posterior_uid: str
    version: int
    state: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    support: Mapping[str, Any]
    mechanism: Mapping[str, Any]
    reliability: Mapping[str, Any]
    parent_posterior_uid: str | None = None
    schema_version: str = POSTERIOR_SCHEMA_VERSION
    content_digest: str = ""
    uid_content_digest: str = ""

    def __post_init__(self) -> None:
        if not self.posterior_uid or not self.posterior_uid.startswith("posterior-"):
            raise ValueError("ScientificPosterior.posterior_uid must be a versioned posterior identity.")
        if int(self.version) < 1:
            raise ValueError("ScientificPosterior.version must be positive.")
        if self.schema_version != POSTERIOR_SCHEMA_VERSION:
            raise ValueError(f"Unsupported posterior schema: {self.schema_version}")
        for component_name in ("state", "uncertainty", "support", "mechanism", "reliability"):
            component = getattr(self, component_name)
            if not isinstance(component, Mapping):
                raise TypeError(f"Posterior component {component_name!r} must be a mapping.")
            if not _is_finite(component):
                raise ValueError(f"Posterior component {component_name!r} contains NaN or Inf.")
        expected_content_digest = _digest(
            {
                "schema_version": self.schema_version,
                "posterior_uid": self.posterior_uid,
                "version": int(self.version),
                "parent_posterior_uid": self.parent_posterior_uid,
                "state": self.state,
                "uncertainty": self.uncertainty,
                "support": self.support,
                "mechanism": self.mechanism,
                "reliability": self.reliability,
            }
        )
        if self.content_digest and self.content_digest != expected_content_digest:
            raise ValueError("ScientificPosterior.content_digest does not match serialized content")
        object.__setattr__(self, "content_digest", expected_content_digest)
        expected_uid_digest: str | None = None
        components = {
            "state": self.state,
            "uncertainty": self.uncertainty,
            "support": self.support,
            "mechanism": self.mechanism,
            "reliability": self.reliability,
        }
        if self.parent_posterior_uid is not None:
            expected_uid_digest = _digest(
                {
                    "schema_version": self.schema_version,
                    "version": int(self.version),
                    "parent_posterior_uid": self.parent_posterior_uid,
                    "state": self.state,
                    "uncertainty": self.uncertainty,
                }
            )
        elif isinstance(self.mechanism.get("graph_digest"), str):
            expected_uid_digest = _digest(
                {
                    "schema_version": self.schema_version,
                    "version": int(self.version),
                    "components": components,
                    "graph_digest": self.mechanism["graph_digest"],
                }
            )
        if self.uid_content_digest:
            if expected_uid_digest is not None and self.uid_content_digest != expected_uid_digest:
                raise ValueError("ScientificPosterior.uid_content_digest does not match serialized content")
            if self.posterior_uid != "posterior-" + self.uid_content_digest:
                raise ValueError("ScientificPosterior.posterior_uid does not match uid_content_digest")
        object.__setattr__(self, "uid_content_digest", self.uid_content_digest)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "posterior_uid": self.posterior_uid,
            "version": int(self.version),
            "parent_posterior_uid": self.parent_posterior_uid,
            "state": _thaw(self.state),
            "uncertainty": _thaw(self.uncertainty),
            "support": _thaw(self.support),
            "mechanism": _thaw(self.mechanism),
            "reliability": _thaw(self.reliability),
            "content_digest": self.content_digest,
            "uid_content_digest": self.uid_content_digest,
        }


@dataclass(frozen=True)
class WorldAction:
    """Typed physical intervention action.

    This object is deliberately separate from :class:`InformationAction`.
    """

    target_set: tuple[str, ...]
    drug_set: tuple[str, ...]
    dose: Any
    schedule: str
    route: str
    action_uid: str | None = None
    schema_version: str = WORLD_ACTION_SCHEMA_VERSION
    # Semantic DSL fields are optional for legacy catalog records.  When
    # present they are included in the content identity and serialized
    # alongside the canonical target/drug/dose fields.
    modality: str = "pharmacological"
    target: tuple[str, ...] = ()
    agent: tuple[str, ...] = ()
    dose_or_intensity: Any = None
    duration: Any | None = None
    combination: tuple[str, ...] = ()
    biological_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != WORLD_ACTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported world-action schema: {self.schema_version}")
        if not self.target_set or not self.drug_set:
            raise ValueError("WorldAction requires non-empty target_set and drug_set.")
        if not self.schedule or not self.route:
            raise ValueError("WorldAction requires schedule and route semantics.")
        if not _is_finite(self.dose):
            raise ValueError("WorldAction dose must be finite.")
        if not self.modality or not isinstance(self.modality, str):
            raise ValueError("WorldAction modality must be a non-empty string.")
        if self.target and tuple(self.target) != tuple(self.target_set):
            raise ValueError("WorldAction target and target_set disagree.")
        if self.agent and tuple(self.agent) != tuple(self.drug_set):
            raise ValueError("WorldAction agent and drug_set disagree.")
        object.__setattr__(self, "target", tuple(self.target) or tuple(self.target_set))
        object.__setattr__(self, "agent", tuple(self.agent) or tuple(self.drug_set))
        object.__setattr__(self, "dose_or_intensity", self.dose if self.dose_or_intensity is None else self.dose_or_intensity)
        object.__setattr__(self, "combination", tuple(self.combination) or tuple(self.drug_set))
        if not _is_finite(self.dose_or_intensity) or not _is_finite(self.duration) or not _is_finite(self.biological_context):
            raise ValueError("WorldAction semantic fields must be finite.")
        identity = {
            "target_set": self.target_set,
            "drug_set": self.drug_set,
            "dose": self.dose,
            "schedule": self.schedule,
            "route": self.route,
        }
        # Keep legacy UIDs stable when callers use only the original v1
        # fields; richer actions receive a content identity over the DSL.
        if any(
            (
                self.modality != "pharmacological",
                self.duration is not None,
                bool(self.biological_context),
                self.combination != tuple(self.drug_set),
                _canonical(self.dose_or_intensity) != _canonical(self.dose),
            )
        ):
            identity["dsl"] = {
                "modality": self.modality,
                "target": self.target,
                "agent": self.agent,
                "dose_or_intensity": self.dose_or_intensity,
                "duration": self.duration,
                "combination": self.combination,
                "biological_context": self.biological_context,
            }
        computed_uid = "world-action-" + _digest(
            identity
        )
        if self.action_uid is None:
            object.__setattr__(self, "action_uid", computed_uid)
        elif self.action_uid != computed_uid:
            raise ValueError("WorldAction.action_uid does not match action content.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WorldAction":
        payload = _require_mapping(payload, "world_action")
        aliases = {
            "target_set": payload.get("target", payload.get("target_set")),
            "drug_set": payload.get("agent", payload.get("drug_set", payload.get("combination"))),
            "dose": payload.get("dose_or_intensity", payload.get("dose")),
            "schedule": payload.get("schedule"),
            "route": payload.get("route"),
        }
        missing = [key for key, value in aliases.items() if value is None]
        if missing:
            raise ValueError(f"WorldAction is missing fields: {missing}")
        return cls(
            target_set=tuple(str(value) for value in aliases["target_set"]),
            drug_set=tuple(str(value) for value in aliases["drug_set"]),
            dose=aliases["dose"],
            schedule=str(aliases["schedule"]),
            route=str(aliases["route"]),
            action_uid=str(payload["action_uid"]) if payload.get("action_uid") else None,
            schema_version=str(
                payload.get("schema_version", WORLD_ACTION_SCHEMA_VERSION)
            ),
            modality=str(payload.get("modality", "pharmacological")),
            target=tuple(str(value) for value in payload.get("target", ())),
            agent=tuple(str(value) for value in payload.get("agent", ())),
            dose_or_intensity=payload.get("dose_or_intensity"),
            duration=payload.get("duration"),
            combination=tuple(str(value) for value in payload.get("combination", ())),
            biological_context=dict(payload.get("biological_context", {})),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_type": "world",
            "action_uid": self.action_uid,
            "target_set": list(self.target_set),
            "drug_set": list(self.drug_set),
            "dose": _canonical(self.dose),
            "schedule": self.schedule,
            "route": self.route,
            "modality": self.modality,
            "target": list(self.target),
            "agent": list(self.agent),
            "dose_or_intensity": _canonical(self.dose_or_intensity),
            "duration": _canonical(self.duration),
            "combination": list(self.combination),
            "biological_context": _canonical(self.biological_context),
        }


@dataclass(frozen=True)
class InformationAction:
    """Typed observation/tool action with no biological transition semantics."""

    tool_id: str
    query: Mapping[str, Any]
    fidelity: str = "diagnostic"
    expected_cost: float = 0.0
    action_uid: str | None = None
    schema_version: str = INFORMATION_ACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INFORMATION_ACTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported information-action schema: {self.schema_version}")
        if not self.tool_id:
            raise ValueError("InformationAction.tool_id is required.")
        if not isinstance(self.query, Mapping):
            raise TypeError("InformationAction.query must be a mapping.")
        forbidden = {"dose", "drug_set", "schedule", "route", "occupancy_program"}
        hidden_physical = sorted(forbidden.intersection(str(key) for key in self.query))
        if hidden_physical:
            raise ValueError(f"InformationAction cannot hide physical intervention fields: {hidden_physical}")
        if not math.isfinite(float(self.expected_cost)) or self.expected_cost < 0:
            raise ValueError("InformationAction.expected_cost must be finite and non-negative.")
        computed_uid = "information-action-" + _digest(
            {"tool_id": self.tool_id, "query": self.query, "fidelity": self.fidelity, "expected_cost": self.expected_cost}
        )
        if self.action_uid is None:
            object.__setattr__(self, "action_uid", computed_uid)
        elif self.action_uid != computed_uid:
            raise ValueError("InformationAction.action_uid does not match action content.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "InformationAction":
        payload = _require_mapping(payload, "information_action")
        required = ("tool_id", "query")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"InformationAction is missing fields: {missing}")
        return cls(
            tool_id=str(payload["tool_id"]),
            query=dict(payload["query"]),
            fidelity=str(payload.get("fidelity", "diagnostic")),
            expected_cost=float(payload.get("expected_cost", 0.0)),
            action_uid=str(payload["action_uid"]) if payload.get("action_uid") else None,
            schema_version=str(
                payload.get("schema_version", INFORMATION_ACTION_SCHEMA_VERSION)
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_type": "information",
            "action_uid": self.action_uid,
            "tool_id": self.tool_id,
            "query": _thaw(self.query),
            "fidelity": self.fidelity,
            "expected_cost": self.expected_cost,
        }


@dataclass(frozen=True)
class ToolResult:
    posterior_uid: str
    action_uid: str
    status: str
    observation: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    partial: bool = False
    observation_uid: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "partial", "failed", "invalid"}:
            raise ValueError(f"Unknown tool result status: {self.status}")
        if self.status in {"completed", "partial"} and not _is_finite(self.observation):
            raise ValueError("Completed tool results must contain finite observations.")
        if not _is_finite(self.uncertainty):
            raise ValueError("Tool result uncertainty must be finite.")
        if self.observation_uid is not None and not str(self.observation_uid):
            raise ValueError("observation_uid must be non-empty when provided")

    @property
    def valid_for_assimilation(self) -> bool:
        return self.status in {"completed", "partial"} and bool(self.observation)

    def to_record(self) -> dict[str, Any]:
        return {
            "posterior_uid": self.posterior_uid,
            "action_uid": self.action_uid,
            "status": self.status,
            "observation": _thaw(self.observation),
            "uncertainty": _thaw(self.uncertainty),
            "error": self.error,
            "provenance": _thaw(self.provenance),
            "partial": bool(self.partial),
            "observation_uid": self.observation_uid,
        }


@dataclass(frozen=True)
class TrajectoryResult:
    posterior_uid: str
    action_uid: str
    trajectory: Any
    uncertainty: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_finite(self.trajectory) or not _is_finite(self.uncertainty):
            raise ValueError("Trajectory and uncertainty must be finite.")

    def to_record(self) -> dict[str, Any]:
        return {
            "posterior_uid": self.posterior_uid,
            "action_uid": self.action_uid,
            "trajectory": _canonical(self.trajectory),
            "uncertainty": _canonical(self.uncertainty),
            "diagnostics": _thaw(self.diagnostics),
        }


@dataclass(frozen=True)
class AssimilationOutcome:
    accepted: bool
    posterior: ScientificPosterior
    posterior_uid_assimilation_parent: str
    reason: str
    tool_result_status: str


class ScientificBeliefKernel:
    """Registry-backed posterior inference and immutable child creation."""

    def __init__(self) -> None:
        self._posteriors: dict[str, ScientificPosterior] = {}

    @property
    def posteriors(self) -> Mapping[str, ScientificPosterior]:
        return MappingProxyType(self._posteriors)

    def register_posterior(self, posterior: ScientificPosterior) -> ScientificPosterior:
        """Register an immutable posterior for an explicit restore/assimilation path.

        Runtime adapters must use this boundary instead of mutating the
        registry directly.  Re-registering the same UID is idempotent only
        when its serialized scientific content is identical.
        """
        if not isinstance(posterior, ScientificPosterior):
            raise TypeError("register_posterior requires a ScientificPosterior")
        existing = self._posteriors.get(posterior.posterior_uid)
        if existing is not None and existing.to_record() != posterior.to_record():
            raise ValueError("Cannot replace registered posterior content for an existing UID")
        self._posteriors[posterior.posterior_uid] = posterior
        return posterior

    def _build_components(self, evidence: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        state = dict(evidence.get("state", {})) if isinstance(evidence.get("state"), Mapping) else {"evidence_digest": _digest(evidence)}
        uncertainty = dict(evidence.get("uncertainty", {})) if isinstance(evidence.get("uncertainty"), Mapping) else {}
        uncertainty.setdefault("epistemic", 0.0)
        uncertainty.setdefault("aleatoric", 0.0)
        support = dict(evidence.get("support", {})) if isinstance(evidence.get("support"), Mapping) else {}
        support.setdefault("evidence_keys", sorted(str(key) for key in evidence))
        support.setdefault("graph_keys", sorted(str(key) for key in graph))
        support.setdefault("support_score", 1.0 if evidence else 0.0)
        mechanism = dict(evidence.get("mechanism", {})) if isinstance(evidence.get("mechanism"), Mapping) else {}
        mechanism.setdefault("posterior_family", "factorized_scientific_state")
        mechanism.setdefault("graph_digest", _digest(graph))
        reliability = dict(evidence.get("reliability", {})) if isinstance(evidence.get("reliability"), Mapping) else {}
        reliability.setdefault("tri_score", 1.0 if evidence else 0.0)
        reliability.setdefault("status", "evidence_bound" if evidence else "insufficient_evidence")
        return {"state": state, "uncertainty": uncertainty, "support": support, "mechanism": mechanism, "reliability": reliability}

    def infer_belief(self, evidence: Mapping[str, Any], graph: Mapping[str, Any]) -> ScientificPosterior:
        evidence = _require_mapping(evidence, "evidence")
        graph = _require_mapping(graph, "graph")
        if not _is_finite(evidence) or not _is_finite(graph):
            raise ValueError("infer_belief rejects NaN/Inf evidence or graph values.")
        components = self._build_components(evidence, graph)
        identity_payload = {
            "schema_version": POSTERIOR_SCHEMA_VERSION,
            "version": 1,
            "components": components,
            "graph_digest": _digest(graph),
        }
        posterior_uid = "posterior-" + _digest(identity_payload)
        posterior = ScientificPosterior(
            posterior_uid=posterior_uid,
            version=1,
            state=_freeze(components["state"]),
            uncertainty=_freeze(components["uncertainty"]),
            support=_freeze(components["support"]),
            mechanism=_freeze(components["mechanism"]),
            reliability=_freeze(components["reliability"]),
            uid_content_digest=posterior_uid.removeprefix("posterior-"),
        )
        return self.register_posterior(posterior)

    def child_from_observation(self, parent: ScientificPosterior, observation: Mapping[str, Any], uncertainty: Mapping[str, Any]) -> ScientificPosterior:
        if parent.posterior_uid not in self._posteriors:
            raise KeyError(f"Unknown parent posterior: {parent.posterior_uid}")
        if not _is_finite(observation) or not _is_finite(uncertainty):
            raise ValueError("Cannot create posterior from non-finite observation.")
        state = _thaw(parent.state)
        observations = list(state.get("assimilated_observations", [])) if isinstance(state, Mapping) else []
        observations.append(_canonical(observation))
        state = dict(state) if isinstance(state, Mapping) else {"parent_state": state}
        state["assimilated_observations"] = observations
        merged_uncertainty = dict(_thaw(parent.uncertainty))
        merged_uncertainty.update(_thaw(uncertainty))
        version = parent.version + 1
        identity_payload = {
            "schema_version": POSTERIOR_SCHEMA_VERSION,
            "version": version,
            "parent_posterior_uid": parent.posterior_uid,
            "state": state,
            "uncertainty": merged_uncertainty,
        }
        child = ScientificPosterior(
            posterior_uid="posterior-" + _digest(identity_payload),
            version=version,
            parent_posterior_uid=parent.posterior_uid,
            state=_freeze(state),
            uncertainty=_freeze(merged_uncertainty),
            support=_freeze({**_thaw(parent.support), "last_assimilation": "accepted"}),
            mechanism=_freeze(_thaw(parent.mechanism)),
            reliability=_freeze(_thaw(parent.reliability)),
            uid_content_digest=_digest(identity_payload),
        )
        return self.register_posterior(child)

    def get(self, posterior_uid: str) -> ScientificPosterior:
        try:
            return self._posteriors[str(posterior_uid)]
        except KeyError as exc:
            raise KeyError(f"Unknown posterior_uid: {posterior_uid}") from exc


class WorldChannel:
    def __init__(self, belief: ScientificBeliefKernel, rollout_fn: Callable[[ScientificPosterior, WorldAction], Any] | None = None) -> None:
        self.belief = belief
        self.rollout_fn = rollout_fn
        self.last_posterior_uid: str | None = None
        self._posterior_validator: Callable[[str], ScientificPosterior] | None = None
        self._trace_recorder: Callable[..., dict[str, Any]] | None = None

    def _bind_controller(
        self,
        posterior_validator: Callable[[str], ScientificPosterior],
        trace_recorder: Callable[..., dict[str, Any]],
    ) -> None:
        if not callable(posterior_validator) or not callable(trace_recorder):
            raise TypeError("WorldChannel controller hooks must be callable.")
        self._posterior_validator = posterior_validator
        self._trace_recorder = trace_recorder

    def rollout(self, posterior_uid: str, world_action: WorldAction | Mapping[str, Any]) -> TrajectoryResult:
        posterior = (
            self._posterior_validator(str(posterior_uid))
            if self._posterior_validator is not None
            else self.belief.get(posterior_uid)
        )
        action = world_action if isinstance(world_action, WorldAction) else WorldAction.from_mapping(world_action)
        if not isinstance(action, WorldAction):  # defensive for callers with custom subclasses
            raise TypeError("rollout requires a WorldAction, not an information action.")
        raw = self.rollout_fn(posterior, action) if self.rollout_fn is not None else None
        if isinstance(raw, TrajectoryResult):
            result = raw
            if result.posterior_uid != posterior.posterior_uid or result.action_uid != action.action_uid:
                raise ValueError("Rollout callback returned a mismatched posterior or action identity.")
        elif isinstance(raw, Mapping):
            result = TrajectoryResult(
                posterior_uid=posterior.posterior_uid,
                action_uid=str(action.action_uid),
                trajectory=raw.get("trajectory", []),
                uncertainty=raw.get("uncertainty", {"aleatoric": 0.0, "epistemic": 0.0}),
                diagnostics=dict(raw.get("diagnostics", {})),
            )
        else:
            # Unconfigured execution is explicit and finite, never fabricated as evidence.
            result = TrajectoryResult(
                posterior_uid=posterior.posterior_uid,
                action_uid=str(action.action_uid),
                trajectory=[0.0],
                uncertainty={"aleatoric": 0.0, "epistemic": 0.0},
                diagnostics={"status": "unconfigured_world_executor"},
            )
        self.last_posterior_uid = posterior.posterior_uid
        if self._trace_recorder is not None:
            self._trace_recorder(
                posterior.posterior_uid,
                "rollout",
                posterior_uid_rollout=posterior.posterior_uid,
                action_uid=result.action_uid,
            )
        return result


class InformationChannel:
    def __init__(self, belief: ScientificBeliefKernel, tools: Mapping[str, Callable[[ScientificPosterior, InformationAction], Any]] | None = None) -> None:
        self.belief = belief
        self.tools: dict[str, Callable[[ScientificPosterior, InformationAction], Any]] = dict(tools or {})
        self.last_posterior_uid: str | None = None
        self._posterior_validator: Callable[[str], ScientificPosterior] | None = None
        self._trace_recorder: Callable[..., dict[str, Any]] | None = None

    def _bind_controller(
        self,
        posterior_validator: Callable[[str], ScientificPosterior],
        trace_recorder: Callable[..., dict[str, Any]],
    ) -> None:
        if not callable(posterior_validator) or not callable(trace_recorder):
            raise TypeError("InformationChannel controller hooks must be callable.")
        self._posterior_validator = posterior_validator
        self._trace_recorder = trace_recorder

    def register_tool(self, tool_id: str, handler: Callable[[ScientificPosterior, InformationAction], Any]) -> None:
        if not tool_id or not callable(handler):
            raise ValueError("register_tool requires a non-empty tool_id and callable handler.")
        self.tools[str(tool_id)] = handler

    def predict_tool_outcome(self, posterior_uid: str, information_action: InformationAction | Mapping[str, Any]) -> ToolResult:
        posterior = (
            self._posterior_validator(str(posterior_uid))
            if self._posterior_validator is not None
            else self.belief.get(posterior_uid)
        )
        action = information_action if isinstance(information_action, InformationAction) else InformationAction.from_mapping(information_action)
        if not isinstance(action, InformationAction):
            raise TypeError("predict_tool_outcome requires an InformationAction, not a world action.")
        handler = self.tools.get(action.tool_id)
        if handler is None:
            result = ToolResult(
                posterior_uid=posterior.posterior_uid,
                action_uid=str(action.action_uid),
                status="failed",
                error="tool_executor_unconfigured",
                provenance={"tool_id": action.tool_id, "failure_visible": True},
            )
        else:
            try:
                raw = handler(posterior, action)
                if isinstance(raw, ToolResult):
                    result = raw
                elif isinstance(raw, Mapping):
                    status = str(raw.get("status", "completed"))
                    result = ToolResult(
                        posterior_uid=posterior.posterior_uid,
                        action_uid=str(action.action_uid),
                        status=status,
                        observation=dict(raw.get("observation", {})),
                        uncertainty=dict(raw.get("uncertainty", {})),
                        error=str(raw["error"]) if raw.get("error") is not None else None,
                        provenance=dict(raw.get("provenance", {})),
                        partial=bool(raw.get("partial", status == "partial")),
                    )
                else:
                    raise TypeError("Tool handler must return ToolResult or mapping.")
                if result.posterior_uid != posterior.posterior_uid or result.action_uid != action.action_uid:
                    raise ValueError("Tool handler returned mismatched posterior or action identity.")
            except Exception as exc:  # tool failures are visible evidence, not silent fallbacks
                result = ToolResult(
                    posterior_uid=posterior.posterior_uid,
                    action_uid=str(action.action_uid),
                    status="failed",
                    error=f"tool_execution_error:{type(exc).__name__}",
                    provenance={"tool_id": action.tool_id, "failure_visible": True},
                )
        self.last_posterior_uid = posterior.posterior_uid
        if self._trace_recorder is not None:
            self._trace_recorder(
                posterior.posterior_uid,
                "predict_tool_outcome",
                posterior_uid_query=posterior.posterior_uid,
                action_uid=result.action_uid,
                tool_result_status=result.status,
            )
        return result


class DecisionController:
    """Decision surface that requires one posterior identity for every operation."""

    def __init__(self, belief: ScientificBeliefKernel, world: WorldChannel, information: InformationChannel) -> None:
        self.belief = belief
        self.world = world
        self.information = information
        self._current_posterior_uid: str | None = None
        self.trace: list[dict[str, Any]] = []
        self.world._bind_controller(self._require, self._record)
        self.information._bind_controller(self._require, self._record)

    @property
    def current_posterior_uid(self) -> str | None:
        return self._current_posterior_uid

    def bind(self, posterior: ScientificPosterior) -> None:
        self.belief.get(posterior.posterior_uid)
        if (
            self._current_posterior_uid is not None
            and self._current_posterior_uid != posterior.posterior_uid
        ):
            raise ValueError(
                "Decision-consistency violation: a controller cannot rebind an active posterior."
            )
        self._current_posterior_uid = posterior.posterior_uid

    def restore_posterior(self, posterior_uid: str, *, reason: str = "rollback") -> ScientificPosterior:
        """Restore a previously registered posterior through an explicit audit boundary."""
        posterior = self.belief.get(posterior_uid)
        previous = self._current_posterior_uid
        self._current_posterior_uid = posterior.posterior_uid
        self._record(
            posterior.posterior_uid,
            "restore_posterior",
            posterior_uid_result=posterior.posterior_uid,
            previous_posterior_uid=previous,
            reason=reason,
        )
        return posterior

    def register_posterior(
        self,
        posterior: ScientificPosterior,
        *,
        activate: bool = False,
        expected_parent_posterior_uid: str | None = None,
        assimilation_status: str | None = None,
        reason: str = "registered_posterior_restored",
    ) -> ScientificPosterior:
        """Register and optionally activate a posterior through the runtime API.

        ``activate=True`` is the only supported way for an adapter to restore
        a child posterior after a controlled update.  The expected parent is
        checked against both the child lineage and the currently active
        controller state, preventing a hidden state rebind.
        """
        if not activate:
            return self.belief.register_posterior(posterior)
        if not isinstance(posterior, ScientificPosterior):
            raise TypeError("register_posterior requires a ScientificPosterior")
        expected_parent = expected_parent_posterior_uid
        if expected_parent is not None and posterior.parent_posterior_uid != expected_parent:
            raise ValueError("Registered posterior parent does not match the expected assimilation parent")
        if expected_parent is not None:
            self.belief.get(expected_parent)
        if self._current_posterior_uid is not None:
            if expected_parent is None or self._current_posterior_uid != expected_parent:
                raise ValueError("Cannot activate a posterior from a different active parent")
        if assimilation_status is not None:
            if expected_parent is None:
                raise ValueError("Assimilation registration requires an explicit parent posterior")
        self.belief.register_posterior(posterior)
        self._current_posterior_uid = posterior.posterior_uid
        if assimilation_status is not None:
            self._record(
                expected_parent,
                "assimilate_observation",
                posterior_uid_assimilation_parent=expected_parent,
                posterior_uid_result=posterior.posterior_uid,
                accepted=True,
                reason=reason,
                tool_result_status=assimilation_status,
            )
        return posterior

    def _require(self, posterior_uid: str) -> ScientificPosterior:
        posterior = self.belief.get(posterior_uid)
        if self._current_posterior_uid is None:
            self._current_posterior_uid = posterior.posterior_uid
        if self._current_posterior_uid != posterior.posterior_uid:
            raise ValueError(
                f"Decision-consistency violation: current posterior {self._current_posterior_uid} != requested {posterior.posterior_uid}"
            )
        return posterior

    def _record(self, posterior_uid: str, operation: str, **payload: Any) -> dict[str, Any]:
        requested_uid = str(posterior_uid)
        row = {
            "schema_version": DECISION_TRACE_SCHEMA_VERSION,
            "operation": operation,
            # The invariant is evaluated for the operation being recorded.  The
            # channel ``last_*`` values are retained as diagnostics below and
            # never become a stale hidden decision state for a later operation.
            "posterior_uid_rollout": payload.pop("posterior_uid_rollout", requested_uid),
            "posterior_uid_query": payload.pop("posterior_uid_query", requested_uid),
            "posterior_uid_assimilation_parent": payload.pop("posterior_uid_assimilation_parent", requested_uid),
            "posterior_uid_decision": payload.pop("posterior_uid_decision", requested_uid),
            "posterior_uid_for_intervention_ranking": payload.pop(
                "posterior_uid_for_intervention_ranking", requested_uid
            ),
            "channel_last_posterior_uid_rollout": self.world.last_posterior_uid,
            "channel_last_posterior_uid_query": self.information.last_posterior_uid,
            **payload,
        }
        self.assert_consistent(row, allow_unset=True)
        self.trace.append(row)
        return row

    @staticmethod
    def assert_consistent(record: Mapping[str, Any], allow_unset: bool = False) -> None:
        values = [record.get(field) for field in _INVARIANT_FIELDS]
        if allow_unset:
            values = [value for value in values if value is not None]
        if values and len(set(str(value) for value in values)) != 1:
            raise ValueError(f"Decision-consistency invariant failed for fields {_INVARIANT_FIELDS}: {values}")

    def rank_interventions(self, posterior_uid: str, actions: Iterable[WorldAction | Mapping[str, Any]]) -> list[dict[str, Any]]:
        self._require(posterior_uid)
        rows: list[dict[str, Any]] = []
        for rank, item in enumerate(actions):
            action = item if isinstance(item, WorldAction) else WorldAction.from_mapping(item)
            score = float(self.belief.get(posterior_uid).reliability.get("tri_score", 0.0))
            rows.append({"action_uid": action.action_uid, "score": score, "rank": rank, "posterior_uid": posterior_uid})
        self._record(posterior_uid, "rank_interventions", ranked_count=len(rows))
        return rows

    def value_information_actions(self, posterior_uid: str, actions: Iterable[InformationAction | Mapping[str, Any]]) -> list[dict[str, Any]]:
        self._require(posterior_uid)
        rows: list[dict[str, Any]] = []
        for rank, item in enumerate(actions):
            action = item if isinstance(item, InformationAction) else InformationAction.from_mapping(item)
            utility = 1.0 / (1.0 + float(action.expected_cost))
            rows.append({"action_uid": action.action_uid, "utility": utility, "rank": rank, "posterior_uid": posterior_uid})
        self._record(posterior_uid, "value_information_actions", valued_count=len(rows))
        return rows

    def select_tool(self, posterior_uid: str, action: InformationAction | Mapping[str, Any]) -> InformationAction:
        self._require(posterior_uid)
        selected = action if isinstance(action, InformationAction) else InformationAction.from_mapping(action)
        self._record(posterior_uid, "select_tool", action_uid=selected.action_uid)
        return selected

    def assimilate_observation(self, posterior_uid: str, result: ToolResult) -> AssimilationOutcome:
        parent = self._require(posterior_uid)
        if result.posterior_uid != parent.posterior_uid:
            raise ValueError("Tool result posterior_uid does not match assimilation parent.")
        if not result.valid_for_assimilation:
            outcome = AssimilationOutcome(False, parent, parent.posterior_uid, "invalid_or_failed_tool_result_rollback", result.status)
            self._record(parent.posterior_uid, "assimilate_observation", posterior_uid_assimilation_parent=parent.posterior_uid, accepted=False, reason=outcome.reason, tool_result_status=result.status)
            return outcome
        child = self.belief.child_from_observation(parent, result.observation, result.uncertainty)
        self._current_posterior_uid = child.posterior_uid
        outcome = AssimilationOutcome(True, child, parent.posterior_uid, "partial_result_assimilated" if result.partial or result.status == "partial" else "observation_assimilated", result.status)
        # Assimilation is a transition event: the invariant is evaluated on
        # the parent scientific state consumed by the observation.  The child
        # identity is explicit and becomes the required identity for the next
        # operation; it is never hidden in a channel or decision component.
        self._record(
            parent.posterior_uid,
            "assimilate_observation",
            posterior_uid_assimilation_parent=parent.posterior_uid,
            posterior_uid_result=child.posterior_uid,
            accepted=True,
            reason=outcome.reason,
            tool_result_status=result.status,
        )
        return outcome

    def compute_reliability(self, posterior_uid: str) -> dict[str, Any]:
        posterior = self._require(posterior_uid)
        payload = {"posterior_uid": posterior_uid, **_thaw(posterior.reliability)}
        self._record(posterior_uid, "compute_reliability", reliability=payload)
        return payload

    def _decision(self, posterior_uid: str, operation: str, **payload: Any) -> dict[str, Any]:
        self._require(posterior_uid)
        return self._record(posterior_uid, operation, **payload)

    def accept(self, posterior_uid: str, reason: str = "") -> dict[str, Any]:
        return self._decision(posterior_uid, "accept", decision_label="accept", reason=reason)

    def reject(self, posterior_uid: str, reason: str = "") -> dict[str, Any]:
        return self._decision(posterior_uid, "reject", decision_label="reject", reason=reason)

    def acquire(self, posterior_uid: str, action_uid: str | None = None) -> dict[str, Any]:
        return self._decision(posterior_uid, "acquire", decision_label="active_learning", action_uid=action_uid)

    def escalate(self, posterior_uid: str, reason: str = "") -> dict[str, Any]:
        return self._decision(posterior_uid, "escalate", decision_label="external_validation", reason=reason)

    def abstain(self, posterior_uid: str, reason: str = "") -> dict[str, Any]:
        return self._decision(posterior_uid, "abstain", decision_label="insufficient_evidence", reason=reason)

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "schema_version": DECISION_TRACE_SCHEMA_VERSION,
            "current_posterior_uid": self._current_posterior_uid,
            "runtime_capabilities": {
                "world_executor_configured": self.world.rollout_fn is not None,
                "information_tool_ids": sorted(self.information.tools),
            },
            "posteriors": [posterior.to_record() for posterior in self.belief.posteriors.values()],
            "trace": copy.deepcopy(self.trace),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        rollout_fn: Callable[[ScientificPosterior, WorldAction], Any] | None = None,
        tools: Mapping[
            str, Callable[[ScientificPosterior, InformationAction], Any]
        ]
        | None = None,
    ) -> "DecisionController":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != DECISION_TRACE_SCHEMA_VERSION:
            raise ValueError("Unsupported decision-consistency checkpoint schema.")
        runtime_capabilities = payload.get("runtime_capabilities")
        if not isinstance(runtime_capabilities, Mapping):
            raise ValueError(
                "Checkpoint is missing the required runtime_capabilities contract."
            )
        world_executor_configured = runtime_capabilities.get(
            "world_executor_configured"
        )
        information_tool_ids = runtime_capabilities.get("information_tool_ids")
        if (
            not isinstance(world_executor_configured, bool)
            or not isinstance(information_tool_ids, list)
            or any(
                not isinstance(tool_id, str) or not tool_id
                for tool_id in information_tool_ids
            )
            or len(set(information_tool_ids)) != len(information_tool_ids)
        ):
            raise ValueError("Checkpoint runtime_capabilities contract is invalid.")
        expected_world_executor = world_executor_configured
        expected_tool_ids = set(information_tool_ids)
        provided_tool_ids = set((tools or {}).keys())
        if expected_world_executor != (rollout_fn is not None):
            raise ValueError(
                "Checkpoint runtime capability mismatch: rollout_fn binding does not match the saved runtime."
            )
        if expected_tool_ids != provided_tool_ids:
            raise ValueError(
                "Checkpoint runtime capability mismatch: tool registry IDs do not match the saved runtime."
            )
        posterior_rows = payload.get("posteriors")
        if not isinstance(posterior_rows, list) or not posterior_rows:
            raise ValueError("Checkpoint must contain a non-empty posterior registry")
        belief = ScientificBeliefKernel()
        staged_posteriors: dict[str, ScientificPosterior] = {}
        for index, row in enumerate(posterior_rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"Checkpoint posterior row {index} is not a mapping")
            required_fields = {
                "posterior_uid",
                "version",
                "parent_posterior_uid",
                "state",
                "uncertainty",
                "support",
                "mechanism",
                "reliability",
                "content_digest",
                "uid_content_digest",
            }
            missing_fields = sorted(required_fields.difference(row))
            if missing_fields:
                raise ValueError(
                    f"Checkpoint posterior row {index} is missing fields: {missing_fields}"
                )
            posterior = ScientificPosterior(
                posterior_uid=str(row["posterior_uid"]),
                version=int(row["version"]),
                parent_posterior_uid=row.get("parent_posterior_uid"),
                state=_freeze(row["state"]),
                uncertainty=_freeze(row["uncertainty"]),
                support=_freeze(row["support"]),
                mechanism=_freeze(row["mechanism"]),
                reliability=_freeze(row["reliability"]),
                schema_version=str(row.get("schema_version", POSTERIOR_SCHEMA_VERSION)),
                content_digest=str(row.get("content_digest", "")),
                uid_content_digest=str(row.get("uid_content_digest", "")),
            )
            if posterior.posterior_uid in staged_posteriors:
                raise ValueError(
                    f"Checkpoint contains duplicate posterior UID: {posterior.posterior_uid}"
                )
            staged_posteriors[posterior.posterior_uid] = posterior
            belief.register_posterior(posterior)

        # Validate the immutable posterior graph after staging so row order is
        # irrelevant, while still rejecting missing parents, version jumps,
        # self-cycles, and cyclic ancestry.
        for posterior in staged_posteriors.values():
            parent_uid = posterior.parent_posterior_uid
            if parent_uid is None:
                if posterior.version != 1:
                    raise ValueError(
                        f"Root posterior {posterior.posterior_uid} must have version 1"
                    )
                continue
            if parent_uid == posterior.posterior_uid:
                raise ValueError("Checkpoint posterior lineage cannot self-reference")
            parent = staged_posteriors.get(parent_uid)
            if parent is None:
                raise ValueError(
                    f"Checkpoint posterior {posterior.posterior_uid} references missing parent {parent_uid}"
                )
            if posterior.version != parent.version + 1:
                raise ValueError(
                    f"Checkpoint posterior {posterior.posterior_uid} has an invalid version transition"
                )
            seen: set[str] = set()
            cursor: ScientificPosterior | None = posterior
            while cursor.parent_posterior_uid is not None:
                if cursor.posterior_uid in seen:
                    raise ValueError("Checkpoint posterior lineage contains a cycle")
                seen.add(cursor.posterior_uid)
                cursor = staged_posteriors.get(cursor.parent_posterior_uid)
                if cursor is None:
                    raise ValueError("Checkpoint posterior lineage contains a missing ancestor")

        # A checkpoint is a persisted scientific state, so every row must carry
        # the UID digest that binds its identity to the serialized posterior.
        # This check intentionally follows graph validation so lineage-specific
        # diagnostics remain actionable for malformed ancestry fixtures.
        for posterior in staged_posteriors.values():
            if not posterior.uid_content_digest:
                raise ValueError(
                    f"Checkpoint posterior {posterior.posterior_uid} is missing uid_content_digest"
                )

        controller = cls(
            belief,
            WorldChannel(belief, rollout_fn=rollout_fn),
            InformationChannel(belief, tools=tools),
        )
        controller._current_posterior_uid = payload.get("current_posterior_uid")
        if not isinstance(controller._current_posterior_uid, str):
            raise ValueError("Checkpoint current_posterior_uid must identify a registered posterior")
        belief.get(controller._current_posterior_uid)
        trace_rows = payload.get("trace")
        if not isinstance(trace_rows, list):
            raise ValueError("Checkpoint trace must be a list")
        known_operations = {
            "rollout",
            "predict_tool_outcome",
            "rank_interventions",
            "value_information_actions",
            "select_tool",
            "assimilate_observation",
            "compute_reliability",
            "accept",
            "reject",
            "acquire",
            "escalate",
            "abstain",
            "restore_posterior",
        }
        controller.trace = []
        for index, row in enumerate(trace_rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"Checkpoint trace row {index} is not a mapping")
            if row.get("schema_version") != DECISION_TRACE_SCHEMA_VERSION:
                raise ValueError(f"Checkpoint trace row {index} has an unsupported schema")
            operation = row.get("operation")
            if operation not in known_operations:
                raise ValueError(f"Checkpoint trace row {index} has an unsupported operation")
            invariant_values = [row.get(field) for field in _INVARIANT_FIELDS]
            if any(value is None for value in invariant_values):
                raise ValueError(f"Checkpoint trace row {index} is missing posterior invariants")
            if len({str(value) for value in invariant_values}) != 1:
                raise ValueError(f"Checkpoint trace row {index} violates posterior invariants")
            for field_name in (*_INVARIANT_FIELDS, "posterior_uid_result", "previous_posterior_uid"):
                referenced_uid = row.get(field_name)
                if referenced_uid is not None:
                    belief.get(str(referenced_uid))
            if operation == "assimilate_observation":
                accepted = row.get("accepted")
                result_uid = row.get("posterior_uid_result")
                parent_uid = row.get("posterior_uid_assimilation_parent")
                if not isinstance(accepted, bool):
                    raise ValueError(f"Checkpoint trace row {index} has invalid assimilation status")
                if accepted:
                    if not isinstance(result_uid, str) or not result_uid:
                        raise ValueError(f"Checkpoint trace row {index} is missing assimilation result UID")
                    child = staged_posteriors[result_uid]
                    parent = staged_posteriors[str(parent_uid)]
                    if child.parent_posterior_uid != parent.posterior_uid or child.version != parent.version + 1:
                        raise ValueError(f"Checkpoint trace row {index} has invalid assimilation lineage")
                elif result_uid is not None and str(result_uid) != str(parent_uid):
                    raise ValueError(f"Checkpoint trace row {index} has an invalid rejected assimilation result")
            elif operation == "restore_posterior":
                if not isinstance(row.get("posterior_uid_result"), str):
                    raise ValueError(f"Checkpoint trace row {index} is missing restore result UID")
            cls.assert_consistent(row, allow_unset=True)
            controller.trace.append(dict(row))
        return controller


class DecisionConsistentTwin:
    """Convenience facade exposing the three canonical kernels."""

    def __init__(
        self,
        rollout_fn: Callable[[ScientificPosterior, WorldAction], Any] | None = None,
        tools: Mapping[str, Callable[[ScientificPosterior, InformationAction], Any]] | None = None,
    ) -> None:
        self.belief = ScientificBeliefKernel()
        self.world = WorldChannel(self.belief, rollout_fn=rollout_fn)
        self.information = InformationChannel(self.belief, tools=tools)
        self.decisions = DecisionController(self.belief, self.world, self.information)

    def infer_belief(self, evidence: Mapping[str, Any], graph: Mapping[str, Any]) -> ScientificPosterior:
        posterior = self.belief.infer_belief(evidence, graph)
        self.decisions.bind(posterior)
        return posterior

    def register_posterior(
        self,
        posterior: ScientificPosterior,
        *,
        activate: bool = False,
        expected_parent_posterior_uid: str | None = None,
        assimilation_status: str | None = None,
        reason: str = "registered_posterior_restored",
    ) -> ScientificPosterior:
        """Expose the guarded posterior registration/restore boundary."""
        return self.decisions.register_posterior(
            posterior,
            activate=activate,
            expected_parent_posterior_uid=expected_parent_posterior_uid,
            assimilation_status=assimilation_status,
            reason=reason,
        )

    def rollout(self, posterior_uid: str, world_action: WorldAction | Mapping[str, Any]) -> TrajectoryResult:
        return self.world.rollout(posterior_uid, world_action)

    def predict_tool_outcome(self, posterior_uid: str, information_action: InformationAction | Mapping[str, Any]) -> ToolResult:
        return self.information.predict_tool_outcome(
            posterior_uid, information_action
        )

    def assimilate_observation(self, posterior_uid: str, result: ToolResult) -> AssimilationOutcome:
        return self.decisions.assimilate_observation(posterior_uid, result)

    def save_checkpoint(self, path: str | Path) -> None:
        self.decisions.save_checkpoint(path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        rollout_fn: Callable[[ScientificPosterior, WorldAction], Any] | None = None,
        tools: Mapping[
            str, Callable[[ScientificPosterior, InformationAction], Any]
        ]
        | None = None,
    ) -> "DecisionConsistentTwin":
        controller = DecisionController.load_checkpoint(
            path,
            rollout_fn=rollout_fn,
            tools=tools,
        )
        twin = cls()
        twin.belief = controller.belief
        twin.world = controller.world
        twin.information = controller.information
        twin.decisions = controller
        return twin


def infer_belief(evidence: Mapping[str, Any], graph: Mapping[str, Any]) -> ScientificPosterior:
    """Stateless convenience entry point for contract and smoke tests."""
    return ScientificBeliefKernel().infer_belief(evidence, graph)


__all__ = [
    "AssimilationOutcome",
    "DecisionConsistentTwin",
    "DecisionController",
    "InformationAction",
    "ScientificBeliefKernel",
    "ScientificPosterior",
    "ToolResult",
    "TrajectoryResult",
    "WorldAction",
    "WorldChannel",
    "InformationChannel",
    "infer_belief",
]
