"""Typed scientific tool fabric with strict trust boundaries and receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Mapping
from uuid import uuid4

TRUST_BOUNDARIES = frozenset(
    (
        "neuronsifter-twin-mcp",
        "neuronsifter-intervention-mcp",
        "neuronsifter-evidence-mcp",
        "neuronsifter-memory-mcp",
        "neuronsifter-hpc-mcp",
        "neuronsifter-science-tools-mcp",
    )
)


def _uid(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "tool_call_" + hashlib.sha256(payload.encode()).hexdigest()[:24]

@dataclass(frozen=True)
class ScientificToolReceipt:
    """Auditable result envelope returned by every registered tool."""
    tool_call_uid: str
    tool_name: str
    tool_version: str
    status: str
    posterior_uid: str
    input_manifest_uid: str | None = None
    world_action_uid: str | None = None
    information_action_uid: str | None = None
    run_uid: str | None = None
    artifact_refs: tuple[str, ...] = ()
    result_summary: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    support: Mapping[str, Any] = field(default_factory=dict)
    evidence_grade: str | None = None
    runtime: float | None = None
    cost: float | None = None
    failure_class: str | None = None
    provenance_hash: str = ""
    observation_uid: str = ""

    def _identity_payload(self) -> dict[str, Any]:
        """Return every receipt field that contributes to scientific identity."""
        return {
            "tool_call_uid": self.tool_call_uid,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "status": self.status,
            "posterior_uid": self.posterior_uid,
            "input_manifest_uid": self.input_manifest_uid,
            "world_action_uid": self.world_action_uid,
            "information_action_uid": self.information_action_uid,
            "run_uid": self.run_uid,
            "artifact_refs": tuple(self.artifact_refs),
            "result_summary": self.result_summary,
            "uncertainty": self.uncertainty,
            "support": self.support,
            "evidence_grade": self.evidence_grade,
            "runtime": self.runtime,
            "cost": self.cost,
            "failure_class": self.failure_class,
        }

    def computed_provenance_hash(self) -> str:
        payload = json.dumps(
            self._identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def __post_init__(self) -> None:
        if not self.posterior_uid:
            raise ValueError("posterior_uid is required for every scientific tool receipt")
        if not self.tool_call_uid or not self.tool_name:
            raise ValueError("tool_call_uid and tool_name are required")
        if self.status not in {
            "succeeded",
            "completed",
            "partial",
            "failed",
            "invalid",
            "submitted",
            "running",
            "cancelled",
        }:
            raise ValueError(f"invalid receipt status: {self.status}")
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        for name, value in (("runtime", self.runtime), ("cost", self.cost)):
            if value is not None and (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative when provided")
        expected = self.computed_provenance_hash()
        if self.provenance_hash and self.provenance_hash != expected:
            raise ValueError("provenance_hash does not match the receipt identity")
        object.__setattr__(self, "provenance_hash", expected)
        expected_observation_uid = "observation-" + expected[:24]
        if self.observation_uid and self.observation_uid != expected_observation_uid:
            raise ValueError("observation_uid does not match the receipt provenance")
        object.__setattr__(self, "observation_uid", expected_observation_uid)

    def to_record(self) -> dict[str, Any]:
        return {
            "tool_call_uid": self.tool_call_uid,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "status": self.status,
            "input_manifest_uid": self.input_manifest_uid,
            "posterior_uid": self.posterior_uid,
            "world_action_uid": self.world_action_uid,
            "information_action_uid": self.information_action_uid,
            "run_uid": self.run_uid,
            "artifact_refs": list(self.artifact_refs),
            "result_summary": dict(self.result_summary),
            "uncertainty": dict(self.uncertainty),
            "support": dict(self.support),
            "evidence_grade": self.evidence_grade,
            "runtime": self.runtime,
            "cost": self.cost,
            "failure_class": self.failure_class,
            "provenance_hash": self.provenance_hash,
            "observation_uid": self.observation_uid,
        }

@dataclass(frozen=True)
class _ToolSpec:
    boundary: str
    handler: Callable[[Mapping[str, Any]], Any]
    version: str
    mutates_twin: bool
    long_running: bool

class ScientificToolFabric:
    """Registry and execution facade; unrestricted shell execution is impossible."""
    def __init__(self) -> None:
        self._tools: dict[str, _ToolSpec] = {}
        self._tasks: dict[str, ScientificToolReceipt] = {}
        self._receipts: dict[str, ScientificToolReceipt] = {}

    def register(
        self,
        name: str,
        *,
        boundary: str,
        handler: Callable[[Mapping[str, Any]], Any],
        version: str = "1",
        mutates_twin: bool = False,
        long_running: bool = False,
    ) -> None:
        if boundary not in TRUST_BOUNDARIES: raise ValueError(f"unknown trust boundary: {boundary}")
        forbidden_names = {"shell", "exec", "execute", "command", "run_shell", "shell_exec"}
        if (
            not name
            or "shell" in name.lower()
            or str(name).lower() in forbidden_names
            or name in self._tools
        ):
            raise ValueError("tool name is empty, unrestricted, or already registered")
        self._tools[name] = _ToolSpec(boundary, handler, version, mutates_twin, long_running)

    def list_tools(self, *, boundary: str | None = None) -> tuple[str, ...]:
        if boundary is not None and boundary not in TRUST_BOUNDARIES:
            raise ValueError(f"unknown trust boundary: {boundary}")
        return tuple(
            sorted(
                name
                for name, spec in self._tools.items()
                if boundary is None or spec.boundary == boundary
            )
        )

    def invoke(
        self,
        name: str,
        *,
        posterior_uid: str,
        inputs: Mapping[str, Any] | None = None,
    ) -> ScientificToolReceipt:
        spec = self._require(name)
        payload = dict(inputs or {})
        if spec.mutates_twin:
            parent_uid = payload.get("parent_posterior_uid")
            if not parent_uid:
                raise ValueError("mutating tools require parent_posterior_uid")
            if str(parent_uid) != str(posterior_uid):
                raise ValueError("mutating tool parent_posterior_uid must match the active posterior")
        call_uid = _uid({"name": name, "posterior": posterior_uid, "inputs": payload})
        cached = self._receipts.get(call_uid)
        if cached is not None:
            return cached
        receipt_kwargs = {
            "input_manifest_uid": str(payload["input_manifest_uid"])
            if payload.get("input_manifest_uid")
            else None,
            "world_action_uid": str(payload["world_action_uid"])
            if payload.get("world_action_uid")
            else None,
            "information_action_uid": str(payload["information_action_uid"])
            if payload.get("information_action_uid")
            else None,
        }
        try:
            result = spec.handler(payload)
            if not isinstance(result, Mapping):
                raise TypeError("scientific tool handlers must return a mapping receipt payload")
            summary = dict(result)
            declared_status = summary.get("status")
            status = "submitted" if spec.long_running else str(declared_status or "succeeded")
            allowed_statuses = {
                "succeeded",
                "completed",
                "partial",
                "failed",
                "invalid",
                "submitted",
                "running",
                "cancelled",
            }
            if status not in allowed_statuses:
                raise ValueError(f"scientific tool returned unsupported status: {status}")
            receipt = ScientificToolReceipt(
                call_uid,
                name,
                spec.version,
                status,
                posterior_uid,
                **receipt_kwargs,
                result_summary=summary,
                uncertainty=(
                    dict(summary.get("uncertainty", {}))
                    if isinstance(summary.get("uncertainty"), Mapping)
                    else {}
                ),
                support=(
                    dict(summary.get("support", {}))
                    if isinstance(summary.get("support"), Mapping)
                    else {}
                ),
                evidence_grade=(
                    str(summary["evidence_grade"])
                    if summary.get("evidence_grade") is not None
                    else None
                ),
                runtime=(
                    float(summary["runtime"])
                    if isinstance(summary.get("runtime"), (int, float))
                    else None
                ),
                cost=(
                    float(summary["cost"])
                    if isinstance(summary.get("cost"), (int, float))
                    else None
                ),
                failure_class=(
                    str(summary["failure_class"])
                    if summary.get("failure_class") is not None
                    else None
                ),
                run_uid=(str(uuid4()) if spec.long_running else None),
            )
        except Exception as exc:
            receipt = ScientificToolReceipt(
                call_uid,
                name,
                spec.version,
                "failed",
                posterior_uid,
                **receipt_kwargs,
                failure_class=(
                    "invalid_tool_receipt"
                    if isinstance(exc, (TypeError, ValueError))
                    else type(exc).__name__
                ),
                result_summary={"error": str(exc)},
            )
        self._receipts[call_uid] = receipt
        if spec.long_running and receipt.run_uid: self._tasks[receipt.run_uid] = receipt
        return receipt

    def submit_task(
        self,
        name: str,
        *,
        posterior_uid: str,
        inputs: Mapping[str, Any] | None = None,
    ) -> ScientificToolReceipt:
        if not self._require(name).long_running:
            raise ValueError("submit_task requires a long-running registered tool")
        return self.invoke(name, posterior_uid=posterior_uid, inputs=inputs)

    def task_status(self, run_uid: str) -> ScientificToolReceipt:
        try:
            return self._tasks[run_uid]
        except KeyError as exc:
            raise KeyError(f"unknown scientific task: {run_uid}") from exc

    def _require(self, name: str) -> _ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unregistered scientific tool: {name}") from exc
