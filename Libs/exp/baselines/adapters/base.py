from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BaselineCapability:
    information_available: tuple[str, ...]
    information_unavailable_by_design: tuple[str, ...]
    axis_conversion: str
    feature_mapping: str
    mask_handling: str
    action_representation: str
    uncertainty_support: str
    semantic_loss: str


@dataclass(frozen=True)
class CompatibilityDecision:
    status: str
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"applicable", "not_applicable", "pending"}:
            raise ValueError(f"Unsupported compatibility status: {self.status}")
        if not self.reason:
            raise ValueError("Compatibility decisions require an explicit reason.")


class NeuronSifterBaselineAdapter(ABC):
    """Uniform adapter boundary; incompatible tasks must fail visibly."""

    @abstractmethod
    def capability(self) -> BaselineCapability:
        raise NotImplementedError

    @abstractmethod
    def validate_compatibility(self, task: Mapping[str, Any]) -> CompatibilityDecision:
        raise NotImplementedError

    @abstractmethod
    def materialize_inputs(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train(self, run_context: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, checkpoint: str, eval_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def normalize_outputs(self, result: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def resource_usage(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def require_compatible(self, task: Mapping[str, Any]) -> None:
        decision = self.validate_compatibility(task)
        if decision.status != "applicable":
            raise ValueError(f"Baseline task compatibility is {decision.status}: {decision.reason}")
