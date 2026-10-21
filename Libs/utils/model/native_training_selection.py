"""Development-only checkpoint selection and finite scientific stopping."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from Libs.utils.data.raw_data_registry import resolve_repo_path


@dataclass
class NativeTrainingSelection:
    policy: dict
    history: list[dict] = field(default_factory=list)
    best: dict | None = None
    last: dict | None = None
    stale_epochs: int = 0
    terminal: str = "RUNNING"

    def update(self, *, role: str, epoch: int, steps: int, training_loss: float,
               development_loss: float, checkpoint_path: str, elapsed_seconds: float) -> bool:
        if role != "development":
            raise ValueError("Checkpoint selection requires development groups only")
        if not all(math.isfinite(v) for v in (training_loss, development_loss, elapsed_seconds)):
            raise FloatingPointError("Nonfinite scientific selection evidence")
        if self.last and (epoch <= self.last["epoch"] or steps <= self.last["steps"]):
            raise ValueError("Selection snapshots require increasing epochs and steps")
        row = {"epoch": epoch, "steps": steps, "training_loss": training_loss,
               "development_loss": development_loss, "checkpoint_path": checkpoint_path,
               "elapsed_seconds": elapsed_seconds}
        self.last = dict(row)
        self.history.append({key:value for key,value in row.items() if key != "checkpoint_path"})
        improved = self.best is None or development_loss < self.best["development_loss"] - self.policy["min_delta"]
        self.history[-1]["selected_best"] = improved
        if improved:
            self.best, self.stale_epochs = dict(row), 0
        else:
            self.stale_epochs += 1
        if steps >= self.policy["max_steps"] or elapsed_seconds >= self.policy["max_seconds"]:
            self.terminal = "INCOMPLETE_BUDGET_CAP"
        elif (epoch >= self.policy["minimum_epochs"] and steps >= self.policy["minimum_steps"]
              and self.stale_epochs >= self.policy["patience"]):
            self.terminal = "DEVELOPMENT_PLATEAU_REQUIRES_SUFFICIENCY_REVIEW"
        elif epoch >= self.policy["max_epochs"]:
            self.terminal = "INCOMPLETE_EPOCH_CAP"
        return self.terminal != "RUNNING"

    def payload(self, *, artifact_root: Path | None = None) -> dict:
        payload = {"selection_role": "development", "best": self.best, "last": self.last,
                   "history": self.history, "terminal": self.terminal,
                   "sufficient_training": False, "scientific_twin": "NOT_RUN"}
        if artifact_root is not None:
            # Keep live paths untouched for checkpoint retirement. Only emitted
            # references move with the manifest and its retained checkpoint files.
            root = Path(artifact_root).absolute()
            payload["checkpoint_path_base"] = "manifest_directory"
            for key in ("best", "last"):
                row = payload[key]
                if row is not None:
                    relative = Path(row["checkpoint_path"]).absolute().relative_to(root)
                    payload[key] = dict(row, checkpoint_path=str(relative))
        return payload


def training_checkpoint_path(manifest_path: Path, selection: dict, reference: str) -> Path:
    """Resolve emitted native references and the historical repository-based format."""
    base = selection.get("checkpoint_path_base", "repository")
    if base not in {"repository", "manifest_directory"}:
        raise ValueError("Unknown native training checkpoint path base")
    path = Path(reference)
    if base == "manifest_directory" and not path.is_absolute():
        return (Path(manifest_path).parent / path).resolve()
    return resolve_repo_path(path)


def validation_checkpoint_due(policy: dict, steps: int, last_checkpoint_step: int,
                              epoch: int, *, epoch_end: bool, budget_capped: bool) -> bool:
    """The manuscript scope validates on shared optimizer steps; old epoch mode remains."""
    if steps==last_checkpoint_step:
        return False
    interval=policy["validation"].get("interval_optimizer_steps")
    if interval is not None and (isinstance(interval,bool) or not isinstance(interval,int) or interval<1):
        raise ValueError("Validation optimizer-step interval must be a positive integer")
    if budget_capped or epoch_end and epoch>=policy["selection"]["max_epochs"]:
        return True
    return epoch_end if interval is None else steps-last_checkpoint_step>=interval


def optimizer_learning_rate_ratio(steps: int, policy: dict) -> float:
    """Shared warmup/cosine clock in completed optimizer updates."""
    schedule=policy["learning_rate_schedule"]
    warmup=schedule["warmup_steps"]
    maximum=policy["selection"]["max_steps"]
    if steps<warmup:
        return (steps+1)/max(warmup,1)
    progress=min(1.,(steps-warmup)/max(1,maximum-warmup))
    return schedule["minimum_ratio"]+(1-schedule["minimum_ratio"])*(1+math.cos(math.pi*progress))/2
