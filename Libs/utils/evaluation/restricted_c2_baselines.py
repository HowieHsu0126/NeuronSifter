"""Small, deterministic C2 baseline adapters used for software qualification.

These adapters exercise the real restricted current-clamp trajectory contract.
They are intentionally limited to adapter qualification: callers must keep
final-holdout trajectories and formal challenge thresholds out of this path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LinearStateSpaceModel:
    """Action-conditioned ridge state-transition model for a canonical grid."""

    coefficients: np.ndarray
    state_shape: tuple[int, ...]


class ActionConditionedGru(nn.Module):
    """Minimal recurrent rollout baseline over flattened canonical state grids."""

    def __init__(self, state_size: int, hidden_size: int) -> None:
        super().__init__()
        self.recurrent = nn.GRU(state_size + 1, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, state_size)

    def forward(self, states: torch.Tensor, action_values: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError(f"GRU states must be rank three, got {tuple(states.shape)}.")
        if action_values.shape != states.shape[:2] + (1,):
            raise ValueError(
                "GRU action values must have shape (batch, time, 1), "
                f"got {tuple(action_values.shape)} for states {tuple(states.shape)}."
            )
        sequence = torch.cat((states, action_values), dim=-1)
        encoded, _ = self.recurrent(sequence)
        return self.readout(encoded)


def flatten_state_grid(trajectories: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flatten the non-simulation/time axes while preserving trajectory order."""

    array = np.asarray(trajectories, dtype=np.float64)
    if array.ndim != 5:
        raise ValueError(
            "Restricted C2 trajectories must use simulation x time x region x compartment x species axes, "
            f"got shape {tuple(array.shape)}."
        )
    if array.shape[0] == 0 or array.shape[1] < 2 or any(size <= 0 for size in array.shape[2:]):
        raise ValueError(f"Restricted C2 trajectories have an invalid shape: {tuple(array.shape)}.")
    if not np.isfinite(array).all():
        raise ValueError("Restricted C2 trajectories contain NaN or Inf values.")
    state_shape = tuple(int(value) for value in array.shape[2:])
    return array.reshape(array.shape[0], array.shape[1], -1), state_shape


def persistence_rollout(initial_states: np.ndarray, time_count: int) -> np.ndarray:
    """Carry the first observed state forward without action information."""

    states = np.asarray(initial_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] == 0:
        raise ValueError(f"Persistence requires non-empty two-dimensional initial states, got {tuple(states.shape)}.")
    if int(time_count) < 2:
        raise ValueError(f"Persistence requires at least two time points, got {time_count}.")
    return np.repeat(states[:, None, :], int(time_count), axis=1)


def fit_linear_state_space(
    train_trajectories: np.ndarray,
    action_values: np.ndarray,
    *,
    ridge_penalty: float,
) -> LinearStateSpaceModel:
    """Fit an action-conditioned ridge model to train-only state deltas."""

    if float(ridge_penalty) <= 0.0:
        raise ValueError("Linear state-space ridge_penalty must be positive.")
    flattened, state_shape = flatten_state_grid(train_trajectories)
    actions = np.asarray(action_values, dtype=np.float64).reshape(-1)
    if len(actions) != flattened.shape[0]:
        raise ValueError(
            "Linear state-space action count must match trajectory count, "
            f"got {len(actions)} and {flattened.shape[0]}."
        )
    if not np.isfinite(actions).all():
        raise ValueError("Linear state-space action values contain NaN or Inf.")

    current = flattened[:, :-1, :].reshape(-1, flattened.shape[-1])
    next_state = flattened[:, 1:, :].reshape(-1, flattened.shape[-1])
    repeated_actions = np.repeat(actions, flattened.shape[1] - 1)[:, None]
    design = np.concatenate((current, repeated_actions, np.ones((len(current), 1))), axis=1)
    deltas = next_state - current
    gram = design.T @ design
    gram += np.eye(gram.shape[0], dtype=np.float64) * float(ridge_penalty)
    coefficients = np.linalg.solve(gram, design.T @ deltas)
    if not np.isfinite(coefficients).all():
        raise ValueError("Linear state-space fit produced NaN or Inf coefficients.")
    return LinearStateSpaceModel(coefficients=coefficients, state_shape=state_shape)


def linear_state_space_rollout(
    model: LinearStateSpaceModel,
    initial_states: np.ndarray,
    action_values: np.ndarray,
    time_count: int,
) -> np.ndarray:
    """Roll out the fitted state-space model autoregressively."""

    states = np.asarray(initial_states, dtype=np.float64)
    actions = np.asarray(action_values, dtype=np.float64).reshape(-1)
    expected_state_size = int(np.prod(model.state_shape))
    if states.ndim != 2 or states.shape[1] != expected_state_size:
        raise ValueError(
            "Linear state-space initial states do not match the fitted grid, "
            f"got {tuple(states.shape)} and expected state width {expected_state_size}."
        )
    if len(actions) != states.shape[0] or not np.isfinite(actions).all():
        raise ValueError("Linear state-space rollout actions are invalid.")
    if int(time_count) < 2:
        raise ValueError(f"Linear state-space rollout requires at least two time points, got {time_count}.")

    prediction = np.empty((states.shape[0], int(time_count), states.shape[1]), dtype=np.float64)
    prediction[:, 0, :] = states
    for time_index in range(1, int(time_count)):
        previous = prediction[:, time_index - 1, :]
        design = np.concatenate((previous, actions[:, None], np.ones((len(previous), 1))), axis=1)
        prediction[:, time_index, :] = previous + design @ model.coefficients
    if not np.isfinite(prediction).all():
        raise ValueError("Linear state-space rollout produced NaN or Inf values.")
    return prediction


def train_gru_smoke(
    train_trajectories: np.ndarray,
    action_values: np.ndarray,
    *,
    hidden_size: int,
    epoch_count: int,
    learning_rate: float,
    seed: int,
) -> ActionConditionedGru:
    """Run a bounded train-only GRU smoke without selecting on holdout data."""

    if int(hidden_size) <= 0 or int(epoch_count) <= 0 or float(learning_rate) <= 0.0:
        raise ValueError("GRU smoke hyperparameters must be positive.")
    flattened, _ = flatten_state_grid(train_trajectories)
    actions = np.asarray(action_values, dtype=np.float32).reshape(-1)
    if len(actions) != flattened.shape[0] or not np.isfinite(actions).all():
        raise ValueError("GRU smoke action values are invalid.")

    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_thread_count = torch.get_num_threads()
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(1)
    try:
        model = ActionConditionedGru(flattened.shape[-1], int(hidden_size))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        source = torch.as_tensor(flattened[:, :-1, :], dtype=torch.float32)
        target = torch.as_tensor(flattened[:, 1:, :], dtype=torch.float32)
        action_tensor = torch.as_tensor(
            np.repeat(actions[:, None, None], source.shape[1], axis=1),
            dtype=torch.float32,
        )
        model.train()
        for _ in range(int(epoch_count)):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source, action_tensor)
            loss = torch.mean((prediction - target) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("GRU smoke loss is non-finite.")
            loss.backward()
            optimizer.step()
        return model.eval()
    finally:
        torch.use_deterministic_algorithms(previous_determinism, warn_only=False)
        torch.set_num_threads(previous_thread_count)


def gru_rollout(
    model: ActionConditionedGru,
    initial_states: np.ndarray,
    action_values: np.ndarray,
    time_count: int,
) -> np.ndarray:
    """Generate an autoregressive GRU rollout from one observed initial state."""

    states = np.asarray(initial_states, dtype=np.float32)
    actions = np.asarray(action_values, dtype=np.float32).reshape(-1)
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] == 0:
        raise ValueError(f"GRU rollout requires non-empty two-dimensional initial states, got {tuple(states.shape)}.")
    if len(actions) != states.shape[0] or not np.isfinite(actions).all():
        raise ValueError("GRU rollout actions are invalid.")
    if int(time_count) < 2:
        raise ValueError(f"GRU rollout requires at least two time points, got {time_count}.")

    current = torch.as_tensor(states, dtype=torch.float32)
    action = torch.as_tensor(actions[:, None], dtype=torch.float32)
    output = np.empty((states.shape[0], int(time_count), states.shape[1]), dtype=np.float32)
    output[:, 0, :] = states
    hidden: torch.Tensor | None = None
    with torch.no_grad():
        for time_index in range(1, int(time_count)):
            sequence = torch.cat((current, action), dim=1).unsqueeze(1)
            encoded, hidden = model.recurrent(sequence, hidden)
            current = model.readout(encoded[:, -1, :])
            output[:, time_index, :] = current.detach().cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("GRU rollout produced NaN or Inf values.")
    return output.astype(np.float64, copy=False)


def prediction_sha256(prediction: np.ndarray) -> str:
    """Hash a finite prediction tensor without persisting challenge predictions."""

    array = np.ascontiguousarray(np.asarray(prediction, dtype=np.float32))
    if not np.isfinite(array).all():
        raise ValueError("Prediction hash requires finite values.")
    return hashlib.sha256(array.tobytes()).hexdigest()


def mean_squared_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return a strict finite MSE for a shape-matched validation-only rollout."""

    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    if predicted.shape != observed.shape:
        raise ValueError(f"Prediction shape {predicted.shape} does not match target shape {observed.shape}.")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("MSE requires finite prediction and target tensors.")
    return float(np.mean((predicted - observed) ** 2))
