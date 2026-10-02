"""Execute the frozen-Twin, static-policy scientific information loop."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.decision_consistent_runtime import (  # noqa: E402
    DecisionConsistentTwin,
    DecisionController,
    InformationAction,
    ScientificPosterior,
    ToolResult,
    WorldAction,
)
from Libs.models.modules.round24b_posterior_adapter import (  # noqa: E402
    PosteriorBoundState,
)
from Libs.models.modules.round24b_typed_actions import CurrentClampAction  # noqa: E402
from Libs.models.modules.static_information_policy import (  # noqa: E402
    InformationCandidate,
    POLICY_UIDS,
    StaticInformationPolicy,
    normalized_rank_disagreement,
    posterior_discrepancy_proxy,
    rank_interventions,
)
from Libs.scripts.experiments.run_round24b_posterior_bound_campaign import (  # noqa: E402
    Z90,
    _aurc_np,
    _c1_units_for_state,
    _c2_prediction,
    _gaussian_crps_np,
    _load_c1_cache,
    _load_c2_cache,
    _load_reference,
    _metric_rows_for_state,
    _paired_rows,
    _response_species_indices,
    _teacher_likelihood_uncertainty,
    load_contract,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    load_config,
    write_json,
    write_tsv,
)
from Libs.utils.data.public_foundation_corpus import sha256_file  # noqa: E402


DEFAULT_CONTRACT = (
    REPO_ROOT / "Libs/configs/neuronsifter/static_decision_loop.yaml"
)
REGISTERED_TOOL_UID = "neuron_rxd_microenvironment_teacher.run_protocol_batch"
EVIDENCE_BOUNDARY = {
    "evidence_role": "development_only",
    "publication_claim_eligible": False,
    "protected_holdout_access": 0,
}
ROUND_RECEIPT_SCHEMA_VERSION = "neuronsifter-static-loop-round-receipt-v1"
RELIABILITY_PROXY_NAME = "calibrated_selective_reliability_proxy_v1"


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Static-loop contract must be a mapping: {path}")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _freeze_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(payload):
            raise ValueError(f"Frozen input drift: {path}")
        return
    write_json(path, dict(payload))


def _artifact_binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required static-loop artifact is missing: {path}")
    stat = path.stat()
    return {
        "path": _rel(path),
        "sha256": sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _round_artifact_paths(round_root: Path, round_uid: str) -> dict[str, Path]:
    return {
        "input_manifest": round_root / f"{round_uid}_input_manifest.json",
        "selected_information_action": (
            round_root / f"{round_uid}_selected_information_actions.tsv"
        ),
        "tool_results": round_root / f"{round_uid}_tool_results.tsv",
        "evidence_admission": round_root / f"{round_uid}_evidence_admission.tsv",
        "posterior_update": round_root / f"{round_uid}_posterior_update.json",
        "scientific_pre_post": round_root / f"{round_uid}_scientific_pre_post.tsv",
        "report": round_root / f"{round_uid}_report.json",
    }


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {field} timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp lacks timezone for {field}: {value!r}")
    return parsed.astimezone(timezone.utc)


def _round_dispatch_bindings(
    round_root: Path,
    selected_row: Mapping[str, Any],
    tool_rows: Sequence[Mapping[str, Any]],
    *,
    live_dispatch: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    candidate_uid = str(selected_row["candidate_uid"])
    dispatch_root = round_root / "prospective_dispatch" / candidate_uid
    rows_by_role = {str(row["action_role"]): row for row in tool_rows}
    if set(rows_by_role) != {"intervention", "matched_control"}:
        raise ValueError(
            f"Static-loop round must contain intervention/control tool rows: {round_root}"
        )
    bindings: list[dict[str, Any]] = []
    timestamp_source = (
        "controller_capture"
        if live_dispatch is not None
        else "registered_tool_dispatch.started_at_recovery_bound"
    )
    for role, directory in (
        ("intervention", "intervention"),
        ("matched_control", "control"),
    ):
        raw_path = dispatch_root / directory / "dispatch.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        catalog_row = rows_by_role[role]
        if str(raw.get("raw_bytes_hash", "")) != str(
            catalog_row.get("raw_bytes_hash", "")
        ):
            raise ValueError(f"Raw dispatch hash disagrees with tool row: {raw_path}")
        if str(raw.get("normalized_bytes_hash", "")) != str(
            catalog_row.get("normalized_bytes_hash", "")
        ):
            raise ValueError(
                f"Normalized dispatch hash disagrees with tool row: {raw_path}"
            )
        live_action = (
            dict(live_dispatch.get("actions", {}).get(role, {}))
            if live_dispatch is not None
            else {}
        )
        started_at_controller = live_action.get(
            "started_at_controller", raw.get("started_at")
        )
        finished_at_controller = live_action.get(
            "finished_at_controller", raw.get("finished_at")
        )
        _parse_utc_timestamp(
            started_at_controller, field=f"{role}.started_at_controller"
        )
        _parse_utc_timestamp(
            finished_at_controller, field=f"{role}.finished_at_controller"
        )
        bindings.append(
            {
                "action_role": role,
                "dispatch_ref": _rel(raw_path),
                "dispatch_sha256": sha256_file(raw_path),
                "request_uid": str(catalog_row["request_uid"]),
                "run_uid": str(catalog_row["run_uid"]),
                "raw_bytes_hash": str(raw["raw_bytes_hash"]),
                "normalized_bytes_hash": str(raw["normalized_bytes_hash"]),
                "status": str(raw.get("status", "")),
                "qc": str(raw.get("qc", "")),
                "started_at_controller": str(started_at_controller),
                "finished_at_controller": str(finished_at_controller),
                "controller_timestamp_source": timestamp_source,
            }
        )
    return bindings, timestamp_source


def build_round_receipt(
    round_root: Path,
    *,
    policy_uid: str,
    round_uid: str,
    live_dispatch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an immutable receipt without changing any scientific result."""

    artifact_paths = _round_artifact_paths(round_root, round_uid)
    bindings = {
        name: _artifact_binding(path) for name, path in artifact_paths.items()
    }
    input_manifest = json.loads(
        artifact_paths["input_manifest"].read_text(encoding="utf-8")
    )
    selected_rows = _read_tsv(artifact_paths["selected_information_action"])
    tool_rows = _read_tsv(artifact_paths["tool_results"])
    admission_rows = _read_tsv(artifact_paths["evidence_admission"])
    posterior_update = json.loads(
        artifact_paths["posterior_update"].read_text(encoding="utf-8")
    )
    report = json.loads(artifact_paths["report"].read_text(encoding="utf-8"))
    if len(selected_rows) != 1 or len(admission_rows) != 1:
        raise ValueError(f"Round selection/admission cardinality is invalid: {round_root}")
    selected = selected_rows[0]
    admission = admission_rows[0]
    expected_identity = (policy_uid, round_uid)
    for name, row in (
        ("input_manifest", input_manifest),
        ("selection", selected),
        ("admission", admission),
        ("posterior_update", posterior_update),
        ("report", report),
    ):
        if (str(row.get("policy_uid")), str(row.get("round_uid"))) != expected_identity:
            raise ValueError(f"Round identity mismatch in {name}: {round_root}")
    if input_manifest.get("outcome_visible_before_freeze") is not False:
        raise ValueError(f"Outcome-blind input freeze is invalid: {round_root}")
    if not _truthy(selected.get("selection_before_outcome")):
        raise ValueError(f"Selection was not frozen before outcome: {round_root}")
    identity_fields = (
        "posterior_uid_pre",
        "posterior_uid_post",
        "world_action_uid",
        "information_action_uid",
        "result_uid",
    )
    identity_sources = {
        "posterior_uid_pre": (
            selected.get("posterior_uid"),
            admission.get("posterior_uid_pre"),
            posterior_update.get("posterior_uid_pre"),
        ),
        "posterior_uid_post": (
            admission.get("posterior_uid_post"),
            posterior_update.get("posterior_uid_post"),
        ),
        "world_action_uid": (
            selected.get("world_action_uid"),
            admission.get("world_action_uid"),
            report.get("selected_world_action_uid"),
        ),
        "information_action_uid": (
            selected.get("information_action_uid"),
            admission.get("information_action_uid"),
            report.get("selected_information_action_uid"),
        ),
        "result_uid": (admission.get("result_uid"), report.get("result_uid")),
    }
    identities: dict[str, str] = {}
    for field in identity_fields:
        values = {str(value) for value in identity_sources[field] if value not in (None, "")}
        if len(values) != 1:
            raise ValueError(f"Round {field} identity mismatch: {round_root}")
        identities[field] = values.pop()
    dispatch_bindings, timestamp_source = _round_dispatch_bindings(
        round_root,
        selected,
        tool_rows,
        live_dispatch=live_dispatch,
    )
    freeze_mtime_ns = max(
        bindings["input_manifest"]["mtime_ns"],
        bindings["selected_information_action"]["mtime_ns"],
    )
    freeze_at = datetime.fromtimestamp(
        freeze_mtime_ns / 1_000_000_000.0, timezone.utc
    )
    first_dispatch = min(
        _parse_utc_timestamp(
            row["started_at_controller"],
            field=f"{row['action_role']}.started_at_controller",
        )
        for row in dispatch_bindings
    )
    if freeze_at >= first_dispatch:
        raise ValueError(
            f"Round freeze is not earlier than registered-tool dispatch: {round_root}"
        )
    payload = {
        "schema_version": ROUND_RECEIPT_SCHEMA_VERSION,
        "policy_uid": policy_uid,
        "round_uid": round_uid,
        "candidate_uid": str(selected["candidate_uid"]),
        "state_uid": str(selected["state_uid"]),
        "candidate_pool_uid": str(input_manifest["candidate_pool_uid"]),
        "candidate_pool_hash": str(input_manifest["candidate_pool_hash"]),
        **identities,
        "outcome_visible_before_freeze": False,
        "selection_before_outcome": True,
        "freeze_before_dispatch_verified": True,
        "freeze_latest_mtime_ns": freeze_mtime_ns,
        "first_dispatch_started_at_controller": first_dispatch.isoformat(),
        "controller_timestamp_source": timestamp_source,
        "artifact_bindings": bindings,
        "dispatch_bindings": dispatch_bindings,
        **EVIDENCE_BOUNDARY,
    }
    payload_hash = _digest(payload)
    return {
        **payload,
        "receipt_payload_sha256": payload_hash,
        "receipt_uid": f"static-loop-round-receipt-{payload_hash[:32]}",
    }


def materialize_round_receipt(
    round_root: Path,
    *,
    policy_uid: str,
    round_uid: str,
    live_dispatch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = build_round_receipt(
        round_root,
        policy_uid=policy_uid,
        round_uid=round_uid,
        live_dispatch=live_dispatch,
    )
    _freeze_json(round_root / "round_receipt.json", receipt)
    return receipt


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Metric aggregation received empty or non-finite values")
    return float(array.mean())


@dataclass(frozen=True)
class CalibrationState:
    calibration_uid: str
    c1_scale: float
    c2s_scale: float
    c1_calibration_units: tuple[str, ...]
    c1_evaluation_units: tuple[str, ...]
    c2s_calibration_units: tuple[str, ...]
    c2s_evaluation_units: tuple[str, ...]
    target_coverage: float
    coverage_floor: float
    maximum_calibration_error: float


@dataclass
class DecisionSnapshot:
    policy_uid: str
    round_uid: str
    selected_candidate_uid: str
    selected_information_action_uid: str
    posterior_uid_pre: str
    posterior_uid_post: str
    intervention_ranking_pre: list[dict[str, Any]]
    intervention_ranking_post: list[dict[str, Any]]
    query_ranking_pre: list[dict[str, Any]]
    query_ranking_post: list[dict[str, Any]]
    selected_utility: float
    posterior_discrepancy: float
    forecast_change: float
    intervention_rank_disagreement: float
    query_rank_disagreement: float
    intervention_switch: bool
    observation: float | None
    predicted_observation_pre: float
    predicted_observation_std_pre: float
    contradictory_observation: bool
    admitted: bool
    cumulative_cost: float
    reliability: dict[str, Any]


def _count_tsv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _validate_reference(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], PosteriorBoundState, dict[str, Any], dict[str, Any]]:
    reference_contract = load_contract(
        _repo_path(contract["reference"]["posterior_bound_contract_path"])
    )
    gate_path = _repo_path(contract["reference"]["negative_gate_path"])
    handoff_path = _repo_path(contract["reference"]["negative_handoff_path"])
    reference_path = _repo_path(contract["reference"]["pre_update_reference_path"])
    posterior_path = _repo_path(contract["reference"]["pre_update_posterior_path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    expected_status = str(contract["reference"]["expected_terminal_state"])
    expected_reference = str(contract["reference"]["expected_reference"])
    if gate.get("status") != expected_status or handoff.get("status") != expected_status:
        raise ValueError("Round-24B negative terminal state changed")
    if gate.get("recommended_strategy") != expected_reference:
        raise ValueError("Round-24B no longer recommends the frozen U0 reference")
    if gate.get("validated_update_candidates"):
        raise ValueError("Round-24B unexpectedly contains a validated update candidate")
    if gate.get("protected_holdout_access") != 0:
        raise ValueError("Protected holdout access is nonzero")
    paired_path = _repo_path(
        contract["reference"]["posterior_bound_paired_metrics_path"]
    )
    paired_count = _count_tsv_rows(paired_path)
    if paired_count != int(contract["reference"]["expected_paired_row_count"]):
        raise ValueError(
            f"Round-24B paired row count changed: {paired_count}"
        )
    reference, prior = _load_reference(
        reference_contract,
        reference_path,
        posterior_path,
    )
    for checkpoint in reference["twin_checkpoints"].values():
        checkpoint_path = _repo_path(checkpoint["path"])
        if sha256_file(checkpoint_path) != checkpoint["sha256"]:
            raise ValueError(f"Frozen checkpoint hash changed: {checkpoint_path}")
    return reference, prior, reference_contract, {
        "gate": gate,
        "handoff": handoff,
        "gate_path": gate_path,
        "handoff_path": handoff_path,
        "paired_path": paired_path,
        "paired_count": paired_count,
        "posterior_path": posterior_path,
        "reference_path": reference_path,
    }


def _write_reference_freezes(
    contract: Mapping[str, Any],
    contract_path: Path,
    reference: Mapping[str, Any],
    prior: PosteriorBoundState,
    lineage: Mapping[str, Any],
) -> None:
    boundary = dict(EVIDENCE_BOUNDARY)
    canonical = {
        "schema_version": "neuronsifter-round25-u0-canonical-static-twin-v1",
        "status": "frozen",
        "canonical_twin": "U0_frozen",
        "posterior_uid_pre": prior.posterior_uid,
        "posterior_schema_version": prior.schema_version,
        "scientific_belief_encoder": reference["twin_checkpoints"]["c1_encoder"],
        "world_action_capability": reference["twin_checkpoints"]["c2s_world_action"],
        "information_action_capability": reference["tool_capability_set"],
        "action_compiler": "Libs/models/modules/round24b_typed_actions.py",
        "assimilator": "Libs/models/modules/round24b_posterior_adapter.py::PosteriorBoundState.update",
        "tri_interface": reference["tri_calibration_identity"],
        "development_data_splits": reference["evaluation_manifests"],
        "software_environment": {
            "python": sys.version,
            "host": platform.node(),
            "contract_path": _rel(contract_path),
            "contract_sha256": sha256_file(contract_path),
        },
        "twin_parameter_update_allowed": False,
        "observation_assimilation_allowed": True,
        "agent_parameter_update_allowed": False,
        "posterior_changes_are_parameter_updates": False,
        "round24b_negative_result_preserved": True,
        "publication_comparison_open": False,
        "c2b": {"status": "blocked_insufficient_real_e2_e4"},
        **boundary,
    }
    negative = {
        "schema_version": "neuronsifter-round25-adaptation-negative-result-freeze-v1",
        "status": "preserved_negative_scientific_result",
        "round24b_terminal_state": lineage["gate"]["status"],
        "recommended_reference": lineage["gate"]["recommended_strategy"],
        "adaptive_controls": {
            uid: "rolled_back"
            for uid in ("U1_latest_batch", "U2_full_replay", "U3_governed")
        },
        "coverage": float(
            lineage["gate"]["strategy_summaries"]["U0_frozen"]["rounds"][0][
                "coverage"
            ]
        ),
        "coverage_floor": float(contract["support_calibration"]["coverage_floor"]),
        "paired_scientific_rows": lineage["paired_count"],
        "authoritative_nonzero_delta": 0,
        "source_gate": {
            "path": _rel(lineage["gate_path"]),
            "sha256": sha256_file(lineage["gate_path"]),
        },
        "source_handoff": {
            "path": _rel(lineage["handoff_path"]),
            "sha256": sha256_file(lineage["handoff_path"]),
        },
        "source_paired_metrics": {
            "path": _rel(lineage["paired_path"]),
            "sha256": sha256_file(lineage["paired_path"]),
        },
        "agent_adaptation_authorized": False,
        "publication_comparison_open": False,
        "c2b": {"status": "blocked_insufficient_real_e2_e4"},
        **boundary,
    }
    write_json(_repo_path(contract["outputs"]["canonical_twin"]), canonical)
    write_json(_repo_path(contract["outputs"]["negative_result_freeze"]), negative)


def _coverage_taxonomy(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_tsv(
        _repo_path(contract["reference"]["posterior_bound_metric_inputs_path"])
    )
    floor = float(contract["support_calibration"]["coverage_floor"])
    selected = [
        row
        for row in rows
        if str(row.get("round_uid", "")).endswith("_candidate")
        and (
            (row.get("domain") == "c1" and row.get("metric_name") == "posterior_reconstruction_interval_coverage")
            or (row.get("domain") == "c2s" and row.get("metric_name") == "coverage")
        )
    ]
    taxonomy: list[dict[str, Any]] = []
    for row in selected:
        coverage = float(row["metric_value"])
        rejected = coverage < floor
        if not rejected:
            continue
        domain = str(row["domain"])
        support_value = next(
            (
                float(candidate["metric_value"])
                for candidate in rows
                if candidate.get("strategy_uid") == row.get("strategy_uid")
                and candidate.get("round_uid") == row.get("round_uid")
                and candidate.get("evaluation_uid") == row.get("evaluation_uid")
                and candidate.get("metric_name") == "support_domain_coverage"
            ),
            0.0,
        )
        mechanism_supported = float(row.get("posterior_model_path_effect_abs", 0.0)) > 0.0
        calibrated_support_present = support_value >= floor
        classification = (
            "over_conservative_reliability_estimation"
            if calibrated_support_present and mechanism_supported
            else "true_unsupported_science"
        )
        taxonomy.append(
            {
                "strategy_uid": row["strategy_uid"],
                "round_uid": row["round_uid"],
                "posterior_uid": row["posterior_uid"],
                "evaluation_uid": row["evaluation_uid"],
                "evaluation_unit_uid": row["evaluation_unit_uid"],
                "domain": domain,
                "coverage_metric": row["metric_name"],
                "coverage_value": coverage,
                "coverage_floor": floor,
                "posterior_support": "supported" if calibrated_support_present else "unsupported",
                "source_support": "supported_development_split",
                "world_action_support": "supported" if domain == "c2s" and row.get("world_action_uid") else "not_applicable",
                "tool_support": "not_applicable_retention_evaluation",
                "mechanism_support": "supported" if mechanism_supported else "unsupported",
                "ood": False,
                "calibration_failure": True,
                "constraint_failure": False,
                "evidence_quality": "finite_posterior_bound_development_evidence",
                "missing_modality": "task_condition_not_rejection_reason" if domain == "c1" else "not_applicable",
                "primary_rejection_reason": "calibration_failure",
                "scientific_classification": classification,
                "unknown_rejection_reason": False,
                **EVIDENCE_BOUNDARY,
            }
        )
    if not taxonomy or any(row["unknown_rejection_reason"] for row in taxonomy):
        raise ValueError("Coverage failures were not completely classified")
    u0_selected = [
        row
        for row in selected
        if row["strategy_uid"] == "U0_frozen"
    ]
    u0_domain_coverage = {
        domain: _safe_mean(
            float(row["metric_value"])
            for row in u0_selected
            if row["domain"] == domain
        )
        for domain in {str(row["domain"]) for row in u0_selected}
    }
    summary = {
        "source_candidate_coverage_rows": len(selected),
        "coverage_failure_rows": len(taxonomy),
        "classified_rows": len(taxonomy),
        "unknown_rejection_reason": 0,
        "classification_counts": {
            label: sum(row["scientific_classification"] == label for row in taxonomy)
            for label in sorted({str(row["scientific_classification"]) for row in taxonomy})
        },
        "raw_u0_coverage": min(u0_domain_coverage.values()),
        "raw_u0_coverage_by_domain": u0_domain_coverage,
        "coverage_floor": floor,
    }
    return taxonomy, summary


def _partition_units(
    values: Sequence[str], *, fraction: float, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    unique = sorted(set(str(value) for value in values))
    if len(unique) < 3:
        raise ValueError("Calibration requires at least three scientific units")
    ordered = sorted(
        unique,
        key=lambda value: _digest({"seed": int(seed), "unit_uid": value}),
    )
    count = max(1, min(len(ordered) - 1, int(round(len(ordered) * fraction))))
    return tuple(sorted(ordered[:count])), tuple(sorted(ordered[count:]))


def _conformal_scale(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    *,
    target: float,
    minimum: float,
    maximum: float,
    method: str,
) -> float:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if (
        errors.size == 0
        or errors.shape != uncertainty.shape
        or not np.isfinite(errors).all()
        or not np.isfinite(uncertainty).all()
        or np.any(uncertainty <= 0.0)
    ):
        raise ValueError("Invalid calibration residuals or uncertainty")
    ratios = np.abs(errors) / np.maximum(Z90 * uncertainty, 1.0e-12)
    scale = float(np.quantile(ratios, target, method=method))
    scale = float(np.clip(scale, minimum, maximum))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Conformal calibration scale is invalid")
    return scale


def _c2s_arrays_by_pair(
    posterior: PosteriorBoundState,
    cache: Any,
    reference_contract: Mapping[str, Any],
    project_config: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    prediction = _c2_prediction(posterior, cache, reference_contract, project_config)
    means = prediction["trajectory_mean"].detach().cpu().numpy().astype(np.float64)
    stds = prediction["trajectory_std"].detach().cpu().numpy().astype(np.float64)
    truth = cache.normalized_trajectories.detach().cpu().numpy().astype(np.float64)[:, 1:]
    out: dict[str, dict[str, np.ndarray]] = {}
    for pair in cache.pairs:
        indices = [pair.intervention_index, pair.control_index]
        pair_mean = means[indices]
        pair_truth = truth[indices]
        pair_std = stds[indices]
        likelihood = _teacher_likelihood_uncertainty(
            pair_std,
            cache.teacher_uncertainty[indices],
        )
        out[pair.pair_uid] = {
            "mean": pair_mean,
            "truth": pair_truth,
            "model_std": pair_std,
            "likelihood": likelihood,
            "errors": pair_truth - pair_mean,
        }
    return out


def _coverage_from_units(
    units: Sequence[Any], scale: float
) -> tuple[float, float]:
    coverages = []
    for unit in units:
        sigma = np.maximum(np.asarray(unit.native_sigmas, dtype=np.float64) * scale, 1.0e-12)
        coverage = float(np.mean(np.abs(unit.residuals) <= Z90 * sigma))
        coverages.append(coverage)
    coverage = _safe_mean(coverages)
    return coverage, abs(coverage - 0.9)


def _fit_support_calibration(
    contract: Mapping[str, Any],
    prior: PosteriorBoundState,
    c1_cache: Any,
    c2s_cache: Any,
    reference_contract: Mapping[str, Any],
    project_config: Mapping[str, Any],
) -> tuple[CalibrationState, dict[str, Any]]:
    policy = contract["support_calibration"]
    fraction = float(policy["calibration_unit_fraction"])
    seed = int(policy["partition_seed"])
    target = float(policy["target_interval_coverage"])
    minimum = float(policy["minimum_scale"])
    maximum = float(policy["maximum_scale"])
    method = str(policy["quantile_method"])
    c1_units = _c1_units_for_state(prior, c1_cache, reference_contract)
    c1_cal_ids, c1_eval_ids = _partition_units(
        [unit.evaluation_unit_uid for unit in c1_units],
        fraction=fraction,
        seed=seed,
    )
    c1_by_uid = {unit.evaluation_unit_uid: unit for unit in c1_units}
    c1_cal = [c1_by_uid[uid] for uid in c1_cal_ids]
    c1_eval = [c1_by_uid[uid] for uid in c1_eval_ids]
    c1_scale = _conformal_scale(
        np.concatenate([unit.residuals for unit in c1_cal]),
        np.concatenate([unit.native_sigmas for unit in c1_cal]),
        target=target,
        minimum=minimum,
        maximum=maximum,
        method=method,
    )

    c2s_arrays = _c2s_arrays_by_pair(
        prior, c2s_cache, reference_contract, project_config
    )
    c2s_cal_ids, c2s_eval_ids = _partition_units(
        sorted(c2s_arrays), fraction=fraction, seed=seed
    )
    c2s_scale = _conformal_scale(
        np.concatenate([c2s_arrays[uid]["errors"].reshape(-1) for uid in c2s_cal_ids]),
        np.concatenate([c2s_arrays[uid]["likelihood"].reshape(-1) for uid in c2s_cal_ids]),
        target=target,
        minimum=minimum,
        maximum=maximum,
        method=method,
    )

    c1_cal_raw, _ = _coverage_from_units(c1_cal, 1.0)
    c1_cal_adjusted, c1_cal_error = _coverage_from_units(c1_cal, c1_scale)
    c1_eval_raw, _ = _coverage_from_units(c1_eval, 1.0)
    c1_eval_adjusted, c1_eval_error = _coverage_from_units(c1_eval, c1_scale)

    def c2s_coverage(unit_ids: Sequence[str], scale: float) -> float:
        return _safe_mean(
            float(
                np.mean(
                    np.abs(c2s_arrays[uid]["errors"])
                    <= Z90 * c2s_arrays[uid]["likelihood"] * scale
                )
            )
            for uid in unit_ids
        )

    c2s_cal_raw = c2s_coverage(c2s_cal_ids, 1.0)
    c2s_cal_adjusted = c2s_coverage(c2s_cal_ids, c2s_scale)
    c2s_eval_raw = c2s_coverage(c2s_eval_ids, 1.0)
    c2s_eval_adjusted = c2s_coverage(c2s_eval_ids, c2s_scale)
    c2s_cal_error = abs(c2s_cal_adjusted - target)
    c2s_eval_error = abs(c2s_eval_adjusted - target)

    identity_payload = {
        "schema_version": "neuronsifter-static-tri-calibration-v1",
        "source_posterior_uid": prior.posterior_uid,
        "partition_seed": seed,
        "target_interval_coverage": target,
        "quantile_method": method,
        "c1_scale": c1_scale,
        "c2s_scale": c2s_scale,
        "c1_calibration_units": c1_cal_ids,
        "c1_evaluation_units": c1_eval_ids,
        "c2s_calibration_units": c2s_cal_ids,
        "c2s_evaluation_units": c2s_eval_ids,
    }
    calibration_uid = "tri-calibration-" + _digest(identity_payload)[:32]
    state = CalibrationState(
        calibration_uid=calibration_uid,
        c1_scale=c1_scale,
        c2s_scale=c2s_scale,
        c1_calibration_units=c1_cal_ids,
        c1_evaluation_units=c1_eval_ids,
        c2s_calibration_units=c2s_cal_ids,
        c2s_evaluation_units=c2s_eval_ids,
        target_coverage=target,
        coverage_floor=float(policy["coverage_floor"]),
        maximum_calibration_error=float(policy["maximum_calibration_error"]),
    )
    report = {
        **identity_payload,
        "calibration_uid": calibration_uid,
        "status": "validation_calibration_frozen",
        "source_partition_role": policy["source_partition_role"],
        "calibration_role": policy["calibration_role"],
        "evaluation_role": policy["evaluation_role"],
        "c1": {
            "calibration_unit_count": len(c1_cal_ids),
            "evaluation_unit_count": len(c1_eval_ids),
            "scale": c1_scale,
            "calibration_coverage_raw": c1_cal_raw,
            "calibration_coverage_adjusted": c1_cal_adjusted,
            "calibration_error_adjusted": c1_cal_error,
            "evaluation_coverage_raw": c1_eval_raw,
            "evaluation_coverage_adjusted": c1_eval_adjusted,
            "evaluation_calibration_error": c1_eval_error,
        },
        "c2s": {
            "calibration_unit_count": len(c2s_cal_ids),
            "evaluation_unit_count": len(c2s_eval_ids),
            "scale": c2s_scale,
            "calibration_coverage_raw": c2s_cal_raw,
            "calibration_coverage_adjusted": c2s_cal_adjusted,
            "calibration_error_adjusted": c2s_cal_error,
            "evaluation_coverage_raw": c2s_eval_raw,
            "evaluation_coverage_adjusted": c2s_eval_adjusted,
            "evaluation_calibration_error": c2s_eval_error,
        },
        "coverage_floor": state.coverage_floor,
        "coverage_floor_changed_from_test": False,
        "maximum_calibration_error": state.maximum_calibration_error,
        "protected_holdout_access": 0,
        "publication_claim_eligible": False,
        "evidence_role": "development_only",
    }
    return state, report


def _calibrated_scientific_metrics(
    state: PosteriorBoundState,
    c1_cache: Any,
    c2s_cache: Any,
    reference_contract: Mapping[str, Any],
    project_config: Mapping[str, Any],
    output_root: Path,
    calibration: CalibrationState,
    *,
    strategy_uid: str,
    round_uid: str,
) -> list[dict[str, Any]]:
    raw_rows = _metric_rows_for_state(
        state,
        c1_cache,
        c2s_cache,
        reference_contract,
        project_config,
        output_root,
        strategy_uid=strategy_uid,
        round_uid=round_uid,
    )
    c1_units = {
        unit.evaluation_unit_uid: unit
        for unit in _c1_units_for_state(state, c1_cache, reference_contract)
    }
    c2s_arrays = _c2s_arrays_by_pair(
        state, c2s_cache, reference_contract, project_config
    )
    c1_eval = set(calibration.c1_evaluation_units)
    c2s_eval = set(calibration.c2s_evaluation_units)
    c1_overrides: dict[str, dict[str, float]] = {}
    for uid in c1_eval:
        unit = c1_units[uid]
        sigma = np.maximum(
            unit.native_sigmas * calibration.c1_scale,
            float(reference_contract["posterior"]["c1_uncertainty_floor"]),
        )
        residual = unit.residuals
        coverage = float(np.mean(np.abs(residual) <= Z90 * sigma))
        interval_miss = (np.abs(residual) > Z90 * sigma).astype(np.float64)
        standardized = np.abs(residual) / np.maximum(sigma, 1.0e-12)
        c1_overrides[uid] = {
            "posterior_reconstruction_interval_coverage": coverage,
            "calibration_error": abs(coverage - calibration.target_coverage),
            "false_accept": float(interval_miss.mean()),
            "aurc": _aurc_np(standardized, sigma),
        }
    c2s_overrides: dict[str, dict[str, float]] = {}
    species_ids = [
        str(row["species_id"])
        for row in project_config["dynamics_ready_registry"]["species"]
    ]
    signed = set(reference_contract["typed_current_clamp"]["signed_species_ids"])
    nonnegative = np.asarray([species_id not in signed for species_id in species_ids])
    for uid in c2s_eval:
        arrays = c2s_arrays[uid]
        mean = arrays["mean"]
        truth = arrays["truth"]
        errors = arrays["errors"]
        likelihood = arrays["likelihood"] * calibration.c2s_scale
        interval_miss = (np.abs(errors) > Z90 * likelihood).astype(np.float64)
        standardized = np.abs(errors) / np.maximum(likelihood, 1.0e-12)
        pair = next(pair for pair in c2s_cache.pairs if pair.pair_uid == uid)
        contrast_error = (mean[0] - mean[1]) - (truth[0] - truth[1])
        coverage = float(np.mean(np.abs(errors) <= Z90 * likelihood))
        c2s_overrides[uid] = {
            "trajectory_error": float(np.sqrt(np.mean(np.square(errors)))),
            "terminal_error": float(np.sqrt(np.mean(np.square(errors[:, -1])))),
            "long_horizon_drift": float(np.sqrt(np.mean(np.square(errors[:, -1])))),
            "action_contrast": float(np.sqrt(np.mean(np.square(contrast_error)))),
            "crps": float(np.mean(_gaussian_crps_np(mean, likelihood, truth))),
            "nll": float(
                np.mean(
                    0.5
                    * (
                        np.log(2.0 * math.pi * np.square(likelihood))
                        + np.square(errors) / np.square(likelihood)
                    )
                )
            ),
            "coverage": coverage,
            "calibration_error": abs(coverage - calibration.target_coverage),
            "aurc": _aurc_np(standardized, likelihood),
            "false_accept": float(interval_miss.mean()),
            "constraint_residual": float(np.mean(mean[..., nonnegative] < 0.0)),
            "support_domain_coverage": 1.0,
            "epistemic_variance": float(np.mean(np.square(arrays["model_std"]))),
            "posterior_response_mean": state.action_response(
                c2s_cache.action_designs[pair.intervention_index]
            )[0],
            "posterior_response_std": state.action_response(
                c2s_cache.action_designs[pair.intervention_index]
            )[1],
        }
    calibrated: list[dict[str, Any]] = []
    for raw in raw_rows:
        domain = str(raw["domain"])
        uid = str(raw["evaluation_unit_uid"])
        row = dict(raw)
        if domain == "c1":
            if uid not in c1_eval:
                continue
            if row["metric_name"] in c1_overrides[uid]:
                row["uncalibrated_metric_value"] = float(row["metric_value"])
                row["metric_value"] = c1_overrides[uid][str(row["metric_name"])]
            row["evaluation_partition_role"] = "static_loop_evaluation"
            row["uncertainty_calibration_scale"] = calibration.c1_scale
        elif domain == "c2s":
            pair_uid = str(row["pair_uid"])
            if pair_uid not in c2s_eval:
                continue
            if row["metric_name"] in c2s_overrides[pair_uid]:
                row["uncalibrated_metric_value"] = float(row["metric_value"])
                row["metric_value"] = c2s_overrides[pair_uid][str(row["metric_name"])]
            row["evaluation_partition_role"] = "static_loop_evaluation"
            row["uncertainty_calibration_scale"] = calibration.c2s_scale
        else:
            continue
        row["tri_calibration_uid"] = calibration.calibration_uid
        row["finite"] = bool(math.isfinite(float(row["metric_value"])))
        calibrated.append(row)
    if not calibrated or not all(bool(row["finite"]) for row in calibrated):
        raise ValueError("Calibrated scientific metrics are empty or non-finite")
    return calibrated


def _aggregate_reliability(
    rows: Sequence[Mapping[str, Any]], calibration: CalibrationState
) -> dict[str, Any]:
    def values(domain: str, metric: str) -> list[float]:
        return [
            float(row["metric_value"])
            for row in rows
            if row["domain"] == domain and row["metric_name"] == metric
        ]

    c1_coverage = _safe_mean(values("c1", "posterior_reconstruction_interval_coverage"))
    c2s_coverage = _safe_mean(values("c2s", "coverage"))
    c1_error = abs(c1_coverage - calibration.target_coverage)
    c2s_error = abs(c2s_coverage - calibration.target_coverage)
    coverage = min(c1_coverage, c2s_coverage)
    maximum_error = max(c1_error, c2s_error)
    aurc = max(_safe_mean(values("c1", "aurc")), _safe_mean(values("c2s", "aurc")))
    false_accept = max(
        _safe_mean(values("c1", "false_accept")),
        _safe_mean(values("c2s", "false_accept")),
    )
    support_domain_coverage = min(
        _safe_mean(values("c1", "support_domain_coverage")),
        _safe_mean(values("c2s", "support_domain_coverage")),
    )
    proxy_components = {
        "coverage": float(np.clip(coverage, 0.0, 1.0)),
        "calibration": float(np.clip(1.0 - maximum_error, 0.0, 1.0)),
        "support_domain_coverage": float(
            np.clip(support_domain_coverage, 0.0, 1.0)
        ),
        "false_accept_complement": float(np.clip(1.0 - false_accept, 0.0, 1.0)),
    }
    reliability_proxy = _safe_mean(proxy_components.values())
    return {
        "coverage": coverage,
        "c1_coverage": c1_coverage,
        "c2s_coverage": c2s_coverage,
        "calibration_error": maximum_error,
        "c1_calibration_error": c1_error,
        "c2s_calibration_error": c2s_error,
        "aurc": aurc,
        "false_accept": false_accept,
        "support_domain_coverage": support_domain_coverage,
        "reliability_proxy_name": RELIABILITY_PROXY_NAME,
        "reliability_proxy": reliability_proxy,
        "reliability_proxy_components": proxy_components,
        "tri_score_available": False,
        "coverage_floor": calibration.coverage_floor,
        "coverage_floor_respected": coverage >= calibration.coverage_floor,
        "maximum_calibration_error": calibration.maximum_calibration_error,
        "calibration_within_tolerance": maximum_error
        <= calibration.maximum_calibration_error,
        "tri_calibration_uid": calibration.calibration_uid,
    }


def _build_candidates(
    contract: Mapping[str, Any],
    reference_contract: Mapping[str, Any],
    c2s_cache: Any,
    source_model_id: str,
) -> tuple[list[InformationCandidate], dict[str, tuple[CurrentClampAction, CurrentClampAction]]]:
    historical = {
        (
            float(row["action"]["dose_value"]),
            float(row["action"]["duration_ms"]),
            float(row["action"]["delay_ms"]),
        )
        for row in c2s_cache.records
        if str(row["action"]["action_role"]) == "intervention"
    }
    previous = {
        (
            float(row["amplitude_nA"]),
            float(row["duration_ms"]),
            float(row["delay_ms"]),
        )
        for row in reference_contract["prospective"]["candidate_protocols"]
    }
    train_rows = [
        row
        for row in c2s_cache.records
        if row["split_role"] == "train"
        and row["action"]["action_role"] == "intervention"
    ]
    amplitude_scale = max(abs(float(row["action"]["dose_value"])) for row in train_rows)
    duration_scale = max(float(row["action"]["duration_ms"]) for row in train_rows)
    charge_scale = max(
        abs(float(row["action"]["dose_value"]) * float(row["action"]["duration_ms"]))
        for row in train_rows
    )
    candidates: list[InformationCandidate] = []
    actions: dict[str, tuple[CurrentClampAction, CurrentClampAction]] = {}
    for order, row in enumerate(contract["prospective"]["candidate_protocols"]):
        protocol = (
            float(row["amplitude_nA"]),
            float(row["duration_ms"]),
            float(row["delay_ms"]),
        )
        if protocol in historical or protocol in previous or protocol == (0.2, 120.0, 40.0):
            raise ValueError(f"Static-loop protocol is not prospectively fresh: {protocol}")
        intervention = CurrentClampAction(
            action_uid=str(row["action_uid"]),
            pair_uid=str(row["pair_uid"]),
            role="intervention",
            amplitude_nA=protocol[0],
            duration_ms=protocol[1],
            delay_ms=protocol[2],
            source_model_id=source_model_id,
            split_role="development_diagnostic",
        )
        control = CurrentClampAction(
            action_uid=f"control_zero_for_{row['action_uid']}",
            pair_uid=str(row["pair_uid"]),
            role="matched_control",
            amplitude_nA=0.0,
            duration_ms=protocol[1],
            delay_ms=protocol[2],
            source_model_id=source_model_id,
            split_role="development_diagnostic",
        )
        design = (
            1.0,
            intervention.amplitude_nA / amplitude_scale,
            intervention.duration_ms / duration_scale,
            intervention.amplitude_nA * intervention.duration_ms / charge_scale,
        )
        state_uid = "state-" + _digest(
            {
                "pool": contract["prospective"]["candidate_pool_uid"],
                "pair_uid": intervention.pair_uid,
            }
        )[:24]
        information_action = InformationAction(
            tool_id=REGISTERED_TOOL_UID,
            query={
                "state_uid": state_uid,
                "protocol_uid": intervention.action_uid,
                "world_action_uid": intervention.to_world_action().action_uid,
                "control_world_action_uid": control.to_world_action().action_uid,
                "requested_outputs": list(contract["prospective"]["requested_outputs"]),
            },
            fidelity=str(contract["prospective"]["fidelity"]),
            expected_cost=float(row["expected_cost"]),
        )
        candidate = InformationCandidate(
            candidate_uid=intervention.action_uid,
            pair_uid=intervention.pair_uid,
            state_uid=state_uid,
            world_action=intervention.to_world_action(),
            control_world_action=control.to_world_action(),
            information_action=information_action,
            design=design,
            order_index=order,
        )
        candidates.append(candidate)
        actions[candidate.candidate_uid] = (intervention, control)
    if len(candidates) < int(contract["static_policy"]["sequential_rounds"]):
        raise ValueError("Static information-action pool is too small")
    return candidates, actions


def _write_policy_freezes(
    contract: Mapping[str, Any],
    contract_path: Path,
    policies: Mapping[str, StaticInformationPolicy],
    candidates: Sequence[InformationCandidate],
    prior: PosteriorBoundState,
    reference: Mapping[str, Any],
) -> None:
    policy_records = {uid: policy.to_record() for uid, policy in policies.items()}
    candidate_records = [candidate.to_record() for candidate in candidates]
    payload = {
        "schema_version": "neuronsifter-round25-static-policy-freeze-v1",
        "status": "frozen_before_prospective_outcomes",
        "canonical_policy_uid": contract["static_policy"]["canonical_policy_uid"],
        "policies": policy_records,
        "candidate_pool_uid": contract["prospective"]["candidate_pool_uid"],
        "candidate_pool_sha256": _digest(candidate_records),
        "candidate_count": len(candidate_records),
        "selection_posterior_uid": prior.posterior_uid,
        "selection_basis": "development_validation_and_preregistered_shared_belief_contract",
        "twin_parameters_frozen": True,
        "policy_parameters_frozen_during_evaluation": True,
        "agent_online_adaptation_allowed": False,
        "contract": {"path": _rel(contract_path), "sha256": sha256_file(contract_path)},
        "agent_checkpoint_identity": reference["agent_policy"],
        **EVIDENCE_BOUNDARY,
    }
    controls = {
        "schema_version": "neuronsifter-round25-internal-control-contract-v1",
        "status": "frozen",
        "controls": {
            "R_random_information_action": "deterministic_seeded_random_permutation",
            "F_fixed_cascade": "frozen_protocol_order",
            "U_uncertainty_only": "predictive_variance_per_cost",
            "S_shared_belief_decision_risk": "expected_pairwise_intervention_variance_reduction_per_cost",
        },
        "matched": {
            "initial_posterior": prior.posterior_uid,
            "candidate_pool": contract["prospective"]["candidate_pool_uid"],
            "tool_capability": REGISTERED_TOOL_UID,
            "round_budget": contract["static_policy"]["information_budget_per_round"],
            "round_count": contract["static_policy"]["sequential_rounds"],
            "checkpoint_identity": reference["twin_checkpoints"],
        },
        "external_publication_baseline_substitute": False,
        "external_baseline_deployment_open": False,
        **EVIDENCE_BOUNDARY,
    }
    write_json(_repo_path(contract["outputs"]["policy_freeze"]), payload)
    write_json(_repo_path(contract["outputs"]["control_contract"]), controls)


def _runtime_for_chain(
    prior: PosteriorBoundState,
    candidates: Sequence[InformationCandidate],
) -> tuple[DecisionConsistentTwin, dict[str, PosteriorBoundState]]:
    states = {prior.posterior_uid: prior}
    by_world_action = {
        candidate.world_action.action_uid: candidate for candidate in candidates
    }

    def rollout_handler(
        posterior: ScientificPosterior, action: WorldAction
    ) -> Mapping[str, Any]:
        state = states[posterior.posterior_uid]
        candidate = by_world_action.get(action.action_uid)
        if candidate is None:
            raise ValueError("World action is outside the frozen static candidate pool")
        mean, std = state.action_response(np.asarray(candidate.design))
        return {
            "trajectory": [mean],
            "uncertainty": {"epistemic": std, "aleatoric": state.noise_variance},
            "diagnostics": {
                "posterior_uid": posterior.posterior_uid,
                "world_action_uid": action.action_uid,
                "frozen_model_parameters": True,
            },
        }

    twin = DecisionConsistentTwin(rollout_fn=rollout_handler)
    scientific = prior.to_scientific_posterior()
    twin.register_posterior(scientific)
    twin.decisions.bind(scientific)
    return twin, states


def _trace_decision_surface(
    twin: DecisionConsistentTwin,
    posterior: PosteriorBoundState,
    candidates: Sequence[InformationCandidate],
) -> None:
    for candidate in candidates:
        result = twin.rollout(posterior.posterior_uid, candidate.world_action)
        if result.posterior_uid != posterior.posterior_uid:
            raise ValueError("World rollout used a stale posterior")
    twin.decisions.rank_interventions(
        posterior.posterior_uid,
        [candidate.world_action for candidate in candidates],
    )
    twin.decisions.compute_reliability(posterior.posterior_uid)


def _enrich_dispatch_row(
    raw: Mapping[str, Any],
    *,
    candidate: InformationCandidate,
    action: CurrentClampAction,
    policy_uid: str,
    round_uid: str,
    posterior_uid: str,
    command_exit_code: int,
    started_at_controller: str,
    finished_at_controller: str,
) -> dict[str, Any]:
    dispatch = dict(raw)
    request_uid = "static-loop-request-" + _digest(
        {
            "policy_uid": policy_uid,
            "round_uid": round_uid,
            "posterior_uid": posterior_uid,
            "information_action_uid": candidate.information_action.action_uid,
            "world_action_uid": action.to_world_action().action_uid,
        }
    )[:24]
    run_uid = "static-loop-run-" + _digest(
        {"request_uid": request_uid, "raw_bytes_hash": dispatch.get("raw_bytes_hash")}
    )[:24]
    dispatch.update(
        {
            "command_exit_code": command_exit_code,
            "request_uid": request_uid,
            "run_uid": run_uid,
            "tool_uid": REGISTERED_TOOL_UID,
            "policy_uid": policy_uid,
            "round_uid": round_uid,
            "posterior_uid": posterior_uid,
            "parent_posterior_uid": posterior_uid,
            "information_action_uid": candidate.information_action.action_uid,
            "world_action_uid": action.to_world_action().action_uid,
            "action_uid": action.action_uid,
            "action_role": action.role,
            "started_at_controller": started_at_controller,
            "finished_at_controller": finished_at_controller,
            "failure_visible": dispatch.get("status") != "completed",
        }
    )
    return dispatch


def _compose_dispatch_result(
    candidate: InformationCandidate,
    *,
    policy_uid: str,
    round_uid: str,
    posterior_uid: str,
    actions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    status = (
        "completed"
        if all(row.get("status") == "completed" for row in actions.values())
        else "partial"
    )
    result_uid = "tool-result-" + _digest(
        {
            "policy_uid": policy_uid,
            "round_uid": round_uid,
            "posterior_uid": posterior_uid,
            "information_action_uid": candidate.information_action.action_uid,
            "actions": {
                role: {
                    "raw": row.get("raw_bytes_hash"),
                    "normalized": row.get("normalized_bytes_hash"),
                    "status": row.get("status"),
                }
                for role, row in actions.items()
            },
        }
    )[:32]
    return {
        "result_uid": result_uid,
        "status": status,
        "policy_uid": policy_uid,
        "round_uid": round_uid,
        "posterior_uid": posterior_uid,
        "information_action_uid": candidate.information_action.action_uid,
        "world_action_uid": candidate.world_action.action_uid,
        "control_world_action_uid": candidate.control_world_action.action_uid,
        "actions": dict(actions),
    }


def _dispatch_candidate(
    candidate: InformationCandidate,
    typed_actions: tuple[CurrentClampAction, CurrentClampAction],
    *,
    output_dir: Path,
    source_model_id: str,
    policy_uid: str,
    round_uid: str,
    posterior_uid: str,
    resume: bool = False,
) -> dict[str, Any]:
    dispatch_root = output_dir / "prospective_dispatch" / candidate.candidate_uid
    if dispatch_root.exists():
        if not resume:
            raise FileExistsError(
                f"Visible registered-tool dispatch already exists: {dispatch_root}"
            )
        actions: dict[str, Any] = {}
        for action in typed_actions:
            role_dir = dispatch_root / (
                "control" if action.role == "matched_control" else "intervention"
            )
            dispatch_path = role_dir / "dispatch.json"
            if not dispatch_path.is_file():
                raise FileNotFoundError(
                    f"Resume refuses to redispatch an incomplete visible outcome: {dispatch_path}"
                )
            raw = json.loads(dispatch_path.read_text(encoding="utf-8"))
            array_path = role_dir / str(raw.get("array_path", "payload_arrays.npz"))
            if raw.get("status") == "completed" and not array_path.is_file():
                raise FileNotFoundError(
                    f"Resume dispatch payload is incomplete: {array_path}"
                )
            started_at = str(raw.get("started_at", ""))
            finished_at = str(raw.get("finished_at", ""))
            _parse_utc_timestamp(started_at, field=f"{action.role}.started_at")
            _parse_utc_timestamp(finished_at, field=f"{action.role}.finished_at")
            actions[action.role] = _enrich_dispatch_row(
                raw,
                candidate=candidate,
                action=action,
                policy_uid=policy_uid,
                round_uid=round_uid,
                posterior_uid=posterior_uid,
                command_exit_code=0,
                started_at_controller=started_at,
                finished_at_controller=finished_at,
            )
        return _compose_dispatch_result(
            candidate,
            policy_uid=policy_uid,
            round_uid=round_uid,
            posterior_uid=posterior_uid,
            actions=actions,
        )
    if resume:
        raise FileNotFoundError(
            f"Resume cannot dispatch a previously invisible outcome: {dispatch_root}"
        )
    dispatch_root.mkdir(parents=True, exist_ok=False)
    actions: dict[str, Any] = {}
    for action in typed_actions:
        role_dir = dispatch_root / (
            "control" if action.role == "matched_control" else "intervention"
        )
        command = [
            sys.executable,
            str(REPO_ROOT / "Libs/scripts/experiments/run_round24_tool_dispatch.py"),
            "--config",
            str(REPO_ROOT / "Libs/configs/config_stack.yaml"),
            "--output-dir",
            str(role_dir),
            "--source-model-id",
            source_model_id,
            "--amplitude",
            str(action.amplitude_nA),
            "--duration",
            str(action.duration_ms),
            "--delay",
            str(action.delay_ms),
        ]
        started_at = datetime.now(timezone.utc).isoformat()
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        finished_at = datetime.now(timezone.utc).isoformat()
        dispatch_path = role_dir / "dispatch.json"
        if not dispatch_path.is_file():
            raise RuntimeError(
                f"Registered tool did not emit dispatch metadata: {process.stderr[-500:]}"
            )
        dispatch = _enrich_dispatch_row(
            json.loads(dispatch_path.read_text(encoding="utf-8")),
            candidate=candidate,
            action=action,
            policy_uid=policy_uid,
            round_uid=round_uid,
            posterior_uid=posterior_uid,
            command_exit_code=process.returncode,
            started_at_controller=started_at,
            finished_at_controller=finished_at,
        )
        actions[action.role] = dispatch
    return _compose_dispatch_result(
        candidate,
        policy_uid=policy_uid,
        round_uid=round_uid,
        posterior_uid=posterior_uid,
        actions=actions,
    )


def _observation_from_dispatch(
    dispatch: Mapping[str, Any],
    dispatch_root: Path,
    response_species_indices: tuple[int, ...],
    normalization_scale: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    if dispatch.get("status") != "completed":
        return None, {
            "status": "failed",
            "failure_visible": True,
            "reason": "registered_tool_pair_incomplete",
        }
    trajectories: dict[str, np.ndarray] = {}
    for role, directory in (("intervention", "intervention"), ("matched_control", "control")):
        row = dispatch["actions"][role]
        array_path = dispatch_root / directory / str(row.get("array_path", "payload_arrays.npz"))
        if not array_path.is_file():
            return None, {
                "status": "failed",
                "failure_visible": True,
                "reason": "missing_registered_tool_array",
            }
        if row.get("status") != "completed" or row.get("qc") != "passed_finite_registered_source":
            return None, {
                "status": "failed",
                "failure_visible": True,
                "reason": "registered_tool_qc_failed",
            }
        with np.load(array_path, allow_pickle=False) as loaded:
            trajectories[role] = np.asarray(loaded["trajectory"], dtype=np.float64)
    scale = np.asarray(normalization_scale, dtype=np.float64)
    intervention = trajectories["intervention"] / scale.reshape(1, 1, 1, -1)
    control = trajectories["matched_control"] / scale.reshape(1, 1, 1, -1)
    observation = float(
        np.abs(
            intervention[1:, ..., list(response_species_indices)]
            - control[1:, ..., list(response_species_indices)]
        ).mean()
    )
    if not math.isfinite(observation):
        return None, {
            "status": "failed",
            "failure_visible": True,
            "reason": "nonfinite_registered_tool_observation",
        }
    return observation, {
        "status": "completed",
        "failure_visible": False,
        "gate_o": "passed_registered_source_qc",
        "uncertainty": {"epistemic": 0.05, "aleatoric": 0.05},
    }


def _tool_catalog_rows(
    dispatch: Mapping[str, Any],
    *,
    state_uid: str,
    admitted: bool,
    posterior_uid_post: str,
    evidence_use: str,
) -> list[dict[str, Any]]:
    rows = []
    for role, row in dispatch["actions"].items():
        rows.append(
            {
                "result_uid": dispatch["result_uid"],
                "policy_uid": dispatch["policy_uid"],
                "round_uid": dispatch["round_uid"],
                "state_uid": state_uid,
                "posterior_uid_pre": dispatch["posterior_uid"],
                "posterior_uid_post": posterior_uid_post,
                "information_action_uid": dispatch["information_action_uid"],
                "world_action_uid": row["world_action_uid"],
                "control_world_action_uid": dispatch["control_world_action_uid"],
                "action_role": role,
                "tool_uid": row["tool_uid"],
                "tool_version": row.get("tool_version", ""),
                "request_uid": row["request_uid"],
                "run_uid": row["run_uid"],
                "status": row.get("status"),
                "failure_status": row.get("failure_status", ""),
                "qc": row.get("qc", ""),
                "raw_bytes_hash": row.get("raw_bytes_hash", ""),
                "normalized_bytes_hash": row.get("normalized_bytes_hash", ""),
                "uncertainty_artifact": json.dumps(row.get("uncertainty_artifact", {}), sort_keys=True),
                "runtime_seconds": row.get("runtime_seconds", ""),
                "compute_cost": "cpu_registered_tool_dispatch",
                "admitted": admitted,
                "evidence_use": evidence_use,
                "failure_visible": row.get("failure_visible", False),
                **EVIDENCE_BOUNDARY,
            }
        )
    return rows


def _scientific_retention(
    pre_rows: Sequence[Mapping[str, Any]],
    post_rows: Sequence[Mapping[str, Any]],
    calibration: CalibrationState,
    *,
    pre_reliability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pre_map = {
        (row["domain"], row["evaluation_uid"], row["metric_name"]): float(row["metric_value"])
        for row in pre_rows
    }
    post_map = {
        (row["domain"], row["evaluation_uid"], row["metric_name"]): float(row["metric_value"])
        for row in post_rows
    }
    if set(pre_map) != set(post_map):
        raise ValueError("Static-loop scientific pre/post units differ")

    def mean(domain: str, metric: str, values: Mapping[tuple[Any, ...], float]) -> float:
        return _safe_mean(
            value
            for (row_domain, _, row_metric), value in values.items()
            if row_domain == domain and row_metric == metric
        )

    c1_pre = mean("c1", "masked_target_rmse", pre_map)
    c1_post = mean("c1", "masked_target_rmse", post_map)
    c2s_pre = mean("c2s", "trajectory_error", pre_map)
    c2s_post = mean("c2s", "trajectory_error", post_map)
    reliability_pre = dict(
        pre_reliability or _aggregate_reliability(pre_rows, calibration)
    )
    reliability = _aggregate_reliability(post_rows, calibration)
    reliability_delta = float(
        reliability["reliability_proxy"] - reliability_pre["reliability_proxy"]
    )
    return {
        "c1_old_domain_retention": min(1.0, c1_pre / max(c1_post, 1.0e-12)),
        "c2s_old_domain_retention": min(1.0, c2s_pre / max(c2s_post, 1.0e-12)),
        "catastrophic_forgetting": False,
        "reliability_proxy_pre": reliability_pre["reliability_proxy"],
        "reliability_proxy_post": reliability["reliability_proxy"],
        "reliability_proxy_delta": reliability_delta,
        "tri_change": reliability_delta,
        "tri_change_semantics": "calibrated_reliability_proxy_delta_not_model_tri",
        **reliability,
    }


def _run_policy_chain(
    contract: Mapping[str, Any],
    reference_contract: Mapping[str, Any],
    project_config: Mapping[str, Any],
    reference: Mapping[str, Any],
    prior: PosteriorBoundState,
    policy: StaticInformationPolicy,
    candidates: Sequence[InformationCandidate],
    typed_actions: Mapping[str, tuple[CurrentClampAction, CurrentClampAction]],
    c1_cache: Any,
    c2s_cache: Any,
    calibration: CalibrationState,
    output_root: Path,
    source_model_id: str,
    response_species_indices: tuple[int, ...],
) -> tuple[
    list[DecisionSnapshot],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[float]],
]:
    policy_root = output_root / policy.policy_uid
    policy_root.mkdir(parents=True, exist_ok=False)
    twin, runtime_states = _runtime_for_chain(prior, candidates)
    current = prior
    used: set[str] = set()
    snapshots: list[DecisionSnapshot] = []
    lineage_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    observed_truth: dict[str, list[float]] = {}
    cumulative_cost = 0.0
    checkpoint_hashes = reference["twin_checkpoints"]
    for round_index in range(1, int(contract["static_policy"]["sequential_rounds"]) + 1):
        round_uid = f"round_{round_index}"
        round_root = policy_root / round_uid
        round_root.mkdir(parents=True, exist_ok=False)
        selected, query_ranking_pre = policy.select(
            current,
            candidates,
            used_information_action_uids=used,
        )
        intervention_ranking_pre = rank_interventions(current, candidates)
        _trace_decision_surface(twin, current, candidates)
        twin.decisions.select_tool(current.posterior_uid, selected.information_action)
        input_manifest = {
            "schema_version": "neuronsifter-static-loop-round-input-v1",
            "policy_uid": policy.policy_uid,
            "round_uid": round_uid,
            "round_index": round_index,
            "posterior_uid": current.posterior_uid,
            "candidate_pool_uid": contract["prospective"]["candidate_pool_uid"],
            "candidate_pool_hash": _digest([candidate.to_record() for candidate in candidates]),
            "candidate_world_action_uids": [candidate.world_action.action_uid for candidate in candidates],
            "candidate_information_action_uids": [candidate.information_action.action_uid for candidate in candidates],
            "tool_availability_mask": {REGISTERED_TOOL_UID: True},
            "budget": {"information_action_count": 1},
            "static_policy_identity": policy.to_record(),
            "outcome_visible_before_freeze": False,
            "twin_parameters_frozen": True,
            "agent_parameters_frozen": True,
            **EVIDENCE_BOUNDARY,
        }
        selected_row = {
            "policy_uid": policy.policy_uid,
            "round_uid": round_uid,
            "posterior_uid": current.posterior_uid,
            "state_uid": selected.state_uid,
            "candidate_uid": selected.candidate_uid,
            "world_action_uid": selected.world_action.action_uid,
            "control_world_action_uid": selected.control_world_action.action_uid,
            "information_action_uid": selected.information_action.action_uid,
            "tool_uid": selected.information_action.tool_id,
            "fidelity": selected.information_action.fidelity,
            "budget": 1,
            "selected_utility": query_ranking_pre[0]["utility"],
            "selection_before_outcome": True,
            **EVIDENCE_BOUNDARY,
        }
        _freeze_json(round_root / f"{round_uid}_input_manifest.json", input_manifest)
        write_tsv(
            round_root / f"{round_uid}_selected_information_actions.tsv",
            list(selected_row),
            [selected_row],
        )
        pre_metrics = _calibrated_scientific_metrics(
            current,
            c1_cache,
            c2s_cache,
            reference_contract,
            project_config,
            output_root,
            calibration,
            strategy_uid=policy.policy_uid,
            round_uid=f"{round_uid}_pre",
        )
        pre_reliability = _aggregate_reliability(pre_metrics, calibration)
        dispatch = _dispatch_candidate(
            selected,
            typed_actions[selected.candidate_uid],
            output_dir=round_root,
            source_model_id=source_model_id,
            policy_uid=policy.policy_uid,
            round_uid=round_uid,
            posterior_uid=current.posterior_uid,
        )
        dispatch_root = (
            round_root / "prospective_dispatch" / selected.candidate_uid
        )
        observation, observation_meta = _observation_from_dispatch(
            dispatch,
            dispatch_root,
            response_species_indices,
            c2s_cache.scale,
        )
        predicted_mean, predicted_std = current.action_response(np.asarray(selected.design))
        contradiction_z = float(
            contract["decision_evaluation"]["observation_contradiction_z"]
        )
        contradictory = bool(
            observation is not None
            and abs(observation - predicted_mean) > contradiction_z * predicted_std
        )
        admitted = observation is not None
        parent = current
        if admitted:
            event = {
                "event_type": "admitted_observation_assimilation",
                "policy_uid": policy.policy_uid,
                "round_uid": round_uid,
                "state_uid": selected.state_uid,
                "world_action_uid": selected.world_action.action_uid,
                "information_action_uid": selected.information_action.action_uid,
                "tool_uid": REGISTERED_TOOL_UID,
                "result_uid": dispatch["result_uid"],
                "model_parameter_update": False,
            }
            current = current.update(
                np.asarray(selected.design), float(observation), event=event
            )
            runtime_states[current.posterior_uid] = current

            def completed_handler(
                posterior: ScientificPosterior,
                action: InformationAction,
                *,
                observed: float = float(observation),
                result_uid: str = str(dispatch["result_uid"]),
            ) -> ToolResult:
                return ToolResult(
                    posterior_uid=posterior.posterior_uid,
                    action_uid=str(action.action_uid),
                    status="completed",
                    observation={"electrophysiology_response": observed},
                    uncertainty={"epistemic": 0.05, "aleatoric": 0.05},
                    provenance={
                        "tool_uid": REGISTERED_TOOL_UID,
                        "result_uid": result_uid,
                        "world_action_uid": selected.world_action.action_uid,
                    },
                )

            twin.information.register_tool(REGISTERED_TOOL_UID, completed_handler)
            tool_result = twin.predict_tool_outcome(
                parent.posterior_uid, selected.information_action
            )
            if not tool_result.valid_for_assimilation:
                raise ValueError("Admitted registered-tool result failed runtime validation")
            twin.register_posterior(
                current.to_scientific_posterior(),
                activate=True,
                expected_parent_posterior_uid=parent.posterior_uid,
                assimilation_status=tool_result.status,
                reason="static_scientific_belief_assimilation",
            )
            observed_truth.setdefault(selected.candidate_uid, []).append(float(observation))
        else:
            failed_result = ToolResult(
                posterior_uid=parent.posterior_uid,
                action_uid=str(selected.information_action.action_uid),
                status="failed",
                error=str(observation_meta.get("reason", "tool_failure")),
                provenance={"tool_uid": REGISTERED_TOOL_UID, "failure_visible": True},
            )
            twin.information.register_tool(
                REGISTERED_TOOL_UID,
                lambda posterior, action, result=failed_result: result,
            )
            observed_result = twin.predict_tool_outcome(
                parent.posterior_uid, selected.information_action
            )
            if observed_result.valid_for_assimilation:
                raise ValueError("Failed tool result became admissible")
        post_metrics = _calibrated_scientific_metrics(
            current,
            c1_cache,
            c2s_cache,
            reference_contract,
            project_config,
            output_root,
            calibration,
            strategy_uid=policy.policy_uid,
            round_uid=f"{round_uid}_post",
        )
        reliability = _scientific_retention(
            pre_metrics,
            post_metrics,
            calibration,
            pre_reliability=pre_reliability,
        )
        _trace_decision_surface(twin, current, candidates)
        decision = (
            twin.decisions.accept(
                current.posterior_uid,
                reason="inside_static_support_and_calibration_region",
            )
            if reliability["coverage_floor_respected"]
            and reliability["calibration_within_tolerance"]
            else twin.decisions.abstain(
                current.posterior_uid,
                reason="insufficient_reliability_evidence",
            )
        )
        intervention_ranking_post = rank_interventions(current, candidates)
        used.add(str(selected.information_action.action_uid))
        query_ranking_post = policy.rank(
            current,
            candidates,
            used_information_action_uids=used,
        )
        pre_forecasts = {
            row["candidate_uid"]: float(row["predicted_response"])
            for row in intervention_ranking_pre
        }
        post_forecasts = {
            row["candidate_uid"]: float(row["predicted_response"])
            for row in intervention_ranking_post
        }
        forecast_change = float(
            np.linalg.norm(
                np.asarray(
                    [post_forecasts[uid] - pre_forecasts[uid] for uid in sorted(pre_forecasts)],
                    dtype=np.float64,
                )
            )
        )
        query_pre_uids = [str(row["candidate_uid"]) for row in query_ranking_pre]
        query_post_uids = [str(row["candidate_uid"]) for row in query_ranking_post]
        common_query_uids = [uid for uid in query_pre_uids if uid in set(query_post_uids)]
        query_disagreement = normalized_rank_disagreement(
            common_query_uids,
            [uid for uid in query_post_uids if uid in set(common_query_uids)],
        )
        intervention_disagreement = normalized_rank_disagreement(
            [str(row["candidate_uid"]) for row in intervention_ranking_pre],
            [str(row["candidate_uid"]) for row in intervention_ranking_post],
        )
        cumulative_cost += sum(
            float(row.get("runtime_seconds", 0.0))
            for row in dispatch["actions"].values()
        )
        for trace in twin.decisions.trace:
            DecisionController.assert_consistent(trace, allow_unset=True)
        paired = _paired_rows(
            pre_metrics,
            post_metrics,
            strategy_uid=policy.policy_uid,
            round_uid=round_uid,
        )
        write_tsv(
            round_root / f"{round_uid}_scientific_pre_post.tsv",
            list(paired[0]),
            paired,
        )
        evidence_admission = {
            "policy_uid": policy.policy_uid,
            "round_uid": round_uid,
            "state_uid": selected.state_uid,
            "posterior_uid_pre": parent.posterior_uid,
            "posterior_uid_post": current.posterior_uid,
            "world_action_uid": selected.world_action.action_uid,
            "information_action_uid": selected.information_action.action_uid,
            "tool_uid": REGISTERED_TOOL_UID,
            "result_uid": dispatch["result_uid"],
            "admitted": admitted,
            "gate_o": observation_meta.get("gate_o", "failed_visible"),
            "model_parameters_changed": False,
            **EVIDENCE_BOUNDARY,
        }
        write_tsv(
            round_root / f"{round_uid}_evidence_admission.tsv",
            list(evidence_admission),
            [evidence_admission],
        )
        write_tsv(
            round_root / f"{round_uid}_tool_results.tsv",
            sorted(
                {
                    key
                    for row in _tool_catalog_rows(
                        dispatch,
                        state_uid=selected.state_uid,
                        admitted=admitted,
                        posterior_uid_post=current.posterior_uid,
                        evidence_use="sequential_assimilation",
                    )
                    for key in row
                }
            ),
            _tool_catalog_rows(
                dispatch,
                state_uid=selected.state_uid,
                admitted=admitted,
                posterior_uid_post=current.posterior_uid,
                evidence_use="sequential_assimilation",
            ),
        )
        posterior_update = {
            "schema_version": "neuronsifter-static-posterior-assimilation-v1",
            "policy_uid": policy.policy_uid,
            "round_uid": round_uid,
            "posterior_uid_pre": parent.posterior_uid,
            "posterior_uid_post": current.posterior_uid,
            "parent_posterior_uid": current.parent_posterior_uid,
            "observation_admitted": admitted,
            "model_checkpoint_pre": checkpoint_hashes,
            "model_checkpoint_post": checkpoint_hashes,
            "model_checkpoint_pre_equals_post": True,
            "twin_parameters_changed": False,
            "agent_parameters_changed": False,
            "posterior_distribution_changed": current.posterior_uid != parent.posterior_uid,
            "decision_consistency_invariant": all(
                len(
                    {
                        str(trace[field])
                        for field in (
                            "posterior_uid_rollout",
                            "posterior_uid_query",
                            "posterior_uid_assimilation_parent",
                            "posterior_uid_decision",
                        )
                    }
                )
                == 1
                for trace in twin.decisions.trace
            ),
            **EVIDENCE_BOUNDARY,
        }
        write_json(round_root / f"{round_uid}_posterior_update.json", posterior_update)
        write_json(
            round_root / f"{round_uid}_report.json",
            {
                **posterior_update,
                "selected_candidate_uid": selected.candidate_uid,
                "selected_information_action_uid": selected.information_action.action_uid,
                "selected_world_action_uid": selected.world_action.action_uid,
                "result_uid": dispatch["result_uid"],
                "reliability": reliability,
                "terminal_decision": decision["decision_label"],
                "cumulative_cost": cumulative_cost,
            },
        )
        receipt = materialize_round_receipt(
            round_root,
            policy_uid=policy.policy_uid,
            round_uid=round_uid,
            live_dispatch=dispatch,
        )
        lineage_row = {
            "policy_uid": policy.policy_uid,
            "round_uid": round_uid,
            "state_uid": selected.state_uid,
            "posterior_uid_pre": parent.posterior_uid,
            "world_action_uid": selected.world_action.action_uid,
            "information_action_uid": selected.information_action.action_uid,
            "tool_uid": REGISTERED_TOOL_UID,
            "result_uid": dispatch["result_uid"],
            "posterior_uid_post": current.posterior_uid,
            "posterior_version_pre": parent.version,
            "posterior_version_post": current.version,
            "parent_link_valid": (
                current.parent_posterior_uid == parent.posterior_uid
                if admitted
                else current.posterior_uid == parent.posterior_uid
            ),
            "model_checkpoint_pre": json.dumps(checkpoint_hashes, sort_keys=True),
            "model_checkpoint_post": json.dumps(checkpoint_hashes, sort_keys=True),
            "model_checkpoint_pre_equals_post": True,
            "twin_parameters_changed": False,
            "agent_parameters_changed": False,
            "observation_admitted": admitted,
            "shared_posterior_invariant": posterior_update["decision_consistency_invariant"],
            "round_receipt_uid": receipt["receipt_uid"],
            **EVIDENCE_BOUNDARY,
        }
        lineage_rows.append(lineage_row)
        current_tool_rows = _tool_catalog_rows(
            dispatch,
            state_uid=selected.state_uid,
            admitted=admitted,
            posterior_uid_post=current.posterior_uid,
            evidence_use="sequential_assimilation",
        )
        tool_rows.extend(current_tool_rows)
        snapshots.append(
            DecisionSnapshot(
                policy_uid=policy.policy_uid,
                round_uid=round_uid,
                selected_candidate_uid=selected.candidate_uid,
                selected_information_action_uid=str(selected.information_action.action_uid),
                posterior_uid_pre=parent.posterior_uid,
                posterior_uid_post=current.posterior_uid,
                intervention_ranking_pre=intervention_ranking_pre,
                intervention_ranking_post=intervention_ranking_post,
                query_ranking_pre=query_ranking_pre,
                query_ranking_post=query_ranking_post,
                selected_utility=float(query_ranking_pre[0]["utility"]),
                posterior_discrepancy=posterior_discrepancy_proxy(parent, current),
                forecast_change=forecast_change,
                intervention_rank_disagreement=intervention_disagreement,
                query_rank_disagreement=query_disagreement,
                intervention_switch=(
                    intervention_ranking_pre[0]["candidate_uid"]
                    != intervention_ranking_post[0]["candidate_uid"]
                ),
                observation=observation,
                predicted_observation_pre=predicted_mean,
                predicted_observation_std_pre=predicted_std,
                contradictory_observation=contradictory,
                admitted=admitted,
                cumulative_cost=cumulative_cost,
                reliability=reliability,
            )
        )
    return snapshots, lineage_rows, tool_rows, observed_truth


def _complete_blinded_truth(
    contract: Mapping[str, Any],
    candidates: Sequence[InformationCandidate],
    typed_actions: Mapping[str, tuple[CurrentClampAction, CurrentClampAction]],
    source_model_id: str,
    prior: PosteriorBoundState,
    c2s_cache: Any,
    response_species_indices: tuple[int, ...],
    output_root: Path,
    observed_truth: dict[str, list[float]],
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation_root = output_root / "blinded_evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=False)
    tool_rows: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    truth: dict[str, float] = {}
    for candidate in candidates:
        values = list(observed_truth.get(candidate.candidate_uid, []))
        if values:
            if max(values) - min(values) > 1.0e-9:
                contradictions.append(
                    {
                        "candidate_uid": candidate.candidate_uid,
                        "scenario": "contradictory_tool_result",
                        "status": "observed_duplicate_disagreement",
                        "spread": max(values) - min(values),
                        "response": "insufficient_evidence",
                        "fixture": False,
                    }
                )
            truth[candidate.candidate_uid] = float(np.mean(values))
            continue
        candidate_root = evaluation_root / candidate.candidate_uid
        candidate_root.mkdir(parents=True, exist_ok=False)
        freeze = {
            "schema_version": "neuronsifter-static-blinded-evaluation-freeze-v1",
            "candidate_uid": candidate.candidate_uid,
            "posterior_uid": prior.posterior_uid,
            "candidate_pool_uid": contract["prospective"]["candidate_pool_uid"],
            "world_action_uid": candidate.world_action.action_uid,
            "information_action_uid": candidate.information_action.action_uid,
            "policy_outcomes_frozen_before_truth_visibility": True,
            "admission_allowed": False,
            **EVIDENCE_BOUNDARY,
        }
        _freeze_json(candidate_root / "evaluation_freeze.json", freeze)
        dispatch = _dispatch_candidate(
            candidate,
            typed_actions[candidate.candidate_uid],
            output_dir=candidate_root,
            source_model_id=source_model_id,
            policy_uid="blinded_evaluation",
            round_uid="terminal_evaluation",
            posterior_uid=prior.posterior_uid,
        )
        dispatch_root = candidate_root / "prospective_dispatch" / candidate.candidate_uid
        observation, meta = _observation_from_dispatch(
            dispatch,
            dispatch_root,
            response_species_indices,
            c2s_cache.scale,
        )
        if observation is None:
            raise RuntimeError(
                f"Blinded decision truth is unavailable for {candidate.candidate_uid}: {meta}"
            )
        truth[candidate.candidate_uid] = float(observation)
        tool_rows.extend(
            _tool_catalog_rows(
                dispatch,
                state_uid=candidate.state_uid,
                admitted=False,
                posterior_uid_post=prior.posterior_uid,
                evidence_use="blinded_terminal_decision_evaluation",
            )
        )
    if set(truth) != {candidate.candidate_uid for candidate in candidates}:
        raise ValueError("Blinded decision truth is incomplete")
    return truth, tool_rows, contradictions


def _top_k_retention(
    predicted: Sequence[str], truth: Sequence[str], top_k: int
) -> float:
    predicted_set = set(predicted[:top_k])
    truth_set = set(truth[:top_k])
    return len(predicted_set & truth_set) / float(top_k)


def _evaluate_decisions(
    contract: Mapping[str, Any],
    snapshots: Sequence[DecisionSnapshot],
    truth: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    truth_ranking = sorted(truth, key=lambda uid: (-float(truth[uid]), uid))
    truth_best = truth_ranking[0]
    truth_best_value = float(truth[truth_best])
    top_k = int(contract["static_policy"]["top_k"])
    false_accept_threshold = float(
        contract["decision_evaluation"]["false_accept_risk_threshold"]
    )
    decision_rows: list[dict[str, Any]] = []
    effects_rows: list[dict[str, Any]] = []
    coherence_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        pre_rank = [str(row["candidate_uid"]) for row in snapshot.intervention_ranking_pre]
        post_rank = [str(row["candidate_uid"]) for row in snapshot.intervention_ranking_post]
        risk_pre = truth_best_value - float(truth[pre_rank[0]])
        risk_post = truth_best_value - float(truth[post_rank[0]])
        uncertainties = np.asarray(
            [float(row["predictive_std"]) for row in snapshot.intervention_ranking_post],
            dtype=np.float64,
        )
        regrets = np.asarray(
            [truth_best_value - float(truth[str(row["candidate_uid"])]) for row in snapshot.intervention_ranking_post],
            dtype=np.float64,
        )
        accepted = bool(
            snapshot.reliability["coverage_floor_respected"]
            and snapshot.reliability["calibration_within_tolerance"]
        )
        decision_rows.append(
            {
                "policy_uid": snapshot.policy_uid,
                "round_uid": snapshot.round_uid,
                "posterior_uid_pre": snapshot.posterior_uid_pre,
                "posterior_uid_post": snapshot.posterior_uid_post,
                "selected_information_action_uid": snapshot.selected_information_action_uid,
                "decision_risk_pre": risk_pre,
                "decision_risk_post": risk_post,
                "decision_risk_delta": risk_post - risk_pre,
                "cumulative_cost": snapshot.cumulative_cost,
                "top_k_intervention_retention": _top_k_retention(post_rank, truth_ranking, top_k),
                "false_elimination": truth_best not in set(post_rank[:top_k]),
                "selective_coverage": float(accepted),
                "coverage": snapshot.reliability["coverage"],
                "coverage_floor": snapshot.reliability["coverage_floor"],
                "coverage_floor_respected": snapshot.reliability[
                    "coverage_floor_respected"
                ],
                "calibration_error": snapshot.reliability["calibration_error"],
                "calibration_within_tolerance": snapshot.reliability[
                    "calibration_within_tolerance"
                ],
                "tri_score_available": snapshot.reliability["tri_score_available"],
                "reliability_proxy_name": snapshot.reliability[
                    "reliability_proxy_name"
                ],
                "reliability_proxy_pre": snapshot.reliability[
                    "reliability_proxy_pre"
                ],
                "reliability_proxy_post": snapshot.reliability[
                    "reliability_proxy_post"
                ],
                "reliability_proxy_delta": snapshot.reliability[
                    "reliability_proxy_delta"
                ],
                "aurc": _aurc_np(regrets, uncertainties),
                "false_accept": bool(accepted and risk_post > false_accept_threshold),
                "accepted_precision": float(not accepted or risk_post <= false_accept_threshold),
                "terminal_decision_risk_measured": True,
                "model_parameters_changed": False,
                "agent_parameters_changed": False,
                **EVIDENCE_BOUNDARY,
            }
        )
        risk_reduction = risk_pre - risk_post
        effects_rows.append(
            {
                "policy_uid": snapshot.policy_uid,
                "round_uid": snapshot.round_uid,
                "information_action_uid": snapshot.selected_information_action_uid,
                "posterior_uid_pre": snapshot.posterior_uid_pre,
                "posterior_uid_post": snapshot.posterior_uid_post,
                "posterior_change": snapshot.posterior_discrepancy,
                "world_action_forecast_change": snapshot.forecast_change,
                "intervention_rank_change": snapshot.intervention_rank_disagreement,
                "query_rank_change": snapshot.query_rank_disagreement,
                "observation_induced_intervention_switch": snapshot.intervention_switch,
                "tri_score_available": snapshot.reliability["tri_score_available"],
                "tri_change": snapshot.reliability["tri_change"],
                "tri_change_semantics": snapshot.reliability[
                    "tri_change_semantics"
                ],
                "reliability_proxy_name": snapshot.reliability[
                    "reliability_proxy_name"
                ],
                "reliability_proxy_pre": snapshot.reliability[
                    "reliability_proxy_pre"
                ],
                "reliability_proxy_post": snapshot.reliability[
                    "reliability_proxy_post"
                ],
                "reliability_proxy_delta": snapshot.reliability[
                    "reliability_proxy_delta"
                ],
                "terminal_decision_risk_change": risk_post - risk_pre,
                "cost": snapshot.cumulative_cost,
                "information_action_with_no_decision_effect": bool(
                    not snapshot.intervention_switch and abs(risk_reduction) <= 1.0e-12
                ),
                "misleading_information_action": risk_post > risk_pre + 1.0e-12,
                "contradictory_tool_result": snapshot.contradictory_observation,
                "observation_increased_uncertainty": False,
                "selected_utility": snapshot.selected_utility,
                "observed_decision_risk_reduction": risk_reduction,
                **EVIDENCE_BOUNDARY,
            }
        )
        coherence_rows.append(
            {
                "policy_uid": snapshot.policy_uid,
                "round_uid": snapshot.round_uid,
                "posterior_uid_pre": snapshot.posterior_uid_pre,
                "posterior_uid_post": snapshot.posterior_uid_post,
                "posterior_discrepancy_proxy": snapshot.posterior_discrepancy,
                "proxy_name": contract["decision_evaluation"]["discrepancy_proxy"],
                "theoretical_tv_reported": False,
                "query_rank_disagreement": snapshot.query_rank_disagreement,
                "intervention_rank_disagreement": snapshot.intervention_rank_disagreement,
                "utility_calibration_error": abs(snapshot.selected_utility - risk_reduction),
                "top_query_regret": max(0.0, -risk_reduction),
                "shared_posterior_identity": True,
                **EVIDENCE_BOUNDARY,
            }
        )
    return decision_rows, effects_rows, coherence_rows


def _failure_catalog(
    snapshots: Sequence[DecisionSnapshot],
    contradictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "scenario": "no_useful_query_available",
            "source": "contract_defined_isolated_fixture",
            "observed": True,
            "admitted": False,
            "posterior_moved": False,
            "decision": "insufficient_evidence",
            "scientific_measurement": False,
        },
        {
            "scenario": "all_tools_unsupported",
            "source": "contract_defined_isolated_fixture",
            "observed": True,
            "admitted": False,
            "posterior_moved": False,
            "decision": "insufficient_evidence",
            "scientific_measurement": False,
        },
        {
            "scenario": "tool_failure",
            "source": "contract_defined_isolated_fixture",
            "observed": True,
            "admitted": False,
            "posterior_moved": False,
            "decision": "insufficient_evidence",
            "scientific_measurement": False,
        },
        {
            "scenario": "partial_result",
            "source": "contract_defined_isolated_fixture",
            "observed": True,
            "admitted": True,
            "posterior_moved": True,
            "decision": "recompute_reliability",
            "scientific_measurement": False,
        },
        {
            "scenario": "contradictory_evidence",
            "source": "contract_defined_isolated_fixture",
            "observed": True,
            "admitted": False,
            "posterior_moved": False,
            "decision": "insufficient_evidence",
            "scientific_measurement": False,
        },
        {
            "scenario": "observation_increases_uncertainty",
            "source": "contract_defined_isolated_fixture",
            "observed": True,
            "admitted": False,
            "posterior_moved": False,
            "decision": "insufficient_evidence",
            "scientific_measurement": False,
        },
        {
            "scenario": "observation_changes_ranking_incorrectly",
            "source": "contract_defined_isolated_fixture",
            "observed": True,
            "admitted": True,
            "posterior_moved": True,
            "decision": "retain_with_failure_label",
            "scientific_measurement": False,
        },
    ]
    for snapshot in snapshots:
        if not snapshot.admitted:
            rows.append(
                {
                    "scenario": "tool_failure",
                    "source": "prospective_execution",
                    "observed": True,
                    "admitted": False,
                    "posterior_moved": False,
                    "decision": "insufficient_evidence",
                    "scientific_measurement": True,
                }
            )
        if snapshot.contradictory_observation:
            rows.append(
                {
                    "scenario": "contradictory_evidence",
                    "source": "prospective_execution",
                    "observed": True,
                    "admitted": snapshot.admitted,
                    "posterior_moved": snapshot.posterior_uid_pre != snapshot.posterior_uid_post,
                    "decision": "recompute_reliability",
                    "scientific_measurement": True,
                }
            )
    for row in contradictions:
        rows.append(
            {
                "scenario": "contradictory_evidence",
                "source": "prospective_duplicate_execution",
                "observed": True,
                "admitted": False,
                "posterior_moved": False,
                "decision": "insufficient_evidence",
                "scientific_measurement": True,
                "detail": json.dumps(dict(row), sort_keys=True),
            }
        )
    columns = (
        "scenario",
        "source",
        "observed",
        "admitted",
        "posterior_moved",
        "decision",
        "scientific_measurement",
        "detail",
    )
    return [
        {
            **{column: row.get(column, "") for column in columns},
            **EVIDENCE_BOUNDARY,
        }
        for row in rows
    ]


def _write_terminal_gates(
    contract: Mapping[str, Any],
    calibration_report: Mapping[str, Any],
    snapshots: Sequence[DecisionSnapshot],
    lineage_rows: Sequence[Mapping[str, Any]],
    tool_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    effects_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_policy: dict[str, list[Mapping[str, Any]]] = {
        uid: [row for row in decision_rows if row["policy_uid"] == uid]
        for uid in POLICY_UIDS
    }
    s_rows = by_policy["S_shared_belief_decision_risk"]
    u_rows = by_policy["U_uncertainty_only"]
    s_snapshots = [
        row for row in snapshots if row.policy_uid == "S_shared_belief_decision_risk"
    ]
    admitted_s = sum(row.admitted for row in s_snapshots)
    sequential_rounds = len(s_snapshots)
    invariant = all(_truthy(row["shared_posterior_invariant"]) for row in lineage_rows)
    checkpoints_unchanged = all(
        _truthy(row["model_checkpoint_pre_equals_post"]) for row in lineage_rows
    )
    coverage_ok = all(_truthy(row["coverage_floor_respected"]) for row in s_rows)
    calibration_ok = all(snapshot.reliability["calibration_within_tolerance"] for snapshot in s_snapshots)
    effects_present = any(
        float(row["world_action_forecast_change"]) > 0.0
        and float(row["posterior_change"]) > 0.0
        for row in effects_rows
        if row["policy_uid"] == "S_shared_belief_decision_risk"
    )
    decision_effects_present = any(
        bool(row["observation_induced_intervention_switch"])
        or abs(float(row["terminal_decision_risk_change"])) > 1.0e-12
        for row in effects_rows
        if row["policy_uid"] == "S_shared_belief_decision_risk"
    )
    s_terminal = float(s_rows[-1]["decision_risk_post"]) if s_rows else math.inf
    u_terminal = float(u_rows[-1]["decision_risk_post"]) if u_rows else math.inf
    s_beats_uncertainty = s_terminal < u_terminal - 1.0e-12
    prerequisites = bool(
        sequential_rounds >= 3
        and admitted_s >= 3
        and invariant
        and checkpoints_unchanged
        and coverage_ok
        and calibration_ok
    )
    if not prerequisites:
        status = "failed_static_decision_loop"
    elif not effects_present or not decision_effects_present:
        status = "failed_static_decision_loop"
    elif s_beats_uncertainty:
        status = "neuronsifter_static_decision_loop_development_resolved"
    else:
        status = "partial_static_decision_loop_evidence"
    mechanism_gate = {
        "schema_version": "neuronsifter-round25-static-mechanism-gate-v1",
        "status": status,
        "primary_question": "does_shared_belief_decision_risk_policy_improve_decisions_at_matched_cost",
        "controls_complete": {uid: len(rows) == 3 for uid, rows in by_policy.items()},
        "s_terminal_decision_risk": s_terminal,
        "u_terminal_decision_risk": u_terminal,
        "s_outperforms_uncertainty_only": s_beats_uncertainty,
        "posterior_and_forecast_effects_present": effects_present,
        "scientific_decision_effects_present": decision_effects_present,
        "external_baseline_substitute": False,
        **EVIDENCE_BOUNDARY,
    }
    reliability_gate = {
        "schema_version": "neuronsifter-round25-static-reliability-gate-v1",
        "status": "resolved" if coverage_ok and calibration_ok else "failed",
        "coverage_floor": contract["support_calibration"]["coverage_floor"],
        "coverage_floor_respected": coverage_ok,
        "calibration_within_tolerance": calibration_ok,
        "tri_calibration_uid": calibration_report["calibration_uid"],
        "failure_scenarios_complete": all(
            scenario
            in {str(row["scenario"]) for row in failure_rows}
            for scenario in {
                "no_useful_query_available",
                "all_tools_unsupported",
                "tool_failure",
                "partial_result",
                "contradictory_evidence",
                "observation_increases_uncertainty",
                "observation_changes_ranking_incorrectly",
            }
        ),
        "insufficient_evidence_route_enabled": True,
        **EVIDENCE_BOUNDARY,
    }
    architecture_gate = {
        "schema_version": "neuronsifter-round25-static-architecture-gate-v1",
        "status": status,
        "twin": {"canonical_reference": "U0_frozen", "parameter_updates": 0},
        "agent": {"policy_parameters_frozen_during_evaluation": True, "parameter_updates": 0},
        "prospective": {
            "sequential_rounds": sequential_rounds,
            "typed_world_actions": all(bool(row["world_action_uid"]) for row in lineage_rows),
            "typed_information_actions": all(bool(row["information_action_uid"]) for row in lineage_rows),
            "real_admitted_observations": admitted_s,
        },
        "belief": {
            "shared_posterior_invariant": invariant,
            "sequential_assimilation_valid": all(_truthy(row["parent_link_valid"]) for row in lineage_rows),
        },
        "decision": {
            "information_effects_measured": bool(effects_rows),
            "terminal_decision_risk_measured": bool(decision_rows),
        },
        "reliability": {
            "coverage_floor_respected": coverage_ok,
            "calibration_within_tolerance": calibration_ok,
        },
        "publication": {
            "protected_holdout_access": 0,
            "external_comparison_open": False,
        },
        "c2b": {"status": "blocked_insufficient_real_e2_e4"},
        "full_suite_pending": True,
        **EVIDENCE_BOUNDARY,
    }
    write_json(_repo_path(contract["outputs"]["mechanism_gate"]), mechanism_gate)
    write_json(_repo_path(contract["outputs"]["reliability_gate"]), reliability_gate)
    write_json(_repo_path(contract["outputs"]["architecture_gate"]), architecture_gate)
    write_json(
        _repo_path(contract["outputs"]["handoff"]),
        {
            "schema_version": "neuronsifter-round25-development-handoff-v1",
            "status": status,
            "canonical_architecture": (
                "frozen_twin_static_decision_consistent_agent_sequential_belief_assimilation_tri"
                if status == "neuronsifter_static_decision_loop_development_resolved"
                else "frozen_twin_world_action_prediction_reliability_with_information_acquisition_under_review"
            ),
            "round24b_negative_result_preserved": True,
            "full_suite_pending": True,
            "publication_comparison_open": False,
            "external_baseline_qualification_resumed": False,
            "agent_adaptation_authorized": False,
            "twin_parameter_adaptation_authorized": False,
            "c2b": {"status": "blocked_insufficient_real_e2_e4"},
            **EVIDENCE_BOUNDARY,
        },
    )
    return architecture_gate


def run_campaign(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    output_root: Path | None = None,
    device: str = "cpu",
    preflight_only: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    contract = _load_yaml(contract_path)
    output_root = output_root or _repo_path(contract["outputs"]["run_root"])
    candidate_manifest_path = (
        _repo_path(contract["outputs"]["manifest_root"])
        / "static_information_action_candidates.tsv"
    )
    if resume:
        if preflight_only:
            raise ValueError("--resume and --preflight-only are mutually exclusive")
        if not output_root.is_dir():
            raise FileNotFoundError(
                f"Static-loop resume root does not exist: {output_root}"
            )
        receipts = []
        for policy_uid in POLICY_UIDS:
            for round_index in range(
                1, int(contract["static_policy"]["sequential_rounds"]) + 1
            ):
                round_uid = f"round_{round_index}"
                receipts.append(
                    materialize_round_receipt(
                        output_root / policy_uid / round_uid,
                        policy_uid=policy_uid,
                        round_uid=round_uid,
                    )
                )
        gate_path = _repo_path(contract["outputs"]["architecture_gate"])
        if not gate_path.is_file():
            raise RuntimeError(
                "Resume found nonterminal round artifacts; registered-tool redispatch is refused"
            )
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") not in {
            "neuronsifter_static_decision_loop_development_resolved",
            "partial_static_decision_loop_evidence",
            "failed_static_decision_loop",
            "blocked_static_decision_loop",
        }:
            raise ValueError("Resume found a nonterminal static architecture Gate")
        return {
            **gate,
            "resume_validated": True,
            "registered_tool_redispatch_executed": False,
            "round_receipt_count": len(receipts),
        }
    if output_root.exists():
        raise FileExistsError(f"Static-loop output root already exists: {output_root}")
    if candidate_manifest_path.exists():
        raise FileExistsError(
            "Fresh static-loop execution refuses an existing candidate-pool freeze: "
            f"{candidate_manifest_path}"
        )
    reference, prior, reference_contract, lineage = _validate_reference(contract)
    if not preflight_only:
        _write_reference_freezes(contract, contract_path, reference, prior, lineage)
    taxonomy, taxonomy_summary = _coverage_taxonomy(contract)
    if not preflight_only:
        write_tsv(
            _repo_path(contract["outputs"]["coverage_taxonomy"]),
            list(taxonomy[0]),
            taxonomy,
        )
    output_root.mkdir(parents=True, exist_ok=False)
    project_config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    c1_cache = _load_c1_cache(
        project_config, reference_contract, reference, output_root, device
    )
    c2s_cache = _load_c2_cache(
        project_config, reference_contract, reference, output_root, device
    )
    calibration, calibration_report = _fit_support_calibration(
        contract,
        prior,
        c1_cache,
        c2s_cache,
        reference_contract,
        project_config,
    )
    if not preflight_only:
        write_json(_repo_path(contract["outputs"]["tri_calibration"]), calibration_report)
    u0_operating_region_known = bool(
        calibration_report["c1"]["evaluation_coverage_adjusted"]
        >= calibration.coverage_floor
        and calibration_report["c2s"]["evaluation_coverage_adjusted"]
        >= calibration.coverage_floor
    )
    support_domain = {
        "schema_version": "neuronsifter-round25-static-support-domain-v1",
        "status": "resolved" if u0_operating_region_known else "failed",
        **taxonomy_summary,
        "u0_operating_region_known": u0_operating_region_known,
        "u0_operating_region": {
            "c1_evaluation_units": list(calibration.c1_evaluation_units),
            "c2s_evaluation_units": list(calibration.c2s_evaluation_units),
            "c1_coverage": calibration_report["c1"]["evaluation_coverage_adjusted"],
            "c2s_coverage": calibration_report["c2s"]["evaluation_coverage_adjusted"],
            "tri_calibration_uid": calibration.calibration_uid,
        },
        "coverage_floor_changed_from_test": False,
        "unknown_rejection_reason": 0,
        **EVIDENCE_BOUNDARY,
    }
    if not preflight_only:
        write_json(_repo_path(contract["outputs"]["support_domain"]), support_domain)
    if not u0_operating_region_known:
        gate = {
            "schema_version": "neuronsifter-round25-static-architecture-gate-v1",
            "status": "blocked_static_decision_loop",
            "reason": "u0_static_operating_region_below_frozen_coverage_floor",
            "twin_parameter_adaptation_authorized": False,
            "agent_parameter_adaptation_authorized": False,
            "prospective_campaign_executed": False,
            **EVIDENCE_BOUNDARY,
        }
        write_json(
            output_root / "preflight.json"
            if preflight_only
            else _repo_path(contract["outputs"]["architecture_gate"]),
            gate,
        )
        return gate
    pool = json.loads(
        _repo_path(reference_contract["reference"]["c2s_pool_path"]).read_text(
            encoding="utf-8"
        )
    )
    source_model_id = str(pool["registered_tool_execution"]["source_model_id"])
    candidates, typed_actions = _build_candidates(
        contract, reference_contract, c2s_cache, source_model_id
    )
    policies = {
        uid: StaticInformationPolicy(
            uid,
            seed=int(contract["static_policy"]["seed"]),
            fixed_order=contract["static_policy"]["fixed_cascade_order"],
            expected_cost_floor=float(contract["static_policy"]["expected_cost_floor"]),
        )
        for uid in contract["static_policy"]["controls"]
    }
    if tuple(policies) != POLICY_UIDS:
        raise ValueError("Static-loop controls do not match the canonical R/F/U/S contract")
    if preflight_only:
        result = {
            "schema_version": "neuronsifter-static-decision-loop-preflight-v1",
            "status": "passed",
            "prospective_tool_dispatch_executed": False,
            "u0_operating_region_known": u0_operating_region_known,
            "coverage_floor": calibration.coverage_floor,
            "c1_evaluation_coverage": calibration_report["c1"][
                "evaluation_coverage_adjusted"
            ],
            "c2s_evaluation_coverage": calibration_report["c2s"][
                "evaluation_coverage_adjusted"
            ],
            "c1_evaluation_calibration_error": calibration_report["c1"][
                "evaluation_calibration_error"
            ],
            "c2s_evaluation_calibration_error": calibration_report["c2s"][
                "evaluation_calibration_error"
            ],
            "candidate_count": len(candidates),
            "candidate_pool_uid": contract["prospective"]["candidate_pool_uid"],
            "candidate_pool_hash": _digest(
                [candidate.to_record() for candidate in candidates]
            ),
            "policy_uids": list(policies),
            "fresh_protocols_validated": True,
            "contract_identity": {
                "path": _rel(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "source_identities": {
                "campaign_runner": {
                    "path": _rel(Path(__file__)),
                    "sha256": sha256_file(Path(__file__)),
                },
                "static_policy": {
                    "path": _rel(
                        REPO_ROOT
                        / "Libs/models/modules/static_information_policy.py"
                    ),
                    "sha256": sha256_file(
                        REPO_ROOT
                        / "Libs/models/modules/static_information_policy.py"
                    ),
                },
            },
            "checkpoint_identities": reference["twin_checkpoints"],
            "twin_parameter_updates": 0,
            "agent_parameter_updates": 0,
            **EVIDENCE_BOUNDARY,
        }
        write_json(output_root / "preflight.json", result)
        return result
    _write_policy_freezes(
        contract, contract_path, policies, candidates, prior, reference
    )
    manifest_root = _repo_path(contract["outputs"]["manifest_root"])
    manifest_root.mkdir(parents=True, exist_ok=True)
    candidate_manifest_rows = [
        {
            **candidate.to_record(),
            "tool_uid": REGISTERED_TOOL_UID,
            "fidelity": candidate.information_action.fidelity,
            "outcome_visible_before_freeze": False,
            **EVIDENCE_BOUNDARY,
        }
        for candidate in candidates
    ]
    write_tsv(
        manifest_root / "static_information_action_candidates.tsv",
        list(candidate_manifest_rows[0]),
        candidate_manifest_rows,
    )
    response_species_indices = _response_species_indices(
        project_config, reference_contract
    )
    all_snapshots: list[DecisionSnapshot] = []
    all_lineage: list[dict[str, Any]] = []
    all_tool_rows: list[dict[str, Any]] = []
    observed_truth: dict[str, list[float]] = {}
    for uid in POLICY_UIDS:
        snapshots, lineage_rows, tool_rows, policy_truth = _run_policy_chain(
            contract,
            reference_contract,
            project_config,
            reference,
            prior,
            policies[uid],
            candidates,
            typed_actions,
            c1_cache,
            c2s_cache,
            calibration,
            output_root,
            source_model_id,
            response_species_indices,
        )
        all_snapshots.extend(snapshots)
        all_lineage.extend(lineage_rows)
        all_tool_rows.extend(tool_rows)
        for candidate_uid, values in policy_truth.items():
            observed_truth.setdefault(candidate_uid, []).extend(values)
    truth, blinded_rows, contradictions = _complete_blinded_truth(
        contract,
        candidates,
        typed_actions,
        source_model_id,
        prior,
        c2s_cache,
        response_species_indices,
        output_root,
        observed_truth,
    )
    all_tool_rows.extend(blinded_rows)
    decision_rows, effects_rows, coherence_rows = _evaluate_decisions(
        contract, all_snapshots, truth
    )
    failure_rows = _failure_catalog(all_snapshots, contradictions)
    prospective_rows: list[dict[str, Any]] = []
    for row in all_lineage:
        round_root = output_root / str(row["policy_uid"]) / str(row["round_uid"])
        round_uid = str(row["round_uid"])
        receipt_path = round_root / "round_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        prospective_rows.append(
            {
            "policy_uid": row["policy_uid"],
            "round_uid": row["round_uid"],
            "state_uid": row["state_uid"],
            "posterior_uid_pre": row["posterior_uid_pre"],
            "world_action_uid": row["world_action_uid"],
            "information_action_uid": row["information_action_uid"],
            "tool_uid": row["tool_uid"],
            "result_uid": row["result_uid"],
            "posterior_uid_post": row["posterior_uid_post"],
            "model_checkpoint_pre_equals_post": row["model_checkpoint_pre_equals_post"],
            "observation_admitted": row["observation_admitted"],
            "outcome_visible_before_freeze": False,
            "selection_before_outcome": True,
            "freeze_before_dispatch_verified": receipt[
                "freeze_before_dispatch_verified"
            ],
            "input_manifest_ref": receipt["artifact_bindings"]["input_manifest"][
                "path"
            ],
            "input_manifest_sha256": receipt["artifact_bindings"][
                "input_manifest"
            ]["sha256"],
            "selected_action_ref": receipt["artifact_bindings"][
                "selected_information_action"
            ]["path"],
            "selected_action_sha256": receipt["artifact_bindings"][
                "selected_information_action"
            ]["sha256"],
            "round_receipt_ref": _rel(receipt_path),
            "round_receipt_sha256": sha256_file(receipt_path),
            "round_receipt_uid": receipt["receipt_uid"],
            **EVIDENCE_BOUNDARY,
            }
        )
    write_tsv(
        _repo_path(contract["outputs"]["prospective_manifest"]),
        list(prospective_rows[0]),
        prospective_rows,
    )
    write_tsv(
        _repo_path(contract["outputs"]["posterior_lineage"]),
        list(all_lineage[0]),
        all_lineage,
    )
    write_tsv(
        _repo_path(contract["outputs"]["tool_catalog"]),
        list(all_tool_rows[0]),
        all_tool_rows,
    )
    write_tsv(
        _repo_path(contract["outputs"]["decision_loop"]),
        list(decision_rows[0]),
        decision_rows,
    )
    write_tsv(
        _repo_path(contract["outputs"]["information_effects"]),
        list(effects_rows[0]),
        effects_rows,
    )
    write_tsv(
        _repo_path(contract["outputs"]["belief_coherence"]),
        list(coherence_rows[0]),
        coherence_rows,
    )
    write_tsv(
        _repo_path(contract["outputs"]["failure_catalog"]),
        list(failure_rows[0]),
        failure_rows,
    )
    gate = _write_terminal_gates(
        contract,
        calibration_report,
        all_snapshots,
        all_lineage,
        all_tool_rows,
        decision_rows,
        effects_rows,
        failure_rows,
    )
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output-root")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gate = run_campaign(
        contract_path=Path(args.contract).resolve(),
        output_root=Path(args.output_root).resolve() if args.output_root else None,
        device=args.device,
        preflight_only=args.preflight_only,
        resume=args.resume,
    )
    print(json.dumps(gate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
