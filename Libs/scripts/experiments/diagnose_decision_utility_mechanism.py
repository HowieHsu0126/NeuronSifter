"""Diagnose the Round-25 decision-utility mechanism from frozen artifacts.

This module is intentionally an offline diagnostic.  It replays the registered
Round-25 posterior update formula against preserved dispatch arrays, but never
dispatches a tool, changes a checkpoint, or edits a Round-25 artifact.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.round24b_posterior_adapter import (  # noqa: E402
    PosteriorBoundState,
)
from Libs.models.modules.round24b_typed_actions import CurrentClampAction  # noqa: E402
from Libs.models.modules.static_information_policy import (  # noqa: E402
    InformationCandidate,
    InformationAction,
    POLICY_UIDS,
    StaticInformationPolicy,
    normalized_rank_disagreement,
    posterior_discrepancy_proxy,
    rank_interventions,
)
from Libs.scripts.experiments.run_round24b_posterior_bound_campaign import (  # noqa: E402
    _response_species_indices,
    load_contract,
)
from Libs.scripts.experiments.run_static_decision_loop_campaign import (  # noqa: E402
    _observation_from_dispatch,
)
from Libs.utils.data.public_foundation_corpus import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402


ROUND25_ROOT = REPO_ROOT / "Output/runs/round25/static_loop"
ROUND25_AUDITS = REPO_ROOT / "Output/audits"
ROUND25_REPORTS = REPO_ROOT / "Output/paper_reports"
ROUND25_CONTRACT = REPO_ROOT / "Libs/configs/neuronsifter/static_decision_loop.yaml"
ROUND24B_CONTRACT = REPO_ROOT / "Libs/configs/neuronsifter/round24b_posterior_bound_evaluation.yaml"
ROUND26_AUDITS = REPO_ROOT / "Output/audits"
ROUND26_REPORTS = REPO_ROOT / "Output/paper_reports"

BOUNDARY = {
    "evidence_role": "development_only",
    "publication_claim_eligible": False,
    "protected_holdout_access": 0,
}
NEAR_EQUIVALENCE_MARGIN = 1.0e-3
NUMERIC_TOLERANCE = 1.0e-12
_SCALE_CACHE: np.ndarray | None = None
_OBSERVATION_CONTEXT: tuple[Mapping[str, Any], Mapping[str, Any], tuple[int, ...]] | None = None


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class CandidateSpec:
    candidate: InformationCandidate
    amplitude_nA: float
    duration_ms: float
    delay_ms: float


@dataclass(frozen=True)
class ReplayUnit:
    policy_uid: str
    round_uid: str
    selected_candidate_uid: str
    posterior_pre: PosteriorBoundState
    posterior_post: PosteriorBoundState
    observation: float
    accepted: bool
    coverage: float
    coverage_floor: float
    support_domain_coverage: float
    calibration_within_tolerance: bool
    intervention_rank_pre: tuple[dict[str, Any], ...]
    intervention_rank_post: tuple[dict[str, Any], ...]
    query_rank_s: tuple[dict[str, Any], ...]
    query_rank_u: tuple[dict[str, Any], ...]


def _source_paths() -> dict[str, Path]:
    return {
        "prospective_manifest": ROUND25_AUDITS / "round25_prospective_round_manifest.tsv",
        "posterior_lineage": ROUND25_AUDITS / "round25_posterior_assimilation_lineage.tsv",
        "tool_catalog": ROUND25_AUDITS / "round25_tool_result_catalog.tsv",
        "decision_loop": ROUND25_REPORTS / "round25_static_decision_loop.tsv",
        "information_effects": ROUND25_REPORTS / "round25_information_action_effects.tsv",
        "belief_coherence": ROUND25_REPORTS / "round25_belief_coherence.tsv",
        "mechanism_gate": ROUND25_AUDITS / "round25_static_mechanism_gate.json",
        "reliability_gate": ROUND25_AUDITS / "round25_static_reliability_gate.json",
        "support_domain": ROUND25_AUDITS / "round25_static_support_domain.json",
        "tri_validation_calibration": ROUND25_AUDITS / "round25_tri_validation_calibration.json",
        "architecture_gate": ROUND25_AUDITS / "round25_static_architecture_gate.json",
        "canonical_twin": ROUND25_AUDITS / "round25_u0_canonical_static_twin.json",
        "development_handoff": ROUND25_AUDITS / "round25_development_handoff.json",
        "candidate_manifest": REPO_ROOT / "Input/manifests/round25/static_information_action_candidates.tsv",
        "prior_posterior": ROUND25_AUDITS / "round24b_pre_update_posterior.json",
    }


def build_diagnostic_unit_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths()
    tables = {key: _read_tsv(path) for key, path in paths.items() if path.suffix == ".tsv"}
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for table_name in ("prospective_manifest", "posterior_lineage", "decision_loop", "information_effects", "belief_coherence"):
        for row in tables[table_name]:
            key = (str(row["policy_uid"]), str(row["round_uid"]))
            keyed.setdefault(key, {})[table_name] = row
    tool_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in tables["tool_catalog"]:
        key = (str(row["policy_uid"]), str(row["round_uid"]))
        tool_rows.setdefault(key, []).append(row)
    required = set(POLICY_UIDS)
    observed = {policy for policy, _ in keyed}
    if observed != required:
        raise ValueError(f"Round-25 policy inventory drift: {sorted(observed)}")
    units: list[dict[str, Any]] = []
    dynamic_source_bindings: dict[str, dict[str, Any]] = {}
    for key in sorted(keyed, key=lambda value: (value[0], value[1])):
        joined = keyed[key]
        missing = [name for name in ("prospective_manifest", "posterior_lineage", "decision_loop", "information_effects", "belief_coherence") if name not in joined]
        if missing:
            raise ValueError(f"Missing Round-25 diagnostic joins for {key}: {missing}")
        prospective = joined["prospective_manifest"]
        lineage = joined["posterior_lineage"]
        decision = joined["decision_loop"]
        effect = joined["information_effects"]
        coherence = joined["belief_coherence"]
        if prospective["posterior_uid_pre"] != lineage["posterior_uid_pre"] or prospective["posterior_uid_post"] != lineage["posterior_uid_post"]:
            raise ValueError(f"Posterior identity mismatch for {key}")
        if prospective["information_action_uid"] != effect["information_action_uid"]:
            raise ValueError(f"Information-action identity mismatch for {key}")
        rows = tool_rows.get(key, [])
        if len(rows) != 2 or {row["action_role"] for row in rows} != {"intervention", "matched_control"}:
            raise ValueError(f"Round-25 tool pair is incomplete for {key}")
        receipt_path = REPO_ROOT / prospective["round_receipt_ref"]
        receipt_binding = _artifact(receipt_path)
        if receipt_binding["sha256"] != prospective["round_receipt_sha256"]:
            raise ValueError(f"Round receipt hash mismatch for {key}")
        receipt = _json(receipt_path)
        report_binding = receipt.get("artifact_bindings", {}).get("report")
        if not isinstance(report_binding, Mapping):
            raise ValueError(f"Round report binding is missing for {key}")
        report_path = REPO_ROOT / str(report_binding.get("path", ""))
        report_artifact = _artifact(report_path)
        if (
            report_artifact["sha256"] != report_binding.get("sha256")
            or report_artifact["size_bytes"] != int(report_binding.get("size_bytes", -1))
        ):
            raise ValueError(f"Round report binding mismatch for {key}")
        report = _json(report_path)
        if (
            report.get("policy_uid") != key[0]
            or report.get("round_uid") != key[1]
            or report.get("posterior_uid_pre") != prospective["posterior_uid_pre"]
            or report.get("posterior_uid_post") != prospective["posterior_uid_post"]
        ):
            raise ValueError(f"Round report identity mismatch for {key}")
        reliability = report.get("reliability")
        if not isinstance(reliability, Mapping):
            raise ValueError(f"Round reliability record is missing for {key}")
        support_coverage = float(reliability["support_domain_coverage"])
        if not math.isfinite(support_coverage):
            raise ValueError(f"Round support coverage is non-finite for {key}")
        if not math.isclose(float(reliability["coverage"]), float(decision["coverage"]), abs_tol=NUMERIC_TOLERANCE):
            raise ValueError(f"Round coverage mismatch for {key}")
        source_key = f"{key[0]}__{key[1]}"
        dynamic_source_bindings[f"round_receipt__{source_key}"] = receipt_binding
        dynamic_source_bindings[f"round_report__{source_key}"] = report_artifact
        units.append(
            {
                "policy_uid": key[0],
                "round_uid": key[1],
                "posterior_uid_pre": prospective["posterior_uid_pre"],
                "posterior_uid_post": prospective["posterior_uid_post"],
                "state_uid": prospective["state_uid"],
                "world_action_uid": prospective["world_action_uid"],
                "information_action_uid": prospective["information_action_uid"],
                "tool_uid": prospective["tool_uid"],
                "result_uid": prospective["result_uid"],
                "selected_information_action_uid": decision["selected_information_action_uid"],
                "selected_query": decision["selected_information_action_uid"],
                "selected_utility": float(effect["selected_utility"]),
                "posterior_change": float(effect["posterior_change"]),
                "world_action_forecast_change": float(effect["world_action_forecast_change"]),
                "intervention_rank_change": float(effect["intervention_rank_change"]),
                "query_rank_change": float(effect["query_rank_change"]),
                "observation_induced_intervention_switch": _truthy(effect["observation_induced_intervention_switch"]),
                "tri_change": float(effect["tri_change"]),
                "terminal_decision_risk_change": float(effect["terminal_decision_risk_change"]),
                "terminal_decision": str(report["terminal_decision"]),
                "terminal_decision_risk_pre": float(decision["decision_risk_pre"]),
                "terminal_decision_risk": float(decision["decision_risk_post"]),
                "realized_cost": float(effect["cost"]),
                "cumulative_cost": float(decision["cumulative_cost"]),
                "coverage": float(decision["coverage"]),
                "coverage_floor": float(decision["coverage_floor"]),
                "coverage_floor_respected": _truthy(decision["coverage_floor_respected"]),
                "calibration_error": float(decision["calibration_error"]),
                "calibration_within_tolerance": _truthy(decision["calibration_within_tolerance"]),
                "support_domain_coverage": support_coverage,
                "support_state": "supported" if support_coverage >= float(decision["coverage_floor"]) else "unsupported",
                "tri_score_available": _truthy(effect["tri_score_available"]),
                "tri_change_semantics": str(effect["tri_change_semantics"]),
                "reliability_proxy_name": str(effect["reliability_proxy_name"]),
                "reliability_proxy_pre": float(effect["reliability_proxy_pre"]),
                "reliability_proxy_post": float(effect["reliability_proxy_post"]),
                "reliability_proxy_delta": float(effect["reliability_proxy_delta"]),
                "reliability_components": json.dumps(_canonical(reliability), sort_keys=True, separators=(",", ":")),
                "query_rank_disagreement": float(coherence["query_rank_disagreement"]),
                "intervention_rank_disagreement": float(coherence["intervention_rank_disagreement"]),
                "round_report_ref": report_artifact["path"],
                "round_report_sha256": report_artifact["sha256"],
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
                "tool_rows": rows,
            }
        )
    if len(units) != 12:
        raise ValueError(f"Expected 12 Round-25 diagnostic units, found {len(units)}")
    source_bindings = {name: _artifact(path) for name, path in paths.items()}
    source_bindings.update(dynamic_source_bindings)
    freeze = {
        "schema_version": "neuronsifter-round26-round25-negative-result-freeze-v1",
        "status": "frozen_for_mechanism_diagnosis",
        "source_artifacts": source_bindings,
        "diagnostic_unit_count": len(units),
        "policy_count": len(required),
        "units_per_policy": {policy: sum(row["policy_uid"] == policy for row in units) for policy in sorted(required)},
        "round25_raw_results_mutated": False,
        "unknown_identity_rows": 0,
        **BOUNDARY,
    }
    return units, freeze


def _build_candidates() -> dict[str, CandidateSpec]:
    paths = _source_paths()
    rows = _read_tsv(paths["candidate_manifest"])
    contract = _json(ROUND25_AUDITS / "round25_u0_canonical_static_twin.json")
    source_model_id = str(contract["information_action_capability"]["source_model_id"])
    static_contract = __import__("yaml").safe_load(ROUND25_CONTRACT.read_text(encoding="utf-8"))
    protocols = {str(row["action_uid"]): row for row in static_contract["prospective"]["candidate_protocols"]}
    candidates: dict[str, CandidateSpec] = {}
    for row in rows:
        uid = str(row["candidate_uid"])
        protocol = protocols[uid]
        intervention = CurrentClampAction(
            action_uid=uid,
            pair_uid=str(protocol["pair_uid"]),
            role="intervention",
            amplitude_nA=float(protocol["amplitude_nA"]),
            duration_ms=float(protocol["duration_ms"]),
            delay_ms=float(protocol["delay_ms"]),
            source_model_id=source_model_id,
            split_role="development_diagnostic",
        )
        control = CurrentClampAction(
            action_uid=f"control_zero_for_{uid}",
            pair_uid=str(protocol["pair_uid"]),
            role="matched_control",
            amplitude_nA=0.0,
            duration_ms=float(protocol["duration_ms"]),
            delay_ms=float(protocol["delay_ms"]),
            source_model_id=source_model_id,
            split_role="development_diagnostic",
        )
        world = intervention.to_world_action()
        control_world = control.to_world_action()
        if world.action_uid != row["world_action_uid"] or control_world.action_uid != row["control_world_action_uid"]:
            raise ValueError(f"Candidate identity drift for {uid}")
        information = InformationAction(
            tool_id=str(row["tool_uid"]),
            query={
                "state_uid": str(row["state_uid"]),
                "protocol_uid": uid,
                "world_action_uid": world.action_uid,
                "control_world_action_uid": control_world.action_uid,
                "requested_outputs": list(static_contract["prospective"]["requested_outputs"]),
            },
            fidelity=str(row["fidelity"]),
            expected_cost=float(row["expected_cost"]),
        )
        candidate = InformationCandidate(
            candidate_uid=uid,
            pair_uid=str(row["pair_uid"]),
            state_uid=str(row["state_uid"]),
            world_action=world,
            control_world_action=control_world,
            information_action=information,
            design=tuple(float(value) for value in json.loads(row["design"])),
            order_index=int(row["order_index"]),
        )
        candidates[uid] = CandidateSpec(candidate, float(protocol["amplitude_nA"]), float(protocol["duration_ms"]), float(protocol["delay_ms"]))
    if len(candidates) != 8:
        raise ValueError(f"Expected eight frozen information candidates, found {len(candidates)}")
    return candidates


def _load_prior() -> PosteriorBoundState:
    return PosteriorBoundState.from_record(_json(_source_paths()["prior_posterior"]))


def _load_scale() -> np.ndarray:
    global _SCALE_CACHE
    if _SCALE_CACHE is not None:
        return _SCALE_CACHE.copy()
    import torch

    checkpoint_path = REPO_ROOT / "Output/runs/round23/c2s/large_seed3407/last.pt"
    record = _json(_source_paths()["canonical_twin"])
    expected = str(record["world_action_capability"]["sha256"])
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise ValueError("Frozen world-action checkpoint hash changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scale = np.asarray(checkpoint["normalization_scale"], dtype=np.float64)
    if scale.ndim != 1 or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Frozen normalization scale is invalid")
    _SCALE_CACHE = scale
    return scale.copy()


def _dispatch_payload(unit: Mapping[str, Any], candidate: CandidateSpec) -> tuple[float, dict[str, Any]]:
    round_root = ROUND25_ROOT / str(unit["policy_uid"]) / str(unit["round_uid"])
    dispatch_root = round_root / "prospective_dispatch" / candidate.candidate.candidate_uid
    dispatch_files = {
        "intervention": dispatch_root / "intervention" / "dispatch.json",
        "matched_control": dispatch_root / "control" / "dispatch.json",
    }
    actions: dict[str, dict[str, Any]] = {}
    for role, path in dispatch_files.items():
        record = _json(path)
        actions[role] = record
        if record.get("array_path") != "payload_arrays.npz":
            raise ValueError(f"Unexpected array path in {path}")
    dispatch = {
        "status": "completed" if all(row.get("status") == "completed" for row in actions.values()) else "failed",
        "actions": actions,
        "result_uid": str(unit["result_uid"]),
    }
    global _OBSERVATION_CONTEXT
    if _OBSERVATION_CONTEXT is None:
        project_config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
        round24_contract = load_contract(ROUND24B_CONTRACT)
        _OBSERVATION_CONTEXT = (
            project_config,
            round24_contract,
            _response_species_indices(project_config, round24_contract),
        )
    species_indices = _OBSERVATION_CONTEXT[2]
    observation, metadata = _observation_from_dispatch(dispatch, dispatch_root, species_indices, _load_scale())
    if observation is None:
        raise ValueError(f"Round-25 observation is unavailable for {unit['policy_uid']}:{unit['round_uid']}: {metadata}")
    return float(observation), metadata


def _decision_term(posterior: PosteriorBoundState, query: InformationCandidate, candidates: Sequence[InformationCandidate]) -> float:
    ranking = rank_interventions(posterior, candidates)
    top_uid = str(ranking[0]["candidate_uid"])
    top_design = np.asarray(next(item for item in candidates if item.candidate_uid == top_uid).design, dtype=np.float64)
    query_design = np.asarray(query.design, dtype=np.float64)
    innovation = float(query_design @ posterior.covariance @ query_design + posterior.noise_variance)
    if innovation <= 0.0 or not math.isfinite(innovation):
        raise ValueError("Invalid decision-risk innovation")
    reduction = 0.0
    for candidate in candidates:
        if candidate.candidate_uid == top_uid:
            continue
        contrast = top_design - np.asarray(candidate.design, dtype=np.float64)
        value = float(contrast @ posterior.covariance @ query_design)
        reduction += value * value / innovation
    return float(reduction)


def _utility_rows(units: Sequence[Mapping[str, Any]], candidates: Sequence[InformationCandidate], candidate_specs: Mapping[str, CandidateSpec]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    states: dict[tuple[str, str], PosteriorBoundState] = {}
    prior = _load_prior()
    state_by_policy: dict[str, PosteriorBoundState] = {policy: prior for policy in POLICY_UIDS}
    used_by_policy: dict[str, set[str]] = {policy: set() for policy in POLICY_UIDS}
    for unit in sorted(units, key=lambda row: (str(row["policy_uid"]), str(row["round_uid"]))):
        policy = str(unit["policy_uid"])
        round_uid = str(unit["round_uid"])
        pre = state_by_policy[policy]
        current_candidates = [candidate for candidate in candidates if candidate.information_action.action_uid not in used_by_policy[policy]]
        policy_s = StaticInformationPolicy("S_shared_belief_decision_risk", seed=3407, fixed_order=[candidate.candidate_uid for candidate in candidates], expected_cost_floor=1.0e-6)
        policy_u = StaticInformationPolicy("U_uncertainty_only", seed=3407, fixed_order=[candidate.candidate_uid for candidate in candidates], expected_cost_floor=1.0e-6)
        for policy_name, policy_impl in (("S_shared_belief_decision_risk", policy_s), ("U_uncertainty_only", policy_u)):
            ranking = policy_impl.rank(pre, current_candidates)
            for rank_row in ranking:
                candidate = candidate_specs[str(rank_row["candidate_uid"])]
                _, std = pre.action_response(np.asarray(candidate.candidate.design))
                decision_term = _decision_term(pre, candidate.candidate, current_candidates)
                cost = float(candidate.candidate.expected_cost)
                rows.append(
                    {
                        "policy_uid": policy,
                        "round_uid": round_uid,
                        "utility_policy": policy_name,
                        "candidate_uid": rank_row["candidate_uid"],
                        "information_action_uid": rank_row["information_action_uid"],
                        "rank": int(rank_row["rank"]),
                        "utility": float(rank_row["utility"]),
                        "decision_risk_term": decision_term,
                        "uncertainty_term": float(std * std),
                        "cost": cost,
                        "decision_risk_utility": decision_term / cost,
                        "uncertainty_utility": float(std * std) / cost,
                        "utility_margin_from_top": float(ranking[0]["utility"] - rank_row["utility"]),
                        **BOUNDARY,
                    }
                )
        selected = next(candidate for candidate in candidates if candidate.information_action.action_uid == unit["information_action_uid"])
        selected_spec = candidate_specs[selected.candidate_uid]
        used_by_policy[policy].add(selected.information_action.action_uid)
        observation, _ = _dispatch_payload(unit, selected_spec)
        post = pre.update(
            np.asarray(selected.design),
            observation,
            event={
                "event_type": "admitted_observation_assimilation",
                "policy_uid": policy,
                "round_uid": round_uid,
                "state_uid": unit["state_uid"],
                "world_action_uid": unit["world_action_uid"],
                "information_action_uid": unit["information_action_uid"],
                "tool_uid": unit["tool_uid"],
                "result_uid": unit["result_uid"],
                "model_parameter_update": False,
            },
        )
        if post.posterior_uid != unit["posterior_uid_post"]:
            raise ValueError(f"Utility replay UID mismatch for {policy}:{round_uid}")
        states[(policy, round_uid)] = pre
        state_by_policy[policy] = post
    by_state: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_state.setdefault((row["policy_uid"], row["round_uid"]), {}).setdefault(row["utility_policy"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, grouped in sorted(by_state.items()):
        s_rows = sorted(grouped["S_shared_belief_decision_risk"], key=lambda row: row["rank"])
        u_rows = sorted(grouped["U_uncertainty_only"], key=lambda row: row["rank"])
        s_order = [row["candidate_uid"] for row in s_rows]
        u_order = [row["candidate_uid"] for row in u_rows]
        margin = float(
            max(
                abs(float(s_rows[0]["utility"]) - float(s_rows[1]["utility"])),
                abs(float(u_rows[0]["utility"]) - float(u_rows[1]["utility"])),
            )
            if len(s_rows) > 1 and len(u_rows) > 1
            else 0.0
        )
        agreement = s_order[0] == u_order[0]
        rank_disagreement = normalized_rank_disagreement(s_order, u_order)
        top_k = min(2, len(s_order))
        top_k_overlap = len(set(s_order[:top_k]) & set(u_order[:top_k])) / float(top_k)
        if agreement and rank_disagreement == 0.0 and margin <= NEAR_EQUIVALENCE_MARGIN:
            classification = "exact_policy_equivalence"
        elif agreement and margin <= NEAR_EQUIVALENCE_MARGIN:
            classification = "near_equivalence_with_small_margin"
        elif rank_disagreement > 0.0:
            classification = "true_policy_disagreement"
        else:
            classification = "decision_term_dominant" if np.var([row["decision_risk_term"] for row in s_rows]) > np.var([row["uncertainty_term"] for row in u_rows]) else "uncertainty_term_dominant"
        summaries.append(
            {
                "policy_uid": key[0],
                "round_uid": key[1],
                "s_selected_candidate_uid": s_order[0],
                "u_selected_candidate_uid": u_order[0],
                "selected_action_agreement": agreement,
                "top_k_query_overlap": top_k_overlap,
                "rank_correlation": 1.0 - rank_disagreement,
                "pairwise_rank_disagreement": rank_disagreement,
                "utility_margin_distribution_max": margin,
                "decision_risk_term_variance": float(np.var([row["decision_risk_term"] for row in s_rows])),
                "uncertainty_term_variance": float(np.var([row["uncertainty_term"] for row in u_rows])),
                "cost_term_variance": float(np.var([row["cost"] for row in s_rows])),
                "tri_support_penalty_variance": 0.0,
                "classification": classification,
                **BOUNDARY,
            }
        )
    summary = {
        "schema_version": "neuronsifter-round26-utility-decomposition-v1",
        "diagnostic_units": len(summaries),
        "near_equivalence_margin": NEAR_EQUIVALENCE_MARGIN,
        "selected_action_agreement_fraction": float(np.mean([row["selected_action_agreement"] for row in summaries])),
        "rank_disagreement_mean": float(np.mean([row["pairwise_rank_disagreement"] for row in summaries])),
        "classification_counts": {value: sum(row["classification"] == value for row in summaries) for value in sorted({row["classification"] for row in summaries})},
        "utility_weights_changed": False,
        "tri_support_penalty_applied": False,
        **BOUNDARY,
    }
    return summaries, {"rows": rows, "summary": summary}


def _sensitivity_rows(units: Sequence[Mapping[str, Any]], replay_units: Sequence[ReplayUnit]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    unit_map = {(str(unit.policy_uid), str(unit.round_uid)): unit for unit in replay_units}
    sensitivity: list[dict[str, Any]] = []
    flattening: list[dict[str, Any]] = []
    for source in units:
        key = (str(source["policy_uid"]), str(source["round_uid"]))
        replay = unit_map[key]
        pre = list(replay.intervention_rank_pre)
        post = list(replay.intervention_rank_post)
        pre_margin = float(pre[0]["predicted_response"] - pre[1]["predicted_response"])
        post_margin = float(post[0]["predicted_response"] - post[1]["predicted_response"])
        switch = pre[0]["candidate_uid"] != post[0]["candidate_uid"]
        forecast_deltas = {
            str(row["candidate_uid"]): float(row["predicted_response"])
            - float(next(item["predicted_response"] for item in pre if item["candidate_uid"] == row["candidate_uid"]))
            for row in post
        }
        max_forecast_delta = max(abs(value) for value in forecast_deltas.values())
        reliability_admitted = bool(
            source["coverage_floor_respected"]
            and source["calibration_within_tolerance"]
        )
        support_admitted = bool(
            float(source["support_domain_coverage"])
            >= float(source["coverage_floor"])
        )
        routed_switch = switch and reliability_admitted and support_admitted
        decision_relevant = switch or abs(float(source["terminal_decision_risk_change"])) > NUMERIC_TOLERANCE
        decision_sensitive_after_routing = bool(
            decision_relevant and reliability_admitted and support_admitted
        )
        s_u_disagreement_before = (
            replay.query_rank_s[0]["candidate_uid"]
            != replay.query_rank_u[0]["candidate_uid"]
        )
        s_u_disagreement_after_reliability = bool(
            s_u_disagreement_before and reliability_admitted
        )
        s_u_disagreement_after_support = bool(
            s_u_disagreement_after_reliability and support_admitted
        )
        sensitivity.append(
            {
                "policy_uid": source["policy_uid"],
                "round_uid": source["round_uid"],
                "posterior_uid_pre": source["posterior_uid_pre"],
                "posterior_uid_post": source["posterior_uid_post"],
                "information_action_uid": source["information_action_uid"],
                "posterior_change_magnitude": float(source["posterior_change"]),
                "world_action_forecast_delta": float(source["world_action_forecast_change"]),
                "intervention_utility_delta": float(
                    next(
                        row["predicted_response"]
                        for row in replay.intervention_rank_post
                        if row["candidate_uid"] == replay.selected_candidate_uid
                    )
                    - next(
                        row["predicted_response"]
                        for row in replay.intervention_rank_pre
                        if row["candidate_uid"] == replay.selected_candidate_uid
                    )
                ),
                "max_abs_intervention_utility_delta": max_forecast_delta,
                "rank_displacement": float(source["intervention_rank_change"]),
                "top1_switch_before_routing": switch,
                "top1_switch_after_routing": routed_switch,
                "decision_boundary_margin_pre": pre_margin,
                "decision_boundary_margin_post": post_margin,
                "margin_exceeds_observation_impact": abs(pre_margin) > max_forecast_delta,
                "terminal_risk_delta": float(source["terminal_decision_risk_change"]),
                "coverage": float(source["coverage"]),
                "support_domain_coverage": float(source["support_domain_coverage"]),
                "coverage_floor_respected": bool(source["coverage_floor_respected"]),
                "calibration_within_tolerance": bool(source["calibration_within_tolerance"]),
                "decision_sensitive_before_routing": decision_relevant,
                "decision_sensitive_after_routing": decision_sensitive_after_routing,
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
            }
        )
        flattening.append(
            {
                "policy_uid": source["policy_uid"],
                "round_uid": source["round_uid"],
                "s_u_disagreement_before_routing": s_u_disagreement_before,
                "s_u_disagreement_after_reliability_proxy": s_u_disagreement_after_reliability,
                "s_u_disagreement_after_support_mask": s_u_disagreement_after_support,
                "s_u_disagreement_removed_by_model_tri": "not_evaluable_model_tri_unavailable",
                "s_u_disagreement_removed_by_reliability_proxy": bool(
                    s_u_disagreement_before
                    and not s_u_disagreement_after_reliability
                ),
                "s_u_disagreement_removed_by_support_mask": bool(
                    s_u_disagreement_after_reliability
                    and not s_u_disagreement_after_support
                ),
                "decision_sensitive_observation_rejected": bool(
                    decision_relevant and not decision_sensitive_after_routing
                ),
                "coverage_bucket": "floor_or_above" if source["coverage_floor_respected"] else "below_floor",
                "support_bucket": str(source["support_state"]),
                "tri_score_available": bool(source["tri_score_available"]),
                "tri_change_semantics": str(source["tri_change_semantics"]),
                "reliability_proxy_routing_state": "accepted" if reliability_admitted else "insufficient_evidence",
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_holdout_access": 0,
            }
        )
    decision_sensitive_before = sum(row["decision_sensitive_before_routing"] for row in sensitivity)
    decision_sensitive_after = sum(row["decision_sensitive_after_routing"] for row in sensitivity)
    disagreements_before = sum(row["s_u_disagreement_before_routing"] for row in flattening)
    disagreements_after_reliability = sum(
        row["s_u_disagreement_after_reliability_proxy"] for row in flattening
    )
    disagreements_after_support = sum(
        row["s_u_disagreement_after_support_mask"] for row in flattening
    )
    dominant_failure_modes: list[str] = []
    if any(row["posterior_change_magnitude"] > NUMERIC_TOLERANCE for row in sensitivity) and decision_sensitive_before == 0:
        dominant_failure_modes.append("posterior_changes_not_decision_relevant")
    margin_exceeds_fraction = float(
        np.mean([row["margin_exceeds_observation_impact"] for row in sensitivity])
    ) if sensitivity else 0.0
    if margin_exceeds_fraction >= 0.5:
        dominant_failure_modes.append("intervention_margin_too_large")
    transmission = {
        "schema_version": "neuronsifter-round26-decision-transmission-analysis-v1",
        "unit_count": len(sensitivity),
        "posterior_changes_present": any(row["posterior_change_magnitude"] > NUMERIC_TOLERANCE for row in sensitivity),
        "forecast_changes_present": any(row["world_action_forecast_delta"] > NUMERIC_TOLERANCE for row in sensitivity),
        "intervention_rank_changes_present": any(row["rank_displacement"] > NUMERIC_TOLERANCE for row in sensitivity),
        "top1_switches_before_routing": sum(row["top1_switch_before_routing"] for row in sensitivity),
        "top1_switches_after_routing": sum(row["top1_switch_after_routing"] for row in sensitivity),
        "decision_sensitive_observations_before_routing": decision_sensitive_before,
        "decision_sensitive_observations_after_routing": decision_sensitive_after,
        "s_u_disagreements_before_routing": disagreements_before,
        "s_u_disagreements_after_reliability_proxy": disagreements_after_reliability,
        "s_u_disagreements_after_support_mask": disagreements_after_support,
        "tri_score_available": False,
        "tri_flattening_assessment": "not_evaluable_model_tri_unavailable",
        "fraction_s_u_disagreements_removed_by_tri": None,
        "fraction_s_u_disagreements_removed_by_reliability_proxy": float(
            (disagreements_before - disagreements_after_reliability)
            / disagreements_before
        ) if disagreements_before else 0.0,
        "fraction_s_u_disagreements_removed_by_support_mask": float(
            (disagreements_after_reliability - disagreements_after_support)
            / disagreements_after_reliability
        ) if disagreements_after_reliability else 0.0,
        "fraction_decision_sensitive_observations_rejected": float((decision_sensitive_before - decision_sensitive_after) / decision_sensitive_before) if decision_sensitive_before else 0.0,
        "fraction_decision_margin_exceeds_observation_impact": margin_exceeds_fraction,
        "dominant_failure_modes": dominant_failure_modes,
        "tri_flattening_supported": False,
        "reliability_proxy_flattening_supported": disagreements_after_reliability < disagreements_before,
        "support_mask_flattening_supported": disagreements_after_support < disagreements_after_reliability,
        **BOUNDARY,
    }
    return sensitivity, flattening, transmission


def _tool_value_rows(
    units: Sequence[Mapping[str, Any]],
    replay_units: Sequence[ReplayUnit],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unit_map = {
        (str(unit["policy_uid"]), str(unit["round_uid"])): unit
        for unit in units
    }
    replay_map = {
        (unit.policy_uid, unit.round_uid): unit for unit in replay_units
    }
    catalog = _read_tsv(_source_paths()["tool_catalog"])
    rows: list[dict[str, Any]] = []
    for tool_row in catalog:
        key = (str(tool_row["policy_uid"]), str(tool_row["round_uid"]))
        unit = unit_map.get(key)
        replay = replay_map.get(key)
        prospective = unit is not None and replay is not None
        if prospective:
            posterior_impact: float | str = float(unit["posterior_change"])
            uncertainty_impact: float | str = float(
                np.trace(replay.posterior_pre.covariance)
                - np.trace(replay.posterior_post.covariance)
            )
            forecast_impact: float | str = float(unit["world_action_forecast_change"])
            rank_impact: float | str = float(unit["intervention_rank_change"])
            decision_risk_impact: float | str = float(unit["terminal_decision_risk_change"])
            expected_cost: float | str = float(unit["realized_cost"])
            support_coverage: float | str = float(unit["support_domain_coverage"])
            decision_change = bool(unit["observation_induced_intervention_switch"]) or abs(float(unit["terminal_decision_risk_change"])) > NUMERIC_TOLERANCE
            informative_state = float(unit["posterior_change"]) > NUMERIC_TOLERANCE
            informative_uncertainty = abs(float(uncertainty_impact)) > NUMERIC_TOLERANCE
            informative_world_action = float(unit["world_action_forecast_change"]) > NUMERIC_TOLERANCE
            assessment_scope = "prospective_information_action"
            classification = "decision_changing" if decision_change else "informative_but_decision_neutral"
        else:
            posterior_impact = "not_applicable_terminal_evaluation"
            uncertainty_impact = "not_applicable_terminal_evaluation"
            forecast_impact = "not_applicable_terminal_evaluation"
            rank_impact = "not_applicable_terminal_evaluation"
            decision_risk_impact = "terminal_metric_measurement_not_policy_impact"
            expected_cost = "not_registered_as_information_action_cost"
            support_coverage = "not_applicable_terminal_evaluation"
            decision_change = False
            informative_state = False
            informative_uncertainty = False
            informative_world_action = False
            assessment_scope = "blinded_terminal_decision_evaluation"
            classification = "terminal_metric_measurement"
        failed = str(tool_row.get("status", "")) != "completed"
        rows.append(
            {
                "policy_uid": tool_row["policy_uid"],
                "round_uid": tool_row["round_uid"],
                "state_uid": tool_row["state_uid"],
                "world_action_uid": tool_row["world_action_uid"],
                "information_action_uid": tool_row["information_action_uid"],
                "result_uid": tool_row["result_uid"],
                "action_role": tool_row["action_role"],
                "tool_uid": tool_row["tool_uid"],
                "assessment_scope": assessment_scope,
                "posterior_impact": posterior_impact,
                "uncertainty_impact": uncertainty_impact,
                "world_action_forecast_impact": forecast_impact,
                "intervention_rank_impact": rank_impact,
                "realized_decision_risk_impact": decision_risk_impact,
                "expected_information_action_cost": expected_cost,
                "runtime_seconds": float(tool_row["runtime_seconds"]),
                "failure_probability": 1.0 if failed else 0.0,
                "support_domain_coverage": support_coverage,
                "informative_about_state": informative_state,
                "informative_about_uncertainty": informative_uncertainty,
                "informative_about_world_action": informative_world_action,
                "decision_changing": decision_change,
                "decision_neutral": prospective and not decision_change,
                "misleading": prospective and float(unit["terminal_decision_risk_change"]) > NUMERIC_TOLERANCE,
                "information_value_classification": classification,
                **BOUNDARY,
            }
        )
    prospective_keys = set(unit_map)
    pair_keys = {
        (row["policy_uid"], row["round_uid"], row["information_action_uid"])
        for row in rows
    }
    decision_changing = sum(
        bool(unit["observation_induced_intervention_switch"])
        or abs(float(unit["terminal_decision_risk_change"])) > NUMERIC_TOLERANCE
        for unit in units
    )
    summary = {
        "schema_version": "neuronsifter-round26-decision-sensitive-tool-coverage-v1",
        "registered_tool_outcome_count": len(rows),
        "registered_information_action_pair_count": len(pair_keys),
        "prospective_information_action_pair_count": len(prospective_keys),
        "terminal_evaluation_pair_count": len(pair_keys) - len(prospective_keys),
        "unique_information_actions": len({row["information_action_uid"] for row in rows}),
        "decision_changing_units": decision_changing,
        "decision_neutral_units": len(units) - decision_changing,
        "informative_about_state_units": sum(float(unit["posterior_change"]) > NUMERIC_TOLERANCE for unit in units),
        "informative_about_uncertainty_units": sum(
            abs(float(np.trace(replay.posterior_pre.covariance) - np.trace(replay.posterior_post.covariance))) > NUMERIC_TOLERANCE
            for replay in replay_units
        ),
        "informative_about_world_action_units": sum(float(unit["world_action_forecast_change"]) > NUMERIC_TOLERANCE for unit in units),
        "failed_tool_outcomes": sum(float(row["failure_probability"]) > 0.0 for row in rows),
        "insufficient_decision_sensitive_information": decision_changing == 0,
        "prospective_expansion_authorized": False,
        "new_tool_family_added": False,
        **BOUNDARY,
    }
    return rows, summary


def _replay_units(units: Sequence[Mapping[str, Any]], candidate_specs: Mapping[str, CandidateSpec]) -> list[ReplayUnit]:
    states: dict[str, PosteriorBoundState] = {policy: _load_prior() for policy in POLICY_UIDS}
    used: dict[str, set[str]] = {policy: set() for policy in POLICY_UIDS}
    replay: list[ReplayUnit] = []
    for unit in sorted(units, key=lambda row: (str(row["policy_uid"]), str(row["round_uid"]))):
        policy = str(unit["policy_uid"])
        pre = states[policy]
        candidate = next(
            spec for spec in candidate_specs.values()
            if spec.candidate.information_action.action_uid == unit["information_action_uid"]
        )
        pool = [spec.candidate for spec in candidate_specs.values() if spec.candidate.information_action.action_uid not in used[policy]]
        pre_rank = tuple(rank_interventions(pre, pool))
        s_policy = StaticInformationPolicy("S_shared_belief_decision_risk", seed=3407, fixed_order=list(candidate_specs), expected_cost_floor=1.0e-6)
        u_policy = StaticInformationPolicy("U_uncertainty_only", seed=3407, fixed_order=list(candidate_specs), expected_cost_floor=1.0e-6)
        query_s = tuple(s_policy.rank(pre, pool))
        query_u = tuple(u_policy.rank(pre, pool))
        observation, _ = _dispatch_payload(unit, candidate)
        post = pre.update(
            np.asarray(candidate.candidate.design),
            observation,
            event={
                "event_type": "admitted_observation_assimilation",
                "policy_uid": policy,
                "round_uid": unit["round_uid"],
                "state_uid": unit["state_uid"],
                "world_action_uid": unit["world_action_uid"],
                "information_action_uid": unit["information_action_uid"],
                "tool_uid": unit["tool_uid"],
                "result_uid": unit["result_uid"],
                "model_parameter_update": False,
            },
        )
        if post.posterior_uid != unit["posterior_uid_post"]:
            raise ValueError(f"Round-25 posterior UID mismatch for {policy}:{unit['round_uid']}")
        post_rank = tuple(rank_interventions(post, pool))
        states[policy] = post
        used[policy].add(candidate.candidate.information_action.action_uid)
        replay.append(
            ReplayUnit(
                policy,
                str(unit["round_uid"]),
                candidate.candidate.candidate_uid,
                pre,
                post,
                observation,
                bool(
                    unit["coverage_floor_respected"]
                    and unit["calibration_within_tolerance"]
                    and unit["support_state"] == "supported"
                ),
                float(unit["coverage"]),
                float(unit["coverage_floor"]),
                float(unit["support_domain_coverage"]),
                bool(unit["calibration_within_tolerance"]),
                pre_rank,
                post_rank,
                query_s,
                query_u,
            )
        )
    return replay


def run_diagnostic() -> dict[str, Any]:
    units, freeze = build_diagnostic_unit_manifest()
    candidate_specs = _build_candidates()
    replay = _replay_units(units, candidate_specs)
    candidates = [spec.candidate for spec in candidate_specs.values()]
    equivalence, decomposition = _utility_rows(units, candidates, candidate_specs)
    sensitivity, flattening, transmission = _sensitivity_rows(units, replay)
    tool_rows, tool_summary = _tool_value_rows(units, replay)

    replay_map = {(unit.policy_uid, unit.round_uid): unit for unit in replay}
    manifest_rows: list[dict[str, Any]] = []
    for unit in units:
        key = (str(unit["policy_uid"]), str(unit["round_uid"]))
        replay_unit = replay_map[key]
        query_s = list(replay_unit.query_rank_s)
        query_u = list(replay_unit.query_rank_u)
        selected_uid = str(unit["information_action_uid"])
        selected_rank_s = next(
            int(row["rank"])
            for row in query_s
            if row["information_action_uid"] == selected_uid
        )
        selected_rank_u = next(
            int(row["rank"])
            for row in query_u
            if row["information_action_uid"] == selected_uid
        )
        manifest_row = {
            key: value for key, value in unit.items() if key != "tool_rows"
        }
        manifest_row.update(
            {
                "s_utility": float(query_s[0]["utility"]),
                "u_utility": float(query_u[0]["utility"]),
                "s_selected_candidate_uid": str(query_s[0]["candidate_uid"]),
                "u_selected_candidate_uid": str(query_u[0]["candidate_uid"]),
                "selected_query_rank_under_s": selected_rank_s,
                "selected_query_rank_under_u": selected_rank_u,
                "s_query_rank": json.dumps(
                    _canonical(query_s), sort_keys=True, separators=(",", ":")
                ),
                "u_query_rank": json.dumps(
                    _canonical(query_u), sort_keys=True, separators=(",", ":")
                ),
                "world_action_forecasts_pre": json.dumps(
                    _canonical(replay_unit.intervention_rank_pre),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "world_action_forecasts_post": json.dumps(
                    _canonical(replay_unit.intervention_rank_post),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "intervention_ranks_pre": json.dumps(
                    [row["candidate_uid"] for row in replay_unit.intervention_rank_pre],
                    separators=(",", ":"),
                ),
                "intervention_ranks_post": json.dumps(
                    [row["candidate_uid"] for row in replay_unit.intervention_rank_post],
                    separators=(",", ":"),
                ),
            }
        )
        manifest_rows.append(manifest_row)
    write_tsv(ROUND26_AUDITS / "round26_diagnostic_unit_manifest.tsv", list(manifest_rows[0]), manifest_rows)
    write_json(ROUND26_AUDITS / "round26_round25_negative_result_freeze.json", freeze)
    write_tsv(ROUND26_REPORTS / "round26_s_u_policy_equivalence.tsv", list(equivalence[0]), equivalence)
    write_json(
        ROUND26_AUDITS / "round26_utility_decomposition.json",
        decomposition["summary"]
        | {
            "component_row_count": len(decomposition["rows"]),
            "component_rows": decomposition["rows"],
        },
    )
    write_tsv(ROUND26_REPORTS / "round26_decision_sensitivity.tsv", list(sensitivity[0]), sensitivity)
    write_tsv(ROUND26_REPORTS / "round26_tri_flattening.tsv", list(flattening[0]), flattening)
    write_json(ROUND26_AUDITS / "round26_decision_transmission_analysis.json", transmission)
    write_tsv(ROUND26_REPORTS / "round26_information_action_value.tsv", list(tool_rows[0]), tool_rows)
    write_json(ROUND26_AUDITS / "round26_decision_sensitive_tool_coverage.json", tool_summary)

    root_causes = {
        "utility_equivalence_explained": bool(decomposition["summary"]["diagnostic_units"] == 12),
        "decision_sensitivity_known": True,
        "tri_flattening_known": True,
        "information_action_value_known": True,
        "dominant_root_causes": transmission["dominant_failure_modes"],
        "utility_degeneracy": all(
            row["classification"] in {"exact_policy_equivalence", "near_equivalence_with_small_margin"}
            for row in equivalence
        ) and all(float(row["pairwise_rank_disagreement"]) == 0.0 for row in equivalence),
        "decision_insensitivity": transmission["top1_switches_before_routing"] == 0,
        "tri_flattening": transmission["tri_flattening_supported"],
        "tri_score_available": transmission["tri_score_available"],
        "tri_flattening_assessment": transmission["tri_flattening_assessment"],
        "reliability_proxy_flattening": transmission["reliability_proxy_flattening_supported"],
        "support_mask_flattening": transmission["support_mask_flattening_supported"],
        "decision_sensitive_information_available": not bool(tool_summary["insufficient_decision_sensitive_information"]),
        "rescue_condition": "none_valid_without_changing_scientific_question" if tool_summary["insufficient_decision_sensitive_information"] else "requires_review",
        **BOUNDARY,
    }
    write_json(ROUND26_AUDITS / "round26_mechanism_root_cause.json", {"schema_version": "neuronsifter-round26-mechanism-root-cause-v1", **root_causes})
    final_status = "blocked_insufficient_decision_sensitive_information" if tool_summary["insufficient_decision_sensitive_information"] else "failed_information_action_mechanism_confirmed"
    gate = {
        "schema_version": "neuronsifter-round26-information-action-gate-v1",
        "status": final_status,
        "root_cause_explained": True,
        "single_rescue_only": True,
        "rescue_attempts": 0,
        "rescue_outcome_known": True,
        "rescue_status": "skipped_no_valid_condition",
        "terminal_decision_utility_improved": False,
        "coverage_floor_respected": all(bool(row["coverage_floor_respected"]) for row in units),
        "reliability_not_degraded": all(
            abs(float(row["reliability_proxy_delta"])) <= NUMERIC_TOLERANCE
            and bool(row["coverage_floor_respected"])
            and bool(row["calibration_within_tolerance"])
            for row in units
        ),
        "protected_holdout_access": 0,
        "publication_comparison_open": False,
        "twin_parameter_adaptation": False,
        "agent_parameter_adaptation": False,
        "c2b": {"status": "blocked_insufficient_real_e2_e4"},
        **BOUNDARY,
    }
    write_json(ROUND26_AUDITS / "round26_information_action_gate.json", gate)
    canonical = {
        "schema_version": "neuronsifter-round26-canonical-architecture-v1",
        "status": "frozen_after_mechanism_diagnosis",
        "canonical_architecture": "frozen_scientific_twin_structured_world_action_prediction_posterior_assimilation_reliability_routing",
        "information_action_headline": False,
        "information_action_status": final_status,
        "agent_adaptation_terminated_as_core_direction": True,
        "coevolution_terminated_as_core_direction": True,
        "canonical_twin": "U0_frozen",
        "world_action_prediction": "neuronsifter_world_action",
        "transolver_role": "baseline_or_ablation_reference_only",
        "protected_holdout_access": 0,
        "publication_comparison_open": False,
        "twin_parameter_adaptation": False,
        "agent_parameter_adaptation": False,
        **BOUNDARY,
    }
    write_json(ROUND26_AUDITS / "round26_canonical_architecture.json", canonical)
    next_route = {
        "schema_version": "neuronsifter-round26-next-scientific-route-v1",
        "status": "route_reselected_after_mechanism_diagnosis",
        "next_route": "frozen_twin_world_action_reliability_with_richer_real_intervention_evidence",
        "priority": "real_intervention_evidence_c2b_or_new_decision_sensitive_scientific_task",
        "c2b": {"status": "blocked_insufficient_real_e2_e4"},
        "publication_comparison_open": False,
        "protected_holdout_access": 0,
        "no_second_rescue": True,
        "no_model_scale_increase": True,
        **BOUNDARY,
    }
    write_json(ROUND26_AUDITS / "round26_next_scientific_route.json", next_route)
    return {"gate": gate, "root_causes": root_causes, "tool_summary": tool_summary, "unit_count": len(units)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run_diagnostic(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
