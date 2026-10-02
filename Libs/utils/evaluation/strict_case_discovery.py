from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Libs.utils.data.raw_data_registry import resolve_repo_path


SUPPORTED_STRICT_CASE_CHECKS = frozenset(
    {
        "target_or_compound_anchor_in_top_k",
        "tri_accepted_anchor_present",
    }
)


def normalize_anchor_values(values: list[object]) -> list[str]:
    return [
        str(value).strip()
        for value in values
        if str(value).strip()
    ]


def uppercase_anchor_values(values: list[object]) -> set[str]:
    return {
        str(value).strip().upper()
        for value in values
        if str(value).strip()
    }


def strict_case_entries(config: dict, disease_program: str = "") -> list[dict[str, object]]:
    strict_rows: list[dict[str, object]] = []
    for registry_key, case_config in sorted(config.get("case_study_registry", {}).items()):
        reporting_mode = str(case_config.get("reporting_mode", "")).strip()
        if reporting_mode != "strict_discovery":
            continue
        case_disease_program = str(case_config.get("disease_program", "")).strip()
        if disease_program and case_disease_program != disease_program:
            continue
        expected_checks = normalize_anchor_values(case_config.get("expected_ranking_checks", []))
        unknown_checks = sorted(set(expected_checks) - SUPPORTED_STRICT_CASE_CHECKS)
        if unknown_checks:
            raise ValueError(
                f"Unknown strict case-study checks for {registry_key}: {', '.join(unknown_checks)}"
            )
        strict_rows.append(
            {
                "registry_key": registry_key,
                "case_study_name": str(case_config.get("case_study_name", registry_key)).strip(),
                "disease_program": case_disease_program,
                "anchor_compounds": normalize_anchor_values(case_config.get("anchor_compounds", [])),
                "anchor_targets": normalize_anchor_values(case_config.get("anchor_targets", [])),
                "anchor_compounds_upper": uppercase_anchor_values(case_config.get("anchor_compounds", [])),
                "anchor_targets_upper": uppercase_anchor_values(case_config.get("anchor_targets", [])),
                "anchor_references": normalize_anchor_values(case_config.get("anchor_references", [])),
                "top_k": int(case_config.get("top_k", 0)),
                "expected_ranking_checks": expected_checks,
                "reporting_mode": reporting_mode,
                "forbid_anchor_forced_include": bool(case_config.get("forbid_anchor_forced_include", False)),
            }
        )
    return strict_rows


def action_ready_policy_payload(config: dict) -> dict[str, object]:
    action_config = config["screening_pipeline"]["action_ready_registry"]
    return {
        "action_ready_registry": {
            "minimum_atlas_program_coverage": float(action_config["minimum_atlas_program_coverage"]),
            "minimum_atlas_program_feature_hits": int(action_config["minimum_atlas_program_feature_hits"]),
            "maximum_single_compounds": int(action_config["maximum_single_compounds"]),
            "maximum_pairwise_compounds": int(action_config["maximum_pairwise_compounds"]),
            "maximum_pairwise_regimens": int(action_config["maximum_pairwise_regimens"]),
            "forced_include_compounds": normalize_anchor_values(action_config.get("forced_include_compounds", [])),
            "forced_include_targets": normalize_anchor_values(action_config.get("forced_include_targets", [])),
        },
        "strict_case_policies": [
            {
                "case_study_name": str(case_row["case_study_name"]),
                "disease_program": str(case_row["disease_program"]),
                "anchor_compounds": list(case_row["anchor_compounds"]),
                "anchor_targets": list(case_row["anchor_targets"]),
                "top_k": int(case_row["top_k"]),
                "expected_ranking_checks": list(case_row["expected_ranking_checks"]),
                "reporting_mode": str(case_row["reporting_mode"]),
                "forbid_anchor_forced_include": bool(case_row["forbid_anchor_forced_include"]),
            }
            for case_row in strict_case_entries(config)
        ],
    }


def action_ready_policy_hash(config: dict) -> str:
    payload = action_ready_policy_payload(config)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def forced_anchor_overlap_rows(config: dict) -> list[dict[str, object]]:
    action_config = config["screening_pipeline"]["action_ready_registry"]
    forced_compounds = uppercase_anchor_values(action_config.get("forced_include_compounds", []))
    forced_targets = uppercase_anchor_values(action_config.get("forced_include_targets", []))
    overlap_rows: list[dict[str, object]] = []
    for case_row in strict_case_entries(config):
        if not case_row["forbid_anchor_forced_include"]:
            continue
        compound_overlap = sorted(case_row["anchor_compounds_upper"] & forced_compounds)
        target_overlap = sorted(case_row["anchor_targets_upper"] & forced_targets)
        if not compound_overlap and not target_overlap:
            continue
        overlap_rows.append(
            {
                "case_study_name": case_row["case_study_name"],
                "disease_program": case_row["disease_program"],
                "compound_overlap": compound_overlap,
                "target_overlap": target_overlap,
            }
        )
    return overlap_rows


def assert_no_forced_anchor_overlap(config: dict) -> None:
    overlap_rows = forced_anchor_overlap_rows(config)
    if overlap_rows:
        raise ValueError(
            "Strict case-study anchors overlap with forced_include lists: "
            + json.dumps(overlap_rows, ensure_ascii=True, sort_keys=True)
        )


def action_ready_manifest_payload(config: dict) -> dict[str, object]:
    payload = action_ready_policy_payload(config)
    return {
        "manifest_name": "action_ready_build_manifest",
        "config_hash": action_ready_policy_hash(config),
        "forced_include_compounds": payload["action_ready_registry"]["forced_include_compounds"],
        "forced_include_targets": payload["action_ready_registry"]["forced_include_targets"],
        "strict_case_policies": payload["strict_case_policies"],
    }


def action_ready_manifest_path(config: dict) -> Path:
    action_root = resolve_repo_path(config["screening_pipeline"]["action_ready_registry"]["local_root"])
    relative_paths = config["screening_pipeline"]["action_ready_registry"]["relative_paths"]
    return action_root / relative_paths["build_manifest"]


def load_action_ready_manifest(config: dict) -> dict[str, object] | None:
    manifest_path = action_ready_manifest_path(config)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def action_ready_contamination_status(config: dict, manifest: dict[str, object] | None) -> str:
    overlap_rows = forced_anchor_overlap_rows(config)
    if overlap_rows:
        return "anchor_force_overlap"
    if manifest is None:
        return "manifest_missing"
    if str(manifest.get("config_hash", "")) != action_ready_policy_hash(config):
        return "manifest_mismatch"
    expected_payload = action_ready_manifest_payload(config)
    if manifest.get("forced_include_compounds", []) != expected_payload["forced_include_compounds"]:
        return "manifest_policy_mismatch"
    if manifest.get("forced_include_targets", []) != expected_payload["forced_include_targets"]:
        return "manifest_policy_mismatch"
    if manifest.get("strict_case_policies", []) != expected_payload["strict_case_policies"]:
        return "manifest_policy_mismatch"
    return "clean"


def assert_clean_action_ready_manifest(config: dict) -> dict[str, object]:
    manifest = load_action_ready_manifest(config)
    contamination_status = action_ready_contamination_status(config, manifest)
    if contamination_status != "clean":
        raise ValueError(f"Action-ready manifest is not clean for strict discovery: {contamination_status}")
    return manifest or {}


def compound_uid_by_preferred_name(compound_rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        str(row["preferred_name"]).strip().upper(): str(row["compound_uid"]).strip()
        for row in compound_rows
        if str(row.get("preferred_name", "")).strip()
    }


def anchor_compound_uids(case_row: dict[str, object], compound_rows: list[dict[str, str]]) -> set[str]:
    compound_lookup = compound_uid_by_preferred_name(compound_rows)
    return {
        compound_lookup[compound_name.upper()]
        for compound_name in case_row["anchor_compounds"]
        if compound_name.upper() in compound_lookup
    }


def pipe_delimited_overlap_mask(series: pd.Series, expected_values_upper: set[str]) -> np.ndarray:
    if not expected_values_upper:
        return np.zeros(series.shape[0], dtype=bool)
    return series.astype(str).apply(
        lambda value: any(token.strip().upper() in expected_values_upper for token in value.split("|") if token.strip())
    ).to_numpy(dtype=bool)


def target_anchor_mask(target_frame: pd.DataFrame, case_row: dict[str, object]) -> np.ndarray:
    return target_frame["canonical_entity_label"].astype(str).str.upper().isin(case_row["anchor_targets_upper"]).to_numpy(dtype=bool)


def regimen_anchor_mask(
    regimen_frame: pd.DataFrame,
    case_row: dict[str, object],
    compound_rows: list[dict[str, str]],
) -> np.ndarray:
    compound_uids = anchor_compound_uids(case_row, compound_rows)
    target_mask = pipe_delimited_overlap_mask(regimen_frame["target_symbol_set"], case_row["anchor_targets_upper"])
    compound_mask = pipe_delimited_overlap_mask(regimen_frame["compound_uid_set"], {value.upper() for value in compound_uids})
    return target_mask | compound_mask


def assert_no_forced_anchor_regimens(
    regimen_frame: pd.DataFrame,
    case_row: dict[str, object],
    compound_rows: list[dict[str, str]],
    context_name: str,
) -> None:
    anchor_mask = regimen_anchor_mask(regimen_frame, case_row, compound_rows)
    if not anchor_mask.any():
        return
    forced_mask = regimen_frame["selection_reason"].astype(str).eq("forced_include").to_numpy(dtype=bool)
    violating_frame = regimen_frame.loc[anchor_mask & forced_mask, ["regimen_uid", "target_symbol_set", "compound_uid_set", "selection_reason"]].copy()
    if violating_frame.empty:
        return
    raise ValueError(
        f"Strict discovery anchor regimen remains forced_include for {context_name}: "
        + violating_frame.head(5).to_json(orient="records")
    )


def compute_anchor_recovery_metrics(
    score_vector: np.ndarray,
    anchor_mask: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    if score_vector.ndim != 1:
        raise ValueError("score_vector must be one-dimensional.")
    if anchor_mask.ndim != 1:
        raise ValueError("anchor_mask must be one-dimensional.")
    if score_vector.shape[0] != anchor_mask.shape[0]:
        raise ValueError("score_vector and anchor_mask must have the same length.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    candidate_count = int(anchor_mask.sum())
    metrics = {
        "anchor_candidate_count": float(candidate_count),
        "anchor_hit_at_top_k": 0.0,
        "anchor_count_in_top_k": 0.0,
    }
    if candidate_count == 0:
        return metrics
    capped_top_k = min(int(top_k), int(score_vector.shape[0]))
    order = np.argsort(-score_vector, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, order.shape[0] + 1)
    anchor_ranks = ranks[anchor_mask]
    metrics["anchor_hit_at_top_k"] = float(np.any(anchor_ranks <= capped_top_k))
    metrics["anchor_count_in_top_k"] = float(np.sum(anchor_ranks <= capped_top_k))
    metrics["best_anchor_rank"] = float(anchor_ranks.min())
    return metrics


def compute_tri_accepted_anchor_metrics(
    score_vector: np.ndarray,
    anchor_mask: np.ndarray,
    tri_accepted_mask: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    if tri_accepted_mask.ndim != 1:
        raise ValueError("tri_accepted_mask must be one-dimensional.")
    if tri_accepted_mask.shape[0] != score_vector.shape[0]:
        raise ValueError("tri_accepted_mask and score_vector must have the same length.")
    metrics = {
        "tri_accepted_anchor_hit_at_top_k": 0.0,
    }
    accepted_anchor_mask = anchor_mask & tri_accepted_mask
    if not accepted_anchor_mask.any():
        return metrics
    accepted_indices = np.flatnonzero(tri_accepted_mask)
    accepted_scores = score_vector[accepted_indices]
    accepted_anchor_subset = accepted_anchor_mask[accepted_indices]
    accepted_metrics = compute_anchor_recovery_metrics(accepted_scores, accepted_anchor_subset, top_k)
    metrics["tri_accepted_anchor_hit_at_top_k"] = accepted_metrics["anchor_hit_at_top_k"]
    metrics["tri_accepted_best_anchor_rank"] = accepted_metrics["best_anchor_rank"]
    return metrics
