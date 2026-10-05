from __future__ import annotations

import numpy as np
import pandas as pd


ROUTE_PRIORITY = {
    "intrathecal": 0,
    "intravenous": 1,
    "oral": 2,
}
SCHEDULE_PRIORITY = {
    "daily": 0,
    "twice_daily": 1,
    "weekly": 2,
}


def target_program_prior_scores(target_program_frame: pd.DataFrame) -> np.ndarray:
    if target_program_frame.empty:
        return np.empty((0,), dtype=np.float32)
    return pd.to_numeric(
        target_program_frame["target_evidence_score"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float32)


def regimen_prior_scores(regimen_frame: pd.DataFrame) -> np.ndarray:
    if regimen_frame.empty:
        return np.empty((0,), dtype=np.float32)
    target_evidence = pd.to_numeric(regimen_frame["aggregate_target_evidence_score"], errors="coerce").fillna(0.0)
    clue_reversal = pd.to_numeric(regimen_frame["clue_reversal_proxy_score"], errors="coerce").fillna(0.0)
    pk_feasibility = pd.to_numeric(regimen_frame["pk_feasibility_score"], errors="coerce").fillna(0.0)
    return (target_evidence + clue_reversal + pk_feasibility).to_numpy(dtype=np.float32)


def regimen_diversity_group_keys(regimen_frame: pd.DataFrame) -> list[str]:
    if regimen_frame.empty:
        return []
    return [
        f"{target_symbol_set}::{route_name}::{schedule_name}"
        for target_symbol_set, route_name, schedule_name in zip(
            regimen_frame["target_symbol_set"].astype(str).tolist(),
            regimen_frame["route"].astype(str).tolist(),
            regimen_frame["schedule"].astype(str).tolist(),
            strict=False,
        )
    ]


def ordered_anchor_compound_uids(
    strict_case_rows: list[dict[str, object]],
    compound_rows: list[dict[str, str]],
) -> list[str]:
    preferred_name_to_uid = {
        str(row["preferred_name"]).strip().upper(): str(row["compound_uid"]).strip()
        for row in compound_rows
        if str(row.get("preferred_name", "")).strip()
    }
    ordered_compound_uids: list[str] = []
    seen_compound_uids: set[str] = set()
    for case_row in strict_case_rows:
        for compound_name in case_row.get("anchor_compounds", []):
            compound_uid = preferred_name_to_uid.get(str(compound_name).strip().upper(), "")
            if not compound_uid or compound_uid in seen_compound_uids:
                continue
            ordered_compound_uids.append(compound_uid)
            seen_compound_uids.add(compound_uid)
    return ordered_compound_uids


def preserved_case_study_regimen_indices(
    regimen_frame: pd.DataFrame,
    strict_case_rows: list[dict[str, object]],
    compound_rows: list[dict[str, str]],
    canonical_dose_level_id: str = "dose_100",
) -> list[int]:
    if regimen_frame.empty:
        return []

    anchor_compound_uids = ordered_anchor_compound_uids(strict_case_rows, compound_rows)
    if not anchor_compound_uids:
        return []

    candidate_frame = regimen_frame.copy()
    candidate_frame["row_index"] = np.arange(candidate_frame.shape[0], dtype=np.int64)
    candidate_frame["regimen_size_numeric"] = pd.to_numeric(candidate_frame["regimen_size"], errors="coerce").fillna(0.0)
    candidate_frame["pk_feasibility_numeric"] = pd.to_numeric(candidate_frame["pk_feasibility_score"], errors="coerce").fillna(0.0)
    candidate_frame["route_priority"] = candidate_frame["route"].astype(str).map(ROUTE_PRIORITY).fillna(len(ROUTE_PRIORITY)).astype(int)
    candidate_frame["schedule_priority"] = candidate_frame["schedule"].astype(str).map(SCHEDULE_PRIORITY).fillna(len(SCHEDULE_PRIORITY)).astype(int)
    candidate_frame["dose_priority"] = (
        candidate_frame["dose_level_id"].astype(str).ne(str(canonical_dose_level_id)).astype(int)
    )
    candidate_frame = candidate_frame.loc[
        candidate_frame["selection_reason"].astype(str).eq("case_study_seed")
        & candidate_frame["regimen_size_numeric"].eq(1.0)
    ].copy()
    if candidate_frame.empty:
        return []

    preserved_indices: list[int] = []
    seen_indices: set[int] = set()
    for compound_uid in anchor_compound_uids:
        compound_frame = candidate_frame.loc[
            candidate_frame["compound_uid_set"].astype(str).eq(compound_uid)
        ].copy()
        if compound_frame.empty:
            continue
        compound_frame = compound_frame.sort_values(
            by=[
                "route_priority",
                "schedule_priority",
                "dose_priority",
                "pk_feasibility_numeric",
                "row_index",
            ],
            ascending=[True, True, True, False, True],
        )
        for (_, _, _), pair_frame in compound_frame.groupby(
            ["compound_uid_set", "route", "schedule"],
            sort=False,
        ):
            selected_index = int(pair_frame.iloc[0]["row_index"])
            if selected_index in seen_indices:
                continue
            preserved_indices.append(selected_index)
            seen_indices.add(selected_index)
    return preserved_indices


def diversified_candidate_indices(
    observed_scores: np.ndarray,
    prior_scores: np.ndarray,
    group_keys: list[str],
    count: int,
) -> np.ndarray:
    if observed_scores.ndim != 1 or prior_scores.ndim != 1:
        raise ValueError("Candidate selection expects one-dimensional score vectors.")
    if observed_scores.shape[0] != prior_scores.shape[0]:
        raise ValueError("Observed and prior score vectors must have the same length.")
    if observed_scores.shape[0] != len(group_keys):
        raise ValueError("group_keys length must match score vectors.")
    if count <= 0 or observed_scores.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    if observed_scores.shape[0] <= count:
        return np.arange(observed_scores.shape[0], dtype=np.int64)

    ranking_frame = pd.DataFrame(
        {
            "row_index": np.arange(observed_scores.shape[0], dtype=np.int64),
            "observed_score": observed_scores.astype(np.float64),
            "prior_score": prior_scores.astype(np.float64),
            "group_key": [str(value) for value in group_keys],
        }
    )
    ranking_frame = ranking_frame.sort_values(
        ["observed_score", "prior_score", "row_index"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    grouped_best = ranking_frame.groupby("group_key", sort=False, as_index=False).first()
    selected_indices = grouped_best["row_index"].astype(np.int64).tolist()[:count]
    selected_lookup = set(selected_indices)

    if len(selected_indices) < count:
        remaining = ranking_frame.loc[~ranking_frame["row_index"].isin(selected_lookup), "row_index"].astype(np.int64).tolist()
        selected_indices.extend(remaining[: count - len(selected_indices)])

    return np.asarray(selected_indices[:count], dtype=np.int64)
