from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Libs.utils.data.raw_data_registry import ensure_directory, read_tsv, resolve_repo_path, write_tsv
from Libs.utils.model.screening_core import stable_uid


def structural_prior_root(config: dict) -> Path:
    return resolve_repo_path(config["structural_prior_registry"]["local_root"])


def structural_prior_region_rows(config: dict) -> list[dict[str, object]]:
    region_label_lookup = {
        str(row["region_id"]): str(row["region_label"])
        for row in config["spatial_atlas_registry"]["regions"]
    }
    rows: list[dict[str, object]] = []
    for row in config["structural_prior_registry"]["disease_region_priors"]:
        disease_program = str(row["disease_program"])
        region_id = str(row["region_id"])
        rows.append(
            {
                "structural_prior_uid": stable_uid("structural_prior", disease_program, region_id),
                "disease_program": disease_program,
                "region_id": region_id,
                "region_label": region_label_lookup[region_id],
                "program_support_score": float(row["program_support_score"]),
                "disease_vulnerability_score": float(row["disease_vulnerability_score"]),
                "transport_centrality_score": float(row["transport_centrality_score"]),
                "clearance_constraint_score": float(row["clearance_constraint_score"]),
                "inflammation_propensity_score": float(row["inflammation_propensity_score"]),
                "neurotransmitter_relevance_score": float(row["neurotransmitter_relevance_score"]),
                "source_evidence": str(row["source_evidence"]),
                "note": str(row["note"]),
            }
        )
    return rows


def structural_prior_local_response_head_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in config["structural_prior_registry"]["local_response_heads"]:
        rows.append(
            {
                "local_response_head_uid": str(row["local_response_head_uid"]),
                "disease_program": str(row["disease_program"]),
                "region_id": str(row["region_id"]),
                "response_head_label": str(row["response_head_label"]),
                "response_role": str(row["response_role"]),
                "source_reference_url": str(row["source_reference_url"]),
                "note": str(row["note"]),
            }
        )
    return rows


def write_structural_prior_catalogs(
    config: dict,
    region_rows: list[dict[str, object]],
    head_rows: list[dict[str, object]],
) -> None:
    root = structural_prior_root(config)
    relative_paths = config["structural_prior_registry"]["relative_paths"]
    ensure_directory(root)
    write_tsv(root / relative_paths["region_prior_catalog"], config["schemas"]["structural_prior_region_fields"], region_rows)
    write_tsv(root / relative_paths["local_response_head_catalog"], config["schemas"]["structural_prior_local_response_head_fields"], head_rows)


def structural_prior_region_frame(config: dict) -> pd.DataFrame:
    root = structural_prior_root(config)
    relative_paths = config["structural_prior_registry"]["relative_paths"]
    return pd.read_csv(root / relative_paths["region_prior_catalog"], sep="\t")


def structural_prior_local_response_head_frame(config: dict) -> pd.DataFrame:
    root = structural_prior_root(config)
    relative_paths = config["structural_prior_registry"]["relative_paths"]
    return pd.read_csv(root / relative_paths["local_response_head_catalog"], sep="\t")


def load_structural_prior_catalogs(config: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    root = structural_prior_root(config)
    relative_paths = config["structural_prior_registry"]["relative_paths"]
    return (
        read_tsv(root / relative_paths["region_prior_catalog"]),
        read_tsv(root / relative_paths["local_response_head_catalog"]),
    )


def structural_prior_lookup(config: dict, disease_program: str) -> dict[str, dict[str, float]]:
    frame = structural_prior_region_frame(config)
    filtered = frame.loc[frame["disease_program"] == str(disease_program)].copy()
    rows: dict[str, dict[str, float]] = {}
    if filtered.empty:
        return rows
    feature_names = list(config["structural_prior_registry"]["region_feature_names"])
    for feature_name in feature_names:
        values = filtered[feature_name].to_numpy(dtype=np.float32)
        minimum = float(values.min())
        maximum = float(values.max())
        filtered[f"{feature_name}_normalized"] = (values - minimum) / max(maximum - minimum, 1e-6)
    for row in filtered.itertuples(index=False):
        rows[str(row.region_id)] = {
            feature_name: float(getattr(row, feature_name))
            for feature_name in feature_names
        }
        rows[str(row.region_id)].update(
            {
                f"{feature_name}_normalized": float(getattr(row, f"{feature_name}_normalized"))
                for feature_name in feature_names
            }
        )
        rows[str(row.region_id)]["structural_prior_uid"] = str(row.structural_prior_uid)
    return rows


def structural_prior_uid_lookup(config: dict, disease_program: str) -> dict[str, str]:
    frame = structural_prior_region_frame(config)
    filtered = frame.loc[frame["disease_program"] == str(disease_program)].copy()
    return {
        str(row.region_id): str(row.structural_prior_uid)
        for row in filtered.itertuples(index=False)
    }


def structural_prior_tensor(config: dict, target_region_ids: list[str], disease_program: str) -> np.ndarray:
    frame = structural_prior_region_frame(config)
    filtered = frame.loc[frame["disease_program"] == str(disease_program)].copy()
    feature_names = list(config["structural_prior_registry"]["region_feature_names"])
    summary_by_region = (
        filtered.groupby("region_id")[feature_names]
        .mean()
        .reindex(target_region_ids)
        .fillna(0.0)
    )
    values = summary_by_region.to_numpy(dtype=np.float32)
    if values.size == 0:
        return np.empty((0, len(feature_names)), dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def structural_prior_summary(config: dict, disease_program: str) -> list[dict[str, object]]:
    frame = structural_prior_region_frame(config)
    filtered = frame.loc[frame["disease_program"] == str(disease_program)].copy()
    return filtered.to_dict(orient="records")
