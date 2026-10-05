from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from Libs.utils.data.raw_data_registry import ensure_directory, read_tsv, resolve_repo_path, write_json, write_tsv
from Libs.utils.model.screening_core import stable_uid


def active_learning_config(config: dict) -> dict[str, object]:
    return config["screening_pipeline"]["screening_benchmark"]["active_learning"]


def active_learning_root(config: dict) -> Path:
    return resolve_repo_path(str(active_learning_config(config)["artifact_root"]))


def round_path(config: dict, path_key: str, round_uid: str) -> Path:
    relative_template = str(active_learning_config(config)["relative_paths"][path_key])
    return active_learning_root(config) / relative_template.format(round_uid=round_uid)


def _numeric(frame: pd.DataFrame, column_name: str, default_value: float = 0.0) -> np.ndarray:
    if column_name not in frame.columns:
        return np.full((frame.shape[0],), float(default_value), dtype=np.float64)
    return pd.to_numeric(frame[column_name], errors="coerce").fillna(float(default_value)).to_numpy(dtype=np.float64)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty((0,), dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1.0e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def quota_counts(total_count: int, quota_config: dict[str, object]) -> dict[str, int]:
    quota_names = ["exploitation", "exploration", "sentinel"]
    raw_values = {name: float(quota_config[name]) * int(total_count) for name in quota_names}
    base_values = {name: int(np.floor(value)) for name, value in raw_values.items()}
    remainder_count = int(total_count) - sum(base_values.values())
    ordered_remainders = sorted(
        quota_names,
        key=lambda name: (raw_values[name] - base_values[name], quota_names.index(name)),
        reverse=True,
    )
    for name in ordered_remainders[:remainder_count]:
        base_values[name] += 1
    return base_values


def mandatory_source_keys(config: dict) -> set[str]:
    active_config = active_learning_config(config)
    return {str(value) for value in active_config["mandatory_public_sources"]} | {
        str(value) for value in active_config["mandatory_tool_sources"]
    }


def python_package_available(raw_root: str) -> bool:
    package_name = raw_root.split(":", 1)[1]
    importlib.import_module(package_name)
    return True


def external_binary_available(raw_root: str) -> bool:
    return raw_root.startswith("external_binary:")


def adapter_raw_root_available(raw_root: str) -> bool:
    if raw_root.startswith("python_package:"):
        return python_package_available(raw_root)
    if raw_root.startswith("external_binary:"):
        return external_binary_available(raw_root)
    return resolve_repo_path(raw_root).exists()


def validate_mandatory_adapters(config: dict, strict_processed: bool) -> list[dict[str, object]]:
    registry_config = config["validation"]["external_adapter_registry"]
    adapters = {str(row["source_key"]): row for row in registry_config["adapters"]}
    missing_sources = sorted(mandatory_source_keys(config) - set(adapters))
    if missing_sources:
        raise ValueError(f"Active-learning mandatory adapters are missing from validation config: {missing_sources}")
    validations: list[dict[str, object]] = []
    for source_key in sorted(mandatory_source_keys(config)):
        row = adapters[source_key]
        raw_root = str(row["raw_root"])
        raw_available = adapter_raw_root_available(raw_root)
        if not raw_available:
            raise FileNotFoundError(f"Mandatory active-learning adapter raw root is unavailable: {source_key} -> {raw_root}")
        processed_path = resolve_repo_path(str(row["processed_artifact_path"]))
        processed_required = str(row["required_processed_artifact_flag"]) == "yes"
        processed_available = processed_path.exists() and (processed_path.is_dir() or processed_path.stat().st_size > 0)
        if strict_processed and processed_required and not processed_available:
            raise FileNotFoundError(
                f"Mandatory active-learning adapter processed artifact is unavailable: {source_key} -> {processed_path}"
            )
        validations.append(
            {
                "source_key": source_key,
                "adapter_role": str(row["adapter_role"]),
                "raw_root": raw_root,
                "processed_artifact_path": str(row["processed_artifact_path"]),
                "processed_required": processed_required,
                "processed_available": processed_available,
            }
        )
    return validations


def validate_required_binaries(config: dict) -> list[dict[str, object]]:
    binary_config = config["validation"]["candidate_validation_execution"]["binaries"]
    validations: list[dict[str, object]] = []
    for binary_name, binary_path in sorted(binary_config.items()):
        path_text = str(binary_path)
        resolved_path = Path(path_text) if Path(path_text).is_absolute() else Path(shutil.which(path_text) or "")
        available = resolved_path.exists() and resolved_path.is_file() and resolved_path.stat().st_size >= 0
        if not available:
            raise FileNotFoundError(f"Mandatory active-learning binary is unavailable: {binary_name} -> {path_text}")
        validations.append({"binary_name": binary_name, "binary_path": path_text, "available": True})
    return validations


def build_active_candidate_pool(
    round_uid: str,
    intervention_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    priority_frame: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    if intervention_frame.empty:
        raise ValueError("Active-learning candidate pool requires non-empty intervention_program_catalog.")
    if priority_frame.empty:
        raise ValueError("Active-learning candidate pool requires non-empty candidate priority catalog.")
    merged = intervention_frame.merge(
        priority_frame,
        on="regimen_uid",
        how="inner",
        suffixes=("", "_priority"),
    )
    if merged.empty:
        raise ValueError("Active-learning candidate pool found no regimen overlap with model priority catalog.")
    if not validation_frame.empty and "intervention_program_uid" in validation_frame.columns:
        validation_columns = [
            column_name
            for column_name in [
                "intervention_program_uid",
                "validation_stage",
                "affinity_evidence_score",
                "admet_readiness_score",
                "pbpk_readiness_score",
            ]
            if column_name in validation_frame.columns
        ]
        merged = merged.merge(validation_frame[validation_columns], on="intervention_program_uid", how="left")
    frame = merged.reset_index(drop=True)
    predicted_score = _numeric(frame, "predicted_score_mean")
    tri_score = _numeric(frame, "tri_score_mean")
    pkpd_consistency = _numeric(frame, "pkpd_consistency_mean")
    epistemic_std = _numeric(frame, "epistemic_std_mean")
    predictive_entropy = _numeric(frame, "predictive_entropy_mean")
    mutual_information = _numeric(frame, "mutual_information_mean")
    affinity = _numeric(frame, "affinity_evidence_score")
    admet = _numeric(frame, "admet_readiness_score")
    pbpk = _numeric(frame, "pbpk_readiness_score")
    priority_seed = active_learning_config(config)["priority_seed"]
    readiness_weights = priority_seed["validation_readiness_weights"]
    validation_readiness = np.clip(
        float(readiness_weights["affinity_evidence_score"]) * affinity
        + float(readiness_weights["admet_readiness_score"]) * admet
        + float(readiness_weights["pbpk_readiness_score"]) * pbpk,
        float(priority_seed["score_min"]),
        float(priority_seed["score_max"]),
    )
    rollout_uncertainty = _normalize(epistemic_std + predictive_entropy)
    ood_score = _normalize(1.0 - pkpd_consistency + predictive_entropy)
    mechanism_uncertainty = _normalize(mutual_information + epistemic_std)
    counterfactual_disagreement = _normalize(np.abs(predicted_score - pkpd_consistency))
    rank_flip_risk = _normalize((1.0 - tri_score) * predicted_score + mutual_information)
    expected_information_gain = _normalize(rollout_uncertainty + mechanism_uncertainty + counterfactual_disagreement)
    risk_penalty = _normalize((1.0 - pkpd_consistency) + ood_score)
    decision_gate = config["screening_pipeline"]["screening_benchmark"]["decision_gate"]
    labels = decision_gate["decision_labels"]
    validation_stage = frame["validation_stage"].fillna("unvalidated").astype(str) if "validation_stage" in frame.columns else pd.Series(["unvalidated"] * frame.shape[0])
    decision_labels = []
    for index in range(frame.shape[0]):
        if ood_score[index] >= float(decision_gate["ood_reject_threshold"]):
            decision_labels.append(str(labels["insufficient_evidence"]))
        elif predicted_score[index] >= float(decision_gate["score_active_learning_threshold"]) and tri_score[index] < float(decision_gate["tri_accept_threshold"]):
            decision_labels.append(str(labels["active_learning"]))
        elif predicted_score[index] >= float(decision_gate["score_accept_threshold"]) and validation_stage.iloc[index] not in set(decision_gate["accepted_validation_stages"]):
            decision_labels.append(str(labels["external_validation"]))
        elif predicted_score[index] >= float(decision_gate["score_accept_threshold"]):
            decision_labels.append(str(labels["accept"]))
        else:
            decision_labels.append(str(labels["reject"]))
    rows: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        candidate_uid = stable_uid("active_candidate", round_uid, row["intervention_program_uid"], row["regimen_uid"])
        evidence_payload = {
            "selection_reason": str(row.get("selection_reason", "")),
            "evidence_summary": str(row.get("evidence_summary", "")),
            "validation_stage": str(validation_stage.iloc[index]),
        }
        rows.append(
            {
                "round_uid": round_uid,
                "candidate_uid": candidate_uid,
                "sample_state_uid": str(row.get("sample_state_uid", "public_ad_state")),
                "intervention_program_uid": str(row["intervention_program_uid"]),
                "regimen_uid": str(row["regimen_uid"]),
                "target_program_set": str(row["target_program_set"]),
                "target_program_label_set": str(row["target_program_label_set"]),
                "compound_set": str(row["compound_set"]),
                "route": str(row["route"]),
                "schedule": str(row["schedule"]),
                "dose_level_id": str(row["dose_level_id"]),
                "predicted_score": round(float(predicted_score[index]), 8),
                "tri_score": round(float(tri_score[index]), 8),
                "pkpd_consistency": round(float(pkpd_consistency[index]), 8),
                "ranking_stability": round(float(1.0 - rollout_uncertainty[index]), 8),
                "rollout_uncertainty": round(float(rollout_uncertainty[index]), 8),
                "ood_score": round(float(ood_score[index]), 8),
                "mechanism_uncertainty": round(float(mechanism_uncertainty[index]), 8),
                "counterfactual_disagreement": round(float(counterfactual_disagreement[index]), 8),
                "rank_flip_risk": round(float(rank_flip_risk[index]), 8),
                "expected_information_gain": round(float(expected_information_gain[index]), 8),
                "validation_readiness": round(float(validation_readiness[index]), 8),
                "risk_penalty": round(float(risk_penalty[index]), 8),
                "decision_label": decision_labels[index],
                "source_holdout_role": str(row.get("source_holdout_role", "active_learning_candidate")),
                "evidence_vector": json.dumps(evidence_payload, sort_keys=True, ensure_ascii=True),
                "candidate_status": "ready_for_acquisition",
            }
        )
    return pd.DataFrame(rows, columns=config["schemas"]["active_learning_candidate_pool_fields"])


def score_acquisition_tasks(round_uid: str, candidate_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    if candidate_frame.empty:
        raise ValueError("Acquisition scoring requires non-empty active candidate pool.")
    active_config = active_learning_config(config)
    weights = active_config["weights"]
    portfolio = active_config["simulator_portfolio"]
    rows: list[dict[str, object]] = []
    candidate_numeric = {
        column_name: _numeric(candidate_frame, column_name)
        for column_name in [
            "expected_information_gain",
            "rank_flip_risk",
            "tri_score",
            "counterfactual_disagreement",
            "ood_score",
            "mechanism_uncertainty",
            "risk_penalty",
        ]
    }
    diversity_values = _normalize(
        1.0
        / np.sqrt(
            candidate_frame["target_program_set"].astype(str).map(candidate_frame["target_program_set"].astype(str).value_counts()).to_numpy(dtype=np.float64)
        )
    )
    for candidate_index, candidate_row in candidate_frame.iterrows():
        tri_uncertainty = 1.0 - float(candidate_numeric["tri_score"][candidate_index])
        for simulator_config in portfolio:
            simulator_cost = float(simulator_config["cost_weight"])
            base_score = (
                float(weights["expected_information_gain"]) * float(candidate_numeric["expected_information_gain"][candidate_index])
                + float(weights["rank_flip_risk"]) * float(candidate_numeric["rank_flip_risk"][candidate_index])
                + float(weights["tri_uncertainty"]) * tri_uncertainty
                + float(weights["counterfactual_disagreement"]) * float(candidate_numeric["counterfactual_disagreement"][candidate_index])
                + float(weights["ood_gap"]) * float(candidate_numeric["ood_score"][candidate_index])
                + float(weights["mechanism_expert_disagreement"]) * float(candidate_numeric["mechanism_uncertainty"][candidate_index])
                + float(weights["diversity"]) * float(diversity_values[candidate_index])
                - float(weights["simulator_cost_penalty"]) * simulator_cost
                - float(weights["risk_penalty"]) * float(candidate_numeric["risk_penalty"][candidate_index])
            )
            exploitation_score = float(candidate_row["predicted_score"]) + float(candidate_numeric["rank_flip_risk"][candidate_index]) + tri_uncertainty
            exploration_score = float(candidate_numeric["ood_score"][candidate_index]) + float(candidate_numeric["mechanism_uncertainty"][candidate_index])
            sentinel_score = float(candidate_numeric["risk_penalty"][candidate_index]) + (1.0 - float(candidate_row["predicted_score"]))
            batch_role = max(
                [
                    ("exploitation", exploitation_score),
                    ("exploration", exploration_score),
                    ("sentinel", sentinel_score),
                ],
                key=lambda item: item[1],
            )[0]
            request_uid = stable_uid(
                "active_learning_request",
                round_uid,
                candidate_row["candidate_uid"],
                simulator_config["simulator_family"],
            )
            rows.append(
                {
                    "round_uid": round_uid,
                    "request_uid": request_uid,
                    "candidate_uid": str(candidate_row["candidate_uid"]),
                    "batch_role": batch_role,
                    "simulator_family": str(simulator_config["simulator_family"]),
                    "fidelity_level": str(simulator_config["fidelity_level"]),
                    "request_role": str(simulator_config["request_role"]),
                    "training_label_policy": str(simulator_config["training_label_policy"]),
                    "acquisition_score": round(base_score, 8),
                    "exploitation_score": round(exploitation_score, 8),
                    "exploration_score": round(exploration_score, 8),
                    "sentinel_score": round(sentinel_score, 8),
                    "expected_information_gain": candidate_row["expected_information_gain"],
                    "rank_flip_risk": candidate_row["rank_flip_risk"],
                    "tri_uncertainty": round(tri_uncertainty, 8),
                    "counterfactual_disagreement": candidate_row["counterfactual_disagreement"],
                    "ood_gap": candidate_row["ood_score"],
                    "mechanism_expert_disagreement": candidate_row["mechanism_uncertainty"],
                    "diversity_bonus": round(float(diversity_values[candidate_index]), 8),
                    "simulator_cost": round(simulator_cost, 8),
                    "risk_penalty": candidate_row["risk_penalty"],
                    "decision_label": str(candidate_row["decision_label"]),
                    "route": str(candidate_row["route"]),
                    "schedule": str(candidate_row["schedule"]),
                    "target_program_set": str(candidate_row["target_program_set"]),
                    "artifact_path": str(round_path(config, "acquisition_scores_template", round_uid)),
                }
            )
    return pd.DataFrame(rows, columns=config["schemas"]["active_learning_acquisition_fields"])


def select_request_batch(round_uid: str, acquisition_frame: pd.DataFrame, candidate_frame: pd.DataFrame, config: dict, total_count: int) -> pd.DataFrame:
    if acquisition_frame.empty:
        raise ValueError("Active-learning request selection requires non-empty acquisition scores.")
    active_config = active_learning_config(config)
    quota = quota_counts(total_count, active_config["batch_quota"])
    remaining_frame = acquisition_frame.copy()
    selected_frames: list[pd.DataFrame] = []
    for batch_role, role_count in quota.items():
        role_policy = active_config["batch_role_policy"][batch_role]
        score_column = str(role_policy["score_column"])
        if score_column not in remaining_frame.columns:
            raise ValueError(f"Active-learning acquisition score column is missing for {batch_role}: {score_column}")
        role_frame = remaining_frame.copy()
        if role_frame.shape[0] < role_count:
            raise ValueError(f"Insufficient {batch_role} acquisition tasks for quota {role_count}: found {role_frame.shape[0]}")
        preferred_labels = {str(value) for value in role_policy["preferred_decision_labels"]}
        preferred_label_bonus = float(role_policy["preferred_label_bonus"])
        role_selection_score = pd.to_numeric(role_frame[score_column], errors="raise").to_numpy(dtype=np.float64)
        preferred_mask = role_frame["decision_label"].astype(str).isin(preferred_labels).to_numpy(dtype=np.float64)
        role_frame = role_frame.assign(
            role_selection_score=role_selection_score + preferred_label_bonus * preferred_mask,
            batch_role=batch_role,
        )
        role_frame = role_frame.sort_values(
            by=["role_selection_score", "acquisition_score", "simulator_cost", "request_uid"],
            ascending=[False, False, True, True],
        ).head(role_count)
        selected_frames.append(role_frame)
        remaining_frame = remaining_frame.loc[~remaining_frame["request_uid"].isin(set(role_frame["request_uid"].astype(str)))]
    selected = pd.concat(selected_frames, ignore_index=True)
    candidate_lookup = candidate_frame.set_index("candidate_uid").to_dict(orient="index")
    request_defaults = active_learning_config(config)["simulation_request_defaults"]
    request_rows: list[dict[str, object]] = []
    for row in selected.to_dict(orient="records"):
        candidate = candidate_lookup[str(row["candidate_uid"])]
        request_rows.append(
            {
                "request_uid": row["request_uid"],
                "round_uid": round_uid,
                "candidate_uid": row["candidate_uid"],
                "intervention_program_uid": candidate["intervention_program_uid"],
                "sample_state_uid": candidate["sample_state_uid"],
                "regimen_uid": candidate["regimen_uid"],
                "target_program_set": candidate["target_program_set"],
                "compound_set": candidate["compound_set"],
                "route": candidate["route"],
                "schedule": candidate["schedule"],
                "dose_level_id": candidate["dose_level_id"],
                "simulator_family": row["simulator_family"],
                "fidelity_level": row["fidelity_level"],
                "request_role": row["request_role"],
                "training_label_policy": row["training_label_policy"],
                "requested_outputs": row["request_role"],
                "time_grid": str(request_defaults["time_grid"]),
                "region_grid": str(request_defaults["region_grid"]),
                "compartment_grid": str(request_defaults["compartment_grid"]),
                "species_list": str(request_defaults["species_list"]),
                "parameter_prior_uid": str(request_defaults["parameter_prior_uid"]),
                "uncertainty_policy": str(request_defaults["uncertainty_policy"]),
                "seed": stable_uid("seed", row["request_uid"])[:8],
                "budget_class": row["batch_role"],
                "source_holdout_role": candidate["source_holdout_role"],
                "execution_status": "pending_dispatch",
                "artifact_path": str(round_path(config, "request_batch_template", round_uid)),
            }
        )
    return pd.DataFrame(request_rows, columns=config["schemas"]["active_learning_simulation_request_fields"])


def ingest_active_learning_results(round_uid: str, request_frame: pd.DataFrame, result_frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if request_frame.empty:
        raise ValueError("Active-learning result ingest requires non-empty request batch.")
    if result_frame.empty:
        raise ValueError("Active-learning result ingest requires non-empty simulator/tool result manifest.")
    merged = request_frame.merge(result_frame, on=["request_uid", "round_uid", "candidate_uid", "simulator_family"], how="inner")
    if merged.shape[0] != request_frame.shape[0]:
        raise ValueError("Active-learning result manifest does not cover every requested task.")
    grade_counts = merged["qc_grade"].astype(str).value_counts().to_dict()
    teacher_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        result_uid = str(row.get("result_uid") or stable_uid("active_learning_result", row["request_uid"], row["trajectory_hash"]))
        result_rows.append({field_name: row.get(field_name, "") for field_name in config["schemas"]["active_learning_simulation_result_fields"]} | {"result_uid": result_uid})
        if str(row["training_label_policy"]) == "downstream_validation_only":
            manifest_action = "candidate_validation_update"
            teacher_family = "candidate_validation_teacher"
        else:
            manifest_action = "teacher_family_update"
            teacher_family = str(row["request_role"])
        teacher_rows.append(
            {
                "update_uid": stable_uid("active_learning_teacher_update", round_uid, result_uid),
                "round_uid": round_uid,
                "result_uid": result_uid,
                "request_uid": row["request_uid"],
                "teacher_family": teacher_family,
                "simulator_family": row["simulator_family"],
                "trajectory_artifact_path": row["trajectory_artifact_path"],
                "teacher_uncertainty_path": row["teacher_uncertainty_path"],
                "qc_grade": row["qc_grade"],
                "qc_use": row["qc_use"],
                "sample_weight": row["sample_weight"],
                "training_label_policy": row["training_label_policy"],
                "manifest_action": manifest_action,
                "note": row.get("note", ""),
            }
        )
    qc_summary = {
        "round_uid": round_uid,
        "request_count": int(request_frame.shape[0]),
        "result_count": int(merged.shape[0]),
        "grade_counts": {str(key): int(value) for key, value in grade_counts.items()},
        "status": "active_learning_results_ingested",
    }
    return (
        pd.DataFrame(result_rows, columns=config["schemas"]["active_learning_simulation_result_fields"]),
        pd.DataFrame(teacher_rows, columns=config["schemas"]["active_learning_teacher_update_fields"]),
        qc_summary,
    )


def artifact_digest(path: Path) -> str:
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        files = [file_path for file_path in sorted(path.rglob("*")) if file_path.is_file()]
        if not files:
            raise FileNotFoundError(f"Active-learning artifact directory has no files: {path}")
        for file_path in files:
            stat = file_path.stat()
            digest.update(str(file_path.resolve().relative_to(path.resolve())).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        return digest.hexdigest()
    raise FileNotFoundError(f"Active-learning artifact path is unavailable: {path}")


def external_adapter_processed_path(config: dict, adapter_source_key: str) -> Path:
    adapters = {
        str(row["source_key"]): row
        for row in config["validation"]["external_adapter_registry"]["adapters"]
    }
    if adapter_source_key not in adapters:
        raise ValueError(f"Active-learning result linker missing external adapter source: {adapter_source_key}")
    return resolve_repo_path(str(adapters[adapter_source_key]["processed_artifact_path"]))


def candidate_validation_execution_catalog_path(config: dict, relative_path_key: str) -> Path:
    execution_config = config["validation"]["candidate_validation_execution"]
    benchmark_name = str(config["screening_pipeline"]["screening_benchmark"]["benchmark_name"])
    execution_root = resolve_repo_path(str(execution_config["output_root_template"]).format(benchmark_name=benchmark_name))
    return execution_root / str(execution_config["relative_paths"][relative_path_key])


def full_microenvironment_registry_path(config: dict, relative_path_key: str) -> Path:
    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    return resolve_repo_path(str(registry["local_root"])) / str(registry["relative_paths"][relative_path_key])


def resolved_result_artifact_path(config: dict, simulator_family: str) -> Path:
    result_config = active_learning_config(config)["result_linking"]
    artifact_sources = result_config["artifact_sources"]
    if simulator_family not in artifact_sources:
        raise ValueError(f"Active-learning result linker lacks artifact source for simulator_family={simulator_family}")
    source_config = artifact_sources[simulator_family]
    source_type = str(source_config["source_type"])
    if source_type == "external_adapter_processed":
        return external_adapter_processed_path(config, str(source_config["adapter_source_key"]))
    if source_type == "candidate_validation_execution_catalog":
        return candidate_validation_execution_catalog_path(config, str(source_config["relative_path_key"]))
    if source_type == "full_microenvironment_registry":
        return full_microenvironment_registry_path(config, str(source_config["relative_path_key"]))
    raise ValueError(f"Unsupported active-learning result artifact source_type: {source_type}")


def build_completed_result_manifest(round_uid: str, request_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    if request_frame.empty:
        raise ValueError("Completed active-learning result manifest requires non-empty request batch.")
    result_config = active_learning_config(config)["result_linking"]
    axis_hashes = result_config["axis_hashes"]
    policy_by_label = result_config["result_policy_by_training_label_policy"]
    rows: list[dict[str, object]] = []
    for request_row in request_frame.to_dict(orient="records"):
        simulator_family = str(request_row["simulator_family"])
        artifact_path = resolved_result_artifact_path(config, simulator_family)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Completed active-learning artifact is unavailable for {simulator_family}: {artifact_path}")
        training_label_policy = str(request_row["training_label_policy"])
        if training_label_policy not in policy_by_label:
            raise ValueError(f"Completed active-learning result policy is missing for {training_label_policy}.")
        result_policy = policy_by_label[training_label_policy]
        artifact_hash = artifact_digest(artifact_path)
        result_uid = stable_uid("active_learning_completed_result", request_row["request_uid"], artifact_hash)
        rows.append(
            {
                "result_uid": result_uid,
                "request_uid": str(request_row["request_uid"]),
                "round_uid": round_uid,
                "candidate_uid": str(request_row["candidate_uid"]),
                "simulator_family": simulator_family,
                "trajectory_artifact_path": str(artifact_path),
                "trajectory_hash": artifact_hash,
                "time_axis_hash": str(axis_hashes["time_axis_hash"]),
                "region_axis_hash": str(axis_hashes["region_axis_hash"]),
                "compartment_axis_hash": str(axis_hashes["compartment_axis_hash"]),
                "species_axis_hash": str(axis_hashes["species_axis_hash"]),
                "teacher_uncertainty_path": str(artifact_path),
                "solver_metadata_path": str(artifact_path),
                "runtime_seconds": "",
                "failure_status": "completed",
                "qc_grade": str(result_policy["qc_grade"]),
                "qc_use": str(result_policy["qc_use"]),
                "sample_weight": float(result_policy["sample_weight"]),
                "provenance_level": "completed_public_tool_or_simulator_artifact",
                "note": "Linked from completed mandatory active-learning simulator/tool artifact.",
            }
        )
    return pd.DataFrame(rows, columns=config["schemas"]["active_learning_simulation_result_fields"])


def build_round_report(round_uid: str, config: dict) -> dict[str, object]:
    candidate_path = round_path(config, "candidate_pool_template", round_uid)
    acquisition_path = round_path(config, "acquisition_scores_template", round_uid)
    request_path = round_path(config, "request_batch_template", round_uid)
    result_path = round_path(config, "result_manifest_template", round_uid)
    teacher_update_path = round_path(config, "teacher_update_template", round_uid)
    paths = [candidate_path, acquisition_path, request_path, result_path, teacher_update_path]
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Active-learning round report is missing required artifacts: {missing_paths}")
    candidate_frame = pd.DataFrame(read_tsv(candidate_path))
    acquisition_frame = pd.DataFrame(read_tsv(acquisition_path))
    request_frame = pd.DataFrame(read_tsv(request_path))
    result_frame = pd.DataFrame(read_tsv(result_path))
    grade_counts = result_frame["qc_grade"].astype(str).value_counts().to_dict()
    report = {
        "round_uid": round_uid,
        "status": "round_artifacts_available",
        "candidate_count": int(candidate_frame.shape[0]),
        "acquisition_task_count": int(acquisition_frame.shape[0]),
        "request_count": int(request_frame.shape[0]),
        "result_count": int(result_frame.shape[0]),
        "grade_a_b_count": int(grade_counts.get("A", 0) + grade_counts.get("B", 0)),
        "grade_c_count": int(grade_counts.get("C", 0)),
        "grade_d_count": int(grade_counts.get("D", 0)),
        "grade_s_count": int(grade_counts.get("S", 0)),
        "source_holdout_rollout_error_delta": "requires_empirical_result",
        "top_k_rank_stability_delta": "requires_empirical_result",
        "tri_selective_risk_delta": "requires_empirical_result",
        "counterfactual_direction_hit_delta": "requires_empirical_result",
        "ood_false_accept_delta": "requires_empirical_result",
        "publication_artifact_status": "requires_empirical_result",
        "artifact_path": str(round_path(config, "round_report_template", round_uid)),
    }
    return report


def write_round_report(config: dict, round_uid: str) -> Path:
    report = build_round_report(round_uid, config)
    report_path = round_path(config, "round_report_template", round_uid)
    write_json(report_path, report)
    return report_path


def write_frame(path: Path, fields: list[str], frame: pd.DataFrame) -> None:
    ensure_directory(path.parent)
    write_tsv(path, fields, frame.to_dict(orient="records"))
