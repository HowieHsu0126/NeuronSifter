from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from Libs.utils.evaluation.candidate_validation_execution import execution_lookup_map as candidate_validation_execution_lookup
from Libs.utils.data.raw_data_registry import normalize_text, read_tsv, resolve_repo_path


MISSING_TOKENS = {"", "na", "nan", "null", "none", "n/a"}


def yes_no_text(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip()
    if text.lower() in {"true", "yes", "1"}:
        return "yes"
    if text.lower() in {"false", "no", "0"}:
        return "no"
    return text


def stable_uid(*parts: object) -> str:
    normalized_parts = [str(part).strip() for part in parts if str(part).strip()]
    joined = "::".join(normalized_parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    slug_seed = "_".join(normalized_parts[:4]) if normalized_parts else "platform"
    slug = normalize_text(slug_seed).replace(" ", "_")[:80]
    return f"{slug}_{digest}"


def relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(resolve_repo_path(".")).as_posix()


def parse_numeric(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if normalize_text(text) in MISSING_TOKENS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def bundle_lookup(config: dict) -> dict[str, dict[str, str]]:
    root = resolve_repo_path(config["model_input_registry"]["local_root"])
    rows = read_tsv(root / config["model_input_registry"]["relative_paths"]["study_bundle_catalog"])
    return {str(row["study_uid"]): row for row in rows}


def sample_alignment_lookup(config: dict) -> dict[tuple[str, str], dict[str, str]]:
    root = resolve_repo_path(config["dynamics_ready_registry"]["local_root"])
    rows = read_tsv(root / config["dynamics_ready_registry"]["relative_paths"]["sample_alignment_catalog"])
    return {
        (str(row["study_uid"]), str(row["sample_uid"])): row
        for row in rows
    }


def disease_programs_for_source_accession(config: dict, source_accession: str) -> list[str]:
    benchmark_programs = config.get("benchmark_registry", {}).get("disease_programs", {})
    matched_programs: list[str] = []
    for disease_program, program_config in benchmark_programs.items():
        accession_pool = {
            *(str(value) for value in program_config.get("primary_source_accessions", [])),
            *(str(value) for value in program_config.get("auxiliary_source_accessions", [])),
        }
        if str(source_accession) in accession_pool:
            matched_programs.append(str(disease_program))
    if matched_programs:
        return sorted(matched_programs)
    return [str(config["screening_pipeline"]["screening_benchmark"]["active_disease_program"])]


def foundation_twin_config(config: dict) -> dict[str, object]:
    if "foundation_twin" not in config:
        raise KeyError("foundation_twin configuration is required for pan-neuro screening twin contracts.")
    return config["foundation_twin"]


def disease_family_for_program(config: dict, disease_program: str) -> str:
    twin_config = foundation_twin_config(config)
    family_lookup = {
        str(program_name): str(family_name)
        for program_name, family_name in twin_config.get("disease_program_to_family", {}).items()
    }
    if disease_program in family_lookup:
        return family_lookup[disease_program]
    return "pan_neuro"


def supported_foundation_modalities(config: dict) -> list[str]:
    twin_config = foundation_twin_config(config)
    return [str(value) for value in twin_config.get("supported_modalities", [])]


def factor_ids(config: dict) -> list[str]:
    twin_config = foundation_twin_config(config)
    return [str(row["factor_id"]) for row in twin_config.get("world_state_factors", [])]


def factor_slice_manifest(config: dict) -> str:
    slices = {
        factor_id: {
            "slice_label": factor_id,
            "hidden_state_support": True,
        }
        for factor_id in factor_ids(config)
    }
    return json.dumps(slices, sort_keys=True, ensure_ascii=True)


def clipped_unit_interval(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def factor_target_payload(
    config: dict,
    observed_modality: str,
    region_scope: str,
    alignment_row: dict[str, str],
) -> str:
    severity = parse_numeric(alignment_row.get("baseline_severity_score", "")) or 0.2
    normalized_regions = str(region_scope).lower()
    hippocampal_bonus = 0.1 if any(token in normalized_regions for token in ["hippocampus", "entorhinal", "ca1", "ca3"]) else 0.0
    cortical_bonus = 0.1 if any(token in normalized_regions for token in ["cortex", "temporal", "prefrontal"]) else 0.0
    modality_bonus = 0.1 if observed_modality == "metabolomics" else 0.0
    support_payload = factor_supervision_mask_payload(config, observed_modality)
    payload = {
        "z_tx": clipped_unit_interval(0.5 + 0.5 * severity),
        "z_pk": clipped_unit_interval(0.15 + modality_bonus + 0.2 * severity),
        "z_env": clipped_unit_interval(0.3 + 0.6 * severity),
        "z_ephys": clipped_unit_interval(0.2 + 0.5 * severity + hippocampal_bonus),
        "z_morph": clipped_unit_interval(0.2 + 0.4 * severity + max(hippocampal_bonus, cortical_bonus)),
    }
    masked_payload = {
        factor_id: float(payload[factor_id]) * float(support_payload[factor_id])
        for factor_id in payload
    }
    return json.dumps(masked_payload, sort_keys=True, ensure_ascii=True)


def factor_supervision_mask_payload(config: dict, observed_modality: str) -> dict[str, float]:
    return {
        str(row["factor_id"]): 1.0 if observed_modality in {str(value) for value in row.get("primary_modalities", [])} else 0.0
        for row in foundation_twin_config(config).get("world_state_factors", [])
    }


def factor_teacher_source_manifest(config: dict, observed_modality: str) -> str:
    support_payload = factor_supervision_mask_payload(config, observed_modality)
    manifest = {}
    for factor_id, support_value in support_payload.items():
        if support_value >= 1.0:
            source_kind = "direct_observation"
            evidence_level = "direct_sample_observation"
        elif factor_id in {"z_pk", "z_env", "z_ephys"}:
            source_kind = "simulation_teacher"
            evidence_level = "provisional_teacher"
        elif factor_id == "z_morph":
            source_kind = "crossmodal_anchor"
            evidence_level = "weak_anchor_or_hidden"
        else:
            source_kind = "hidden_state_only"
            evidence_level = "hidden_state"
        manifest[factor_id] = {
            "source_kind": source_kind,
            "observed_modality": observed_modality,
            "direct_supervision": bool(support_value >= 1.0),
            "evidence_level": evidence_level,
        }
    return json.dumps(manifest, sort_keys=True, ensure_ascii=True)


def factor_uncertainty_payload(config: dict, observed_modality: str) -> str:
    support_payload = factor_supervision_mask_payload(config, observed_modality)
    return json.dumps(
        {
            factor_id: float(1.0 - support_value)
            for factor_id, support_value in support_payload.items()
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def posterior_slice_manifest(config: dict, state_store_path: Path) -> str:
    manifest = {
        factor_id: {
            "slice_label": factor_id,
            "state_store_path": relative_repo_path(state_store_path),
        }
        for factor_id in factor_ids(config)
    }
    return json.dumps(manifest, sort_keys=True, ensure_ascii=True)


def factor_evidence_payload(config: dict, observed_modality: str) -> str:
    source_payload = json.loads(factor_teacher_source_manifest(config, observed_modality))
    evidence = {
        factor_id: {
            "evidence_level": str(source_row["evidence_level"]),
            "source_kind": str(source_row["source_kind"]),
            "uncertainty_provenance": "direct_measurement_scaler" if bool(source_row["direct_supervision"]) else "teacher_or_missing_modality_uncertainty",
        }
        for factor_id, source_row in source_payload.items()
    }
    return json.dumps(evidence, sort_keys=True, ensure_ascii=True)


def unit_context_payload(config: dict, observation_row: dict[str, object], semantics_by_uid: dict[str, dict[str, object]]) -> str:
    semantics_uid = str(observation_row["semantics_uid"])
    semantics_row = semantics_by_uid[semantics_uid]
    source_payload = json.loads(factor_teacher_source_manifest(config, str(observation_row["modality"])))
    payload: dict[str, dict[str, object]] = {}
    for factor_id, source_row in source_payload.items():
        if bool(source_row["direct_supervision"]):
            payload[factor_id] = {
                "semantics_uid": semantics_uid,
                "canonical_unit": str(semantics_row["canonical_unit"]),
                "raw_unit": str(semantics_row["raw_unit"]),
                "proxy_flag": str(semantics_row["proxy_unit_flag"]) == "yes",
                "standardized_space": True,
                "inverse_transform_artifact_path": str(semantics_row["inverse_transform_artifact_path"]),
            }
            continue
        payload[factor_id] = {
            "semantics_uid": "",
            "canonical_unit": "latent_factor_unit",
            "raw_unit": "",
            "proxy_flag": True,
            "standardized_space": True,
            "inverse_transform_artifact_path": "",
        }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def bundle_modality(bundle_row: dict[str, str]) -> str:
    source_key = str(bundle_row["source_key"])
    if source_key in {"geo", "arrayexpress"}:
        return "transcriptomics"
    return "metabolomics"


def measurement_semantics_label(bundle_row: dict[str, str]) -> str:
    values = [
        str(bundle_row.get("selected_measurement_label", "")).strip(),
        str(bundle_row.get("matrix_semantics", "")).strip(),
    ]
    label = next((value for value in values if value), "")
    if not label:
        raise ValueError(f"Bundle lacks measurement semantics fields: {bundle_row}")
    return label


def measurement_semantics(bundle_row: dict[str, str]) -> str:
    return measurement_semantics_label(bundle_row)


def measurement_semantics_uid(bundle_row: dict[str, str]) -> str:
    if str(bundle_row.get("semantics_uid", "")).strip():
        return str(bundle_row["semantics_uid"])
    return stable_uid(
        "measurement_semantics",
        bundle_row.get("study_uid", ""),
        bundle_row.get("selected_matrix_uid", ""),
        bundle_row.get("source_accession", ""),
        measurement_semantics_label(bundle_row),
        bundle_row.get("matrix_semantics", ""),
    )


def measurement_semantics_definition(config: dict, matrix_semantics: str) -> dict[str, object]:
    definitions = config["measurement_semantics_registry"]["matrix_semantics_definitions"]
    if matrix_semantics not in definitions:
        raise ValueError(f"Missing measurement semantics definition for matrix_semantics={matrix_semantics!r}")
    return dict(definitions[matrix_semantics])


def measurement_semantics_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, bundle_row in sorted(bundle_lookup(config).items(), key=lambda item: (item[1]["source_key"], item[1]["source_accession"])):
        matrix_semantics = str(bundle_row["matrix_semantics"])
        definition = measurement_semantics_definition(config, matrix_semantics)
        scaler_artifact_path = str(bundle_row.get("scaler_artifact_path", "")).strip()
        if not scaler_artifact_path:
            scaler_artifact_path = str(Path(str(bundle_row["bundle_root"])) / str(config["model_input_registry"]["bundle_filenames"]["feature_scaler"]))
        semantics_uid = measurement_semantics_uid(bundle_row)
        label = str(bundle_row.get("measurement_semantics_label", "")).strip() or measurement_semantics_label(bundle_row)
        rows.append(
            {
                "semantics_uid": semantics_uid,
                "study_uid": str(bundle_row["study_uid"]),
                "source_key": str(bundle_row["source_key"]),
                "source_accession": str(bundle_row["source_accession"]),
                "selected_matrix_uid": str(bundle_row["selected_matrix_uid"]),
                "modality": bundle_modality(bundle_row),
                "raw_measurement_label": str(bundle_row["selected_measurement_label"]),
                "measurement_semantics_label": label,
                "matrix_semantics": matrix_semantics,
                "biological_quantity": str(definition["biological_quantity"]),
                "measurement_entity_namespace": str(definition["measurement_entity_namespace"]),
                "raw_unit": str(definition["raw_unit"]),
                "canonical_unit": str(definition["canonical_unit"]),
                "unit_conversion_chain": str(definition["unit_conversion_chain"]),
                "training_transform": str(bundle_row.get("normalization_method", definition["default_training_transform"])),
                "standardization_statistics_path": scaler_artifact_path,
                "standardization_summary": json.dumps(
                    {
                        "fit_policy": str(bundle_row.get("scaler_fit_policy", "")),
                        "normalization_method": str(bundle_row.get("normalization_method", definition["default_training_transform"])),
                        "statistics_artifact": scaler_artifact_path,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "inverse_transform_artifact_path": scaler_artifact_path,
                "inverse_transform": str(definition["inverse_transform"]),
                "uncertainty_policy": str(definition["uncertainty_policy"]),
                "uncertainty_semantics": str(definition["uncertainty_semantics"]),
                "valid_min": str(definition["valid_min"]),
                "valid_max": str(definition["valid_max"]),
                "raw_matrix_path": str(bundle_row["matrix_raw_path"]),
                "norm_matrix_path": str(bundle_row["matrix_norm_path"]),
                "sample_index_path": str(bundle_row["sample_index_path"]),
                "feature_index_path": str(bundle_row["feature_index_path"]),
                "scaler_artifact_path": scaler_artifact_path,
                "provenance_level": "model_input_bundle_public_matrix",
                "public_only_flag": str(bundle_row["public_only_flag"]),
                "proxy_unit_flag": yes_no_text(definition["proxy_unit_flag"]),
                "physical_interpretation_policy": str(definition["physical_interpretation_policy"]),
                "note": "Measurement semantics are registry-backed and reversible through the per-feature scaler artifact.",
            }
        )
    return rows


def measurement_semantics_lookup(config: dict) -> dict[str, dict[str, object]]:
    return {str(row["semantics_uid"]): row for row in measurement_semantics_rows(config)}


def region_mapping_confidence(region_scope: str) -> str:
    if not region_scope:
        return "none"
    if "|" in region_scope:
        return "multi_region"
    if region_scope == "systemic_circulation":
        return "high_proxy"
    return "high"


def sample_observation_rows(config: dict) -> list[dict[str, object]]:
    bundle_by_study = bundle_lookup(config)
    alignment_by_sample = sample_alignment_lookup(config)
    rows: list[dict[str, object]] = []
    for study_uid, bundle_row in sorted(bundle_by_study.items(), key=lambda item: (item[1]["source_key"], item[1]["source_accession"])):
        sample_index = pd.read_parquet(resolve_repo_path(bundle_row["sample_index_path"]))
        modality = bundle_modality(bundle_row)
        semantics_uid = measurement_semantics_uid(bundle_row)
        semantics_label = str(bundle_row.get("measurement_semantics_label", "")).strip() or measurement_semantics_label(bundle_row)
        disease_programs = disease_programs_for_source_accession(config, str(bundle_row["source_accession"]))
        for sample_row in sample_index.itertuples(index=False):
            sample_series = pd.Series(sample_row._asdict())
            alignment_row = alignment_by_sample.get((study_uid, str(sample_series["sample_uid"])), {})
            region_scope = str(alignment_row.get("observed_region_set", "")).strip()
            for disease_program in disease_programs:
                rows.append(
                    {
                        "observation_uid": stable_uid("observation", disease_program, study_uid, sample_series["sample_uid"]),
                        "study_uid": study_uid,
                        "sample_uid": str(sample_series["sample_uid"]),
                        "source_key": str(bundle_row["source_key"]),
                        "source_accession": str(bundle_row["source_accession"]),
                        "disease_program": disease_program,
                        "modality": modality,
                        "tissue_or_biofluid": str(sample_series.get("tissue_or_biofluid", "")).strip(),
                        "region_scope": region_scope,
                        "region_mapping_confidence": region_mapping_confidence(region_scope),
                        "semantics_uid": semantics_uid,
                        "measurement_semantics_label": semantics_label,
                        "public_only_flag": str(bundle_row["public_only_flag"]),
                        "observation_role": "sample_evidence",
                        "compatibility_alias_flag": "no",
                        "note": str(sample_series.get("sample_title", "")).strip()[:240],
                    }
                )
    return rows


def world_state_ready_rows(config: dict) -> list[dict[str, object]]:
    modalities = supported_foundation_modalities(config)
    alignment_by_sample = sample_alignment_lookup(config)
    semantics_by_uid = measurement_semantics_lookup(config)
    rows: list[dict[str, object]] = []
    for observation_row in sample_observation_rows(config):
        observed_modality = str(observation_row["modality"])
        missing_modalities = [modality for modality in modalities if modality != observed_modality]
        world_state_uid = stable_uid("world_state", observation_row["observation_uid"])
        state_store_path = resolve_repo_path(
            str(config["data"]["world_state_ready_registry"]["state_store_template"]).format(
                disease_program=observation_row["disease_program"],
                world_state_uid=world_state_uid,
            )
        )
        alignment_row = alignment_by_sample.get(
            (str(observation_row["study_uid"]), str(observation_row["sample_uid"])),
            {},
        )
        rows.append(
            {
                "world_state_uid": world_state_uid,
                "observation_uid": str(observation_row["observation_uid"]),
                "study_uid": str(observation_row["study_uid"]),
                "sample_uid": str(observation_row["sample_uid"]),
                "source_accession": str(observation_row["source_accession"]),
                "disease_program": str(observation_row["disease_program"]),
                "disease_family": disease_family_for_program(config, str(observation_row["disease_program"])),
                "semantics_uid": str(observation_row["semantics_uid"]),
                "measurement_semantics_label": str(observation_row["measurement_semantics_label"]),
                "observed_modality_set": observed_modality,
                "missing_modality_set": "|".join(missing_modalities),
                "world_state_version": "foundation_twin_world_state_v1",
                "hidden_state_support_flag": "yes",
                "cross_scene_compatibility_flag": "yes",
                "factor_slice_manifest": factor_slice_manifest(config),
                "factor_target_payload": factor_target_payload(
                    config,
                    observed_modality,
                    str(observation_row["region_scope"]),
                    alignment_row,
                ),
                "factor_supervision_mask_payload": json.dumps(
                    factor_supervision_mask_payload(config, observed_modality),
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "factor_teacher_source_manifest": factor_teacher_source_manifest(config, observed_modality),
                "factor_uncertainty_payload": factor_uncertainty_payload(config, observed_modality),
                "factor_evidence_payload": factor_evidence_payload(config, observed_modality),
                "unit_context_payload": unit_context_payload(config, observation_row, semantics_by_uid),
                "posterior_slice_manifest": posterior_slice_manifest(config, state_store_path),
                "state_store_path": relative_repo_path(state_store_path),
                "rollout_ready_flag": "yes",
                "note": str(observation_row["note"]),
            }
        )
    return rows


def crossmodal_anchor_rows(config: dict) -> list[dict[str, object]]:
    twin_config = foundation_twin_config(config)
    disease_families = [str(value) for value in twin_config.get("disease_families", [])]
    anchor_sources = list(twin_config.get("crossmodal_anchor_sources", []))
    rows: list[dict[str, object]] = []
    for disease_family in disease_families:
        for source_row in anchor_sources:
            anchor_source = str(source_row["source_id"])
            anchor_scope = str(source_row["anchor_scope"])
            paired_anchor = "patchseq" in anchor_source
            available_modalities = anchor_scope.replace("_", "|")
            anchor_type = "paired_sample_cell" if paired_anchor else "weak_teacher_source"
            confidence_provenance = "public_patchseq_cell_level_pairing" if paired_anchor else "public_source_level_teacher_anchor"
            alignment_target = "supervised_contrastive_retrieval" if paired_anchor else "distributional_alignment_only"
            rows.append(
                {
                    "crossmodal_anchor_uid": stable_uid("crossmodal_anchor", disease_family, anchor_source),
                    "disease_family": disease_family,
                    "anchor_source": anchor_source,
                    "anchor_scope": anchor_scope,
                    "source_sample_or_cell_id": f"{anchor_source}:{disease_family}:public_anchor",
                    "available_modality_set": available_modalities,
                    "data_path": str(source_row.get("data_path", "")),
                    "anchor_type": anchor_type,
                    "split_role": "alignment_validation" if paired_anchor else "teacher_or_distribution_reference",
                    "confidence_provenance": confidence_provenance,
                    "alignment_target": alignment_target,
                    "metric_artifact_path": f"Output/paper_reports/crossscale_alignment_audit_{anchor_source}_{disease_family}.json",
                    "transcriptomic_anchor_id": f"{anchor_source}:{disease_family}:tx",
                    "morphology_anchor_id": f"{anchor_source}:{disease_family}:morph",
                    "electrophysiology_anchor_id": f"{anchor_source}:{disease_family}:ephys",
                    "supervision_role": str(source_row["supervision_role"]),
                    "anchor_status": "catalog_materialized_public_source",
                    "confidence_score": 1.0 if "patchseq" in anchor_source else 0.7,
                    "note": "Pan-neuro cross-modal anchor source registered for foundation twin alignment.",
                }
            )
    return rows


def simulation_teacher_rows(config: dict) -> list[dict[str, object]]:
    root = resolve_repo_path(config["dynamics_ready_registry"]["local_root"])
    simulation_rows = read_tsv(root / config["dynamics_ready_registry"]["relative_paths"]["simulation_catalog"])
    rows: list[dict[str, object]] = []
    for simulation_row in simulation_rows:
        disease_program = str(simulation_row["disease_program"])
        rows.append(
            {
                "simulation_teacher_uid": stable_uid("simulation_teacher", simulation_row["simulation_uid"]),
                "disease_program": disease_program,
                "disease_family": disease_family_for_program(config, disease_program),
                "teacher_family": "ad_microenvironment_mechanistic_teacher",
                "teacher_role": "legacy_solver_compatibility_teacher",
                "source_model_id": str(simulation_row["simulation_uid"]),
                "source_repository": "dynamics_ready_registry",
                "provenance_level": "configuration_generated_legacy_solver",
                "teacher_evidence_level": "legacy_compatibility",
                "calibration_role": "compatibility_baseline",
                "calibration_status": "not_external_calibrated",
                "observed_or_synthetic": "synthetic_legacy_mesoscale_solver",
                "source_simulation_uid": str(simulation_row["simulation_uid"]),
                "regimen_uid": str(simulation_row["regimen_uid"]),
                "factor_scope": "z_pk|z_env|z_ephys",
                "route": str(simulation_row["route"]),
                "schedule": str(simulation_row["schedule"]),
                "dose_level_id": str(simulation_row["dose_level_id"]),
                "trajectory_store_path": str(simulation_row["trajectory_store_path"]),
                "supervision_role": "solver_teacher_trajectory",
                "consistency_targets": "non_negativity|conservation|flux_consistency|long_horizon_stability",
                "uncertainty_policy": "legacy_solver_default_uncertainty",
                "teacher_loss_weight": "0.1",
                "main_pretraining_eligibility": "calibration_or_compatibility_only",
                "note": str(simulation_row["note"]),
            }
        )
    return rows


def counterfactual_benchmark_rows(config: dict) -> list[dict[str, object]]:
    action_root = resolve_repo_path(config["screening_pipeline"]["action_ready_registry"]["local_root"])
    regimen_rows = read_tsv(action_root / config["screening_pipeline"]["action_ready_registry"]["relative_paths"]["regimen_catalog"])
    rows: list[dict[str, object]] = []
    regimen_frame = pd.DataFrame(regimen_rows)
    if regimen_frame.empty:
        return rows
    group_columns = ["target_symbol_set", "compound_uid_set"]
    for _, group_frame in regimen_frame.groupby(group_columns, dropna=False):
        if group_frame.shape[0] < 2:
            continue
        ordered_group = group_frame.sort_values(by=["route", "schedule", "dose_level_id", "regimen_uid"], ascending=True)
        baseline_row = ordered_group.iloc[0].to_dict()
        disease_program = str(config["screening_pipeline"]["screening_benchmark"]["active_disease_program"])
        for counterfactual_row in ordered_group.iloc[1:].to_dict(orient="records"):
            rows.append(
                {
                    "counterfactual_benchmark_uid": stable_uid(
                        "counterfactual_benchmark",
                        baseline_row["regimen_uid"],
                        counterfactual_row["regimen_uid"],
                    ),
                    "disease_program": disease_program,
                    "disease_family": disease_family_for_program(config, disease_program),
                    "benchmark_name": "route_schedule_dose_counterfactual",
                    "intervention_family": str(baseline_row["target_symbol_set"]),
                    "baseline_regimen_uid": str(baseline_row["regimen_uid"]),
                    "counterfactual_regimen_uid": str(counterfactual_row["regimen_uid"]),
                    "objective_name": "intervention_screening_gain",
                    "target_metric": "rerank_score",
                    "validity_contract": "counterfactual_action_consistency_v1",
                    "note": "Counterfactual benchmark pair derived from regimen action variants with shared target and compound sets.",
                }
            )
    return rows


def feedback_event_rows(config: dict) -> list[dict[str, object]]:
    registry_config = config["data"]["feedback_ready_registry"]
    catalog_path = resolve_repo_path(registry_config["local_root"]) / registry_config["relative_paths"]["feedback_event_catalog"]
    if not catalog_path.exists():
        return []
    return [
        {
            "feedback_event_uid": str(row["feedback_event_uid"]),
            "disease_program": str(row["disease_program"]),
            "disease_family": str(row["disease_family"]),
            "sample_uid": str(row["sample_uid"]),
            "intervention_program_uid": str(row["intervention_program_uid"]),
            "regimen_uid": str(row["regimen_uid"]),
            "predicted_rollout_summary": str(row["predicted_rollout_summary"]),
            "realized_outcome_summary": str(row["realized_outcome_summary"]),
            "discrepancy_summary": str(row["discrepancy_summary"]),
            "calibration_impact_score": str(row["calibration_impact_score"]),
            "update_eligibility_flag": str(row["update_eligibility_flag"]),
            "update_decision": str(row["update_decision"]),
            "event_timestamp": str(row["event_timestamp"]),
            "note": str(row.get("note", "")),
        }
        for row in read_tsv(catalog_path)
    ]


def preferred_target_program_lookup(target_program_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    frame = pd.DataFrame(target_program_rows).copy()
    if frame.empty:
        return {}
    frame["target_evidence_score_numeric"] = pd.to_numeric(frame["target_evidence_score"], errors="coerce").fillna(0.0)
    frame["mapped_feature_count_numeric"] = pd.to_numeric(frame["mapped_feature_count"], errors="coerce").fillna(0.0)
    frame["coverage_fraction_numeric"] = pd.to_numeric(frame["coverage_fraction"], errors="coerce").fillna(0.0)
    frame["direct_priority"] = frame["source_repository"].astype(str).eq("drug_target_direct").astype(int)
    frame = frame.sort_values(
        by=[
            "canonical_entity_id",
            "direct_priority",
            "target_evidence_score_numeric",
            "mapped_feature_count_numeric",
            "coverage_fraction_numeric",
            "target_program_uid",
        ],
        ascending=[True, False, False, False, False, True],
    )
    best_rows = frame.drop_duplicates(subset=["canonical_entity_id"], keep="first")
    return {
        str(row["canonical_entity_id"]): {
            "target_program_uid": str(row["target_program_uid"]),
            "canonical_entity_label": str(row["canonical_entity_label"]),
        }
        for row in best_rows.to_dict(orient="records")
    }


def target_uid_to_entity_lookup(target_rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        str(row["target_uid"]): str(row["canonical_entity_id"])
        for row in target_rows
    }


def route_exposure_targets(prior_config: dict, route_name: str, schedule_name: str, dose_multiplier: float) -> dict[str, float]:
    exposure_targets: dict[str, float] = {}
    occupancy_targets: dict[str, float] = {}
    dose_exposure_scale = dose_multiplier ** float(prior_config["dose_exposure_exponent"])
    dose_occupancy_scale = dose_multiplier ** float(prior_config["dose_occupancy_exponent"])
    for compartment_name in prior_config["compartment_order"]:
        exposure_targets[compartment_name] = (
            float(prior_config["exposure_target_by_compartment"][compartment_name])
            * float(prior_config["route_exposure_multiplier"][route_name][compartment_name])
            * float(prior_config["schedule_exposure_multiplier"][schedule_name])
            * dose_exposure_scale
        )
        occupancy_targets[compartment_name] = min(
            float(prior_config["occupancy_upper_bound"]),
            max(
                float(prior_config["occupancy_lower_bound"]),
                float(prior_config["occupancy_target_by_compartment"][compartment_name])
                * float(prior_config["route_occupancy_multiplier"][route_name])
                * float(prior_config["schedule_occupancy_multiplier"][schedule_name])
                * dose_occupancy_scale,
            ),
        )
    return {
        "exposure_target_by_compartment": exposure_targets,
        "occupancy_target_by_compartment": occupancy_targets,
    }


def intervention_program_rows(config: dict) -> list[dict[str, object]]:
    action_root = resolve_repo_path(config["screening_pipeline"]["action_ready_registry"]["local_root"])
    drug_target_root = resolve_repo_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"])
    regimen_rows = read_tsv(action_root / config["screening_pipeline"]["action_ready_registry"]["relative_paths"]["regimen_catalog"])
    target_program_rows = read_tsv(action_root / config["screening_pipeline"]["action_ready_registry"]["relative_paths"]["target_program_catalog"])
    target_rows = read_tsv(drug_target_root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["target_catalog"])
    preferred_program_by_entity = preferred_target_program_lookup(target_program_rows)
    target_uid_to_entity = target_uid_to_entity_lookup(target_rows)
    prior_config = config["screening_pipeline"]["screening_benchmark"]["pkpd_prior_consistency"]

    rows: list[dict[str, object]] = []
    for regimen_row in regimen_rows:
        target_program_uids = []
        target_program_labels = []
        for target_uid in str(regimen_row["target_uid_set"]).split("|"):
            if target_uid not in target_uid_to_entity:
                continue
            entity_id = target_uid_to_entity[target_uid]
            if entity_id not in preferred_program_by_entity:
                continue
            payload = preferred_program_by_entity[entity_id]
            target_program_uids.append(str(payload["target_program_uid"]))
            target_program_labels.append(str(payload["canonical_entity_label"]))
        target_program_uid_set = "|".join(sorted(set(target_program_uids)))
        target_program_label_set = "|".join(sorted(set(target_program_labels)))
        dose_multiplier = float(regimen_row["dose_multiplier"])
        occupancy_constraints = route_exposure_targets(
            prior_config,
            str(regimen_row["route"]),
            str(regimen_row["schedule"]),
            dose_multiplier,
        )
        rows.append(
            {
                "intervention_program_uid": stable_uid("intervention_program", regimen_row["regimen_uid"]),
                "regimen_uid": str(regimen_row["regimen_uid"]),
                "drug_program_uid_set": str(regimen_row["drug_program_uid_set"]),
                "target_program_set": target_program_uid_set,
                "target_program_label_set": target_program_label_set,
                "compound_set": str(regimen_row["compound_uid_set"]),
                "route": str(regimen_row["route"]),
                "schedule": str(regimen_row["schedule"]),
                "dose_level_id": str(regimen_row["dose_level_id"]),
                "dose_multiplier": dose_multiplier,
                "occupancy_constraints": json.dumps(occupancy_constraints, sort_keys=True, ensure_ascii=True),
                "pkpd_feasibility_score": str(regimen_row["pk_feasibility_score"]),
                "evidence_summary": json.dumps(
                    {
                        "aggregate_target_evidence_score": float(regimen_row["aggregate_target_evidence_score"]),
                        "aggregate_affinity_score": (
                            None
                            if str(regimen_row["aggregate_affinity_score"]).strip() in {"", "nan"}
                            else float(regimen_row["aggregate_affinity_score"])
                        ),
                        "clue_reversal_proxy_score": float(regimen_row["clue_reversal_proxy_score"]),
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "selection_reason": str(regimen_row["selection_reason"]),
                "note": str(regimen_row["note"]),
            }
        )
    return rows


def compound_feature_frame(config: dict) -> pd.DataFrame:
    root = resolve_repo_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"])
    path = root / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_feature_catalog"]
    return pd.DataFrame(read_tsv(path))


def compound_pk_frame(config: dict) -> pd.DataFrame:
    root = resolve_repo_path(config["screening_pipeline"]["pkpd_ready_registry"]["local_root"])
    path = root / config["screening_pipeline"]["pkpd_ready_registry"]["relative_paths"]["compound_pk_prior_catalog"]
    return pd.DataFrame(read_tsv(path))


def compound_feature_lookup_map(feature_frame: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    lookup: dict[str, dict[str, list[float]]] = {}
    if feature_frame.empty:
        return lookup
    for row in feature_frame.to_dict(orient="records"):
        compound_uid = str(row["compound_uid"])
        feature_name = str(row["feature_name"])
        feature_value = parse_numeric(row["feature_value"])
        if feature_value is None:
            continue
        lookup.setdefault(compound_uid, {}).setdefault(feature_name, []).append(feature_value)
    return lookup


def compound_pk_lookup_map(pk_frame: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    lookup: dict[str, list[dict[str, object]]] = {}
    if pk_frame.empty:
        return lookup
    for row in pk_frame.to_dict(orient="records"):
        compound_uid = str(row["compound_uid"])
        lookup.setdefault(compound_uid, []).append(row)
    return lookup


def admet_summary_payload(compound_uid_set: str, feature_lookup: dict[str, dict[str, list[float]]]) -> tuple[str, float]:
    compound_uids = [value for value in str(compound_uid_set).split("|") if value]
    if not compound_uids:
        return "", 0.0
    feature_value_by_name = {}
    for compound_uid in compound_uids:
        for feature_name, values in feature_lookup.get(compound_uid, {}).items():
            feature_value_by_name.setdefault(feature_name, []).extend(values)
    if not feature_value_by_name:
        return "", 0.0
    averaged = {
        feature_name: float(np.mean(values))
        for feature_name, values in feature_value_by_name.items()
    }
    lipinski_pass = (
        averaged.get("molecular_weight", 0.0) <= 500.0
        and averaged.get("alogp", 0.0) <= 5.0
        and averaged.get("hba", 0.0) <= 10.0
        and averaged.get("hbd", 0.0) <= 5.0
    )
    cns_bias_pass = (
        averaged.get("tpsa", 999.0) <= 90.0
        and averaged.get("alogp", -999.0) >= 1.0
        and averaged.get("alogp", 999.0) <= 4.5
    )
    readiness_score = 0.5 * float(lipinski_pass) + 0.5 * float(cns_bias_pass)
    payload = {
        "compound_count": len(compound_uids),
        "lipinski_pass": bool(lipinski_pass),
        "cns_bias_pass": bool(cns_bias_pass),
        "mean_molecular_weight": (
            round(averaged["molecular_weight"], 3) if "molecular_weight" in averaged else None
        ),
        "mean_alogp": round(averaged["alogp"], 3) if "alogp" in averaged else None,
        "mean_tpsa": round(averaged["tpsa"], 3) if "tpsa" in averaged else None,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True), readiness_score


def pbpk_summary_payload(compound_uid_set: str, pk_lookup: dict[str, list[dict[str, object]]]) -> tuple[str, float]:
    compound_uids = [value for value in str(compound_uid_set).split("|") if value]
    if not compound_uids:
        return "", 0.0
    subset_rows = [
        row
        for compound_uid in compound_uids
        for row in pk_lookup.get(compound_uid, [])
    ]
    if not subset_rows:
        return "", 0.0
    subset = pd.DataFrame(subset_rows)
    subset["permeability_score_numeric"] = pd.to_numeric(subset["permeability_score"], errors="coerce").fillna(0.0)
    subset["exposure_scale_numeric"] = pd.to_numeric(subset["exposure_scale"], errors="coerce").fillna(0.0)
    subset["clearance_scale_numeric"] = pd.to_numeric(subset["clearance_scale"], errors="coerce").fillna(0.0)
    readiness_score = float(
        0.5 * subset["permeability_score_numeric"].mean()
        + 0.25 * np.clip(subset["exposure_scale_numeric"].mean(), 0.0, 1.5) / 1.5
        + 0.25 * (1.0 - min(subset["clearance_scale_numeric"].mean(), 2.0) / 2.0)
    )
    payload = {
        "route_count": int(subset["route"].astype(str).nunique()),
        "mean_permeability_score": round(float(subset["permeability_score_numeric"].mean()), 3),
        "mean_exposure_scale": round(float(subset["exposure_scale_numeric"].mean()), 3),
        "mean_clearance_scale": round(float(subset["clearance_scale_numeric"].mean()), 3),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True), float(np.clip(readiness_score, 0.0, 1.0))


def structure_validation_payload(execution_row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(execution_row.get("structure_confidence", "")).strip(),
        str(execution_row.get("docking_score", "")).strip(),
        str(execution_row.get("rescoring_score", "")).strip(),
    )


def validation_stage(
    structure_confidence: str,
    docking_score: str,
    rescoring_score: str,
    affinity_evidence: str,
    admet_summary: str,
    pbpk_summary: str,
    execution_status: str,
) -> str:
    if rescoring_score:
        return "rescored"
    if docking_score:
        return "docked"
    if execution_status:
        return execution_status
    if structure_confidence and admet_summary and pbpk_summary:
        return "structure_profiled"
    if affinity_evidence and admet_summary and pbpk_summary:
        return "knowledge_profiled"
    if admet_summary and pbpk_summary:
        return "admet_pbpk_profiled"
    if admet_summary:
        return "admet_profiled"
    if affinity_evidence:
        return "affinity_profiled"
    return "knowledge_ready"


def candidate_validation_rows(config: dict) -> list[dict[str, object]]:
    twin_root = resolve_repo_path(config["twin"]["intervention_program_registry"]["local_root"])
    twin_path = twin_root / config["twin"]["intervention_program_registry"]["relative_paths"]["intervention_program_catalog"]
    intervention_rows = read_tsv(twin_path)
    action_root = resolve_repo_path(config["screening_pipeline"]["action_ready_registry"]["local_root"])
    regimen_rows = read_tsv(action_root / config["screening_pipeline"]["action_ready_registry"]["relative_paths"]["regimen_catalog"])
    regimen_by_uid = {str(row["regimen_uid"]): row for row in regimen_rows}
    feature_frame = compound_feature_frame(config)
    pk_frame = compound_pk_frame(config)
    feature_lookup = compound_feature_lookup_map(feature_frame)
    pk_lookup = compound_pk_lookup_map(pk_frame)
    execution_lookup = candidate_validation_execution_lookup(config)

    rows: list[dict[str, object]] = []
    for row in intervention_rows:
        regimen_row = regimen_by_uid[str(row["regimen_uid"])]
        execution_row = execution_lookup.get(str(row["intervention_program_uid"]), {})
        structure_confidence, docking_score, rescoring_score = structure_validation_payload(execution_row)
        execution_status = str(execution_row.get("execution_status", "execution_pending")).strip() or "execution_pending"
        affinity_evidence_score = 0.0
        aggregate_affinity_score = parse_numeric(regimen_row.get("aggregate_affinity_score", ""))
        if aggregate_affinity_score is not None:
            affinity_evidence_score = float(np.exp(-math.log10(max(aggregate_affinity_score, 1.0)) / 4.0))
        admet_summary, admet_readiness_score = admet_summary_payload(row["compound_set"], feature_lookup)
        pbpk_summary = str(execution_row.get("pbpk_summary", "")).strip()
        pbpk_readiness_value = parse_numeric(execution_row.get("pbpk_readiness_score", ""))
        pbpk_readiness_score = 0.0 if pbpk_readiness_value is None else float(pbpk_readiness_value)
        affinity_evidence = ""
        if affinity_evidence_score > 0.0:
            affinity_evidence = json.dumps(
                {
                    "affinity_evidence_score": round(affinity_evidence_score, 4),
                    "aggregate_affinity_score": (
                        None
                        if str(regimen_row["aggregate_affinity_score"]).strip() in {"", "nan"}
                        else float(regimen_row["aggregate_affinity_score"])
                    ),
                },
                sort_keys=True,
                ensure_ascii=True,
            )
        rows.append(
            {
                "candidate_validation_uid": stable_uid("candidate_validation", row["intervention_program_uid"]),
                "intervention_program_uid": str(row["intervention_program_uid"]),
                "regimen_uid": str(row["regimen_uid"]),
                "target_program_set": str(row["target_program_set"]),
                "compound_set": str(row["compound_set"]),
                "structure_confidence": structure_confidence,
                "docking_score": docking_score,
                "rescoring_score": rescoring_score,
                "affinity_evidence": affinity_evidence,
                "affinity_evidence_score": round(float(affinity_evidence_score), 4),
                "admet_summary": admet_summary,
                "admet_readiness_score": round(float(admet_readiness_score), 4),
                "pbpk_summary": pbpk_summary,
                "pbpk_readiness_score": round(float(pbpk_readiness_score), 4),
                "validation_stage": validation_stage(
                    structure_confidence,
                    docking_score,
                    rescoring_score,
                    affinity_evidence,
                    admet_summary,
                    pbpk_summary,
                    execution_status,
                ),
                "note": str(execution_row.get("note", row["selection_reason"])),
            }
        )
    return rows


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.empty((0,), dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def validation_rerank_frame(
    candidate_frame: pd.DataFrame,
    predicted_scores: np.ndarray,
    tri_scores: np.ndarray,
    validation_frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    if candidate_frame.empty:
        return candidate_frame.copy()
    predicted_scores = np.asarray(predicted_scores, dtype=np.float64).reshape(-1)
    tri_scores = np.asarray(tri_scores, dtype=np.float64).reshape(-1)
    if predicted_scores.shape[0] != candidate_frame.shape[0]:
        raise ValueError("predicted_scores must align with candidate_frame rows.")
    if tri_scores.shape[0] != candidate_frame.shape[0]:
        raise ValueError("tri_scores must align with candidate_frame rows.")
    merged = candidate_frame.copy().reset_index(drop=True)
    merged["predicted_score"] = predicted_scores
    merged["tri_score"] = tri_scores
    if not validation_frame.empty:
        validation_columns = [
            "intervention_program_uid",
            "validation_stage",
            "affinity_evidence_score",
            "admet_readiness_score",
            "pbpk_readiness_score",
            "structure_confidence",
        ]
        merged = merged.merge(
            validation_frame[validation_columns].drop_duplicates(subset=["intervention_program_uid"]),
            on="intervention_program_uid",
            how="left",
        )
    else:
        merged["validation_stage"] = ""
        merged["affinity_evidence_score"] = 0.0
        merged["admet_readiness_score"] = 0.0
        merged["pbpk_readiness_score"] = 0.0
        merged["structure_confidence"] = ""
    merged["affinity_evidence_score"] = pd.to_numeric(merged["affinity_evidence_score"], errors="coerce").fillna(0.0)
    merged["admet_readiness_score"] = pd.to_numeric(merged["admet_readiness_score"], errors="coerce").fillna(0.0)
    merged["pbpk_readiness_score"] = pd.to_numeric(merged["pbpk_readiness_score"], errors="coerce").fillna(0.0)
    model_component = _normalize(merged["predicted_score"].to_numpy(dtype=np.float64))
    tri_component = _normalize(merged["tri_score"].to_numpy(dtype=np.float64))
    affinity_component = _normalize(merged["affinity_evidence_score"].to_numpy(dtype=np.float64))
    admet_component = _normalize(merged["admet_readiness_score"].to_numpy(dtype=np.float64))
    pbpk_component = _normalize(merged["pbpk_readiness_score"].to_numpy(dtype=np.float64))
    merged["rerank_score"] = (
        float(weights["model_score"]) * model_component
        + float(weights["tri_score"]) * tri_component
        + float(weights["affinity_evidence"]) * affinity_component
        + float(weights["admet_readiness"]) * admet_component
        + float(weights["pbpk_readiness"]) * pbpk_component
    )
    return merged.sort_values(by=["rerank_score", "tri_score", "predicted_score"], ascending=[False, False, False]).reset_index(drop=True)
