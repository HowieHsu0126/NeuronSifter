from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, write_json
from Libs.utils.model.neuronsifter_scale_registry import canonical_model_name, registry as neuronsifter_scale_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the AD screening system production bundle.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--model-version", default="", help="Optional model version override.")
    return parser.parse_args()


def optional_tsv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str)


def pan_neuro_completeness(config: dict, benchmark_name: str) -> dict[str, object]:
    declared_programs = {
        str(program_name)
        for program_name in config["foundation_twin"]["disease_program_to_family"]
    }
    configured_programs = {
        str(program_name): dict(program_config)
        for program_name, program_config in config["benchmark_registry"].get("disease_programs", {}).items()
    }
    benchmark_root = resolve_path(config["benchmark_registry"]["local_root"])
    cohort_frame = optional_tsv_frame(benchmark_root / config["benchmark_registry"]["relative_paths"]["cohort_catalog"])
    fold_frame = optional_tsv_frame(benchmark_root / config["benchmark_registry"]["relative_paths"]["fold_catalog"])
    world_state_frame = optional_tsv_frame(
        resolve_path(config["data"]["world_state_ready_registry"]["local_root"])
        / config["data"]["world_state_ready_registry"]["relative_paths"]["world_state_catalog"]
    )
    simulation_teacher_frame = optional_tsv_frame(
        resolve_path(config["data"]["simulation_teacher_ready_registry"]["local_root"])
        / config["data"]["simulation_teacher_ready_registry"]["relative_paths"]["simulation_teacher_catalog"]
    )
    counterfactual_frame = optional_tsv_frame(
        resolve_path(config["data"]["counterfactual_benchmark_ready_registry"]["local_root"])
        / config["data"]["counterfactual_benchmark_ready_registry"]["relative_paths"]["counterfactual_benchmark_catalog"]
    )
    capability_frame = optional_tsv_frame(
        Path(config["screening_pipeline"]["screening_benchmark"]["artifact_root"]).resolve()
        / benchmark_name
        / config["screening_pipeline"]["screening_benchmark"]["relative_paths"]["capability_catalog"]
    )
    program_status: dict[str, dict[str, object]] = {}
    for disease_program in sorted(declared_programs):
        reasons: list[str] = []
        if disease_program not in configured_programs:
            reasons.append("benchmark_program_missing")
            program_status[disease_program] = {"status": "incomplete", "reasons": reasons}
            continue
        program_config = configured_programs[disease_program]
        cohort_rows = cohort_frame.loc[cohort_frame["disease_program"].astype(str) == disease_program].copy() if not cohort_frame.empty else pd.DataFrame()
        if cohort_rows.empty:
            reasons.append("cohort_catalog_missing")
        minimum_included_primary = int(program_config.get("minimum_included_primary_study_count", 1))
        included_primary_count = int(
            cohort_rows.loc[
                cohort_rows["cohort_role"].astype(str).eq("primary_train_eval")
                & cohort_rows["inclusion_flag"].astype(str).eq("yes")
            ].shape[0]
        ) if not cohort_rows.empty else 0
        if included_primary_count < minimum_included_primary:
            reasons.append("included_primary_cohort_insufficient")
        fold_count = int(fold_frame.loc[fold_frame["disease_program"].astype(str) == disease_program].shape[0]) if not fold_frame.empty else 0
        if fold_count <= 0:
            reasons.append("fold_contract_missing")
        world_state_count = int(world_state_frame.loc[world_state_frame["disease_program"].astype(str) == disease_program].shape[0]) if not world_state_frame.empty else 0
        if world_state_count <= 0:
            reasons.append("world_state_rows_missing")
        completeness_gate = dict(program_config.get("completeness_gate", {}))
        if completeness_gate.get("require_simulation_teacher_rows", False):
            teacher_count = int(simulation_teacher_frame.loc[simulation_teacher_frame["disease_program"].astype(str) == disease_program].shape[0]) if not simulation_teacher_frame.empty else 0
            if teacher_count <= 0:
                reasons.append("simulation_teacher_rows_missing")
        if completeness_gate.get("require_counterfactual_rows", False):
            counterfactual_count = int(counterfactual_frame.loc[counterfactual_frame["disease_program"].astype(str) == disease_program].shape[0]) if not counterfactual_frame.empty else 0
            if counterfactual_count <= 0:
                reasons.append("counterfactual_rows_missing")
        if completeness_gate.get("require_capability_rows", False):
            capability_rows = capability_frame.loc[capability_frame["disease_program"].astype(str) == disease_program].copy() if not capability_frame.empty else pd.DataFrame()
            if capability_rows.empty:
                reasons.append("capability_rows_missing")
            else:
                hard_gate_rows = capability_rows.loc[
                    capability_rows["capability_family"].astype(str).eq("hard_gate")
                    & capability_rows["metric_name"].astype(str).eq("contract_pass_fraction")
                ].copy()
                required_hard_gates = {
                    "structured_world_state",
                    "causal_dynamics",
                    "long_horizon_action_rollout",
                    "counterfactual_planning",
                }
                available_hard_gates = {str(value) for value in hard_gate_rows["capability_name"].astype(str).tolist()}
                if available_hard_gates != required_hard_gates:
                    reasons.append("hard_gate_summary_missing")
                elif not hard_gate_rows["pass_flag"].astype(str).eq("yes").all():
                    reasons.append("hard_gates_not_all_passed")
        program_status[disease_program] = {
            "status": "complete" if not reasons else "incomplete",
            "reasons": reasons,
            "included_primary_cohort_count": included_primary_count,
            "fold_count": fold_count,
            "world_state_row_count": world_state_count,
        }
    incomplete_programs = [
        disease_program
        for disease_program, payload in program_status.items()
        if payload["status"] != "complete"
    ]
    return {
        "status": "complete" if not incomplete_programs else "incomplete",
        "incomplete_disease_programs": sorted(incomplete_programs),
        "program_status": program_status,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    serving_config = config["screening_pipeline"]["screening_serving"]
    benchmark_config = config["screening_pipeline"]["screening_benchmark"]
    model_version = args.model_version or serving_config["default_model_version"]
    production_dir = Path(benchmark_config["artifact_root"]).resolve() / benchmark_config["benchmark_name"] / "production" / "ad_screening_system"
    if not (production_dir / "model_weights.pt").exists():
        raise FileNotFoundError(f"Missing production ad_screening_system under {production_dir}")
    bundle_root = (REPO_ROOT / serving_config["production_root_template"].format(benchmark_name=benchmark_config["benchmark_name"], model_version=model_version)).resolve()
    ensure_directory(bundle_root)

    filenames = serving_config["bundle_filenames"]
    shutil.copy2(production_dir / "model_weights.pt", bundle_root / filenames["model_weights"])
    required_bundle_assets = [
        (resolve_path(config["screening_pipeline"]["action_ready_registry"]["local_root"]) / config["screening_pipeline"]["action_ready_registry"]["relative_paths"]["target_program_catalog"], filenames["target_program_catalog"]),
        (resolve_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"]) / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["target_catalog"], filenames["target_catalog"]),
        (resolve_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"]) / config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]["compound_catalog"], filenames["compound_catalog"]),
        (resolve_path(config["screening_pipeline"]["action_ready_registry"]["local_root"]) / config["screening_pipeline"]["action_ready_registry"]["relative_paths"]["regimen_catalog"], filenames["regimen_catalog"]),
        (resolve_path(config["data"]["sample_observation_registry"]["local_root"]) / config["data"]["sample_observation_registry"]["relative_paths"]["sample_observation_catalog"], filenames["sample_observation_catalog"]),
        (resolve_path(config["measurement_semantics_registry"]["local_root"]) / config["measurement_semantics_registry"]["relative_paths"]["measurement_semantics_catalog"], filenames["measurement_semantics_catalog"]),
        (resolve_path(config["twin"]["intervention_program_registry"]["local_root"]) / config["twin"]["intervention_program_registry"]["relative_paths"]["intervention_program_catalog"], filenames["intervention_program_catalog"]),
        (resolve_path(config["validation"]["candidate_validation_registry"]["local_root"]) / config["validation"]["candidate_validation_registry"]["relative_paths"]["candidate_validation_catalog"], filenames["candidate_validation_catalog"]),
        (resolve_path(config["spatial_atlas_registry"]["local_root"]) / config["spatial_atlas_registry"]["relative_paths"]["region_catalog"], filenames["region_catalog"]),
        (resolve_path(config["dynamics_ready_registry"]["local_root"]) / config["dynamics_ready_registry"]["relative_paths"]["species_catalog"], filenames["species_catalog"]),
        (resolve_path(config["dynamics_ready_registry"]["local_root"]) / config["dynamics_ready_registry"]["relative_paths"]["case_validation_catalog"], filenames["case_validation_catalog"]),
        (resolve_path(config["structural_prior_registry"]["local_root"]) / config["structural_prior_registry"]["relative_paths"]["region_prior_catalog"], filenames["region_prior_catalog"]),
        (resolve_path(config["structural_prior_registry"]["local_root"]) / config["structural_prior_registry"]["relative_paths"]["local_response_head_catalog"], filenames["local_response_head_catalog"]),
        (resolve_path(config["validation"]["external_adapter_registry"]["local_root"]) / config["validation"]["external_adapter_registry"]["relative_paths"]["external_adapter_catalog"], filenames["external_adapter_catalog"]),
    ]
    for source_path, output_name in required_bundle_assets:
        shutil.copy2(source_path, bundle_root / output_name)
    optional_bundle_assets = [
        (
            resolve_path(config["data"]["world_state_ready_registry"]["local_root"])
            / config["data"]["world_state_ready_registry"]["relative_paths"]["world_state_catalog"],
            filenames["world_state_catalog"],
        ),
        (
            resolve_path(config["data"]["crossmodal_anchor_ready_registry"]["local_root"])
            / config["data"]["crossmodal_anchor_ready_registry"]["relative_paths"]["crossmodal_anchor_catalog"],
            filenames["crossmodal_anchor_catalog"],
        ),
        (
            resolve_path(config["data"]["simulation_teacher_ready_registry"]["local_root"])
            / config["data"]["simulation_teacher_ready_registry"]["relative_paths"]["simulation_teacher_catalog"],
            filenames["simulation_teacher_catalog"],
        ),
        (
            resolve_path(config["data"]["counterfactual_benchmark_ready_registry"]["local_root"])
            / config["data"]["counterfactual_benchmark_ready_registry"]["relative_paths"]["counterfactual_benchmark_catalog"],
            filenames["counterfactual_benchmark_catalog"],
        ),
        (
            resolve_path(config["data"]["feedback_ready_registry"]["local_root"])
            / config["data"]["feedback_ready_registry"]["relative_paths"]["feedback_event_catalog"],
            filenames["feedback_event_catalog"],
        ),
    ]
    for source_path, output_name in optional_bundle_assets:
        if source_path.exists():
            shutil.copy2(source_path, bundle_root / output_name)

    base_shared_dir = Path(config["benchmark_registry"]["artifact_root"]).resolve() / config["benchmark_registry"]["benchmark_name"] / "production" / "shared_latent_dual_encoder_classifier"
    shutil.copy2(base_shared_dir / "model_weights.pt", bundle_root / "base_shared_latent_model_weights.pt")
    shutil.copy2(base_shared_dir / "transcriptomics_entity_index.parquet", bundle_root / "transcriptomics_entity_index.parquet")
    shutil.copy2(base_shared_dir / "metabolomics_entity_index.parquet", bundle_root / "metabolomics_entity_index.parquet")

    preprocessing_manifest = {
        "benchmark_name": benchmark_config["benchmark_name"],
        "model_version": model_version,
        "active_disease_program": benchmark_config["active_disease_program"],
        "disease_families": config["foundation_twin"]["disease_families"],
        "tri_threshold": benchmark_config["tri_threshold"],
        "supported_modalities": ["transcriptomics", "metabolomics"],
        "foundation_target_modalities": config["foundation_twin"]["supported_modalities"],
        "screening_endpoints": serving_config["public_endpoints"] + serving_config["compatibility_endpoints"],
        "structural_prior_source": "human_atlas_plus_disease_vulnerability",
        "observation_contract": "sample_observation_catalog",
        "measurement_semantics_contract": "measurement_semantics_catalog",
        "world_state_contract": "world_state_catalog",
        "feedback_contract": "feedback_event_catalog",
        "artifact_policy": {"forbid_obsolete_alias_fallback": True},
        "long_horizon_rollout": benchmark_config["long_horizon_rollout"],
        "ensemble_uq": benchmark_config["ensemble_uq"],
        "foundation_twin": {
            "twin_role": config["foundation_twin"]["twin_role"],
            "world_state_factors": [row["factor_id"] for row in config["foundation_twin"]["world_state_factors"]],
            "capability_hierarchy": config["foundation_twin"]["capability_hierarchy"],
        },
    }
    completeness_payload = pan_neuro_completeness(config, benchmark_config["benchmark_name"])
    preprocessing_manifest["pan_neuro_completeness"] = completeness_payload
    positive_label = "Alzheimer disease"
    label_dictionary = {"0": "normal aging", "1": positive_label}
    model_card = {
        "model_name": canonical_model_name(config),
        "model_version": model_version,
        "checkpoint_id": f"{model_version}:{neuronsifter_scale_registry(config)['default_training_checkpoint_scale']}",
        "checkpoint_scale": str(neuronsifter_scale_registry(config)["default_training_checkpoint_scale"]),
        "default_inference_profile": str(neuronsifter_scale_registry(config)["default_inference_profile"]),
        "foundation_claim_status": str(neuronsifter_scale_registry(config)["foundation_candidate_claim_policy"]["current_status"]),
        "problem_definition": "public pan-neuro multiscale screening twin for action-conditioned drug screening and counterfactual planning",
        "supervision": "multimodal sample evidence plus solver-grounded teacher trajectories and intervention validation reranking",
        "structural_prior": "Human atlas plus disease vulnerability and multiscale solver grounding",
        "intended_use": "intervention-facing research screening, counterfactual rollout, reliability-aware prioritization, and online feedback ingestion",
        "clinical_use": "not_validated_for_clinical_decision_support",
        "pan_neuro_completeness_status": str(completeness_payload["status"]),
        "incomplete_disease_programs": list(completeness_payload["incomplete_disease_programs"]),
    }
    write_json(bundle_root / filenames["preprocessing_manifest"], preprocessing_manifest)
    write_json(bundle_root / filenames["label_dictionary"], label_dictionary)
    write_json(bundle_root / filenames["model_card"], model_card)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=bundle_root,
        model_version=model_version,
    )


def resolve_path(path_value: str) -> Path:
    return (REPO_ROOT / path_value).resolve()


if __name__ == "__main__":
    main()
