from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import ensure_directory, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate publication-readiness contracts for NeuronSifter.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", default="Output/audits/publication_readiness_contracts.json")
    return parser.parse_args()


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def require_subset(required_values: list[str], available_values: set[str], label: str) -> None:
    missing_values = sorted(set(required_values) - available_values)
    if missing_values:
        raise ValueError(f"{label} missing required values: {missing_values}")


def require_float_equal(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1.0e-9:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def require_true(actual: object, label: str) -> None:
    if bool(actual) is not True:
        raise ValueError(f"{label} expected true, found {actual!r}")


def require_positive_int(actual: object, label: str) -> None:
    if int(actual) <= 0:
        raise ValueError(f"{label} expected positive integer, found {actual!r}")


def require_contains(required_value: str, available_values: list[object], label: str) -> None:
    available_text = {str(value) for value in available_values}
    if required_value not in available_text:
        raise ValueError(f"{label} missing {required_value!r}; found {sorted(available_text)}")


def resolve_artifact_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validate_tsv_schema(path: Path, fields: list[str]) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_equal(reader.fieldnames, fields, f"{path} fieldnames")
        row_count = sum(1 for _ in reader)
    return {
        "artifact_path": str(path),
        "row_count": row_count,
        "schema_status": "schema_valid",
    }


def validate_json_artifact(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "artifact_path": str(path),
        "payload_type": type(payload).__name__,
        "schema_status": "json_parse_valid",
    }


def read_json_artifact(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} expected JSON object, found {type(payload).__name__}")
    return payload


def walk_json_keys(payload: object) -> list[str]:
    keys: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.append(str(key))
            keys.extend(walk_json_keys(value))
    elif isinstance(payload, list):
        for value in payload:
            keys.extend(walk_json_keys(value))
    return keys


def validate_adni_publication_boundary(config: dict) -> list[dict[str, object]]:
    if "adni_controlled_access" not in config:
        return []
    adni_config = config["adni_controlled_access"]
    require_equal(config["source_policy"]["strict_public_only_flag"], True, "source_policy.strict_public_only_flag")
    require_equal(adni_config["governance"]["publication_boundary"], "preserve_public_only_mainline", "ADNI publication boundary")
    checkpoint_config = adni_config["application"]["screening_checkpoint"]
    require_equal(
        checkpoint_config["selection_policy"],
        "current_public_mainline_production",
        "ADNI screening checkpoint selection policy",
    )
    require_equal(checkpoint_config["expected_scope"], "production", "ADNI screening checkpoint scope")
    require_equal(checkpoint_config["expected_fold_name"], "production", "ADNI screening checkpoint fold name")
    require_equal(checkpoint_config["expected_model_name"], "ad_screening_system", "ADNI screening checkpoint model name")
    require_equal(
        checkpoint_config["expected_model_contract_version"],
        config["screening_pipeline"]["screening_benchmark"]["model_contract_version"],
        "ADNI screening checkpoint contract version",
    )
    require_equal(checkpoint_config["state_dict_strict"], True, "ADNI screening checkpoint strict state load")
    validator_config = adni_config["validators"]
    suffixes = tuple(str(value) for value in validator_config["public_text_scan_suffixes"])
    forbidden_fragments = [str(value) for value in validator_config["forbidden_public_path_fragments"]]
    forbidden_keys = {str(value) for value in validator_config["forbidden_public_json_keys"]}
    scanned_file_count = 0
    for root_value in validator_config["public_scan_roots"]:
        root_path = resolve_artifact_path(root_value)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            scanned_file_count += 1
            relative_text = path.resolve().relative_to(REPO_ROOT).as_posix() if path.resolve().is_relative_to(REPO_ROOT) else str(path)
            if any(fragment in relative_text for fragment in forbidden_fragments):
                raise ValueError(f"ADNI participant-level artifact path appears under a public publication root: {relative_text}")
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                overlapping_keys = forbidden_keys & set(walk_json_keys(payload))
                if overlapping_keys:
                    raise ValueError(f"ADNI participant-level JSON keys appear under a public publication root: {relative_text}")
    return [
        {
            "gate_name": "adni_controlled_access_publication_boundary",
            "gate_status": "pass",
            "scanned_public_file_count": scanned_file_count,
            "checkpoint_selection_policy": str(checkpoint_config["selection_policy"]),
        }
    ]


def validate_existing_report_artifacts(config: dict) -> list[dict[str, object]]:
    readiness_config = config["paper"]["publication_readiness"]
    schema_config = config["schemas"]
    report_dir = REPO_ROOT / str(readiness_config["report_outputs"]["report_dir"])
    schema_names = dict(
        readiness_config.get("report_outputs", {}).get(
            "output_schemas",
            readiness_config.get("hardening_gates", {}).get("output_schemas", {}),
        )
    )
    validations: list[dict[str, object]] = []
    for artifact_name in [str(value) for value in readiness_config["report_outputs"]["required_artifacts"]]:
        artifact_path = report_dir / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Required publication report artifact is missing: {artifact_path}")
        if artifact_name.endswith(".tsv"):
            schema_key = str(schema_names[artifact_name.removesuffix(".tsv")])
            validations.append(validate_tsv_schema(artifact_path, [str(value) for value in schema_config[schema_key]]))
            continue
        validations.append(validate_json_artifact(artifact_path))
    return validations


def validate_framework_hardening_gates(config: dict) -> list[dict[str, object]]:
    readiness_config = config["paper"]["publication_readiness"]
    hardening_gates = readiness_config["hardening_gates"]
    report_dir = REPO_ROOT / str(readiness_config["report_outputs"]["report_dir"])
    measurement_payload = read_json_artifact(report_dir / "measurement_semantics_audit.json")
    scaler_payload = read_json_artifact(report_dir / "scaler_leakage_audit.json")
    dynamics_payload = read_json_artifact(report_dir / "dynamics_calibration_audit.json")
    alignment_payload = read_json_artifact(report_dir / "crossscale_alignment_audit.json")
    claim_payload = read_json_artifact(report_dir / "claim_discipline_audit.json")

    measurement_gate = hardening_gates["measurement_semantics_registry"]
    if bool(measurement_gate["require_registry_catalog"]):
        require_equal(measurement_payload["status"], "contract_satisfied", "measurement semantics audit status")
        require_positive_int(measurement_payload["semantics_row_count"], "measurement semantics row count")
        catalog_path = resolve_artifact_path(measurement_payload["catalog_path"])
        if not catalog_path.exists():
            raise FileNotFoundError(f"Measurement semantics catalog is missing: {catalog_path}")
    if bool(measurement_gate["require_sample_semantics_coverage"]):
        require_equal(measurement_payload["coverage_status"], "complete", "measurement semantics coverage")
        require_positive_int(measurement_payload["sample_semantics_coverage_count"], "sample semantics coverage count")
    if bool(measurement_gate["require_world_state_semantics_coverage"]):
        require_equal(measurement_payload["coverage_status"], "complete", "world-state semantics coverage")
        require_positive_int(measurement_payload["world_state_semantics_coverage_count"], "world-state semantics coverage count")

    scaler_gate = hardening_gates["scaler_leakage"]
    if bool(scaler_gate["require_bundle_scaler_artifacts"]):
        require_equal(scaler_payload["status"], "bundle_scaler_artifacts_materialized", "scaler leakage audit status")
        require_true(scaler_payload["all_scaler_artifacts_exist"], "all scaler artifacts exist")
    if bool(scaler_gate["require_inverse_transform_artifacts"]):
        require_true(scaler_payload["all_inverse_transform_artifacts_declared"], "all inverse transform artifacts declared")
    require_equal(scaler_payload["leakage_guard"], scaler_gate["leakage_guard"], "scaler leakage guard")
    require_true(scaler_payload["all_leakage_guards_match_policy"], "scaler artifact leakage guards")

    dynamics_gate = hardening_gates["dynamics_calibration"]
    if bool(dynamics_gate["require_provisional_main_dynamics_label"]):
        require_contains("calibration_pending", list(dynamics_payload["mainline_calibration_statuses"]), "mainline dynamics calibration status")
        require_contains("configuration_generated_provisional", list(dynamics_payload["mainline_teacher_evidence_levels"]), "mainline dynamics evidence level")
    if bool(dynamics_gate["require_teacher_loss_weight_audit"]):
        require_equal(dynamics_payload["teacher_loss_weight_field_status"], "present", "teacher loss weight field")
        if float(dynamics_payload["mean_teacher_loss_weight"]) <= 0.0:
            raise ValueError(f"mean_teacher_loss_weight must be positive: {dynamics_payload['mean_teacher_loss_weight']!r}")
    if bool(dynamics_gate["require_external_calibration_claim_only_when_validated"]):
        if str(dynamics_payload["external_calibration_claim_status"]) != "external_calibration_validated":
            require_equal(dynamics_payload["external_calibration_claim_status"], "not_claimed", "external calibration claim status")
            require_equal(claim_payload["dynamics_accuracy_claim_scope"], "mechanistic_teacher_pretraining_only", "dynamics claim scope")
            require_contains("external_calibrated_dynamics_accuracy", list(claim_payload["prohibited_evidence_upgrades"]), "prohibited evidence upgrades")

    alignment_gate = hardening_gates["crossscale_alignment"]
    if bool(alignment_gate["require_paired_anchor_count"]):
        require_positive_int(alignment_payload["paired_anchor_count"], "paired anchor count")
    if bool(alignment_gate["require_weak_anchor_distinction"]):
        require_equal(alignment_payload["weak_anchor_policy"], "distributional_alignment_only", "weak anchor policy")
        require_positive_int(alignment_payload["weak_anchor_count"], "weak anchor count")
    if bool(alignment_gate["require_registry_level_alignment_only_for_weak_sources"]):
        if int(alignment_payload["weak_anchor_count"]) > 0:
            require_equal(alignment_payload["full_crossscale_alignment_claim_status"], "not_claimed", "full crossscale alignment claim status")
            require_equal(claim_payload["crossscale_alignment_claim_scope"], "paired_sample_sources_plus_distributional_weak_sources", "crossscale claim scope")
            require_contains("full_crossscale_alignment", list(claim_payload["prohibited_evidence_upgrades"]), "prohibited evidence upgrades")

    require_equal(claim_payload["wording_status"], "evidence_scoped_guardrail_contract_declared", "claim discipline wording status")
    require_equal(claim_payload["measurement_semantics_status"], measurement_payload["status"], "claim discipline measurement status")
    require_equal(claim_payload["scaler_leakage_status"], scaler_payload["status"], "claim discipline scaler status")
    require_equal(claim_payload["dynamics_calibration_status"], dynamics_payload["status"], "claim discipline dynamics status")
    require_equal(claim_payload["crossscale_alignment_status"], alignment_payload["status"], "claim discipline crossscale status")
    return [
        {
            "gate_name": "measurement_semantics_registry",
            "gate_status": "pass",
            "artifact_path": str(report_dir / "measurement_semantics_audit.json"),
        },
        {
            "gate_name": "scaler_leakage",
            "gate_status": "pass",
            "artifact_path": str(report_dir / "scaler_leakage_audit.json"),
        },
        {
            "gate_name": "dynamics_calibration_claim_scope",
            "gate_status": "pass",
            "artifact_path": str(report_dir / "dynamics_calibration_audit.json"),
        },
        {
            "gate_name": "crossscale_alignment_claim_scope",
            "gate_status": "pass",
            "artifact_path": str(report_dir / "crossscale_alignment_audit.json"),
        },
        {
            "gate_name": "claim_discipline_scope_consistency",
            "gate_status": "pass",
            "artifact_path": str(report_dir / "claim_discipline_audit.json"),
        },
    ]


def build_report(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    readiness_config = config["paper"]["publication_readiness"]
    contract = config["supplemental_milestone_contracts"]["publication_readiness"]

    positioning = readiness_config["research_positioning"]
    require_equal(positioning["method_claim"], contract["required_method_claim"], "method claim")
    require_equal(positioning["disease_scope"], contract["required_disease_scope"], "disease scope")
    require_equal(positioning["endpoint_policy"], contract["required_endpoint_policy"], "endpoint policy")
    require_equal(positioning["data_boundary"], contract["required_data_boundary"], "data boundary")

    submission_policy = readiness_config["submission_policy"]
    require_equal(submission_policy["readiness_failure_policy"], contract["required_submission_failure_policy"], "submission failure policy")
    require_equal(str(submission_policy["readiness_freeze_date"]), str(contract["required_readiness_freeze_date"]), "readiness freeze date")
    materialization_policy = submission_policy["formal_result_materialization_policy"]
    materialization_contract = contract["required_formal_result_materialization_policy"]
    require_equal(bool(materialization_policy["fail_fast_before_main_tables"]), bool(materialization_contract["fail_fast_before_main_tables"]), "formal result fail-fast policy")
    require_equal([str(value) for value in materialization_policy["ready_result_statuses"]], [str(value) for value in materialization_contract["ready_result_statuses"]], "formal ready result statuses")
    require_subset([str(value) for value in materialization_contract["required_surfaces"]], set(str(value) for value in materialization_policy["required_surfaces"]), "formal result surfaces")

    claim_ids = {str(row["claim_id"]) for row in readiness_config["core_claims"]}
    require_subset([str(value) for value in contract["required_core_claims"]], claim_ids, "core claims")
    for claim_config in readiness_config["core_claims"]:
        require_equal(claim_config["required_bootstrap_ci"], contract["required_bootstrap_interval"], f"{claim_config['claim_id']} bootstrap interval")
        require_equal(claim_config["acceptance_delta_policy"], "ci_excludes_zero_and_direction_consistent", f"{claim_config['claim_id']} acceptance policy")

    strict_splits = set(str(value) for value in readiness_config["strict_splits"]["split_names"])
    require_subset([str(value) for value in contract["required_strict_splits"]], strict_splits, "strict splits")

    statistical_contract = readiness_config["statistical_contract"]
    require_equal(statistical_contract["primary_interval"], contract["required_bootstrap_interval"], "primary interval")
    require_equal(int(statistical_contract["bootstrap_sample_count"]), int(contract["required_bootstrap_sample_count"]), "bootstrap sample count")
    require_equal(statistical_contract["acceptance_rule"], "claim_delta_ci_must_exclude_zero", "statistical acceptance rule")

    g5_config = readiness_config["g5_foundation_pretraining"]
    require_equal(g5_config["effective_sample_gate"], contract["required_g5_gate"], "G5 effective gate")
    require_equal(int(g5_config["store_trajectory_min_count"]), int(contract["required_g5_store_trajectory_min_count"]), "G5 store count")
    require_equal(g5_config["headline_profile"], contract["required_g5_headline_profile"], "G5 headline profile")
    require_subset([str(value) for value in contract["required_g5_gate_terms"]], set(str(value) for value in g5_config["required_gate_terms"]), "G5 gate terms")
    for quota_name, expected_value in contract["required_g5_quota"].items():
        require_float_equal(float(g5_config["expansion_quota"][quota_name]), float(expected_value), f"G5 quota {quota_name}")
    require_float_equal(sum(float(value) for value in g5_config["expansion_quota"].values()), float(g5_config["quota_sum"]), "G5 quota sum")
    require_equal(bool(g5_config["candidate_boundary"]["allow_molecule_generation"]), False, "G5 molecule generation boundary")
    require_equal(bool(g5_config["candidate_boundary"]["allow_analog_expansion"]), False, "G5 analog expansion boundary")
    require_equal(bool(g5_config["candidate_boundary"]["allow_private_patient_records"]), False, "G5 public-only boundary")

    metric_catalog = readiness_config["metric_catalog"]
    require_subset([str(value) for value in contract["required_uq_metrics"]], set(str(value) for value in metric_catalog["uncertainty"]), "UQ metrics")
    require_subset([str(value) for value in contract["required_tri_metrics"]], set(str(value) for value in metric_catalog["tri"]), "TRI metrics")

    baseline_ids = {str(row["baseline_id"]) for row in readiness_config["baseline_grid"]["baselines"]}
    require_subset([str(value) for value in contract["required_baselines"]], baseline_ids, "baseline grid")
    require_equal(readiness_config["baseline_grid"]["action_ablation_policy"]["shared_backbone"], "neuronsifter_large", "action ablation backbone")

    tri_config = readiness_config["tri_calibration"]
    require_equal(tri_config["threshold_source_split"], "validation", "TRI threshold source")
    require_equal(tri_config["test_policy"], "evaluate_only_no_threshold_refit", "TRI test policy")
    require_equal(tri_config["source_holdout_policy"], "evaluate_only_no_threshold_refit", "TRI source holdout policy")

    required_artifacts = [str(value) for value in readiness_config["report_outputs"]["required_artifacts"]]
    require_subset([str(value) for value in contract["required_report_artifacts"]], set(required_artifacts), "report artifacts")

    doc_paths = [REPO_ROOT / str(value) for value in contract["required_publication_docs"]]
    missing_docs = [str(path) for path in doc_paths if not path.exists()]
    if missing_docs:
        raise ValueError(f"publication docs missing required files: {missing_docs}")

    artifact_validations = validate_existing_report_artifacts(config)
    hardening_gate_validations = validate_framework_hardening_gates(config)
    controlled_access_validations = validate_adni_publication_boundary(config)
    return {
        "contract_status": "publication_readiness_contract_satisfied",
        "method_claim": str(positioning["method_claim"]),
        "disease_scope": str(positioning["disease_scope"]),
        "strict_split_count": len(strict_splits),
        "core_claim_count": len(claim_ids),
        "baseline_count": len(baseline_ids),
        "g5_store_trajectory_min_count": int(g5_config["store_trajectory_min_count"]),
        "artifact_validations": artifact_validations,
        "hardening_gate_validations": hardening_gate_validations,
        "controlled_access_validations": controlled_access_validations,
    }


def main() -> None:
    args = parse_args()
    report = build_report(REPO_ROOT / args.config)
    output_path = REPO_ROOT / args.output
    ensure_directory(output_path.parent)
    write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
