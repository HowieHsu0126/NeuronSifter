from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import resolve_repo_path
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic


PASS = "pass"
FAIL = "fail"
BLOCKER = "blocker"
MAJOR = "major"
MINOR = "minor"


@dataclass(frozen=True)
class AlignmentCheck:
    check_id: str
    check_name: str
    status: str
    severity: str
    evidence: list[str]
    required_fix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multimodal twin data alignment for the paper-grade suite.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--md-output", default="")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def artifact_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_json_object(path_value: object) -> dict[str, object]:
    payload = json.loads(artifact_path(path_value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {artifact_path(path_value)}")
    return payload


def read_tsv_rows(path_value: object) -> list[dict[str, str]]:
    with artifact_path(path_value).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def tsv_row_count(path_value: object) -> int:
    path = artifact_path(path_value)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def registry_config(config: dict[str, object], registry_name: str) -> dict[str, object]:
    candidates = [
        config.get(registry_name, {}),
        config.get("data", {}).get(registry_name, {}) if isinstance(config.get("data", {}), dict) else {},
        config.get("twin", {}).get(registry_name, {}) if isinstance(config.get("twin", {}), dict) else {},
        config.get("validation", {}).get(registry_name, {}) if isinstance(config.get("validation", {}), dict) else {},
        config.get("screening_pipeline", {}).get(registry_name, {}) if isinstance(config.get("screening_pipeline", {}), dict) else {},
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def registry_artifact(config: dict[str, object], registry_name: str, relative_key: str) -> Path:
    payload = registry_config(config, registry_name)
    if not payload:
        return REPO_ROOT / "__missing_registry__" / registry_name / relative_key
    relative_paths = dict(payload.get("relative_paths", {}))
    root = resolve_repo_path(str(payload.get("local_root", "")))
    return root / str(relative_paths.get(relative_key, f"__missing__/{relative_key}"))


def check_row(
    check_id: str,
    check_name: str,
    condition: bool,
    evidence: list[str],
    required_fix: str,
    severity: str = BLOCKER,
) -> AlignmentCheck:
    return AlignmentCheck(
        check_id=check_id,
        check_name=check_name,
        status=PASS if condition else FAIL,
        severity=MINOR if condition else severity,
        evidence=evidence,
        required_fix="No fix required." if condition else required_fix,
    )


def simulation_artifact_paths(config: dict[str, object]) -> dict[str, Path]:
    audit_config = config["ai_ready_preprocessing"]["simulation_audit"]
    root = resolve_repo_path(str(audit_config["local_root"]))
    return {
        str(name): root / str(path_value)
        for name, path_value in dict(audit_config["relative_paths"]).items()
    }


def training_code_paths() -> list[Path]:
    return [
        REPO_ROOT / "Libs/exp/training/pretrain_microenvironment_world_model.py",
        REPO_ROOT / "Libs/exp/training/train_screening_system.py",
        REPO_ROOT / "Libs/exp/evaluation/evaluate_screening_system.py",
        REPO_ROOT / "Libs/scripts/services/serve_screening_api.py",
    ]


def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def full_only_main_pretraining_count(pair_rows: list[dict[str, str]], qc_rows: list[dict[str, str]]) -> int:
    full_only_source_uids = {
        str(row.get("full_source_uid", ""))
        for row in pair_rows
        if str(row.get("paired_status", "")) == "full_only"
    }
    return sum(
        1
        for row in qc_rows
        if str(row.get("source_uid", "")) in full_only_source_uids
        and str(row.get("qc_use", "")) == "main_pretraining"
    )


def main_pretraining_nonpaired_count(qc_rows: list[dict[str, str]]) -> int:
    if not qc_rows or "paired_status" not in qc_rows[0]:
        return 0
    return sum(
        1
        for row in qc_rows
        if str(row.get("qc_use", "")) == "main_pretraining"
        and str(row.get("paired_status", "")) != "paired"
    )


def build_report(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    sim_paths = simulation_artifact_paths(config)
    qc_summary = read_json_object(sim_paths["qc_summary"])
    qc_rows = read_tsv_rows(sim_paths["qc_catalog"])
    pair_rows = read_tsv_rows(sim_paths["pair_alignment_audit"])
    uncertainty_rows = read_tsv_rows(sim_paths["teacher_uncertainty_audit"])
    checks: list[AlignmentCheck] = []

    registry_requirements = {
        "processed_registry": "registry_summary",
        "cohort_ready_registry": "study_summary",
        "feature_ready_registry": "study_summary",
        "analysis_ready_registry": "study_summary",
        "model_input_registry": "study_bundle_catalog",
        "alignment_ready_registry": "aligned_matrix_catalog",
        "drug_target_ready_registry": "target_catalog",
        "pkpd_ready_registry": "compound_pk_prior_catalog",
        "action_ready_registry": "regimen_catalog",
        "candidate_validation_registry": "candidate_validation_catalog",
        "world_state_ready_registry": "world_state_catalog",
        "simulation_teacher_ready_registry": "simulation_teacher_catalog",
        "full_microenvironment_teacher_ready_registry": "full_microenvironment_teacher_catalog",
    }
    registry_evidence = []
    missing_registry_artifacts = []
    for registry_name, relative_key in registry_requirements.items():
        payload = registry_config(config, registry_name)
        path = registry_artifact(config, registry_name, relative_key)
        exists = bool(payload) and path.exists() and tsv_row_count(path) > 0
        registry_evidence.append(f"{registry_name}:{repo_relative(path)}:rows={tsv_row_count(path)}")
        if not exists:
            missing_registry_artifacts.append(f"{registry_name}.{relative_key}")
    checks.append(
        check_row(
            "ALIGN-001",
            "Registry-first data flow artifacts exist.",
            not missing_registry_artifacts,
            registry_evidence,
            "Regenerate the missing registry builders and validators before formal suite launch: "
            + ", ".join(missing_registry_artifacts),
        )
    )

    raw_training_matches = []
    for path in training_code_paths():
        text = source_text(path)
        if "Input/raw" in text:
            raw_training_matches.append(repo_relative(path))
    checks.append(
        check_row(
            "ALIGN-002",
            "Training and serving code do not read raw files directly.",
            not raw_training_matches,
            raw_training_matches or ["no Input/raw reads in training/evaluation/serving code"],
            "Remove direct raw-data reads from training/evaluation/serving paths and route them through processed registries.",
        )
    )

    model_input_count = tsv_row_count(registry_artifact(config, "model_input_registry", "study_bundle_catalog"))
    alignment_count = tsv_row_count(registry_artifact(config, "alignment_ready_registry", "aligned_matrix_catalog"))
    anchor_count = tsv_row_count(registry_artifact(config, "crossmodal_anchor_ready_registry", "crossmodal_anchor_catalog"))
    pkpd_count = tsv_row_count(registry_artifact(config, "pkpd_ready_registry", "compound_pk_prior_catalog"))
    action_count = tsv_row_count(registry_artifact(config, "action_ready_registry", "regimen_catalog"))
    spatial_count = tsv_row_count(registry_artifact(config, "spatial_atlas_registry", "region_catalog"))
    if spatial_count == 0:
        spatial_count = tsv_row_count(registry_artifact(config, "structural_prior_registry", "region_prior_catalog"))
    checks.append(
        check_row(
            "ALIGN-003",
            "Multimodal, spatial, PK/PD, and action registries are jointly materialized.",
            all(count > 0 for count in [model_input_count, alignment_count, anchor_count, pkpd_count, action_count, spatial_count]),
            [
                f"model_input_rows={model_input_count}",
                f"alignment_rows={alignment_count}",
                f"crossmodal_anchor_rows={anchor_count}",
                f"pkpd_rows={pkpd_count}",
                f"action_rows={action_count}",
                f"spatial_rows={spatial_count}",
            ],
            "Regenerate model-input, alignment, crossmodal-anchor, PK/PD, action, and spatial registries before launch.",
        )
    )

    full_only_main_count = full_only_main_pretraining_count(pair_rows, qc_rows)
    nonpaired_main_count = main_pretraining_nonpaired_count(qc_rows)
    checks.append(
        check_row(
            "ALIGN-004",
            "Paired full and mesoscopic teacher rows gate main pretraining.",
            full_only_main_count == 0 and nonpaired_main_count == 0,
            [
                f"paired_count={qc_summary.get('paired_count')}",
                f"paired_full_only_count={qc_summary.get('paired_full_only_count')}",
                f"full_only_main_pretraining_rows={full_only_main_count}",
                f"main_pretraining_nonpaired_rows={nonpaired_main_count}",
            ],
            "Exclude unpaired full-only rows from main pretraining and keep Grade D rows rejected.",
        )
    )

    teacher_family_config = config["screening_pipeline"]["screening_benchmark"]["training"]["teacher_family_training"]
    missing_uncertainty_families = []
    for family in teacher_family_config["families"]:
        family_key = str(family["family_key"])
        if str(family.get("store_format", "")) == "zarr_v2" and not str(family.get("uncertainty_array_name", "")).strip():
            missing_uncertainty_families.append(family_key)
    checks.append(
        check_row(
            "ALIGN-005",
            "Teacher uncertainty is declared and audited for trajectory-backed teacher families.",
            not missing_uncertainty_families and len(uncertainty_rows) > 0,
            [
                f"uncertainty_audit_rows={len(uncertainty_rows)}",
                "missing_uncertainty_families=" + ",".join(missing_uncertainty_families),
            ],
            "Declare uncertainty_array_name for every zarr-backed teacher family and regenerate simulation_teacher_uncertainty_audit.tsv.",
        )
    )

    pretrain_text = source_text(REPO_ROOT / "Libs/exp/training/pretrain_microenvironment_world_model.py")
    train_text = source_text(REPO_ROOT / "Libs/exp/training/train_screening_system.py")
    eval_text = source_text(REPO_ROOT / "Libs/exp/evaluation/evaluate_screening_system.py")
    serving_text = source_text(REPO_ROOT / "Libs/scripts/services/serve_screening_api.py")
    dynamics_tokens_present = all(
        token in pretrain_text + train_text + eval_text + serving_text
        for token in [
            "teacher_family_training",
            "pretraining_objectives",
            "solver_grounded_dynamics",
            "rollout_counterfactuals",
            "plan_interventions",
        ]
    )
    checks.append(
        check_row(
            "ALIGN-006",
            "Dynamics evidence enters loader, objectives, metrics, and serving.",
            dynamics_tokens_present,
            [
                "loader=teacher_family_training",
                "objectives=pretraining_objectives",
                "metrics=solver_grounded_dynamics",
                "serving=rollout_counterfactuals,plan_interventions",
            ],
            "Restore dynamics loaders, training objectives, evaluation metrics, and serving rollout/planning paths.",
        )
    )

    species = [str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"]]
    compartments = [
        str(value)
        for value in config["screening_pipeline"]["screening_benchmark"]["pkpd_prior_consistency"]["compartment_order"]
    ]
    expected_full_species = {
        "free_drug",
        "target_bound_drug",
        "misfolded_protein_load",
        "aggregate_burden",
        "neuroinflammation",
        "clearance_flux",
        "circuit_instability",
        "amyloid_beta_soluble",
        "amyloid_plaque",
        "phosphorylated_tau",
        "inflammatory_mediator",
        "calcium_concentration",
        "membrane_voltage",
        "spike_rate",
        "vascular_perivascular_transport",
    }
    checks.append(
        check_row(
            "ALIGN-007",
            "State-grid species and compartment coverage match the AD contract.",
            len(species) == 15 and len(compartments) == 6 and expected_full_species <= set(species),
            [
                f"species_count={len(species)}",
                f"compartment_count={len(compartments)}",
                "missing_species=" + ",".join(sorted(expected_full_species - set(species))),
            ],
            "Update configs, datasets, validators, and reports together if the state-grid contract changes.",
        )
    )

    checks.append(
        check_row(
            "ALIGN-008",
            "Effective counts preserve G2 pass-with-warnings and G5 not-passed boundaries.",
            str(qc_summary.get("readiness_status")) == "G2_pass_with_warnings"
            and int(qc_summary.get("train_eligible_count", 0)) == 222336
            and int(qc_summary.get("weak_teacher_count", -1)) == 0
            and int(qc_summary.get("total_split_leakage_count", -1)) == 0,
            [
                f"readiness_status={qc_summary.get('readiness_status')}",
                f"train_eligible_count={qc_summary.get('train_eligible_count')}",
                f"weak_teacher_count={qc_summary.get('weak_teacher_count')}",
                f"reject_count={qc_summary.get('reject_count')}",
                f"total_split_leakage_count={qc_summary.get('total_split_leakage_count')}",
            ],
            "Regenerate simulation QC/readiness artifacts and freeze reports before changing formal-suite status.",
        )
    )

    failed_checks = [check for check in checks if check.status != PASS]
    return {
        "audit_name": "multimodal_twin_data_alignment",
        "config_path": repo_relative(artifact_path(config_path)),
        "status": "pass" if not failed_checks else "fail",
        "failed_check_count": len(failed_checks),
        "checks": [asdict(check) for check in checks],
    }


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Multimodal Twin Data Alignment Audit",
        "",
        f"- status: `{report['status']}`",
        f"- failed_check_count: `{report['failed_check_count']}`",
        "",
        "| check_id | status | severity | check_name | evidence | required_fix |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["checks"]:
        evidence = "<br>".join(str(value) for value in row["evidence"])
        lines.append(
            f"| {row['check_id']} | {row['status']} | {row['severity']} | "
            f"{row['check_name']} | {evidence} | {row['required_fix']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs = config.get("paper_grade_full_suite", {}).get("output_paths", {})
    json_output = args.json_output or str(outputs.get("multimodal_alignment_json", "Output/audits/multimodal_twin_data_alignment_audit.json"))
    md_output = args.md_output or str(outputs.get("multimodal_alignment_md", "Output/audits/multimodal_twin_data_alignment_audit.md"))
    report = build_report(args.config)
    write_json(artifact_path(json_output), report)
    write_text_atomic(artifact_path(md_output), markdown_report(report))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
