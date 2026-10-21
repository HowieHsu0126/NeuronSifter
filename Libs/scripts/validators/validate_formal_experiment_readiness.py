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

from Libs.utils.data.g0_read_only_artifacts import resolve_read_only_artifact
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import resolve_repo_path
from Libs.utils.data.raw_data_registry import write_json
from Libs.utils.data.raw_data_registry import write_text_atomic
from Libs.utils.data.trajectory_numeric_lineage import load_canonical_metric_values


IMPLEMENTED = "implemented"
PARTIAL = "partially_implemented"
MISSING = "missing"
INCONSISTENT = "inconsistent"
OVERCLAIMED = "overclaimed"

BLOCKER = "blocker"
MAJOR = "major"
MINOR = "minor"


@dataclass(frozen=True)
class RequirementRow:
    requirement_id: str
    requirement: str
    expected_implementation: str
    actual_implementation_files: list[str]
    actual_tests_validators: list[str]
    artifact_evidence: list[str]
    status: str
    severity: str
    required_fix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate formal NeuronSifter experiment readiness.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--json-output", default="Output/audits/formal_experiment_readiness_audit.json")
    parser.add_argument("--md-output", default="Output/audits/formal_experiment_readiness_audit.md")
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


def path_exists(path_value: object) -> bool:
    return artifact_path(path_value).exists()


def read_json(path_value: object) -> dict[str, object]:
    payload = json.loads(artifact_path(path_value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {artifact_path(path_value)}")
    return payload


def read_tsv(path_value: object) -> list[dict[str, str]]:
    with artifact_path(path_value).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def tsv_header(path_value: object) -> list[str]:
    with artifact_path(path_value).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        return next(reader)


def file_text(path_value: object) -> str:
    return artifact_path(path_value).read_text(encoding="utf-8")


def row(
    requirement_id: str,
    requirement: str,
    expected: str,
    files: list[str],
    tests: list[str],
    evidence: list[str],
    condition: bool,
    required_fix: str,
    fail_status: str = MISSING,
    fail_severity: str = BLOCKER,
) -> RequirementRow:
    return RequirementRow(
        requirement_id=requirement_id,
        requirement=requirement,
        expected_implementation=expected,
        actual_implementation_files=files,
        actual_tests_validators=tests,
        artifact_evidence=evidence,
        status=IMPLEMENTED if condition else fail_status,
        severity=MINOR if condition else fail_severity,
        required_fix="No fix required." if condition else required_fix,
    )


def required_simulation_paths(config: dict[str, object]) -> dict[str, Path]:
    audit_config = config["ai_ready_preprocessing"]["simulation_audit"]
    root = resolve_repo_path(str(audit_config["local_root"]))
    return {
        str(name): root / str(path_value)
        for name, path_value in dict(audit_config["relative_paths"]).items()
    }


def report_artifact_paths(config: dict[str, object]) -> dict[str, Path]:
    readiness = config["paper"]["publication_readiness"]
    report_dir = resolve_repo_path(str(readiness["report_outputs"]["report_dir"]))
    return {
        str(name): report_dir / str(name)
        for name in readiness["report_outputs"]["required_artifacts"]
    }


def serving_bundle_path(config: dict[str, object]) -> Path:
    serving_config = config["screening_pipeline"]["screening_serving"]
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    return resolve_repo_path(
        str(serving_config["production_root_template"]).format(
            benchmark_name=str(benchmark["benchmark_name"]),
            model_version=str(serving_config["default_model_version"]),
        )
    )


def forbidden_report_claims(config: dict[str, object]) -> list[str]:
    report_dir = resolve_repo_path(str(config["paper"]["publication_readiness"]["report_outputs"]["report_dir"]))
    fragments = [
        "private clinical data",
        "direct clinical endpoint supervision",
        "prospective clinical validation",
        "pd/ftd/als readiness",
        "pd readiness",
        "ftd readiness",
        "als readiness",
        "1m trajectory readiness passed",
        "g5 passed",
        "materialized 1m trajectory gate passed",
    ]
    if not report_dir.exists():
        return []
    matches: list[str] = []
    for path in sorted(report_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".tex", ".txt", ".tsv"}:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for fragment in fragments:
            if fragment in text:
                matches.append(f"{repo_relative(path)}::{fragment}")
    return matches


def tsv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def required_bundle_filenames(config: dict[str, object]) -> list[str]:
    filenames = config["screening_pipeline"]["screening_serving"]["bundle_filenames"]
    required_keys = [
        "target_program_catalog",
        "target_catalog",
        "compound_catalog",
        "regimen_catalog",
        "region_catalog",
        "species_catalog",
        "candidate_validation_catalog",
        "world_state_catalog",
        "simulation_teacher_catalog",
        "external_adapter_catalog",
        "model_weights",
        "model_card",
    ]
    return [str(filenames[key]) for key in required_keys]


def submission_scripts_with_job_names() -> list[Path]:
    scripts = sorted((REPO_ROOT / "Libs" / "exp" / "submission").glob("*.sh"))
    return [
        path
        for path in scripts
        if "--job-name" in path.read_text(encoding="utf-8")
    ]


def build_requirement_rows(config: dict[str, object]) -> list[RequirementRow]:
    rows: list[RequirementRow] = []
    project_prefix = (REPO_ROOT / "PROJECT_PREFIX").read_text(encoding="utf-8").strip() if (REPO_ROOT / "PROJECT_PREFIX").exists() else ""
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    architecture = benchmark["model_architecture"]
    transition_backbone = architecture["rollout_operator"]["transition_backbone"]
    decision_gate = benchmark["decision_gate"]
    serving = config["screening_pipeline"]["screening_serving"]
    sim_paths = required_simulation_paths(config)
    qc_summary_path = sim_paths["qc_summary"]
    split_leakage_path = sim_paths["split_leakage_audit"]
    qc_catalog_path = sim_paths["qc_catalog"]
    qc_summary = read_json(qc_summary_path)
    canonical_counts = load_canonical_metric_values(REPO_ROOT / "Libs/configs/config_stack.yaml")
    split_leakage = read_json(split_leakage_path)
    qc_rows = read_tsv(qc_catalog_path)
    report_paths = report_artifact_paths(config)

    world_state_factors = config["foundation_twin"]["world_state_factors"]
    hidden_factor_count = sum(1 for item in world_state_factors if bool(item.get("hidden_state_support")))
    rows.append(
        row(
            "WM-001",
            "Structured world-state representation exists.",
            "Factorized hidden world-state with explicit factor IDs and runtime posterior support.",
            [
                "Libs/configs/core/project.yaml",
                "Libs/models/modules/world_state.py",
                "Libs/models/networks/screening_system.py",
            ],
            ["Libs/tests/test_foundation_twin_contracts.py"],
            [f"hidden_state_factor_count={hidden_factor_count}"],
            hidden_factor_count >= 5 and path_exists("Libs/models/modules/world_state.py"),
            "Restore factorized z_tx/z_pk/z_env/z_ephys/z_morph support and world-state posterior runtime code.",
        )
    )
    rows.append(
        row(
            "WM-002",
            "Hidden state and uncertainty are explicit, not surface-only feature fitting.",
            "Training and serving expose uncertainty weighting, posterior factors, and stochastic summaries.",
            [
                "Libs/exp/training/pretrain_microenvironment_world_model.py",
                "Libs/models/modules/world_state.py",
                "Libs/scripts/services/serve_screening_api.py",
            ],
            ["Libs/tests/test_screening_uq.py", "Libs/tests/test_foundation_training_integration.py"],
            [
                f"uncertainty_weighting_enabled={benchmark['training']['uncertainty_weighting']['enabled']}",
                f"required_world_state_factors={len(world_state_factors)}",
            ],
            bool(benchmark["training"]["uncertainty_weighting"]["enabled"])
            and "stochastic_uncertainty_summary" in file_text("Libs/scripts/services/serve_screening_api.py"),
            "Wire teacher uncertainty and stochastic summary fields through training and serving.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )
    rows.append(
        row(
            "WM-003",
            "Action-conditioned dynamics p(s_next | s, a) is implemented.",
            "Field rollout consumes regimen/action features and NeuronSifter transition backbone.",
            [
                "Libs/models/modules/field_rollout.py",
                "Libs/models/modules/neuronsifter_world_action.py",
                "Libs/configs/model/screening_model.yaml",
            ],
            ["Libs/tests/test_screening_system_multiscale.py"],
            [
                f"transition_kind={transition_backbone['kind']}",
                f"active_action_policy={architecture['rollout_operator']['action_representation']['active_policy']}",
            ],
            transition_backbone["kind"] == "neuronsifter_world_action"
            and architecture["rollout_operator"]["action_representation"]["active_policy"] == "occupancy_program",
            "Use occupancy-program action conditioning with the NeuronSifter transition backbone.",
            fail_status=INCONSISTENT,
        )
    )
    rows.append(
        row(
            "WM-004",
            "Long-horizon action-controllable rollout, counterfactual simulation, and planning exist.",
            "Serving exposes rollout and intervention planning endpoints backed by runtime functions.",
            ["Libs/scripts/services/serve_screening_api.py", "Libs/configs/model/screening_model.yaml"],
            ["Libs/tests/test_foundation_twin_contracts.py"],
            [",".join(str(value) for value in serving["public_endpoints"])],
            all(endpoint in serving["public_endpoints"] for endpoint in ["/rollout_counterfactuals", "/plan_interventions"])
            and "def rollout_counterfactuals" in file_text("Libs/scripts/services/serve_screening_api.py")
            and "def plan_interventions" in file_text("Libs/scripts/services/serve_screening_api.py"),
            "Restore counterfactual rollout and planner endpoints plus runtime handlers.",
        )
    )
    rows.append(
        row(
            "WM-005",
            "Physical, logical, numerical, and leakage constraints are checked.",
            "Simulation numeric audit, pairing audit, and leakage audit are materialized and validated.",
            [
                "Libs/scripts/builders/build_simulation_readiness_audit.py",
                "Libs/scripts/validators/validate_simulation_readiness_audit.py",
                "Input/processed/simulation_audit/",
            ],
            ["Libs/scripts/validators/validate_simulation_readiness_audit.py"],
            [
                repo_relative(sim_paths["numeric_audit"]),
                repo_relative(sim_paths["pair_alignment_audit"]),
                repo_relative(split_leakage_path),
            ],
            sim_paths["numeric_audit"].exists()
            and sim_paths["pair_alignment_audit"].exists()
            and int(qc_summary["total_split_leakage_count"]) == 0,
            "Regenerate simulation numeric, pairing, and leakage audits and fix any failed gate.",
        )
    )

    rows.append(
        row(
            "BOUNDARY-001",
            "AD-first public-only boundary is consistent.",
            "Config keeps strict public-only mainline and AD-first publication disease scope.",
            ["Libs/configs/core/project.yaml", "Libs/configs/paper/paper.yaml"],
            ["Libs/scripts/validators/validate_publication_readiness_contracts.py"],
            [
                f"strict_public_only_flag={config['source_policy']['strict_public_only_flag']}",
                f"publication_disease_scope={config['paper']['publication_readiness']['research_positioning']['disease_scope']}",
            ],
            bool(config["source_policy"]["strict_public_only_flag"])
            and not bool(config["foundation_twin"]["public_data_boundary"]["controlled_access_required"])
            and str(config["paper"]["publication_readiness"]["research_positioning"]["disease_scope"]) == "ad_control",
            "Restore strict public-only policy and AD-first publication scope.",
            fail_status=INCONSISTENT,
        )
    )
    forbidden_matches = forbidden_report_claims(config)
    rows.append(
        row(
            "BOUNDARY-002",
            "Public report wording avoids prohibited private/clinical/generalization claims.",
            "Paper reports do not claim controlled private training, direct clinical supervision, prospective validation, or PD/FTD/ALS readiness.",
            ["Output/paper_reports/"],
            [
                "Libs/scripts/reports/build_paper_results.py",
                "Libs/scripts/validators/validate_publication_readiness_contracts.py",
            ],
            forbidden_matches if forbidden_matches else ["forbidden_public_report_claim_matches=0"],
            len(forbidden_matches) == 0,
            "Remove or evidence-scope forbidden public claim fragments from generated reports.",
            fail_status=OVERCLAIMED,
        )
    )
    rows.append(
        row(
            "BOUNDARY-003",
            "NeuronSifter is mainline and Transolver remains baseline/ablation/reference.",
            "Main transition kind is NeuronSifter; Transolver is only in ablation/reference fields.",
            ["Libs/configs/model/screening_model.yaml", "Libs/models/modules/transolver_operator.py"],
            ["Libs/scripts/validators/validate_supplemental_milestone_contracts.py"],
            [
                f"transition_kind={transition_backbone['kind']}",
                f"transolver_ablation_kind={transition_backbone['transolver_ablation_template']['kind']}",
            ],
            transition_backbone["kind"] == "neuronsifter_world_action"
            and transition_backbone["transolver_ablation_template"]["kind"] == "transolver_microenvironment_operator",
            "Demote Transolver to ablation/reference and keep NeuronSifter as the configured mainline.",
            fail_status=INCONSISTENT,
        )
    )
    design_doc_paths = ["Docs/design/framework.md", "world_model.md"]
    missing_design_docs = [path for path in design_doc_paths if not path_exists(path)]
    design_doc_text = "\n".join(file_text(path) for path in design_doc_paths if path_exists(path))
    required_design_terms = [
        "AD-first",
        "public-only",
        "NeuronSifter",
        "neuronsifter_world_action",
        "occupancy program",
        "TRI reliability gate",
        "G2_pass_with_warnings",
        "G5_not_passed",
    ]
    missing_design_terms = [
        term
        for term in required_design_terms
        if term.lower() not in design_doc_text.lower()
    ]
    rows.append(
        row(
            "DOC-001",
            "Framework and world-model contract documents are present and aligned with current gates.",
            "Docs/design/framework.md and world_model.md must describe public-only AD-first scope, NeuronSifter mainline, occupancy actions, TRI, and G2/G5 separation.",
            design_doc_paths,
            ["Libs/scripts/validators/validate_formal_experiment_readiness.py"],
            [
                f"missing_design_docs={missing_design_docs}",
                f"missing_design_terms={missing_design_terms}",
            ],
            not missing_design_docs and not missing_design_terms,
            "Restore the formal framework/world-model docs from current config and audit artifacts without upgrading G5 status.",
            fail_status=MISSING,
            fail_severity=MAJOR,
        )
    )

    teacher_family_catalogs = [
        str(item["catalog_path"])
        for item in benchmark["training"]["teacher_family_training"]["families"]
        if str(item["family_key"]) in {"mesoscopic_solver_teacher", "full_microenvironment_teacher"}
    ]
    rows.append(
        row(
            "DATA-001",
            "Raw data cannot enter training without registry-first flow.",
            "Training uses model/action/simulation registries and QC-approved simulation catalog, not Input/raw.",
            ["Libs/exp/training/pretrain_microenvironment_world_model.py", "Libs/exp/training/train_screening_system.py"],
            ["Libs/tests/test_foundation_training_integration.py"],
            [f"teacher_family_catalogs={teacher_family_catalogs}"],
            "Input/raw" not in file_text("Libs/exp/training/pretrain_microenvironment_world_model.py")
            and set(teacher_family_catalogs) == {"Input/processed/simulation_audit/simulation_qc_catalog.tsv"},
            "Route training through processed registries and the QC catalog; remove direct raw simulation reads.",
            fail_status=INCONSISTENT,
        )
    )
    required_sim_names = [
        "qc_catalog",
        "pair_alignment_audit",
        "numeric_audit",
        "teacher_uncertainty_audit",
        "inventory_audit",
        "coverage_audit",
        "split_leakage_audit",
        "qc_summary",
    ]
    missing_sim_artifacts = [name for name in required_sim_names if not sim_paths[name].exists()]
    rows.append(
        row(
            "DATA-002",
            "Simulation readiness artifacts are materialized by code.",
            "QC catalog, inventory, coverage, leakage, pairing, uncertainty, numeric, and summary artifacts exist.",
            ["Libs/scripts/builders/build_simulation_readiness_audit.py", "Input/processed/simulation_audit/"],
            ["Libs/scripts/validators/validate_simulation_readiness_audit.py"],
            [f"missing_simulation_artifacts={missing_sim_artifacts}"],
            not missing_sim_artifacts,
            "Run the simulation readiness builder and repair any missing generated artifact.",
        )
    )
    grading = config["ai_ready_preprocessing"]["simulation_audit"]["grading"]
    grade_use = {str(key): str(value) for key, value in dict(grading["qc_use_by_grade"]).items()}
    grade_weight = {str(key): float(value) for key, value in dict(grading["sample_weight_by_grade"]).items()}
    inconsistent_qc_rows = [
        row_payload["trajectory_uid"]
        for row_payload in qc_rows
        if str(row_payload["qc_use"]) != grade_use[str(row_payload["qc_grade"])]
        or abs(float(row_payload["sample_weight"]) - grade_weight[str(row_payload["qc_grade"])]) > 1.0e-9
    ]
    rows.append(
        row(
            "DATA-003",
            "QC grade A/B/C/D/S use and weight semantics are enforced.",
            "QC catalog rows match YAML-declared training use and sample weights.",
            ["Libs/configs/data/ai_ready_preprocessing.yaml", "Input/processed/simulation_audit/simulation_qc_catalog.tsv"],
            ["Libs/scripts/validators/validate_simulation_readiness_audit.py"],
            [
                f"qc_row_count={len(qc_rows)}",
                f"inconsistent_qc_row_count={len(inconsistent_qc_rows)}",
            ],
            len(inconsistent_qc_rows) == 0,
            "Regenerate the QC catalog or fix grade/use/weight mapping in the builder.",
            fail_status=INCONSISTENT,
        )
    )
    leakage_fields = [
        "trajectory_random_uid_overlap",
        "family_holdout_family_overlap",
        "regimen_holdout_regimen_overlap",
        "target_program_holdout_overlap",
        "content_hash_overlap",
        "source_holdout_role_overlap",
    ]
    leakage_total = sum(int(split_leakage[field]) for field in leakage_fields)
    rows.append(
        row(
            "DATA-004",
            "UID/content/family/regimen/target-program/source-holdout leakage is zero.",
            "Split leakage audit covers all required leakage axes and reports zero overlap.",
            ["Input/processed/simulation_audit/simulation_split_leakage_audit.json"],
            ["Libs/scripts/validators/validate_simulation_readiness_audit.py"],
            [f"{field}={split_leakage[field]}" for field in leakage_fields],
            leakage_total == 0 and str(split_leakage["leakage_status"]) == "pass",
            "Stop formal launch and repair split construction until all leakage counts are zero.",
            fail_status=INCONSISTENT,
        )
    )
    g5_config = config["paper"]["publication_readiness"]["g5_foundation_pretraining"]
    g5_min_count = int(g5_config["store_trajectory_min_count"])
    g5_materialized = (
        canonical_counts["raw_materialized_full_microenvironment_trajectory_count"] >= g5_min_count
        and canonical_counts["raw_materialized_mesoscopic_trajectory_count"] >= g5_min_count
    )
    g5_report_path = report_paths["g5_effective_sample_audit.json"]
    canonical_g5_report_path = (REPO_ROOT / "Output/paper_reports/g5_effective_sample_audit.json").resolve()
    if g5_report_path.resolve() == canonical_g5_report_path:
        g5_report_path = resolve_read_only_artifact(
            config, "g5_effective_sample_audit", REPO_ROOT
        ).physical_path
    g5_report_overclaims = False
    g5_report_has_counts = False
    if g5_report_path.exists():
        g5_report = read_json(g5_report_path)
        g5_report_overclaims = str(g5_report.get("gate_status", "")).lower() in {"passed", "pass", "g5_pass"} and not g5_materialized
        g5_report_has_counts = all(
            key in g5_report
            for key in [
                "current_full_microenvironment_trajectory_count",
                "current_mesoscopic_trajectory_count",
                "current_train_eligible_record_count",
                "materialized_trajectory_gate_status",
                "effective_sample_counts",
                "effective_coverage",
                "source_holdout_leakage_counts",
            ]
        )
    rows.append(
        row(
            "DATA-005",
            "G2 100k readiness is separated from G5 1M materialized readiness.",
            "Reports must read current counts from artifacts and must not mark G5 passed below 1M.",
            [
                "Input/processed/simulation_audit/simulation_qc_summary.json",
                "Output/paper_reports/g5_effective_sample_audit.json",
            ],
            [
                "Libs/scripts/reports/build_paper_results.py",
                "Libs/scripts/validators/validate_formal_experiment_readiness.py",
            ],
            [
                f"g2_readiness_status={qc_summary['readiness_status']}",
                f"full_count={canonical_counts['raw_materialized_full_microenvironment_trajectory_count']}",
                f"mesoscopic_count={canonical_counts['raw_materialized_mesoscopic_trajectory_count']}",
                f"g5_min_count={g5_min_count}",
                f"g5_materialized={g5_materialized}",
            ],
            not g5_report_overclaims and (not g5_report_path.exists() or g5_report_has_counts),
            "Regenerate G5 audit with current artifact counts and keep materialized trajectory gate not_passed.",
            fail_status=OVERCLAIMED if g5_report_overclaims else PARTIAL,
        )
    )
    effective_sample_keys = {
        "unique_trajectory_families",
        "unique_regimens",
        "unique_target_programs",
        "high_confidence_trajectories",
        "medium_confidence_trajectories",
        "full_microenvironment_trajectories",
        "mesoscopic_trajectories",
        "full_mesoscopic_paired_trajectories",
        "counterfactual_pair_count",
        "train_eligible_grade_ab_records",
        "weak_calibration_records",
        "reject_records",
        "stress_only_records",
    }
    effective_coverage_keys = {
        "full_pair_coverage",
        "mesoscopic_pair_coverage",
        "counterfactual_pair_coverage",
        "qc_grade_a_b_rate",
        "high_or_medium_confidence_rate",
        "source_holdout_leakage_status",
    }
    missing_effective_sample_keys = sorted(effective_sample_keys)
    missing_effective_coverage_keys = sorted(effective_coverage_keys)
    effective_counts_match_summary = False
    source_holdout_leakage_status = "missing"
    if g5_report_path.exists():
        g5_report = read_json(g5_report_path)
        effective_counts = dict(g5_report.get("effective_sample_counts", {}))
        effective_coverage = dict(g5_report.get("effective_coverage", {}))
        missing_effective_sample_keys = sorted(effective_sample_keys - set(effective_counts))
        missing_effective_coverage_keys = sorted(effective_coverage_keys - set(effective_coverage))
        source_holdout_leakage_status = str(effective_coverage.get("source_holdout_leakage_status", "missing"))
        effective_counts_match_summary = (
            not missing_effective_sample_keys
            and not missing_effective_coverage_keys
            and int(effective_counts["unique_trajectory_families"]) == int(qc_summary["unique_trajectory_family_count"])
            and int(effective_counts["unique_regimens"]) == int(qc_summary["unique_regimen_count"])
            and int(effective_counts["unique_target_programs"]) == int(qc_summary["unique_target_program_count"])
            and int(effective_counts["high_confidence_trajectories"]) == int(qc_summary["high_confidence_trajectory_count"])
            and int(effective_counts["medium_confidence_trajectories"]) == int(qc_summary["medium_confidence_trajectory_count"])
            and int(effective_counts["train_eligible_grade_ab_records"]) == canonical_counts["qc_eligible_trajectory_count"]
            and source_holdout_leakage_status == "pass"
        )
    rows.append(
        row(
            "REPORT-001",
            "Paper reports output effective sample counts, not only trajectory row counts.",
            "G5 audit reports unique trajectory families, regimens, target programs, confidence tiers, pair coverage, training-use counts, and source-holdout leakage status from simulation artifacts.",
            [
                "Libs/scripts/reports/build_paper_results.py",
                "Output/paper_reports/g5_effective_sample_audit.json",
                "Input/processed/simulation_audit/simulation_qc_summary.json",
            ],
            [
                "Libs/tests/test_publication_readiness_contracts.py",
                "Libs/scripts/validators/validate_formal_experiment_readiness.py",
            ],
            [
                f"missing_effective_sample_keys={missing_effective_sample_keys}",
                f"missing_effective_coverage_keys={missing_effective_coverage_keys}",
                f"source_holdout_leakage_status={source_holdout_leakage_status}",
            ],
            effective_counts_match_summary,
            "Regenerate G5 effective sample audit from simulation QC, pairing, and leakage artifacts.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )

    rows.append(
        row(
            "NEURONSIFTER-001",
            "NeuronSifter mainline architecture is configured and implemented.",
            "Graph-spectral, temporal-wavelet, AdaLN/DiT, mechanism routing, and diffusion sampler code is present.",
            ["Libs/models/modules/neuronsifter_world_action.py", "Libs/configs/model/screening_model.yaml"],
            ["Libs/tests/test_screening_system_multiscale.py"],
            [
                f"operator_registry_mainline={transition_backbone['operator_registry']['mainline_kind']}",
                f"main_rollout_sampler={transition_backbone['operator_registry']['main_rollout_sampler_kind']}",
            ],
            transition_backbone["operator_registry"]["mainline_kind"] == "neuronsifter_world_action"
            and transition_backbone["operator_registry"]["main_rollout_sampler_kind"] == "quantized_mddm_rollout_operator"
            and "trajectory_tokenizer_diagnostics" in file_text("Libs/models/modules/neuronsifter_world_action.py"),
            "Restore the native NeuronSifter operator and quantized MDDM rollout binding.",
        )
    )
    required_diagnostics = [
        "expert_utilization",
        "router_entropy",
        "operator_splitting_species_attribution",
        "operator_splitting_coupling_matrix",
        "operator_splitting_score_norm_by_expert",
        "operator_splitting_uncertainty_by_expert",
        "operator_splitting_case_activation",
    ]
    neuronsifter_text = file_text("Libs/models/modules/neuronsifter_world_action.py")
    missing_diagnostics = [name for name in required_diagnostics if name not in neuronsifter_text]
    rows.append(
        row(
            "NEURONSIFTER-002",
            "Mechanism experts operate through a shared coupled trajectory model and expose diagnostics.",
            "Configured expert bank has coupling masks and emits required operator-splitting diagnostics.",
            ["Libs/models/modules/neuronsifter_world_action.py", "Libs/configs/model/screening_model.yaml"],
            ["Libs/tests/test_screening_system_multiscale.py"],
            [
                f"operator_splitting_enabled={transition_backbone['operator_splitting']['enabled']}",
                f"missing_diagnostics={missing_diagnostics}",
            ],
            bool(transition_backbone["operator_splitting"]["enabled"])
            and not missing_diagnostics
            and bool(transition_backbone["operator_splitting"]["coupling_masks"]),
            "Wire required expert diagnostics through the shared NeuronSifter operator-splitting path.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )
    sampler = transition_backbone["diffusion_sampler"]
    rows.append(
        row(
            "NEURONSIFTER-003",
            "Stochastic outputs include std, quantiles, ranking stability, sample count, and denoising step count.",
            "Diffusion sampler and quantized MDDM diagnostics expose stochastic rollout evidence.",
            [
                "Libs/models/modules/neuronsifter_world_action.py",
                "Libs/models/modules/quantized_mddm_rollout_operator.py",
            ],
            ["Libs/tests/test_screening_system_multiscale.py"],
            [
                f"sample_count={sampler['sample_count']}",
                f"denoising_step_count={sampler['denoising_step_count']}",
                f"quantile_levels={sampler['quantile_levels']}",
            ],
            int(sampler["sample_count"]) >= 64
            and int(sampler["denoising_step_count"]) >= 100
            and "trajectory_quantiles" in file_text("Libs/models/modules/quantized_mddm_rollout_operator.py"),
            "Set the formal stochastic sampler to at least 64 samples x 100 denoising steps and expose quantile diagnostics.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )
    profile_config = transition_backbone["neuronsifter_profiles"]
    rows.append(
        row(
            "NEURONSIFTER-004",
            "XL, Large, Student profiles and distillation path are declared consistently.",
            "Scale registry and screening config expose active profile and enabled distillation.",
            ["Libs/configs/model/neuronsifter_scale_registry.yaml", "Libs/configs/model/screening_model.yaml"],
            ["Libs/tests/test_config_stack_contracts.py"],
            [
                f"active_screening_profile={profile_config['active_screening_profile']}",
                f"distillation_enabled={profile_config['distillation']['enabled']}",
            ],
            str(profile_config["active_screening_profile"]) in {"Large", "neuronsifter_large"} and bool(profile_config["distillation"]["enabled"]),
            "Restore Large active screening profile and XL/Large/Student distillation declaration.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )

    action_policy = architecture["rollout_operator"]["action_representation"]
    rows.append(
        row(
            "ACTION-001",
            "Action abstraction is target set, drug set, dose, schedule, and route.",
            "Action-ready config and regimen features cover target, drug, dose, schedule, and route dimensions.",
            ["Libs/configs/model/screening_model.yaml", "Libs/utils/evaluation/screening_datasets.py"],
            ["Libs/tests/test_screening_system_multiscale.py"],
            [
                f"active_policy={action_policy['active_policy']}",
                f"route_count={len(config['screening_pipeline']['pkpd_ready_registry']['routes'])}",
                f"schedule_count={len(config['screening_pipeline']['pkpd_ready_registry']['schedules'])}",
                f"dose_count={len(config['screening_pipeline']['pkpd_ready_registry']['dose_grid'])}",
            ],
            action_policy["active_policy"] == "occupancy_program"
            and {"target_identity", "drug_identity", "scalar_exposure"} <= set(action_policy["policies"]["occupancy_program"]["retained_feature_groups"]),
            "Use the occupancy-program action policy and retain target/drug/exposure feature groups.",
            fail_status=INCONSISTENT,
        )
    )
    rows.append(
        row(
            "ACTION-002",
            "Occupancy-conditioned path is distinct from bare drug-ID conditioning.",
            "Config includes drug-ID ablation while mainline retains occupancy-program conditioning, with tests for feature masks.",
            [
                "Libs/configs/exp/experiment_families/action_conditioning_drug_identity.yaml",
                "Libs/tests/test_screening_system_multiscale.py",
            ],
            ["Libs/tests/test_screening_system_multiscale.py::test_action_representation_policy_masks_regimen_features"],
            [
                f"available_action_policies={sorted(action_policy['policies'].keys())}",
            ],
            "drug_id" in action_policy["policies"]
            and "test_action_representation_policy_masks_regimen_features" in file_text("Libs/tests/test_screening_system_multiscale.py"),
            "Add a drug-ID ablation and a regression test proving it masks different features than occupancy conditioning.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )
    rows.append(
        row(
            "ACTION-003",
            "Route-ordering evaluation aligns through simulation catalog regimen UID.",
            "Route ordering tests use regimen UID aligned catalog axes rather than trajectory-index fallback.",
            ["Libs/tests/test_neurodegeneration_twin.py", "Libs/utils/model/dynamics_fields.py"],
            ["Libs/tests/test_neurodegeneration_twin.py::test_route_ordering_exposure_aligns_replicate_axis_from_catalog"],
            ["route_ordering_alignment_test_present=true"],
            "test_route_ordering_exposure_aligns_replicate_axis_from_catalog" in file_text("Libs/tests/test_neurodegeneration_twin.py")
            and "regimen_uid" in file_text("Libs/utils/model/dynamics_fields.py"),
            "Align route-ordering evaluation through regimen UID and retain the regression test.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )

    teacher_families = {
        str(item["family_key"])
        for item in benchmark["training"]["teacher_family_training"]["families"]
    }
    required_teacher_families = {
        "mesoscopic_solver_teacher",
        "full_microenvironment_teacher",
        "osp_pbpk_teacher",
        "neuron_rxd_teacher",
        "chembl_parameterized_teacher",
    }
    objectives = set(benchmark["training"]["pretraining_objectives"])
    rows.append(
        row(
            "TRAIN-001",
            "World-model pretraining reads the QC-approved catalog and uses teacher uncertainty.",
            "Teacher families use processed QC catalog paths, sample weights, and uncertainty/precision loss weighting.",
            ["Libs/exp/training/pretrain_microenvironment_world_model.py", "Libs/configs/model/screening_model.yaml"],
            ["Libs/tests/test_foundation_training_integration.py"],
            [
                f"teacher_families={sorted(teacher_families)}",
                f"objectives={sorted(objectives)}",
            ],
            required_teacher_families <= teacher_families
            and bool(benchmark["training"]["teacher_family_training"]["enabled"])
            and "uncertainty_weighted_mse" in file_text("Libs/exp/training/pretrain_microenvironment_world_model.py"),
            "Restore all required teacher families and uncertainty-weighted loss use.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )
    required_objectives = {
        "masked_state_reconstruction_weight",
        "next_state_prediction_weight",
        "long_horizon_rollout_weight",
        "teacher_distillation_weight",
        "counterfactual_consistency_weight",
        "router_regularization_weight",
    }
    rows.append(
        row(
            "TRAIN-002",
            "Pretraining objectives cover reconstruction, prediction, rollout, distillation, counterfactual consistency, router, and PK/PD.",
            "Configured objectives and training code compute the required loss families.",
            ["Libs/configs/model/screening_model.yaml", "Libs/exp/training/pretrain_microenvironment_world_model.py"],
            ["Libs/tests/test_foundation_training_integration.py"],
            [f"required_objective_missing={sorted(required_objectives - objectives)}"],
            required_objectives <= objectives
            and "pkpd_consistency_head" in file_text("Libs/models/modules/world_state.py")
            and "pkpd_prior_loss" in file_text("Libs/exp/training/train_screening_system.py"),
            "Add missing objective weights in YAML and wire the corresponding loss into training.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )
    train_text = file_text("Libs/exp/training/train_screening_system.py")
    supervised_terms = [
        "target_program_rows",
        "compound_rows",
        "regimen_rows",
        "candidate_validation",
        "world_state",
        "simulation_teacher",
        "counterfactual_benchmark",
        "feedback",
    ]
    missing_supervised_terms = [term for term in supervised_terms if term not in train_text]
    rows.append(
        row(
            "TRAIN-003",
            "Supervised screening consumes target, compound, regimen, validation, world-state, teacher, counterfactual, and feedback registries.",
            "Training entrypoint loads and uses all formal registry families and preserves strict-case leakage checks.",
            ["Libs/exp/training/train_screening_system.py", "Libs/utils/evaluation/strict_case_discovery.py"],
            ["Libs/tests/test_foundation_training_integration.py", "Libs/tests/test_strict_case_discovery.py"],
            [f"missing_training_terms={missing_supervised_terms}"],
            not missing_supervised_terms and "assert_no_forced_anchor_regimens" in train_text,
            "Wire missing registry family into supervised training and preserve strict-case anchor leakage assertions.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )

    validator_paths = [
        "Libs/scripts/validators/validate_processed_registry.py",
        "Libs/scripts/validators/validate_model_input_registry.py",
        "Libs/scripts/validators/validate_simulation_readiness_audit.py",
        "Libs/scripts/validators/validate_supplemental_milestone_contracts.py",
        "Libs/scripts/validators/validate_publication_readiness_contracts.py",
        "Libs/scripts/validators/validate_candidate_validation_registry.py",
        "Libs/scripts/validators/validate_external_adapter_registry.py",
        "Libs/scripts/validators/validate_world_state_ready_registry.py",
        "Libs/scripts/validators/validate_simulation_teacher_ready_registry.py",
        "Libs/scripts/validators/validate_formal_experiment_readiness.py",
    ]
    missing_validators = [path for path in validator_paths if not path_exists(path)]
    rows.append(
        row(
            "VALID-001",
            "Registry and readiness validators cover critical registries and gates.",
            "Validators exist for processed/model-input/simulation/supplemental/publication/external/world-state/teacher/formal readiness.",
            validator_paths,
            validator_paths,
            [f"missing_validators={missing_validators}"],
            not missing_validators,
            "Add missing validators before formal experiment launch.",
        )
    )
    missing_report_artifacts = [
        repo_relative(path)
        for path in report_paths.values()
        if not path.exists()
    ]
    rows.append(
        row(
            "VALID-002",
            "Paper report artifacts are complete before formal launch.",
            "Every publication-readiness required report artifact is materialized and parseable.",
            ["Libs/scripts/reports/build_paper_results.py", "Output/paper_reports/"],
            ["Libs/scripts/validators/validate_publication_readiness_contracts.py"],
            [
                f"required_report_artifact_count={len(report_paths)}",
                f"missing_report_artifact_count={len(missing_report_artifacts)}",
                *missing_report_artifacts[:20],
            ],
            not missing_report_artifacts,
            "Run framework hardening, ADNI aggregate validation if required, and paper result builders until all required artifacts exist.",
            fail_status=MISSING,
        )
    )
    fsdp_path = report_paths.get("fsdp_smoke_summary.json")
    fsdp_status = "missing"
    if fsdp_path is not None and fsdp_path.exists():
        fsdp_status = str(read_json(fsdp_path).get("smoke_status", "missing"))
    gpu_smoke_path = report_paths.get("gpu_stochastic_smoke_summary.json")
    gpu_smoke_status = "missing"
    gpu_smoke_checks_pass = False
    gpu_smoke_missing_diagnostics: list[str] = ["gpu_stochastic_smoke_summary.json"]
    gpu_smoke_counts = "missing"
    if gpu_smoke_path is not None and gpu_smoke_path.exists():
        gpu_smoke_payload = read_json(gpu_smoke_path)
        finite_checks = dict(gpu_smoke_payload.get("finite_checks", {}))
        gpu_smoke_status = str(gpu_smoke_payload.get("smoke_status", "missing"))
        gpu_smoke_missing_diagnostics = [
            str(value)
            for value in list(gpu_smoke_payload.get("missing_operator_diagnostics", []))
        ]
        observed_sample_count = int(gpu_smoke_payload.get("observed_sample_count", 0))
        observed_denoising_step_count = int(gpu_smoke_payload.get("observed_denoising_step_count", 0))
        required_sample_count = int(gpu_smoke_payload.get("required_sample_count", sampler["sample_count"]))
        required_denoising_step_count = int(gpu_smoke_payload.get("required_denoising_step_count", sampler["denoising_step_count"]))
        gpu_smoke_counts = f"{observed_sample_count}x{observed_denoising_step_count}"
        gpu_smoke_checks_pass = (
            gpu_smoke_status == "passed"
            and observed_sample_count == required_sample_count
            and observed_denoising_step_count == required_denoising_step_count
            and observed_sample_count >= int(sampler["sample_count"])
            and observed_denoising_step_count >= int(sampler["denoising_step_count"])
            and all(bool(value) for value in finite_checks.values())
            and not gpu_smoke_missing_diagnostics
        )
    rows.append(
        row(
            "VALID-003",
            "GPU stochastic smoke evidence is present for the configured formal smoke.",
            "Report records finite GPU smoke with 64 samples, 100 denoising steps, quantiles, ranking stability, and operator-splitting diagnostics; formal launch requires pass.",
            [
                "Output/paper_reports/fsdp_smoke_summary.json",
                "Output/paper_reports/gpu_stochastic_smoke_summary.json",
                "Libs/exp/submission/submit_neuronsifter_fsdp_smoke.sh",
                "Libs/scripts/validators/validate_neuronsifter_stochastic_smoke.py",
            ],
            ["Libs/tests/test_screening_system_multiscale.py"],
            [
                f"fsdp_smoke_status={fsdp_status}",
                f"gpu_stochastic_smoke_status={gpu_smoke_status}",
                f"gpu_stochastic_smoke_counts={gpu_smoke_counts}",
                f"missing_operator_diagnostics={gpu_smoke_missing_diagnostics}",
            ],
            fsdp_status == "passed" and gpu_smoke_checks_pass,
            "Run validate_neuronsifter_stochastic_smoke.py on gpu03/gpu1 and regenerate paper reports.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )

    endpoints = set(str(value) for value in serving["public_endpoints"])
    required_endpoints = {
        "/infer_world_state",
        "/rollout_counterfactuals",
        "/plan_interventions",
        "/explain_decision",
        "/ingest_feedback",
    }
    rows.append(
        row(
            "SERVE-001",
            "Serving API exposes required world-state, rollout, planning, explain, and feedback endpoints.",
            "Config and serve_screening_api handlers cover the five public endpoints.",
            ["Libs/scripts/services/serve_screening_api.py", "Libs/configs/model/screening_model.yaml"],
            ["Libs/tests/test_foundation_twin_contracts.py"],
            [f"missing_endpoints={sorted(required_endpoints - endpoints)}"],
            required_endpoints <= endpoints
            and all(endpoint.strip("/") in file_text("Libs/scripts/services/serve_screening_api.py") for endpoint in required_endpoints),
            "Restore all required serving endpoints in config and HTTP dispatch.",
        )
    )
    bundle_root = serving_bundle_path(config)
    required_filenames = required_bundle_filenames(config)
    missing_bundle_assets = [
        filename
        for filename in required_filenames
        if not (bundle_root / filename).exists()
    ]
    rows.append(
        row(
            "SERVE-002",
            "Serving bundle contains catalogs, model card, and weights required for formal inference.",
            "Default serving bundle includes target/regimen/species/candidate/world-state/simulation/external catalogs and weights.",
            [repo_relative(bundle_root), "Libs/exp/export/export_screening_bundle.py"],
            ["Libs/tests/test_platform_contracts.py"],
            [
                f"bundle_root={repo_relative(bundle_root)}",
                f"missing_bundle_assets={missing_bundle_assets}",
            ],
            not missing_bundle_assets,
            "Regenerate the serving bundle with all required registry assets and model weights.",
            fail_status=MISSING,
        )
    )
    service_text = file_text("Libs/scripts/services/serve_screening_api.py")
    rows.append(
        row(
            "SERVE-003",
            "Decision gate uses score, TRI, PK/PD consistency, OOD risk, and validation stage.",
            "Decision label logic gates high score by reliability, PK/PD, OOD, and validation stage.",
            ["Libs/scripts/services/serve_screening_api.py", "Libs/configs/model/screening_model.yaml"],
            ["Libs/tests/test_foundation_twin_contracts.py"],
            [
                f"decision_labels={decision_gate['decision_labels']}",
                f"accepted_validation_stages={decision_gate['accepted_validation_stages']}",
            ],
            all(term in service_text for term in ["predicted_score", "tri_score", "pkpd_consistency", "ood_score", "validation_stage"])
            and all(label in decision_gate["decision_labels"] for label in ["accept", "reject", "active_learning", "external_validation", "insufficient_evidence"]),
            "Restore full decision-gate inputs and five-way label set.",
            fail_status=INCONSISTENT,
        )
    )
    rows.append(
        row(
            "SERVE-004",
            "Missing expert diagnostics are explicit, not fabricated.",
            "Serving returns diagnostic missing status when runtime diagnostics are unavailable.",
            ["Libs/scripts/services/serve_screening_api.py"],
            ["Libs/tests/test_foundation_twin_contracts.py"],
            ["diagnostic_status=not_provided_by_runtime_bundle"],
            "not_provided_by_runtime_bundle" in service_text and "case_level_expert_diagnostics missing required fields" in service_text,
            "Return explicit diagnostic missing status and reject partial supplied diagnostics.",
            fail_status=PARTIAL,
            fail_severity=MAJOR,
        )
    )

    script_paths = submission_scripts_with_job_names()
    bad_prefix_scripts = [
        repo_relative(path)
        for path in script_paths
        if "PROJECT_PREFIX" not in path.read_text(encoding="utf-8")
    ]
    cluster_config = config["cluster_registry"]
    rows.append(
        row(
            "SLURM-001",
            "Slurm scripts use PROJECT_PREFIX and YAML cluster config.",
            "Submission scripts derive job names from PROJECT_PREFIX and resources from config-loaded cluster profiles.",
            [repo_relative(path) for path in script_paths] + ["PROJECT_PREFIX", "Libs/configs/runtime/cluster.yaml"],
            ["Libs/tests/test_config_stack_contracts.py"],
            [
                f"project_prefix_file={project_prefix}",
                f"project_prefix_config={config['project']['prefix']}",
                f"bad_prefix_scripts={bad_prefix_scripts}",
            ],
            project_prefix == str(config["project"]["prefix"])
            and not bad_prefix_scripts
            and "dynamics" in cluster_config
            and "active_learning" in cluster_config,
            "Read PROJECT_PREFIX before submission and load resources from cluster.yaml.",
            fail_status=INCONSISTENT,
        )
    )
    suite_text = file_text("Libs/exp/submission/submit_publication_claim_suite.sh")
    rows.append(
        row(
            "SLURM-002",
            "Formal experiment launch runs readiness and freeze replay gates before submission.",
            "Publication/formal suite invokes simulation readiness, supplemental contracts, formal readiness, and formal freeze no-write replay before submitting formal jobs.",
            ["Libs/exp/submission/submit_publication_claim_suite.sh"],
            ["Libs/tests/test_formal_experiment_readiness.py"],
            [
                "preflight_script=Libs/scripts/validators/validate_simulation_readiness_audit.py",
                "preflight_script=Libs/scripts/validators/validate_supplemental_milestone_contracts.py",
                "preflight_script=Libs/scripts/validators/validate_formal_experiment_readiness.py",
                "preflight_script=Libs/scripts/validators/validate_formal_experiment_freeze.py --no-write",
            ],
            "validate_simulation_readiness_audit.py" in suite_text
            and "validate_supplemental_milestone_contracts.py" in suite_text
            and "validate_formal_experiment_readiness.py" in suite_text
            and "validate_formal_experiment_freeze.py" in suite_text
            and "--no-write" in suite_text,
            "Wire simulation readiness, supplemental, formal readiness, and freeze replay validators into the real submission path before any sbatch call.",
            fail_status=MISSING,
        )
    )
    return rows


def build_report(config_path: str | Path = "Libs/configs/config_stack.yaml") -> dict[str, object]:
    config = load_config(config_path)
    rows = build_requirement_rows(config)
    blocker_rows = [item for item in rows if item.status != IMPLEMENTED and item.severity == BLOCKER]
    major_rows = [item for item in rows if item.status != IMPLEMENTED and item.severity == MAJOR]
    minor_rows = [item for item in rows if item.status != IMPLEMENTED and item.severity == MINOR]
    sim_paths = required_simulation_paths(config)
    qc_summary = read_json(sim_paths["qc_summary"])
    canonical_counts = load_canonical_metric_values(config_path)
    g5_min_count = int(config["paper"]["publication_readiness"]["g5_foundation_pretraining"]["store_trajectory_min_count"])
    g5_materialized = (
        canonical_counts["raw_materialized_full_microenvironment_trajectory_count"] >= g5_min_count
        and canonical_counts["raw_materialized_mesoscopic_trajectory_count"] >= g5_min_count
    )
    status_counts: dict[str, int] = {}
    for item in rows:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
    return {
        "audit_name": "formal_experiment_readiness_audit",
        "readiness_status": "formal_experiment_ready" if not blocker_rows and not major_rows else "formal_experiment_blocked",
        "formal_experiment_launch_allowed": not blocker_rows and not major_rows,
        "blocker_count": len(blocker_rows),
        "major_count": len(major_rows),
        "minor_count": len(minor_rows),
        "status_counts": status_counts,
        "g2_status": str(qc_summary["readiness_status"]),
        "g2_gate": str(qc_summary["readiness_gate"]),
        "g5_status": "G5_pass" if g5_materialized else "G5_not_passed",
        "g5_store_trajectory_min_count": g5_min_count,
        "current_full_microenvironment_trajectory_count": canonical_counts["raw_materialized_full_microenvironment_trajectory_count"],
        "current_mesoscopic_trajectory_count": canonical_counts["raw_materialized_mesoscopic_trajectory_count"],
        "current_paired_trajectory_count": canonical_counts["paired_full_mesoscopic_count"],
        "current_train_eligible_record_count": canonical_counts["qc_eligible_trajectory_count"],
        "requirements": [asdict(item) for item in rows],
    }


def escape_md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Formal Experiment Readiness Audit",
        "",
        f"- readiness_status: `{report['readiness_status']}`",
        f"- formal_experiment_launch_allowed: `{report['formal_experiment_launch_allowed']}`",
        f"- blocker_count: `{report['blocker_count']}`",
        f"- major_count: `{report['major_count']}`",
        f"- g2_status: `{report['g2_status']}`",
        f"- g5_status: `{report['g5_status']}`",
        f"- current_full_microenvironment_trajectory_count: `{report['current_full_microenvironment_trajectory_count']}`",
        f"- current_mesoscopic_trajectory_count: `{report['current_mesoscopic_trajectory_count']}`",
        f"- current_paired_trajectory_count: `{report['current_paired_trajectory_count']}`",
        f"- current_train_eligible_record_count: `{report['current_train_eligible_record_count']}`",
        "",
        "| requirement_id | status | severity | requirement | artifact_evidence | required_fix |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["requirements"]:
        evidence = "; ".join(str(value) for value in item["artifact_evidence"])
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md(item["requirement_id"]),
                    escape_md(item["status"]),
                    escape_md(item["severity"]),
                    escape_md(item["requirement"]),
                    escape_md(evidence),
                    escape_md(item["required_fix"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args.config)
    json_output = artifact_path(args.json_output)
    md_output = artifact_path(args.md_output)
    write_json(json_output, report)
    write_text_atomic(md_output, markdown_report(report))
    print(json.dumps(report, indent=2, sort_keys=False))
    if not bool(report["formal_experiment_launch_allowed"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
