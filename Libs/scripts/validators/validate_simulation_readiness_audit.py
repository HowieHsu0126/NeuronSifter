from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate simulation QC, leakage, pairing, and readiness audit outputs.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def required_paths(config: dict[str, object]) -> dict[str, Path]:
    audit_config = config["ai_ready_preprocessing"]["simulation_audit"]
    root = resolve_repo_path(str(audit_config["local_root"]))
    return {
        name: root / str(path_value)
        for name, path_value in dict(audit_config["relative_paths"]).items()
    }


def validate_qc_rows(rows: list[dict[str, str]], config: dict[str, object]) -> None:
    require(bool(rows), "simulation_qc_catalog must not be empty.")
    header = list(rows[0].keys())
    require(header == config["schemas"]["simulation_qc_catalog_fields"], f"simulation_qc_catalog header mismatch: {header}")
    required_grades = set(config["ai_ready_preprocessing"]["simulation_audit"]["grading"]["qc_use_by_grade"])
    required_uses = set(config["ai_ready_preprocessing"]["simulation_audit"]["grading"]["sample_weight_by_grade"])
    observed_grades = {row["qc_grade"] for row in rows}
    require(observed_grades <= required_grades, f"Unexpected QC grades: {sorted(observed_grades - required_grades)}")
    for row in rows:
        require(row["trajectory_uid"], f"QC row missing trajectory_uid: {row}")
        require(row["trajectory_family_uid"], f"QC row missing trajectory_family_uid: {row}")
        require(row["content_hash"], f"QC row missing content_hash: {row}")
        require(row["qc_use"] in set(config["ai_ready_preprocessing"]["simulation_audit"]["grading"]["qc_use_by_grade"].values()), f"Unexpected qc_use: {row}")
        if row["qc_use"] == "main_pretraining":
            require(row["qc_grade"] in {"A", "B"}, f"Main pretraining row must be Grade A/B: {row}")
        require(float(row["sample_weight"]) >= 0.0, f"Negative sample weight: {row}")


def validate_leakage(leakage: dict[str, object], contract: dict[str, object]) -> None:
    if bool(contract["require_zero_uid_leakage"]):
        require(int(leakage["trajectory_random_uid_overlap"]) == 0, f"UID leakage detected: {leakage}")
    if bool(contract["require_zero_family_leakage"]):
        require(int(leakage["family_holdout_family_overlap"]) == 0, f"Family leakage detected: {leakage}")
    if bool(contract["require_zero_regimen_leakage"]):
        require(int(leakage["regimen_holdout_regimen_overlap"]) == 0, f"Regimen leakage detected: {leakage}")
    if bool(contract["require_zero_target_program_leakage"]):
        require(int(leakage["target_program_holdout_overlap"]) == 0, f"Target-program leakage detected: {leakage}")
    if bool(contract["require_zero_source_holdout_leakage"]):
        require(int(leakage["source_holdout_role_overlap"]) == 0, f"Source-holdout leakage detected: {leakage}")


def validate_summary(summary: dict[str, object], contract: dict[str, object], config: dict[str, object]) -> None:
    gate = config["ai_ready_preprocessing"]["simulation_audit"]["readiness_gates"][str(contract["required_readiness_gate"])]
    require(str(summary["readiness_gate"]) == str(contract["required_readiness_gate"]), f"Unexpected readiness gate: {summary}")
    require(str(summary["readiness_status"]) in set(contract["required_readiness_statuses"]), f"Readiness status is not accepted: {summary}")
    require(int(summary["full_microenvironment_trajectory_count"]) >= int(gate["full_min_count"]), f"Full trajectory count below readiness gate: {summary}")
    require(int(summary["unique_trajectory_uid_count"]) > 0, f"No unique trajectories reported: {summary}")
    require(int(summary["train_eligible_count"]) > 0, f"No main-pretraining eligible trajectories reported: {summary}")
    require(int(summary["paired_count"]) > 0, f"No paired mesoscopic-full trajectories reported: {summary}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    paths = required_paths(config)
    for name, path in paths.items():
        require(path.exists(), f"Missing simulation audit output {name}: {path}")
    qc_rows = read_tsv(paths["qc_catalog"])
    validate_qc_rows(qc_rows, config)
    contract = config["supplemental_milestone_contracts"]["simulation_readiness_audit"]
    leakage = read_json(paths["split_leakage_audit"])
    validate_leakage(leakage, contract)
    summary = read_json(paths["qc_summary"])
    validate_summary(summary, contract, config)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=paths["qc_summary"],
        readiness_status=summary["readiness_status"],
        qc_row_count=len(qc_rows),
    )


if __name__ == "__main__":
    main()
