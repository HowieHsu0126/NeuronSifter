from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the simulation-teacher registry catalog.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    root = resolve_repo_path(config["data"]["simulation_teacher_ready_registry"]["local_root"])
    path = root / config["data"]["simulation_teacher_ready_registry"]["relative_paths"]["simulation_teacher_catalog"]
    if not path.exists():
        raise FileNotFoundError(f"Missing simulation-teacher catalog: {path}")
    validate_header(path, config["schemas"]["simulation_teacher_fields"])
    rows = read_tsv(path)
    if not rows:
        raise ValueError("simulation_teacher_catalog.tsv must not be empty.")
    for row in rows:
        if row["supervision_role"] != "solver_teacher_trajectory":
            raise ValueError(f"Simulation teacher row must be solver grounded: {row}")
        for field_name in ["teacher_role", "source_model_id", "provenance_level", "calibration_role", "observed_or_synthetic"]:
            if not row[field_name]:
                raise ValueError(f"Simulation teacher row is missing {field_name}: {row}")
        if row["calibration_role"] not in {"compatibility_baseline", "mechanism_teacher", "gold_calibration"}:
            raise ValueError(f"Unexpected simulation teacher calibration_role: {row}")
        if row["teacher_evidence_level"] not in {"empirical_measured", "validated_mechanistic", "repository_generated", "configuration_generated_provisional", "legacy_compatibility"}:
            raise ValueError(f"Unexpected simulation teacher evidence level: {row}")
        if row["calibration_status"] not in {"external_calibrated", "repository_protocol_validated", "not_external_calibrated", "calibration_pending"}:
            raise ValueError(f"Unexpected simulation teacher calibration status: {row}")
        if not row["uncertainty_policy"]:
            raise ValueError(f"Simulation teacher row is missing uncertainty_policy: {row}")
        if float(row["teacher_loss_weight"]) < 0.0:
            raise ValueError(f"Simulation teacher loss weight must be non-negative: {row}")
        if row["main_pretraining_eligibility"] not in {"main_pretraining", "calibration_or_compatibility_only", "reject"}:
            raise ValueError(f"Unexpected simulation teacher main pretraining eligibility: {row}")
        if not row["trajectory_store_path"]:
            raise ValueError(f"Simulation teacher row is missing trajectory_store_path: {row}")
        if "conservation" not in row["consistency_targets"]:
            raise ValueError(f"Simulation teacher row is missing conservation target: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        simulation_teacher_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
