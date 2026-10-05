from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the first-class measurement semantics registry.")
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
    registry = config["measurement_semantics_registry"]
    path = resolve_repo_path(registry["local_root"]) / registry["relative_paths"]["measurement_semantics_catalog"]
    if not path.exists():
        raise FileNotFoundError(f"Missing measurement semantics catalog: {path}")
    validate_header(path, config["schemas"]["measurement_semantics_fields"])
    rows = read_tsv(path)
    if not rows:
        raise ValueError("measurement_semantics_catalog.tsv must not be empty.")
    semantics_uids = {row["semantics_uid"] for row in rows}
    if len(semantics_uids) != len(rows):
        raise ValueError("measurement_semantics_catalog.tsv contains duplicate semantics_uid values.")
    allowed_semantics = set(config["measurement_semantics_registry"]["matrix_semantics_definitions"])
    for row in rows:
        if row["matrix_semantics"] not in allowed_semantics:
            raise ValueError(f"Measurement semantics row uses undeclared matrix_semantics: {row}")
        if row["measurement_semantics_label"] == str(registry["label_policy"]["fallback_label"]):
            raise ValueError(f"Measurement semantics row fell back to prohibited unknown label: {row}")
        if row["public_only_flag"] != "yes":
            raise ValueError(f"Measurement semantics row violates public-only contract: {row}")
        for semantics_field in ["unit_conversion_chain", "standardization_summary", "uncertainty_semantics"]:
            if not row[semantics_field]:
                raise ValueError(f"Measurement semantics row is missing {semantics_field}: {row}")
        json.loads(row["standardization_summary"])
        for path_field in ["raw_matrix_path", "norm_matrix_path", "sample_index_path", "feature_index_path", "scaler_artifact_path", "standardization_statistics_path"]:
            if not row[path_field]:
                raise ValueError(f"Measurement semantics row is missing {path_field}: {row}")
        if row["standardization_statistics_path"] != row["scaler_artifact_path"]:
            raise ValueError(f"Measurement semantics row has inconsistent scaler/statistics paths: {row}")
        if bool(registry["scaler_artifact_policy"]["inverse_transform_required"]) and not row["inverse_transform_artifact_path"]:
            raise ValueError(f"Measurement semantics row lacks inverse-transform artifact: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        measurement_semantics_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
