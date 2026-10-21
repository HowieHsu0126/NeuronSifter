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
    parser = argparse.ArgumentParser(description="Validate the canonical sample-observation contract catalog.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    root = resolve_repo_path(config["data"]["sample_observation_registry"]["local_root"])
    path = root / config["data"]["sample_observation_registry"]["relative_paths"]["sample_observation_catalog"]
    if not path.exists():
        raise FileNotFoundError(f"Missing sample-observation catalog: {path}")
    validate_header(path, config["schemas"]["sample_observation_fields"])
    rows = read_tsv(path)
    if not rows:
        raise ValueError("sample_observation_catalog.tsv must not be empty.")
    semantics_registry = config["measurement_semantics_registry"]
    semantics_path = resolve_repo_path(semantics_registry["local_root"]) / semantics_registry["relative_paths"]["measurement_semantics_catalog"]
    if not semantics_path.exists():
        raise FileNotFoundError(f"Missing measurement semantics catalog: {semantics_path}")
    semantics_rows = read_tsv(semantics_path)
    semantics_by_uid = {row["semantics_uid"]: row for row in semantics_rows}
    observation_uids = {row["observation_uid"] for row in rows}
    if len(observation_uids) != len(rows):
        raise ValueError("sample_observation_catalog.tsv contains duplicate observation_uid values.")
    for row in rows:
        if row["public_only_flag"] != "yes":
            raise ValueError(f"Sample observation violates public-only contract: {row}")
        if row["compatibility_alias_flag"] != "no":
            raise ValueError(f"Canonical sample observation cannot be a compatibility alias: {row}")
        if row["semantics_uid"] not in semantics_by_uid:
            raise ValueError(f"Sample observation references unknown semantics_uid: {row}")
        if not row["measurement_semantics_label"]:
            raise ValueError(f"Sample observation is missing measurement_semantics_label: {row}")
        if row["measurement_semantics_label"] == str(semantics_registry["label_policy"]["fallback_label"]):
            raise ValueError(f"Sample observation fell back to prohibited unknown measurement label: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        sample_observation_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
