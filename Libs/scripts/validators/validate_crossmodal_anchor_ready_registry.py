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
    parser = argparse.ArgumentParser(description="Validate the foundation cross-modal anchor registry catalog.")
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
    root = resolve_repo_path(config["data"]["crossmodal_anchor_ready_registry"]["local_root"])
    path = root / config["data"]["crossmodal_anchor_ready_registry"]["relative_paths"]["crossmodal_anchor_catalog"]
    if not path.exists():
        raise FileNotFoundError(f"Missing cross-modal anchor catalog: {path}")
    validate_header(path, config["schemas"]["crossmodal_anchor_fields"])
    rows = read_tsv(path)
    if not rows:
        raise ValueError("crossmodal_anchor_catalog.tsv must not be empty.")
    supported_families = set(config["foundation_twin"]["disease_families"])
    supported_sources = {
        str(row["source_id"])
        for row in config["foundation_twin"]["crossmodal_anchor_sources"]
    }
    for row in rows:
        if row["disease_family"] not in supported_families:
            raise ValueError(f"Unknown disease family in crossmodal anchor row: {row}")
        if row["anchor_source"] not in supported_sources:
            raise ValueError(f"Unknown anchor source in crossmodal anchor row: {row}")
        if not row["source_sample_or_cell_id"]:
            raise ValueError(f"Crossmodal anchor row is missing source_sample_or_cell_id: {row}")
        if not row["available_modality_set"]:
            raise ValueError(f"Crossmodal anchor row is missing available_modality_set: {row}")
        if row["anchor_type"] not in {"paired_sample_cell", "weak_teacher_source"}:
            raise ValueError(f"Unexpected crossmodal anchor_type: {row}")
        if row["anchor_type"] == "paired_sample_cell" and "patchseq" not in row["anchor_source"]:
            raise ValueError(f"Only Patch-seq may be declared paired_sample_cell in the public anchor contract: {row}")
        if row["anchor_type"] != "paired_sample_cell" and row["alignment_target"] == "supervised_contrastive_retrieval":
            raise ValueError(f"Weak crossmodal anchors cannot claim paired retrieval supervision: {row}")
        if not row["metric_artifact_path"]:
            raise ValueError(f"Crossmodal anchor row is missing metric_artifact_path: {row}")
        if not row["transcriptomic_anchor_id"] or not row["morphology_anchor_id"] or not row["electrophysiology_anchor_id"]:
            raise ValueError(f"Crossmodal anchor row is missing required anchor IDs: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        crossmodal_anchor_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
