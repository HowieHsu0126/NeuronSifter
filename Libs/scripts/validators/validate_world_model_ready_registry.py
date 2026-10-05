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
    parser = argparse.ArgumentParser(description="Validate the public world-model-ready study catalog.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
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
    world_model_root = resolve_repo_path(config["world_model_ready_registry"]["local_root"])
    study_catalog_path = world_model_root / config["world_model_ready_registry"]["relative_paths"]["study_catalog"]
    if not study_catalog_path.exists():
        raise FileNotFoundError(f"World-model-ready study catalog is missing: {study_catalog_path}")
    validate_header(study_catalog_path, config["schemas"]["world_model_ready_study_fields"])

    study_rows = read_tsv(study_catalog_path)
    bundle_rows = read_tsv(resolve_repo_path(config["model_input_registry"]["local_root"]) / config["model_input_registry"]["relative_paths"]["study_bundle_catalog"])
    pipeline_logger.progress(
        "loaded_world_model_inputs",
        study_row_count=len(study_rows),
        bundle_row_count=len(bundle_rows),
    )
    bundle_by_study = {row["study_uid"]: row for row in bundle_rows}
    if {row["study_uid"] for row in study_rows} != set(bundle_by_study):
        raise ValueError("World-model-ready study catalog does not match the model-input bundle set.")

    for row in study_rows:
        bundle_row = bundle_by_study[row["study_uid"]]
        if row["world_model_role"] != bundle_row["world_model_role"]:
            raise ValueError(f"world_model_role mismatch for study {row['study_uid']}")
        if row["subject_linkable_flag"] != bundle_row["subject_linkable_flag"]:
            raise ValueError(f"subject_linkable_flag mismatch for study {row['study_uid']}")
        if row["atlas_prior_ready_flag"] != bundle_row["atlas_prior_ready_flag"]:
            raise ValueError(f"atlas_prior_ready_flag mismatch for study {row['study_uid']}")
        if row["public_only_flag"] != bundle_row["public_only_flag"]:
            raise ValueError(f"public_only_flag mismatch for study {row['study_uid']}")
        if row["model_input_bundle_path"] != bundle_row["bundle_root"]:
            raise ValueError(f"model_input_bundle_path mismatch for study {row['study_uid']}")
        if not row["field_alignment_uid"]:
            raise ValueError(f"field_alignment_uid missing for study {row['study_uid']}")
        if not row["field_alignment_bundle_path"]:
            raise ValueError(f"field_alignment_bundle_path missing for study {row['study_uid']}")
        alignment_path = resolve_repo_path(row["field_alignment_bundle_path"])
        if not alignment_path.exists():
            raise FileNotFoundError(f"Missing field alignment bundle for study {row['study_uid']}: {alignment_path}")
        if not row["structural_prior_uid"]:
            raise ValueError(f"structural_prior_uid missing for study {row['study_uid']}")
        if not row["structural_prior_bundle_path"]:
            raise ValueError(f"structural_prior_bundle_path missing for study {row['study_uid']}")
        structural_prior_path = resolve_repo_path(row["structural_prior_bundle_path"])
        if not structural_prior_path.exists():
            raise FileNotFoundError(f"Missing structural prior bundle for study {row['study_uid']}: {structural_prior_path}")
        if not row["observed_region_set"]:
            raise ValueError(f"observed_region_set missing for study {row['study_uid']}")

    pipeline_logger.progress(
        "world_model_ready_validated",
        artifact=study_catalog_path,
        study_row_count=len(study_rows),
        bundle_row_count=len(bundle_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=study_catalog_path,
        study_row_count=len(study_rows),
        bundle_row_count=len(bundle_rows),
    )

if __name__ == "__main__":
    main()
