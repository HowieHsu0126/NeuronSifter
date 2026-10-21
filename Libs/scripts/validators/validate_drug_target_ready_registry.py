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
    parser = argparse.ArgumentParser(description="Validate the drug-target-ready registry outputs.")
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
    root = resolve_repo_path(config["screening_pipeline"]["drug_target_ready_registry"]["local_root"])
    relative_paths = config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]
    path_map = {
        "target_catalog": root / relative_paths["target_catalog"],
        "compound_catalog": root / relative_paths["compound_catalog"],
        "drug_target_edge_catalog": root / relative_paths["drug_target_edge_catalog"],
        "target_evidence_catalog": root / relative_paths["target_evidence_catalog"],
        "compound_feature_catalog": root / relative_paths["compound_feature_catalog"],
        "drug_target_exclusion_catalog": root / relative_paths["drug_target_exclusion_catalog"],
    }
    for path in path_map.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing drug-target-ready output: {path}")

    validate_header(path_map["target_catalog"], config["schemas"]["drug_target_target_fields"])
    validate_header(path_map["compound_catalog"], config["schemas"]["drug_target_compound_fields"])
    validate_header(path_map["drug_target_edge_catalog"], config["schemas"]["drug_target_edge_fields"])
    validate_header(path_map["target_evidence_catalog"], config["schemas"]["drug_target_evidence_fields"])
    validate_header(path_map["compound_feature_catalog"], config["schemas"]["drug_target_compound_feature_fields"])
    validate_header(path_map["drug_target_exclusion_catalog"], config["schemas"]["drug_target_exclusion_fields"])

    target_rows = read_tsv(path_map["target_catalog"])
    compound_rows = read_tsv(path_map["compound_catalog"])
    edge_rows = read_tsv(path_map["drug_target_edge_catalog"])
    evidence_rows = read_tsv(path_map["target_evidence_catalog"])
    feature_rows = read_tsv(path_map["compound_feature_catalog"])
    exclusion_rows = read_tsv(path_map["drug_target_exclusion_catalog"])
    pipeline_logger.progress(
        "loaded_drug_target_inputs",
        target_row_count=len(target_rows),
        compound_row_count=len(compound_rows),
        edge_row_count=len(edge_rows),
        evidence_row_count=len(evidence_rows),
        feature_row_count=len(feature_rows),
        exclusion_row_count=len(exclusion_rows),
    )
    target_uids = {row["target_uid"] for row in target_rows}
    compound_uids = {row["compound_uid"] for row in compound_rows}
    if not target_rows:
        raise ValueError("target_catalog.tsv must not be empty.")
    if not compound_rows:
        raise ValueError("compound_catalog.tsv must not be empty.")
    if not edge_rows:
        raise ValueError("drug_target_edge_catalog.tsv must not be empty.")
    if not evidence_rows:
        raise ValueError("target_evidence_catalog.tsv must not be empty.")

    for row in target_rows:
        if not row["canonical_entity_id"] or not row["canonical_entity_label"]:
            raise ValueError(f"Invalid target row without canonical mapping: {row}")

    for row in compound_rows:
        if not row["canonical_compound_id"] or not row["canonical_id_namespace"]:
            raise ValueError(f"Invalid compound row without canonical identity: {row}")

    for row in edge_rows:
        if row["target_uid"] not in target_uids:
            raise ValueError(f"Edge references unknown target_uid: {row['edge_uid']}")
        if row["compound_uid"] not in compound_uids:
            raise ValueError(f"Edge references unknown compound_uid: {row['edge_uid']}")
        if float(row["evidence_score"]) < 0:
            raise ValueError(f"Negative evidence_score in edge row: {row['edge_uid']}")

    for row in evidence_rows:
        if row["target_uid"] not in target_uids:
            raise ValueError(f"Evidence references unknown target_uid: {row['evidence_uid']}")
        if float(row["evidence_score"]) < 0:
            raise ValueError(f"Negative evidence_score in evidence row: {row['evidence_uid']}")

    pipeline_logger.progress(
        "drug_target_ready_validated",
        artifact=path_map["drug_target_edge_catalog"],
        target_row_count=len(target_rows),
        compound_row_count=len(compound_rows),
        edge_row_count=len(edge_rows),
        evidence_row_count=len(evidence_rows),
        feature_row_count=len(feature_rows),
        exclusion_row_count=len(exclusion_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path_map["drug_target_edge_catalog"],
        target_row_count=len(target_rows),
        compound_row_count=len(compound_rows),
        edge_row_count=len(edge_rows),
        evidence_row_count=len(evidence_rows),
        feature_row_count=len(feature_rows),
        exclusion_row_count=len(exclusion_rows),
    )

if __name__ == "__main__":
    main()
