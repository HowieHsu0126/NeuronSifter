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
    parser = argparse.ArgumentParser(description="Validate the foundation world-state registry catalog.")
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
    root = resolve_repo_path(config["data"]["world_state_ready_registry"]["local_root"])
    path = root / config["data"]["world_state_ready_registry"]["relative_paths"]["world_state_catalog"]
    if not path.exists():
        raise FileNotFoundError(f"Missing world-state catalog: {path}")
    validate_header(path, config["schemas"]["world_state_fields"])
    rows = read_tsv(path)
    if not rows:
        raise ValueError("world_state_catalog.tsv must not be empty.")
    semantics_registry = config["measurement_semantics_registry"]
    semantics_path = resolve_repo_path(semantics_registry["local_root"]) / semantics_registry["relative_paths"]["measurement_semantics_catalog"]
    if not semantics_path.exists():
        raise FileNotFoundError(f"Missing measurement semantics catalog: {semantics_path}")
    semantics_by_uid = {row["semantics_uid"]: row for row in read_tsv(semantics_path)}
    seen_uids = {row["world_state_uid"] for row in rows}
    if len(seen_uids) != len(rows):
        raise ValueError("world_state_catalog.tsv contains duplicate world_state_uid values.")
    required_factor_ids = {
        str(row["factor_id"])
        for row in config["foundation_twin"]["world_state_factors"]
    }
    for row in rows:
        if row["semantics_uid"] not in semantics_by_uid:
            raise ValueError(f"world-state row references unknown semantics_uid: {row}")
        if not row["measurement_semantics_label"]:
            raise ValueError(f"world-state row is missing measurement_semantics_label: {row}")
        manifest = json.loads(row["factor_slice_manifest"])
        manifest_factor_ids = set(manifest)
        if manifest_factor_ids != required_factor_ids:
            raise ValueError(f"world-state factor manifest mismatch: {row}")
        factor_targets = json.loads(row["factor_target_payload"])
        if set(factor_targets) != required_factor_ids:
            raise ValueError(f"world-state factor targets mismatch: {row}")
        factor_masks = json.loads(row["factor_supervision_mask_payload"])
        if set(factor_masks) != required_factor_ids:
            raise ValueError(f"world-state factor supervision mask mismatch: {row}")
        if sum(float(value) > 0.0 for value in factor_masks.values()) <= 0:
            raise ValueError(f"world-state row must expose at least one directly supervised factor: {row}")
        factor_sources = json.loads(row["factor_teacher_source_manifest"])
        if set(factor_sources) != required_factor_ids:
            raise ValueError(f"world-state factor teacher-source manifest mismatch: {row}")
        factor_uncertainty = json.loads(row["factor_uncertainty_payload"])
        if set(factor_uncertainty) != required_factor_ids:
            raise ValueError(f"world-state factor uncertainty payload mismatch: {row}")
        factor_evidence = json.loads(row["factor_evidence_payload"])
        if set(factor_evidence) != required_factor_ids:
            raise ValueError(f"world-state factor evidence payload mismatch: {row}")
        unit_context = json.loads(row["unit_context_payload"])
        if set(unit_context) != required_factor_ids:
            raise ValueError(f"world-state unit context payload mismatch: {row}")
        for factor_id, mask_value in factor_masks.items():
            factor_context = unit_context[factor_id]
            if float(mask_value) > 0.0 and factor_context["semantics_uid"] != row["semantics_uid"]:
                raise ValueError(f"Direct world-state factor lacks measurement semantics context: {row}")
            if "canonical_unit" not in factor_context or "proxy_flag" not in factor_context:
                raise ValueError(f"World-state factor lacks unit context fields: {row}")
        posterior_manifest = json.loads(row["posterior_slice_manifest"])
        if set(posterior_manifest) != required_factor_ids:
            raise ValueError(f"world-state posterior slice manifest mismatch: {row}")
        if row["hidden_state_support_flag"] != "yes":
            raise ValueError(f"world-state row must support hidden state inference: {row}")
        if row["cross_scene_compatibility_flag"] != "yes":
            raise ValueError(f"world-state row must declare cross-scene compatibility: {row}")
        if row["rollout_ready_flag"] != "yes":
            raise ValueError(f"world-state row must be rollout ready: {row}")
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=path,
        world_state_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
