from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit startup collection completion against manual-access registry rows.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    processed_root = resolve_repo_path(config["processed_registry"]["local_root"])
    manual_access_path = processed_root / config["processed_registry"]["relative_paths"]["manual_access_catalog"]
    registry_summary_path = processed_root / config["processed_registry"]["relative_paths"]["registry_summary"]
    manual_access_rows = read_tsv(manual_access_path)
    registry_summary_rows = read_tsv(registry_summary_path)
    incomplete_prefixes = tuple(config["collection_completion_audit"]["incomplete_status_prefixes"])
    manual_required_source_keys = set(config["collection_completion_audit"]["manual_required_source_keys"])
    ignored_source_keys = set(config["collection_completion_audit"].get("ignored_source_keys", []))
    reprobe_prefix = str(config["collection_completion_audit"]["reprobe_asset_prefix"])

    unresolved_rows = [
        row
        for row in manual_access_rows
        if row["status"].startswith(incomplete_prefixes) and row["source_key"] not in ignored_source_keys
    ]
    blocking_rows = [row for row in unresolved_rows if row["manual_action_required_flag"] == "yes"]
    prepared_rows = [row for row in manual_access_rows if row["resolution_status"] == "prepared"]
    manual_pending_rows = [
        row
        for row in blocking_rows
        if row["source_key"] in manual_required_source_keys
    ]
    automatable_blocking_rows = [
        row
        for row in blocking_rows
        if row["source_key"] not in manual_required_source_keys
    ]
    automatable_complete_flag = len(automatable_blocking_rows) == 0
    manual_pending_flag = len(manual_pending_rows) > 0
    collection_complete_flag = automatable_complete_flag and not manual_pending_flag
    if collection_complete_flag:
        collection_status = str(config["collection_completion_audit"]["collection_complete_status"])
    elif automatable_complete_flag:
        collection_status = str(config["collection_completion_audit"]["manual_pending_status"])
    else:
        collection_status = str(config["collection_completion_audit"]["automatable_incomplete_status"])
    reprobed_sources = sorted(
        {
            row["source_name"]
            for row in manual_access_rows
            if row["asset_name"].startswith(reprobe_prefix) or row["last_reprobe_date"]
        }
    )
    nemo_summary_row = next((row for row in registry_summary_rows if row["source_key"] == "nemo_archive"), None)
    payload = {
        "collection_status": collection_status,
        "collection_complete_flag": collection_complete_flag,
        "automatable_complete_flag": automatable_complete_flag,
        "manual_pending_flag": manual_pending_flag,
        "blocking_row_count": len(blocking_rows),
        "prepared_row_count": len(prepared_rows),
        "blocking_sources": sorted({row["source_name"] for row in blocking_rows}),
        "blocking_accessions": sorted({row["source_accession"] for row in blocking_rows if row["source_accession"]}),
        "automatable_blocking_sources": sorted({row["source_name"] for row in automatable_blocking_rows}),
        "manual_pending_sources": sorted({row["source_name"] for row in manual_pending_rows}),
        "manual_pending_asset_count": len(manual_pending_rows),
        "reprobed_sources": reprobed_sources,
        "nemo_public_discovery_complete_flag": nemo_summary_row["nemo_public_discovery_complete_flag"] if nemo_summary_row else "no",
        "nemo_primary_open_candidate_count": int(nemo_summary_row["nemo_primary_open_candidate_count"]) if nemo_summary_row else 0,
        "nemo_primary_materialized_asset_count": int(nemo_summary_row["nemo_primary_materialized_asset_count"]) if nemo_summary_row else 0,
        "nemo_reference_materialized_asset_count": int(nemo_summary_row["nemo_reference_materialized_asset_count"]) if nemo_summary_row else 0,
        "audit_source": str(manual_access_path.relative_to(REPO_ROOT)),
        "blocking_rows": blocking_rows,
    }
    output_path = resolve_repo_path(config["collection_completion_audit"]["output_path"])
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
