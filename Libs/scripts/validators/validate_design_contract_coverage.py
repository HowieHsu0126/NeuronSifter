from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.iclr_contract_union import build_design_coverage_report, resolve_repo_path
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exact ICLR design-document contract coverage.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def build_report(config_path: Path | str) -> dict[str, object]:
    return build_design_coverage_report(load_config(config_path))


def _summary_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for document in report["documents"]:
        row = dict(document)
        rows.append(
            {
                "path": row.get("path", ""),
                "document_status": row.get("document_status", ""),
                "source_status": row.get("source_status", ""),
                "precedence_id": row.get("precedence_id", ""),
                "canonical_contract_ids": "|".join(str(value) for value in row.get("canonical_contract_ids", [])),
                "owning_milestone_id": row.get("owning_milestone_id", ""),
                "owning_direction_id": row.get("owning_direction_id", ""),
                "downstream_consumer_ids": "|".join(str(value) for value in row.get("downstream_consumer_ids", [])),
                "responsible_validator_ids": "|".join(str(value) for value in row.get("responsible_validator_ids", [])),
                "mandatory_requirement_ids": "|".join(str(value) for value in row.get("mandatory_requirement_ids", [])),
                "conflict_ids": "|".join(str(dict(value).get("conflict_id", "")) for value in row.get("conflicts", [])),
                "supersedes": "|".join(str(value) for value in row.get("supersedes", [])),
                "coverage_status": row.get("coverage_status", ""),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = build_design_coverage_report(config)
    report["producer_command"] = f"python Libs/scripts/validators/validate_design_contract_coverage.py --config {args.config}"
    coverage_config = config["validation"]["design_contract_coverage"]
    output_json = resolve_repo_path(coverage_config["output_json"])
    output_tsv = resolve_repo_path(coverage_config["output_summary_tsv"])
    write_json(output_json, report)
    fieldnames = [
        "path",
        "document_status",
        "source_status",
        "precedence_id",
        "canonical_contract_ids",
        "owning_milestone_id",
        "owning_direction_id",
        "downstream_consumer_ids",
        "responsible_validator_ids",
        "mandatory_requirement_ids",
        "conflict_ids",
        "supersedes",
        "coverage_status",
    ]
    write_tsv(output_tsv, fieldnames, _summary_rows(report))
    print(f"design_contract_coverage_status={report['status']}")
    print(f"design_contract_discovered_count={report['discovered_document_count']}")
    print(f"design_contract_declared_count={report['declared_document_count']}")
    print(f"design_contract_blocker_count={report['blocker_count']}")
    print(f"design_contract_output={output_json.relative_to(REPO_ROOT).as_posix()}")
    print(f"design_contract_summary={output_tsv.relative_to(REPO_ROOT).as_posix()}")
    if report["blocker_count"]:
        print(json.dumps(report["blockers"], sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
