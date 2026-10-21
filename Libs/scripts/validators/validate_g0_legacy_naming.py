from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.g0_naming import (
    build_naming_remediation_report,
    report_json,
    write_baseline_inventory,
)
from Libs.utils.data.raw_data_registry import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical NeuronSifter naming and approved legacy migration literals.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--mode", choices=["baseline", "validate"], default="validate")
    parser.add_argument("--baseline-revision")
    parser.add_argument("--inventory-output", default="Output/audits/g0_legacy_naming_inventory.tsv")
    parser.add_argument("--residual-output", default="Output/audits/naming_residual_audit.tsv")
    parser.add_argument("--report-json", default="Output/audits/g0_naming_remediation_report.json")
    parser.add_argument("--report-markdown", default="Output/audits/g0_naming_remediation_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    inventory_path = (REPO_ROOT / args.inventory_output).resolve()
    config = load_config(config_path)
    if args.mode == "baseline":
        configured_revision = str(config["iclr_design_contract_union"]["g0_naming_audit"]["baseline_revision"])
        summary = write_baseline_inventory(config, inventory_path, str(args.baseline_revision or configured_revision))
        print(report_json(summary))
        return
    report = build_naming_remediation_report(
        config,
        config_path,
        inventory_path,
        inventory_path,
        (REPO_ROOT / args.residual_output).resolve(),
        (REPO_ROOT / args.report_json).resolve(),
        (REPO_ROOT / args.report_markdown).resolve(),
    )
    print(report_json(report))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
