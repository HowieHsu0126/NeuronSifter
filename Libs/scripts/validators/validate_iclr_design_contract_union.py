from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.iclr_contract_union import build_union_consistency_report, load_union_context, resolve_repo_path
from Libs.utils.data.raw_data_registry import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the canonical ICLR-v2 design contract union.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, provenance = load_union_context(args.config)
    report = build_union_consistency_report(config, provenance, args.config)
    report["producer_command"] = f"python Libs/scripts/validators/validate_iclr_design_contract_union.py --config {args.config}"
    output_path = resolve_repo_path(config["iclr_design_contract_union"]["audit_contract"]["output_paths"]["contract_union"])
    write_json(output_path, report)
    print(f"iclr_contract_union_status={report['status']}")
    print(f"iclr_contract_union_blocker_count={report['blocker_count']}")
    print(f"iclr_contract_union_document_count={report['document_count']}")
    print(f"iclr_contract_union_output={output_path.relative_to(REPO_ROOT).as_posix()}")
    if report["blocker_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
