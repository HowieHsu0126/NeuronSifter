from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.digital_twin_evidence_fabric import (
    build_evidence_fabric_candidate,
    validate_evidence_fabric_outputs,
    write_evidence_fabric_outputs,
    write_validation_report,
)
from Libs.utils.data.raw_data_registry import load_config_with_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the read-only digital_twin_evidence_fabric_v1 inventory and graph.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, provenance = load_config_with_provenance(args.config)
    owner = "Libs/configs/data/digital_twin_evidence_fabric.yaml"
    if provenance["include_order"].count(owner) != 1:
        raise ValueError(f"DT evidence fabric config must be included exactly once: {owner}")
    inventory, graph, role_rows = build_evidence_fabric_candidate(config)
    write_evidence_fabric_outputs(config, inventory, graph, role_rows)
    report = validate_evidence_fabric_outputs(config)
    write_validation_report(config, report)
    print(
        f"{report['status']} fabric_identity={report['fabric_identity']} "
        f"records={report['record_count']} blockers={report['blocker_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
