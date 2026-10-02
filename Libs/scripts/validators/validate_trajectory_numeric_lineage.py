from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.trajectory_numeric_lineage import (
    build_numeric_lineage_audit,
    validate_numeric_lineage_report,
    write_numeric_lineage_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical trajectory numeric lineage and provenance.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--allow-blocked", action="store_true", help="Write and report unresolved lineage without returning a failing exit code.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_numeric_lineage_audit(args.config)
    artifact_paths = write_numeric_lineage_artifacts(report, args.config)
    blockers = validate_numeric_lineage_report(report, args.config)
    report["validator_blockers"] = blockers
    report["validator_blocker_count"] = len(blockers)
    write_numeric_lineage_artifacts(report, args.config)
    print(
        {
            "status": report["status"],
            "unresolved_conflict_count": report["unresolved_conflict_count"],
            "validator_blocker_count": len(blockers),
            "artifacts": {name: str(path.relative_to(REPO_ROOT)) for name, path in artifact_paths.items()},
        }
    )
    if blockers or report["status"] != "pass":
        if args.allow_blocked:
            return
        raise SystemExit("Trajectory numeric lineage remains fail-closed; inspect Output/audits/trajectory_numeric_lineage_audit.json")


if __name__ == "__main__":
    main()
