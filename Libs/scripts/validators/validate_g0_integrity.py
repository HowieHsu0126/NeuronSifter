from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.g0_integrity import build_g0_integrity_audit, write_g0_integrity_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the immutable Phase 0 G0 integrity re-freeze audit.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Materialize a fail-closed G0 report without treating an immutable remaining blocker as command failure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_g0_integrity_audit(args.config)
    output_path = write_g0_integrity_audit(report, args.config)
    print(
        {
            "status": report["status"],
            "original_blocker_count": report["original_blocker_count"],
            "remaining_original_blocker_count": report["remaining_original_blocker_count"],
            "remaining_original_blocker_ids": report["remaining_original_blocker_ids"],
            "output": output_path.relative_to(REPO_ROOT).as_posix(),
        }
    )
    if report["status"] != "passed" and not args.allow_blocked:
        raise SystemExit("G0 integrity remains fail-closed; inspect Output/audits/iclr_phase0_g0_integrity_audit.json")


if __name__ == "__main__":
    main()
