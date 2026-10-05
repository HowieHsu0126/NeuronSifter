from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.claim_gated_wording import build_claim_wording_audit, write_claim_wording_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate gate-qualified public claim wording.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, rows = build_claim_wording_audit(args.config)
    artifacts = write_claim_wording_artifacts(report, rows, args.config)
    print(
        {
            "status": report["status"],
            "violation_count": report["violation_count"],
            "artifacts": {name: str(path.relative_to(REPO_ROOT)) for name, path in artifacts.items()},
        }
    )
    if report["status"] != "pass":
        raise SystemExit("Claim-gated wording violations require remediation.")


if __name__ == "__main__":
    main()
