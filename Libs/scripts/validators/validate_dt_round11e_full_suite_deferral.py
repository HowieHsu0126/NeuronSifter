"""Validate fail-closed Round-11E full-suite deferral artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.round11e_full_suite_deferral import validate_full_suite_deferral  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Round-11E full-suite deferral without executing pytest.")
    parser.add_argument("--audit-root", default="Output/audits")
    args = parser.parse_args()
    report = validate_full_suite_deferral(REPO_ROOT / args.audit_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
