"""Validate the Round-11G0 exact-clean execution identity gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json  # noqa: E402
from Libs.utils.exp.round11g_closure import validate_round11g0_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--evidence-root",
        default=str(REPO_ROOT),
        help="Root worktree holding storage, lock, recovery, and commit evidence.",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    report = validate_round11g0_gate(root, Path(args.evidence_root))
    if args.write:
        write_json(root / "Output/audits/dt_round11g0_gate.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "execution_identity_restored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
