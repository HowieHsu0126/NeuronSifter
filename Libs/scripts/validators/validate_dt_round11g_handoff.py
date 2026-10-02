"""Validate that the frozen Round-11F handoff is usable by Round-11G."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json  # noqa: E402
from Libs.utils.exp.round11g_closure import validate_handoff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--evidence-root",
        default=None,
        help="Optional repository root holding frozen Round-11D/E audit inputs.",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    evidence_root = Path(args.evidence_root).resolve() if args.evidence_root else root
    report = validate_handoff(root, evidence_root)
    if args.write:
        write_json(root / "Output/audits/dt_round11g_handoff_validation.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
