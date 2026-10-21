"""Build Round-11F canonical controller and evidence-union governance artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.round11f_controller_lineage import build_round11f_bundle, write_round11f_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--evidence-root",
        default=None,
        help="Optional repository root holding frozen Output/audits inputs.",
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    evidence_root = Path(args.evidence_root).resolve() if args.evidence_root else repo_root
    bundle = build_round11f_bundle(repo_root, evidence_root)
    write_round11f_bundle(repo_root, bundle)
    print(
        json.dumps(
            {
                "status": bundle["validation"]["status"],
                "round11f_gate_status": bundle["controller_authority_map"]["status"],
                "canonical_controller_count": bundle["controller_authority_map"]["canonical_controller_count"],
                "canonical_gate_writer_count": bundle["controller_authority_map"]["canonical_gate_writer_count"],
                "negative_case_rejection_count": bundle["validation"]["negative_case_rejection_count"],
                "round11g_handoff": "Output/audits/dt_round11g_handoff.json",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
