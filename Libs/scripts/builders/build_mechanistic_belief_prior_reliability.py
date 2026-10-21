#!/usr/bin/env python3
"""Build frozen Mechanistic Belief Twin prior and reliability authorities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.model.mechanistic_belief_prior_reliability import (  # noqa: E402
    AUTHORITY_PATH,
    CONTRACT_PATH,
    DECISION_TABLE_PATH,
    PriorReliabilityError,
    build_prior_reliability_authority,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path(CONTRACT_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        authority, decision_table, created = build_prior_reliability_authority(
            repository_root=args.repository_root,
            contract_path=args.contract,
        )
    except PriorReliabilityError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "created": created,
                "authority": AUTHORITY_PATH,
                "authority_sha256": authority["authority_sha256"],
                "decision_table": DECISION_TABLE_PATH,
                "decision_table_sha256": decision_table["decision_table_sha256"],
                "terminal_count": len(decision_table["decisions"]),
                "terminal_issued_in_this_round": decision_table[
                    "terminal_issued_in_this_round"
                ],
                **authority["access_counters"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
