#!/usr/bin/env python3
"""Validate the frozen six-publication metadata-only cutoff receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.mechanistic_belief_construct_firewall import (  # noqa: E402
    CONTRACT_PATH,
    RECEIPT_PATH,
    ConstructFirewallError,
    validate_cutoff_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path(CONTRACT_PATH))
    parser.add_argument("--receipt", type=Path, default=Path(RECEIPT_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = validate_cutoff_receipt(
            repository_root=args.repository_root,
            contract_path=args.contract,
            receipt_path=args.receipt,
        )
    except ConstructFirewallError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "record_count": len(receipt["records"]),
                "receipt_sha256": receipt["receipt_sha256"],
                "strictly_before_cutoff": receipt["derivation"]["strictly_before_cutoff"],
                **receipt["access_counters"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
