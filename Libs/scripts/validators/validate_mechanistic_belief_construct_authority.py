#!/usr/bin/env python3
"""Validate the frozen mechanistic-belief construct and firewall authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.mechanistic_belief_construct_firewall import (  # noqa: E402
    AUTHORITY_PATH,
    CONTRACT_PATH,
    RECEIPT_PATH,
    ConstructFirewallError,
    validate_construct_authority,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path(CONTRACT_PATH))
    parser.add_argument("--receipt", type=Path, default=Path(RECEIPT_PATH))
    parser.add_argument("--authority", type=Path, default=Path(AUTHORITY_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        authority = validate_construct_authority(
            repository_root=args.repository_root,
            contract_path=args.contract,
            receipt_path=args.receipt,
            authority_path=args.authority,
        )
    except ConstructFirewallError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "authority_sha256": authority["authority_sha256"],
                "method_uid": authority["method_uid"],
                "historical_authority_count": len(authority["historical_authorities"]),
                "strictly_before_cutoff": authority["cutoff_receipt"]["strictly_before_cutoff"],
                **authority["construct"]["access_policy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
