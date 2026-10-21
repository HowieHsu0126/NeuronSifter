#!/usr/bin/env python3
"""Build the metadata-only mechanistic-belief construct authorities."""

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
    build_construct_authority,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path(CONTRACT_PATH))
    parser.add_argument("--metadata-source", choices=("official",), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt, authority, created = build_construct_authority(
            repository_root=args.repository_root,
            contract_path=args.contract,
            metadata_source=args.metadata_source,
        )
    except ConstructFirewallError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "created": created,
                "receipt": RECEIPT_PATH,
                "receipt_sha256": receipt["receipt_sha256"],
                "authority": AUTHORITY_PATH,
                "authority_sha256": authority["authority_sha256"],
                "record_count": len(receipt["records"]),
                "strictly_before_cutoff": receipt["derivation"]["strictly_before_cutoff"],
                "source_content_access": receipt["access_counters"]["source_content_access"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
