"""Perform the independent Phase 1 public-corpus storage measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_storage import scan_storage_roots, safe_repo_path
from Libs.utils.data.raw_data_registry import load_config, write_json
from Libs.scripts.validators.validate_public_foundation_corpus_storage_ledger import (
    validate_ledger_payload,
)


DEFAULT_CONFIG = "Libs/configs/config_stack.yaml"
DEFAULT_OUTPUT = "Output/audits/public_foundation_corpus_storage_ledger.json"


def build(
    repo_root: Path = REPO_ROOT,
    *,
    config_path: str = DEFAULT_CONFIG,
) -> dict[str, object]:
    config = load_config(safe_repo_path(repo_root, config_path))
    policy = config["public_foundation_corpus"]["storage_policy"]
    ledger = scan_storage_roots(
        repo_root,
        roots=policy["scan_root_paths"],
        cap_bytes=int(policy["decimal_byte_cap"]),
        exclude_relative_paths=policy["scan_excluded_paths"],
        previous_confirmed_usage_bytes=int(policy["previous_confirmed_usage_bytes"]),
        largest_directory_limit=int(policy["largest_directory_limit"]),
    )
    schema_path = safe_repo_path(repo_root, str(policy["storage_ledger_schema_path"]))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = validate_ledger_payload(ledger, storage_policy=policy, schema=schema)
    ledger["validator_status"] = "passed" if report["status"] == "passed" else "blocked"
    return ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the Public Foundation Corpus storage footprint.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = build(REPO_ROOT, config_path=args.config)
    write_json(safe_repo_path(REPO_ROOT, args.output), ledger)
    print(
        json.dumps(
            {
                "scan_uid": ledger["scan_uid"],
                "scan_status": ledger["scan_status"],
                "usage_bytes": ledger["usage_bytes"],
                "capacity_limit_bytes": ledger["capacity_limit_bytes"],
                "partial_scan": ledger["partial_scan"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
