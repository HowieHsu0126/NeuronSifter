"""Validate bounded mechanism-basis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Libs.utils.data.mechanism_basis_qualification import (
    CONTRACT_PATH,
    MechanismBasisQualificationError,
    validate_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", default=CONTRACT_PATH)
    parser.add_argument("--stage", choices=("authorization", "ledger", "final"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_stage(
            repository_root=args.repository_root,
            contract_path=args.contract,
            stage=args.stage,
        )
    except MechanismBasisQualificationError as exc:
        print(json.dumps({"status": "MECHANISM_BASIS_VALIDATION_FAILED", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
