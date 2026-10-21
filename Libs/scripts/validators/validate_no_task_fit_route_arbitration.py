"""Read-only validation for no-task-fit route arbitration artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_no_task_fit_route_arbitration import (  # noqa: E402
    ARBITRATION_OUTPUT,
    CANDIDATE_OUTPUT,
    LINEAGE_OUTPUT,
    RouteArbitrationError,
    _load_contract,
    _resolve_path,
    canonical_json_bytes,
    compute_arbitration_outputs,
    sha256_bytes,
    validate_output_schema,
)


def _load_output(repository_root: Path, relative: str) -> tuple[dict[str, Any], bytes]:
    path = _resolve_path(
        repository_root,
        relative,
        expected_presence="present",
        label="arbitration output",
    )
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RouteArbitrationError(
            f"invalid UTF-8 JSON output {relative}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RouteArbitrationError(
            f"arbitration output must be a JSON object: {relative}"
        )
    return payload, raw


def validate_route_arbitration(
    *,
    repository_root: Path,
    contract_path: Path | str,
    parent_validator: Callable[..., Mapping[str, Any]] | None = None,
    asset_git_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve(strict=True)
    expected_lineage, expected_arbitration, expected_registry, schemas = (
        compute_arbitration_outputs(
            repository_root=repository_root,
            contract_path=contract_path,
            parent_validator=parent_validator,
            asset_git_provenance=asset_git_provenance,
        )
    )
    contract_file = Path(contract_path)
    if not contract_file.is_absolute():
        contract_file = repository_root / contract_file
    contract, contract_raw = _load_contract(contract_file.resolve(strict=True))

    observed_lineage, lineage_raw = _load_output(repository_root, LINEAGE_OUTPUT)
    observed_arbitration, arbitration_raw = _load_output(
        repository_root, ARBITRATION_OUTPUT
    )
    observed_registry, registry_raw = _load_output(repository_root, CANDIDATE_OUTPUT)

    validate_output_schema(
        observed_lineage, schemas["reusable_asset_lineage"], label="asset lineage"
    )
    validate_output_schema(
        observed_arbitration, schemas["route_arbitration"], label="route arbitration"
    )
    validate_output_schema(
        observed_registry, schemas["candidate_registry"], label="candidate registry"
    )

    expected_payloads = {
        LINEAGE_OUTPUT: expected_lineage,
        ARBITRATION_OUTPUT: expected_arbitration,
        CANDIDATE_OUTPUT: expected_registry,
    }
    observed_payloads = {
        LINEAGE_OUTPUT: observed_lineage,
        ARBITRATION_OUTPUT: observed_arbitration,
        CANDIDATE_OUTPUT: observed_registry,
    }
    for relative, expected in expected_payloads.items():
        observed = observed_payloads[relative]
        if observed != expected or canonical_json_bytes(
            observed
        ) != canonical_json_bytes(expected):
            raise RouteArbitrationError(
                f"recomputed arbitration output drift: {relative}"
            )

    contract_sha256 = sha256_bytes(contract_raw)
    if observed_lineage["contract"]["sha256"] != contract_sha256:
        raise RouteArbitrationError("lineage contract hash drift")
    lineage_sha256 = sha256_bytes(lineage_raw)
    if observed_arbitration["asset_lineage"]["sha256"] != lineage_sha256:
        raise RouteArbitrationError("route arbitration asset-lineage hash drift")
    arbitration_sha256 = sha256_bytes(arbitration_raw)
    if observed_registry["route_arbitration"]["sha256"] != arbitration_sha256:
        raise RouteArbitrationError("candidate registry arbitration hash drift")
    if observed_registry["asset_lineage"]["sha256"] != lineage_sha256:
        raise RouteArbitrationError("candidate registry asset-lineage hash drift")
    if (
        observed_lineage["parent_authority"]["sha256"]
        != contract["parent_authority"]["sha256"]
    ):
        raise RouteArbitrationError("lineage parent-authority hash drift")
    if (
        observed_arbitration["parent_authority"]["sha256"]
        != contract["parent_authority"]["sha256"]
    ):
        raise RouteArbitrationError("arbitration parent-authority hash drift")
    if (
        observed_registry["parent_authority"]["sha256"]
        != contract["parent_authority"]["sha256"]
    ):
        raise RouteArbitrationError("candidate registry parent-authority hash drift")

    dispositions = {
        row["route_id"]: row["disposition"] for row in observed_arbitration["routes"]
    }
    return {
        "schema_version": "neuronsifter-no-task-fit-route-arbitration-validation-v1",
        "status": "passed",
        "asset_count": observed_lineage["asset_count"],
        "present_asset_count": observed_lineage["present_asset_count"],
        "absent_asset_count": observed_lineage["absent_asset_count"],
        "route_dispositions": dispositions,
        "candidate_count": observed_registry["candidate_count"],
        "terminal_status": observed_registry["status"],
        "six_context_outcome_access": observed_registry["access_boundary"][
            "six_context_outcome_access"
        ],
        "protected_holdout_access": observed_registry["access_boundary"][
            "protected_holdout_access"
        ],
        "blocker_count": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = validate_route_arbitration(
            repository_root=args.repository_root,
            contract_path=args.contract,
        )
    except RouteArbitrationError as exc:
        raise SystemExit(
            f"no-task-fit route arbitration validation failed: {exc}"
        ) from exc
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
