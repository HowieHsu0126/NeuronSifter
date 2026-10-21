"""Validate the frozen no-task-fit parent-authority binding without writes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_no_task_fit_parent_authority import (  # noqa: E402
    AGGREGATE_ACCESS,
    ARTIFACT_COUNT,
    AUTHORIZATION_CLOSURE,
    CONTRACT_PATH,
    EXPECTED_ARTIFACT_IDENTITIES,
    EXPECTED_SOURCE_TRACKING_STATE,
    LIMITATIONS,
    MANIFEST_PATH,
    NEXT_ACTION,
    SCHEMA_PATH,
    TERMINAL_DECISIONS,
    AuthorityBindingError,
    _aggregate_access,
    _load_and_check_schema,
    _load_json_bytes,
    _load_yaml,
    _resolve_config_path,
    _resolve_repository_path,
    sha256_bytes,
    validate_artifact_payload,
    validate_contract,
    validate_manifest_payload,
)


def _require_equal(observed: object, expected: object, *, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise AuthorityBindingError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def validate_bound_parent_authority(
    *,
    repository_root: Path,
    contract_path: Path | str,
    manifest_path: Path | str,
) -> dict[str, Any]:
    repository_root = repository_root.resolve(strict=True)
    resolved_contract, contract_relative = _resolve_config_path(
        repository_root,
        contract_path,
        label="authority contract",
    )
    if contract_relative != CONTRACT_PATH:
        raise AuthorityBindingError(f"authority contract must be {CONTRACT_PATH}")
    contract_raw = resolved_contract.read_bytes()
    contract = _load_yaml(resolved_contract)
    records = validate_contract(contract)

    resolved_schema = _resolve_repository_path(
        repository_root,
        contract["manifest"]["schema_path"],
        must_exist=True,
        label="manifest schema",
    )
    schema, schema_sha256 = _load_and_check_schema(resolved_schema)

    resolved_manifest, manifest_relative = _resolve_config_path(
        repository_root,
        manifest_path,
        label="aggregate manifest",
    )
    if manifest_relative != MANIFEST_PATH:
        raise AuthorityBindingError(f"aggregate manifest must be {MANIFEST_PATH}")
    manifest = _load_json_bytes(
        resolved_manifest.read_bytes(), label="aggregate manifest"
    )
    validate_manifest_payload(manifest, schema)

    _require_equal(
        manifest["contract"]["path"], contract_relative, label="manifest contract path"
    )
    _require_equal(
        manifest["contract"]["sha256"],
        sha256_bytes(contract_raw),
        label="manifest contract SHA-256",
    )
    _require_equal(
        manifest["contract"]["schema_path"], SCHEMA_PATH, label="manifest schema path"
    )
    _require_equal(
        manifest["contract"]["schema_sha256"],
        schema_sha256,
        label="manifest schema SHA-256",
    )
    _require_equal(
        manifest["contract"]["artifact_count"],
        ARTIFACT_COUNT,
        label="contract artifact count",
    )
    _require_equal(
        manifest["terminal_decisions"], TERMINAL_DECISIONS, label="terminal decisions"
    )
    _require_equal(
        manifest["authorization_closure"],
        AUTHORIZATION_CLOSURE,
        label="authorization closure",
    )
    _require_equal(
        manifest["limitations"], LIMITATIONS, label="interpretation limitations"
    )
    _require_equal(manifest["next_action"], NEXT_ACTION, label="next action")

    manifest_records = manifest["artifacts"]
    if (
        not isinstance(manifest_records, list)
        or len(manifest_records) != ARTIFACT_COUNT
    ):
        raise AuthorityBindingError(
            "aggregate manifest must contain exactly nine artifact records"
        )
    manifest_by_path: dict[str, Mapping[str, Any]] = {}
    seen_uids: set[str] = set()
    for manifest_record in manifest_records:
        if not isinstance(manifest_record, dict):
            raise AuthorityBindingError("aggregate artifact record must be an object")
        path = manifest_record["destination_path"]
        uid = manifest_record["authority_uid"]
        if path in manifest_by_path:
            raise AuthorityBindingError(f"duplicate aggregate artifact path: {path}")
        if uid in seen_uids:
            raise AuthorityBindingError(f"duplicate aggregate authority UID: {uid}")
        manifest_by_path[path] = manifest_record
        seen_uids.add(uid)

    payloads: list[dict[str, Any]] = []
    for contract_record in records:
        relative = contract_record["destination_path"]
        if relative not in manifest_by_path:
            raise AuthorityBindingError(
                f"aggregate manifest omits allowlisted artifact: {relative}"
            )
        manifest_record = manifest_by_path[relative]
        expected_category, expected_schema, expected_status = (
            EXPECTED_ARTIFACT_IDENTITIES[relative]
        )
        expected_record_values = {
            "authority_uid": contract_record["authority_uid"],
            "category": expected_category,
            "source_path": contract_record["source_path"],
            "destination_path": relative,
            "schema_version": expected_schema,
            "status": expected_status,
            "expected_sha256": contract_record["sha256"],
            "expected_bytes": contract_record["bytes"],
            "source_tracking_state": EXPECTED_SOURCE_TRACKING_STATE,
            "source_matches_contract": True,
            "destination_matches_source": True,
        }
        for field, expected in expected_record_values.items():
            _require_equal(
                manifest_record[field], expected, label=f"{relative}.{field}"
            )

        destination = _resolve_repository_path(
            repository_root,
            relative,
            must_exist=True,
            label="bound destination artifact",
        )
        raw = destination.read_bytes()
        observed_sha256 = sha256_bytes(raw)
        _require_equal(
            observed_sha256, contract_record["sha256"], label=f"{relative} SHA-256"
        )
        _require_equal(
            len(raw), contract_record["bytes"], label=f"{relative} byte count"
        )
        _require_equal(
            manifest_record["source_sha256"],
            observed_sha256,
            label=f"{relative}.source_sha256",
        )
        _require_equal(
            manifest_record["destination_sha256"],
            observed_sha256,
            label=f"{relative}.destination_sha256",
        )
        _require_equal(
            manifest_record["source_bytes"], len(raw), label=f"{relative}.source_bytes"
        )
        _require_equal(
            manifest_record["destination_bytes"],
            len(raw),
            label=f"{relative}.destination_bytes",
        )

        payload = _load_json_bytes(raw, label=relative)
        fact_results = validate_artifact_payload(payload, contract_record)
        _require_equal(
            manifest_record["required_fact_results"],
            fact_results,
            label=f"{relative}.required_fact_results",
        )
        payloads.append(payload)

    _require_equal(
        set(manifest_by_path),
        set(EXPECTED_ARTIFACT_IDENTITIES),
        label="aggregate artifact path set",
    )
    _require_equal(
        _aggregate_access(payloads),
        AGGREGATE_ACCESS,
        label="recomputed aggregate access",
    )
    _require_equal(
        manifest["aggregate_access"],
        AGGREGATE_ACCESS,
        label="manifest aggregate access",
    )
    _require_equal(
        manifest["artifact_read_scope"]["source_artifact_paths_opened"],
        list(EXPECTED_ARTIFACT_IDENTITIES),
        label="source artifact read scope",
    )
    _require_equal(
        manifest["artifact_read_scope"]["non_allowlisted_source_artifact_paths_opened"],
        [],
        label="non-allowlisted source artifact reads",
    )

    return {
        "schema_version": "neuronsifter-no-task-fit-parent-authority-validation-v1",
        "status": "passed",
        "manifest_status": manifest["status"],
        "artifact_count": len(manifest_records),
        "required_fact_count": sum(
            len(record["required_fact_results"]) for record in manifest_records
        ),
        "blocker_count": 0,
        "six_context_outcome_access": manifest["aggregate_access"][
            "six_context_outcome_access"
        ],
        "protected_holdout_access": manifest["aggregate_access"][
            "protected_holdout_access"
        ],
        "next_action": manifest["next_action"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = validate_bound_parent_authority(
            repository_root=args.repository_root,
            contract_path=args.contract,
            manifest_path=args.manifest,
        )
    except AuthorityBindingError as exc:
        raise SystemExit(
            f"no-task-fit parent-authority validation failed: {exc}"
        ) from exc
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
