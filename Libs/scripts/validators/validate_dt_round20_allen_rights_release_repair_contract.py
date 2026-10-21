#!/usr/bin/env python3
"""Validate the frozen Round-20 Allen rights and release repair contract."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

# Support the repository's documented direct-script validator invocation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.round20_track_contracts import ALLEN_FAMILY_UIDS
from Libs.utils.data.trajectory_contracts import is_sha256


CONFIG_RELATIVE = Path("Libs/configs/validation/round20_allen_rights_release_repair.yaml")
OUTPUT_RELATIVE = Path("Output/audits/dt_round20_allen_rights_release_repair_contract.json")
EXPECTED_INPUT_UIDS = frozenset(
    {
        "historical_migration",
        "rights_matrix_v1",
        "release_inventory_v1",
        "retrieval_manifest_v1",
        "official_endpoint_probe_v1",
        "exact_suite_execution_508462",
        "endpoint_main_promotion_v2",
        "exact_suite_failed_attempt_ledger_v1",
        "duplicate_confirmation_508468",
    }
)
EXPECTED_TEST_IDENTITIES = frozenset(
    {
        "Libs.tests.test_dt_round20_allen_rights_release_repair::test_repair_contract_is_canonical_and_fail_closed",
        "Libs.tests.test_dt_round20_allen_rights_release_repair::test_repair_contract_rejects_authorization_and_identity_mutations",
        "Libs.tests.test_dt_round20_allen_rights_release_repair::test_repair_contract_verifies_all_frozen_input_hashes",
        "Libs.tests.test_dt_round20_allen_rights_release_repair::test_repair_contract_reports_missing_and_hash_mismatched_inputs",
    }
)
FALSE_AUTHORIZATIONS = frozenset(
    {"payload_download", "training", "evidence_fabric_sealing", "dt_core_evaluation", "remote_push"}
)
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _blocker(code: str, message: str, **context: object) -> dict[str, object]:
    return {"code": code, "message": message, **context}


def load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("round20_allen_rights_release_repair"), Mapping):
        raise ValueError("round20_allen_rights_release_repair mapping is required")
    return dict(payload["round20_allen_rights_release_repair"])


def validate_contract_definition(contract: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if contract.get("version") != "neuronsifter-round20-allen-rights-release-repair-v1":
        blockers.append(_blocker("contract_version", "Repair contract version is not frozen."))
    if contract.get("status") != "frozen_before_repair_execution":
        blockers.append(_blocker("contract_status", "Repair contract is not frozen before execution."))
    if not GIT_COMMIT_RE.fullmatch(str(contract.get("base_main_commit", ""))):
        blockers.append(_blocker("base_main_commit", "Base main commit must be a 40-character Git identity."))

    inputs = [_mapping(value) for value in _sequence(contract.get("input_artifacts"))]
    input_uids = [str(value.get("artifact_uid")) for value in inputs]
    if len(inputs) != 9 or set(input_uids) != EXPECTED_INPUT_UIDS or len(input_uids) != len(set(input_uids)):
        blockers.append(_blocker("input_identity_inventory", "Exactly nine unique frozen input identities are required."))
    for item in inputs:
        path = Path(str(item.get("path", "")))
        if not item.get("artifact_uid") or not item.get("role"):
            blockers.append(_blocker("input_identity_fields", "Input identity requires an artifact UID and role."))
        if path.is_absolute() or ".." in path.parts or not str(path):
            blockers.append(_blocker("input_identity_path", "Input paths must be repository-relative and traversal-free.", path=str(path)))
        if not is_sha256(item.get("sha256")):
            blockers.append(_blocker("input_identity_hash", "Input identity requires a SHA-256 hash.", path=str(path)))
        if not GIT_COMMIT_RE.fullmatch(str(item.get("source_commit", ""))):
            blockers.append(_blocker("input_source_commit", "Input identity requires a source commit.", path=str(path)))

    family_uids = [str(value) for value in _sequence(contract.get("family_uids"))]
    if len(family_uids) != 12 or set(family_uids) != ALLEN_FAMILY_UIDS or len(family_uids) != len(set(family_uids)):
        blockers.append(_blocker("family_inventory_drift", "The canonical 12-family Allen inventory is not exact."))

    source_policy = _mapping(contract.get("source_policy"))
    if source_policy.get("access_class") != "public_uncontrolled" or source_policy.get("official_hosts_only") is not True:
        blockers.append(_blocker("public_official_source_policy", "Only official public uncontrolled sources are allowed."))
    for key in ("general_terms_satisfy_dataset_specific_rights", "historical_release_hint_promotable", "mutable_latest_query_allowed"):
        if source_policy.get(key) is not False:
            blockers.append(_blocker("source_policy_promotion", f"{key} must remain false."))
    for key in ("release_identity_requires_official_immutable_evidence", "terms_and_release_snapshots_hash_required", "checked_at_execution_time_required"):
        if source_policy.get(key) is not True:
            blockers.append(_blocker("source_policy_provenance", f"{key} must remain true."))
    if set(_sequence(source_policy.get("allowed_host_suffixes"))) != {"alleninstitute.org", "brain-map.org"}:
        blockers.append(_blocker("official_host_inventory", "Official Allen host suffixes are not exact."))

    configuration = _mapping(contract.get("configuration"))
    if configuration.get("maximum_requests") != 48 or configuration.get("max_concurrent_requests") != 1:
        blockers.append(_blocker("request_budget", "Request count and serial concurrency must remain frozen."))
    if configuration.get("request_timeout_seconds") != 20 or configuration.get("retry_count") != 0:
        blockers.append(_blocker("request_retry_policy", "Timeout must be 20 seconds and retries must remain disabled."))
    if configuration.get("maximum_snapshot_bytes_per_response") != 1048576 or configuration.get("maximum_total_snapshot_bytes") != 25165824:
        blockers.append(_blocker("snapshot_budget", "Snapshot byte budgets are not exact."))
    if configuration.get("redirect_to_non_official_host_allowed") is not False:
        blockers.append(_blocker("redirect_policy", "Redirects to non-official hosts must remain blocked."))

    tests = _mapping(contract.get("test_membership"))
    if tests.get("parent_identity_count") != 618 or tests.get("expected_addition_count") != 4 or tests.get("expected_identity_count") != 622:
        blockers.append(_blocker("test_identity_counts", "Frozen test identity counts are not exact."))
    if set(_sequence(tests.get("identities"))) != EXPECTED_TEST_IDENTITIES or tests.get("mandatory_skips_allowed") is not False:
        blockers.append(_blocker("test_identity_inventory", "Repair test identities are incomplete or skips are allowed."))

    authorization = _mapping(contract.get("authorization"))
    for key in FALSE_AUTHORIZATIONS:
        if authorization.get(key) is not False:
            blockers.append(_blocker("unauthorized_execution", f"{key} must remain false.", authorization=key))
    for key in ("bounded_official_metadata_probe", "dataset_specific_terms_resolution", "immutable_release_identity_resolution"):
        if authorization.get(key) is not True:
            blockers.append(_blocker("missing_repair_authorization", f"{key} must remain explicitly authorized.", authorization=key))
    return sorted(blockers, key=lambda value: (str(value["code"]), str(value.get("path", ""))))


def validate_input_identities(contract: Mapping[str, object], *, repository: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    for item in sorted((_mapping(value) for value in _sequence(contract.get("input_artifacts"))), key=lambda value: str(value.get("artifact_uid"))):
        relative = Path(str(item.get("path", "")))
        path = repository / relative
        expected = str(item.get("sha256", ""))
        actual = file_sha256(path) if path.is_file() else None
        if actual is None:
            status = "missing"
            blockers.append(_blocker("input_identity_missing", "Frozen input artifact is missing.", artifact_uid=item.get("artifact_uid"), path=str(relative)))
        elif actual != expected:
            status = "hash_mismatch"
            blockers.append(_blocker("input_hash_mismatch", "Frozen input artifact hash does not match.", artifact_uid=item.get("artifact_uid"), path=str(relative), expected_sha256=expected, actual_sha256=actual))
        else:
            status = "verified"
        identities.append(
            {
                "artifact_uid": str(item.get("artifact_uid")),
                "path": str(relative),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "source_commit": str(item.get("source_commit")),
                "role": str(item.get("role")),
                "status": status,
            }
        )
    return identities, sorted(blockers, key=lambda value: (str(value["code"]), str(value.get("path", ""))))


def build_report(contract: Mapping[str, object], *, repository: Path, config_path: Path, candidate_commit: str) -> dict[str, object]:
    definition_blockers = validate_contract_definition(contract)
    identities, identity_blockers = validate_input_identities(contract, repository=repository)
    blockers = sorted(definition_blockers + identity_blockers, key=lambda value: (str(value["code"]), str(value.get("path", ""))))
    authorization = _mapping(contract.get("authorization"))
    passed = not blockers
    return {
        "schema_version": "neuronsifter-round20-allen-rights-release-repair-contract-v1",
        "artifact_uid": "dt_round20_allen_rights_release_repair_contract",
        "status": "passed" if passed else "blocked_contract_or_input_identity",
        "gate": "allen_rights_release_repair_contract_frozen" if passed else "blocked_allen_rights_release_repair_contract",
        "candidate_commit": candidate_commit,
        "config_sha256": file_sha256(config_path),
        "input_identities": identities,
        "family_count": len(_sequence(contract.get("family_uids"))),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "scope": {
            "access_class": "public_uncontrolled",
            "official_hosts_only": True,
            "payload_download_authorized": False,
            "training_authorized": False,
            "evidence_fabric_sealing_authorized": False,
            "dt_core_evaluation_authorized": False,
            "remote_push_authorized": False,
        },
        "authorization": authorization,
        "next_action": (
            "execute_bounded_family_by_family_official_metadata_resolution_without_payload_retrieval"
            if passed
            else "repair_contract_or_input_identity_blockers_under_a_new_artifact_identity"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_RELATIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT_RELATIVE)
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else repository / args.config
    output_path = args.output if args.output.is_absolute() else repository / args.output
    candidate_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    contract = load_contract(config_path)
    report = build_report(
        contract,
        repository=repository,
        config_path=config_path,
        candidate_commit=candidate_commit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gate": report["gate"], "blocker_count": report["blocker_count"], "output": str(output_path)}, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
