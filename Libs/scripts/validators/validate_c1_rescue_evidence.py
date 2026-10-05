"""Read-only fail-closed validator for the C1 rescue corpus evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1_bindings import (  # noqa: E402
    build_c1_observation_binding_artifacts,
    serialize_binding_manifest,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256, hash_path  # noqa: E402
from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import read_tsv, write_json  # noqa: E402


HISTORICAL_BUNDLE_PATH = "Output/audits/public_foundation_corpus_v2_evidence_bundle.json"
HISTORICAL_BLOCK_PATH = "Output/audits/public_foundation_corpus_v2_canonical_observation_blocks.json"
HISTORICAL_CROSSWALK_PATH = "Output/audits/public_foundation_corpus_v2_crosswalk_delta.json"
HISTORICAL_ACTION_SUBSET_PATH = "Output/audits/c1_v2_qualified_action_subset_manifest.json"
HISTORICAL_SPLIT_PATH = "Output/audits/public_foundation_corpus_v2_frozen_splits.json"
HISTORICAL_QC_PATH = "Output/audits/public_foundation_corpus_v2_qc_ess_statistics.json"
CONFIG_PATH = "Libs/configs/data/public_foundation_corpus_rescue.yaml"
OUTPUT_PATH = "Output/audits/public_foundation_corpus_rescue_evidence_validation.json"
HISTORICAL_STUDENT_PATH = "Output/results/c1/runs/student_b8_atlas_20260722_v1/run_result.json"
HISTORICAL_STUDENT_SHA256 = "8b2071247ababbbdbbd90200fdd8765bef8a11aacca771f40c77f6a37b946989"
RESCUE_BUNDLE_PATH = "Output/audits/public_foundation_corpus_rescue_evidence_bundle.json"
RESCUE_CORPUS_UID = "public_foundation_corpus_rescue"
RESCUE_EVIDENCE_NAMES = (
    "source_qualification",
    "crosswalk_delta",
    "canonical_observation_blocks",
    "structured_action_catalog",
    "qualified_action_subset",
    "frozen_splits",
    "leakage_audit",
    "qc_ess_statistics",
    "storage_ledger",
    "controlled_isolation",
    "parent_v1_evidence_bundle",
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _semantic_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "producer", "corpus_hash", "corpus_identity_hash"}
    }


def _blocker(code: str, path: object, message: object) -> dict[str, str]:
    return {"code": code, "path": str(path), "message": str(message)}


def _validate_artifact_ref(
    repo_root: Path,
    name: str,
    reference: Mapping[str, object],
    *,
    accepted_payload_statuses: set[str] | None = None,
    allow_missing_payload_status: bool = False,
) -> dict[str, object]:
    path = safe_repo_path(repo_root, str(reference["path"]))
    if sha256_file(path) != str(reference["sha256"]):
        raise ValueError(f"artifact SHA-256 drift: {name}")
    payload = _load_json(path)
    expected_semantic = reference.get("semantic_hash")
    if expected_semantic is not None and canonical_sha256(_semantic_payload(payload)) != expected_semantic:
        raise ValueError(f"artifact semantic hash drift: {name}")
    if payload.get("status") != reference.get("status") and not (
        allow_missing_payload_status and payload.get("status") is None
    ) and (
        accepted_payload_statuses is None
        or str(payload.get("status")) not in accepted_payload_statuses
    ):
        raise ValueError(f"artifact status drift: {name}")
    return payload


def _controlled_reachability(payload: Mapping[str, object], *, label: str) -> int:
    value = payload.get(
        "controlled_reachability_count",
        payload.get("controlled_reachability"),
    )
    if value is None or int(value) != 0:
        raise ValueError(f"{label} controlled reachability is nonzero or absent")
    findings = payload.get("route_findings")
    if not isinstance(findings, list) or not findings:
        raise ValueError(f"{label} route findings are absent")
    if any(
        not isinstance(row, Mapping) or int(row.get("controlled_match_count", -1)) != 0
        for row in findings
    ):
        raise ValueError(f"{label} contains a controlled route match")
    return 0


def _validate_current_controlled_isolation_reference(
    repo_root: Path,
    reference: Mapping[str, object],
) -> dict[str, object]:
    """Validate the frozen C1 snapshot and its current pass-state successor."""
    config = yaml.safe_load((repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    policy = config["public_foundation_corpus_rescue"]["frozen_evidence_snapshots"][
        "controlled_isolation"
    ]
    if not isinstance(policy, Mapping):
        raise ValueError("controlled-isolation frozen snapshot policy is malformed")
    if str(reference.get("path")) != str(policy.get("live_path")):
        raise ValueError("controlled-isolation live path is not policy-bound")
    if reference.get("status") not in {"pass", "passed"}:
        raise ValueError("controlled-isolation frozen reference is not passed")

    snapshot_path = safe_repo_path(repo_root, str(policy["path"]))
    snapshot_sha256 = sha256_file(snapshot_path)
    expected_sha256 = str(policy["sha256"])
    expected_bytes = int(policy["bytes"])
    if (
        snapshot_path.stat().st_size != expected_bytes
        or snapshot_sha256 != expected_sha256
        or snapshot_sha256 != str(reference.get("sha256"))
        or expected_bytes != int(reference.get("bytes", -1))
    ):
        raise ValueError("controlled-isolation frozen snapshot binding drift")
    snapshot = _load_json(snapshot_path)
    if snapshot.get("status") not in {"pass", "passed"}:
        raise ValueError("controlled-isolation frozen snapshot is not passed")
    _controlled_reachability(snapshot, label="frozen controlled-isolation snapshot")

    live_path = safe_repo_path(repo_root, str(policy["live_path"]))
    live = _load_json(live_path)
    if live.get("status") not in {"pass", "passed"}:
        raise ValueError("current controlled-isolation audit is not passed")
    _controlled_reachability(live, label="current controlled-isolation audit")

    if sha256_file(live_path) != snapshot_sha256:
        mutable_binding_keys = {
            "config_sha256",
            "resolved_config_sha256",
            "config_graph_sha256",
        }
        frozen_semantics = {
            key: value for key, value in snapshot.items() if key not in mutable_binding_keys
        }
        live_semantics = {
            key: value for key, value in live.items() if key not in mutable_binding_keys
        }
        if live_semantics != frozen_semantics:
            raise ValueError(
                "current controlled-isolation audit is not a binding-only supersession"
            )
        if policy.get("live_supersession_policy") != (
            "require_pass_and_zero_controlled_reachability"
        ):
            raise ValueError("controlled-isolation supersession policy is not fail closed")
    return live


def _validate_versioned_block_contract(
    manifest: Mapping[str, object], row: Mapping[str, object]
) -> None:
    origin = row.get("corpus_origin")
    if origin not in {"public_foundation_corpus_v1", "public_foundation_corpus_v2"}:
        raise ValueError(f"unknown corpus origin: {row.get('canonical_uid')}")
    if manifest.get("role") != "qualified_for_mainline" or manifest.get("qc_grade") not in {
        "A",
        "B",
    }:
        raise ValueError(f"block role/grade is ineligible: {row['canonical_uid']}")
    if (
        row.get("accepted_value_block_format") is not True
        or row.get("semantic_axes_complete") is not True
        or manifest.get("semantic_axes_complete") is not True
        or manifest.get("value_block_format") not in {"zarr", "parquet"}
    ):
        raise ValueError(f"block value/semantic contract failed: {row['canonical_uid']}")
    if origin == "public_foundation_corpus_v2" and (
        manifest.get("real_value_matrix") is not True or manifest.get("blocker_codes")
    ):
        raise ValueError(f"rescue block contract failed: {row['canonical_uid']}")


def _validate_block(repo_root: Path, row: Mapping[str, object]) -> dict[str, object]:
    path = safe_repo_path(repo_root, str(row["manifest_path"]))
    if sha256_file(path) != str(row["manifest_sha256"]):
        raise ValueError(f"block manifest hash drift: {row['canonical_uid']}")
    manifest = _load_json(path)
    for key in (
        "canonical_uid",
        "family_id",
        "source_id",
        "source_accession",
        "publication_accession_group",
        "mirror_group",
        "block_sha256",
        "qc_grade",
        "observation_count",
        "biological_unit_count",
        "effective_sample_size",
    ):
        if manifest.get(key) != row.get(key):
            raise ValueError(f"block identity mismatch: {row['canonical_uid']}:{key}")
    _validate_versioned_block_contract(manifest, row)
    files = manifest.get("block_files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"block file list is absent: {row['canonical_uid']}")
    content: list[dict[str, object]] = []
    for record in files:
        if "relative_path" in record:
            target = path.parent / str(record["relative_path"])
            content_schema = "relative_path_with_file_count"
        elif "path" in record:
            target = safe_repo_path(repo_root, str(record["path"]))
            try:
                target.relative_to(path.parent)
            except ValueError as exc:
                raise ValueError(f"block payload escapes manifest directory: {target}") from exc
            content_schema = "repo_path_single_file"
        else:
            raise ValueError(f"block payload path is absent: {row['canonical_uid']}")
        count, byte_count, digest = hash_path(target)
        if (
            count != int(record.get("file_count", 1))
            or byte_count != int(record["bytes"])
            or digest != str(record["sha256"])
        ):
            raise ValueError(f"block payload drift: {target}")
        normalized = {"name": record["name"], "bytes": byte_count, "sha256": digest}
        if content_schema == "relative_path_with_file_count":
            normalized["file_count"] = count
        content.append(normalized)
    if canonical_sha256(content) != str(manifest["block_sha256"]):
        raise ValueError(f"block content address drift: {row['canonical_uid']}")
    return manifest


def _validate_splits(repo_root: Path, splits: Mapping[str, object]) -> None:
    query = splits["query_domain"]
    query_path = safe_repo_path(repo_root, str(query["path"]))
    if sha256_file(query_path) != str(query["sha256"]):
        raise ValueError("V2 query-domain hash drift")
    query_rows = read_tsv(query_path)
    if len(query_rows) != int(query["record_count"]) or len(query_rows) != 7191:
        raise ValueError("V2 query-domain count drift")
    if len({row["canonical_uid"] for row in query_rows}) != len(query_rows):
        raise ValueError("V2 query-domain canonical UID duplication")
    if len({row["biological_unit_uid"] for row in query_rows}) != len(query_rows):
        raise ValueError("V2 query-domain biological-unit duplication")
    for manifest in splits["manifests"]:
        expected_hash = canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        )
        if expected_hash != manifest["manifest_hash"]:
            raise ValueError(f"V2 split manifest hash drift: {manifest['split_uid']}")
        roles: dict[str, list[dict[str, str]]] = {}
        for role, binding in manifest["role_bindings"].items():
            path = safe_repo_path(repo_root, str(binding["path"]))
            if sha256_file(path) != str(binding["sha256"]):
                raise ValueError(f"V2 split role hash drift: {manifest['split_uid']}:{role}")
            rows = read_tsv(path)
            if len(rows) != int(binding["record_count"]):
                raise ValueError(f"V2 split role count drift: {manifest['split_uid']}:{role}")
            roles[role] = rows
        role_uids = [{row["canonical_uid"] for row in roles[role]} for role in ("train", "validation", "test")]
        if (role_uids[0] & role_uids[1]) or (role_uids[0] & role_uids[2]) or (role_uids[1] & role_uids[2]):
            raise ValueError(f"V2 biological unit crosses split roles: {manifest['split_uid']}")
        if manifest["split_uid"] in {"regimen_holdout", "target_program_holdout"}:
            if any(
                row["regimen_uid"].startswith("not_applicable_control_context:")
                for role in ("validation", "test")
                for row in roles[role]
            ):
                raise ValueError("Actionless control entered a regimen/target holdout role")


def _validate_rescue_identity(bundle: Mapping[str, object]) -> dict[str, object]:
    """Validate the current rescue identity without using the historical validator."""
    identity = bundle.get("identity_payload")
    if not isinstance(identity, Mapping):
        raise ValueError("current rescue identity_payload is missing")
    if bundle.get("corpus_uid") != RESCUE_CORPUS_UID:
        raise ValueError("current rescue corpus UID is not the configured semantic identity")
    corpus_hash = str(bundle.get("corpus_hash", ""))
    if len(corpus_hash) != 64 or canonical_sha256(identity) != corpus_hash:
        raise ValueError("current rescue corpus identity does not derive from identity_payload")
    if bundle.get("corpus_identity_hash") != corpus_hash:
        raise ValueError("current rescue corpus_identity_hash is not bound to corpus_hash")
    if identity.get("corpus_uid") != RESCUE_CORPUS_UID:
        raise ValueError("current rescue identity payload has a different corpus UID")
    for key in ("parent_corpus_uid", "parent_corpus_identity_hash", "scope"):
        if not identity.get(key) or identity.get(key) != bundle.get(key):
            raise ValueError(f"current rescue identity field is not bundle-bound: {key}")
    if bundle.get("status") != "valid_fail_closed" or bundle.get("blockers") != []:
        raise ValueError("current rescue bundle is not valid_fail_closed without blockers")
    return dict(identity)


def _validate_rescue_block(
    repo_root: Path,
    row: Mapping[str, object],
    *,
    parent_corpus_uid: str,
    rescue_corpus_uid: str,
    verify_content_hashes: bool,
) -> dict[str, object]:
    manifest_path = safe_repo_path(repo_root, str(row["manifest_path"]))
    if sha256_file(manifest_path) != str(row["manifest_sha256"]):
        raise ValueError(f"rescue block manifest hash drift: {row.get('canonical_uid')}")
    manifest = _load_json(manifest_path)
    for key in (
        "canonical_uid",
        "family_id",
        "source_id",
        "source_accession",
        "publication_accession_group",
        "mirror_group",
        "block_sha256",
        "qc_grade",
        "observation_count",
        "biological_unit_count",
        "effective_sample_size",
    ):
        if manifest.get(key) != row.get(key):
            raise ValueError(f"rescue block identity mismatch: {row.get('canonical_uid')}:{key}")
    if row.get("corpus_origin") not in {parent_corpus_uid, rescue_corpus_uid}:
        raise ValueError(f"rescue block has an unknown corpus origin: {row.get('canonical_uid')}")
    if (
        manifest.get("role") != "qualified_for_mainline"
        or manifest.get("qc_grade") not in {"A", "B"}
        or manifest.get("admission_allowed") is not True
        or manifest.get("semantic_axes_complete") is not True
        or manifest.get("value_block_format") not in {"zarr", "parquet"}
    ):
        raise ValueError(f"rescue block role, grade, or value contract failed: {row.get('canonical_uid')}")
    records = manifest.get("block_files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"rescue block file list is absent: {row.get('canonical_uid')}")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"rescue block file record is malformed: {row.get('canonical_uid')}")
        if "relative_path" in record:
            target = safe_repo_path(manifest_path.parent, str(record["relative_path"]))
        elif "path" in record:
            target = safe_repo_path(repo_root, str(record["path"]))
        else:
            raise ValueError(f"rescue block file path is absent: {row.get('canonical_uid')}")
        if verify_content_hashes:
            count, byte_count, digest = hash_path(target)
            if (
                count != int(record.get("file_count", 1))
                or byte_count != int(record["bytes"])
                or digest != str(record["sha256"])
            ):
                raise ValueError(f"rescue block payload drift: {target}")
    return manifest


def _validate_rescue_splits(
    repo_root: Path,
    splits: Mapping[str, object],
    *,
    expected_query_count: int,
) -> dict[str, str]:
    query = splits.get("query_domain")
    if not isinstance(query, Mapping):
        raise ValueError("current rescue query-domain binding is missing")
    query_path = safe_repo_path(repo_root, str(query["path"]))
    if sha256_file(query_path) != str(query["sha256"]):
        raise ValueError("current rescue query-domain hash drift")
    query_rows = read_tsv(query_path)
    if len(query_rows) != int(query["record_count"]) or len(query_rows) != expected_query_count:
        raise ValueError("current rescue query-domain count drift")
    if len({row["canonical_uid"] for row in query_rows}) != len(query_rows):
        raise ValueError("current rescue query-domain canonical UID duplication")
    if len({row["biological_unit_uid"] for row in query_rows}) != len(query_rows):
        raise ValueError("current rescue query-domain biological-unit duplication")
    manifests = splits.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("current rescue split manifest list is missing")
    split_hashes: dict[str, str] = {}
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            raise ValueError("current rescue split manifest record is malformed")
        split_uid = str(manifest["split_uid"])
        expected_hash = canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        )
        if expected_hash != manifest.get("manifest_hash"):
            raise ValueError(f"current rescue split manifest hash drift: {split_uid}")
        split_hashes[split_uid] = str(manifest["manifest_hash"])
        bindings = manifest.get("role_bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError(f"current rescue split role bindings are missing: {split_uid}")
        role_uids: dict[str, set[str]] = {}
        for role in ("train", "validation", "test"):
            binding = bindings.get(role)
            if not isinstance(binding, Mapping):
                raise ValueError(f"current rescue split role is missing: {split_uid}:{role}")
            path = safe_repo_path(repo_root, str(binding["path"]))
            if sha256_file(path) != str(binding["sha256"]):
                raise ValueError(f"current rescue split role hash drift: {split_uid}:{role}")
            rows = read_tsv(path)
            if len(rows) != int(binding["record_count"]):
                raise ValueError(f"current rescue split role count drift: {split_uid}:{role}")
            role_uids[role] = {row["biological_unit_uid"] for row in rows}
        if any(role_uids[left] & role_uids[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"current rescue biological unit crosses split roles: {split_uid}")
    return split_hashes


def _split_role_content_membership(
    repo_root: Path,
    splits: Mapping[str, object],
    *,
    label: str,
) -> dict[str, dict[str, dict[str, int]]]:
    """Return a stable split-role membership map for semantic migrations.

    Canonical UIDs and storage paths are intentionally allowed to change during
    the rescue migration.  The content hash of each biological-unit row is the
    stable identity used to prove that no unit moved between train,
    validation, or test.
    """

    manifests = splits.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError(f"{label} split manifests are missing")
    membership: dict[str, dict[str, dict[str, int]]] = {}
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            raise ValueError(f"{label} split manifest is malformed")
        split_uid = str(manifest.get("split_uid", ""))
        if not split_uid or split_uid in membership:
            raise ValueError(f"{label} split UID is missing or duplicated: {split_uid}")
        bindings = manifest.get("role_bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError(f"{label} split role bindings are missing: {split_uid}")
        role_membership: dict[str, dict[str, int]] = {}
        for role in ("train", "validation", "test"):
            binding = bindings.get(role)
            if not isinstance(binding, Mapping):
                raise ValueError(f"{label} split role is missing: {split_uid}:{role}")
            path = safe_repo_path(repo_root, str(binding.get("path", "")))
            if sha256_file(path) != str(binding.get("sha256", "")):
                raise ValueError(f"{label} split role hash drift: {split_uid}:{role}")
            rows = read_tsv(path)
            if len(rows) != int(binding.get("record_count", -1)):
                raise ValueError(f"{label} split role count drift: {split_uid}:{role}")
            content_hashes: Counter[str] = Counter()
            for row in rows:
                content_hash = str(row.get("content_hash", ""))
                if len(content_hash) != 64 or any(
                    character not in "0123456789abcdef" for character in content_hash.lower()
                ):
                    raise ValueError(
                        f"{label} split row has no stable content hash: {split_uid}:{role}"
                    )
                content_hashes[content_hash] += 1
            role_membership[role] = dict(sorted(content_hashes.items()))
        membership[split_uid] = role_membership
    return dict(sorted(membership.items()))


def _validate_rescue_migration(repo_root: Path, bundle: Mapping[str, object]) -> dict[str, object]:
    """Record the explicit semantic migration without treating the old bundle as proof."""
    historical_path = safe_repo_path(repo_root, HISTORICAL_BUNDLE_PATH)
    historical = _load_json(historical_path)
    current_identity = bundle["identity_payload"]
    historical_identity = historical.get("identity_payload")
    if not isinstance(historical_identity, Mapping):
        raise ValueError("historical rescue identity payload is absent")
    current_blocks = current_identity.get("block_hashes")
    historical_blocks = historical_identity.get("block_hashes")
    if not isinstance(current_blocks, Mapping) or not isinstance(historical_blocks, Mapping):
        raise ValueError("rescue block identity maps are absent")
    if sorted(map(str, current_blocks.values())) != sorted(map(str, historical_blocks.values())):
        raise ValueError("semantic rescue migration changed block content identities")
    if current_identity.get("parent_corpus_identity_hash") != historical_identity.get("parent_corpus_identity_hash"):
        raise ValueError("semantic rescue migration changed the parent corpus identity")
    for field in ("qualified_block_count", "grade_ab_observation_count", "grade_ab_effective_sample_size", "qualified_action_count"):
        if bundle.get(field) != historical.get(field):
            raise ValueError(f"semantic rescue migration changed frozen aggregate count: {field}")
    current_hash = str(bundle["corpus_hash"])
    historical_hash = str(historical.get("corpus_hash"))
    if current_hash == historical_hash:
        raise ValueError("semantic rescue migration did not produce a distinct current identity")
    current_evidence = bundle.get("evidence")
    if not isinstance(current_evidence, Mapping):
        raise ValueError("current rescue evidence map is absent during migration validation")
    current_split_reference = current_evidence.get("frozen_splits")
    if not isinstance(current_split_reference, Mapping):
        raise ValueError("current rescue split reference is absent during migration validation")
    current_split_path = safe_repo_path(repo_root, str(current_split_reference.get("path", "")))
    if sha256_file(current_split_path) != str(current_split_reference.get("sha256", "")):
        raise ValueError("current rescue split artifact hash drift during migration validation")
    historical_evidence = historical.get("evidence")
    if not isinstance(historical_evidence, Mapping):
        raise ValueError("historical rescue evidence map is absent during migration validation")
    historical_split_reference = historical_evidence.get("frozen_splits")
    if not isinstance(historical_split_reference, Mapping):
        raise ValueError("historical rescue split reference is absent during migration validation")
    historical_split_path = safe_repo_path(repo_root, str(historical_split_reference.get("path", "")))
    if sha256_file(historical_split_path) != str(historical_split_reference.get("sha256", "")):
        raise ValueError("historical rescue split artifact hash drift during migration validation")
    current_split_membership = _split_role_content_membership(
        repo_root,
        _load_json(current_split_path),
        label="current rescue",
    )
    historical_split_membership = _split_role_content_membership(
        repo_root,
        _load_json(historical_split_path),
        label="historical rescue",
    )
    if current_split_membership != historical_split_membership:
        raise ValueError(
            "semantic rescue migration changed split-role content membership"
        )
    current_split_membership_hash = canonical_sha256(current_split_membership)
    historical_split_membership_hash = canonical_sha256(historical_split_membership)
    return {
        "status": "verified",
        "migration_type": "semantic_identifier_and_storage_path_migration",
        "historical_bundle_path": HISTORICAL_BUNDLE_PATH,
        "historical_bundle_sha256": sha256_file(historical_path),
        "historical_corpus_hash": historical_hash,
        "current_corpus_hash": current_hash,
        "same_parent_identity": True,
        "same_block_content_identity_multiset": True,
        "same_split_role_content_membership": True,
        "current_split_membership_hash": current_split_membership_hash,
        "historical_split_membership_hash": historical_split_membership_hash,
        "historical_bundle_used_as_provenance_only": True,
    }


def validate_rescue_payload(
    bundle: Mapping[str, object],
    repo_root: Path = REPO_ROOT,
    *,
    verify_large_block_payloads: bool = True,
) -> dict[str, object]:
    """Validate the current semantic rescue bundle independently of historical contracts."""
    blockers: list[dict[str, str]] = []
    identity: dict[str, object] = {}
    artifacts: dict[str, dict[str, object]] = {}
    block_rows: list[Mapping[str, object]] = []
    migration: dict[str, object] = {}
    try:
        identity = _validate_rescue_identity(bundle)
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(_blocker("c1_rescue_semantic_identity_invalid", RESCUE_BUNDLE_PATH, exc))
    try:
        evidence = bundle["evidence"]
        if not isinstance(evidence, Mapping):
            raise ValueError("current rescue evidence map is missing")
        for name in RESCUE_EVIDENCE_NAMES:
            reference = evidence.get(name)
            if not isinstance(reference, Mapping):
                raise ValueError(f"current rescue evidence reference is missing: {name}")
            if name == "controlled_isolation":
                artifacts[name] = _validate_current_controlled_isolation_reference(
                    repo_root,
                    reference,
                )
            else:
                artifacts[name] = _validate_artifact_ref(
                    repo_root,
                    name,
                    reference,
                    allow_missing_payload_status=name == "parent_v1_evidence_bundle",
                )
        controlled = artifacts["controlled_isolation"]
        controlled_reachability = controlled.get(
            "controlled_reachability_count",
            controlled.get("controlled_reachability"),
        )
        if controlled_reachability != 0 or evidence["controlled_isolation"].get("controlled_reachability") != 0:
            raise ValueError("current rescue controlled reachability is nonzero")
        parent = artifacts["parent_v1_evidence_bundle"]
        if parent.get("corpus_hash") != bundle.get("parent_corpus_identity_hash"):
            raise ValueError("current rescue parent corpus identity drift")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_semantic_evidence_reference_invalid", RESCUE_BUNDLE_PATH, exc))
    try:
        block_artifact = artifacts["canonical_observation_blocks"]
        raw_rows = block_artifact.get("qualified_blocks")
        if not isinstance(raw_rows, list) or len(raw_rows) != int(bundle["qualified_block_count"]):
            raise ValueError("current rescue qualified block count drift")
        block_rows = [row for row in raw_rows if isinstance(row, Mapping)]
        if len(block_rows) != len(raw_rows) or len({str(row["canonical_uid"]) for row in block_rows}) != len(block_rows):
            raise ValueError("current rescue block identities are not unique")
        for row in block_rows:
            _validate_rescue_block(
                repo_root,
                row,
                parent_corpus_uid=str(bundle["parent_corpus_uid"]),
                rescue_corpus_uid=RESCUE_CORPUS_UID,
                verify_content_hashes=verify_large_block_payloads,
            )
        if identity.get("block_hashes") != {str(row["canonical_uid"]): row["block_sha256"] for row in block_rows}:
            raise ValueError("current rescue block identity map drift")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_semantic_block_validation_failed", RESCUE_BUNDLE_PATH, exc))
    try:
        crosswalk = artifacts["crosswalk_delta"]
        if identity.get("crosswalk_delta_hash") != canonical_sha256(_semantic_payload(crosswalk)):
            raise ValueError("current rescue crosswalk identity drift")
        table = crosswalk.get("table")
        if not isinstance(table, Mapping) or len(read_tsv(safe_repo_path(repo_root, str(table["path"])))) != int(crosswalk["record_count"]):
            raise ValueError("current rescue crosswalk table count drift")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_semantic_crosswalk_invalid", RESCUE_BUNDLE_PATH, exc))
    try:
        actions = artifacts["structured_action_catalog"]
        catalogs = actions.get("qualified_catalogs")
        if not isinstance(catalogs, list) or sum(int(row["record_count"]) for row in catalogs) != int(bundle["qualified_action_count"]):
            raise ValueError("current rescue action catalog count drift")
        if identity.get("action_catalog_hash") != canonical_sha256(catalogs):
            raise ValueError("current rescue action catalog identity drift")
        for row in catalogs:
            path = safe_repo_path(repo_root, str(row["path"]))
            if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
                raise ValueError(f"current rescue action catalog content drift: {row.get('asset_id')}")
        action_subset = artifacts["qualified_action_subset"]
        expected_subset_hash = canonical_sha256({"corpus_hash": bundle["corpus_hash"], "qualified_catalogs": catalogs})
        if action_subset.get("corpus_hash") != bundle["corpus_hash"] or action_subset.get("qualified_action_subset_hash") != expected_subset_hash:
            raise ValueError("current rescue qualified action subset identity drift")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_semantic_action_catalog_invalid", RESCUE_BUNDLE_PATH, exc))
    try:
        split_hashes = _validate_rescue_splits(
            repo_root,
            artifacts["frozen_splits"],
            expected_query_count=int(float(bundle["grade_ab_effective_sample_size"])),
        )
        if identity.get("query_domain_hash") != artifacts["frozen_splits"]["query_domain"]["sha256"] or identity.get("split_hashes") != split_hashes:
            raise ValueError("current rescue split identity drift")
        leakage = artifacts["leakage_audit"]
        if leakage.get("status") != "passed" or leakage.get("zero_overlap") is not True or leakage.get("blocker_codes"):
            raise ValueError("current rescue leakage audit is not clean")
        if identity.get("leakage_hash") != canonical_sha256(_semantic_payload(leakage)):
            raise ValueError("current rescue leakage identity drift")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_semantic_split_or_leakage_invalid", RESCUE_BUNDLE_PATH, exc))
    try:
        qc = artifacts["qc_ess_statistics"]
        if (
            qc.get("status") != "passed"
            or qc.get("grade_ab_observation_count") != bundle.get("grade_ab_observation_count")
            or qc.get("grade_ab_effective_sample_size") != bundle.get("grade_ab_effective_sample_size")
            or qc.get("grade_ab_action_count") != bundle.get("qualified_action_count")
            or qc.get("blocker_codes")
            or any(qc.get(key, 0) != 0 for key in ("synthetic_observation_count", "weak_calibration_observation_count", "controlled_observation_count"))
        ):
            raise ValueError("current rescue QC/ESS contract is not clean")
        storage = artifacts["storage_ledger"]
        if storage.get("status") != "passed" or storage.get("cap_status") != "passed" or storage.get("partial_scan") is not False or storage.get("blocker_codes"):
            raise ValueError("current rescue storage contract is not complete")
        if identity.get("qc_ess_hash") != canonical_sha256(_semantic_payload(qc)) or identity.get("storage_policy_hash") != canonical_sha256(_semantic_payload(storage)):
            raise ValueError("current rescue QC/storage identity drift")
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(_blocker("c1_rescue_semantic_qc_or_storage_invalid", RESCUE_BUNDLE_PATH, exc))
    try:
        binding = bundle["c1_observation_binding"]
        if not isinstance(binding, Mapping):
            raise ValueError("current rescue observation binding is missing")
        manifest_ref = binding["manifest"]
        table_ref = binding["row_table"]
        manifest_path = safe_repo_path(repo_root, str(manifest_ref["path"]))
        table_path = safe_repo_path(repo_root, str(table_ref["path"]))
        if sha256_file(manifest_path) != str(manifest_ref["sha256"]) or sha256_file(table_path) != str(table_ref["sha256"]):
            raise ValueError("current rescue observation binding file hash drift")
        manifest = _load_json(manifest_path)
        for key, expected in (("corpus_uid", RESCUE_CORPUS_UID), ("corpus_hash", bundle["corpus_hash"]), ("block_count", bundle["qualified_block_count"]), ("biological_unit_count", int(float(bundle["grade_ab_effective_sample_size"]))), ("observation_count", bundle["grade_ab_observation_count"]), ("binding_hash", binding["binding_hash"])):
            if manifest.get(key) != expected:
                raise ValueError(f"current rescue observation binding drift: {key}")
        if manifest.get("row_binding_table", {}).get("record_count") != int(float(bundle["grade_ab_effective_sample_size"])):
            raise ValueError("current rescue observation binding row count drift")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_semantic_observation_binding_invalid", RESCUE_BUNDLE_PATH, exc))
    try:
        migration = _validate_rescue_migration(repo_root, bundle)
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_semantic_migration_invalid", RESCUE_BUNDLE_PATH, exc))
    return {
        "schema_version": "neuronsifter-public-foundation-corpus-rescue-evidence-validation",
        "validation_scope": "current_semantic_rescue_identity_and_payload_integrity",
        "status": "valid_fail_closed" if not blockers else "blocked",
        "corpus_uid": bundle.get("corpus_uid", "unknown"),
        "corpus_identity_hash": bundle.get("corpus_hash", "unknown"),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "semantic_migration": migration,
        "verified_counts": {
            "qualified_blocks": len(block_rows),
            "observations": bundle.get("grade_ab_observation_count"),
            "biological_units": bundle.get("grade_ab_effective_sample_size"),
            "qualified_actions": bundle.get("qualified_action_count"),
        },
    }


def validate_rescue(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    bundle_path = safe_repo_path(repo_root, RESCUE_BUNDLE_PATH)
    bundle = _load_json(bundle_path)
    report = validate_rescue_payload(bundle, repo_root)
    report["bundle_path"] = RESCUE_BUNDLE_PATH
    report["bundle_sha256"] = sha256_file(bundle_path)
    return report


def validate_payload(
    bundle: Mapping[str, object],
    repo_root: Path = REPO_ROOT,
    *,
    verify_large_block_payloads: bool = True,
) -> dict[str, object]:
    blockers: list[dict[str, str]] = []
    details: dict[str, object] = {}
    try:
        identity = bundle["identity_payload"]
        if canonical_sha256(identity) != bundle.get("corpus_hash"):
            raise ValueError("corpus identity does not derive from identity_payload")
        if bundle.get("corpus_uid") != "public_foundation_corpus_v2":
            raise ValueError("unexpected historical rescue corpus UID")
        if bundle.get("parent_corpus_identity_hash") != (
            "06ae88daf3c44be2c2bb96e51c939554e160b205138ec96f471adb5d82eee791"
        ):
            raise ValueError("V1 parent corpus identity drift")
        details["corpus_identity_hash"] = bundle["corpus_hash"]
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(_blocker("c1_rescue_identity_invalid", HISTORICAL_BUNDLE_PATH, exc))

    artifacts: dict[str, dict[str, object]] = {}
    try:
        evidence = bundle["evidence"]
        for name in (
            "source_qualification",
            "crosswalk_delta",
            "canonical_observation_blocks",
            "structured_action_catalog",
            "qualified_action_subset",
            "frozen_splits",
            "leakage_audit",
            "qc_ess_statistics",
            "storage_ledger",
        ):
            artifacts[name] = _validate_artifact_ref(repo_root, name, evidence[name])
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_evidence_reference_invalid", HISTORICAL_BUNDLE_PATH, exc))

    try:
        controlled = bundle["evidence"]["controlled_isolation"]
        controlled_path = safe_repo_path(repo_root, str(controlled["path"]))
        if sha256_file(controlled_path) != str(controlled["sha256"]):
            raise ValueError("controlled-isolation hash drift")
        if controlled.get("controlled_reachability") != 0:
            raise ValueError("controlled reachability is nonzero")
        parent = bundle["evidence"]["parent_v1_evidence_bundle"]
        parent_path = safe_repo_path(repo_root, str(parent["path"]))
        if sha256_file(parent_path) != str(parent["sha256"]):
            raise ValueError("parent V1 evidence hash drift")
        if sha256_file(repo_root / HISTORICAL_STUDENT_PATH) != HISTORICAL_STUDENT_SHA256:
            raise ValueError("V1 Student smoke evidence drift")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_parent_or_isolation_binding_invalid", HISTORICAL_BUNDLE_PATH, exc))

    block_rows: list[dict[str, object]] = []
    try:
        blocks_artifact = artifacts["canonical_observation_blocks"]
        block_rows = blocks_artifact["qualified_blocks"]
        if len(block_rows) != 8:
            raise ValueError("qualified V2 block count is not eight")
        if verify_large_block_payloads:
            manifests = [_validate_block(repo_root, row) for row in block_rows]
        else:
            manifests = [
                _load_json(safe_repo_path(repo_root, str(row["manifest_path"])))
                for row in block_rows
            ]
        control = next(
            manifest
            for manifest in manifests
            if manifest["canonical_uid"] == "pfc_arrayexpress_e_mtab_6772_controls_rescue"
        )
        independence = control.get("source_independence")
        if not isinstance(independence, Mapping):
            raise ValueError("E-MTAB control overlap declaration is absent")
        if independence.get("independence_claim_allowed") is not False:
            raise ValueError("same E-MTAB study matrix is incorrectly counted as independent")
        if independence.get("shared_source_accession") != "E-MTAB-6772":
            raise ValueError("E-MTAB underlying accession identity drift")
    except (KeyError, StopIteration, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_block_validation_failed", HISTORICAL_BLOCK_PATH, exc))

    try:
        crosswalk = artifacts["crosswalk_delta"]
        rows = read_tsv(safe_repo_path(repo_root, str(crosswalk["table"]["path"])))
        if len(rows) != int(crosswalk["record_count"]):
            raise ValueError("V2 crosswalk row count drift")
        if sum(row["canonical_value"] == "pubchem:5743" for row in rows) != 6:
            raise ValueError("V2 exact dexamethasone crosswalk must contain five CLUE plus one GEO row")
        if sum(row["canonical_value"] == "UBERON:0002421" for row in rows) != 2:
            raise ValueError("V2 hippocampus crosswalk anchor count drift")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_crosswalk_invalid", HISTORICAL_CROSSWALK_PATH, exc))

    try:
        actions = artifacts["qualified_action_subset"]
        if actions["record_count"] != 7124 or actions["catalog_count"] != 3:
            raise ValueError("V2 qualified action subset count drift")
        expected = canonical_sha256(
            {
                "corpus_hash": bundle["corpus_hash"],
                "qualified_catalogs": actions["qualified_catalogs"],
            }
        )
        if expected != actions["qualified_action_subset_hash"]:
            raise ValueError("V2 qualified action subset hash drift")
        for catalog in actions["qualified_catalogs"]:
            path = safe_repo_path(repo_root, str(catalog["path"]))
            if path.stat().st_size != int(catalog["bytes"]) or sha256_file(path) != catalog["sha256"]:
                raise ValueError(f"V2 action catalog drift: {catalog['asset_id']}")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_action_subset_invalid", HISTORICAL_ACTION_SUBSET_PATH, exc))

    try:
        splits = artifacts["frozen_splits"]
        _validate_splits(repo_root, splits)
        leakage = artifacts["leakage_audit"]
        if leakage.get("status") != "passed" or leakage.get("zero_overlap") is not True:
            raise ValueError("V2 leakage audit is not passed with zero overlap")
        if leakage.get("blocker_codes"):
            raise ValueError("V2 leakage audit retains blockers")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_split_or_leakage_invalid", HISTORICAL_SPLIT_PATH, exc))

    try:
        qc = artifacts["qc_ess_statistics"]
        if (
            qc["grade_ab_observation_count"] != 909484
            or qc["grade_ab_effective_sample_size"] != 7191.0
            or qc["grade_ab_action_count"] != 7124
        ):
            raise ValueError("V2 QC/ESS totals drift")
        if any(
            qc[key] != 0
            for key in (
                "synthetic_observation_count",
                "weak_calibration_observation_count",
                "context_prior_observation_count",
                "controlled_observation_count",
            )
        ):
            raise ValueError("forbidden observations entered V2 mainline")
        storage = artifacts["storage_ledger"]
        if storage.get("cap_status") != "passed" or storage.get("partial_scan") is not False:
            raise ValueError("V2 storage evidence is not complete and within cap")
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(_blocker("c1_rescue_qc_ess_storage_invalid", HISTORICAL_QC_PATH, exc))

    try:
        if (
            bundle.get("qualified_cross_source_anchor_count") != 1
            or bundle.get("qualified_perturbation_transfer_anchor_count") != 1
        ):
            raise ValueError("V2 rescue anchor counts drift")
        action_anchor = bundle["perturbation_transfer_anchor"]
        if action_anchor["v1_clue_action_count"] != 5 or action_anchor["v2_geo_action_count"] != 1:
            raise ValueError("V2 dexamethasone identity-qualified action count drift")
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(_blocker("c1_rescue_anchor_invalid", HISTORICAL_BUNDLE_PATH, exc))

    try:
        binding = bundle["c1_observation_binding"]
        manifest_path = safe_repo_path(repo_root, str(binding["manifest"]["path"]))
        table_path = safe_repo_path(repo_root, str(binding["row_table"]["path"]))
        if sha256_file(manifest_path) != binding["manifest"]["sha256"]:
            raise ValueError("V2 observation-binding manifest hash drift")
        if sha256_file(table_path) != binding["row_table"]["sha256"]:
            raise ValueError("V2 observation-binding table hash drift")
        manifest = _load_json(manifest_path)
        if (
            manifest["block_count"] != 8
            or manifest["biological_unit_count"] != 7191
            or manifest["observation_count"] != 909484
            or manifest["binding_hash"] != binding["binding_hash"]
        ):
            raise ValueError("V2 observation-binding totals drift")
        if block_rows and "frozen_splits" in artifacts:
            rebuilt_text, rebuilt_manifest = build_c1_observation_binding_artifacts(
                repo_root,
                bundle,
                block_rows,
                artifacts["frozen_splits"]["query_domain"],
                row_table_path=str(manifest["row_binding_table"]["path"]),
                verify_files=False,
            )
            rebuilt_manifest["derivation_policy"] = manifest["derivation_policy"]
            if rebuilt_text != table_path.read_text(encoding="utf-8"):
                raise ValueError("V2 observation-binding rows differ from live reconstruction")
            if serialize_binding_manifest(rebuilt_manifest) != manifest_path.read_text(encoding="utf-8"):
                raise ValueError("V2 observation-binding manifest differs from live reconstruction")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_rescue_observation_binding_invalid", HISTORICAL_BUNDLE_PATH, exc))

    return {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-evidence-validation-v1",
        "status": "valid_fail_closed" if not blockers else "blocked",
        "corpus_uid": bundle.get("corpus_uid", "unknown"),
        "corpus_identity_hash": bundle.get("corpus_hash", "unknown"),
        "parent_corpus_identity_hash": bundle.get("parent_corpus_identity_hash", "unknown"),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "details": details,
    }


def validate(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    return validate_rescue(repo_root)


def mutation_probe(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    bundle = _load_json(repo_root / HISTORICAL_BUNDLE_PATH)
    mutated = deepcopy(bundle)
    mutated["identity_payload"]["block_hashes"][
        "pfc_geo_gse262992_dex_control_rescue"
    ] = "0" * 64
    report = validate_payload(mutated, repo_root, verify_large_block_payloads=False)
    return {
        "status": "passed" if report["status"] == "blocked" else "failed",
        "injected_mutation": "v2_block_hash",
        "validator_status": report["status"],
        "detected_blocker_codes": [row["code"] for row in report["blockers"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    report = validate(REPO_ROOT)
    bundle = _load_json(safe_repo_path(REPO_ROOT, RESCUE_BUNDLE_PATH))
    mutated = deepcopy(bundle)
    mutated["identity_payload"]["block_hashes"] = {
        **mutated["identity_payload"]["block_hashes"],
        "semantic_mutation_probe": "0" * 64,
    }
    report["mutation_probe"] = {
        "status": "passed"
        if validate_rescue_payload(mutated, REPO_ROOT, verify_large_block_payloads=False)["status"] == "blocked"
        else "failed",
        "injected_mutation": "current_rescue_block_identity",
    }
    if report["mutation_probe"]["status"] != "passed":
        report["status"] = "blocked"
        report["blockers"].append(
            _blocker("c1_rescue_semantic_mutation_probe_failed", RESCUE_BUNDLE_PATH, "identity mutation was not detected")
        )
        report["blocker_count"] = len(report["blockers"])
    write_json(safe_repo_path(REPO_ROOT, args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "valid_fail_closed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
