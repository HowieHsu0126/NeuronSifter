"""Fail-closed C1 bindings for the frozen public transcriptomics corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from Libs.scripts.validators.validate_public_foundation_corpus_evidence import (
    validate_bundle_payload,
)
from Libs.scripts.validators.validate_c1_rescue_evidence import (
    validate_rescue_payload,
    validate_payload as validate_v2_bundle_payload,
)
from Libs.utils.data.public_foundation_corpus import CORE_FAMILIES, canonical_sha256, hash_path
from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file
from Libs.utils.data.foundation_state_c1_bindings import validate_c1_observation_binding


C1_SCHEMA_PATH = "Libs/configs/schema/foundation_state_c1.schema.json"
ACTION_CATALOG_FIELDS = (
    "asset_id",
    "path",
    "record_count",
    "bytes",
    "sha256",
    "source_id",
    "source_accession",
    "canonical_observation_uid",
    "role",
    "action_linkage_complete",
)
QUERY_DOMAIN_FIELDS = (
    "canonical_uid",
    "content_hash",
    "family_uid",
    "source_id",
    "source_accession",
    "publication_accession_group",
    "mirror_group",
    "biological_unit_uid",
    "regimen_uid",
    "target_program_proxy",
    "source_holdout_role",
    "split_role",
)


@dataclass(frozen=True)
class C1Unit:
    canonical_uid: str
    content_hash: str
    family_uid: str
    source_id: str
    source_accession: str
    publication_accession_group: str
    mirror_group: str
    biological_unit_uid: str
    regimen_uid: str
    target_program_proxy: str
    source_holdout_role: str
    split_role: str


@dataclass(frozen=True)
class C1LivePreflightAttestation:
    """In-process capability produced by one complete live corpus attestation."""

    repo_root: str
    contract_digest: str
    preflight_digest: str
    corpus_preflight_digest: str
    corpus_hash: str
    observation_binding_hash: str
    vocabulary_manifest_sha256: str


def _c1_contract_digest(config: Mapping[str, object]) -> str:
    contract = config.get("foundation_state_c1")
    if not isinstance(contract, Mapping):
        raise ValueError("foundation_state_c1 contract is missing")
    return canonical_sha256(contract)


def _blocker(code: str, path: object, message: object) -> dict[str, str]:
    return {"code": code, "path": str(path), "message": str(message)}


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _block_content_hash(manifest: Mapping[str, object]) -> str:
    records = manifest.get("block_files")
    if not isinstance(records, list) or not records:
        raise ValueError("Frozen block manifest has no block_files records")
    normalized: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Frozen block manifest contains a malformed block_files record")
        row = {
            "name": str(record["name"]),
            "bytes": int(record["bytes"]),
            "sha256": str(record["sha256"]),
        }
        if "file_count" in record:
            row["file_count"] = int(record["file_count"])
        normalized.append(row)
    return canonical_sha256(normalized)


def _block_payload_path(
    repo_root: Path,
    manifest_path: Path,
    record: Mapping[str, object],
) -> Path:
    if str(record.get("relative_path") or ""):
        return safe_repo_path(manifest_path.parent, str(record["relative_path"]))
    if str(record.get("path") or ""):
        return safe_repo_path(repo_root, str(record["path"]))
    raise ValueError(f"Block file {record.get('name')} has no path binding")


def _validate_block(
    repo_root: Path,
    block: Mapping[str, object],
    *,
    expected_block_hash: str,
    required_modality: str,
    required_roles: set[str],
    required_grades: set[str],
    verify_content_hashes: bool,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    uid = str(block.get("canonical_uid") or "<missing>")
    role = str(block.get("role") or "")
    grade = str(block.get("qc_grade") or "")
    if role not in required_roles:
        blockers.append(_blocker("c1_block_role_forbidden", uid, role or "missing role"))
    if grade not in required_grades:
        blockers.append(_blocker("c1_block_grade_forbidden", uid, grade or "missing grade"))
    if block.get("observation_eligible") is not True:
        blockers.append(_blocker("c1_block_not_observation_eligible", uid, "observation_eligible must be true"))
    if str(block.get("block_sha256") or "") != expected_block_hash:
        blockers.append(_blocker("c1_block_identity_mismatch", uid, block.get("block_sha256")))

    try:
        manifest_path = safe_repo_path(repo_root, str(block["manifest_path"]))
        if sha256_file(manifest_path) != str(block["manifest_sha256"]):
            raise ValueError("manifest sha256 drift")
        manifest = _load_json(manifest_path)
        identity_fields = (
            "canonical_uid",
            "family_id",
            "source_id",
            "source_accession",
            "role",
            "qc_grade",
            "observation_count",
            "effective_sample_size",
            "block_sha256",
        )
        for field in identity_fields:
            if manifest.get(field) != block.get(field):
                raise ValueError(f"manifest/evidence mismatch for {field}")
        axes = manifest.get("semantic_axes")
        if not isinstance(axes, Mapping):
            raise ValueError("semantic_axes must be a mapping")
        modality = axes.get("modality")
        if not isinstance(modality, Mapping) or modality.get("value") != required_modality:
            raise ValueError(f"block modality is not {required_modality}")
        records = manifest.get("block_files")
        if not isinstance(records, list) or not records:
            raise ValueError("block_files must be a non-empty list")
        if verify_content_hashes:
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError("malformed block_files record")
                payload_path = _block_payload_path(repo_root, manifest_path, record)
                count, byte_count, digest = hash_path(payload_path)
                if count != int(record.get("file_count", 1)):
                    raise ValueError(f"file count drift for {record['name']}")
                if byte_count != int(record["bytes"]):
                    raise ValueError(f"byte count drift for {record['name']}")
                if digest != str(record["sha256"]):
                    raise ValueError(f"content hash drift for {record['name']}")
        if _block_content_hash(manifest) != expected_block_hash:
            raise ValueError("block_sha256 does not derive from the frozen manifest")
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_frozen_block_drift", uid, exc))
    return blockers


def _normalized_action_catalogs(
    actions: Mapping[str, object],
    repo_root: Path,
    *,
    required_roles: set[str],
) -> list[dict[str, object]]:
    raw_catalogs = actions.get("qualified_catalogs")
    if not isinstance(raw_catalogs, list) or not raw_catalogs:
        raise ValueError("qualified action catalogs are missing")
    normalized: list[dict[str, object]] = []
    for raw in raw_catalogs:
        if not isinstance(raw, Mapping):
            raise ValueError("qualified action catalog contains a malformed record")
        row = {field: raw[field] for field in ACTION_CATALOG_FIELDS}
        row["asset_id"] = str(row["asset_id"])
        row["path"] = str(row["path"])
        row["record_count"] = int(row["record_count"])
        row["bytes"] = int(row["bytes"])
        row["sha256"] = str(row["sha256"])
        row["source_id"] = str(row["source_id"])
        row["source_accession"] = str(row["source_accession"])
        row["canonical_observation_uid"] = str(row["canonical_observation_uid"])
        row["role"] = str(row["role"])
        row["action_linkage_complete"] = bool(row["action_linkage_complete"])
        if row["role"] not in required_roles:
            raise ValueError(f"qualified action role is forbidden: {row['role']}")
        if row["action_linkage_complete"] is not True:
            raise ValueError(f"qualified action linkage is incomplete: {row['asset_id']}")
        action_path = safe_repo_path(repo_root, row["path"])
        if action_path.stat().st_size != row["bytes"]:
            raise ValueError(f"qualified action byte count drift: {row['asset_id']}")
        if sha256_file(action_path) != row["sha256"]:
            raise ValueError(f"qualified action content drift: {row['asset_id']}")
        normalized.append(row)
    normalized.sort(key=lambda row: str(row["asset_id"]))
    if len({str(row["asset_id"]) for row in normalized}) != len(normalized):
        raise ValueError("qualified action asset IDs are not unique")
    return normalized


def build_qualified_action_subset_manifest(
    bundle: Mapping[str, object],
    actions: Mapping[str, object],
    repo_root: Path,
    *,
    required_roles: set[str] | None = None,
) -> dict[str, object]:
    roles = required_roles or {"qualified_for_mainline"}
    catalogs = _normalized_action_catalogs(actions, repo_root, required_roles=roles)
    record_count = sum(int(row["record_count"]) for row in catalogs)
    if record_count != int(actions.get("qualified_record_count", -1)):
        raise ValueError("qualified action catalog counts do not sum to the audit total")
    hash_payload = {
        "corpus_hash": str(bundle["corpus_hash"]),
        "qualified_catalogs": catalogs,
    }
    return {
        "schema_version": (
            "neuronsifter-c1-qualified-action-subset-v2"
            if str(bundle["corpus_uid"]) == "public_foundation_corpus_rescue"
            else "neuronsifter-c1-qualified-action-subset-v1"
        ),
        **({"status": "passed"} if str(bundle["corpus_uid"]) == "public_foundation_corpus_rescue" else {}),
        "corpus_uid": str(bundle["corpus_uid"]),
        "corpus_hash": str(bundle["corpus_hash"]),
        "role": "qualified_for_mainline",
        "record_count": record_count,
        "catalog_count": len(catalogs),
        "qualified_catalogs": catalogs,
        "qualified_action_subset_hash": canonical_sha256(hash_payload),
    }


def validate_c1_contract(config: Mapping[str, object], repo_root: Path) -> list[dict[str, str]]:
    contract = config.get("foundation_state_c1")
    if not isinstance(contract, Mapping):
        return [_blocker("c1_contract_missing", "foundation_state_c1", "C1 config is absent")]
    blockers: list[dict[str, str]] = []
    try:
        schema_path = str(contract.get("schema_path") or C1_SCHEMA_PATH)
        schema = _load_json(safe_repo_path(repo_root, schema_path))
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(contract),
            key=lambda error: (list(error.absolute_path), error.message),
        )
        blockers.extend(
            _blocker(
                "c1_contract_schema_violation",
                ".".join(map(str, error.absolute_path)) or "$",
                error.message,
            )
            for error in errors
        )
    except (TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(
            _blocker(
                "c1_contract_schema_unavailable",
                contract.get("schema_path", C1_SCHEMA_PATH),
                exc,
            )
        )
    return blockers


def headline_preregistration_ready(contract: Mapping[str, object]) -> bool:
    evaluation = contract.get("evaluation")
    return (
        isinstance(evaluation, Mapping)
        and evaluation.get("source_shortcut_tolerance_status") == "preregistered"
        and isinstance(evaluation.get("source_shortcut_tolerance"), (int, float))
        and not isinstance(evaluation.get("source_shortcut_tolerance"), bool)
        and math.isfinite(float(evaluation["source_shortcut_tolerance"]))
    )


def _preflight_report(
    contract: Mapping[str, object],
    frozen: Mapping[str, object],
    blockers: Sequence[Mapping[str, str]],
    actual: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "corpus_uid": frozen.get("corpus_uid"),
        "corpus_hash": frozen.get("identity_hash"),
        "qualified_block_count": frozen.get("required_block_count"),
        "family_count": frozen.get("required_family_count"),
        "grade_ab_observation_count": frozen.get("required_grade_ab_observation_count"),
        "grade_ab_ess": frozen.get("required_grade_ab_ess"),
        "qualified_action_count": frozen.get("required_qualified_action_count"),
        "qualified_action_subset_hash": frozen.get("qualified_action_subset_hash"),
        "crosswalk": frozen.get("required_crosswalk"),
        "query_domain": frozen.get("required_query_domain"),
        "blocks": frozen.get("required_blocks"),
        "split_manifests": frozen.get("required_split_manifests"),
        "observation_binding": frozen.get("required_observation_binding"),
    }
    status = "passed" if not blockers else "blocked"
    shortcut_tolerance_ready = headline_preregistration_ready(contract)
    return {
        "schema_version": "neuronsifter-c1-foundation-corpus-preflight-v1",
        "status": status,
        "blocker_count": len(blockers),
        "blockers": list(blockers),
        "scope": frozen.get("scope"),
        "expected": expected,
        "actual": dict(actual),
        "corpus_uid": actual.get("corpus_uid"),
        "corpus_hash": actual.get("corpus_hash"),
        "qualified_block_count": actual.get("qualified_block_count"),
        "grade_ab_observation_count": actual.get("grade_ab_observation_count"),
        "grade_ab_ess": actual.get("grade_ab_ess"),
        "qualified_action_count": actual.get("qualified_action_count"),
        "observation_binding_hash": (
            actual.get("observation_binding", {}).get("binding_hash")
            if isinstance(actual.get("observation_binding"), Mapping)
            else None
        ),
        "training_allowed": status == "passed",
        "headline_training_allowed": status == "passed" and shortcut_tolerance_ready,
        "headline_training_blocker": (
            None
            if status != "passed" or shortcut_tolerance_ready
            else "source_shortcut_tolerance_not_preregistered"
        ),
        "g3_status": "not_evaluated",
        "g3_claim_promotion_allowed": False,
        "full_multimodal_claim_allowed": False,
        "cross_modal_prediction_claim_allowed": False,
    }


def build_c1_preflight_report(
    config: Mapping[str, object],
    repo_root: Path,
    *,
    bundle_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    contract = config.get("foundation_state_c1")
    if not isinstance(contract, Mapping):
        blockers = validate_c1_contract(config, repo_root)
        return _preflight_report({}, {}, blockers, {})
    frozen = contract.get("frozen_corpus")
    if not isinstance(frozen, Mapping):
        return _preflight_report(
            contract,
            {},
            [_blocker("c1_frozen_corpus_missing", "foundation_state_c1.frozen_corpus", "mapping required")],
            {},
        )
    blockers = validate_c1_contract(config, repo_root)
    actual: dict[str, object] = {
        "corpus_uid": None,
        "corpus_hash": None,
        "gate_status": None,
        "qualified_block_count": None,
        "family_count": None,
        "grade_ab_observation_count": None,
        "grade_ab_ess": None,
        "qualified_action_count": None,
        "qualified_action_subset_hash": None,
        "crosswalk": None,
        "query_domain": None,
        "blocks": None,
        "split_manifests": None,
        "observation_binding": None,
    }
    if blockers:
        return _preflight_report(contract, frozen, blockers, actual)

    try:
        bundle_path = safe_repo_path(repo_root, str(frozen["evidence_bundle_path"]))
        bundle = dict(bundle_payload) if bundle_payload is not None else _load_json(bundle_path)
        actual["corpus_uid"] = bundle.get("corpus_uid")
        actual["corpus_hash"] = bundle.get("corpus_hash")
        if str(bundle.get("corpus_uid")) == "public_foundation_corpus_rescue":
            evidence_report = validate_rescue_payload(bundle, repo_root)
        elif str(bundle.get("schema_version", "")).startswith(
            "neuronsifter-public-foundation-corpus-v2-"
        ):
            evidence_report = validate_v2_bundle_payload(bundle, repo_root)
        else:
            evidence_report = validate_bundle_payload(
                bundle,
                repo_root,
                allow_controlled_isolation_refresh=(
                    frozen.get("allowed_governance_reference_refresh")
                    == ["controlled_isolation"]
                ),
            )
        if evidence_report.get("status") != "valid_fail_closed":
            blockers.append(
                _blocker(
                    "c1_evidence_bundle_invalid",
                    frozen["evidence_bundle_path"],
                    evidence_report.get("blockers"),
                )
            )
        if bundle.get("corpus_uid") != frozen["corpus_uid"]:
            blockers.append(_blocker("c1_corpus_uid_mismatch", "corpus_uid", bundle.get("corpus_uid")))
        if bundle.get("corpus_hash") != frozen["identity_hash"]:
            blockers.append(_blocker("c1_corpus_identity_mismatch", "corpus_hash", bundle.get("corpus_hash")))
        derived_gates = bundle.get("derived_gates")
        if not isinstance(derived_gates, Mapping):
            raise ValueError("derived_gates must be a mapping")
        actual_gates = {
            str(key): value.get("status") if isinstance(value, Mapping) else value
            for key, value in derived_gates.items()
        }
        if str(bundle.get("corpus_uid")) == "public_foundation_corpus_rescue":
            v2_gate_names = {
                "G1": "iclr_v2.public_corpus.G1",
                "G2_reduced": "iclr_v2.public_corpus.G2",
                "G2_full_multimodal": "iclr_v2.public_corpus.G2_full_multimodal",
            }
            actual_gates = {
                v2_gate_names.get(key, key): value
                for key, value in actual_gates.items()
            }
        actual["gate_status"] = actual_gates
        if actual_gates != dict(frozen["required_gate_status"]):
            blockers.append(_blocker("c1_gate_status_mismatch", "derived_gates", actual_gates))

        evidence = bundle.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("evidence must be a mapping")
        crosswalk_ref = evidence.get("entity_unit_crosswalk") or evidence.get(
            "crosswalk_delta"
        )
        if not isinstance(crosswalk_ref, Mapping):
            raise ValueError("crosswalk evidence binding is missing")
        crosswalk = _load_json(
            safe_repo_path(repo_root, str(crosswalk_ref["path"]))
        )
        crosswalk_binding = crosswalk.get("crosswalk_table_binding") or crosswalk.get(
            "table"
        )
        if not isinstance(crosswalk_binding, Mapping):
            raise ValueError("crosswalk table binding is missing")
        crosswalk_actual = {
            "mapping_version": crosswalk.get("mapping_version"),
            "table_path": crosswalk_binding.get("path"),
            "table_sha256": crosswalk_binding.get("sha256"),
            "record_count": crosswalk.get("record_count"),
        }
        actual["crosswalk"] = crosswalk_actual
        if crosswalk_actual != dict(frozen["required_crosswalk"]):
            blockers.append(_blocker("c1_crosswalk_binding_mismatch", "entity_unit_crosswalk", crosswalk_actual))
        crosswalk_path = safe_repo_path(repo_root, str(crosswalk_binding["path"]))
        if sha256_file(crosswalk_path) != str(crosswalk_binding["sha256"]):
            blockers.append(_blocker("c1_crosswalk_table_hash_mismatch", crosswalk_binding["path"], "content drift"))

        observation_artifact = _load_json(
            safe_repo_path(repo_root, str(evidence["canonical_observation_blocks"]["path"]))
        )
        raw_blocks = observation_artifact.get("blocks")
        if not isinstance(raw_blocks, list):
            raise ValueError("canonical observation blocks must be a list")
        qualified = [
            row
            for row in raw_blocks
            if isinstance(row, Mapping) and row.get("observation_eligible") is True
        ]
        actual["qualified_block_count"] = len(qualified)
        actual_families = {str(row.get("family_id")) for row in qualified}
        actual["family_count"] = len(actual_families)
        actual_blocks = {
            str(row.get("canonical_uid")): str(row.get("block_sha256"))
            for row in qualified
        }
        actual["blocks"] = actual_blocks
        if len(qualified) != int(frozen["required_block_count"]):
            blockers.append(_blocker("c1_block_count_mismatch", "qualified_blocks", len(qualified)))
        if len(actual_families) != int(frozen["required_family_count"]) or actual_families != set(CORE_FAMILIES):
            blockers.append(_blocker("c1_family_set_mismatch", "qualified_blocks", sorted(actual_families)))
        expected_blocks = {str(key): str(value) for key, value in frozen["required_blocks"].items()}
        if actual_blocks != expected_blocks:
            blockers.append(_blocker("c1_frozen_block_set_mismatch", "qualified_blocks", actual_blocks))
        required_roles = {str(value) for value in frozen["required_roles"]}
        required_grades = {str(value) for value in frozen["required_qc_grades"]}
        for block in qualified:
            uid = str(block.get("canonical_uid"))
            blockers.extend(
                _validate_block(
                    repo_root,
                    block,
                    expected_block_hash=expected_blocks.get(uid, ""),
                    required_modality=str(frozen["required_modality"]),
                    required_roles=required_roles,
                    required_grades=required_grades,
                    verify_content_hashes=bool(frozen["verify_block_content_hashes"]),
                )
            )

        qc = _load_json(safe_repo_path(repo_root, str(evidence["qc_ess_statistics"]["path"])))
        actual_qc = (
            int(qc.get("grade_ab_observation_count", -1)),
            float(qc.get("grade_ab_effective_sample_size", -1)),
            int(
                qc.get(
                    "synthetic_teacher_observation_count",
                    qc.get("synthetic_observation_count", -1),
                )
            ),
        )
        actual["grade_ab_observation_count"] = actual_qc[0]
        actual["grade_ab_ess"] = actual_qc[1]
        expected_qc = (
            int(frozen["required_grade_ab_observation_count"]),
            float(frozen["required_grade_ab_ess"]),
            0,
        )
        if actual_qc != expected_qc:
            blockers.append(_blocker("c1_qc_ess_mismatch", "qc_ess_statistics", actual_qc))
        block_counts = (
            sum(int(row.get("observation_count", -1)) for row in qualified),
            sum(float(row.get("effective_sample_size", -1)) for row in qualified),
        )
        if block_counts != expected_qc[:2]:
            blockers.append(_blocker("c1_block_count_sum_mismatch", "qualified_blocks", block_counts))

        actions = _load_json(
            safe_repo_path(repo_root, str(evidence["structured_action_catalog"]["path"]))
        )
        actual["qualified_action_count"] = int(actions.get("qualified_record_count", -1))
        if actual["qualified_action_count"] != int(frozen["required_qualified_action_count"]):
            blockers.append(
                _blocker(
                    "c1_qualified_action_count_mismatch",
                    "qualified_record_count",
                    actual["qualified_action_count"],
                )
            )
        rebuilt_action_manifest = build_qualified_action_subset_manifest(
            bundle,
            actions,
            repo_root,
            required_roles=required_roles,
        )
        actual["qualified_action_subset_hash"] = rebuilt_action_manifest["qualified_action_subset_hash"]
        if actual["qualified_action_subset_hash"] != frozen["qualified_action_subset_hash"]:
            blockers.append(
                _blocker(
                    "c1_qualified_action_subset_hash_mismatch",
                    "qualified_action_subset_hash",
                    actual["qualified_action_subset_hash"],
                )
            )
        action_manifest_path = safe_repo_path(
            repo_root,
            str(frozen["qualified_action_subset_manifest_path"]),
        )
        stored_action_manifest = _load_json(action_manifest_path)
        action_schema = _load_json(
            safe_repo_path(repo_root, str(frozen["qualified_action_subset_schema_path"]))
        )
        Draft202012Validator.check_schema(action_schema)
        action_errors = sorted(
            Draft202012Validator(action_schema).iter_errors(stored_action_manifest),
            key=lambda error: (list(error.absolute_path), error.message),
        )
        blockers.extend(
            _blocker(
                "c1_action_subset_manifest_schema_violation",
                ".".join(map(str, error.absolute_path)) or "$",
                error.message,
            )
            for error in action_errors
        )
        if stored_action_manifest != rebuilt_action_manifest:
            blockers.append(
                _blocker(
                    "c1_action_subset_manifest_drift",
                    frozen["qualified_action_subset_manifest_path"],
                    "stored manifest does not equal the canonical reconstruction",
                )
            )

        splits = _load_json(safe_repo_path(repo_root, str(evidence["frozen_splits"]["path"])))
        query_binding = splits.get("query_domain")
        if not isinstance(query_binding, Mapping):
            raise ValueError("frozen split query-domain binding is missing")
        query_actual = {
            "path": query_binding.get("path"),
            "sha256": query_binding.get("sha256"),
            "record_count": query_binding.get("record_count"),
        }
        actual["query_domain"] = query_actual
        if query_actual != dict(frozen["required_query_domain"]):
            blockers.append(_blocker("c1_query_domain_binding_mismatch", "query_domain", query_actual))
        query_path = safe_repo_path(repo_root, str(query_binding["path"]))
        if sha256_file(query_path) != str(query_binding["sha256"]):
            blockers.append(_blocker("c1_query_domain_hash_mismatch", query_binding["path"], "content drift"))
        query_units = load_c1_units(query_path)
        if len(query_units) != int(query_binding["record_count"]):
            blockers.append(_blocker("c1_query_domain_count_mismatch", query_binding["path"], len(query_units)))

        binding_blockers, binding_actual = validate_c1_observation_binding(
            config,
            repo_root,
            bundle=bundle,
            qualified_blocks=qualified,
            # _validate_block above already hashes every frozen payload.  The
            # binding reconstruction still checks row/feature structure here.
            verify_files=False,
        )
        blockers.extend(binding_blockers)
        actual["observation_binding"] = binding_actual

        raw_manifests = splits.get("manifests")
        if not isinstance(raw_manifests, list):
            raise ValueError("frozen split manifests must be a list")
        actual_split_hashes: dict[str, str] = {}
        expected_split_hashes = {
            str(key): str(value) for key, value in frozen["required_split_manifests"].items()
        }
        for manifest in raw_manifests:
            if not isinstance(manifest, Mapping):
                raise ValueError("frozen split manifest contains a malformed record")
            split_uid = str(manifest.get("split_uid") or "")
            manifest_hash = str(manifest.get("manifest_hash") or "")
            actual_split_hashes[split_uid] = manifest_hash
            recomputed_manifest_hash = canonical_sha256(
                {key: value for key, value in manifest.items() if key != "manifest_hash"}
            )
            if manifest_hash != recomputed_manifest_hash:
                blockers.append(_blocker("c1_split_manifest_hash_invalid", split_uid, recomputed_manifest_hash))
            role_bindings = manifest.get("role_bindings")
            if not isinstance(role_bindings, Mapping) or set(role_bindings) != {"train", "validation", "test"}:
                blockers.append(_blocker("c1_split_role_binding_set_mismatch", split_uid, role_bindings))
                continue
            split_units: list[C1Unit] = []
            for role in ("train", "validation", "test"):
                binding = role_bindings[role]
                if not isinstance(binding, Mapping):
                    blockers.append(_blocker("c1_split_binding_malformed", f"{split_uid}:{role}", binding))
                    continue
                role_path = safe_repo_path(repo_root, str(binding["path"]))
                role_units = load_c1_split_units(query_path, role_path)
                role_count = len(role_units)
                split_units.extend(role_units)
                if sha256_file(role_path) != str(binding["sha256"]):
                    blockers.append(_blocker("c1_split_binding_hash_mismatch", f"{split_uid}:{role}", binding["path"]))
                if role_count != int(binding["record_count"]):
                    blockers.append(_blocker("c1_split_binding_count_mismatch", f"{split_uid}:{role}", role_count))
                role_uri = manifest.get(f"{role}_uids_uri")
                if role_uri is not None and str(role_uri) != str(binding["path"]):
                    blockers.append(_blocker("c1_split_binding_path_mismatch", f"{split_uid}:{role}", binding["path"]))
            split_uids = [unit.canonical_uid for unit in split_units]
            if len(split_uids) != len(set(split_uids)):
                blockers.append(
                    _blocker("c1_split_role_overlap", split_uid, "canonical UID occurs in multiple roles")
                )
            split_domain_hash = canonical_sha256(
                sorted(unit.content_hash for unit in split_units)
            )
            expected_domain_hash = manifest.get("corpus_hash") or manifest.get(
                "corpus_domain_hash"
            )
            if split_domain_hash != str(expected_domain_hash):
                blockers.append(
                    _blocker("c1_split_domain_hash_mismatch", split_uid, split_domain_hash)
                )
        actual["split_manifests"] = actual_split_hashes
        if actual_split_hashes != expected_split_hashes:
            blockers.append(_blocker("c1_frozen_split_set_mismatch", "split_manifests", actual_split_hashes))

        leakage = _load_json(safe_repo_path(repo_root, str(evidence["leakage_audit"]["path"])))
        overlaps = leakage.get("overlap_counts")
        if not isinstance(overlaps, Mapping):
            raise ValueError("leakage overlap_counts must be a mapping")
        if leakage.get("status") != "passed" or any(int(value) != 0 for value in overlaps.values()):
            blockers.append(_blocker("c1_leakage_gate_not_passed", "leakage_audit", overlaps))
        adversarial = leakage.get("adversarial_injection")
        if not isinstance(adversarial, Mapping) or adversarial.get("all_required_keys_detected") is not True:
            blockers.append(
                _blocker(
                    "c1_adversarial_leakage_missing",
                    "leakage_audit",
                    "required injected overlaps were not all detected",
                )
            )
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append(_blocker("c1_preflight_reconstruction_failed", "$", exc))

    return _preflight_report(contract, frozen, blockers, actual)


def _validated_c1_preflight_snapshots(
    config: Mapping[str, object], repo_root: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Return validated live C1 preflight snapshots without reloading them."""

    repo_root = repo_root.resolve()
    contract = config.get("foundation_state_c1")
    if not isinstance(contract, Mapping):
        raise ValueError("foundation_state_c1 contract is missing")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("foundation_state_c1 artifacts contract is missing")
    execution = contract.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("foundation_state_c1 execution contract is missing")
    # The corpus report is an input to the unified execution preflight.  Live
    # authorization must compare the persisted unified artifact against the
    # unified builder, using its matching schema.
    preflight_path = safe_repo_path(
        repo_root, str(execution["unified_preflight_path"])
    )
    stored = _load_json(preflight_path)
    # The unified preflight is a frozen execution authorization.  Runtime
    # loaders validate that authorization and its lower-level corpus evidence;
    # they do not recompute the corpus hash tree.  Recomputing it here both
    # defeats the declared node-local dependency contract and incorrectly
    # compares a historical regression source snapshot with the current BASELINE_FIRST
    # candidate identity.  The BASELINE_FIRST runner binds the latter separately.
    from Libs.utils.data.raw_data_registry import load_config

    config_path = repo_root / "Libs/configs/config_stack.yaml"
    resolved_config = load_config(config_path)
    if canonical_sha256(config) != canonical_sha256(resolved_config):
        raise ValueError("C1 live config differs from the frozen config stack")
    schema_path = safe_repo_path(
        repo_root, str(artifacts["unified_preflight_schema_path"])
    )
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(stored),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        raise ValueError(
            "C1 persisted preflight violates its schema: "
            + "; ".join(error.message for error in errors)
        )

    corpus_path = safe_repo_path(repo_root, str(artifacts["preflight_output_path"]))
    corpus = _load_json(corpus_path)
    corpus_schema = _load_json(
        safe_repo_path(repo_root, str(artifacts["preflight_schema_path"]))
    )
    Draft202012Validator.check_schema(corpus_schema)
    corpus_errors = sorted(
        Draft202012Validator(corpus_schema).iter_errors(corpus),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if corpus_errors:
        raise ValueError(
            "C1 corpus preflight violates its schema: "
            + "; ".join(error.message for error in corpus_errors)
        )

    frozen = contract.get("frozen_corpus")
    if not isinstance(frozen, Mapping):
        raise ValueError("foundation_state_c1 frozen corpus contract is missing")
    expected_bindings = {
        "corpus_uid": frozen.get("corpus_uid"),
        "corpus_identity_hash": frozen.get("identity_hash"),
        "block_hashes": frozen.get("required_blocks"),
        "split_hashes": frozen.get("required_split_manifests"),
        "action_subset_hash": frozen.get("qualified_action_subset_hash"),
    }
    observed_bindings = {key: stored.get(key) for key in expected_bindings}
    if observed_bindings != expected_bindings:
        raise ValueError("C1 unified preflight differs from the frozen C1 contract")
    if (
        corpus.get("status") != "passed"
        or corpus.get("training_allowed") is not True
        or corpus.get("corpus_hash") != stored.get("corpus_identity_hash")
        or corpus.get("qualified_block_count") != len(expected_bindings["block_hashes"])
    ):
        raise ValueError("C1 corpus preflight does not support the frozen unified preflight")
    if stored.get("status") != "passed" or stored.get("training_allowed") is not True:
        raise ValueError(
            "C1 live corpus preflight does not allow training: "
            + str(stored.get("blockers"))
        )
    live_report = dict(stored)
    live_report["corpus_hash"] = corpus["corpus_hash"]
    live_report["observation_binding_hash"] = corpus["observation_binding_hash"]
    live_report["scope"] = corpus.get("scope", live_report.get("scope"))
    return live_report, dict(stored), dict(corpus)


def require_live_c1_preflight(
    config: Mapping[str, object], repo_root: Path
) -> dict[str, object]:
    """Rebuild and verify the persisted C1 preflight before any data access."""

    return _validated_c1_preflight_snapshots(config, repo_root)[0]


def attest_live_c1_preflight(
    config: Mapping[str, object], repo_root: Path
) -> C1LivePreflightAttestation:
    """Perform one complete live check and return a process-local capability."""

    repo_root = repo_root.resolve()
    _, stored, corpus = _validated_c1_preflight_snapshots(config, repo_root)
    contract = config["foundation_state_c1"]
    input_contract = contract["encoder"]["input_contract"]
    vocabulary_path = safe_repo_path(
        repo_root, str(input_contract["vocabulary_manifest_path"])
    )
    vocabulary_hash = str(input_contract["vocabulary_manifest_sha256"])
    if sha256_file(vocabulary_path) != vocabulary_hash:
        raise ValueError("C1 vocabulary manifest hash drift during live attestation")
    return C1LivePreflightAttestation(
        repo_root=repo_root.as_posix(),
        contract_digest=_c1_contract_digest(config),
        preflight_digest=canonical_sha256(stored),
        corpus_preflight_digest=canonical_sha256(corpus),
        corpus_hash=str(corpus["corpus_hash"]),
        observation_binding_hash=str(corpus["observation_binding_hash"]),
        vocabulary_manifest_sha256=vocabulary_hash,
    )


def validate_c1_preflight_attestation(
    config: Mapping[str, object],
    repo_root: Path,
    attestation: C1LivePreflightAttestation,
) -> None:
    """Validate a process-local attestation without rescanning large stores."""

    if type(attestation) is not C1LivePreflightAttestation:
        raise TypeError("C1 preflight attestation must be a live in-process capability")
    repo_root = repo_root.resolve()
    if attestation.repo_root != repo_root.as_posix():
        raise ValueError("C1 preflight attestation repository root drift")
    if attestation.contract_digest != _c1_contract_digest(config):
        raise ValueError("C1 preflight attestation contract drift")
    contract = config["foundation_state_c1"]
    frozen = contract["frozen_corpus"]
    if (
        attestation.corpus_hash != frozen["identity_hash"]
        or attestation.observation_binding_hash
        != frozen["required_observation_binding"]["binding_hash"]
    ):
        raise ValueError("C1 preflight attestation frozen identity drift")
    _, stored, corpus = _validated_c1_preflight_snapshots(config, repo_root)
    if (
        canonical_sha256(stored) != attestation.preflight_digest
        or stored.get("status") != "passed"
        or stored.get("training_allowed") is not True
    ):
        raise ValueError("C1 persisted preflight changed after live attestation")
    if (
        canonical_sha256(corpus) != attestation.corpus_preflight_digest
        or corpus.get("status") != "passed"
        or corpus.get("training_allowed") is not True
        or corpus.get("corpus_hash") != attestation.corpus_hash
        or corpus.get("observation_binding_hash") != attestation.observation_binding_hash
    ):
        raise ValueError("C1 corpus preflight changed after live attestation")
    input_contract = contract["encoder"]["input_contract"]
    configured_vocabulary_hash = str(input_contract["vocabulary_manifest_sha256"])
    if configured_vocabulary_hash != attestation.vocabulary_manifest_sha256:
        raise ValueError("C1 preflight attestation vocabulary identity drift")
    vocabulary_path = safe_repo_path(
        repo_root, str(input_contract["vocabulary_manifest_path"])
    )
    if sha256_file(vocabulary_path) != configured_vocabulary_hash:
        raise ValueError("C1 vocabulary manifest changed after live attestation")


def count_query_domain_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUERY_DOMAIN_FIELDS:
            raise ValueError(f"Unexpected C1 query-domain fields: {reader.fieldnames}")
        return sum(1 for _ in reader)


def _read_c1_units(path: Path, *, require_all_families: bool) -> list[C1Unit]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUERY_DOMAIN_FIELDS:
            raise ValueError(f"Unexpected C1 query-domain fields: {reader.fieldnames}")
        units = [C1Unit(**{field: str(row[field]) for field in QUERY_DOMAIN_FIELDS}) for row in reader]
    if not units:
        raise ValueError("C1 unit table is empty")
    if len({unit.canonical_uid for unit in units}) != len(units):
        raise ValueError("C1 unit table contains duplicate canonical UIDs")
    if any(not unit.biological_unit_uid or not unit.family_uid for unit in units):
        raise ValueError("C1 unit table contains an empty family or biological-unit identity")
    if require_all_families and {unit.family_uid for unit in units} != set(CORE_FAMILIES):
        raise ValueError("C1 query domain does not contain the frozen three-family set")
    return units


def load_c1_units(path: Path) -> list[C1Unit]:
    return _read_c1_units(path, require_all_families=True)


def load_c1_split_units(query_domain_path: Path, role_binding_path: Path) -> list[C1Unit]:
    query_units = {unit.canonical_uid: unit for unit in load_c1_units(query_domain_path)}
    role_units = _read_c1_units(role_binding_path, require_all_families=False)
    for role_unit in role_units:
        query_unit = query_units.get(role_unit.canonical_uid)
        if query_unit is None:
            raise ValueError(f"Split unit is outside the frozen query domain: {role_unit.canonical_uid}")
        for field in QUERY_DOMAIN_FIELDS:
            if field == "split_role":
                continue
            if getattr(query_unit, field) != getattr(role_unit, field):
                raise ValueError(f"Split/query mismatch for {role_unit.canonical_uid}:{field}")
        if role_unit.split_role not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid C1 split role: {role_unit.split_role}")
    return role_units


def family_balanced_unit_weights(units: Sequence[C1Unit]) -> list[float]:
    if not units:
        raise ValueError("Cannot weight an empty C1 unit collection")
    family_units: dict[str, dict[str, list[int]]] = {}
    biological_unit_families: dict[str, str] = {}
    for index, unit in enumerate(units):
        existing_family = biological_unit_families.setdefault(unit.biological_unit_uid, unit.family_uid)
        if existing_family != unit.family_uid:
            raise ValueError("A biological unit cannot belong to multiple C1 families")
        family_units.setdefault(unit.family_uid, {}).setdefault(unit.biological_unit_uid, []).append(index)
    family_count = len(family_units)
    weights = [0.0] * len(units)
    for unit_groups in family_units.values():
        unit_count = len(unit_groups)
        for indices in unit_groups.values():
            row_weight = 1.0 / (family_count * unit_count * len(indices))
            for index in indices:
                weights[index] = row_weight
    return weights


def aggregate_unit_observations(values: Iterable[Sequence[float]]) -> list[float]:
    rows = [list(map(float, row)) for row in values]
    if not rows:
        raise ValueError("Cannot aggregate an empty biological unit")
    feature_count = len(rows[0])
    if feature_count == 0 or any(len(row) != feature_count for row in rows):
        raise ValueError("Within-unit observations must have one non-empty feature shape")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("Within-unit observations must be finite")
    return [sum(row[index] for row in rows) / len(rows) for index in range(feature_count)]


def deterministic_nested_unit_subset(
    units: Sequence[C1Unit],
    fraction: float,
    *,
    seed: str,
) -> list[C1Unit]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("C1 scaling fraction must be in (0, 1]")
    if not units:
        raise ValueError("Cannot scale an empty C1 unit collection")
    family_groups: dict[str, dict[str, list[C1Unit]]] = {}
    for unit in units:
        family_groups.setdefault(unit.family_uid, {}).setdefault(unit.biological_unit_uid, []).append(unit)
    selected: list[C1Unit] = []
    for family_uid in sorted(family_groups):
        biological_units = family_groups[family_uid]
        ordered_uids = sorted(
            biological_units,
            key=lambda uid: hashlib.sha256(f"{seed}|{family_uid}|{uid}".encode("utf-8")).hexdigest(),
        )
        selected_count = max(1, math.ceil(len(ordered_uids) * fraction))
        for biological_uid in ordered_uids[:selected_count]:
            selected.extend(biological_units[biological_uid])
    return selected
