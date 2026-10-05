"""Validate and re-derive the Phase 1 public-corpus evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import CORE_FAMILIES
from Libs.utils.data.public_foundation_evidence import (
    EVIDENCE_KEYS,
    canonical_hash,
    family_assessment_from_rows,
    safe_repo_path,
    sha256_file,
)
from Libs.utils.data.public_foundation_storage import storage_gate_status
from Libs.utils.data.raw_data_registry import read_tsv, write_json
from Libs.scripts.validators.validate_public_foundation_corpus_crosswalk import validate as validate_crosswalk


DEFAULT_BUNDLE_PATH = "Output/audits/public_foundation_corpus_evidence_bundle.json"
DEFAULT_SCHEMA_PATH = "Libs/configs/schema/public_foundation_corpus_evidence.schema.json"
CONTRACT_PATH = "Libs/configs/data/public_foundation_corpus.yaml"
CONTROLLED_ISOLATION_KEY = "controlled_isolation"
CONTROLLED_ISOLATION_SCHEMA_VERSION = "neuronsifter-public-mainline-isolation-audit-v1"
GOVERNANCE_REFRESH_POLICY = "c1-controlled-isolation-refresh-v1"


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _schema(repo_root: Path, path: str = DEFAULT_SCHEMA_PATH) -> dict[str, object]:
    payload = _load_json(safe_repo_path(repo_root, path))
    if not isinstance(payload, dict):
        raise TypeError(f"Schema must be a JSON object: {path}")
    Draft202012Validator.check_schema(payload)
    return payload


def _ref_path(repo_root: Path, value: object) -> Path:
    path = safe_repo_path(repo_root, str(value))
    relative = path.relative_to(repo_root).as_posix()
    if not relative.startswith("Output/audits/") or path.suffix != ".json":
        raise ValueError(f"Evidence reference is outside Output/audits JSON scope: {relative}")
    return path


def _load_ref_artifact(
    repo_root: Path,
    ref: Mapping[str, object],
    *,
    verify_hash: bool = True,
) -> tuple[Path, dict[str, object]]:
    path = _ref_path(repo_root, ref.get("path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256_file(path)
    if verify_hash and actual_sha != ref.get("sha256"):
        raise ValueError(f"Evidence hash mismatch: {path}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Evidence artifact must be a JSON object: {path}")
    declared_count = payload.get("record_count")
    if isinstance(declared_count, int) and declared_count != int(ref.get("record_count", -1)):
        raise ValueError(f"Evidence record count mismatch: {path}")
    return path, payload


def _unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _expected_g1(
    refs: Mapping[str, Mapping[str, object]],
    crosswalk: Mapping[str, object],
    isolation: Mapping[str, object],
    storage: Mapping[str, object],
) -> list[str]:
    blockers: list[str] = []
    for key in ("source_qualification", "legacy_disposition", "entity_unit_crosswalk"):
        if refs[key].get("status") != "passed":
            blockers.append(f"{key}_not_passed")
    if isolation.get("status") != "pass":
        blockers.append("controlled_reachability_nonzero")
    if int(crosswalk.get("critical_mapping_conflict_count", 0)) != 0:
        blockers.append("critical_mapping_conflicts_nonzero")
    if int(isolation.get("controlled_reachability_count", -1)) != 0:
        blockers.append("controlled_reachability_nonzero")
    storage_status = storage_gate_status(storage)
    if storage_status == "pending_revalidation":
        blockers.append("storage_cap_pending_revalidation")
    elif storage_status == "failed_capacity":
        blockers.append("storage_cap_exceeded")
    return _unique(blockers)


def _expected_g2(
    refs: Mapping[str, Mapping[str, object]],
    families: Mapping[str, Mapping[str, object]],
    *,
    g1_blockers: list[str],
    leakage: Mapping[str, object],
    storage: Mapping[str, object],
) -> list[str]:
    blockers: list[str] = []
    if g1_blockers:
        blockers.append("public_corpus_g1_not_passed")
    for family_id in CORE_FAMILIES:
        assessment = families[family_id]
        if assessment.get("status") != "qualified":
            blockers.extend(f"{family_id}:{code}" for code in assessment.get("blocker_codes", []))
    for key, ref in refs.items():
        if key in {"legacy_disposition", "source_qualification", "entity_unit_crosswalk", "controlled_isolation", "leakage_audit"}:
            continue
        if key == "storage_ledger" and storage_gate_status(storage) == "pending_revalidation":
            continue
        if ref.get("status") != "passed":
            blockers.append(f"{key}_not_passed")
    if leakage.get("status") != "passed":
        blockers.append("leakage_audit_not_passed")
    storage_status = storage_gate_status(storage)
    if storage_status == "pending_revalidation":
        blockers.append("storage_cap_pending_revalidation")
    elif storage_status == "failed_capacity":
        blockers.append("storage_cap_exceeded")
    return _unique(blockers)


def validate_bundle_payload(
    payload: object,
    repo_root: Path = REPO_ROOT,
    *,
    schema_path: str = DEFAULT_SCHEMA_PATH,
    verify_storage: bool = False,
    allow_controlled_isolation_refresh: bool = False,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    blockers: list[dict[str, str]] = []
    governance_reference_refresh: dict[str, object] = {"status": "not_requested"}
    schema = _schema(repo_root, schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda error: (list(error.absolute_path), error.message))
    for error in errors:
        blockers.append({"code": "json_schema_violation", "path": ".".join(map(str, error.absolute_path)) or "$", "message": error.message})
    if errors or not isinstance(payload, dict):
        return {"status": "blocked", "blocker_count": len(blockers), "blockers": blockers}

    try:
        contract_sha = sha256_file(safe_repo_path(repo_root, CONTRACT_PATH))
        if payload["contract"]["sha256"] != contract_sha:
            blockers.append({"code": "contract_hash_mismatch", "path": "contract.sha256", "message": "Contract bytes changed after bundle creation."})
        expected_hash = canonical_hash({key: value for key, value in payload.items() if key != "corpus_hash"})
        if payload["corpus_hash"] != expected_hash:
            blockers.append({"code": "corpus_hash_mismatch", "path": "corpus_hash", "message": "Corpus hash is not the canonical hash of the bundle payload."})
        refs = payload["evidence"]
        artifacts: dict[str, dict[str, object]] = {}
        # C1 may refresh only this governance scan after adding its public config;
        # dataset, split, action, QC, and crosswalk references remain hash-bound.
        for key in EVIDENCE_KEYS:
            refresh_allowed = (
                allow_controlled_isolation_refresh
                and key == CONTROLLED_ISOLATION_KEY
            )
            path, artifacts[key] = _load_ref_artifact(
                repo_root,
                refs[key],
                verify_hash=not refresh_allowed,
            )
            if refresh_allowed:
                current_sha = sha256_file(path)
                isolation_artifact = artifacts[key]
                current_record_count = isolation_artifact.get("record_count")
                if not isinstance(current_record_count, int):
                    route_findings = isolation_artifact.get("route_findings")
                    current_record_count = (
                        len(route_findings) if isinstance(route_findings, list) else -1
                    )
                refresh_valid = (
                    refs[key].get("status") == "passed"
                    and isolation_artifact.get("schema_version")
                    == CONTROLLED_ISOLATION_SCHEMA_VERSION
                    and isolation_artifact.get("status") == "pass"
                    and int(
                        isolation_artifact.get("controlled_reachability_count", -1)
                    )
                    == 0
                    and current_record_count == int(refs[key].get("record_count", -1))
                )
                governance_reference_refresh = {
                    "status": "applied",
                    "policy": GOVERNANCE_REFRESH_POLICY,
                    "reference_key": CONTROLLED_ISOLATION_KEY,
                    "frozen_sha256": refs[key].get("sha256"),
                    "current_sha256": current_sha,
                    "current_schema_version": isolation_artifact.get("schema_version"),
                    "current_audit_status": isolation_artifact.get("status"),
                    "current_controlled_reachability_count": isolation_artifact.get(
                        "controlled_reachability_count"
                    ),
                    "record_count_matches_frozen_reference": (
                        current_record_count == int(refs[key].get("record_count", -1))
                    ),
                    "corpus_identity_unchanged": (
                        payload.get("corpus_hash") == expected_hash
                    ),
                }
                if not refresh_valid:
                    blockers.append(
                        {
                            "code": "controlled_isolation_refresh_invalid",
                            "path": "evidence.controlled_isolation",
                            "message": (
                                "Governance refresh requires the public-mainline isolation "
                                "audit v1 to pass with zero controlled reachability."
                            ),
                        }
                    )
        crosswalk = artifacts["entity_unit_crosswalk"]
        crosswalk_report = validate_crosswalk(repo_root)
        if crosswalk_report.get("status") != "passed":
            blockers.append({"code": "crosswalk_contract_not_passed", "path": "evidence.entity_unit_crosswalk", "message": str(crosswalk_report.get("blockers", []))})
        isolation = artifacts["controlled_isolation"]
        observations = artifacts["canonical_observation_blocks"]
        splits = artifacts["frozen_splits"]
        leakage = artifacts["leakage_audit"]
        storage = artifacts["storage_ledger"]
        blocks = observations.get("blocks", [])
        if not isinstance(blocks, list):
            raise ValueError("canonical observation blocks must contain a list")
        family_assessments = {
            family_id: family_assessment_from_rows(
                family_id,
                blocks,
                query_domain_status=str(splits.get("query_domain_status", "not_frozen")),
                nested_split_status="frozen" if splits.get("frozen") is True else "not_frozen",
                leakage_status=str(leakage.get("status", "not_passed")),
            )
            for family_id in CORE_FAMILIES
        }
        if family_assessments != payload["family_assessments"]:
            blockers.append({"code": "family_assessment_recomputed_mismatch", "path": "family_assessments", "message": "Family qualification was not derived from the canonical block rows."})
        split_schema = _load_json(safe_repo_path(repo_root, "Docs/design/iclr/schemas/split_manifest.schema.json"))
        split_validator = Draft202012Validator(split_schema, format_checker=FormatChecker())
        recomputed_overlap: dict[str, int] = {
            str(key): 0 for key in leakage.get("required_zero_overlap_keys", [])
        }
        for index, manifest in enumerate(splits.get("manifests", [])):
            design_manifest = {
                key: manifest[key]
                for key in (
                    "schema_version", "split_uid", "split_type", "corpus_hash",
                    "group_axes", "train_uids_uri", "validation_uids_uri",
                    "test_uids_uri", "holdout_uids_uri", "manifest_hash",
                    "leakage_audit_uri", "frozen",
                )
            }
            for error in split_validator.iter_errors(design_manifest):
                blockers.append({"code": "split_manifest_schema_violation", "path": f"evidence.frozen_splits.manifests.{index}", "message": error.message})
            role_rows: dict[str, list[dict[str, str]]] = {}
            for role, binding in manifest.get("role_bindings", {}).items():
                uid_path = safe_repo_path(repo_root, str(binding.get("path", "")))
                if not uid_path.is_file() or sha256_file(uid_path) != binding.get("sha256"):
                    blockers.append({"code": "split_uid_binding_mismatch", "path": f"evidence.frozen_splits.manifests.{index}.{role}", "message": "Frozen split UID binding is missing or hash-drifted."})
                    continue
                rows = read_tsv(uid_path)
                role_rows[role] = rows
                if len(rows) != int(binding.get("record_count", -1)):
                    blockers.append({"code": "split_uid_count_mismatch", "path": f"evidence.frozen_splits.manifests.{index}.{role}", "message": "Frozen split UID row count drifted."})
            if set(role_rows) == {"train", "validation", "test"}:
                for key in manifest.get("applicable_zero_overlap_keys", []):
                    role_sets = {
                        role: {str(row.get(key) or "") for row in rows if str(row.get(key) or "")}
                        for role, rows in role_rows.items()
                    }
                    if key in {"source_holdout_role", "publication_accession_group", "family_uid"}:
                        overlap_values = (role_sets["train"] | role_sets["validation"]) & role_sets["test"]
                    else:
                        overlap_values = (
                            (role_sets["train"] & role_sets["validation"])
                            | (role_sets["train"] & role_sets["test"])
                            | (role_sets["validation"] & role_sets["test"])
                        )
                    recomputed_overlap[str(key)] = recomputed_overlap.get(str(key), 0) + len(overlap_values)
        if recomputed_overlap != leakage.get("overlap_counts"):
            blockers.append({"code": "leakage_overlap_recomputation_mismatch", "path": "evidence.leakage_audit.overlap_counts", "message": str(recomputed_overlap)})
        if any(int(value) != 0 for value in leakage.get("overlap_counts", {}).values()):
            blockers.append({"code": "canonical_observation_overlap", "path": "evidence.leakage_audit", "message": str(leakage.get("overlap_counts"))})
        if leakage.get("adversarial_injection", {}).get("all_required_keys_detected") is not True:
            blockers.append({"code": "adversarial_leakage_test_missing", "path": "evidence.leakage_audit", "message": "Adversarial duplicate injection was not detected."})
        if verify_storage:
            from Libs.scripts.validators.validate_public_foundation_corpus_storage_ledger import (
                build_report as validate_storage_ledger,
            )

            storage_report = validate_storage_ledger(
                str(refs["storage_ledger"]["path"]),
                repo_root=repo_root,
                verify_current_scan=True,
            )
            if storage_report["status"] != "passed":
                blockers.append({"code": "storage_ledger_revalidation_failed", "path": "evidence.storage_ledger", "message": str(storage_report.get("blockers", []))})
        expected_g1_blockers = _expected_g1(refs, crosswalk, isolation, storage)
        expected_g2_blockers = _expected_g2(
            refs,
            family_assessments,
            g1_blockers=expected_g1_blockers,
            leakage=leakage,
            storage=storage,
        )
        gates = payload["derived_gates"]
        g1_status = "passed" if not expected_g1_blockers else "not_passed"
        g2_status = "PASS_WITH_SCOPE_LIMITATION" if not expected_g2_blockers else "not_passed"
        if gates["iclr_v2.public_corpus.G1"]["status"] != g1_status or gates["iclr_v2.public_corpus.G1"]["blocker_codes"] != expected_g1_blockers:
            blockers.append({"code": "g1_derivation_mismatch", "path": "derived_gates.iclr_v2.public_corpus.G1", "message": "G1 status is not derived from evidence."})
        if gates["iclr_v2.public_corpus.G2"]["status"] != g2_status or gates["iclr_v2.public_corpus.G2"]["blocker_codes"] != expected_g2_blockers:
            blockers.append({"code": "g2_derivation_mismatch", "path": "derived_gates.iclr_v2.public_corpus.G2", "message": "G2 status is not derived from evidence."})
        if gates["iclr_v2.public_corpus.G2_full_multimodal"]["status"] != "not_passed":
            blockers.append({"code": "full_multimodal_gate_open", "path": "derived_gates.iclr_v2.public_corpus.G2_full_multimodal", "message": "Full multimodal extension must remain closed."})
    except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as exc:
        blockers.append({"code": "evidence_reconstruction_failed", "path": "$", "message": str(exc)})

    gate_status = payload.get("derived_gates", {})
    return {
        "status": "valid_fail_closed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "corpus_uid": payload.get("corpus_uid"),
        "corpus_hash": payload.get("corpus_hash"),
        "gate_status": {key: value.get("status") for key, value in gate_status.items() if isinstance(value, Mapping)},
        "governance_reference_refresh": governance_reference_refresh,
    }


def build_report(
    bundle_path: str = DEFAULT_BUNDLE_PATH,
    *,
    repo_root: Path = REPO_ROOT,
    schema_path: str = DEFAULT_SCHEMA_PATH,
    verify_storage: bool = False,
    allow_controlled_isolation_refresh: bool = False,
) -> dict[str, object]:
    path = safe_repo_path(repo_root, bundle_path)
    payload = _load_json(path)
    return validate_bundle_payload(
        payload,
        repo_root,
        schema_path=schema_path,
        verify_storage=verify_storage,
        allow_controlled_isolation_refresh=allow_controlled_isolation_refresh,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Phase 1 public foundation corpus evidence bundle.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output", default="Output/audits/public_foundation_corpus_evidence_validation.json")
    parser.add_argument("--verify-storage", action="store_true")
    parser.add_argument(
        "--allow-controlled-isolation-refresh",
        action="store_true",
        help=(
            "Allow only the C1-governed public-mainline isolation audit refresh; "
            "all corpus-bearing evidence remains hash-bound."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.bundle,
        repo_root=REPO_ROOT,
        schema_path=args.schema,
        verify_storage=args.verify_storage,
        allow_controlled_isolation_refresh=args.allow_controlled_isolation_refresh,
    )
    write_json(safe_repo_path(REPO_ROOT, args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "valid_fail_closed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
