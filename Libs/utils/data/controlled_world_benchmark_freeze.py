"""Content-addressed freeze and unopened access receipts for controlled-world benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from Libs.utils.data.controlled_world_benchmark_manifests import PARTITIONS, validate_benchmark_config
from Libs.utils.data.controlled_world_contracts import ControlledWorldContractError
from Libs.utils.data.controlled_world_runtime import sha256_file
from Libs.utils.data.trajectory_contracts import canonical_sha256, stable_uid

FREEZE_SCHEMA_VERSION = "neuronsifter-controlled-world-freeze-receipt-v1"
ACCESS_RECEIPT_SCHEMA_VERSION = "neuronsifter-controlled-world-access-receipt-v1"


def build_freeze_receipt(
    config: Mapping[str, object],
    *,
    repository_root: Path,
    manifests: Mapping[str, Mapping[str, object]],
    ood_manifest: Mapping[str, object],
    headroom: Mapping[str, object],
    qualification_report: Mapping[str, object],
) -> dict[str, object]:
    bench = validate_benchmark_config(config)
    components = {
        "training_manifest": manifests["training"].get("content_identity"),
        "development_manifest": manifests["development"].get("content_identity"),
        "calibration_manifest": manifests["calibration"].get("content_identity"),
        "test_manifest": manifests["test"].get("content_identity"),
        "ood_manifest": ood_manifest.get("content_identity"),
        "development_headroom": headroom.get("content_identity"),
        "qualification_report": qualification_report.get("qualification_uid"),
        "generator_identity": sha256_file(repository_root / "Libs/utils/data/controlled_world_runtime.py"),
        "benchmark_schema_identity": sha256_file(repository_root / "Libs/configs/schema/controlled_world_benchmark_manifest.schema.yaml"),
        "ood_schema_identity": sha256_file(repository_root / "Libs/configs/schema/controlled_world_ood_manifest.schema.yaml"),
        "freeze_schema_identity": sha256_file(repository_root / "Libs/configs/schema/controlled_world_freeze_receipt.schema.yaml"),
        "qualification_schema_identity": sha256_file(repository_root / "Libs/configs/schema/controlled_world_qualification_report.schema.yaml"),
        "statistic_plan": canonical_sha256(bench["statistical_policy"]),
        "seed_policy": canonical_sha256(bench["split_policy"]["seed_bases"]),
        "terminal_policy": canonical_sha256(bench["terminal_policy"]),
    }
    receipt = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "freeze_uid": stable_uid("controlled_world_benchmark_freeze", components),
        "protocol_version": bench["freeze_policy"]["protocol_version"],
        "first_use_policy": bench["freeze_policy"]["first_use_policy"],
        "evidence_class": "controlled_world_non_empirical",
        "content_addressed_components": components,
        "sealed_test_partitions": list(PARTITIONS),
        "test_manifest_uid": manifests["test"]["manifest_uid"],
        "result_access_before_freeze": 0,
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0},
        "claim_boundary": {"controlled_world_evidence_only": True, "empirical_evidence": False, "therapeutic_efficacy_claim": False},
    }
    _validate_receipt(receipt)
    return receipt


def build_access_receipts(freeze_receipt: Mapping[str, object], test_manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    receipts: dict[str, dict[str, object]] = {}
    for partition in PARTITIONS:
        worlds = [row for row in test_manifest["worlds"] if row["partition"] == partition]
        receipt = {
            "schema_version": ACCESS_RECEIPT_SCHEMA_VERSION,
            "access_receipt_uid": stable_uid("controlled_world_test_access_receipt", {"freeze_uid": freeze_receipt["freeze_uid"], "partition": partition, "worlds": worlds}),
            "freeze_uid": freeze_receipt["freeze_uid"],
            "partition": partition,
            "first_use_policy": "first_and_only_use",
            "partition_world_count": len(worlds),
            "result_opened": False,
            "formal_result_access_count": 0,
            "allowed_current_access": "freeze_identity_and_unopened_receipt_only",
            "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0},
        }
        _validate_receipt(receipt)
        receipts[partition] = receipt
    return receipts


def _validate_receipt(receipt: Mapping[str, object]) -> None:
    access = dict(receipt["access_counters"])
    if access != {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0}:
        raise ControlledWorldContractError("protected_access_nonzero", "Freeze/access receipt counters must remain zero.")
    if receipt.get("result_opened") is True or receipt.get("formal_result_access_count", 0) not in (0, None):
        raise ControlledWorldContractError("test_outcome_access_forbidden", "This round may not open sealed test outcomes.")
