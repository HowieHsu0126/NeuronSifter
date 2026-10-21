"""Validate controlled neural-world benchmark qualification artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.controlled_world_benchmark_freeze import build_access_receipts, build_freeze_receipt  # noqa: E402
from Libs.utils.data.controlled_world_benchmark_manifests import OOD_CLASSES, PARTITIONS, validate_benchmark_config, validate_lineage_separation  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402
from Libs.utils.data.trajectory_contracts import canonical_sha256, stable_uid  # noqa: E402

SCHEMAS = {
    "benchmark": "Libs/configs/schema/controlled_world_benchmark_manifest.schema.yaml",
    "ood": "Libs/configs/schema/controlled_world_ood_manifest.schema.yaml",
    "freeze": "Libs/configs/schema/controlled_world_freeze_receipt.schema.yaml",
    "qualification": "Libs/configs/schema/controlled_world_qualification_report.schema.yaml",
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema(repository_root: Path, key: str) -> Draft202012Validator:
    schema_path = repository_root / SCHEMAS[key]
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _resolve(root: Path, raw: object) -> Path:
    return root / Path(str(raw))


def _durable_release_paths(artifact_root: Path, paths: dict[str, object]) -> list[Path]:
    """Return the complete, deliberately closed release inventory."""
    release = [
        _resolve(artifact_root, paths[key])
        for key in (
            "training_manifest",
            "development_manifest",
            "calibration_manifest",
            "test_manifest",
            "ood_manifest",
            "development_headroom",
            "freeze_receipt",
            "qualification_audit",
            "decision_report",
        )
    ]
    receipt_root = _resolve(artifact_root, paths["access_receipt_directory"])
    release.extend(receipt_root / f"{partition}_unopened_access_receipt.json" for partition in PARTITIONS)
    release.append(_resolve(artifact_root, paths["qualification_audit"]).with_name("controlled_neural_world_benchmark_validation.json"))
    return release


def _require_zero_access(value: object, path: str = "release") -> None:
    """Require every nested protected/test access counter to be exactly zero."""
    counter_keys = {"six_context_outcome_access", "protected_holdout_access", "test_outcome_access"}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in counter_keys and child != 0:
                raise ValueError(f"nonzero access counter at {path}.{key}")
            _require_zero_access(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_zero_access(child, f"{path}[{index}]")


def _validate_durable_release(
    *, repository_root: Path, artifact_root: Path, paths: dict[str, object], config: dict[str, object],
    manifests: dict[str, dict[str, object]], ood: dict[str, object], headroom: dict[str, object],
    freeze: dict[str, object], qualification: dict[str, object]
) -> None:
    """Fail closed unless the exact content-addressed 14-file release is indexed."""
    if artifact_root.resolve() != repository_root.resolve():
        raise ValueError("durable release requires artifact_root == repository_root")
    release = _durable_release_paths(artifact_root, paths)
    if len(release) != 14 or len(set(release)) != 14:
        raise ValueError("durable release inventory must contain exactly 14 unique files")
    for path in release:
        relative = path.resolve().relative_to(repository_root.resolve())
        if not path.is_file():
            raise FileNotFoundError(f"durable release file missing: {relative}")
        indexed = subprocess.run(
            ["git", "show", f":{relative.as_posix()}"], cwd=repository_root,
            check=False, capture_output=True,
        )
        if indexed.returncode != 0:
            raise ValueError(f"durable release file is not Git-tracked: {relative}")
        if indexed.stdout != path.read_bytes():
            raise ValueError(f"durable release file differs from Git index: {relative}")

    bench = validate_benchmark_config(config)
    for role, manifest in manifests.items():
        worlds = manifest["worlds"]
        basis = {"role": role, "worlds": worlds, "policy": bench["split_policy"]}
        if manifest["content_identity"] != canonical_sha256({"role": role, "worlds": worlds}):
            raise ValueError(f"{role} manifest content_identity mismatch")
        if manifest["manifest_uid"] != stable_uid("controlled_world_benchmark_manifest", basis):
            raise ValueError(f"{role} manifest_uid mismatch")
    if ood["content_identity"] != canonical_sha256(ood["records"]):
        raise ValueError("OOD content_identity mismatch")
    if ood["manifest_uid"] != stable_uid("controlled_world_ood_manifest", {"records": ood["records"]}):
        raise ValueError("OOD manifest_uid mismatch")
    headroom_basis = {"records": headroom["records"], "statistics": headroom["statistics"]}
    if headroom["headroom_uid"] != stable_uid("controlled_world_development_headroom", headroom_basis):
        raise ValueError("headroom_uid mismatch")
    content_basis = {"records": headroom["records"], "statistics": headroom["statistics"], "failures": headroom["failure_records"], "undefined": headroom["undefined_cases"]}
    if headroom["content_identity"] != canonical_sha256(content_basis):
        raise ValueError("headroom content_identity mismatch")
    qualification_basis = {"headroom": headroom["content_identity"], "ood": ood["content_identity"], "blockers": qualification["blockers"]}
    if qualification["qualification_uid"] != stable_uid("controlled_world_qualification_report", qualification_basis):
        raise ValueError("qualification_uid mismatch")
    expected_freeze = build_freeze_receipt(
        config, repository_root=repository_root, manifests=manifests, ood_manifest=ood,
        headroom=headroom, qualification_report=qualification,
    )
    if freeze != expected_freeze:
        raise ValueError("freeze receipt does not exactly match the recomputed content-addressed receipt")
    expected_receipts = build_access_receipts(freeze, manifests["test"])
    for partition in PARTITIONS:
        receipt = _load(_resolve(artifact_root, paths["access_receipt_directory"]) / f"{partition}_unopened_access_receipt.json")
        if receipt != expected_receipts[partition]:
            raise ValueError(f"access receipt does not exactly match the recomputed receipt: {partition}")

    terminal = qualification.get("terminal_decision")
    if terminal not in qualification.get("allowed_terminals", []):
        raise ValueError(f"durable release terminal is not allowed: {terminal!r}")
    for payload in [*manifests.values(), ood, headroom, freeze, qualification]:
        _require_zero_access(payload)
    for partition in PARTITIONS:
        receipt = _load(_resolve(artifact_root, paths["access_receipt_directory"]) / f"{partition}_unopened_access_receipt.json")
        if receipt.get("freeze_uid") != freeze["freeze_uid"] or receipt.get("partition") != partition:
            raise ValueError(f"access receipt binding mismatch: {partition}")
        if receipt.get("result_opened") is not False or receipt.get("formal_result_access_count") != 0:
            raise ValueError(f"access receipt was opened: {partition}")
        _require_zero_access(receipt)

    report = _resolve(artifact_root, paths["decision_report"]).read_text(encoding="utf-8")
    required_report_values = (
        str(terminal),
        "Evidence class: controlled-world evidence only; not empirical, therapeutic, or clinical evidence.",
        "Access counters: six_context_outcome_access=0, protected_holdout_access=0, test_outcome_access=0.",
    )
    if any(report.count(value) != 1 for value in required_report_values):
        raise ValueError("decision report must contain exactly one terminal and non-empirical/zero-access boundary")


def validate_controlled_world_benchmark(
    *, config_path: Path, repository_root: Path, artifact_root: Path, require_durable_release: bool = False
) -> dict[str, object]:
    config = load_config(config_path)
    validate_benchmark_config(config)
    paths = dict(config["controlled_world_benchmark_qualification"]["output_paths"])
    blockers: list[dict[str, object]] = []
    manifests: dict[str, dict[str, object]] = {}
    try:
        for role in ("training", "development", "calibration", "test"):
            payload = _load(_resolve(artifact_root, paths[f"{role}_manifest"]))
            _schema(repository_root, "benchmark").validate(payload)
            manifests[role] = payload
        validate_lineage_separation(manifests)
        ood = _load(_resolve(artifact_root, paths["ood_manifest"]))
        _schema(repository_root, "ood").validate(ood)
        if tuple(ood["ood_classes"]) != OOD_CLASSES:
            blockers.append({"code": "ood_classes_incomplete"})
        headroom = _load(_resolve(artifact_root, paths["development_headroom"]))
        freeze = _load(_resolve(artifact_root, paths["freeze_receipt"]))
        _schema(repository_root, "freeze").validate(freeze)
        qualification = _load(_resolve(artifact_root, paths["qualification_audit"]))
        _schema(repository_root, "qualification").validate(qualification)
        for partition in PARTITIONS:
            receipt_path = _resolve(artifact_root, paths["access_receipt_directory"]) / f"{partition}_unopened_access_receipt.json"
            receipt = _load(receipt_path)
            if receipt["result_opened"] is not False or receipt["formal_result_access_count"] != 0:
                blockers.append({"code": "test_outcome_access_forbidden", "partition": partition})
        if qualification["terminal_decision"] not in qualification["allowed_terminals"]:
            blockers.append({"code": "invalid_terminal"})
    except Exception as exc:  # noqa: BLE001 - validator preserves exact blocker for audit instead of hiding failures.
        blockers.append({"code": type(exc).__name__, "message": str(exc)})
    audit = {
        "schema_version": "neuronsifter-controlled-world-benchmark-validation-v1",
        "status": "passed" if not blockers else "failed",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "family_uids": list(config["controlled_world_benchmark_qualification"]["allowed_family_uids"]),
        "sealed_test_partitions": list(PARTITIONS),
        "ood_classes": list(OOD_CLASSES),
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0},
    }
    validation_path = _resolve(artifact_root, paths["qualification_audit"]).with_name("controlled_neural_world_benchmark_validation.json")
    write_json(validation_path, audit)
    if require_durable_release and not blockers:
        try:
            _validate_durable_release(
                repository_root=repository_root,
                artifact_root=artifact_root,
                paths=paths,
                config=config,
                manifests=manifests,
                ood=ood,
                headroom=headroom,
                freeze=freeze,
                qualification=qualification,
            )
        except Exception as exc:  # noqa: BLE001 - durable mode must convert every incomplete-release condition to a blocker.
            blockers.append({"code": "durable_release_invalid", "message": str(exc)})
            audit["status"] = "failed"
            audit["blocker_count"] = len(blockers)
            write_json(validation_path, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--require-durable-release",
        action="store_true",
        help="require the complete 14-file release to exactly match Git's index",
    )
    args = parser.parse_args()
    audit = validate_controlled_world_benchmark(
        config_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        require_durable_release=args.require_durable_release,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
