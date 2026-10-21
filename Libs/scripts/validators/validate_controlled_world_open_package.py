"""Validate the open-only controlled-world action-identification package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.controlled_world_open_package import (  # noqa: E402
    OPEN_PACKAGE_POSITIVE_TERMINAL,
    OPEN_ROLES,
    build_construction_audit,
    construction_report,
    resolve_declared_output,
    validate_open_manifests,
    validate_open_package_config,
)
from Libs.utils.data.raw_data_registry import load_config, write_json, write_text_atomic  # noqa: E402

SCHEMAS = {
    "manifest": "Libs/configs/schema/controlled_world_open_manifest.schema.yaml",
    "receipt": "Libs/configs/schema/controlled_world_open_package_receipt.schema.yaml",
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema(repository_root: Path, key: str) -> Draft202012Validator:
    schema_path = repository_root / SCHEMAS[key]
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_controlled_world_open_package(
    *,
    config_path: Path,
    repository_root: Path,
    artifact_root: Path,
    write_artifacts: bool = True,
) -> dict[str, object]:
    config = load_config(config_path)
    open_cfg = validate_open_package_config(config)
    paths = dict(open_cfg["output_paths"])
    blockers: list[dict[str, object]] = []
    manifests: dict[str, dict[str, object]] = {}
    receipt: dict[str, object] = {}
    try:
        manifest_schema = _schema(repository_root, "manifest")
        for role in OPEN_ROLES:
            path = resolve_declared_output(artifact_root, paths[f"{role}_manifest"], label=f"{role}_manifest")
            payload = _load(path)
            manifest_schema.validate(payload)
            manifests[role] = payload
        receipt_path = resolve_declared_output(artifact_root, paths["package_receipt"], label="package_receipt")
        receipt = _load(receipt_path)
        _schema(repository_root, "receipt").validate(receipt)
        blockers.extend(validate_open_manifests(manifests, config=config, repository_root=repository_root))
        expected_receipt_identities = {role: manifests[role]["content_identity"] for role in OPEN_ROLES}
        if receipt.get("manifest_content_identities") != expected_receipt_identities:
            blockers.append({"code": "package_receipt_manifest_identity_mismatch"})
        if receipt.get("training_authorized") is not False or receipt.get("open_data_qualification") != "pending":
            blockers.append({"code": "package_receipt_downstream_gate_open"})
    except Exception as exc:  # noqa: BLE001 - validator must retain exact blocker.
        blockers.append({"code": type(exc).__name__, "message": str(exc)})
    audit = build_construction_audit(manifests, receipt, blockers, config=config)
    if write_artifacts:
        audit_path = resolve_declared_output(artifact_root, paths["construction_audit"], label="construction_audit")
        report_path = resolve_declared_output(artifact_root, paths["construction_report"], label="construction_report")
        write_json(audit_path, audit)
        write_text_atomic(report_path, construction_report(audit))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()
    audit = validate_controlled_world_open_package(
        config_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
        write_artifacts=not args.read_only,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit.get("terminal_decision") == OPEN_PACKAGE_POSITIVE_TERMINAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
