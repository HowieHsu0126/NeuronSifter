"""Build controlled neural-world benchmark qualification manifests and receipts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.controlled_world_benchmark_freeze import build_access_receipts, build_freeze_receipt  # noqa: E402
from Libs.utils.data.controlled_world_benchmark_manifests import build_ood_manifest, build_split_manifests  # noqa: E402
from Libs.utils.data.controlled_world_headroom_statistics import build_development_headroom, qualification_decision  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


def _resolve(artifact_root: Path, relative: object) -> Path:
    path = Path(str(relative))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Benchmark output path must be repository-relative: {relative}")
    return artifact_root / path


def build_controlled_world_benchmark_manifests(*, config_path: Path, repository_root: Path, artifact_root: Path) -> dict[str, Path]:
    config = load_config(config_path)
    paths = dict(config["controlled_world_benchmark_qualification"]["output_paths"])
    manifests = build_split_manifests(config, repository_root=repository_root)
    ood_manifest = build_ood_manifest(config, manifests, repository_root=repository_root)
    headroom = build_development_headroom(config, manifests["development"], repository_root=repository_root)
    qualification = qualification_decision(headroom, ood_manifest, config)
    freeze = build_freeze_receipt(
        config,
        repository_root=repository_root,
        manifests=manifests,
        ood_manifest=ood_manifest,
        headroom=headroom,
        qualification_report=qualification,
    )
    access_receipts = build_access_receipts(freeze, manifests["test"])

    written: dict[str, Path] = {}
    for role in ("training", "development", "calibration", "test"):
        path = _resolve(artifact_root, paths[f"{role}_manifest"])
        write_json(path, manifests[role])
        written[f"{role}_manifest"] = path
    for key, payload in (("ood_manifest", ood_manifest), ("development_headroom", headroom), ("freeze_receipt", freeze), ("qualification_audit", qualification)):
        path = _resolve(artifact_root, paths[key])
        write_json(path, payload)
        written[key] = path
    receipt_dir = _resolve(artifact_root, paths["access_receipt_directory"])
    for partition, receipt in access_receipts.items():
        path = receipt_dir / f"{partition}_unopened_access_receipt.json"
        write_json(path, receipt)
        written[f"access_receipt.{partition}"] = path
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    paths = build_controlled_world_benchmark_manifests(
        config_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
