"""Validate the generated C1 repository inventory artifacts by reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_c1_repository_inventory import build  # noqa: E402


NAME_SUFFIXES = (
    "repository_inventory.json",
    "reuse_and_gap_map.json",
    "config_path_map.json",
)


def validate(
    repo_root: Path = REPO_ROOT,
    *,
    config_path: str | Path = "Libs/configs/config_stack.yaml",
    output_prefix: str = "c1",
) -> dict[str, object]:
    expected = build(repo_root, config_path=config_path)
    blockers: list[dict[str, str]] = []
    names = tuple(f"{output_prefix}_{suffix}" for suffix in NAME_SUFFIXES)
    for name, reconstructed in zip(names, expected, strict=True):
        path = repo_root / "Output/audits" / name
        if not path.is_file():
            blockers.append({"code": "c1_inventory_artifact_missing", "path": name, "message": "required artifact is missing"})
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != reconstructed:
            blockers.append({"code": "c1_inventory_reconstruction_mismatch", "path": name, "message": "stored artifact differs from live reconstruction"})
    return {"status": "passed" if not blockers else "blocked", "blocker_count": len(blockers), "blockers": blockers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-prefix", default="c1")
    args = parser.parse_args()
    report = validate(
        REPO_ROOT,
        config_path=args.config,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
