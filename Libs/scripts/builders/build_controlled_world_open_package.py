"""Build the open-only controlled-world action-identification package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.controlled_world_open_package import build_controlled_world_open_package  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    paths = build_controlled_world_open_package(
        config=config,
        repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
