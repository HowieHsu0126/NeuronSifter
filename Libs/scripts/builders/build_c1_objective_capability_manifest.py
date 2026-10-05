"""Build the split/role-scoped C1 objective capability manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1_capabilities import (  # noqa: E402
    build_c1_objective_capability_manifest,
)
from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument(
        "--output", default="Output/audits/c1_objective_capability_manifest.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    manifest = build_c1_objective_capability_manifest(config, REPO_ROOT)
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"c1_objective_capability_manifest={output.relative_to(REPO_ROOT).as_posix()}")
    print(f"c1_objective_capability_split_count={len(manifest['splits'])}")


if __name__ == "__main__":
    main()
