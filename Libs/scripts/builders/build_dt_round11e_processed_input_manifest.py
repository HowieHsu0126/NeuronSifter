"""Build the Round-11E read-only generated-input allowlist from discovery evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json  # noqa: E402
from Libs.utils.data.round11e_processed_input_linkage import build_processed_input_manifest  # noqa: E402
from Libs.utils.data.round11e_processed_input_linkage import normalize_logical_path  # noqa: E402


_LOGICAL_PATH = re.compile(r"Input/processed/[A-Za-z0-9._+@=/-]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_paths_from_trace(trace_path: Path) -> list[str]:
    logical_paths = {
        normalize_logical_path(match.group(0))
        for match in _LOGICAL_PATH.finditer(trace_path.read_text(encoding="utf-8", errors="replace"))
    }
    return sorted(logical_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a hash-bound Round-11E Input/processed linkage manifest.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--discovery-trace", required=True)
    parser.add_argument("--discovery-trace-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-entry-count", type=int, default=54)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    trace_path = Path(args.discovery_trace).resolve()
    output_path = Path(args.output).resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"Round-11E input discovery trace is unavailable: {trace_path}")
    logical_paths = logical_paths_from_trace(trace_path)
    if len(logical_paths) != args.expected_entry_count:
        raise ValueError(
            "Round-11E input discovery allowlist count is unexpected: "
            f"expected {args.expected_entry_count}, observed {len(logical_paths)}"
        )
    manifest = build_processed_input_manifest(
        input_root,
        logical_paths,
        discovery_trace_label=args.discovery_trace_label,
        discovery_trace_sha256=sha256_file(trace_path),
    )
    write_json(output_path, manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "entry_count": manifest["entry_count"],
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
