"""Attach explicit, independently generated action programs to a teacher store.

This entrypoint never derives conditioning from the teacher response. The input
NPZ contains independently compiled programs and a manifest joined to QC rows.
"""

from __future__ import annotations

# Standalone script bootstraps the repository path before project imports.
# ruff: noqa: E402

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.occupancy_conditioning import write_conditioning_arrays
from Libs.utils.model.dynamics_store import read_zarr_array_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--store", required=True)
    parser.add_argument("--program-arrays", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    store = Path(args.store).resolve()
    shape = tuple(read_zarr_array_metadata(store, "trajectories")["shape"])
    manifest = json.loads(Path(args.manifest).read_text())
    with np.load(args.program_arrays, allow_pickle=False) as source:
        arrays = {
            key: source[key]
            for key in (
                "occupancy_program",
                "occupancy_support",
                "posterior_context",
                "time_hours",
                "sample_keys",
            )
        }
    write_conditioning_arrays(store, arrays, shape, config, manifest)
    print(
        "Independent conditioning arrays written; existing artifacts were not overwritten."
    )


if __name__ == "__main__":
    main()
