"""Validate a bounded, QC-selected conditional-training sample."""

from __future__ import annotations

# Standalone script bootstraps the repository path before project imports.
# ruff: noqa: E402

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.pretrain_microenvironment_world_model import (
    MicroenvironmentZarrDataset,
    manifest_row_indices,
)
from Libs.utils.data.raw_data_registry import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--store", required=True)
    parser.add_argument("--sample-count", type=int, default=1)
    args = parser.parse_args()
    if args.sample_count < 1:
        raise ValueError("sample-count must be positive.")
    config = load_config(args.config)
    split = config["screening_pipeline"]["screening_benchmark"]["training"][
        "zarr_loader"
    ]["training_split_name"]
    store = Path(args.store).resolve()
    indices = manifest_row_indices(config, store, split)
    if not indices:
        raise ValueError("No QC-approved training trajectories are available.")
    dataset = MicroenvironmentZarrDataset(
        store, config, row_indices=indices[: args.sample_count]
    )
    rows = dataset.__getitems__(list(range(len(dataset))))
    print(
        f"Validated {len(rows)} independent-occupancy training rows; no scientific qualification implied."
    )


if __name__ == "__main__":
    main()
