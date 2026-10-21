"""Bounded metadata admission and native producer-store preparation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.native_diffusion_data import (
    bounded_metadata_report, load_registry, load_family_registries, select_records, load_records, reindex_registry,
)
from Libs.utils.model.microenvironment_teacher import write_native_microenvironment_store
from Libs.utils.data.native_field_normalization import fit_transform, fit_multigrid, bind_artifact, validate_artifact, representation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--store-metadata", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--family", help="Explicit registered physical source family")
    parser.add_argument("--split", choices=("train", "development", "validation", "calibration"), default="development")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fit-field-normalization", action="store_true")
    parser.add_argument("--normalization-artifact", type=Path)
    parser.add_argument("--store-format", choices=("zarr_v2","neuronsifter_chunk_store"), default="zarr_v2")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.family:
        config["native_diffusion_data"]["source_family_uid"] = args.family
    bound = args.max_rows or int(config["native_diffusion_data"]["metadata_row_limit"])
    if args.metadata_only:
        if args.catalog is None:
            parser.error("--metadata-only requires an explicit --catalog")
        report = bounded_metadata_report(args.catalog, bound, config)
        if args.store_metadata is not None:
            metadata = json.loads(args.store_metadata.read_text(encoding="utf-8"))
            report["store_metadata"] = str(args.store_metadata)
            report["store_native_binding_present"] = bool(metadata.get("native_diffusion"))
            report["store_coordinates"] = metadata.get("coordinates", {})
        print(json.dumps(report, indent=2))
        return 2 if report["rejected_rows"] else 0
    if not 0 < bound <= 256 or args.output is None:
        parser.error("store preparation requires --output and max rows in 1..256")
    if args.output.exists():
        parser.error("output exists; refusing to overwrite a teacher store")
    if args.fit_field_normalization and (args.split != "train" or args.normalization_artifact):
        parser.error("normalization fitting requires explicit --split train and no existing scaler")
    if args.fit_field_normalization:
        registries = load_family_registries(config)
        batches = []
        for registry in registries.values():
            if registry["grid"].get("normalization", {}).get("kind") != "physical":
                parser.error("fit requires admitted raw physical/source coordinates")
            records = select_records(registry, config, "train")
            if sum(len(rows) for _, rows, _ in batches) + len(records) > bound:
                parser.error("training rows exceed explicit bound across all families")
            arrays, _ = load_records(registry, records, config)
            batches.append((registry["grid"], records, arrays))
        if any(grid.get("quantity_profile") for grid, _, _ in batches):
            artifact = fit_multigrid(batches, config)
        else:
            artifact = fit_transform(*batches[0], config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"fit_role": "train", "trajectory_count": artifact["trajectory_count"],
                          "artifact": str(args.output)}))
        return 0
    registry = load_registry(config)
    records = select_records(registry, config, args.split)
    if len(records) > bound:
        parser.error("selected records exceed explicit bound; provide a bounded registry")
    arrays, _ = load_records(registry, records, config)
    derived = reindex_registry(registry, records, args.output, args.store_format)
    if args.normalization_artifact:
        derived["grid"] = bind_artifact(registry["grid"], args.normalization_artifact, config)
    write_native_microenvironment_store(args.output, config, derived, arrays, args.store_format)
    print(json.dumps({"rows_written": len(records), "store": str(args.output), "scope": "producer serialization only"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
