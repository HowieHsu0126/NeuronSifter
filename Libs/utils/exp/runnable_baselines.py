from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from Libs.utils.evaluation.benchmark_datasets import assemble_dataset, benchmark_program_config, fold_entity_index, fold_sample_split, load_benchmark_catalogs
from Libs.utils.data.raw_data_registry import ensure_directory, load_config


def baseline_entry(config: dict, baseline_name: str) -> dict:
    for row in config["paper_experiments"].get("runnable_baselines", []):
        if row["baseline_name"] == baseline_name:
            return row
    raise ValueError(f"Unknown runnable baseline: {baseline_name}")


def adapter_mode(entry: dict) -> str:
    if "adapter_mode" not in entry:
        raise ValueError(f"Runnable baseline is missing adapter_mode: {entry}")
    return str(entry["adapter_mode"])


def prepare_baseline_inputs(
    config_path: str,
    baseline_name: str,
    disease_program: str,
    fold_name: str,
    output_dir: Path,
) -> Path:
    config = load_config(config_path)
    config["benchmark_registry"].update(benchmark_program_config(config, disease_program))
    _, fold_rows, _ = load_benchmark_catalogs(config, disease_program)
    target_fold = next(row for row in fold_rows if row["fold_name"] == fold_name)
    split_frame = fold_sample_split(config, fold_name, disease_program)

    ensure_directory(output_dir)
    manifest = {
        "baseline_name": baseline_name,
        "disease_program": disease_program,
        "benchmark_name": config["benchmark_registry"]["benchmark_name"],
        "fold_name": fold_name,
        "held_out_study_uid": target_fold["held_out_study_uid"],
        "prepared_arrays": [],
    }
    for modality in ["transcriptomics", "metabolomics"]:
        entity_index = fold_entity_index(config, fold_name, modality, disease_program)
        for split_role in ["train", "validation", "test"]:
            role_frame = split_frame.loc[split_frame["split_role"] == split_role].copy()
            matrix, labels, metadata = assemble_dataset(config, role_frame, modality, entity_index)
            array_path = output_dir / f"{baseline_name}_{disease_program}_{fold_name}_{modality}_{split_role}.npz"
            np.savez_compressed(
                array_path,
                matrix=matrix.astype(np.float32),
                labels=labels.astype(np.int64),
                sample_uid=metadata["sample_uid"].astype(str).to_numpy() if not metadata.empty else np.asarray([], dtype=str),
                study_uid=metadata["study_uid"].astype(str).to_numpy() if not metadata.empty else np.asarray([], dtype=str),
                canonical_entity_id=entity_index["canonical_entity_id"].astype(str).to_numpy() if not entity_index.empty else np.asarray([], dtype=str),
            )
            manifest["prepared_arrays"].append(str(array_path))
    manifest_path = output_dir / f"{baseline_name}_{disease_program}_{fold_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest_path
