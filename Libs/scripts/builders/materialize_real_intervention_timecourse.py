"""Materialize a quantitative public intervention time course as a canonical block."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256, hash_path  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402
from Libs.utils.model.dynamics_store import initialize_zarr_store, write_zarr_array  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
SOURCE_KEY = "amyloid_peptide_timecourse"
BUILDER_ID = "Libs/scripts/builders/materialize_real_intervention_timecourse.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes the repository: {value}")
    return path


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _record(path: Path, name: str) -> dict[str, object]:
    file_count, byte_count, digest = hash_path(path)
    return {
        "name": name,
        "relative_path": name,
        "file_count": file_count,
        "bytes": byte_count,
        "sha256": digest,
    }


def _write_staged_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    dataframe.to_parquet(path, index=False)


def parse_localized_float(value: str) -> float:
    normalized = value.strip().replace(",", ".")
    result = float(normalized)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return result


def _load_retrieval_bindings(source: Mapping[str, object]) -> dict[str, Path]:
    snapshot_root = _repo_path(str(source["official_snapshot_root"]))
    receipt_path = snapshot_root / "retrieval_record.json"
    receipt = _json(receipt_path)
    if receipt.get("source_id") != source["source_id"] or receipt.get("source_accession") != source["source_accession"]:
        raise ValueError("Retrieval receipt source identity does not match the configured source")
    recorded = {str(row["asset_role"]): row for row in receipt.get("assets", []) if isinstance(row, Mapping)}
    assets = source["assets"]
    assert isinstance(assets, Mapping)
    paths: dict[str, Path] = {}
    for asset_role, raw_contract in assets.items():
        if not isinstance(raw_contract, Mapping):
            raise TypeError(f"Asset contract must be a mapping: {asset_role}")
        path = snapshot_root / str(raw_contract["filename"])
        binding = _binding(path)
        expected = recorded.get(str(asset_role))
        if expected is None or any(binding[key] != expected.get(key) for key in ("path", "bytes", "sha256")):
            raise ValueError(f"Retrieved asset binding drifted: {asset_role}")
        if expected.get("url") != raw_contract["url"]:
            raise ValueError(f"Retrieved asset URL drifted: {asset_role}")
        paths[str(asset_role)] = path
    paths["retrieval_record"] = receipt_path
    return paths


def _read_sdrf(path: Path, source: Mapping[str, object]) -> pd.DataFrame:
    design = source["expected_design"]
    assert isinstance(design, Mapping)
    conditions = design["conditions"]
    assert isinstance(conditions, Mapping)
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    output: list[dict[str, object]] = []
    group_counts: dict[tuple[str, float], int] = defaultdict(int)
    for row in rows:
        raw_condition = str(row["Factor Value[compound]"]).strip()
        if raw_condition not in conditions:
            raise ValueError(f"Unexpected intervention condition: {raw_condition!r}")
        condition_uid = str(conditions[raw_condition])
        time_value = float(str(row["Factor Value [time]"]).strip())
        time_unit = str(row["Unit[time unit]"]).strip()
        if time_unit != "hour":
            raise ValueError(f"Unexpected time unit: {time_unit!r}")
        if str(row["Characteristics[organism]"]).strip() != design["organism"]:
            raise ValueError("SDRF organism differs from the frozen design")
        if str(row["Characteristics[cell line]"]).strip() != design["cell_line"]:
            raise ValueError("SDRF cell line differs from the frozen design")
        if str(row["Array Design REF"]).strip() != design["platform_accession"]:
            raise ValueError("SDRF platform differs from the frozen design")
        group_counts[(condition_uid, time_value)] += 1
        replicate_index = group_counts[(condition_uid, time_value)]
        sample_name = str(row["Source Name"]).strip()
        assay_name = str(row["Assay Name"]).strip()
        is_control = condition_uid == "vehicle_control"
        action_family_uid = f"arrayexpress:E-MTAB-1841:action_family:{condition_uid}"
        world_action_uid = f"arrayexpress:E-MTAB-1841:world_action:{condition_uid}:{time_value:g}h"
        control_group_uid = f"arrayexpress:E-MTAB-1841:vehicle_control:{time_value:g}h"
        output.append(
            {
                "source_uid": "arrayexpress",
                "publication_uid": "arrayexpress:E-MTAB-1841",
                "accession_uid": "E-MTAB-1841",
                "source_collection_uid": "arrayexpress:E-MTAB-1841",
                "source_sample_name": sample_name,
                "source_assay_name": assay_name,
                "source_value_file": str(row["Array Data File"]).strip(),
                "sample_uid": f"arrayexpress:E-MTAB-1841:sample:{sample_name}",
                "observation_uid": f"arrayexpress:E-MTAB-1841:observation:{sample_name}",
                "biological_unit_uid": f"arrayexpress:E-MTAB-1841:biological_unit:{sample_name}",
                "subject_or_donor_uid": "not_applicable_cell_line_experiment",
                "cell_or_tissue_context": "human_SH-SY5Y_neuroblastoma_cell_line",
                "cns_context_class": "human_neuronal_cell_line",
                "species": "Homo sapiens",
                "species_taxonomy_id": "NCBITaxon:9606",
                "cell_line": "SH-SY5Y",
                "cell_type": "neuroblastoma",
                "sex": "female",
                "assay_uid": "arrayexpress:A-MEXP-1171",
                "modality": "transcriptomics",
                "measurement_namespace": "NCBI_Entrez_Gene",
                "value_unit": "unnormalized_Illumina_AVG_Signal",
                "condition_original": raw_condition,
                "condition_uid": condition_uid,
                "observation_role": "matched_control" if is_control else "intervention_endpoint",
                "intervention_type": "vehicle_control" if is_control else "peptide_perturbation",
                "agent_uid": condition_uid,
                "dose_value": 0.0 if is_control else float(design["dose_value"]),
                "dose_unit": str(design["dose_unit"]),
                "time_value": time_value,
                "time_unit": "hour",
                "duration": time_value,
                "duration_unit": "hour",
                "schedule": "continuous_in_vitro_exposure_until_endpoint",
                "route": "in_vitro_media_exposure",
                "action_family_uid": action_family_uid,
                "regimen_family_uid": world_action_uid,
                "world_action_uid": world_action_uid,
                "baseline_state_uid": control_group_uid,
                "control_group_uid": control_group_uid,
                "trajectory_uid": f"arrayexpress:E-MTAB-1841:cross_sectional_timecourse:{condition_uid}",
                "replicate_group_uid": f"arrayexpress:E-MTAB-1841:{condition_uid}:{time_value:g}h",
                "replicate_index": replicate_index,
                "trajectory_unit_semantics": "independent_cross_sectional_biological_replicates_by_time",
                "evidence_class": source["endpoint_evidence_class"] if not is_control else "E2_REAL_ENDPOINT",
                "trajectory_evidence_class": source["evidence_class"],
                "evidence_grade": source["evidence_grade"],
                "uncertainty_semantics": "estimated_from_five_independent_replicates_per_condition_time",
            }
        )
    frame = pd.DataFrame(output)
    expected_times = {float(value) for value in design["time_values_hours"]}
    if len(frame) != int(design["sample_count"]):
        raise ValueError(f"SDRF sample count drifted: {len(frame)}")
    if set(frame["time_value"].astype(float)) != expected_times:
        raise ValueError("SDRF time grid differs from the frozen design")
    expected_group_count = int(design["biological_replicates_per_condition_time"])
    if set(group_counts.values()) != {expected_group_count} or len(group_counts) != len(conditions) * len(expected_times):
        raise ValueError(f"SDRF replicate design drifted: {dict(group_counts)}")
    if frame["sample_uid"].duplicated().any() or frame["observation_uid"].duplicated().any():
        raise ValueError("SDRF sample or observation identities are not unique")
    frame["value_row_index"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _read_expression_matrix(
    path: Path,
    observations: pd.DataFrame,
    probe_column: str,
    signal_suffix: str,
) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next((row for row in reader if row and row[0] == probe_column), None)
        if header is None:
            raise ValueError(f"Expression matrix lacks {probe_column}")
        positions = {name: index for index, name in enumerate(header)}
        signal_fields = [f"{assay}{signal_suffix}" for assay in observations["source_assay_name"].astype(str)]
        missing = [field for field in signal_fields if field not in positions]
        if missing:
            raise ValueError(f"Expression matrix lacks SDRF-bound signals: {missing[:3]}")
        indices = [positions[field] for field in signal_fields]
        probe_ids: list[str] = []
        probe_rows: list[np.ndarray] = []
        max_index = max(indices)
        for row in reader:
            if not row:
                continue
            if len(row) <= max_index:
                raise ValueError("Expression matrix row is shorter than its header")
            probe_ids.append(str(row[positions[probe_column]]).strip())
            probe_rows.append(np.asarray([parse_localized_float(row[index]) for index in indices], dtype=np.float64))
    if not probe_ids or len(probe_ids) != len(set(probe_ids)):
        raise ValueError("Expression matrix probe identities are empty or duplicated")
    matrix = np.vstack(probe_rows).T
    if matrix.shape[0] != len(observations) or not np.isfinite(matrix).all():
        raise ValueError("Expression matrix sample axis or finite-value contract failed")
    return probe_ids, matrix


def _read_adf(path: Path) -> dict[str, tuple[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header_index = next(index for index, row in enumerate(rows) if row and row[0] == "Block Column")
    header = rows[header_index]
    positions = {name: index for index, name in enumerate(header)}
    required = ("Reporter Name", "Reporter Database Entry[entrez]", "Reporter Database Entry[hugo]")
    if any(name not in positions for name in required):
        raise ValueError("Platform annotation lacks reporter, Entrez, or symbol fields")
    result: dict[str, tuple[str, str]] = {}
    for row in rows[header_index + 1 :]:
        if len(row) <= max(positions[name] for name in required):
            continue
        reporter = row[positions["Reporter Name"]].strip()
        if not reporter:
            continue
        if reporter in result:
            raise ValueError(f"Duplicate platform reporter: {reporter}")
        result[reporter] = (
            row[positions["Reporter Database Entry[entrez]"]].strip(),
            row[positions["Reporter Database Entry[hugo]"]].strip(),
        )
    return result


def canonicalize_entrez_features(
    probe_ids: Sequence[str],
    matrix: np.ndarray,
    annotations: Mapping[str, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, int]]:
    positions_by_gene: dict[str, list[int]] = defaultdict(list)
    symbols_by_gene: dict[str, set[str]] = defaultdict(set)
    quarantine: list[dict[str, str]] = []
    exact_probe_count = 0
    for position, probe_id in enumerate(probe_ids):
        annotation = annotations.get(probe_id)
        if annotation is None:
            quarantine.append({"source_feature_id": probe_id, "mapping_status": "unmatched_adf_reporter", "raw_entrez_id": ""})
            continue
        entrez, symbol = annotation
        if not entrez.isdigit():
            quarantine.append({"source_feature_id": probe_id, "mapping_status": "non_numeric_or_missing_adf_entrez", "raw_entrez_id": entrez})
            continue
        exact_probe_count += 1
        positions_by_gene[entrez].append(position)
        if symbol:
            symbols_by_gene[entrez].add(symbol)
    ordered_genes = sorted(positions_by_gene, key=lambda gene: min(positions_by_gene[gene]))
    if not ordered_genes:
        raise ValueError("No probes map to numeric Entrez genes")
    canonical = np.empty((matrix.shape[0], len(ordered_genes)), dtype=np.float64)
    feature_rows: list[dict[str, object]] = []
    for feature_index, gene_id in enumerate(ordered_genes):
        positions = positions_by_gene[gene_id]
        canonical[:, feature_index] = np.median(matrix[:, positions], axis=1)
        feature_rows.append(
            {
                "feature_index": feature_index,
                "feature_namespace": "NCBI_Entrez_Gene",
                "feature_id": f"NCBIGene:{gene_id}",
                "canonical_gene_id": gene_id,
                "source_gene_symbols": "|".join(sorted(symbols_by_gene[gene_id])) or "explicit_unavailable",
                "source_probe_ids": "|".join(probe_ids[position] for position in positions),
                "source_probe_count": len(positions),
                "mapping_method": "exact_A-MEXP-1171_reporter_to_numeric_Entrez_then_median",
                "aggregation_rule": "median_across_probes_mapping_to_same_Entrez_gene",
            }
        )
    quarantine_frame = pd.DataFrame(
        quarantine,
        columns=["source_feature_id", "mapping_status", "raw_entrez_id"],
    )
    diagnostics = {
        "source_probe_count": len(probe_ids),
        "exact_entrez_probe_count": exact_probe_count,
        "canonical_gene_count": len(ordered_genes),
        "unmatched_probe_count": sum(row["mapping_status"] == "unmatched_adf_reporter" for row in quarantine),
        "non_numeric_entrez_count": sum(row["mapping_status"] == "non_numeric_or_missing_adf_entrez" for row in quarantine),
    }
    return pd.DataFrame(feature_rows), quarantine_frame, canonical, diagnostics


def _actions(observations: pd.DataFrame, source: Mapping[str, object]) -> pd.DataFrame:
    treated = observations[observations["observation_role"] == "intervention_endpoint"]
    rows: list[dict[str, object]] = []
    for world_action_uid, group in treated.groupby("world_action_uid", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "world_action_uid": world_action_uid,
                "action_family_uid": first["action_family_uid"],
                "regimen_family_uid": first["regimen_family_uid"],
                "intervention_type": first["intervention_type"],
                "agent_uid": first["agent_uid"],
                "agent_set": first["agent_uid"],
                "target_set": "explicit_unknown",
                "dose_value": first["dose_value"],
                "dose_unit": first["dose_unit"],
                "duration": first["duration"],
                "duration_unit": first["duration_unit"],
                "schedule": first["schedule"],
                "route": first["route"],
                "cell_or_tissue_context": first["cell_or_tissue_context"],
                "exposure_program_ref": "source_observed_constant_media_concentration_1_micromolar",
                "occupancy_program_ref": "explicit_unavailable",
                "action_completeness": source["action_completeness"],
                "publication_uid": first["publication_uid"],
                "accession_uid": first["accession_uid"],
                "control_group_uid": first["control_group_uid"],
                "observation_uids": "|".join(group["observation_uid"].astype(str)),
                "linked_observation_count": len(group),
                "evidence_grade": source["evidence_grade"],
            }
        )
    return pd.DataFrame(rows).sort_values("world_action_uid", kind="mergesort").reset_index(drop=True)


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("Time-course materialization is identity-bound to the repository root")
    config = load_config(CONFIG_PATH)
    source = config["real_intervention_evidence"]["sources"][SOURCE_KEY]
    if not isinstance(source, Mapping):
        raise TypeError("Real-intervention source contract must be a mapping")
    paths = _load_retrieval_bindings(source)
    observations = _read_sdrf(paths["sample_metadata"], source)
    parser = source["parser"]
    assert isinstance(parser, Mapping)
    probes, source_matrix = _read_expression_matrix(
        paths["expression_matrix"],
        observations,
        str(parser["probe_id_column"]),
        str(parser["signal_suffix"]),
    )
    features, quarantine, canonical_matrix, mapping = canonicalize_entrez_features(
        probes,
        source_matrix,
        _read_adf(paths["platform_annotation"]),
    )
    if not np.isfinite(canonical_matrix).all() or float(canonical_matrix.min()) < 0.0:
        raise ValueError("Canonical expression values are non-finite or negative")
    controls = observations[observations["observation_role"] == "matched_control"].copy()
    actions = _actions(observations, source)
    expected_control_count = len(source["expected_design"]["time_values_hours"]) * int(
        source["expected_design"]["biological_replicates_per_condition_time"]
    )
    if len(controls) != expected_control_count or len(actions) != 12:
        raise ValueError("Control or action count differs from the frozen factorial design")

    output_root = _repo_path(str(source["canonical_output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real-intervention-timecourse-", dir=output_root) as temporary:
        stage = Path(temporary)
        store = stage / "expression.zarr"
        initialize_zarr_store(
            store,
            {
                "format": "zarr_v2_uncompressed",
                "source_accession": source["source_accession"],
                "row_axis": "observations.parquet:value_row_index",
                "column_axis": "features.parquet:feature_index",
                "value_semantics": parser["source_value_semantics"],
                "analysis_transform_applied": False,
            },
        )
        write_zarr_array(
            store,
            "values",
            canonical_matrix,
            (int(parser["row_chunk_size"]), len(features)),
            attrs={"feature_namespace": "NCBI_Entrez_Gene", "unit": parser["source_value_semantics"]},
        )
        _write_staged_parquet(observations, stage / "observations.parquet")
        _write_staged_parquet(features, stage / "features.parquet")
        _write_staged_parquet(controls, stage / "controls.parquet")
        _write_staged_parquet(actions, stage / "actions.parquet")
        _write_staged_parquet(quarantine, stage / "feature_quarantine.parquet")
        matrix_manifest = {
            "matrix_shape": [len(observations), len(features)],
            "source_matrix_shape": [len(observations), len(probes)],
            "scalar_value_count": int(canonical_matrix.size),
            "finite_values": True,
            "minimum_value": float(canonical_matrix.min()),
            "maximum_value": float(canonical_matrix.max()),
            "mapping_diagnostics": mapping,
            "source_values_preserved_by": _binding(paths["expression_matrix"]),
            "materialization_transform": "median_across_probes_with_same_numeric_Entrez_gene",
            "analysis_transform_applied": False,
        }
        write_json(stage / "matrix_manifest.json", matrix_manifest)
        output_names = (
            "expression.zarr",
            "observations.parquet",
            "features.parquet",
            "controls.parquet",
            "actions.parquet",
            "feature_quarantine.parquet",
            "matrix_manifest.json",
        )
        records = [_record(stage / name, name) for name in output_names]
        block_sha256 = canonical_sha256(
            [
                {
                    "name": row["name"],
                    "file_count": row["file_count"],
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
                for row in records
            ]
        )
        destination = output_root / "blocks" / str(source["source_id"]) / str(source["source_accession"]) / block_sha256[:16]
        manifest = {
            "schema_version": "neuronsifter-real-intervention-timecourse-block-v1",
            "producer": BUILDER_ID,
            "created_at": _now(),
            "canonical_uid": f"real_intervention:arrayexpress:E-MTAB-1841:{block_sha256[:24]}",
            "content_address": f"sha256:{block_sha256}",
            "block_sha256": block_sha256,
            "source_id": source["source_id"],
            "source_accession": source["source_accession"],
            "publication_accession_group": "arrayexpress::E-MTAB-1841",
            "publication_uid": "arrayexpress:E-MTAB-1841",
            "role": "qualified_real_intervention_development_evidence",
            "admission_allowed": True,
            "quantitative_evidence_classes": [source["endpoint_evidence_class"], source["evidence_class"]],
            "trajectory_unit_semantics": "independent_cross_sectional_biological_replicates_by_time",
            "same_unit_longitudinal_trajectory": False,
            "observation_count": len(observations),
            "intervention_observation_count": int((observations["observation_role"] == "intervention_endpoint").sum()),
            "control_count": len(controls),
            "biological_unit_count": observations["biological_unit_uid"].nunique(),
            "effective_sample_size": float(observations["biological_unit_uid"].nunique()),
            "action_count": len(actions),
            "action_family_count": actions["action_family_uid"].nunique(),
            "trajectory_count": 3,
            "timepoint_count": observations["time_value"].nunique(),
            "value_block_format": "zarr",
            "value_block_path": "expression.zarr",
            "sample_metadata_path": "observations.parquet",
            "feature_metadata_path": "features.parquet",
            "control_metadata_path": "controls.parquet",
            "action_catalog_path": "actions.parquet",
            "feature_quarantine_path": "feature_quarantine.parquet",
            "matrix_manifest_path": "matrix_manifest.json",
            "measurement_namespace": "NCBI_Entrez_Gene",
            "value_unit": parser["source_value_semantics"],
            "qc_grade": source["evidence_grade"],
            "qc_grade_rationale": source["evidence_grade_rationale"],
            "action_completeness": source["action_completeness"],
            "target_status": "explicit_unknown",
            "exposure_status": "source_observed_constant_media_concentration",
            "occupancy_status": "explicit_unavailable",
            "split_role": source["split_role"],
            "source_release": {
                "official_url": source["official_study_url"],
                "release_version": source["source_release_version"],
                "publication_status": "manuscript_in_preparation_in_official_idf",
                "access_class": "public",
            },
            "input_bindings": [_binding(path) for path in paths.values()],
            "block_files": records,
            "qc": {"status": "passed", **matrix_manifest},
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_holdout_access": 0,
        }
        write_json(stage / "block_manifest.json", manifest)
        if destination.exists():
            existing = _json(destination / "block_manifest.json")
            if existing.get("block_sha256") != block_sha256:
                raise ValueError(f"Content-address collision: {destination}")
            manifest = existing
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage), str(destination))
    manifest_path = destination / "block_manifest.json"
    audit = dict(manifest)
    audit["canonical_block_manifest"] = _binding(manifest_path)
    write_json(_repo_path(str(source["block_manifest_audit_path"])), audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = build(REPO_ROOT)
    print(
        json.dumps(
            {
                "status": result["qc"]["status"],
                "canonical_uid": result["canonical_uid"],
                "observation_count": result["observation_count"],
                "control_count": result["control_count"],
                "action_count": result["action_count"],
                "feature_count": result["qc"]["matrix_shape"][1],
                "block_sha256": result["block_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
