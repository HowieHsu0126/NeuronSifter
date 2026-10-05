"""Materialize the independent E-MTAB-6772 perturbation mainline block."""

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
from typing import Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256, hash_path
from Libs.utils.data.public_foundation_evidence import sha256_file
from Libs.utils.data.public_foundation_roles import upsert_accession_requalification_delta
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, write_json
from Libs.utils.evaluation.alignment_benchmark import write_parquet_atomic
from Libs.utils.model.dynamics_store import initialize_zarr_store, write_zarr_array


BUILDER_ID = "Libs/scripts/builders/materialize_phase1f_perturbation_pair_block.py"
MANIFEST_VERSION = "neuronsifter-phase1f-perturbation-pair-mainline-block-v1"
AUDIT_VERSION = "neuronsifter-phase1f-perturbation-pair-audit-v1"
CANONICAL_UID = "pfc_arrayexpress_e_mtab_6772_perturbation_phase1f_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _record(path: Path, name: str) -> dict[str, object]:
    file_count, byte_count, digest = hash_path(path)
    return {"name": name, "relative_path": name, "file_count": file_count, "bytes": byte_count, "sha256": digest}


def _content_hash(records: list[Mapping[str, object]]) -> str:
    return canonical_sha256([
        {"name": row["name"], "file_count": row["file_count"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in records
    ])


def _uid(prefix: str, payload: Mapping[str, object]) -> str:
    return f"{prefix}:{canonical_sha256(dict(payload))[:24]}"


def _read_sdrf(path: Path, policy: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = policy["candidate"]
    action = policy["action"]
    qc = policy["qc"]
    assert isinstance(candidate, Mapping) and isinstance(action, Mapping) and isinstance(qc, Mapping)
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    treated: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    for row in rows:
        sample = str(row["Source Name"]).strip()
        assay = str(row["Assay Name"]).strip()
        value_file = str(row["Array Data File"]).strip()
        compound = str(row["Factor Value[compound]"]).strip()
        disease_raw = str(row["Factor Value[disease]"]).strip()
        genotype_raw = str(row["Factor Value[genotype]"]).strip()
        genotype_class = "5XFAD" if disease_raw == "Alzheimers disease" else "wild_type"
        disease = "Alzheimer_disease_mouse_model" if genotype_class == "5XFAD" else "normal_control"
        common = {
            "source_uid": str(candidate["source_id"]),
            "publication_uid": str(candidate["publication_uid"]),
            "accession_uid": str(candidate["source_accession"]),
            "source_sample_name": sample,
            "sample_uid": f"arrayexpress:{candidate['source_accession']}:{sample}",
            "source_assay_name": assay,
            "source_value_file": value_file,
            "subject_or_donor_uid": "explicit_unavailable_within_declared_pooled_sample",
            "cell_or_tissue_context": "mouse_hippocampus_pooled_biological_sample",
            "cns_context_class": str(candidate["cns_context_class"]),
            "species": "Mus musculus",
            "species_taxonomy_id": "NCBITaxon:10090",
            "tissue": "hippocampus",
            "brain_region_ontology_id": "UBERON:0002421",
            "disease_state": disease,
            "disease_state_original": disease_raw,
            "genotype_class": genotype_class,
            "genotype_original": genotype_raw,
            "assay_uid": str(candidate["assay_uid"]),
            "modality": "transcriptomics",
            "feature_namespace": "NCBI_Entrez_Gene",
            "value_unit": "normalized_Illumina_AVG_Signal",
            "replicate_group_uid": f"arrayexpress:{candidate['source_accession']}:declared_biological_sample:{sample}",
            "technical_replicate_status": "one_assay_per_declared_biological_sample",
            "pooling_status": "source_declares_biological_sample_but_idf_pooling_wording_is_conflicting",
        }
        if compound == "pantethine":
            dose_text = str(row["Factor Value[dose]"]).strip()
            unit_text = str(row["Unit[dose unit]"]).strip()
            if float(dose_text) != float(action["physical_dose"]) or unit_text != "milligram":
                raise ValueError(f"Unexpected pantethine physical dose in {sample}")
            biological_uid = f"arrayexpress:{candidate['source_accession']}:treated_sample:{sample}"
            observation_uid = f"arrayexpress_observation:{candidate['source_accession']}:{sample}"
            treated.append({
                **common,
                "observation_uid": observation_uid,
                "biological_unit_uid": biological_uid,
                "compound_uid": str(candidate["compound_uid"]),
                "compound_name": str(candidate["compound_name"]),
                "dose_value": float(action["physical_dose"]),
                "dose_unit": str(action["dose_unit"]),
                "raw_dose_value": dose_text,
                "raw_dose_unit": unit_text,
                "time_value": float(action["duration"]),
                "time_unit": str(action["duration_unit"]),
                "schedule": str(action["schedule"]),
                "route": str(action["route"]),
                "action_uid": "arrayexpress_action:E-MTAB-6772:pantethine_15mg_5.5month_ip_three_times_week",
                "control_group_uid": f"arrayexpress:{candidate['source_accession']}:vehicle:{genotype_class}",
                "qc_grade": str(qc["grade"]),
            })
        elif compound == "none":
            controls.append({
                **common,
                "control_uid": f"arrayexpress_control:{candidate['source_accession']}:{sample}",
                "control_group_uid": f"arrayexpress:{candidate['source_accession']}:vehicle:{genotype_class}",
                "control_type": "saline_vehicle",
                "compound_status": "vehicle_no_pantethine",
                "counted_in_observation_block": False,
                "counted_in_ess": False,
            })
        else:
            raise ValueError(f"Unexpected compound label in SDRF: {compound!r}")
    observations = pd.DataFrame(treated).sort_values("source_sample_name", kind="mergesort").reset_index(drop=True)
    control_frame = pd.DataFrame(controls).sort_values("source_sample_name", kind="mergesort").reset_index(drop=True)
    observations["value_row_index"] = np.arange(len(observations), dtype=np.int64)
    if len(observations) != int(qc["expected_treated_count"]) or len(control_frame) != int(qc["expected_control_count"]):
        raise ValueError("E-MTAB-6772 treated/control counts drifted")
    if observations["biological_unit_uid"].duplicated().any() or control_frame["control_uid"].duplicated().any():
        raise ValueError("E-MTAB-6772 biological sample identities are not unique")
    if set(observations["control_group_uid"]) != set(control_frame["control_group_uid"]):
        raise ValueError("Treated samples lack genotype-matched vehicle control groups")
    return observations, control_frame


def _read_assay(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "PROBE_ID" not in reader.fieldnames:
            raise ValueError(f"Assay lacks PROBE_ID: {path}")
        signal_field = next((field for field in reader.fieldnames if field.endswith(".AVG_Signal")), None)
        if signal_field is None:
            raise ValueError(f"Assay lacks AVG_Signal: {path}")
        rows = list(reader)
    probes = [str(row["PROBE_ID"]).strip() for row in rows]
    values = np.asarray([float(row[signal_field]) for row in rows], dtype=np.float64)
    if len(probes) != len(set(probes)) or not np.isfinite(values).all():
        raise ValueError(f"Assay probe identity or finite-value contract failed: {path}")
    return probes, values


def _read_adf(path: Path) -> dict[str, tuple[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header_index = next(index for index, row in enumerate(rows) if row and row[0] == "Block Column")
    header = rows[header_index]
    positions = {name: index for index, name in enumerate(header)}
    required = ("Reporter Name", "Reporter Database Entry[entrez]", "Reporter Database Entry[hugo]")
    if any(name not in positions for name in required):
        raise ValueError("A-MEXP-932 lacks required reporter/Entrez/symbol fields")
    result: dict[str, tuple[str, str]] = {}
    for row in rows[header_index + 1:]:
        if len(row) <= max(positions[name] for name in required):
            continue
        reporter = row[positions["Reporter Name"]].strip()
        if reporter in result:
            raise ValueError(f"Duplicate ADF reporter: {reporter}")
        result[reporter] = (
            row[positions["Reporter Database Entry[entrez]"]].strip(),
            row[positions["Reporter Database Entry[hugo]"]].strip(),
        )
    return result


def _canonicalize(
    probe_ids: list[str], matrix: np.ndarray, annotations: Mapping[str, tuple[str, str]]
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
    canonical = np.empty((matrix.shape[0], len(ordered_genes)), dtype=np.float32)
    features: list[dict[str, object]] = []
    for feature_index, gene_id in enumerate(ordered_genes):
        positions = positions_by_gene[gene_id]
        canonical[:, feature_index] = np.median(matrix[:, positions], axis=1).astype(np.float32)
        features.append({
            "feature_index": feature_index,
            "feature_namespace": "NCBI_Entrez_Gene",
            "feature_id": f"NCBIGene:{gene_id}",
            "canonical_gene_id": gene_id,
            "source_gene_symbols": "|".join(sorted(symbols_by_gene[gene_id])) or "explicit_unavailable",
            "source_probe_ids": "|".join(probe_ids[position] for position in positions),
            "source_probe_count": len(positions),
            "mapping_method": "exact_A-MEXP-932_Reporter_Name_to_numeric_Entrez_then_median",
            "mapping_confidence": "exact_reporter_join",
            "aggregation_rule": "median_across_probes_mapping_to_same_Entrez_gene",
        })
    diagnostics = {
        "source_probe_count": len(probe_ids),
        "exact_entrez_probe_count": exact_probe_count,
        "canonical_gene_count": len(ordered_genes),
        "unmatched_probe_count": sum(row["mapping_status"] == "unmatched_adf_reporter" for row in quarantine),
        "non_numeric_entrez_count": sum(row["mapping_status"] == "non_numeric_or_missing_adf_entrez" for row in quarantine),
    }
    return pd.DataFrame(features), pd.DataFrame(quarantine), canonical, diagnostics


def _pair_independence(policy: Mapping[str, object], observations: pd.DataFrame) -> dict[str, object]:
    reference = policy["phase1f_pair_reference"]
    candidate = policy["candidate"]
    assert isinstance(reference, Mapping) and isinstance(candidate, Mapping)
    admission = _load_json(_path(str(reference["admission_audit_path"])))
    binding = admission.get("canonical_block_manifest")
    if not isinstance(binding, Mapping):
        raise ValueError("CLUE M1 admission lacks canonical manifest binding")
    manifest_path = _path(str(binding.get("path", "")))
    if _binding(manifest_path) != dict(binding):
        raise ValueError("CLUE M1 manifest binding has drifted")
    manifest = _load_json(manifest_path)
    for field in ("canonical_uid", "source_id", "source_accession", "publication_accession_group", "mirror_group"):
        if manifest.get(field) != reference[f"expected_{field}"]:
            raise ValueError(f"CLUE M1 reference mismatch for {field}")
    identity_fields = ("source_id", "source_accession", "publication_accession_group", "mirror_group")
    equal_fields = [field for field in identity_fields if str(manifest[field]) == str(candidate[field])]
    m1 = pd.read_parquet(manifest_path.parent / str(manifest["sample_metadata_path"]))
    sample_overlap = set(m1["sample_uid"].astype(str)) & set(observations["sample_uid"].astype(str))
    biological_overlap = set(m1["biological_unit_uid"].astype(str)) & set(observations["biological_unit_uid"].astype(str))
    if equal_fields or sample_overlap or biological_overlap:
        raise ValueError("Perturbation pair independence failed")
    return {
        "status": "passed",
        "m1_manifest": _binding(manifest_path),
        "different_identity_axes": list(identity_fields),
        "identity_equal_fields": [],
        "sample_uid_overlap_count": 0,
        "biological_unit_overlap_count": 0,
        "m1_observation_count": int(manifest["observation_count"]),
        "pair_observation_count": int(len(observations)),
        "mirror_overlap": False,
    }


def _semantic_axes(candidate: Mapping[str, object]) -> dict[str, dict[str, str]]:
    return {
        "study": {"status": "observed", "value": str(candidate["source_accession"])},
        "biological_unit": {"status": "mapped", "value": "source-declared pooled hippocampus biological sample"},
        "species": {"status": "mapped", "value": "NCBITaxon:10090"},
        "disease_state": {"status": "observed", "value": "5XFAD Alzheimer disease mouse model|wild type"},
        "tissue": {"status": "mapped", "value": "UBERON:0002421 hippocampus"},
        "brain_region": {"status": "mapped", "value": "UBERON:0002421"},
        "cell_type": {"status": "not_applicable", "reason": "bulk pooled hippocampus expression"},
        "assay": {"status": "mapped", "value": str(candidate["assay_uid"])},
        "modality": {"status": "observed", "value": "transcriptomics"},
        "feature": {"status": "mapped", "value": "NCBI_Entrez_Gene"},
        "time": {"status": "observed", "value": "5.5 month treatment duration"},
        "dose": {"status": "observed", "value": "15 mg per administration; not converted to mg/kg"},
        "unit": {"status": "mapped", "value": "mg|month"},
        "provenance": {"status": "observed", "value": "ArrayExpress SDRF/IDF/ADF/assays plus PMID:31267473 Methods"},
    }


def build(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], dict[str, object]]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("Perturbation pair materialization is identity-bound to the repository root")
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1f_perturbation_pair_materialization"]
    candidate = policy["candidate"]
    qc = policy["qc"]
    action = policy["action"]
    assert isinstance(candidate, Mapping) and isinstance(qc, Mapping) and isinstance(action, Mapping)

    adf_path = _path(str(candidate["platform_annotation_path"]))
    if adf_path.stat().st_size != int(candidate["expected_platform_annotation_bytes"]) or sha256_file(adf_path) != candidate["expected_platform_annotation_sha256"]:
        raise ValueError("A-MEXP-932 bytes or checksum differ from the frozen contract")
    snapshot_root = _path(str(candidate["official_snapshot_root"]))
    assay_paths = sorted(snapshot_root.glob("[0-9]*_?.txt"))
    if len(assay_paths) != int(qc["expected_assay_count"]):
        raise ValueError("Official assay count differs from the frozen contract")
    observations, controls = _read_sdrf(_path(str(candidate["sample_metadata_path"])), policy)
    sample_order = pd.concat([observations, controls], ignore_index=True)
    available = {path.name: path for path in assay_paths}
    probe_ids: list[str] | None = None
    raw_rows: list[np.ndarray] = []
    for value_file in sample_order["source_value_file"]:
        path = available.get(str(value_file))
        if path is None:
            raise ValueError(f"SDRF-bound assay is missing: {value_file}")
        current_probes, current_values = _read_assay(path)
        if probe_ids is None:
            probe_ids = current_probes
        elif probe_ids != current_probes:
            raise ValueError(f"Probe axis differs in {value_file}")
        raw_rows.append(current_values)
    assert probe_ids is not None
    raw_matrix = np.vstack(raw_rows)
    features, quarantine, canonical_matrix, diagnostics = _canonicalize(probe_ids, raw_matrix, _read_adf(adf_path))
    expected = {
        "source_probe_count": int(qc["expected_source_probe_count"]),
        "exact_entrez_probe_count": int(qc["expected_exact_entrez_probe_count"]),
        "canonical_gene_count": int(qc["expected_canonical_gene_count"]),
        "unmatched_probe_count": int(qc["expected_unmatched_probe_count"]),
        "non_numeric_entrez_count": int(qc["expected_non_numeric_entrez_count"]),
    }
    if diagnostics != expected:
        raise ValueError(f"E-MTAB-6772 mapping counts drifted: expected={expected}, observed={diagnostics}")
    treated_matrix = canonical_matrix[:len(observations)]
    if treated_matrix.shape != (int(qc["expected_treated_count"]), int(qc["expected_canonical_gene_count"])) or not np.isfinite(treated_matrix).all():
        raise ValueError("Canonical treated matrix shape or finite-value contract failed")

    action_uid = str(observations["action_uid"].iloc[0])
    actions = pd.DataFrame([{
        "action_uid": action_uid,
        "intervention_type": "compound_perturbation",
        "agent_uid": candidate["compound_uid"],
        "agent_set": candidate["compound_uid"],
        "target_set_or_explicit_unknown": action["target_set"],
        "target_set": action["target_set"],
        "physical_dose": action["physical_dose"],
        "dose_value": action["physical_dose"],
        "dose_unit": action["dose_unit"],
        "duration": action["duration"],
        "duration_unit": action["duration_unit"],
        "schedule": action["schedule"],
        "route_or_not_applicable": action["route"],
        "route": action["route"],
        "cell_or_tissue_context": "mouse_hippocampus_pooled_biological_sample",
        "assay_uid": candidate["assay_uid"],
        "assay_context": f"mouse_hippocampus_pooled_biological_sample|{candidate['assay_uid']}",
        "state_scope": candidate["cns_context_class"],
        "publication_uid": candidate["publication_uid"],
        "source_publication": candidate["publication_uid"],
        "accession_uid": candidate["source_accession"],
        "source_accession": candidate["source_accession"],
        "exposure_program_ref_or_explicit_unavailable": action["exposure_program_ref"],
        "exposure_program_ref": action["exposure_program_ref"],
        "occupancy_program_ref_or_explicit_unavailable": action["occupancy_program_ref"],
        "occupancy_program_ref": action["occupancy_program_ref"],
        "observation_uids": "|".join(observations["observation_uid"].astype(str)),
        "linked_observation_count": len(observations),
        "control_group_uids": "|".join(sorted(set(observations["control_group_uid"].astype(str)))),
        "qc_grade": qc["grade"],
    }])
    pair = _pair_independence(policy, observations)

    output_root = _path(str(policy["output_root"]))
    staging_root = output_root / ".phase1f_perturbation_pair_staging"
    ensure_directory(staging_root)
    with tempfile.TemporaryDirectory(prefix="e-mtab-6772-", dir=staging_root) as temporary:
        stage = Path(temporary)
        store = stage / "expression.zarr"
        initialize_zarr_store(store, {
            "format": "zarr_v2_uncompressed",
            "source_accession": candidate["source_accession"],
            "row_axis": "observations.parquet:value_row_index",
            "column_axis": "features.parquet:feature_index",
            "value_semantics": "normalized_Illumina_AVG_Signal",
        })
        write_zarr_array(store, "values", treated_matrix.astype(np.float32), (int(policy["parser"]["row_chunk_size"]), len(features)), attrs={"feature_namespace": "NCBI_Entrez_Gene", "unit": "normalized_Illumina_AVG_Signal"})
        write_parquet_atomic(observations, stage / "observations.parquet")
        write_parquet_atomic(features, stage / "features.parquet")
        write_parquet_atomic(controls, stage / "controls.parquet")
        write_parquet_atomic(actions, stage / "actions.parquet")
        write_parquet_atomic(quarantine, stage / "feature_quarantine.parquet")
        matrix_stats = {
            "matrix_shape": [len(observations), len(features)],
            "source_assay_count": len(assay_paths),
            "source_probe_count": len(probe_ids),
            "scalar_value_count": int(treated_matrix.size),
            "finite_values": True,
            "finite_fraction": float(np.isfinite(treated_matrix).mean()),
            "minimum_value": float(treated_matrix.min()),
            "maximum_value": float(treated_matrix.max()),
            "mapping_diagnostics": diagnostics,
            "control_expression_materialized_in_mainline_block": False,
        }
        if matrix_stats["scalar_value_count"] != int(qc["expected_scalar_value_count"]):
            raise ValueError("Canonical scalar value count differs from frozen contract")
        write_json(stage / "matrix_manifest.json", matrix_stats)
        output_names = ("expression.zarr", "observations.parquet", "features.parquet", "controls.parquet", "actions.parquet", "feature_quarantine.parquet", "matrix_manifest.json")
        records = [_record(stage / name, name) for name in output_names]
        block_sha256 = _content_hash(records)
        destination = output_root / "blocks" / str(candidate["source_id"]) / str(candidate["source_accession"]) / block_sha256[:16]
        required_inputs = [
            *assay_paths,
            _path(str(candidate["sample_metadata_path"])),
            _path(str(candidate["study_metadata_path"])),
            adf_path,
            _path(str(candidate["terms_snapshot_path"])),
            _path(str(candidate["pubmed_snapshot_path"])),
            _path(str(candidate["europepmc_snapshot_path"])),
            _path(str(candidate["publication_method_evidence_path"])),
            _path(str(candidate["compound_identity_path"])),
        ]
        protocol_conflicts = [
            {
                "conflict_id": "treatment_schedule_frequency",
                "idf_value": "two_times_per_week",
                "publication_methods_value": "three_times_per_week",
                "resolution": "retain_both_use_peer_reviewed_publication_methods_for_structured_action",
                "idf_evidence": _binding(_path(str(candidate["study_metadata_path"]))),
                "publication_evidence": _binding(_path(str(candidate["publication_method_evidence_path"]))),
            },
            {
                "conflict_id": "pooled_sample_wording",
                "idf_value": "three_hippocampi_from_each_group_were_pooled",
                "sdrf_and_idf_sample_count_value": "twelve_declared_biological_samples",
                "resolution": "preserve_six_treated_declared_samples_as_grade_B_biological_units_without_inventing_individual_mouse_ids",
                "idf_evidence": _binding(_path(str(candidate["study_metadata_path"]))),
                "sdrf_evidence": _binding(_path(str(candidate["sample_metadata_path"]))),
            },
        ]
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "producer": BUILDER_ID,
            "created_at": _now(),
            "canonical_uid": CANONICAL_UID,
            "content_address": f"sha256:{block_sha256}",
            "block_sha256": block_sha256,
            "family_id": candidate["family_id"],
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
            "cns_source": True,
            "cns_context_class": candidate["cns_context_class"],
            "role": "qualified_for_mainline",
            "admission_allowed": True,
            "value_block_format": "zarr",
            "value_block_path": "expression.zarr",
            "sample_metadata_path": "observations.parquet",
            "feature_metadata_path": "features.parquet",
            "control_metadata_path": "controls.parquet",
            "action_catalog_path": "actions.parquet",
            "feature_quarantine_path": "feature_quarantine.parquet",
            "matrix_manifest_path": "matrix_manifest.json",
            "observation_count": len(observations),
            "control_count": len(controls),
            "biological_unit_count": int(observations["biological_unit_uid"].nunique()),
            "effective_sample_size": float(observations["biological_unit_uid"].nunique()),
            "effective_sample_size_formula": policy["biological_unit_policy"]["effective_sample_size_formula"],
            "biological_unit_policy": policy["biological_unit_policy"],
            "action_count": len(actions),
            "action_linkage_complete": True,
            "real_value_matrix": True,
            "compound_dose_time_axes_complete": True,
            "qc_grade": qc["grade"],
            "qc_grade_rationale": qc["grade_rationale"],
            "qc": {**matrix_stats, "status": "passed"},
            "mapping_confidence": "exact_ADF_reporter_to_numeric_Entrez_with_median_probe_aggregation",
            "mapping_version": policy["parser"]["mapping_version"],
            "unit_mapping_version": policy["parser"]["unit_mapping_version"],
            "ontology_versions": {"species": "NCBITaxon:10090", "brain_region": "UBERON:0002421", "feature": "NCBI_Entrez_Gene_A-MEXP-932", "compound": "PubChem_CID_452306", "unit": "physical_mg_month"},
            "semantic_axes": _semantic_axes(candidate),
            "semantic_axes_complete": True,
            "pair_independence": pair,
            "protocol_conflicts": protocol_conflicts,
            "observation_value_join_contract": {"observation_axis": "observations.parquet:value_row_index", "feature_axis": "features.parquet:feature_index|feature_id", "value_axis": "expression.zarr:values[value_row_index,feature_index]", "feature_namespace": "NCBI_Entrez_Gene", "value_unit": "normalized_Illumina_AVG_Signal"},
            "source_release": {"official_url": candidate["official_study_url"], "release_version": candidate["source_release_version"], "retrieval_date": candidate["retrieval_date"], "publication_uid": candidate["publication_uid"], "publication_doi": candidate["publication_doi"], "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution", "access_class": "public", "license_role": policy["license_role"]},
            "parser": policy["parser"],
            "input_bindings": [_binding(path) for path in required_inputs],
            "block_files": records,
            "split_role": "unassigned_pending_nested_split_freeze",
            "blocker_codes": ["nested_split_freeze_pending", "real_domain_leakage_audit_pending"],
        }
        write_json(stage / "block_manifest.json", manifest)
        if destination.exists():
            existing = _load_json(destination / "block_manifest.json")
            if existing.get("block_sha256") != block_sha256:
                raise ValueError(f"Content-address collision: {destination}")
            manifest = existing
        else:
            ensure_directory(destination.parent)
            shutil.move(str(stage), str(destination))

    manifest_path = destination / "block_manifest.json"
    write_json(_path(str(policy["block_manifest_audit_path"])), manifest)
    delta_row = {
        "delta_version": "phase1f-source-qualification-delta-v1",
        "entry_id": "phase1f_arrayexpress_e_mtab_6772_perturbation_mainline",
        "source_id": candidate["source_id"],
        "source_accession": candidate["source_accession"],
        "prior_decision": "deferred",
        "decision": "qualified_for_mainline",
        "admission_allowed": "true",
        "scope": "source_accession_only",
        "frozen_ledger_path": policy["source_ledger_path"],
        "frozen_ledger_sha256": sha256_file(_path(str(policy["source_ledger_path"]))),
        "block_manifest_path": _relative(manifest_path),
        "block_manifest_sha256": sha256_file(manifest_path),
        "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_with_attribution",
        "access_class": "public_with_attribution",
        "status": "accepted",
    }
    upsert_accession_requalification_delta(_path(str(policy["qualification_delta_path"])), delta_row)

    common = {"schema_version": AUDIT_VERSION, "producer": BUILDER_ID, "created_at": _now(), "candidate_uid": "arrayexpress_e_mtab_6772_perturbation_pair", "source_id": candidate["source_id"], "source_accession": candidate["source_accession"], "canonical_uid": CANONICAL_UID, "canonical_block_manifest": _binding(manifest_path)}
    audits = {
        "candidate_audit_path": {**common, "status": "passed_candidate_selected", "role_before": "deferred", "role_after": "qualified_for_mainline", "scope": "source_accession_only", "real_value_matrix": True, "license_role": policy["license_role"], "release_evidence": manifest["source_release"]},
        "qc_audit_path": {**common, "status": "passed", "qc_grade": "B", "thresholds": qc, "results": manifest["qc"], "protocol_conflicts": manifest["protocol_conflicts"]},
        "ess_audit_path": {**common, "status": "passed", "effective_sample_size": manifest["effective_sample_size"], "ess_unit": "unique_treated_declared_biological_sample", "formula": manifest["effective_sample_size_formula"], "treated_sample_count": len(observations), "vehicle_control_count": len(controls), "controls_counted_in_ess": False, "features_counted_in_ess": False, "technical_replicates_counted_in_ess": False},
        "action_linkage_path": {**common, "status": "passed", "action_linkage_complete": True, "qualified_action_count": 1, "linked_observation_count": len(observations), "one_action_to_six_observations": True, "action_catalog": _binding(destination / "actions.parquet"), "target_status": "explicit_unknown", "exposure_status": "explicit_unavailable", "occupancy_status": "explicit_unavailable", "bootstrap_prior_used": False},
        "pair_independence_path": {**common, **pair},
    }
    for key, payload in audits.items():
        write_json(_path(str(policy[key])), payload)
    admission = {
        **common,
        "status": "accepted_second_independent_perturbation_block",
        "candidate_role_before": "deferred",
        "candidate_role_after": "qualified_for_mainline",
        "source_family_role_unchanged": "deferred",
        "scope": "source_accession_only",
        "admission_allowed": True,
        "qc_grade": "B",
        "effective_sample_size": manifest["effective_sample_size"],
        "independence_status": "passed_independent_pair_with_clue_m1",
        "pair_independence": pair,
        "license_role": policy["license_role"],
        "g2_status": "not_passed",
        "g2_remaining_blockers": ["nested_split_freeze_pending", "real_domain_leakage_audit_pending"],
    }
    write_json(_path(str(policy["admission_audit_path"])), admission)
    return manifest, admission


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the E-MTAB-6772 perturbation independent pair block.")
    parser.parse_args()
    manifest, admission = build(REPO_ROOT)
    print(json.dumps({"status": admission["status"], "canonical_uid": manifest["canonical_uid"], "observation_count": manifest["observation_count"], "feature_count": manifest["qc"]["matrix_shape"][1], "effective_sample_size": manifest["effective_sample_size"], "block_sha256": manifest["block_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
