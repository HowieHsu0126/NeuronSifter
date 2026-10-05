"""Build a metadata-only priority registry for therapeutic evidence sources."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402

CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
PRIORITY_CONFIG_PATH = REPO_ROOT / "Libs/configs/data/therapeutic_source_priority.yaml"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/therapeutic_source_priority.schema.json"
BUILDER_PATH = REPO_ROOT / "Libs/scripts/builders/build_therapeutic_source_priority.py"
OUTPUT_TABLE_PATH = REPO_ROOT / "Output/audits/therapeutic_source_priority.tsv"
OUTPUT_AUDIT_PATH = REPO_ROOT / "Output/audits/therapeutic_source_selection.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes repository: {value}")
    return path


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_created_at(audit_path: Path, *, write_output: bool) -> str:
    """Keep repeated materializations byte-stable when only metadata is rebuilt."""

    if write_output and audit_path.is_file():
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping) and isinstance(payload.get("created_at"), str):
            return str(payload["created_at"])
    return _now()


def _load_policy() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config.get("therapeutic_source_priority")
    if not isinstance(policy, dict):
        raise TypeError("Therapeutic source priority must be a mapping")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(policy), key=lambda error: list(error.absolute_path))
    if errors:
        details = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Therapeutic source priority schema validation failed: {details}")
    return policy


def _read_tsv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Source registry is empty: {path}")
    return {str(row["source_accession"]): row for row in rows}


def _flatten_attributes(payload: object, output: list[tuple[str, str]]) -> None:
    if isinstance(payload, dict):
        attributes = payload.get("attributes")
        if isinstance(attributes, list):
            for item in attributes:
                if isinstance(item, Mapping) and "name" in item and "value" in item:
                    output.append((str(item["name"]), str(item["value"])))
        for value in payload.values():
            _flatten_attributes(value, output)
    elif isinstance(payload, list):
        for value in payload:
            _flatten_attributes(value, output)


def _metadata_summary(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "metadata_status": "not_materialized",
            "metadata_path": path.relative_to(REPO_ROOT).as_posix(),
            "metadata_title": "",
            "metadata_factors": [],
            "metadata_description_has_functional_endpoint": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    attributes: list[tuple[str, str]] = []
    _flatten_attributes(payload, attributes)
    title_values = [value for name, value in attributes if name.lower() == "title"]
    factor_values = [
        f"{name}={value}"
        for name, value in attributes
        if re.search(r"treat|compound|dose|factor|protocol", name, flags=re.IGNORECASE)
    ]
    descriptions = [value for name, value in attributes if name.lower() == "description"]
    description = " ".join(descriptions).lower()
    return {
        "metadata_status": "local_study_metadata_available",
        "metadata_path": path.relative_to(REPO_ROOT).as_posix(),
        "metadata_title": title_values[0] if title_values else "",
        "metadata_factors": sorted(set(factor_values)),
        "metadata_description_has_functional_endpoint": bool(
            re.search(r"behavior|cognition|viability|rescue|synap|memory|oxidative stress", description)
        ),
    }


def _source_row(candidate: Mapping[str, object], registry: Mapping[str, Mapping[str, str]]) -> dict[str, object]:
    accession = str(candidate["source_accession"])
    row = registry.get(accession)
    if row is None:
        raise ValueError(f"Selected source is absent from the frozen accession registry: {accession}")
    if row.get("include_flag") != "yes":
        raise ValueError(f"Selected source is not admitted by the accession registry: {accession}")
    return {
        **dict(candidate),
        "registry_title": row.get("title", ""),
        "registry_species": row.get("species", ""),
        "registry_tissue_or_biofluid": row.get("tissue_or_biofluid", ""),
        "registry_modality": row.get("modality", ""),
        "registry_sample_context": row.get("sample_context", ""),
        "registry_subject_linkable": row.get("subject_linkable_flag", ""),
        "registry_provenance_level": row.get("provenance_level", ""),
    }


def build(*, write_output: bool = True) -> dict[str, object]:
    policy = _load_policy()
    registries = policy["registries"]
    geo = _read_tsv(_repo_path(str(registries["geo_accessions"])))
    arrayexpress = _read_tsv(_repo_path(str(registries["arrayexpress_accessions"])))
    details_root = _repo_path(str(registries["arrayexpress_study_details"]))
    rows: list[dict[str, object]] = []
    for candidate in policy["candidates"]:
        repository = str(candidate["source_repository"])
        registry = geo if repository == "GEO" else arrayexpress
        row = _source_row(candidate, registry)
        details = _metadata_summary(details_root / f"{candidate['source_accession']}.json") if repository == "ArrayExpress" else {
            "metadata_status": "not_materialized",
            "metadata_path": "",
            "metadata_title": "",
            "metadata_factors": [],
            "metadata_description_has_functional_endpoint": False,
        }
        rows.append({**row, **details})
    rows.sort(key=lambda row: int(row["priority_rank"]))
    ranks = [int(row["priority_rank"]) for row in rows]
    if ranks != list(range(1, len(rows) + 1)):
        raise ValueError(f"Priority ranks must be contiguous: {ranks}")
    if len({str(row["source_accession"]) for row in rows}) != len(rows):
        raise ValueError("Source accessions must be unique")
    selected = [row for row in rows if row["selection_class"] == "selected_for_materialization"]
    if not selected:
        raise ValueError("At least one source must be selected for materialization")
    if any(bool(row["expected_candidate_pair_count"]) and int(row["expected_candidate_set_count"]) == 0 for row in rows):
        raise ValueError("Candidate pair expectations cannot be nonzero when candidate-set expectation is zero")

    table_rows: list[dict[str, object]] = []
    fields = [
        "priority_rank", "source_repository", "source_accession", "selection_class", "expected_evidence_class",
        "expected_matched_control", "expected_therapeutic_arm_count", "expected_candidate_pair_count",
        "expected_candidate_set_count", "expected_replicate_structure", "expected_uncertainty_source",
        "objective_compatibility", "registry_title", "registry_species", "registry_tissue_or_biofluid",
        "registry_modality", "registry_sample_context", "registry_subject_linkable", "registry_provenance_level",
        "metadata_status", "metadata_path", "metadata_title", "metadata_factors",
        "metadata_description_has_functional_endpoint", "rationale", "main_uncertainty",
    ]
    for row in rows:
        table_row = {field: row[field] for field in fields}
        table_row["metadata_factors"] = json.dumps(row["metadata_factors"], ensure_ascii=True, separators=(",", ":"))
        table_rows.append(table_row)

    table_path = _repo_path(str(policy["outputs"]["priority_table"]))
    audit_path = _repo_path(str(policy["outputs"]["selection_audit"]))
    table_binding = {"path": table_path.relative_to(REPO_ROOT).as_posix(), "sha256": None}
    if write_output:
        write_tsv(table_path, fields, table_rows)
        table_binding = _artifact(table_path)
    audit: dict[str, object] = {
        "schema_version": "neuronsifter-therapeutic-source-selection-audit-v1",
        "created_at": _stable_created_at(audit_path, write_output=write_output),
        "status": "source_priority_frozen_metadata_only",
        "selection_basis": policy["source_selection_basis"],
        "policy_digest": _canonical_hash(policy),
        "policy_binding": _artifact(PRIORITY_CONFIG_PATH),
        "schema_binding": _artifact(SCHEMA_PATH),
        "implementation_binding": _artifact(BUILDER_PATH),
        "registry_bindings": {
            "geo_accessions": _artifact(_repo_path(str(registries["geo_accessions"]))),
            "arrayexpress_accessions": _artifact(_repo_path(str(registries["arrayexpress_accessions"]))),
        },
        "priority_table_binding": table_binding,
        "source_count": len(rows),
        "selected_for_materialization_count": len(selected),
        "expected_candidate_set_count": sum(int(row["expected_candidate_set_count"]) for row in rows),
        "expected_candidate_pair_count": sum(int(row["expected_candidate_pair_count"]) for row in rows),
        "selected_sources": [str(row["source_accession"]) for row in selected],
        "outcome_values_inspected": False,
        "model_predictions_inspected": False,
        "protected_holdout_access": 0,
        "publication_comparison_open": False,
        "benchmark_split_created": False,
        "world_action_training_authorized": False,
        "next_route": "materialize_selected_therapeutic_sources_then_construct_candidate_sets",
        "rows": rows,
    }
    if write_output:
        write_json(audit_path, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(write_output=not args.no_write), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
