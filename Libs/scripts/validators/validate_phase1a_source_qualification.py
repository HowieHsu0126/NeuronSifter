"""Validate the terminal source-governance closure for Phase 1A."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file, safe_repo_path
from Libs.utils.data.raw_data_registry import write_json

SOURCE_ARTIFACT = "Output/audits/public_foundation_corpus_source_qualification.json"
LEDGER = "Output/audits/source_qualification_ledger.tsv"
GRAPH = "Output/audits/source_independence_graph.json"
LICENSE = "Output/audits/license_role_decisions.json"
RELEASE = "Output/audits/source_release_manifest.tsv"
MIRROR = "Output/audits/mirror_group_registry.tsv"
SCHEMA = "Libs/configs/schema/phase1a_source_governance.schema.json"
DECISIONS = frozenset({
    "qualified_for_mainline",
    "qualified_for_weak_calibration",
    "qualified_as_context_prior",
    "deferred",
    "formally_excluded",
})
REQUIRED_SOURCES = frozenset({
    "arrayexpress", "geo", "allen_brain_cell_atlas", "biccn_public", "clue_lincs",
    "drugcomb", "bindingdb", "cellxgene_census", "human_cell_atlas",
    "human_protein_atlas", "jump_cell_painting", "metabolomics_workbench",
    "nemo_archive", "neuromorpho", "openneuro", "sra",
})


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate(repo_root: Path = REPO_ROOT) -> dict:
    repo_root = repo_root.resolve()
    blockers: list[dict[str, str]] = []

    def require(condition: bool, code: str, message: str) -> None:
        if not condition:
            blockers.append({"code": code, "message": message})

    source_path = safe_repo_path(repo_root, SOURCE_ARTIFACT)
    ledger_path = safe_repo_path(repo_root, LEDGER)
    graph_path = safe_repo_path(repo_root, GRAPH)
    license_path = safe_repo_path(repo_root, LICENSE)
    release_path = safe_repo_path(repo_root, RELEASE)
    mirror_path = safe_repo_path(repo_root, MIRROR)
    schema_path = safe_repo_path(repo_root, SCHEMA)
    for path in (source_path, ledger_path, graph_path, license_path, release_path, mirror_path, schema_path):
        require(path.is_file(), "required_artifact_missing", str(path.relative_to(repo_root)))
    if blockers:
        return {"status": "blocked", "blocker_count": len(blockers), "blockers": blockers}

    schema = _json(schema_path)
    schema_errors = sorted(Draft202012Validator(schema).iter_errors(_json(graph_path)), key=lambda e: list(e.absolute_path))
    schema_errors += sorted(Draft202012Validator(schema).iter_errors(_json(license_path)), key=lambda e: list(e.absolute_path))
    require(not schema_errors, "phase1a_json_schema_violation", "; ".join(error.message for error in schema_errors))

    source = _json(source_path)
    ledger = _tsv(ledger_path)
    releases = _tsv(release_path)
    mirrors = _tsv(mirror_path)
    graph = _json(graph_path)
    license_decisions = _json(license_path)
    source_rows = source.get("sources", [])
    source_ids = {str(row.get("source_id")) for row in source_rows}
    ledger_ids = {str(row.get("source_id")) for row in ledger}
    release_ids = {str(row.get("source_id")) for row in releases}
    license_ids = {str(row.get("source_id")) for row in license_decisions.get("decisions", [])}
    graph_ids = {str(row.get("source_id")) for row in graph.get("nodes", [])}
    require(source_ids == REQUIRED_SOURCES, "source_set_not_frozen", f"expected {len(REQUIRED_SOURCES)} source IDs, got {len(source_ids)}")
    require(ledger_ids == source_ids, "ledger_source_set_mismatch", "ledger does not cover exactly the frozen source set")
    require(release_ids == source_ids, "release_source_set_mismatch", "release manifest does not cover exactly the frozen source set")
    require(license_ids == source_ids, "license_source_set_mismatch", "license decisions do not cover exactly the frozen source set")
    require(graph_ids == source_ids, "independence_graph_source_set_mismatch", "independence graph does not cover exactly the frozen source set")
    require(len(ledger) == len(releases) == len(source_rows) == len(REQUIRED_SOURCES), "source_row_count_mismatch", "all Phase 1A ledgers must contain exactly 16 rows")

    for row in ledger:
        source_id = row["source_id"]
        require(row.get("decision") in DECISIONS, "invalid_terminal_decision", f"{source_id}: {row.get('decision')}")
        require(row.get("terminal_status") == "terminal", "source_not_terminal", source_id)
        require(row.get("decision") == row.get("status"), "decision_status_mismatch", source_id)
        require(row.get("license_status") not in {"license_unresolved", ""}, "license_unresolved", source_id)
        require(row.get("role_status") not in {"role_unresolved", ""}, "role_unresolved", source_id)
        require(row.get("release_status") not in {"release_unresolved", ""}, "release_unresolved", source_id)
        require(row.get("provenance_record_id", "") != "", "provenance_missing", source_id)
        if row.get("decision", "").startswith("qualified_"):
            require(row.get("official_url", "") != "", "qualified_source_official_url_missing", source_id)
            require(row.get("raw_sha256", "") not in {"", "None"}, "qualified_source_raw_checksum_missing", source_id)
            require(row.get("processed_sha256", "") not in {"", "None", "not_available_phase1a"}, "qualified_source_processed_checksum_missing", source_id)
            require(row.get("parser_id", "").endswith("registry") or row.get("parser_id", "").startswith("build_"), "qualified_source_parser_missing", source_id)
            require(row.get("parser_version", "") not in {"", "pending"}, "qualified_source_parser_version_missing", source_id)
            require(row.get("mapping_version", "") not in {"", "pending"} and "pending" not in row.get("mapping_version", ""), "qualified_source_mapping_version_missing", source_id)

    for decision in license_decisions.get("decisions", []):
        source_id = str(decision.get("source_id"))
        require(decision.get("raw_asset_release_allowed") is False, "raw_asset_release_not_closed", source_id)
        require(str(decision.get("use_scope", "")) != "", "license_use_scope_missing", source_id)
        require(str(decision.get("role_boundary", "")) != "", "license_role_boundary_missing", source_id)

    require(source.get("status") == "passed", "source_artifact_not_passed", "source qualification artifact is not passed")
    require(source.get("phase1a_status") == "passed", "phase1a_status_not_passed", "Phase 1A source governance is not passed")
    unresolved = source.get("unresolved_counts", {})
    for key in ("license_unresolved", "role_unresolved", "release_unresolved"):
        require(int(unresolved.get(key, -1)) == 0, key, f"{key} must be zero")
    require(graph.get("status") == "passed" and graph.get("mirror_groups_frozen") is True and graph.get("publication_accession_groups_frozen") is True, "independence_graph_not_frozen", "source independence graph is not frozen")
    require(license_decisions.get("status") == "passed", "license_role_artifact_not_passed", "license/role decisions are not passed")

    return {"status": "passed" if not blockers else "blocked", "blocker_count": len(blockers), "blockers": blockers, "source_count": len(source_ids), "decision_counts": source.get("decision_counts", {}), "unresolved_counts": unresolved}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 1A source qualification closure.")
    parser.add_argument("--output", default="Output/audits/phase1a_source_qualification_validation.json")
    args = parser.parse_args()
    report = validate()
    write_json(safe_repo_path(REPO_ROOT, args.output), report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
